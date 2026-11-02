"""generate(prompts, stops) behind two interchangeable backends.

  hf    Hugging Face transformers. Runs on MPS (the M1), CUDA or CPU. Used for local
        smoke tests. Prompts in a batch are LEFT-padded so they all end in the same
        column and generation continues from each prompt's last real token.
  vllm  vLLM, for the Kaggle T4s. Cannot run on the M1, so it is kept minimal and is
        exercised only on Kaggle.

Both return the same Generation record, so the runner never knows which one ran.
Decoding is greedy, as in the paper. Generation stops at the first of: a stop string
(blank line or a new "Question:" prefix, from prompts.py), the model's end token, or
max_new_tokens.

Ported from the pilot (llm_mini_replication/src/generate.py) with the chat-template
and sampling machinery removed: these are base models and plain text completion.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GenConfig:
    model: str = "Qwen/Qwen3-0.6B-Base"
    backend: str = "hf"  # "hf" | "vllm"
    dtype: str = "float16"  # matches the Kaggle T4, which has no bfloat16
    max_new_tokens: int = 256  # DIRECT needs ~5; CoT solutions in the exemplars are < 100
    max_model_len: int = 8192  # vllm only: longest prompt + completion. Telugu 6-shot prompts reach 4,243 tokens.
    batch_size: int = 8  # prompts per forward pass; lower if memory is tight


@dataclass(frozen=True)
class Generation:
    text: str  # completion cut at the first stop string: what extract.py sees
    raw_text: str  # everything generated up to the end token or the cap, for the trace
    n_tokens: int  # tokens actually generated (raw)
    finish_reason: str  # "stop" (stop string) | "eos" (end token) | "length" (hit the cap)


def cut_at_stop(text: str, stops: list[str]) -> tuple[str, bool]:
    """Truncate text at the earliest occurrence of any stop string.

    Returns (truncated_text, found). The backends ask the model to halt on these
    strings, but the halt happens a token late and, in a batch, other sequences keep
    going, so the decoded text can run past the stop. This makes the cut exact.
    """
    positions = [text.find(s) for s in stops if s in text]
    if not positions:
        return text, False
    return text[: min(positions)], True


class HFBackend:
    def __init__(self, cfg: GenConfig):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.cfg = cfg
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:  # some tokenizers ship without one
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype = getattr(torch, cfg.dtype)
        self.model = AutoModelForCausalLM.from_pretrained(cfg.model, dtype=dtype).to(self.device).eval()
        eos = self.model.generation_config.eos_token_id
        self.eos_ids = set(eos if isinstance(eos, list) else [eos])
        print(f"[hf] {cfg.model} on {self.device} in {cfg.dtype}; eos ids {sorted(self.eos_ids)}", flush=True)

    def generate(self, prompts: list[str], stops: list[str]) -> list[Generation]:
        torch, tok, cfg = self.torch, self.tokenizer, self.cfg
        # add_special_tokens=False: a base model gets the raw text, nothing prepended.
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
        n_prompt = enc["input_ids"].shape[1]
        with torch.inference_mode():
            out = self.model.generate(
                **enc,
                max_new_tokens=cfg.max_new_tokens,
                do_sample=False,  # greedy
                stop_strings=stops,  # halts a sequence once a stop string has appeared
                tokenizer=tok,  # required by stop_strings
                pad_token_id=tok.pad_token_id,
            )
        results = []
        for row in out[:, n_prompt:].tolist():
            # Cut at the first end token; anything after it is padding added for the batch.
            end = next((i for i, t in enumerate(row) if t in self.eos_ids), None)
            ids = row[:end] if end is not None else row
            raw = tok.decode(ids, skip_special_tokens=True)
            text, stopped = cut_at_stop(raw, stops)
            if stopped:
                reason = "stop"
            elif end is not None:
                reason = "eos"
            elif len(ids) >= cfg.max_new_tokens:
                reason = "length"
            else:
                reason = "stop"  # halted by stop_strings in a batch; the string was consumed
            results.append(Generation(text=text, raw_text=raw, n_tokens=len(ids), finish_reason=reason))
        if self.device == "mps":
            torch.mps.empty_cache()
        return results


class VLLMBackend:
    def __init__(self, cfg: GenConfig):
        from vllm import LLM

        self.cfg = cfg
        self.llm = LLM(model=cfg.model, dtype=cfg.dtype, max_model_len=cfg.max_model_len, gpu_memory_utilization=0.90)
        self.tokenizer = self.llm.get_tokenizer()
        print(f"[vllm] {cfg.model} in {cfg.dtype}, max_model_len {cfg.max_model_len}", flush=True)

    def generate(self, prompts: list[str], stops: list[str]) -> list[Generation]:
        from vllm import SamplingParams

        sampling = SamplingParams(temperature=0, max_tokens=self.cfg.max_new_tokens, stop=stops)
        outs = self.llm.generate(prompts, sampling, use_tqdm=False)
        results = []
        for o in outs:
            c = o.outputs[0]
            text, _ = cut_at_stop(c.text, stops)  # vllm already excludes the stop string; harmless
            if c.finish_reason == "length":
                reason = "length"
            elif c.stop_reason is None:  # finish_reason "stop" with no stop string = end token
                reason = "eos"
            else:
                reason = "stop"
            results.append(Generation(text=text, raw_text=c.text, n_tokens=len(c.token_ids), finish_reason=reason))
        return results


def make_backend(cfg: GenConfig):
    if cfg.backend == "hf":
        return HFBackend(cfg)
    if cfg.backend == "vllm":
        return VLLMBackend(cfg)
    raise ValueError(f"unknown backend {cfg.backend!r}; choose 'hf' or 'vllm'")


if __name__ == "__main__":
    # Smoke check: python -m src.generate   (from tom/, with torch installed and the model cached)
    # One German item, native_cot, 6-shot, on the local backend.
    import time

    from .data import load_mgsm
    from .prompts import PromptBuilder, PromptSpec

    builder = PromptBuilder()
    item = load_mgsm("de")[0]
    prompt = builder.build(item, PromptSpec("native_cot"))
    stops = builder.stop_sequences("de")

    backend = make_backend(GenConfig())
    n_prompt_tokens = len(backend.tokenizer(prompt, add_special_tokens=False)["input_ids"])
    t0 = time.time()
    [gen] = backend.generate([prompt], stops)
    dt = time.time() - t0
    print(f"prompt: {n_prompt_tokens} tokens; generated {gen.n_tokens} tokens in {dt:.1f}s "
          f"({gen.n_tokens / dt:.1f} tok/s); finish_reason={gen.finish_reason}")
    print("--- completion (cut at stop) ---")
    print(repr(gen.text))
    print("--- raw ---")
    print(repr(gen.raw_text))
    print("gold:", item.answer)

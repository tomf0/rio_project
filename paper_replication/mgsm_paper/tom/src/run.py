"""Run one config end to end: build prompts, generate, extract, append rows to a JSONL.

    python -m src.run config/local_smoke.yaml

One row per (language, cell, item). The file is the source of truth: on restart,
rows already present are skipped, so a killed run resumes where it stopped. Every
row keeps the full prompt and the raw completion, so any number in any later table
can be traced back to the text that produced it.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import yaml

from .data import load_mgsm
from .extract import Extraction, extract_answer, is_correct
from .generate import GenConfig, make_backend
from .prompts import PromptBuilder, PromptSpec


@dataclass(frozen=True)
class RunConfig:
    name: str
    model: str
    backend: str
    dtype: str
    max_new_tokens: int
    batch_size: int
    languages: list[str]
    cells: list[str]
    exemplar_set: str
    n_shot: int
    n_items: int | None  # None = all 250
    output: str
    max_model_len: int = 8192  # vllm only; optional in the YAML

    @classmethod
    def from_yaml(cls, path: str | Path, overrides: list[str] = ()) -> "RunConfig":
        """Load a YAML config; `overrides` are "key=value" strings from the command line.

        Values are parsed as YAML, so `n_items=3`, `backend=hf` and `languages=[de,sw]`
        all work. Used by the Kaggle notebook to run a small check before the full run.
        """
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        for item in overrides:
            key, _, value = item.partition("=")
            if key not in raw:
                raise KeyError(f"unknown config key {key!r}")
            raw[key] = yaml.safe_load(value)
        return cls(**raw)  # a missing or misspelt key fails here, loudly

    def gen_config(self) -> GenConfig:
        return GenConfig(model=self.model, backend=self.backend, dtype=self.dtype,
                         max_new_tokens=self.max_new_tokens, batch_size=self.batch_size,
                         max_model_len=self.max_model_len)


def load_done(path: Path) -> set[tuple[str, str, int]]:
    """Keys of rows already in the output file, for resuming."""
    done = set()
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                done.add((row["lang"], row["cell"], row["item_id"]))
    return done


def solution_language(cell: str, lang: str) -> str:
    """The language the model was asked to write its solution in."""
    return "en" if cell == "en_cot" else lang


def run(cfg: RunConfig) -> None:
    out_path = Path(cfg.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out_path)
    print(f"[run] {cfg.name}: {len(done)} rows already done in {out_path}", flush=True)

    builder = PromptBuilder()
    backend = make_backend(cfg.gen_config())
    tokenizer = backend.tokenizer

    with out_path.open("a", encoding="utf-8") as out:
        for lang in cfg.languages:
            items = load_mgsm(lang)[: cfg.n_items]
            stops = builder.stop_sequences(lang)
            for cell in cfg.cells:
                spec = PromptSpec(cell, exemplar_set=cfg.exemplar_set, n_shot=cfg.n_shot)
                todo = [it for it in items if (lang, cell, it.item_id) not in done]
                if not todo:
                    continue
                t0 = time.time()
                n_correct = 0
                # Batches of batch_size prompts, all same language and cell, so one stop list.
                for start in range(0, len(todo), cfg.batch_size):
                    batch = todo[start : start + cfg.batch_size]
                    prompts = [builder.build(it, spec) for it in batch]
                    gens = backend.generate(prompts, stops)
                    for item, prompt, gen in zip(batch, prompts, gens):
                        try:
                            ex = extract_answer(gen.text, solution_language(cell, lang), cell, question_lang=lang)
                        except Exception as err:  # one odd completion must never kill the run
                            print(f"[run] extraction error on {lang} {cell} item {item.item_id}: {err!r}", flush=True)
                            ex = Extraction(None, "error")
                        correct = is_correct(ex.value, item.answer)
                        n_correct += bool(correct)
                        row = {
                            "run": cfg.name,
                            "model": cfg.model,
                            "lang": lang,
                            "cell": cell,
                            "exemplar_set": cfg.exemplar_set,
                            "n_shot": cfg.n_shot,
                            "item_id": item.item_id,
                            "question": item.question,
                            "gold": item.answer,
                            "prompt": prompt,
                            "n_prompt_tokens": len(tokenizer(prompt, add_special_tokens=False)["input_ids"]),
                            "completion": gen.text,
                            "raw_completion": gen.raw_text,
                            "n_tokens": gen.n_tokens,
                            "finish_reason": gen.finish_reason,
                            "extracted_answer": ex.value,
                            "extraction_method": ex.method,
                            "correct": correct,
                        }
                        out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    out.flush()  # a killed run loses at most the current batch
                print(f"[run] {lang} {cell:11} {len(todo):4d} items  {n_correct:4d} correct  "
                      f"{time.time() - t0:6.1f}s", flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python -m src.run config/<name>.yaml [key=value ...]")
    run(RunConfig.from_yaml(sys.argv[1], sys.argv[2:]))

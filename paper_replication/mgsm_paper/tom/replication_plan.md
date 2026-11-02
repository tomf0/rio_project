# Replication plan: "Language Models are Multilingual Chain-of-Thought Reasoners"

Shi, Suzgun, Freitag, Wang, Srivats, Vosoughi, Chung, Tay, Ruder, Zhou, Das, Wei (2022).
arXiv:2210.03057. ICLR 2023. Benchmark and exemplars: github.com/google-research/url-nlp (`mgsm/`).

Written 2026-11-02, before any code or runs. This is the pre-registration: the claims,
the success criteria and the known departures from the paper are fixed here first, so that
later results are read against a plan rather than the plan being read off the results.

## 1. What the paper claims

Setting: MGSM, 250 GSM8K test problems hand-translated into 10 languages (DE, FR, ES, RU, ZH,
JA, TH, TE, BN, SW) plus English. Few-shot prompting (6 exemplars where they fit), greedy
decoding, answer is an integer. Models: GPT-3 text-davinci-002 and PaLM 8B/62B/540B.

Four solution strategies, crossed with three exemplar choices (Table 2):

| | DIRECT | NATIVE-COT | EN-COT | TRANSLATE-EN |
|---|---|---|---|---|
| Native exemplars (question + solution in the test language) | x | x | x | x |
| English exemplars | x | n/a | x | n/a |
| Multilingual exemplars (one each from EN, DE, FR, ES, RU, ZH) | x | x | x | n/a |

The claims we will test, with the paper's reference numbers (PaLM-540B unless stated):

| # | Claim | Paper's evidence |
|---|---|---|
| C1 | Chain of thought beats direct answering in every language. | NATIVE-COT 48.1 vs DIRECT 18.6 average; holds per language (Table 3). |
| C2 | Reasoning in English is at least as good as reasoning in the question's language. | EN-COT 51.3 vs NATIVE-COT 48.1; EN-COT >= NATIVE-COT in 8 of 10 non-English languages (ZH and TH the exceptions). |
| C3 | Translating the question to English and reasoning in English is best. | TRANSLATE-EN 55.0 (Table 3). |
| C4 | Accuracy is only weakly tied to how much of the language was in pretraining. | Under-represented 44.9 vs high-resource 47.9 at 540B (Fig. 1, Table 3), despite a 1000x frequency range. The gap is much larger for GPT-3 (13.9 vs 39.4 under EN-COT). |
| C5 | Multilingual reasoning emerges with scale. | PaLM 8B 4.0, 62B 20.0, 540B 48.1 average NATIVE-COT (Table 8, Fig. 4). |
| C6 | Multilingual exemplars beat English-only exemplars when reasoning in English. | EN-COT: 48.7 with multilingual exemplars vs 34.7 with English exemplars (Table 4). |

Out of scope: Section 5 (XCOPA, XL-WiC). The pipeline should not make that impossible later.

## 2. Departures from the paper, and why

| Paper | Here | Reason |
|---|---|---|
| text-davinci-002, PaLM 8B/62B/540B | An open-weight *base* model family at 4 sizes (Section 3) | Both original models are shut down (OpenAI Jan 2024, Google Aug 2024). Replication on a successor API model is possible for under $10 but is a chat model at ceiling on GSM8K, and the project rule is free compute only. Base models prompted few-shot are the closest analogue to PaLM. |
| Language frequency from PaLM's own pretraining counts | The paper's high-resource / under-represented grouping only; no frequency axis | Open models do not publish per-language token counts, and a public proxy is not the model's mix. We keep the grouping (which the paper derived from mC4) and drop the Figure 1 scatter. |
| TRANSLATE-EN via Google Translate API (2022) | Google Translate web interface (document upload, one file per language), driven by a browser agent, output committed as data | Free. Different MT system from 2022, so TRANSLATE-EN numbers are not comparable in absolute terms. MGSM's English originals give an oracle-translation upper bound the paper did not have; both are run. |
| Greedy decoding | Greedy decoding | Same. Pilot found greedy makes *thinking* models loop; that does not apply to base models with stop sequences. |
| Per-language exemplar counts for GPT-3 (context limit) | 6 exemplars everywhere | Modern context windows remove the constraint. |

Not a departure, but a difference in the world: GSM8K and MGSM are in the training data of
every modern model. Absolute accuracies will be far above the paper's and the "emergence"
floor (C5) may not exist at the smallest size. We report this rather than tune around it.

## 3. Models

Decided 2026-11-02: Qwen3-Base at 0.6B / 1.7B / 4B / 8B (`Qwen/Qwen3-0.6B-Base` etc.).
Base checkpoints, so no chat template and no thinking mode: plain few-shot completion. The
pilot already used the Qwen3 tokenizer, and the 119-language pretraining mix covers all 11
MGSM languages. Qwen2.5 was the alternative (narrower language coverage).

Given the two-week deadline (Section 7), pass 1 runs 0.6B and 1.7B. 4B and 8B are stretch
goals, decided after the 1.7B results are in; without them C5 is a two-point curve.

float16 on Kaggle. The 7-8B model needs either the 2xT4 Kaggle instance with vLLM tensor
parallelism or a 4-bit quantised checkpoint on one T4; quantisation is a confound to note.
Instruct checkpoints of the same sizes are an optional secondary run, not part of the plan.

## 4. Experimental grid

Per model: 11 languages x 250 items x the cells below. Decided 2026-11-02: the first pass is
the core table (Table 3); the Table 4 exemplar variants are a second pass once the pipeline
is trusted on real output.

| Pass | Cell | Exemplars | Generations |
|---|---|---|---|
| 1 | DIRECT, NATIVE-COT, EN-COT | native | 3 x 2,750 = 8,250 |
| 1 | TRANSLATE-EN (oracle: MGSM English originals) | native | 0 extra: identical to the EN column of EN-COT |
| 1, if time | TRANSLATE-EN (machine translation: Google Translate web interface, document upload, driven by Claude in Chrome) | native | 2,500 (non-English only) |
| 2 | NATIVE-COT, EN-COT | multilingual | 2 x 2,750 = 5,500 |
| 2 | EN-COT | English | 2,500 (non-English only) |

About 8,250 generations per model in pass 1 (10,750 with machine translation), 8,000 more
in pass 2, roughly 1,500 prompt tokens and up to 256 output tokens
each (32 for DIRECT). Stop at a blank line or the next question marker in the prompt language.

Every row records: model, language, cell, item id, full prompt, raw completion, parsed answer,
gold, correct flag, truncated flag, completion token count, and for the COT cells the detected
language of the reasoning (per-sentence, reusing the pilot's `lingua` check), so that C2 is
read only where the model actually reasoned in the requested language.

## 5. Statistics

Priority (Tom, 2026-11-02): point estimates first. The first deliverable is the paper's
Table 3 reproduced as accuracies per language x cell, with HRL/URL/average columns, for each
model size. Uncertainty quantification comes after that table exists and has been read.

Paired design throughout: every item appears in every cell, so comparisons between cells are
paired differences over 250 items (2,750 when pooled across languages). Because the runner
stores one row per item x cell, the CIs below can be added to the analysis script later
without touching the data.

- Accuracy per language x cell. Later: percentile bootstrap CI over items (B = 10,000).
- Differences between cells as paired differences in accuracy. Later: paired bootstrap CI
  and the discordant counts (items right in one cell, wrong in the other).
- High-resource vs under-represented averages use the paper's grouping: HRL = DE, FR, ES, RU,
  ZH, JA; URL = TH, TE, BN, SW.
- C4 is the URL-minus-HRL difference in mean accuracy; no frequency axis (Section 2).
  Later: bootstrapped over items within each group.
- C5 is descriptive: accuracy vs model size per language, no test.
- Greedy decoding is deterministic, so one run per cell; vLLM batching can flip a handful of
  tokens, and we note the vLLM version and batch settings.
- NATIVE-COT accuracy is reported two ways: on all items (the headline, intention-to-treat)
  and on the items where the per-sentence language check says the reasoning was actually in
  the requested language (per-protocol), together with the compliance rate per language.
  Per-protocol is a selected subset, so it is a secondary number. The paper did not measure
  compliance at all; it relied on the six native exemplars to fix the output language, and
  the only trace of the problem is the collapse of NATIVE-COT with multilingual exemplars in
  French and Spanish (7.2 and 10.4, Table 4), which it reports without diagnosing.
- All secondary cuts (truncation, parse failures, compliance) are reported, not hidden.

## 6. Success criteria

Decided in advance; each is read at the largest model unless stated. In the first pass each
criterion is read on point estimates; where a row says "CI", read it as "sign of the
difference" until the bootstrap is added, and say so in the write-up.

| # | Replicates if | Partial if | Fails if |
|---|---|---|---|
| C1 | NATIVE-COT minus DIRECT > 0 with CI excluding 0 in all 11 languages. | In >= 9 languages. | Otherwise, or COT hurts anywhere. |
| C2 | Pooled non-English EN-COT minus NATIVE-COT >= 0 with CI not entirely below 0, and >= 7 of 10 languages individually >= 0. | Pooled >= 0 but fewer languages. | Pooled CI entirely below 0. |
| C3 | TRANSLATE-EN >= EN-COT pooled, CI not entirely below 0. | Oracle-translation >= EN-COT but MT version not. | Both below EN-COT. |
| C4 | URL minus HRL gap <= 10 points at the largest model, and smaller than the gap at the smallest model with non-trivial accuracy. | Gap <= 10 but not shrinking with scale. | Gap > 10 at the largest model. |
| C5 | Average accuracy increases monotonically with size and the smallest model is below 50% of the largest. | Monotone but no clear floor (expected, given contamination). | Non-monotone. |
| C6 | Multilingual-exemplar EN-COT minus English-exemplar EN-COT > 0 pooled over non-English, CI excluding 0. | > 0 but CI includes 0. | <= 0. |

Overall: "the paper's qualitative picture replicates" means C1, C2 and C6 replicate and C4 is
at least partial. C3 and C5 are reported but are the ones most changed by the model swap.
If machine translation is not run in time, C3 is read on the oracle version only and labelled
as such; if only two model sizes run, C5 is descriptive only.

## 7. Compute and schedule

Deadline: results for the RIO supervisors / team about two weeks from 2026-11-02. The write-up
is a short research report for readers who know the paper: methods, tables with CIs, caveats
first.

| Days | Work |
|---|---|
| 1-3 | Data layer (vendor TSVs + exemplars), prompt builder, answer extraction; unit tests on the exemplars themselves. |
| 4-5 | Generation backend and runner ported from the pilot; local smoke test on Qwen3-0.6B-Base, 5 items x 2 languages x 3 cells; read every output by eye. |
| 6-8 | Kaggle notebook; 0.6B pass 1 run; analysis script on its JSONL; fix extraction bugs found on real output. |
| 9-10 | 1.7B pass 1 run; decide on 4B/8B and on machine translation. |
| 11-13 | Table 3 reproduction (point estimates) for each size; compliance table; write-up draft. Bootstrap CIs only if time. |
| 14 | Buffer. |

- Local (Apple M1, 8 GB): smoke tests only, smallest model, 5 items x 2 languages x 3 cells.
- Kaggle (free, 30 GPU-hours/week, 12-hour sessions): all real runs, vLLM, see the pilot's
  `kaggle/README.md` for the install gotchas. Rough budget per model: 1-3 hours for the two
  smallest sizes, 3-5 hours for the 4B, 6-10 hours for the 8B on 2xT4. Run smallest first;
  the runner must resume from a partial JSONL.
- Translation step (Section 2) is a one-off manual task, about an afternoon.

## 8. Decisions still open

Settled 2026-11-02: model family (Qwen3-Base, 0.6B and 1.7B first), pass-1 scope (core
table), TRANSLATE-EN (oracle in pass 1, machine translation via the web interface if time),
no frequency axis for C4, compliance reported both ways, two-week deadline, audience = RIO
supervisors.

Also settled 2026-11-02: code lives in this folder, mirroring the root proposal; pilot code is
copied in and adapted piece by piece rather than imported, so this repo stands alone.

```
paper_replication/mgsm_paper/tom/
  replication_plan.md   this file
  data/mgsm/            vendored TSVs + exemplars.py from url-nlp, plus translated questions
  config/               one YAML per run (model, cells, languages, decoding)
  src/                  data.py  prompts.py  generate.py  extract.py  run.py
  analysis/             statistics and plots
  results/              raw JSONL, committed if small
  kaggle/               notebook and notes for the GPU runs
  sandbox/              annotated drafts Tom transcribes by hand; not imported by anything
```

## 9. Caveats to carry into the write-up

- One model family; family-specific training choices (Qwen's heavy maths data) are confounded
  with scale and with language.
- Contamination inflates absolute numbers and may compress language gaps; it can bias C2 and
  C4 in either direction.
- Token counts are not comparable across languages (pilot measured ~1.1 tokens/word English vs
  ~2.6 Swahili); truncation censors verbose languages harder.
- Answer extraction in 11 languages and number formats is a source of silent error; audit a
  sample by hand per language before trusting any table.
- Call this "a replication of the qualitative claims on open models", not a replication of
  the paper's numbers.

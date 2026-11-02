# Running on Kaggle

Everything here was tested locally on an M1 with Qwen3-0.6B-Base and the `hf` backend. The
`vllm` backend is exercised only on Kaggle. Adapted from the pilot's `kaggle/README.md`.

## One-time setup
1. Kaggle account with phone verification (needed for GPU and internet).
2. This repo pushed to GitHub (the notebook clones it).

## Each run
1. kaggle.com -> Create -> New Notebook -> File -> Import Notebook -> upload `kaggle/run_pass1.ipynb`.
2. Right-hand panel: Accelerator **GPU T4 x2** (x1 is enough for 0.6B/1.7B), Internet **On**.
3. Run cells 1-3 interactively and read the check table: are completions in the right language,
   do they end with a closing phrase, is `finish_reason` mostly `stop`?
4. For the main run, **Save Version -> Save & Run All (Commit)** so it runs headless (max 12 h).
5. Download `results.zip` from the notebook's **Output** tab, unzip so the JSONL lands in
   `results/kaggle/` here, commit it (a few MB), and run the analysis locally.

For a different model, copy `config/kaggle_qwen3_0.6b_pass1.yaml`, change `model`, `name` and
`output`, and point `CONFIG` in cell 3 at it. For the 8B model use T4 x2; vLLM needs
`tensor_parallel_size=2`, which is not yet wired into `GenConfig`.

## If something breaks
- `PyTorch and TorchAudio were compiled with different CUDA versions`: cell 2 removes torchaudio;
  if it recurs, `!pip uninstall -y torchaudio` in a new cell and rerun from cell 3.
- vLLM fails to install or crashes loading the model: cell 2 sets `BACKEND = "hf"` automatically
  when vLLM is absent; if it installed but crashes, set `BACKEND = "hf"` by hand and rerun cell 3.
- Out of GPU memory with hf: lower `batch_size` in the `EXTRA` string (8, then 4).
- Session died mid-run: rerun cell 4. It resumes from the rows already written.
- Many `finish_reason: length` rows: raise `max_new_tokens` in the config and rerun; only the
  missing rows are generated (delete the file first if you want the truncated rows redone).

## Budget
Free tier: about 30 GPU-hours/week, 12 h per session. Pass 1 is 8,250 generations of roughly
1,000 prompt tokens and under 200 output tokens each. Expect 1-2 h for 0.6B with vLLM.

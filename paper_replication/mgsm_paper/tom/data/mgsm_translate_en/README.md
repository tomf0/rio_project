# TRANSLATE-EN questions (machine translation step)

For the paper's TRANSLATE-EN cell (claim C3): the 250 non-English MGSM questions of each
language, machine-translated into English with Google Translate, as the paper did (the paper
used the API in 2022; we use the web interface's document mode).

## Input files
`input/mgsm_<lang>_<Language>.xlsx`, one per language: column A `item_id`, column B `question`.
Rows 0-249 are the test items; rows 1000-1007 are the 8 few-shot exemplar questions (prefix
stripped), so that machine-translated exemplars are available too. The paper's wording leaves
it open whether its TRANSLATE-EN exemplars were the human English originals or machine
translations; the config will default to the human originals (paper Appendix A.2). Generated from the vendored TSVs by the snippet in the project log (2026-11-04).

## How to translate (desktop browser, about 20-30 minutes for all ten)
1. Open https://translate.google.com/?op=docs
2. Source language: pick the file's language (or Detect). Target language: English.
3. Browse / upload one `input/*.xlsx` file. Click Translate. Click Download translation.
4. Save the downloaded file into `output/` with the SAME name as the input file.
5. Repeat for the other nine files.

## Checks before use
- Each output file has 259 rows (header + 250 items + 8 exemplars), ids in order, column B all English.
- Numbers in the questions must be unchanged (spot-check a few against the input).
- `python -m src.translated` (to be written) validates all ten and reports any gaps.

Not committed: the translated files are data derived from a third-party service; keep them
local like the results files, or commit if the team wants them shared (they are small).

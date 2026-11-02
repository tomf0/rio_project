"""Load the MGSM benchmark and its few-shot exemplars from the vendored url-nlp files.

Data files live in ../data/mgsm/ and were copied from
https://github.com/google-research/url-nlp/tree/main/mgsm on 2026-11-02:

  mgsm_<lang>.tsv   250 rows, two tab-separated columns: question, integer answer.
                    No header. Row i is the same GSM8K problem in every language.
  exemplars.py      a nested dict MGSM_EXEMPLARS[lang][str(n)] = {"q": ..., "a": ...}
                    holding 8 few-shot exemplars per language. The strings already
                    carry the language's own prefixes, e.g. "Frage: ..." and
                    "Schritt-für-Schritt-Antwort: ... Die Antwort ist 11."

This module only *loads*. Prompt construction is prompts.py; answer parsing is extract.py.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

# Folder holding the vendored files, resolved relative to this file so that the
# loader works no matter which directory Python was started from.
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "mgsm"

# The 11 MGSM languages, in the column order the paper's Table 3 uses.
LANGUAGES = ("en", "de", "fr", "es", "ru", "zh", "ja", "th", "te", "bn", "sw")

# The paper's grouping (Table 3): HRL = more than 0.1% of PaLM's pretraining tokens.
HIGH_RESOURCE = ("de", "fr", "es", "ru", "zh", "ja")
UNDER_REPRESENTED = ("th", "te", "bn", "sw")

N_ITEMS = 250
N_EXEMPLARS = 8


@dataclass(frozen=True)
class Item:
    """One MGSM test problem in one language."""

    item_id: int  # 0..249; the same problem has the same item_id in every language
    lang: str  # two-letter code from LANGUAGES
    question: str
    answer: int  # gold answer, always an integer in MGSM


@dataclass(frozen=True)
class Exemplar:
    """One few-shot example in one language, exactly as the paper's authors wrote it."""

    index: int  # 1..8, same problem across languages
    lang: str
    question: str  # includes the language's question prefix, e.g. "Question: ..."
    solution: str  # includes the answer prefix and the closing "The answer is N." phrase
    answer: int


def parse_gold_answer(text: str) -> int:
    """Turn the answer column into an int.

    Four of the 250 answers are written with a thousands separator ("2,125",
    "114,200", "276,000", "5,600"), identically in every language file.
    """
    return int(text.strip().replace(",", ""))


def load_mgsm(lang: str, data_dir: Path = DATA_DIR) -> list[Item]:
    """Read mgsm_<lang>.tsv and return its 250 items in file order."""
    path = data_dir / f"mgsm_{lang}.tsv"
    items: list[Item] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            line = line.rstrip("\n")
            if not line:  # tolerate a trailing blank line, nothing else
                continue
            fields = line.split("\t")
            if len(fields) != 2:
                raise ValueError(f"{path.name} line {line_no + 1}: expected 2 columns, got {len(fields)}")
            question, answer = fields
            items.append(Item(item_id=len(items), lang=lang, question=question.strip(), answer=parse_gold_answer(answer)))
    if len(items) != N_ITEMS:
        raise ValueError(f"{path.name}: expected {N_ITEMS} rows, got {len(items)}")
    return items


def load_all_mgsm(langs: tuple[str, ...] = LANGUAGES, data_dir: Path = DATA_DIR) -> dict[str, list[Item]]:
    """Load every language and check the gold answers line up row by row."""
    by_lang = {lang: load_mgsm(lang, data_dir) for lang in langs}
    reference = [item.answer for item in by_lang[langs[0]]]
    for lang, items in by_lang.items():
        answers = [item.answer for item in items]
        if answers != reference:
            raise ValueError(f"gold answers in {lang} do not match {langs[0]}; files are misaligned")
    return by_lang


def _read_exemplar_dict(path: Path) -> dict:
    """Pull MGSM_EXEMPLARS out of exemplars.py without executing the file.

    ast.parse turns the source into a syntax tree; ast.literal_eval then rebuilds
    the dict from the literal on the right-hand side of the assignment. Only
    literals (strings, numbers, dicts, lists) are accepted, so no code can run.
    This matters because the file was downloaded from the internet.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and node.targets[0].id == "MGSM_EXEMPLARS":
            return ast.literal_eval(node.value)
    raise ValueError(f"MGSM_EXEMPLARS not found in {path}")


def load_exemplars(lang: str, data_dir: Path = DATA_DIR) -> list[Exemplar]:
    """Return the 8 few-shot exemplars for one language, in the authors' order (1..8).

    The gold answers for the exemplars are not stored per language in the file; they
    come from the parallel list EXEMPLAR_NUMBER_ANSWERS, which we hard-code here after
    checking it against the file once (see tests).
    """
    exemplar_answers = (11, 29, 39, 9, 33, 8, 8, 5)
    raw = _read_exemplar_dict(data_dir / "exemplars.py")[lang]
    exemplars: list[Exemplar] = []
    for n in range(1, N_EXEMPLARS + 1):
        entry = raw[str(n)]
        exemplars.append(
            Exemplar(index=n, lang=lang, question=entry["q"], solution=entry["a"], answer=exemplar_answers[n - 1])
        )
    return exemplars


if __name__ == "__main__":
    # Smoke check: python -m src.data   (run from the tom/ folder)
    data = load_all_mgsm()
    print(f"loaded {len(data)} languages x {len(data['en'])} items")
    for lang in ("en", "sw", "zh"):
        item = data[lang][0]
        print(f"  {lang} item 0: answer={item.answer} question={item.question[:60]!r}...")
    ex = load_exemplars("de")
    print(f"loaded {len(ex)} German exemplars; first solution ends: {ex[0].solution[-40:]!r}")

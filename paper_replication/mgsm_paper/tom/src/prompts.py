"""Build the paper's few-shot prompts (Shi et al. 2022, Figures 6-8).

Part 1: per-language templates. The question prefix ("Frage: ") and the
chain-of-thought prefix ("Schritt-für-Schritt-Antwort: ") are derived from the
exemplar strings; only the DIRECT answer word is hand-written. Exemplars are
used exactly as the authors published them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .data import LANGUAGES, load_exemplars, Exemplar, Item

# The one hand-written piece of language knowledge: the word before a bare-number
# answer in the DIRECT cell. Six are from the paper's Figure 6 (Russian typo
# "Antwor" corrected; Chinese "解答" kept as the paper used it); the rest are the
# noun from each language's own closing phrase, e.g. "Jibu ni 11" -> "Jibu".
DIRECT_ANSWER_WORD: dict[str, str] = {
    "en": "Answer",
    "de": "Antwort",
    "fr": "Réponse",
    "es": "Respuesta",
    "ru": "Ответ",
    "zh": "解答",
    "ja": "答え",
    "th": "คำตอบ",
    "te": "సమాధానం",
    "bn": "উত্তর",
    "sw": "Jibu",
}

# ASCII ":" or full-width "：" (Chinese, Japanese), plus any spaces after it.
# French writes "Question : "; its space before the colon stays in the prefix word.
_COLON = re.compile(r"[:：]\s*")

def split_prefix(text: str) -> tuple[str, str]:
    """Split 'Frage: Roger hat 5 ...' into ('Frage: ', 'Roger hat 5 ...').

    The first part keeps the colon and the spacing after it, so that
    prefix + body reproduces the original string exactly.
    """
    match = _COLON.search(text)              # first colon in the string, or None
    if match is None:
        raise ValueError(f"no colon found in exemplar text: {text[:40]!r}")
    return text[: match.end()], text[match.end() :]

@dataclass(frozen=True)
class LanguageTemplate:
    lang: str
    question_prefix: str   # "Frage: "   -- used before every question, exemplar or test
    cot_prefix: str        # "Schritt-für-Schritt-Antwort: " -- before chain-of-thought solutions
    direct_prefix: str     # "Antwort: " -- before bare-number answers in the DIRECT cell

def language_template(lang: str) -> LanguageTemplate:
    """Derive the template for one language from its exemplars.

    Reads all 8 exemplars, splits each question and each solution at its colon,
    and insists that all 8 agree on both prefixes. If the authors had used two
    different spellings in one language, this would raise rather than pick one.
    """
    exemplars = load_exemplars(lang)

    # A set comprehension collects the distinct prefixes. We want exactly one.
    question_prefixes = {split_prefix(e.question)[0] for e in exemplars}
    cot_prefixes = {split_prefix(e.solution)[0] for e in exemplars}
    if len(question_prefixes) != 1 or len(cot_prefixes) != 1:
        raise ValueError(f"{lang}: exemplars disagree on prefixes: {question_prefixes} / {cot_prefixes}")

    question_prefix = question_prefixes.pop()   # .pop() takes the single element out of the set
    cot_prefix = cot_prefixes.pop()

    # Build the DIRECT prefix in the same colon style as this language's CoT prefix:
    # ": " for most languages, "：" for zh/ja. _COLON.search(...).group(0) returns
    # the matched text itself, i.e. the colon plus any spaces after it.
    colon_style = _COLON.search(cot_prefix).group(0)
    direct_prefix = DIRECT_ANSWER_WORD[lang] + colon_style

    return LanguageTemplate(
        lang=lang,
        question_prefix=question_prefix,
        cot_prefix=cot_prefix,
        direct_prefix=direct_prefix,
    )

def all_templates() -> dict[str, LanguageTemplate]:
    """Templates for all 11 languages, keyed by code. Built once per run."""
    return {lang: language_template(lang) for lang in LANGUAGES}

# --- Part 2: assembling a full prompt -----------------------------------------

# The paper's solution strategies. TRANSLATE-EN is not a separate cell: it is
# en_cot run on an Item with lang="en" whose question is the translation.
CELLS = ("direct", "native_cot", "en_cot")

EXEMPLAR_SETS = ("native", "english", "multilingual")

# Paper, Figure 6, multilingual column: exemplar i is shown in language i.
MULTILINGUAL_EXEMPLAR_LANGS = ("en", "de", "fr", "es", "ru", "zh")

N_SHOT_DEFAULT = 6      # the paper's setting wherever the context allowed
SEPARATOR = "\n\n"      # blank line between blocks, as in Figures 6-8

@dataclass(frozen=True)
class PromptSpec:
    """Which cell of the paper's grid a prompt belongs to."""

    cell: str                        # one of CELLS
    exemplar_set: str = "native"     # one of EXEMPLAR_SETS
    n_shot: int = N_SHOT_DEFAULT

    def __post_init__(self) -> None:
        # Runs automatically after the dataclass constructor; the place for checks.
        if self.cell not in CELLS:
            raise ValueError(f"unknown cell {self.cell!r}; choose from {CELLS}")
        if self.exemplar_set not in EXEMPLAR_SETS:
            raise ValueError(f"unknown exemplar set {self.exemplar_set!r}; choose from {EXEMPLAR_SETS}")
        if self.exemplar_set == "english" and self.cell == "native_cot":
            # Paper, Table 2: N/A. With English exemplars there is no native solution to show.
            raise ValueError("native_cot with english exemplars is undefined (paper Table 2)")
        if self.exemplar_set == "multilingual" and self.n_shot > len(MULTILINGUAL_EXEMPLAR_LANGS):
            raise ValueError(f"multilingual set has only {len(MULTILINGUAL_EXEMPLAR_LANGS)} exemplars")
        if not 1 <= self.n_shot <= 8:
            raise ValueError("n_shot must be between 1 and 8")

class PromptBuilder:
    """Loads templates and exemplars once, then builds prompts for any item and spec."""

    def __init__(self) -> None:
        self.templates: dict[str, LanguageTemplate] = all_templates()
        self.exemplars: dict[str, list[Exemplar]] = {lang: load_exemplars(lang) for lang in LANGUAGES}

    # -- which language does exemplar number i come from? --------------------
    def exemplar_lang(self, spec: PromptSpec, item_lang: str, i: int) -> str:
        """Language of the i-th exemplar (1-based) under this exemplar set."""
        if spec.exemplar_set == "native":
            return item_lang
        if spec.exemplar_set == "english":
            return "en"
        return MULTILINGUAL_EXEMPLAR_LANGS[i - 1]          # "multilingual"

    def exemplar_block(self, spec: PromptSpec, q_lang: str, i: int) -> str:
        """Question line + solution line for exemplar i, as the paper formats them."""
        exemplar = self.exemplars[q_lang][i - 1]
        # The exemplar question already carries its own prefix ("Frage: ...").
        question_line = exemplar.question

        if spec.cell == "direct":
            # "Antwort: 11" -- the answer word in the QUESTION's language (Figure 6).
            solution_line = self.templates[q_lang].direct_prefix + str(exemplar.answer)
        elif spec.cell == "native_cot":
            # The authors' solution in the same language as the question (Figure 8).
            solution_line = exemplar.solution
        else:  # "en_cot"
            # The authors' ENGLISH solution to the same problem (Figure 7).
            solution_line = self.exemplars["en"][i - 1].solution

        return question_line + "\n" + solution_line

    # -- the prefix the model must continue from -------------------------------
    def answer_prefix(self, spec: PromptSpec, item_lang: str) -> str:
        """The bare prefix that ends the prompt, e.g. 'Antwort: ' or 'Step-by-Step Answer: '."""
        if spec.cell == "direct":
            return self.templates[item_lang].direct_prefix
        if spec.cell == "native_cot":
            return self.templates[item_lang].cot_prefix
        return self.templates["en"].cot_prefix                 # "en_cot"

    # -- the whole thing -----------------------------------------------------
    def build(self, item: Item, spec: PromptSpec) -> str:
        """The full prompt string: n_shot exemplar blocks, then the test question and prefix."""
        blocks = [
            self.exemplar_block(spec, self.exemplar_lang(spec, item.lang, i), i)
            for i in range(1, spec.n_shot + 1)
        ]
        test_block = (
            self.templates[item.lang].question_prefix + item.question
            + "\n"
            + self.answer_prefix(spec, item.lang).rstrip()
        )
        # .rstrip(): the prompt ends at the colon. The model generates the space
        # itself. Ending a prompt with a trailing space tends to hurt tokenizer
        # alignment, because " Roger" is usually one token that includes the space.
        return SEPARATOR.join(blocks + [test_block])

    # -- where generation should stop ------------------------------------------
    def stop_sequences(self, item_lang: str) -> list[str]:
        """Strings that mean 'the answer is finished': a blank line, or the model
        starting to invent a new question in the test language or in English."""
        stops = [SEPARATOR, self.templates[item_lang].question_prefix.rstrip()]
        en_q = self.templates["en"].question_prefix.rstrip()
        if en_q not in stops:
            stops.append(en_q)
        return stops


if __name__ == "__main__":
    # Smoke check: python -m src.prompts   (from tom/)
    from .data import load_mgsm

    print("=== templates ===")
    for lang, t in all_templates().items():
        # !r shows the quotes, so trailing spaces and full-width colons are visible
        print(f"{lang}: q={t.question_prefix!r:16} cot={t.cot_prefix!r:36} direct={t.direct_prefix!r}")
    print()

    builder = PromptBuilder()
    item = load_mgsm("de")[0]                      # Janet's ducks, in German

    print("=== de, native_cot, 2-shot (shortened for display) ===")
    print(builder.build(item, PromptSpec("native_cot", n_shot=2)))
    print()
    print("=== last line of the prompt in each cell (de) ===")
    for cell in CELLS:
        print(f"{cell:11} ...{builder.build(item, PromptSpec(cell)).splitlines()[-1]!r}")
    print()
    print("=== multilingual exemplars, en_cot: first 30 chars of each exemplar question ===")
    spec = PromptSpec("en_cot", exemplar_set="multilingual")
    for i in range(1, 7):
        lang = builder.exemplar_lang(spec, "de", i)
        print(f"  {i} [{lang}] {builder.exemplar_block(spec, lang, i).splitlines()[0][:30]!r}")
    print()
    print("stop sequences for de:", builder.stop_sequences("de"))
    print("stop sequences for zh:", builder.stop_sequences("zh"))
    print("full 6-shot native_cot prompt length (de):", len(builder.build(item, PromptSpec("native_cot"))), "chars")

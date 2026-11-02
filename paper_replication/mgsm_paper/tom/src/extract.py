"""Turn a model completion into an integer answer (or None), recording which rule found it.

Rules: in the DIRECT cell take the first number; in the chain-of-thought cells take the
number after the last closing phrase ("The answer is", "答案是", ...), else the last
number anywhere. Digits in any script are normalised to ASCII first. The closing phrase
is looked up by the language of the SOLUTION, which is "en" under en_cot.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# Unicode category "Nd" = decimal digit in any script (১৮, ๑๘, full-width １８).
def normalise_digits(text: str) -> str:
    """Replace every decimal digit in any script with its ASCII equivalent."""
    out = []
    for ch in text:
        if unicodedata.category(ch) == "Nd":
            out.append(str(unicodedata.digit(ch)))
        else:
            out.append(ch)
    return "".join(out)

# optional minus, digits, optional thousands groups ("1,800", "114 200", "12.000"),
# optional decimals. Dot-thousands (German style) is resolved in to_number().
NUMBER = r"-?\d+(?:[ ,.]\d{3})*(?:\.\d+)?"
NUMBER_RE = re.compile(NUMBER)

# Characters allowed between the closing phrase and the number: whitespace,
# currency symbols, markdown bold, a colon (some models write "The answer is: 11").
_JUNK = r"[\s$€£*:：]*"

# Closing phrase per SOLUTION language, from the exemplar solutions. Tested against all 88.
_PHRASE = {
    "en": r"answer\s+is",
    "de": r"Antwort\s+(?:ist|lautet)",
    "fr": r"réponse\s+est",
    "es": r"respuesta\s+es",
    "ru": r"Ответ\s*[—–-]?",          # dash is optional; the model may drop it
    "zh": r"答案是",
    "ja": r"答えは",
    "th": r"คำตอบคือ",
    "te": r"సమాధానం",
    "bn": r"উত্তর\s+হল",
    "sw": r"Jibu\s+ni",
}

# Compile once: "<phrase><junk>(<number>)". The brackets make the number a
# capture group, so .findall() returns just the numbers.
CLOSING_PHRASE_RE = {
    lang: re.compile(phrase + _JUNK + "(" + NUMBER + ")", re.IGNORECASE)
    for lang, phrase in _PHRASE.items()
}

# Only digits in groups of three separated by dots ("12.000", "1.234.567"): a dot-thousands number.
_DOT_THOUSANDS = re.compile(r"^-?\d{1,3}(?:\.\d{3})+$")


def to_number(text: str) -> int | float:
    """'1,800' -> 1800, '114 200' -> 114200, '12.000' -> 12000, '18.0' -> 18, '2.5' -> 2.5."""
    text = text.replace(",", "").replace(" ", "")
    if _DOT_THOUSANDS.match(text):
        text = text.replace(".", "")
    value = float(text)
    return int(value) if value.is_integer() else value

@dataclass(frozen=True)
class Extraction:
    value: int | float | None   # None = nothing numeric found
    method: str | None          # "closing_phrase" | "closing_phrase_native" | "first_number" | "last_number" | None

def extract_answer(completion: str, solution_lang: str, cell: str, question_lang: str | None = None) -> Extraction:
    """Find the model's final answer in one completion.

    completion     the generated text, already truncated at the stop sequence
    solution_lang  language the SOLUTION was requested in ("en" for en_cot,
                   the item language for native_cot and direct)
    cell           "direct" | "native_cot" | "en_cot"
    question_lang  language of the question; used only when it differs from
                   solution_lang (i.e. en_cot on a non-English item)

    Rules, in order:
      direct cell   the completion should be just the number, so take the FIRST
                    number. (A base model may ramble after it; the first is the answer.)
      cot cells     1. the LAST closing phrase in solution_lang, then its number;
                    2. the LAST closing phrase in question_lang (the model slipped
                       into the question's language), recorded separately;
                    3. else the LAST number anywhere (recorded as a fallback).
    """
    text = normalise_digits(completion)

    if cell == "direct":
        match = NUMBER_RE.search(text)
        if match:
            return Extraction(to_number(match.group(0)), "first_number")
        return Extraction(None, None)

    hits = CLOSING_PHRASE_RE[solution_lang].findall(text)
    if hits:
        return Extraction(to_number(hits[-1]), "closing_phrase")

    if question_lang is not None and question_lang != solution_lang:
        hits = CLOSING_PHRASE_RE[question_lang].findall(text)
        if hits:
            return Extraction(to_number(hits[-1]), "closing_phrase_native")

    numbers = NUMBER_RE.findall(text)
    if numbers:
        return Extraction(to_number(numbers[-1]), "last_number")

    return Extraction(None, None)

def is_correct(value: int | float | None, gold: int) -> bool | None:
    """True/False when an answer was extracted; None when it was not.

    Keeping None distinct lets the analysis report 'unparseable' separately from
    'wrong' (the headline accuracy counts both as incorrect; see plan section 5).
    """
    if value is None:
        return None
    return abs(float(value) - float(gold)) < 1e-6


if __name__ == "__main__":
    # Smoke check: python -m src.extract   (from tom/). Runs the extractor over all 88
    # exemplar solutions (model-perfect completions) and a few awkward hand-made cases.
    from .data import LANGUAGES, load_exemplars
    from .prompts import split_prefix

    failures = 0
    for lang in LANGUAGES:
        for e in load_exemplars(lang):
            body = split_prefix(e.solution)[1]          # drop "Step-by-Step Answer: "
            got = extract_answer(body, lang, "native_cot")
            ok = got.value == e.answer and got.method == "closing_phrase"
            failures += not ok
            if not ok:
                print(f"FAIL {lang} {e.index}: got {got} want {e.answer}  ...{body[-30:]!r}")
    print(f"exemplar solutions: {88 - failures}/88 extracted via closing phrase")

    cases = [
        (" 18", "en", "direct", 18),
        (" $1,800. Question: next", "en", "direct", 1800),
        ("... so she makes 18 dollars. The answer is 18.", "en", "en_cot", 18),
        ("... The answer is: $2,125.", "en", "en_cot", 2125),
        ("... 答えは１８です。", "ja", "native_cot", 18),
        ("... উত্তর হল ১৮।", "bn", "native_cot", 18),
        ("... คำตอบคือ ๑๘", "th", "native_cot", 18),
        ("... Die Antwort lautet 29.", "de", "native_cot", 29),
        ("... Ответ 33.", "ru", "native_cot", 33),
        ("... so 16 - 3 - 4 = 9 eggs, 9 * 2 = 18 dollars.", "en", "en_cot", 18),   # fallback
        ("I don't know.", "en", "en_cot", None),
    ]
    for completion, lang, cell, want in cases:
        got = extract_answer(completion, lang, cell)
        flag = "ok  " if got.value == want else "FAIL"
        print(f"{flag} {got.method or '-':15} {got.value!s:6} <- {completion!r}")

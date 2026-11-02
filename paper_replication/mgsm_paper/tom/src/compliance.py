"""Which language is a chain-of-thought solution actually written in?

Ported from the pilot (llm_mini_replication/src/compliance.py), widened from 4 to the
11 MGSM languages. The method: split the completion into sentence-like segments, drop
segments that are mostly digits and symbols (arithmetic has no language), label each
remaining segment with lingua restricted to the 11 languages, and aggregate by letter
count. The result says which language dominates, what fraction was in the requested
language, and whether that counts as compliant.

Restricting the detector to 11 languages makes it much better at telling them apart,
at the cost of mislabelling drift into a 12th language; low-confidence segments are
labelled "unk" to soften that. Validated on the 88 exemplar solutions: 88/88 correct.
"""

from __future__ import annotations

import re
from functools import lru_cache

from lingua import Language, LanguageDetectorBuilder

LANG_CODES = {
    Language.ENGLISH: "en", Language.GERMAN: "de", Language.FRENCH: "fr", Language.SPANISH: "es",
    Language.RUSSIAN: "ru", Language.CHINESE: "zh", Language.JAPANESE: "ja", Language.THAI: "th",
    Language.TELUGU: "te", Language.BENGALI: "bn", Language.SWAHILI: "sw",
}

# Sentence-ish boundaries: Latin punctuation followed by whitespace; CJK punctuation and the
# Bengali danda (no whitespace follows them); newlines. Thai has no sentence punctuation, so a
# Thai completion is usually one segment, which is fine for labelling.
SPLIT_RE = re.compile(r"(?<=[.!?;])\s+|(?<=[。！？；।॥])|\n+")
LETTER_RE = re.compile(r"[^\W\d_]")  # Unicode letters in any script; no digits or underscore

MIN_LETTERS = 12  # shorter segments are too noisy to label
MIN_CONFIDENCE = 0.6  # below this the segment is "unk"
COMPLY_THRESHOLD = 0.5  # compliant = requested language dominates and covers >= half the letters


@lru_cache(maxsize=1)
def _detector():
    # Built once on first use (it loads language models) and cached.
    return LanguageDetectorBuilder.from_languages(*LANG_CODES).build()


def segments(text: str) -> list[str]:
    """Sentence-like chunks with enough letters to carry a language signal.

    If no chunk reaches MIN_LETTERS (short Chinese sentences, for instance), the whole
    text is labelled as one segment instead of giving up.
    """
    segs = [s.strip() for s in SPLIT_RE.split(text) if len(LETTER_RE.findall(s)) >= MIN_LETTERS]
    if not segs and LETTER_RE.search(text):
        segs = [text.strip()]
    return segs


def label_segment(seg: str) -> tuple[str, float]:
    """(language code or 'unk', confidence of the top language)."""
    values = _detector().compute_language_confidence_values(seg)
    if not values:
        return "unk", 0.0
    top = values[0]
    code = LANG_CODES.get(top.language, "unk")
    return (code if top.value >= MIN_CONFIDENCE else "unk"), float(top.value)


def language_profile(text: str, target: str) -> dict:
    """Language profile of one completion against the language it was requested in.

    Returns dominant_lang, target_frac (share of letters in the target language),
    lang_fracs (all shares), complied (bool), n_segments, and the first segment in a
    language other than the target, for eyeballing. All None if nothing was labellable.
    """
    weights: dict[str, int] = {}
    labelled = []
    for seg in segments(text):
        code, conf = label_segment(seg)
        n = len(LETTER_RE.findall(seg))
        weights[code] = weights.get(code, 0) + n
        labelled.append((code, conf, seg))
    # "unk" segments (low confidence, usually arithmetic) carry no language signal, so
    # dominance and the target share are computed over confidently labelled letters only.
    unk = weights.pop("unk", 0)
    total = sum(weights.values())
    if total == 0:
        return {"dominant_lang": None, "target_frac": None, "lang_fracs": {}, "complied": None,
                "n_segments": len(labelled), "first_noncompliant": None}
    fracs = {k: round(v / total, 3) for k, v in sorted(weights.items(), key=lambda kv: -kv[1])}
    if unk:
        fracs["unk"] = round(unk / (total + unk), 3)  # reported, not counted
    dominant = next(iter(fracs))
    target_frac = fracs.get(target, 0.0)
    first_bad = next((seg for code, _, seg in labelled if code not in (target, "unk")), None)
    return {
        "dominant_lang": dominant,
        "target_frac": target_frac,
        "lang_fracs": fracs,
        "complied": dominant == target and target_frac >= COMPLY_THRESHOLD,
        "n_segments": len(labelled),
        "first_noncompliant": first_bad[:120] if first_bad else None,
    }


if __name__ == "__main__":
    # Smoke check: python -m src.compliance   (from tom/). Exemplar solutions must all comply.
    from .data import LANGUAGES, load_exemplars
    from .prompts import split_prefix

    bad = 0
    for lang in LANGUAGES:
        for e in load_exemplars(lang):
            p = language_profile(split_prefix(e.solution)[1], lang)
            bad += not p["complied"]
    print(f"exemplar solutions compliant with their own language: {88 - bad}/88")
    mixed = " Janets Enten legen 16 Eier pro Tag. That means she has 16 - 3 - 4 = 9 eggs left. The answer is 18."
    print("mixed German/English sample vs target en:", language_profile(mixed, "en"))

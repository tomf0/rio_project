"""Checks on the prompt builder. Run from tom/: python -m pytest tests"""

import pytest

from src.data import LANGUAGES, load_exemplars, load_mgsm
from src.prompts import (
    CELLS,
    DIRECT_ANSWER_WORD,
    MULTILINGUAL_EXEMPLAR_LANGS,
    SEPARATOR,
    PromptBuilder,
    PromptSpec,
    all_templates,
    split_prefix,
)


@pytest.fixture(scope="module")
def builder():
    return PromptBuilder()  # loads everything once for the whole test module


def test_split_prefix_round_trips_every_exemplar_string():
    for lang in LANGUAGES:
        for e in load_exemplars(lang):
            for text in (e.question, e.solution):
                prefix, body = split_prefix(text)
                assert prefix + body == text
                assert prefix.rstrip().endswith((":", "："))


def test_every_language_has_a_template_and_a_direct_word():
    templates = all_templates()
    assert set(templates) == set(LANGUAGES) == set(DIRECT_ANSWER_WORD)
    for t in templates.values():
        assert t.question_prefix.rstrip().endswith((":", "："))
        assert t.cot_prefix.rstrip().endswith((":", "："))
        assert t.direct_prefix.startswith(DIRECT_ANSWER_WORD[t.lang])


def test_full_width_colon_for_chinese_and_japanese():
    templates = all_templates()
    for lang in ("zh", "ja"):
        assert templates[lang].question_prefix.endswith("：")
        assert templates[lang].direct_prefix.endswith("：")
    assert templates["de"].direct_prefix == "Antwort: "


def test_invalid_specs_are_rejected():
    with pytest.raises(ValueError):
        PromptSpec("chain_of_thought")
    with pytest.raises(ValueError):
        PromptSpec("direct", exemplar_set="swahili")
    with pytest.raises(ValueError):
        PromptSpec("native_cot", exemplar_set="english")  # paper Table 2: N/A
    with pytest.raises(ValueError):
        PromptSpec("en_cot", exemplar_set="multilingual", n_shot=7)
    with pytest.raises(ValueError):
        PromptSpec("direct", n_shot=0)


def test_prompt_has_n_shot_blocks_and_ends_with_bare_prefix(builder):
    item = load_mgsm("fr")[3]
    for cell in CELLS:
        for n in (1, 6, 8):
            prompt = builder.build(item, PromptSpec(cell, n_shot=n))
            blocks = prompt.split(SEPARATOR)
            assert len(blocks) == n + 1  # n exemplars + the test block
            assert blocks[-1].startswith(builder.templates["fr"].question_prefix + item.question)
            assert prompt == prompt.rstrip()  # no trailing space after the final colon
            assert prompt.endswith((":", "："))


def test_native_cot_prompt_is_entirely_in_the_item_language(builder):
    item = load_mgsm("de")[0]
    prompt = builder.build(item, PromptSpec("native_cot"))
    for e in load_exemplars("de")[:6]:
        assert e.question in prompt
        assert e.solution in prompt
    assert "Step-by-Step Answer" not in prompt
    assert prompt.endswith("Schritt-für-Schritt-Antwort:")


def test_en_cot_prompt_pairs_native_questions_with_english_solutions(builder):
    item = load_mgsm("sw")[0]
    prompt = builder.build(item, PromptSpec("en_cot"))
    for q, en in zip(load_exemplars("sw")[:6], load_exemplars("en")[:6]):
        assert q.question in prompt
        assert en.solution in prompt
        assert q.solution not in prompt  # no Swahili solutions
    assert prompt.endswith("Step-by-Step Answer:")


def test_direct_prompt_shows_bare_numbers_in_the_question_language(builder):
    item = load_mgsm("ru")[0]
    prompt = builder.build(item, PromptSpec("direct"))
    for e in load_exemplars("ru")[:6]:
        assert f"Ответ: {e.answer}" in prompt
        assert e.solution not in prompt  # no reasoning shown
    assert prompt.endswith("Ответ:")


def test_multilingual_exemplars_follow_the_paper_order(builder):
    spec = PromptSpec("en_cot", exemplar_set="multilingual")
    langs = [builder.exemplar_lang(spec, "th", i) for i in range(1, 7)]
    assert langs == list(MULTILINGUAL_EXEMPLAR_LANGS)
    prompt = builder.build(load_mgsm("th")[0], spec)
    assert prompt.startswith("Question: Roger")  # exemplar 1 in English
    assert "Frage: Es waren neun Computer" in prompt  # exemplar 2 in German


def test_english_exemplars_with_a_non_english_test_question(builder):
    item = load_mgsm("bn")[0]
    prompt = builder.build(item, PromptSpec("en_cot", exemplar_set="english"))
    assert prompt.count("Question: ") == 6
    assert prompt.split(SEPARATOR)[-1].startswith("প্রশ্ন: ")


def test_stop_sequences_include_blank_line_and_question_prefixes(builder):
    stops = builder.stop_sequences("zh")
    assert SEPARATOR in stops
    assert "问题：" in stops
    assert "Question:" in stops
    assert len(builder.stop_sequences("en")) == 2  # English prefix not duplicated

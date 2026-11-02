"""Checks on the vendored MGSM files and the loader. Run from tom/: python -m pytest tests"""

import ast
import re

from src.data import (
    DATA_DIR,
    LANGUAGES,
    HIGH_RESOURCE,
    UNDER_REPRESENTED,
    N_EXEMPLARS,
    N_ITEMS,
    load_all_mgsm,
    load_exemplars,
    load_mgsm,
    parse_gold_answer,
)


def test_language_groups_partition_the_non_english_languages():
    assert set(HIGH_RESOURCE) | set(UNDER_REPRESENTED) == set(LANGUAGES) - {"en"}
    assert not set(HIGH_RESOURCE) & set(UNDER_REPRESENTED)


def test_parse_gold_answer_handles_thousands_separators():
    assert parse_gold_answer("18") == 18
    assert parse_gold_answer("2,125") == 2125
    assert parse_gold_answer(" 276,000\n") == 276000


def test_every_language_has_250_items_with_sequential_ids():
    for lang in LANGUAGES:
        items = load_mgsm(lang)
        assert len(items) == N_ITEMS
        assert [it.item_id for it in items] == list(range(N_ITEMS))
        assert all(it.lang == lang for it in items)
        assert all(it.question for it in items)  # no empty questions


def test_gold_answers_align_across_languages():
    data = load_all_mgsm()  # raises if any language disagrees with English
    assert data["en"][0].answer == 18  # Janet's ducks
    assert data["sw"][0].answer == 18


def test_eight_exemplars_per_language_in_order():
    for lang in LANGUAGES:
        ex = load_exemplars(lang)
        assert [e.index for e in ex] == list(range(1, N_EXEMPLARS + 1))
        assert [e.answer for e in ex] == [11, 29, 39, 9, 33, 8, 8, 5]


def test_hardcoded_exemplar_answers_match_the_file():
    # load_exemplars hard-codes the answers; make sure they still match EXEMPLAR_NUMBER_ANSWERS.
    tree = ast.parse((DATA_DIR / "exemplars.py").read_text(encoding="utf-8"))
    in_file = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign) and node.targets[0].id == "EXEMPLAR_NUMBER_ANSWERS"
    )
    assert in_file == [e.answer for e in load_exemplars("en")]


def test_each_exemplar_solution_ends_with_its_answer():
    # The closing phrase differs by language ("The answer is 11.", "答案是 11。", "คำตอบคือ 11"),
    # but the final number in every solution must be the gold answer.
    for lang in LANGUAGES:
        for e in load_exemplars(lang):
            numbers = re.findall(r"\d+", e.solution)
            assert numbers, (lang, e.index)
            assert int(numbers[-1]) == e.answer, (lang, e.index, e.solution[-40:])

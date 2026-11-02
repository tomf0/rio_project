"""Checks on answer extraction. Run from tom/: python -m pytest tests"""

from src.data import LANGUAGES, load_exemplars
from src.extract import extract_answer, is_correct, normalise_digits, to_number
from src.prompts import split_prefix


def test_normalise_digits_maps_every_script_to_ascii():
    assert normalise_digits("১৮ ๑๘ １８ 18") == "18 18 18 18"
    assert normalise_digits("abc 答え") == "abc 答え"  # non-digits untouched


def test_to_number_handles_separators_and_integral_decimals():
    assert to_number("1,800") == 1800
    assert to_number("114 200") == 114200
    assert to_number("18.0") == 18 and isinstance(to_number("18.0"), int)
    assert to_number("2.5") == 2.5
    assert to_number("-3") == -3


def test_to_number_reads_dot_thousands_as_thousands():
    # German style, which the model copies even in English completions.
    assert to_number("12.000") == 12000
    assert to_number("100.000") == 100000
    assert to_number("1.234.567") == 1234567
    assert to_number("1,234.56") == 1234.56  # comma-thousands with a real decimal is untouched
    assert to_number("1234.5") == 1234.5  # four leading digits: not a thousands pattern


def test_to_number_never_raises_on_regex_matchable_shapes():
    # These shapes match NUMBER but are not valid floats; the first Kaggle run died on one.
    assert to_number("1.234.56") == 1234.56
    assert to_number("12.000.5") == 12000.5
    assert to_number("1.234.5") == 1234.5
    assert extract_answer("Das sind 1.234.56 Euro. Die Antwort ist 1.234.56.", "de", "native_cot").value == 1234.56


def test_every_exemplar_solution_is_extracted_via_its_closing_phrase():
    # 88 model-perfect completions: the phrase rule must fire and give the gold answer.
    for lang in LANGUAGES:
        for e in load_exemplars(lang):
            body = split_prefix(e.solution)[1]
            got = extract_answer(body, lang, "native_cot")
            assert got.method == "closing_phrase", (lang, e.index)
            assert got.value == e.answer, (lang, e.index)


def test_en_cot_uses_the_english_phrase_whatever_the_item_language():
    completion = " Janet sells 16 - 3 - 4 = 9 eggs for 9 * 2 = 18 dollars. The answer is 18."
    got = extract_answer(completion, "en", "en_cot")
    assert (got.value, got.method) == (18, "closing_phrase")


def test_last_closing_phrase_wins():
    completion = "The answer is 5. Wait, 5 + 6 = 11. The answer is 11."
    assert extract_answer(completion, "en", "en_cot").value == 11


def test_phrase_variants_and_junk():
    assert extract_answer("Die Antwort lautet 29.", "de", "native_cot").value == 29
    assert extract_answer("Die Antwort ist 11.", "de", "native_cot").value == 11
    assert extract_answer("Ответ 33.", "ru", "native_cot").value == 33  # dash dropped
    assert extract_answer("The answer is: **$2,125**.", "en", "en_cot").value == 2125
    assert extract_answer("答えは１８個です。", "ja", "native_cot").value == 18
    assert extract_answer("คำตอบคือ ๑๘", "th", "native_cot").value == 18


def test_dot_thousands_in_a_completion():
    assert extract_answer("Der Gewinn ist 12.000 US-Dollar. Die Antwort ist 12.000.", "de", "native_cot").value == 12000
    assert extract_answer("The answer is 52.000.", "en", "en_cot").value == 52000
    assert extract_answer(" 100.000 US-Dollar", "de", "direct").value == 100000


def test_en_cot_falls_back_to_the_question_language_phrase():
    completion = "Das ergibt 2 + 1/2 = 5 Knäuel. Die Antwort ist 5."
    got = extract_answer(completion, "en", "en_cot", question_lang="de")
    assert (got.value, got.method) == (5, "closing_phrase_native")
    # without question_lang the old behaviour remains: last number
    assert extract_answer(completion, "en", "en_cot").method == "last_number"
    # the English phrase still wins when present
    both = "Die Antwort ist 5. The answer is 3."
    assert extract_answer(both, "en", "en_cot", question_lang="de").value == 3


def test_fallback_to_last_number_is_recorded():
    got = extract_answer("So 16 - 3 - 4 = 9 eggs, and 9 * 2 = 18 dollars.", "en", "en_cot")
    assert (got.value, got.method) == (18, "last_number")


def test_direct_cell_takes_the_first_number():
    got = extract_answer(" 18\n\nQuestion: There are 3 cars", "en", "direct")
    assert (got.value, got.method) == (18, "first_number")
    assert extract_answer(" 1,800", "en", "direct").value == 1800


def test_nothing_numeric_gives_none():
    got = extract_answer("I don't know.", "en", "en_cot")
    assert (got.value, got.method) == (None, None)
    assert extract_answer("", "sw", "direct").value is None


def test_is_correct_keeps_unparsed_distinct_from_wrong():
    assert is_correct(None, 18) is None
    assert is_correct(18, 18) is True
    assert is_correct(18.0, 18) is True
    assert is_correct(2.125, 2125) is False

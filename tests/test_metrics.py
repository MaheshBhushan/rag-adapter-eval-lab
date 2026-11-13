from ragadapt.metrics import answer_in_contexts, exact_match, f1, normalize
from ragadapt.prompts import clean_answer


def test_normalize_matches_squad_rules():
    assert normalize("The  Eiffel Tower!") == "eiffel tower"


def test_exact_match_any_reference():
    assert exact_match("Denver Broncos", ["Broncos", "Denver Broncos"]) == 1.0
    assert exact_match("Broncos team", ["Denver Broncos"]) == 0.0


def test_f1_partial_overlap():
    assert f1("Denver Broncos team", ["Denver Broncos"]) == 0.8
    assert f1("nothing", ["Denver Broncos"]) == 0.0


def test_answer_in_contexts():
    assert answer_in_contexts(["Santa Clara"], ["played at Levi's Stadium in Santa Clara, California."]) == 1.0
    assert answer_in_contexts(["Paris"], ["no match here"]) == 0.0


def test_clean_answer():
    assert clean_answer(" Answer: Paris.\nmore text") == "Paris"

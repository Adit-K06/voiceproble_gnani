import pytest

from app.core.scoring import (
    normalize,
    normalize_strict,
    spacing_only_difference,
    wer,
    wer_best,
    wer_best_with_flags,
)


def test_identical_is_zero():
    assert wer("do lakh rupaye", "do lakh rupaye") == 0.0


def test_one_substitution():
    assert wer("do lakh pachaas hazaar", "do lac pachaas hazaar") == 0.25


def test_deletion():
    assert wer("do lakh", "do") == 0.5


def test_insertion():
    assert wer("do lakh", "do lakh rupaye") == 0.5


def test_punctuation_and_case_ignored():
    assert wer("Do lakh, rupaye!", "do lakh rupaye") == 0.0


def test_hindi_nukta_variants_equal():
    # U+095B (precomposed ZA) vs U+091C + U+093C (JA + nukta)
    assert normalize("\u095b") == normalize("\u091c\u093c")


@pytest.mark.parametrize(
    ("references", "hypothesis", "expected_wer", "spacing_only"),
    [
        (["हजार"], "हज़ार", 0.0, False),
        (["पाँच"], "पांच", 0.0, False),
        (["ईएमआई"], "ई एम आई", 0.0, True),
        (["Do lakh, rupaye!"], "do lakh rupaye।", 0.0, False),
        (["पचास रुपये"], "पंद्रह रुपये", 0.5, False),
        (
            [
                "आपका account balance check करने के liye star one dabaye.",
                "आपका अकाउंट बैलेंस चेक करने के लिए स्टार वन दबाएँ।",
            ],
            "आपका अकाउंट बैलेंस चेक करने के लिए स्टार वन दबाएँ।",
            0.0,
            False,
        ),
    ],
    ids=["nukta", "chandrabindu", "spacing", "punctuation-case", "real-substitution", "accepted-script"],
)
def test_lenient_scoring_cases(references, hypothesis, expected_wer, spacing_only):
    score, is_spacing_only = wer_best_with_flags(references, hypothesis)
    assert score == pytest.approx(expected_wer)
    assert is_spacing_only is spacing_only
    assert spacing_only_difference(references[0], hypothesis) is spacing_only


@pytest.mark.parametrize(
    ("reference", "hypothesis"),
    [("हजार", "हज़ार"), ("पाँच", "पांच")],
    ids=["nukta", "chandrabindu"],
)
def test_strict_normalization_preserves_spelling_distinctions(reference, hypothesis):
    assert normalize_strict(reference) != normalize_strict(hypothesis)


def test_wer_best_uses_lowest_accepted_reference():
    references = [
        "आपका account balance check करने के liye star one dabaye.",
        "आपका अकाउंट बैलेंस चेक करने के लिए स्टार वन दबाएँ।",
    ]
    hypothesis = "आपका अकाउंट बैलेंस चेक करने के लिए स्टार वन दबाएँ।"
    assert wer_best(references, hypothesis) == 0.0


def test_empty_reference():
    assert wer("", "") == 0.0

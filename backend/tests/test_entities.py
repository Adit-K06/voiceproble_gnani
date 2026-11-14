import pytest

from app.core.entities import (
    amounts_correct,
    extract_amounts,
    extract_dates,
    extract_digit_sequences,
    extract_ifsc_codes,
    extract_upi_ids,
    is_valid_ifsc,
    normalize_digit_sequence,
    normalize_upi_id,
    parse_hindi_number,
    score_entity_types,
)


HINDI_NUMBERS_0_TO_99 = """
शून्य
एक
दो
तीन
चार
पाँच
छह
सात
आठ
नौ
दस
ग्यारह
बारह
तेरह
चौदह
पंद्रह
सोलह
सत्रह
अठारह
उन्नीस
बीस
इक्कीस
बाईस
तेईस
चौबीस
पच्चीस
छब्बीस
सत्ताईस
अट्ठाईस
उनतीस
तीस
इकतीस
बत्तीस
तैंतीस
चौंतीस
पैंतीस
छत्तीस
सैंतीस
अड़तीस
उनतालीस
चालीस
इकतालीस
बयालीस
तैंतालीस
चवालीस
पैंतालीस
छियालीस
सैंतालीस
अड़तालीस
उनचास
पचास
इक्यावन
बावन
तिरपन
चौवन
पचपन
छप्पन
सत्तावन
अट्ठावन
उनसठ
साठ
इकसठ
बासठ
तिरसठ
चौंसठ
पैंसठ
छियासठ
सड़सठ
अड़सठ
उनहत्तर
सत्तर
इकहत्तर
बहत्तर
तिहत्तर
चौहत्तर
पचहत्तर
छिहत्तर
सतहत्तर
अठहत्तर
उन्नासी
अस्सी
इक्यासी
बयासी
तिरासी
चौरासी
पचासी
छियासी
सत्तासी
अट्ठासी
नवासी
नब्बे
इक्यानवे
बानवे
तिरानवे
चौरानवे
पंचानवे
छियानवे
सत्तानवे
अट्ठानवे
निन्यानवे
""".strip().splitlines()


@pytest.mark.parametrize(
    ("word", "value"),
    [(word, value) for value, word in enumerate(HINDI_NUMBERS_0_TO_99)],
)
def test_hindi_number_words_zero_through_ninety_nine(word, value):
    assert parse_hindi_number(word) == value


@pytest.mark.parametrize(
    ("phrase", "value"),
    [
        ("पन्द्रह", 15),
        ("पचहत्तर", 75),
        ("निन्यानवे", 99),
        ("दो लाख पचास हज़ार", 250000),
        ("पंद्रह हज़ार", 15000),
        ("एक करोड़ बीस लाख", 12000000),
        ("तीन हज़ार पाँच सौ", 3500),
        ("सौ", 100),
        ("एक सौ पंद्रह", 115),
        ("दो लाख पचास हज़ार", 250000),
        ("पंद्रह हज़ार", 15000),
        ("एक करोड़ बीस लाख", 12000000),
        ("तीन हज़ार पाँच सौ", 3500),
        ("पचहत्तर हज़ार", 75000),
        ("बारह लाख", 1200000),
        ("एक हज़ार दो सौ", 1200),
        ("उनतीस", 29),
        ("इकतालीस", 41),
        ("अट्ठासी", 88),
        ("सत्तानवे", 97),
        ("डेढ़ हज़ार", 1500),
        ("ढाई लाख", 250000),
        ("2 lakh 50 hazaar", 250000),
        ("15 hazaar", 15000),
    ],
)
def test_hindi_number_phrases(phrase, value):
    assert parse_hindi_number(phrase) == value


@pytest.mark.parametrize(
    ("phrase", "value"),
    [
        ("०", 0),
        ("९९", 99),
        ("३,५००", 3500),
        ("2,50,000", 250000),
        ("१२,५०,०००", 1250000),
        ("1,234,567", 1234567),
        ("२५०००", 25000),
        ("2,50,000", 250000),
    ],
)
def test_devanagari_and_grouped_digits(phrase, value):
    assert parse_hindi_number(phrase) == value


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("आपके खाते में दो लाख पचास हज़ार रुपये जमा हुए।", [250000]),
        ("पंद्रह हज़ार रुपये", [15000]),
        ("एक करोड़ बीस लाख रुपये", [12000000]),
        ("तीन हज़ार पाँच सौ रुपये", [3500]),
        ("राशि 2,50,000", [250000]),
        ("राशि ₹३५००", [3500]),
        ("बीस तारीख को भुगतान होगा", []),
        ("पिन के लिए पाँच अंक दर्ज करें", []),
    ],
)
def test_extract_amounts(text, expected):
    assert extract_amounts(text) == expected


@pytest.mark.parametrize(
    ("expected", "hypothesis", "is_correct"),
    [
        ([{"type": "amount", "value": 250000}], "दो लाख पचास हज़ार रुपये", True),
        ([{"type": "amount", "value": 3500}], "३,५०० रुपये", True),
        ([{"type": "amount", "value": 10000000}], "एक करोड़ रुपये", True),
        ([{"type": "amount", "value": 3500}], "तीन हज़ार पाँच सौ रुपये", True),
        ([{"type": "amount", "value": 3500}], "तीन हज़ार रुपये", False),
        ([], "बीस तारीख को भुगतान होगा", True),
    ],
)
def test_amounts_correct(expected, hypothesis, is_correct):
    assert amounts_correct(expected, hypothesis) is is_correct


@pytest.mark.parametrize(
    ("spoken", "digits"),
    [
        ("नौ आठ सात", "987"),
        ("नौ-आठ-सात", "९ ८-७"),
        ("1 2-3", "123"),
    ],
)
def test_digit_sequences_ignore_spaces_and_hyphens(spoken, digits):
    assert extract_digit_sequences(spoken) == ["987" if spoken.startswith("नौ") else "123"]
    assert normalize_digit_sequence(digits) == extract_digit_sequences(spoken)[0]


@pytest.mark.parametrize(
    ("expected", "hypothesis"),
    [
        ("20", "बीस तारीख"),
        ("20 November", "बीस नवंबर"),
        ("पंद्रह अप्रैल", "15 April"),
    ],
)
def test_approximate_date_scoring(expected, hypothesis):
    scores = score_entity_types(
        [{"type": "date", "value": expected}], hypothesis
    )
    assert scores["date"] is True


def test_date_extraction_handles_day_marker_and_month():
    assert extract_dates("बीस तारीख") == [(20, None)]
    assert extract_dates("20 November") == [(20, 11)]


@pytest.mark.parametrize(
    "spoken", ["rahul at oksbi", "राहुल एट ओकेएसबीआई"]
)
def test_approximate_upi_id_comparison(spoken):
    assert normalize_upi_id(spoken) == "rahul@oksbi"
    assert extract_upi_ids(spoken) == ["rahul@oksbi"]
    scores = score_entity_types(
        [{"type": "upi", "value": "rahul@oksbi"}], spoken
    )
    assert scores["upi"] is True


@pytest.mark.parametrize(
    ("value", "valid"),
    [("ABCD0001234", True), ("ABCD1001234", False), ("ABC0001234", False)],
)
def test_approximate_ifsc_shape_check(value, valid):
    assert is_valid_ifsc(value) is valid


def test_ifsc_scoring_extracts_shaped_code():
    assert extract_ifsc_codes("Synthetic IFSC: ABCD0001234.") == ["ABCD0001234"]
    assert score_entity_types(
        [{"type": "ifsc", "value": "ABCD0001234"}],
        "Synthetic IFSC: abcd0001234.",
    )["ifsc"] is True


def test_per_type_scores_are_null_when_no_entities_of_type_are_expected():
    scores = score_entity_types([], "कोई entity नहीं")
    assert scores == {
        "amount": None,
        "digit_sequence": None,
        "date": None,
        "upi": None,
        "ifsc": None,
    }


def test_per_type_entity_scoring_is_independent():
    scores = score_entity_types(
        [
            {"type": "amount", "value": 1500},
            {"type": "digit_sequence", "value": "987"},
        ],
        "नौ आठ सात; डेढ़ हज़ार",
    )

    assert scores["amount"] is True
    assert scores["digit_sequence"] is True
    assert scores["date"] is None
    assert scores["upi"] is None
    assert scores["ifsc"] is None

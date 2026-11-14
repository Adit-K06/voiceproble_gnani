"""Entity extraction and approximate scoring for VoiceProbe test transcripts."""

from __future__ import annotations

import unicodedata
from collections import Counter
from fractions import Fraction
import re


def _fold(text: str) -> str:
    """Canonicalize nukta and chandrabindu variants used in number words."""
    text = unicodedata.normalize("NFD", text)
    text = "".join(char for char in text if char != "़")
    text = text.replace("ँ", "ं")
    return unicodedata.normalize("NFC", text).lower()


_RAW_NUMBER_WORDS: dict[str, int] = {
    "शून्य": 0,
    "सिफर": 0,
    "सिफ़र": 0,
    "जीरो": 0,
    "ज़ीरो": 0,
    "एक": 1,
    "दो": 2,
    "तीन": 3,
    "चार": 4,
    "पाँच": 5,
    "पांच": 5,
    "छह": 6,
    "छः": 6,
    "सात": 7,
    "आठ": 8,
    "नौ": 9,
    "दस": 10,
    "ग्यारह": 11,
    "बारह": 12,
    "तेरह": 13,
    "चौदह": 14,
    "पंद्रह": 15,
    "पन्द्रह": 15,
    "सोलह": 16,
    "सत्रह": 17,
    "अठारह": 18,
    "उन्नीस": 19,
    "बीस": 20,
    "इक्कीस": 21,
    "बाईस": 22,
    "बाइस": 22,
    "तेईस": 23,
    "तेइस": 23,
    "चौबीस": 24,
    "चौबिस": 24,
    "पच्चीस": 25,
    "छब्बीस": 26,
    "सत्ताईस": 27,
    "सत्ताइस": 27,
    "अट्ठाईस": 28,
    "अट्ठाइस": 28,
    "उनतीस": 29,
    "उन्तीस": 29,
    "तीस": 30,
    "इकतीस": 31,
    "इकत्तीस": 31,
    "बत्तीस": 32,
    "तैंतीस": 33,
    "तैतीस": 33,
    "चौंतीस": 34,
    "चौतीस": 34,
    "पैंतीस": 35,
    "पैतीस": 35,
    "छत्तीस": 36,
    "सैंतीस": 37,
    "सैतीस": 37,
    "अड़तीस": 38,
    "अड़तीस": 38,
    "उनतालीस": 39,
    "उन्तालीस": 39,
    "चालीस": 40,
    "इकतालीस": 41,
    "बयालीस": 42,
    "तैंतालीस": 43,
    "तैतालीस": 43,
    "चवालीस": 44,
    "चौवालीस": 44,
    "पैंतालीस": 45,
    "पैतालीस": 45,
    "छियालीस": 46,
    "सैंतालीस": 47,
    "सैतालीस": 47,
    "अड़तालीस": 48,
    "अड़तालीस": 48,
    "उनचास": 49,
    "उनन्चास": 49,
    "पचास": 50,
    "इक्यावन": 51,
    "इकावन": 51,
    "बावन": 52,
    "तिरपन": 53,
    "चौवन": 54,
    "पचपन": 55,
    "छप्पन": 56,
    "सत्तावन": 57,
    "अट्ठावन": 58,
    "उनसठ": 59,
    "साठ": 60,
    "इकसठ": 61,
    "बासठ": 62,
    "तिरसठ": 63,
    "चौंसठ": 64,
    "चौसठ": 64,
    "पैंसठ": 65,
    "पैसठ": 65,
    "छियासठ": 66,
    "सड़सठ": 67,
    "सड़सठ": 67,
    "अड़सठ": 68,
    "अड़सठ": 68,
    "उनहत्तर": 69,
    "सत्तर": 70,
    "इकहत्तर": 71,
    "बहत्तर": 72,
    "तिहत्तर": 73,
    "चौहत्तर": 74,
    "पचहत्तर": 75,
    "छिहत्तर": 76,
    "सतहत्तर": 77,
    "अठहत्तर": 78,
    "उन्नासी": 79,
    "अस्सी": 80,
    "इक्यासी": 81,
    "बयासी": 82,
    "तिरासी": 83,
    "चौरासी": 84,
    "पचासी": 85,
    "छियासी": 86,
    "सत्तासी": 87,
    "अट्ठासी": 88,
    "नवासी": 89,
    "नब्बे": 90,
    "इक्यानवे": 91,
    "इकानवे": 91,
    "बानवे": 92,
    "तिरानवे": 93,
    "चौरानवे": 94,
    "पंचानवे": 95,
    "पचानवे": 95,
    "छियानवे": 96,
    "सत्तानवे": 97,
    "अट्ठानवे": 98,
    "निन्यानवे": 99,
}

_NUMBER_WORDS = {_fold(word): value for word, value in _RAW_NUMBER_WORDS.items()}
_SCALE_WORDS = {
    _fold(word): value
    for value, words in (
        (100, ("सौ", "sau")),
        (1_000, ("हजार", "hazaar", "hazar")),
        (100_000, ("लाख", "lakh", "lakhs", "lac", "lacs")),
        (10_000_000, ("करोड़", "करोड", "crore", "crores")),
    )
    for word in words
}
_FRACTION_WORDS = {
    _fold("डेढ़"): Fraction(3, 2),
    _fold("ढाई"): Fraction(5, 2),
}
_CURRENCY_WORDS = {
    _fold(word)
    for word in (
        "रुपया", "रुपये", "रुपए", "रुपियों", "रुपयों", "रु",
        "₹", "rs", "inr", "rupee", "rupees",
    )
}
_DIGIT_TRANSLATION = str.maketrans("०१२३४५६७८९", "0123456789")


def _tokens(text: str) -> list[str]:
    text = _fold(text).translate(_DIGIT_TRANSLATION)
    chars: list[str] = []
    for index, char in enumerate(text):
        category = unicodedata.category(char)
        if char == "₹":
            chars.extend((" ", char, " "))
        elif char.isspace() or category[0] in {"L", "M", "N"}:
            chars.append(char)
        elif (
            char == ","
            and index > 0
            and index + 1 < len(text)
            and text[index - 1].isdigit()
            and text[index + 1].isdigit()
        ):
            chars.append(char)
        else:
            chars.append(" ")
    return "".join(chars).split()


def _tokens_with_punctuation_boundaries(text: str) -> tuple[list[str], set[int]]:
    """Tokenize words/numerals and retain punctuation as phrase boundaries."""
    normalized = _fold(text).translate(_DIGIT_TRANSLATION).lower()
    tokens: list[str] = []
    boundaries: set[int] = set()
    current: list[str] = []
    pending_boundary = False

    def append_token(token: str) -> None:
        nonlocal pending_boundary
        if pending_boundary:
            boundaries.add(len(tokens))
            pending_boundary = False
        tokens.append(token)

    def flush_word() -> None:
        if current:
            append_token("".join(current))
            current.clear()

    for index, char in enumerate(normalized):
        category = unicodedata.category(char)
        if char.isspace():
            flush_word()
        elif char == "₹":
            flush_word()
            append_token(char)
        elif category[0] in {"L", "M", "N"}:
            current.append(char)
        elif (
            char == ","
            and index > 0
            and index + 1 < len(normalized)
            and normalized[index - 1].isdigit()
            and normalized[index + 1].isdigit()
        ):
            current.append(char)
        else:
            flush_word()
            if category[0] == "P":
                pending_boundary = True
    flush_word()
    return tokens, boundaries


def _parse_digit_token(token: str) -> int:
    if "," not in token:
        if not token.isdigit():
            raise ValueError(f"not a numeric token: {token!r}")
        return int(token)

    groups = token.split(",")
    if not all(group.isdigit() for group in groups):
        raise ValueError(f"invalid grouped number: {token!r}")

    indian_grouping = (
        1 <= len(groups[0]) <= 3
        and len(groups[-1]) == 3
        and all(len(group) == 2 for group in groups[1:-1])
    )
    western_grouping = (
        1 <= len(groups[0]) <= 3
        and all(len(group) == 3 for group in groups[1:])
    )
    if not (indian_grouping or western_grouping):
        raise ValueError(f"invalid grouped number: {token!r}")
    return int("".join(groups))


def _is_digit_token(token: str) -> bool:
    try:
        _parse_digit_token(token)
        return True
    except ValueError:
        return False


def _is_number_part(token: str) -> bool:
    return (
        token in _NUMBER_WORDS
        or token in _SCALE_WORDS
        or token in _FRACTION_WORDS
        or _is_digit_token(token)
    )


def parse_hindi_number(text: str) -> int:
    """Parse an integer Hindi amount phrase or a grouped digit string.

    ``डेढ़`` and ``ढाई`` are supported as multipliers before a scale word; an
    unscaled fractional value is rejected because this function returns rupees
    as integers.
    """
    tokens = [token for token in _tokens(text) if token not in _CURRENCY_WORDS]
    if not tokens:
        raise ValueError("no number found")

    if len(tokens) == 1 and _is_digit_token(tokens[0]):
        return _parse_digit_token(tokens[0])

    total = Fraction(0)
    current: Fraction | None = None
    for token in tokens:
        if token in _NUMBER_WORDS:
            current = (current or Fraction(0)) + _NUMBER_WORDS[token]
        elif token in _FRACTION_WORDS:
            current = (current or Fraction(0)) + _FRACTION_WORDS[token]
        elif _is_digit_token(token):
            current = (current or Fraction(0)) + _parse_digit_token(token)
        elif token in _SCALE_WORDS:
            scale = _SCALE_WORDS[token]
            group = Fraction(1) if current is None else current
            if scale == 100:
                current = group * scale
            else:
                total += group * scale
                current = None
        else:
            raise ValueError(f"unknown Hindi number word: {token!r}")

    result = total + (current or Fraction(0))
    if result.denominator != 1:
        raise ValueError("fractional rupees need a scale such as सौ or हजार")
    return int(result)


def extract_amounts(text: str) -> list[int]:
    """Extract amount-like numeric phrases, avoiding bare dates and PINs."""
    tokens, boundaries = _tokens_with_punctuation_boundaries(text)
    amounts: list[int] = []
    index = 0

    while index < len(tokens):
        if not _is_number_part(tokens[index]):
            index += 1
            continue

        start = index
        while (
            index < len(tokens)
            and (index == start or index not in boundaries)
            and _is_number_part(tokens[index])
        ):
            index += 1
        phrase = tokens[start:index]

        has_large_scale = any(
            token in _SCALE_WORDS and _SCALE_WORDS[token] >= 1_000
            for token in phrase
        )
        has_grouping = any("," in token for token in phrase)
        has_large_digit = any(
            _is_digit_token(token) and _parse_digit_token(token) >= 1_000
            for token in phrase
        )
        has_currency = (
            (
                start > 0
                and start not in boundaries
                and tokens[start - 1] in _CURRENCY_WORDS
            )
            or (
                index < len(tokens)
                and index not in boundaries
                and tokens[index] in _CURRENCY_WORDS
            )
        )

        if has_large_scale or has_grouping or has_large_digit or has_currency:
            try:
                amounts.append(parse_hindi_number(" ".join(phrase)))
            except ValueError:
                # In particular, skip an unscaled डेढ़/ढाई amount: the output
                # contract is integer rupees, so rounding would be misleading.
                continue

    return amounts


def amounts_correct(expected_entities: list[dict], hypothesis: str) -> bool:
    """Compare expected amount values with amounts extracted from a transcript."""
    expected = [
        int(entity["value"])
        for entity in expected_entities
        if entity.get("type") == "amount"
    ]
    return Counter(expected) == Counter(extract_amounts(hypothesis))


_DIGIT_WORDS = {
    _fold(word): str(value)
    for value, words in (
        (0, ("शून्य", "सिफर", "जीरो", "zero", "oh")),
        (1, ("एक", "one")),
        (2, ("दो", "two")),
        (3, ("तीन", "three")),
        (4, ("चार", "four")),
        (5, ("पाँच", "पांच", "five")),
        (6, ("छह", "छः", "six")),
        (7, ("सात", "seven")),
        (8, ("आठ", "eight")),
        (9, ("नौ", "nine")),
    )
    for word in words
}


def extract_digit_sequences(text: str) -> list[str]:
    """Convert contiguous spoken/written digits to digit strings.

    Hindi and English digit words, Devanagari/Latin numerals, spaces, commas,
    and hyphens are handled approximately; non-digit words end a sequence.
    """
    sequences: list[str] = []
    current: list[str] = []

    def flush() -> None:
        if current:
            sequences.append("".join(current))
            current.clear()

    for token in _tokens(text):
        if token in _DIGIT_WORDS:
            current.append(_DIGIT_WORDS[token])
        elif _is_digit_token(token):
            current.append(str(_parse_digit_token(token)))
        else:
            flush()
    flush()
    return sequences


def normalize_digit_sequence(value: object) -> str:
    """Normalize an expected digit sequence, ignoring spaces and hyphens."""
    text = str(value)
    sequences = extract_digit_sequences(text)
    if len(sequences) == 1:
        return sequences[0]
    return re.sub(r"[\s-]", "", _fold(text).translate(_DIGIT_TRANSLATION))


_MONTHS = {
    _fold(name): month
    for month, names in (
        (1, ("जनवरी", "january", "jan")),
        (2, ("फ़रवरी", "फरवरी", "फरवरी", "february", "feb")),
        (3, ("मार्च", "march", "mar")),
        (4, ("अप्रैल", "april", "apr")),
        (5, ("मई", "may")),
        (6, ("जून", "june", "jun")),
        (7, ("जुलाई", "july", "jul")),
        (8, ("अगस्त", "august", "aug")),
        (9, ("सितंबर", "सितम्बर", "september", "sept", "sep")),
        (10, ("अक्टूबर", "october", "oct")),
        (11, ("नवंबर", "नवम्बर", "november", "nov")),
        (12, ("दिसंबर", "दिसम्बर", "december", "dec")),
    )
    for name in names
}
_DATE_MARKERS = {
    _fold(word) for word in ("तारीख", "तारिख", "दिन", "date", "day")
}


def _day_number(token: str) -> int | None:
    if _is_digit_token(token):
        day = _parse_digit_token(token)
    else:
        day = _NUMBER_WORDS.get(token)
    return day if day is not None and 1 <= day <= 31 else None


def extract_dates(text: str) -> list[tuple[int, int | None]]:
    """Extract day/month pairs with simple Hindi and English date patterns.

    This is an approximate scorer, not a full date parser. It recognizes day
    words or numerals followed by a date marker or month name, and month names
    followed by a day.
    """
    tokens = _tokens(text)
    dates: list[tuple[int, int | None]] = []

    for index, token in enumerate(tokens):
        month = _MONTHS.get(token)
        if month is not None:
            for neighbor in (index - 1, index + 1):
                if 0 <= neighbor < len(tokens):
                    day = _day_number(tokens[neighbor])
                    if day is not None:
                        dates.append((day, month))
        elif token in _DATE_MARKERS and index > 0:
            day = _day_number(tokens[index - 1])
            if day is not None:
                dates.append((day, None))

    return list(dict.fromkeys(dates))


def _expected_date_values(value: object) -> list[tuple[int, int | None]]:
    if isinstance(value, dict):
        try:
            day = int(value["day"])
            month_value = value.get("month")
            if month_value is None:
                month = None
            elif isinstance(month_value, str) and not month_value.isdigit():
                month = _MONTHS.get(_fold(month_value))
            else:
                month = int(month_value)
            if 1 <= day <= 31 and (month is None or 1 <= month <= 12):
                return [(day, month)]
        except (KeyError, TypeError, ValueError):
            return []

    text = str(value)
    parsed = extract_dates(text)
    if parsed:
        return parsed
    tokens = _tokens(text)
    if len(tokens) == 1:
        day = _day_number(tokens[0])
        if day is not None:
            return [(day, None)]
    return []


def _dates_match(
    expected: list[tuple[int, int | None]],
    actual: list[tuple[int, int | None]],
) -> bool:
    remaining = list(actual)
    for expected_day, expected_month in expected:
        match_index = next(
            (
                index
                for index, (actual_day, actual_month) in enumerate(remaining)
                if actual_day == expected_day
                and (expected_month is None or actual_month == expected_month)
            ),
            None,
        )
        if match_index is None:
            return False
        remaining.pop(match_index)
    return not remaining


# Approximate grapheme-to-Latin transcription used only for UPI-style IDs.
_DEVANAGARI_CONSONANTS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "ng",
    "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "ny",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ल": "l", "व": "v", "श": "sh",
    "ष": "sh", "स": "s", "ह": "h", "ळ": "l",
}
_INDEPENDENT_VOWELS = {
    "अ": "a", "आ": "a", "इ": "i", "ई": "i", "उ": "u",
    "ऊ": "u", "ऋ": "ri", "ए": "e", "ऐ": "ai", "ओ": "o",
    "औ": "au",
}
_MATRAS = {
    "ा": "a", "ि": "i", "ी": "i", "ु": "u", "ू": "u",
    "ृ": "ri", "े": "e", "ै": "ai", "ो": "o", "ौ": "au",
}
_UPI_WORD_ALIASES = {
    _fold("ओकेएसबीआई"): "oksbi",
    _fold("ओकेएसबिआई"): "oksbi",
}
_UPI_MARKERS = {_fold("एट"), "at"}
_UPI_PATTERN = re.compile(
    r"(?<![\w.+-])([\w.+-]+)\s*@\s*([\w.-]+)(?![\w.-])",
    re.IGNORECASE,
)


def _romanize_devanagari(text: str) -> str:
    text = _fold(text).lower()
    output: list[str] = []
    index = 0
    final_inherent_vowel = False

    while index < len(text):
        char = text[index]
        consonant = _DEVANAGARI_CONSONANTS.get(char)
        if consonant is not None:
            output.append(consonant)
            index += 1
            final_inherent_vowel = False
            if index < len(text) and text[index] == "्":
                index += 1
            elif index < len(text) and text[index] in _MATRAS:
                output.append(_MATRAS[text[index]])
                index += 1
            else:
                output.append("a")
                final_inherent_vowel = True
            continue

        vowel = _INDEPENDENT_VOWELS.get(char)
        if vowel is not None:
            output.append(vowel)
            final_inherent_vowel = False
        elif char == "ं":
            output.append("n")
            final_inherent_vowel = False
        elif char.isascii() and char.isalnum():
            output.append(char)
            final_inherent_vowel = False
        index += 1

    if final_inherent_vowel and output and output[-1] == "a":
        output.pop()
    return "".join(output)


def _normalize_upi_part(part: str) -> str:
    normalized = _fold(part).lower()
    normalized = re.sub(r"[\s-]", "", normalized)
    if normalized in _UPI_WORD_ALIASES:
        return _UPI_WORD_ALIASES[normalized]
    romanized = _romanize_devanagari(normalized)
    return re.sub(r"[^a-z0-9._+]+", "", romanized)


def extract_upi_ids(text: str) -> list[str]:
    """Extract written or spoken UPI-style IDs using approximate transliteration."""
    ids = [
        f"{_normalize_upi_part(local)}@{_normalize_upi_part(provider)}"
        for local, provider in _UPI_PATTERN.findall(text)
    ]

    tokens = _tokens(text)
    for index, token in enumerate(tokens):
        if token in _UPI_MARKERS and 0 < index < len(tokens) - 1:
            candidate = (
                f"{_normalize_upi_part(tokens[index - 1])}@"
                f"{_normalize_upi_part(tokens[index + 1])}"
            )
            if candidate not in ids:
                ids.append(candidate)
    return ids


def normalize_upi_id(value: object) -> str:
    """Normalize one UPI value; Devanagari transliteration is approximate."""
    values = extract_upi_ids(str(value))
    return values[0] if values else ""


_IFSC_PATTERN = re.compile(r"^[A-Z]{4}0[A-Z0-9]{6}$")
_IFSC_IN_TEXT = re.compile(r"(?<![A-Z0-9])[A-Z]{4}0[A-Z0-9]{6}(?![A-Z0-9])", re.I)


def normalize_ifsc(value: object) -> str:
    """Uppercase an IFSC and ignore embedded spaces or hyphens."""
    return re.sub(r"[\s-]", "", str(value)).upper()


def is_valid_ifsc(value: object) -> bool:
    """Approximate IFSC shape check (four letters, 0, then six alphanumerics)."""
    return bool(_IFSC_PATTERN.fullmatch(normalize_ifsc(value)))


def extract_ifsc_codes(text: str) -> list[str]:
    """Extract strings matching the approximate IFSC shape."""
    return [match.group(0).upper() for match in _IFSC_IN_TEXT.finditer(text)]


ENTITY_TYPES = ("amount", "digit_sequence", "date", "upi", "ifsc")
_ENTITY_TYPE_ALIASES = {
    "amount": "amount",
    "digit_sequence": "digit_sequence",
    "digit-sequence": "digit_sequence",
    "digits": "digit_sequence",
    "date": "date",
    "date_day": "date",
    "upi": "upi",
    "upi_id": "upi",
    "ifsc": "ifsc",
    "ifsc_code": "ifsc",
}


def score_entity_types(
    expected_entities: list[dict], hypothesis: str
) -> dict[str, bool | None]:
    """Score each supported entity type; unrequested types receive ``None``.

    Date, UPI, and IFSC scores are approximate. Types with expected values are
    compared independently, so one type's extraction does not affect another.
    """
    expected_by_type: dict[str, list[object]] = {}
    unsupported_types: set[str] = set()
    for entity in expected_entities:
        raw_type = str(entity.get("type", "unknown")).lower()
        entity_type = _ENTITY_TYPE_ALIASES.get(raw_type)
        if entity_type is None:
            unsupported_types.add(raw_type)
            continue
        expected_by_type.setdefault(entity_type, []).append(entity.get("value"))

    scores: dict[str, bool | None] = {entity_type: None for entity_type in ENTITY_TYPES}
    for entity_type, expected_values in expected_by_type.items():
        if entity_type == "amount":
            expected = Counter(int(value) for value in expected_values)
            actual = Counter(extract_amounts(hypothesis))
            scores[entity_type] = expected == actual
        elif entity_type == "digit_sequence":
            expected = Counter(
                normalize_digit_sequence(value) for value in expected_values
            )
            scores[entity_type] = expected == Counter(
                extract_digit_sequences(hypothesis)
            )
        elif entity_type == "date":
            expected_dates = [
                date
                for value in expected_values
                for date in _expected_date_values(value)
            ]
            scores[entity_type] = _dates_match(
                expected_dates, extract_dates(hypothesis)
            )
        elif entity_type == "upi":
            expected = Counter(normalize_upi_id(value) for value in expected_values)
            scores[entity_type] = expected == Counter(extract_upi_ids(hypothesis))
        elif entity_type == "ifsc":
            expected = Counter(normalize_ifsc(value) for value in expected_values)
            scores[entity_type] = expected == Counter(extract_ifsc_codes(hypothesis))

    for unsupported_type in unsupported_types:
        scores[unsupported_type] = False
    return scores

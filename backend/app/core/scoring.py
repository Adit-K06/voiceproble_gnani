"""Text normalization and word error rate scoring."""

import unicodedata


def _normalize_text(text: str) -> str:
    """Apply case, punctuation, and whitespace normalization."""
    text = unicodedata.normalize("NFC", text).lower()
    chars = [
        " " if unicodedata.category(char).startswith("P") else char
        for char in text
    ]
    return " ".join("".join(chars).split())


def normalize_strict(text: str) -> str:
    """Normalize Unicode, case, punctuation, and spaces without spelling folds.

    Nukta and chandrabindu are preserved so callers can inspect strict spelling
    differences even though the default scoring path is lenient.
    """
    return _normalize_text(text)


def normalize(text: str) -> str:
    """Apply lenient Hindi spelling normalization plus standard text cleanup.

    Canonical decomposition exposes nukta even in precomposed characters such
    as ``ज़``. After folding, the result is returned in NFC form.
    """
    text = unicodedata.normalize("NFC", text)
    text = unicodedata.normalize("NFD", text)
    text = "".join(char for char in text if char != "\u093c")
    text = text.replace("\u0901", "\u0902")  # chandrabindu -> anusvara
    return _normalize_text(unicodedata.normalize("NFC", text))


def spacing_only_difference(reference: str, hypothesis: str) -> bool:
    """Return True when normalized strings differ only by spaces."""
    ref = normalize(reference)
    hyp = normalize(hypothesis)
    return ref != hyp and ref.replace(" ", "") == hyp.replace(" ", "")


def edit_distance(ref_words: list[str], hyp_words: list[str]) -> int:
    """Levenshtein distance between two word lists (dynamic programming)."""
    rows = len(ref_words) + 1
    cols = len(hyp_words) + 1
    distances = [[0] * cols for _ in range(rows)]

    for i in range(rows):
        distances[i][0] = i
    for j in range(cols):
        distances[0][j] = j

    for i in range(1, rows):
        for j in range(1, cols):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            distances[i][j] = min(
                distances[i - 1][j] + 1,
                distances[i][j - 1] + 1,
                distances[i - 1][j - 1] + cost,
            )

    return distances[rows - 1][cols - 1]


def _word_wer(reference: str, hypothesis: str) -> float:
    ref_words = normalize(reference).split()
    hyp_words = normalize(hypothesis).split()

    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    return edit_distance(ref_words, hyp_words) / len(ref_words)


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate, with spelling and spacing variants normalized."""
    if spacing_only_difference(reference, hypothesis):
        return 0.0
    return _word_wer(reference, hypothesis)


def wer_best_with_flags(
    references: list[str], hypothesis: str
) -> tuple[float, bool]:
    """Return the minimum WER and whether its best match differed only by spaces."""
    if not references:
        raise ValueError("references must contain at least one string")

    scores = [
        (wer(reference, hypothesis), spacing_only_difference(reference, hypothesis))
        for reference in references
    ]
    # Prefer a non-spacing match when multiple references have the same score.
    return min(scores, key=lambda result: (result[0], result[1]))


def wer_best(references: list[str], hypothesis: str) -> float:
    """Return the minimum WER across accepted reference strings."""
    return wer_best_with_flags(references, hypothesis)[0]

"""Word and character error rates on scoring-normalised text.

Raw WER punishes formatting choices that are not recognition errors: "four" vs "4",
"4,471" vs "4471", "free-range" vs "free range", "1kg" vs "one kilo". Like Whisper's English
text normaliser, we normalise both sides before scoring so WER measures mishearing.
"""

import re

from voice_to_order.text import normalise_tokens, word_edit_distance

_ONES = {
    word: value
    for value, word in enumerate(
        [
            "zero",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
            "eleven",
            "twelve",
            "thirteen",
            "fourteen",
            "fifteen",
            "sixteen",
            "seventeen",
            "eighteen",
            "nineteen",
        ]
    )
}
_TENS = {
    word: (index + 2) * 10
    for index, word in enumerate(
        ["twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
    )
}
_SCALES = {"hundred": 100, "thousand": 1000}
_UNITS = {"kg": "kilo", "kilogram": "kilo", "kilograms": "kilo", "g": "gram", "grams": "gram"}
_THOUSANDS_SEPARATOR = re.compile(r"(?<=\d),(?=\d{3}\b)")
_DIGITS_THEN_UNIT = re.compile(r"\b(\d+)([a-z]+)\b")


def _words_to_digits(tokens: list[str]) -> list[str]:
    """ "five hundred" -> "500", "twenty one" -> "21", "four four seven one" -> "4 4 7 1"."""
    out: list[str] = []
    total = current = 0
    in_number = False

    def flush() -> None:
        nonlocal total, current, in_number
        if in_number:
            out.append(str(total + current))
        total = current = 0
        in_number = False

    for token in tokens:
        if token in _ONES or token in _TENS:
            value = _ONES.get(token, _TENS.get(token, 0))
            # A number word continues the current number only where English allows it:
            # after a scale ("five hundred and" -> tens/ones) or ones after tens ("twenty one").
            fits = (
                not in_number
                or (current % 100 == 0 if current else total > 0)
                or (current >= 20 and current % 10 == 0 and value < 10)
            )
            if not fits:
                flush()
            current += value
            in_number = True
        elif token in _SCALES and in_number:
            if _SCALES[token] == 100:
                current *= 100
            else:
                total += current * _SCALES[token]
                current = 0
        else:
            flush()
            out.append(token)
    flush()
    return out


def scoring_tokens(text: str) -> list[str]:
    text = _THOUSANDS_SEPARATOR.sub("", text.lower())
    text = _DIGITS_THEN_UNIT.sub(r"\1 \2", text)
    tokens = [part for token in normalise_tokens(text) for part in token.split("-") if part]
    tokens = [_UNITS.get(token, token) for token in tokens]
    return _words_to_digits(tokens)


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate: (substitutions + deletions + insertions) / reference words."""
    ref = scoring_tokens(reference)
    if not ref:
        raise ValueError("reference has no words")
    return word_edit_distance(ref, scoring_tokens(hypothesis)) / len(ref)


def cer(reference: str, hypothesis: str) -> float:
    """Character error rate over normalised text with single spaces."""
    ref = list(" ".join(scoring_tokens(reference)))
    if not ref:
        raise ValueError("reference has no characters")
    return word_edit_distance(ref, list(" ".join(scoring_tokens(hypothesis)))) / len(ref)


def corpus_errors(reference: str, hypothesis: str) -> tuple[int, int]:
    """(errors, reference words), for corpus-level WER that weights long messages properly."""
    ref = scoring_tokens(reference)
    return word_edit_distance(ref, scoring_tokens(hypothesis)), len(ref)

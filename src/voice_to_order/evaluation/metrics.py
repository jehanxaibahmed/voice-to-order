"""Word and character error rates."""

from voice_to_order.text import normalise_tokens, word_edit_distance


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate: (substitutions + deletions + insertions) / reference words."""
    ref = normalise_tokens(reference)
    if not ref:
        raise ValueError("reference has no words")
    return word_edit_distance(ref, normalise_tokens(hypothesis)) / len(ref)


def cer(reference: str, hypothesis: str) -> float:
    """Character error rate over normalised text with single spaces."""
    ref = list(" ".join(normalise_tokens(reference)))
    if not ref:
        raise ValueError("reference has no characters")
    return word_edit_distance(ref, list(" ".join(normalise_tokens(hypothesis)))) / len(ref)


def corpus_errors(reference: str, hypothesis: str) -> tuple[int, int]:
    """(errors, reference words), for corpus-level WER that weights long messages properly."""
    ref = normalise_tokens(reference)
    return word_edit_distance(ref, normalise_tokens(hypothesis)), len(ref)

"""Word-level text normalisation and alignment shared by consensus and evaluation."""

import re
from dataclasses import dataclass
from enum import StrEnum

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:['\-][a-z0-9]+)*")


def normalise_tokens(text: str) -> list[str]:
    """Lower-case word tokens with punctuation removed ("OM-12," -> "om-12")."""
    return _TOKEN_RE.findall(text.lower())


class Op(StrEnum):
    MATCH = "match"
    SUBSTITUTE = "substitute"
    DELETE = "delete"  # word in reference, missing from hypothesis
    INSERT = "insert"  # extra word in hypothesis


@dataclass(frozen=True)
class AlignedPair:
    op: Op
    ref: str | None
    hyp: str | None
    ref_index: int | None


def align(reference: list[str], hypothesis: list[str]) -> list[AlignedPair]:
    """Levenshtein alignment of two token lists, returned in reference order."""
    rows, cols = len(reference) + 1, len(hypothesis) + 1
    cost = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        cost[i][0] = i
    for j in range(cols):
        cost[0][j] = j
    for i in range(1, rows):
        for j in range(1, cols):
            diff = 0 if reference[i - 1] == hypothesis[j - 1] else 1
            cost[i][j] = min(cost[i - 1][j - 1] + diff, cost[i - 1][j] + 1, cost[i][j - 1] + 1)

    pairs: list[AlignedPair] = []
    i, j = len(reference), len(hypothesis)
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            diff = 0 if reference[i - 1] == hypothesis[j - 1] else 1
            if cost[i][j] == cost[i - 1][j - 1] + diff:
                op = Op.MATCH if diff == 0 else Op.SUBSTITUTE
                pairs.append(AlignedPair(op, reference[i - 1], hypothesis[j - 1], i - 1))
                i, j = i - 1, j - 1
                continue
        if i > 0 and cost[i][j] == cost[i - 1][j] + 1:
            pairs.append(AlignedPair(Op.DELETE, reference[i - 1], None, i - 1))
            i -= 1
        else:
            pairs.append(AlignedPair(Op.INSERT, None, hypothesis[j - 1], None))
            j -= 1
    pairs.reverse()
    return pairs


def word_edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    return sum(1 for pair in align(reference, hypothesis) if pair.op is not Op.MATCH)

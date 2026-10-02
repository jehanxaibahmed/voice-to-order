"""Resolve the requested delivery date from each transcript and take a majority vote.

Delivery dates are where single-engine errors hurt most ("Tuesday" vs "Thursday", "the
13th" vs "the 30th"), so each provider's transcript votes independently instead of trusting
one reconciled text.
"""

import calendar
import datetime as dt
import re
from collections import Counter
from collections.abc import Mapping

from voice_to_order.domain import ConsensusTranscript, DeliveryDateDecision, TranscriptionResult

CONSENSUS_VOTER = "consensus"
MIN_AGREEMENT = 0.5

_WEEKDAYS = {name.lower(): index for index, name in enumerate(calendar.day_name)}
_MONTHS = {name.lower(): index for index, name in enumerate(calendar.month_name) if name}
_MONTHS |= {name.lower(): index for index, name in enumerate(calendar.month_abbr) if name}
_MONTHS["sept"] = 9
_ORDINAL_WORDS = {
    word: index + 1
    for index, word in enumerate(
        [
            "first",
            "second",
            "third",
            "fourth",
            "fifth",
            "sixth",
            "seventh",
            "eighth",
            "ninth",
            "tenth",
            "eleventh",
            "twelfth",
            "thirteenth",
            "fourteenth",
            "fifteenth",
            "sixteenth",
            "seventeenth",
            "eighteenth",
            "nineteenth",
            "twentieth",
            "twenty-first",
            "twenty-second",
            "twenty-third",
            "twenty-fourth",
            "twenty-fifth",
            "twenty-sixth",
            "twenty-seventh",
            "twenty-eighth",
            "twenty-ninth",
            "thirtieth",
            "thirty-first",
        ]
    )
}

_DAY = (
    r"(?P<day>\d{1,2})(?:st|nd|rd|th)?|(?P<dayword>"
    + "|".join(sorted(_ORDINAL_WORDS, key=len, reverse=True))
    + ")"
)
_MONTH = r"(?P<month>" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + r")\.?"
_WEEKDAY = r"(?P<weekday>" + "|".join(_WEEKDAYS) + ")"

_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("day_after_tomorrow", re.compile(r"\bday after tomorrow\b")),
    ("tomorrow", re.compile(r"\btomorrow\b")),
    ("today", re.compile(r"\btoday\b")),
    ("numeric", re.compile(r"\b(?P<nday>\d{1,2})/(?P<nmonth>\d{1,2})(?:/(?P<nyear>\d{2,4}))?\b")),
    ("day_month", re.compile(rf"\b(?:the\s+)?(?:{_DAY})\s+(?:of\s+)?{_MONTH}\b")),
    ("month_day", re.compile(rf"\b{_MONTH}\s+(?:the\s+)?(?:{_DAY})\b")),
    ("weekday", re.compile(rf"\b(?P<qualifier>next|this|on)?\s*{_WEEKDAY}\b")),
    ("day_only", re.compile(rf"\bthe\s+(?:{_DAY})\b")),
]
_DELIVERY_CUE = re.compile(r"\b(deliver\w*|drop(?:ped)? off|need (?:it|them|this)|arrive|for|by)\b")


def _day(match: re.Match[str]) -> int:
    if match.group("day"):
        return int(match.group("day"))
    return _ORDINAL_WORDS[match.group("dayword")]


def _next_with_day(reference: dt.date, day: int, month: int | None = None) -> dt.date | None:
    """The first date on or after ``reference`` with this day (and month, if given)."""
    for offset in range(0, 13):
        year = reference.year + (reference.month - 1 + offset) // 12
        candidate_month = (reference.month - 1 + offset) % 12 + 1
        if month is not None and candidate_month != month:
            continue
        try:
            candidate = dt.date(year, candidate_month, day)
        except ValueError:
            continue
        if candidate >= reference:
            return candidate
    return None


def _resolve(kind: str, match: re.Match[str], reference: dt.date) -> dt.date | None:
    if kind == "today":
        return reference
    if kind == "tomorrow":
        return reference + dt.timedelta(days=1)
    if kind == "day_after_tomorrow":
        return reference + dt.timedelta(days=2)
    if kind == "numeric":
        day, month = int(match.group("nday")), int(match.group("nmonth"))  # UK: day/month
        if year := match.group("nyear"):
            try:
                return dt.date(int(year) + (2000 if len(year) == 2 else 0), month, day)
            except ValueError:
                return None
        return _next_with_day(reference, day, month) if 1 <= month <= 12 else None
    if kind in {"day_month", "month_day"}:
        return _next_with_day(reference, _day(match), _MONTHS[match.group("month")])
    if kind == "day_only":
        return _next_with_day(reference, _day(match))
    if kind == "weekday":
        target = _WEEKDAYS[match.group("weekday")]
        ahead = (target - reference.weekday()) % 7
        qualifier = match.group("qualifier")
        if qualifier == "next":
            # "next Tuesday" means Tuesday of next week.
            start_of_next_week = reference + dt.timedelta(days=7 - reference.weekday())
            return start_of_next_week + dt.timedelta(days=target)
        if qualifier == "this":
            return reference + dt.timedelta(days=ahead)
        return reference + dt.timedelta(days=ahead or 7)
    raise ValueError(f"unknown pattern {kind}")


def parse_delivery_date(text: str, reference: dt.date) -> dt.date | None:
    """Find the delivery date in ``text``, resolving relative phrases against ``reference``.

    When several dates are mentioned, the first one after a delivery cue ("deliver", "for",
    "by", "need it") wins; otherwise the last date mentioned.
    """
    lowered = text.lower()
    found: list[tuple[int, int, dt.date, str]] = []
    taken: list[range] = []
    for kind, pattern in _PATTERNS:
        for match in pattern.finditer(lowered):
            span = range(match.start(), match.end())
            if any(span.start < t.stop and t.start < span.stop for t in taken):
                continue  # already covered by a more specific pattern
            resolved = _resolve(kind, match, reference)
            if resolved is not None:
                taken.append(span)
                found.append((match.start(), match.end(), resolved, kind))
    found.sort()
    # "Tuesday the 13th": the explicit date right after a weekday is the precise one.
    found = [
        entry
        for entry in found
        if entry[3] != "weekday"
        or not any(0 <= other[0] - entry[1] <= 5 for other in found if other is not entry)
    ]
    if not found:
        return None
    cues = [cue.end() for cue in _DELIVERY_CUE.finditer(lowered)]
    for start, _, resolved, _ in found:
        if any(0 <= start - cue <= 40 for cue in cues):
            return resolved
    return found[-1][2]


def vote_delivery_date(
    result: TranscriptionResult,
    consensus: ConsensusTranscript | None,
    reference: dt.date,
) -> DeliveryDateDecision:
    votes: dict[str, dt.date | None] = {
        t.provider: parse_delivery_date(t.text, reference) for t in result.transcripts
    }
    if consensus is not None:
        votes[CONSENSUS_VOTER] = parse_delivery_date(consensus.text, reference)
    return decide(votes)


def decide(votes: Mapping[str, dt.date | None]) -> DeliveryDateDecision:
    """Majority over the dates found; ties go to the consensus transcript's reading."""
    counts = Counter(date for date in votes.values() if date is not None)
    if not counts:
        return DeliveryDateDecision(date=None, votes=dict(votes), agreement=0.0)
    ranked = counts.most_common()
    winner, top = ranked[0]
    tied = [date for date, count in ranked if count == top]
    if len(tied) > 1 and votes.get(CONSENSUS_VOTER) in tied:
        winner = votes[CONSENSUS_VOTER]  # type: ignore[assignment]
    agreement = sum(1 for date in votes.values() if date == winner) / len(votes)
    return DeliveryDateDecision(date=winner, votes=dict(votes), agreement=round(agreement, 3))


def review_reason(decision: DeliveryDateDecision) -> str | None:
    if decision.date is None:
        return "no delivery date found"
    if decision.agreement < MIN_AGREEMENT:
        return f"transcripts disagree on delivery date ({decision.agreement:.0%} agreement)"
    return None

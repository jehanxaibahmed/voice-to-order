import datetime as dt

import pytest

from voice_to_order.domain import ConsensusTranscript, Transcript, TranscriptionResult
from voice_to_order.extraction import parse_delivery_date, review_reason, vote_delivery_date
from voice_to_order.extraction.delivery_date import decide

# Friday 2 October 2026
REF = dt.date(2026, 10, 2)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("can you deliver tomorrow please", dt.date(2026, 10, 3)),
        ("we need it today", dt.date(2026, 10, 2)),
        ("the day after tomorrow is fine", dt.date(2026, 10, 4)),
        ("for Tuesday morning", dt.date(2026, 10, 6)),
        ("on Friday", dt.date(2026, 10, 9)),  # same weekday means next week
        ("this Friday", dt.date(2026, 10, 2)),
        ("next Monday", dt.date(2026, 10, 5)),
        ("next Friday", dt.date(2026, 10, 9)),
        ("by the 14th", dt.date(2026, 10, 14)),
        ("by the first", dt.date(2026, 11, 1)),
        ("on the fourteenth of October", dt.date(2026, 10, 14)),
        ("14th October", dt.date(2026, 10, 14)),
        ("October 14th", dt.date(2026, 10, 14)),
        ("on Sept 3rd", dt.date(2027, 9, 3)),  # already passed this year
        ("deliver 5/10", dt.date(2026, 10, 5)),
        ("deliver 5/10/27", dt.date(2027, 10, 5)),
        ("the 31st", dt.date(2026, 10, 31)),
    ],
)
def test_parses_date_phrases(text: str, expected: dt.date) -> None:
    assert parse_delivery_date(text, REF) == expected


def test_no_date() -> None:
    assert parse_delivery_date("two cases of oat milk please", REF) is None


def test_prefers_date_after_delivery_cue() -> None:
    text = "I rang on Monday but nobody answered, can you deliver on Wednesday"
    assert parse_delivery_date(text, REF) == dt.date(2026, 10, 7)


def test_specific_pattern_wins_over_weekday_inside_it() -> None:
    assert parse_delivery_date("for Tuesday the 13th of October", REF) == dt.date(2026, 10, 13)


def result(*texts: str) -> TranscriptionResult:
    return TranscriptionResult(
        transcripts=[Transcript(provider=f"p{i}", text=t) for i, t in enumerate(texts)]
    )


def test_vote_majority_beats_single_misheard_day() -> None:
    decision = vote_delivery_date(
        result("deliver on Tuesday", "deliver on Thursday", "deliver on Tuesday"),
        ConsensusTranscript(text="deliver on Tuesday", method="llm", sources=[]),
        REF,
    )
    assert decision.date == dt.date(2026, 10, 6)
    assert decision.agreement == 0.75
    assert decision.votes["p1"] == dt.date(2026, 10, 8)
    assert review_reason(decision) is None


def test_tie_goes_to_consensus() -> None:
    tuesday, thursday = dt.date(2026, 10, 6), dt.date(2026, 10, 8)
    decision = decide({"a": tuesday, "b": thursday, "consensus": thursday, "c": tuesday})
    assert decision.date == thursday


def test_no_dates_needs_review() -> None:
    decision = vote_delivery_date(result("hello", "hi"), None, REF)
    assert decision.date is None
    assert review_reason(decision) == "no delivery date found"


def test_low_agreement_needs_review() -> None:
    decision = decide({"a": dt.date(2026, 10, 6), "b": None, "c": None})
    reason = review_reason(decision)
    assert reason is not None and "33% agreement" in reason

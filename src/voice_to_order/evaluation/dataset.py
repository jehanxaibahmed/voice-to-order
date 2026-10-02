"""Load labelled voicemail samples (see samples/README.md for the format)."""

import datetime as dt
from decimal import Decimal
from pathlib import Path

from pydantic import BaseModel

from voice_to_order.domain import Transcript, TranscriptionResult


class ExpectedOrder(BaseModel):
    account_number: str | None
    delivery_date: dt.date | None
    lines: list[tuple[str, Decimal]]


class Sample(BaseModel):
    id: str
    received_at: dt.datetime
    caller_number: str | None = None
    reference: str
    expected: ExpectedOrder
    transcripts: dict[str, str]

    def transcription_result(self) -> TranscriptionResult:
        return TranscriptionResult(
            transcripts=[Transcript(provider=p, text=t) for p, t in self.transcripts.items()]
        )


def load_samples(directory: Path) -> list[Sample]:
    samples = [Sample.model_validate_json(p.read_text()) for p in sorted(directory.glob("*.json"))]
    if not samples:
        raise FileNotFoundError(f"no sample files in {directory}")
    return samples

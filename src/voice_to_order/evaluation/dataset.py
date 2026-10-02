"""Load labelled voicemail samples (see samples/README.md for the format)."""

import datetime as dt
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field, PlainSerializer

from voice_to_order.domain import Transcript, TranscriptionResult

# "synthetic": hand-written transcripts; "recorded": real provider output from `record`.
TranscriptSource = Literal["synthetic", "recorded"]


def _number(value: Decimal) -> int | float:
    return int(value) if value == value.to_integral_value() else float(value)


Quantity = Annotated[Decimal, PlainSerializer(_number, return_type=int | float)]


class ExpectedOrder(BaseModel):
    account_number: str | None
    delivery_date: dt.date | None
    lines: list[tuple[str, Quantity]]


class Sample(BaseModel):
    id: str
    received_at: dt.datetime
    caller_number: str | None = None
    reference: str
    expected: ExpectedOrder
    transcripts: dict[str, str]
    recorded_transcripts: dict[str, str] = Field(default_factory=dict)

    def transcripts_for(self, source: TranscriptSource) -> dict[str, str]:
        return self.transcripts if source == "synthetic" else self.recorded_transcripts

    def transcription_result(self, source: TranscriptSource = "synthetic") -> TranscriptionResult:
        return TranscriptionResult(
            transcripts=[
                Transcript(provider=p, text=t) for p, t in self.transcripts_for(source).items()
            ]
        )


def load_samples(directory: Path) -> list[Sample]:
    samples = [Sample.model_validate_json(p.read_text()) for p in sorted(directory.glob("*.json"))]
    if not samples:
        raise FileNotFoundError(f"no sample files in {directory}")
    return samples


def sample_path(directory: Path, sample_id: str) -> Path:
    return directory / f"{sample_id}.json"


def save_sample(directory: Path, sample: Sample) -> None:
    sample_path(directory, sample.id).write_text(
        sample.model_dump_json(indent=2, exclude_defaults=False) + "\n"
    )

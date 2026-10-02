"""Pydantic models that flow between pipeline stages.

ingest -> AudioFile -> transcribe -> TranscriptionResult -> reconcile -> ConsensusTranscript
       -> extract -> Order (with DeliveryDateDecision)
"""

import datetime as dt
from decimal import Decimal
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class Voicemail(BaseModel):
    """An inbound voicemail as received from the phone system."""

    id: str
    received_at: dt.datetime
    caller_number: str | None = None
    source_path: Path


class AudioFile(BaseModel):
    """A normalised audio file ready for transcription."""

    model_config = ConfigDict(frozen=True)

    path: Path
    format: str
    sample_rate: int
    channels: int
    duration_seconds: float


class Transcript(BaseModel):
    """One provider's transcript of a voicemail."""

    provider: str
    text: str
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    latency_ms: float | None = None


class TranscriptionFailure(BaseModel):
    provider: str
    error: str


class TranscriptionResult(BaseModel):
    """Transcripts from every provider that succeeded, plus the ones that failed."""

    transcripts: list[Transcript]
    failures: list[TranscriptionFailure] = Field(default_factory=list)

    def by_provider(self) -> dict[str, Transcript]:
        return {t.provider: t for t in self.transcripts}


class ConsensusTranscript(BaseModel):
    """The reconciled transcript produced from several provider transcripts."""

    text: str
    method: str
    sources: list[str]
    uncertain_spans: list[str] = Field(default_factory=list)


class Customer(BaseModel):
    name: str | None = None
    company: str | None = None
    account_number: str | None = None
    phone: str | None = None


class OrderLine(BaseModel):
    description: str
    quantity: Decimal = Field(gt=0)
    unit: str | None = None
    product_code: str | None = None
    matched_catalog_sku: str | None = None


class DeliveryDateDecision(BaseModel):
    """Outcome of the per-transcript delivery date vote."""

    date: dt.date | None
    votes: dict[str, dt.date | None]
    agreement: float = Field(ge=0.0, le=1.0)


class Order(BaseModel):
    voicemail_id: str
    customer: Customer
    lines: list[OrderLine]
    delivery: DeliveryDateDecision | None = None
    notes: str | None = None
    needs_review: bool = False
    review_reasons: list[str] = Field(default_factory=list)

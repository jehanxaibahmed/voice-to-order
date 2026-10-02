"""Run every provider concurrently and keep whatever succeeds."""

import asyncio
import logging
import time
from collections.abc import Sequence

from voice_to_order.domain import (
    AudioFile,
    Transcript,
    TranscriptionError,
    TranscriptionFailure,
    TranscriptionResult,
)
from voice_to_order.transcription.base import TranscriptionProvider

logger = logging.getLogger(__name__)


class TranscriptionRunner:
    def __init__(
        self, providers: Sequence[TranscriptionProvider], *, timeout_seconds: float = 60.0
    ) -> None:
        if not providers:
            raise ValueError("at least one transcription provider is required")
        names = [p.name for p in providers]
        if len(set(names)) != len(names):
            raise ValueError(f"provider names must be unique: {names}")
        self._providers = list(providers)
        self._timeout = timeout_seconds

    @property
    def provider_names(self) -> list[str]:
        return [p.name for p in self._providers]

    async def run(self, audio: AudioFile) -> TranscriptionResult:
        outcomes = await asyncio.gather(*(self._run_one(p, audio) for p in self._providers))
        transcripts = [o for o in outcomes if isinstance(o, Transcript)]
        failures = [o for o in outcomes if isinstance(o, TranscriptionFailure)]
        if not transcripts:
            detail = "; ".join(f"{f.provider}: {f.error}" for f in failures)
            raise TranscriptionError("all", f"every provider failed ({detail})")
        return TranscriptionResult(transcripts=transcripts, failures=failures)

    async def _run_one(
        self, provider: TranscriptionProvider, audio: AudioFile
    ) -> Transcript | TranscriptionFailure:
        started = time.perf_counter()
        try:
            transcript = await asyncio.wait_for(provider.transcribe(audio), self._timeout)
        except TimeoutError:
            logger.warning("%s timed out after %.0fs", provider.name, self._timeout)
            return TranscriptionFailure(provider=provider.name, error="timed out")
        except Exception as exc:  # one provider must never sink the others
            logger.warning("%s failed: %s", provider.name, exc)
            return TranscriptionFailure(provider=provider.name, error=str(exc))
        if not transcript.text:
            return TranscriptionFailure(provider=provider.name, error="empty transcript")
        latency_ms = (time.perf_counter() - started) * 1000
        return transcript.model_copy(update={"latency_ms": round(latency_ms, 1)})

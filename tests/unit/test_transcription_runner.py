import asyncio
from pathlib import Path

import pytest

from voice_to_order.domain import AudioFile, Transcript, TranscriptionError
from voice_to_order.transcription import TranscriptionRunner

AUDIO = AudioFile(
    path=Path("/tmp/vm/normalised.wav"),
    format="wav",
    sample_rate=16_000,
    channels=1,
    duration_seconds=2,
)


class StubProvider:
    def __init__(self, name: str, text: str = "", *, delay: float = 0, error: str = "") -> None:
        self.name = name
        self._text = text
        self._delay = delay
        self._error = error

    async def transcribe(self, audio: AudioFile) -> Transcript:
        await asyncio.sleep(self._delay)
        if self._error:
            raise RuntimeError(self._error)
        return Transcript(provider=self.name, text=self._text)


async def test_runs_providers_concurrently() -> None:
    runner = TranscriptionRunner(
        [StubProvider("a", "one", delay=0.2), StubProvider("b", "two", delay=0.2)]
    )
    loop = asyncio.get_running_loop()
    started = loop.time()
    result = await runner.run(AUDIO)
    assert loop.time() - started < 0.35
    assert {t.provider: t.text for t in result.transcripts} == {"a": "one", "b": "two"}
    assert all(t.latency_ms is not None for t in result.transcripts)


async def test_partial_failure_keeps_successes() -> None:
    runner = TranscriptionRunner(
        [StubProvider("ok", "hello"), StubProvider("bad", error="401"), StubProvider("empty")]
    )
    result = await runner.run(AUDIO)
    assert [t.provider for t in result.transcripts] == ["ok"]
    assert {f.provider: f.error for f in result.failures} == {
        "bad": "401",
        "empty": "empty transcript",
    }


async def test_slow_provider_times_out() -> None:
    runner = TranscriptionRunner(
        [StubProvider("fast", "hi"), StubProvider("slow", "late", delay=1)], timeout_seconds=0.1
    )
    result = await runner.run(AUDIO)
    assert result.failures[0].provider == "slow"
    assert result.failures[0].error == "timed out"


async def test_all_failed_raises() -> None:
    runner = TranscriptionRunner([StubProvider("a", error="x"), StubProvider("b", error="y")])
    with pytest.raises(TranscriptionError, match="every provider failed"):
        await runner.run(AUDIO)


def test_requires_unique_providers() -> None:
    with pytest.raises(ValueError):
        TranscriptionRunner([])
    with pytest.raises(ValueError):
        TranscriptionRunner([StubProvider("a"), StubProvider("a")])

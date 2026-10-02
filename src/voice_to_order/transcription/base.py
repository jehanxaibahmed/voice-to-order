"""The port every speech-to-text adapter implements."""

from typing import Protocol, runtime_checkable

import httpx

from voice_to_order.domain import AudioFile, Transcript


@runtime_checkable
class TranscriptionProvider(Protocol):
    @property
    def name(self) -> str: ...

    async def transcribe(self, audio: AudioFile) -> Transcript:
        """Return the provider's transcript. Raise TranscriptionError on failure."""
        ...


def audio_mime_type(audio: AudioFile) -> str:
    return {"wav": "audio/wav", "mp3": "audio/mpeg"}.get(audio.format, "application/octet-stream")


def describe_http_error(exc: httpx.HTTPError) -> str:
    """A short error without the request URL, which can carry long query strings."""
    if isinstance(exc, httpx.HTTPStatusError):
        return f"HTTP {exc.response.status_code} {exc.response.reason_phrase}".strip()
    return f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__

"""The port every speech-to-text adapter implements."""

from typing import Protocol, runtime_checkable

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

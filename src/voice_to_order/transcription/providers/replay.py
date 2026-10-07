"""Replays stored transcripts. Used for offline showcases, evaluation and tests."""

from collections.abc import Mapping

from voice_to_order.domain import AudioFile, Transcript, TranscriptionError


class ReplayProvider:
    """Returns a pre-recorded transcript keyed by voicemail id.

    The voicemail id is the name of the directory the normalised audio lives in, which is
    how ``AudioIngestor`` lays out its work directory.
    """

    def __init__(self, name: str, transcripts: Mapping[str, str]) -> None:
        self._name = name
        self._transcripts = dict(transcripts)

    @property
    def name(self) -> str:
        return self._name

    async def transcribe(self, audio: AudioFile) -> Transcript:
        voicemail_id = audio.path.parent.name
        try:
            text = self._transcripts[voicemail_id]
        except KeyError:
            raise TranscriptionError(self._name, f"no recording for {voicemail_id}") from None
        return Transcript(provider=self._name, text=text)

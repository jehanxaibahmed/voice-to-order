"""OpenAI Whisper transcription API (https://platform.openai.com/docs/api-reference/audio)."""

import httpx

from voice_to_order.domain import AudioFile, Transcript, TranscriptionError
from voice_to_order.transcription.base import audio_mime_type

WHISPER_URL = "https://api.openai.com/v1/audio/transcriptions"


class WhisperProvider:
    name = "whisper"

    def __init__(
        self,
        api_key: str,
        client: httpx.AsyncClient,
        *,
        model: str = "whisper-1",
        language: str = "en",
        vocabulary_hint: str | None = None,
    ) -> None:
        self._api_key = api_key
        self._client = client
        self._model = model
        self._language = language
        self._vocabulary_hint = vocabulary_hint

    async def transcribe(self, audio: AudioFile) -> Transcript:
        data = {"model": self._model, "language": self._language, "response_format": "json"}
        if self._vocabulary_hint:
            # Whisper's prompt biases spelling of uncommon words such as product names.
            data["prompt"] = self._vocabulary_hint
        try:
            response = await self._client.post(
                WHISPER_URL,
                data=data,
                files={"file": (audio.path.name, audio.path.read_bytes(), audio_mime_type(audio))},
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise TranscriptionError(self.name, f"request failed: {exc}") from exc

        try:
            text = response.json()["text"]
        except (KeyError, TypeError, ValueError) as exc:
            raise TranscriptionError(self.name, "unexpected response shape") from exc
        return Transcript(provider=self.name, text=text.strip())

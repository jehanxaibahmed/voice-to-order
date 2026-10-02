"""Deepgram pre-recorded audio API (https://developers.deepgram.com/reference/listen-file)."""

from collections.abc import Sequence

import httpx

from voice_to_order.domain import AudioFile, Transcript, TranscriptionError
from voice_to_order.transcription.base import audio_mime_type, describe_http_error

DEEPGRAM_URL = "https://api.deepgram.com/v1/listen"


class DeepgramProvider:
    name = "deepgram"

    def __init__(
        self,
        api_key: str,
        client: httpx.AsyncClient,
        *,
        model: str = "nova-3",
        language: str = "en",
        keyterms: Sequence[str] = (),
    ) -> None:
        self._api_key = api_key
        self._client = client
        self._model = model
        self._language = language
        self._keyterms = list(keyterms)

    async def transcribe(self, audio: AudioFile) -> Transcript:
        params: list[tuple[str, str | int | float | bool | None]] = [
            ("model", self._model),
            ("language", self._language),
            ("smart_format", "true"),
            ("punctuate", "true"),
            ("numerals", "true"),
        ]
        # Keyterm prompting boosts recognition of product names and codes.
        params += [("keyterm", term) for term in self._keyterms]
        try:
            response = await self._client.post(
                DEEPGRAM_URL,
                params=params,
                content=audio.path.read_bytes(),
                headers={
                    "Authorization": f"Token {self._api_key}",
                    "Content-Type": audio_mime_type(audio),
                },
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise TranscriptionError(self.name, describe_http_error(exc)) from exc

        try:
            alternative = response.json()["results"]["channels"][0]["alternatives"][0]
            return Transcript(
                provider=self.name,
                text=alternative["transcript"].strip(),
                confidence=alternative.get("confidence"),
            )
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise TranscriptionError(self.name, "unexpected response shape") from exc

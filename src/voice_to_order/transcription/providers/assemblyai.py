"""AssemblyAI async transcription API (https://www.assemblyai.com/docs/api-reference).

Upload the file, create a transcript job, then poll until it completes.
"""

import asyncio
from collections.abc import Sequence
from typing import Any

import httpx

from voice_to_order.domain import AudioFile, Transcript, TranscriptionError

ASSEMBLYAI_URL = "https://api.assemblyai.com"


class AssemblyAIProvider:
    name = "assemblyai"

    def __init__(
        self,
        api_key: str,
        client: httpx.AsyncClient,
        *,
        base_url: str = ASSEMBLYAI_URL,
        language: str = "en",
        word_boost: Sequence[str] = (),
        poll_interval_seconds: float = 1.0,
    ) -> None:
        self._client = client
        self._base_url = base_url.rstrip("/")
        self._headers = {"Authorization": api_key}
        self._language = language
        self._word_boost = list(word_boost)
        self._poll_interval = poll_interval_seconds

    async def transcribe(self, audio: AudioFile) -> Transcript:
        try:
            upload = await self._request("POST", "/v2/upload", content=audio.path.read_bytes())
            job: dict[str, Any] = {
                "audio_url": upload["upload_url"],
                "language_code": self._language,
                "punctuate": True,
                "format_text": True,
            }
            if self._word_boost:
                job |= {"word_boost": self._word_boost, "boost_param": "high"}
            created = await self._request("POST", "/v2/transcript", json=job)
            # The runner's timeout bounds this loop.
            while True:
                status = await self._request("GET", f"/v2/transcript/{created['id']}")
                if status["status"] == "completed":
                    return Transcript(
                        provider=self.name,
                        text=(status.get("text") or "").strip(),
                        confidence=status.get("confidence"),
                    )
                if status["status"] == "error":
                    raise TranscriptionError(self.name, status.get("error") or "job failed")
                await asyncio.sleep(self._poll_interval)
        except KeyError as exc:
            raise TranscriptionError(self.name, f"unexpected response: missing {exc}") from exc

    async def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        try:
            response = await self._client.request(
                method, self._base_url + path, headers=self._headers, **kwargs
            )
            response.raise_for_status()
            body: dict[str, Any] = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise TranscriptionError(self.name, f"request failed: {exc}") from exc
        return body

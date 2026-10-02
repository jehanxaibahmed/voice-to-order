"""In-process Whisper via faster-whisper (https://github.com/SYSTRAN/faster-whisper).

Free and offline: needs the optional extra (``pip install -e '.[local]'``). The model is
downloaded from the Hugging Face hub on first use and loaded once, lazily.
"""

import asyncio
import threading
from typing import Any

from voice_to_order.domain import AudioFile, Transcript, TranscriptionError

INSTALL_HINT = "faster-whisper is not installed: run `pip install -e '.[local]'`"


class LocalWhisperProvider:
    name = "whisper-local"

    def __init__(
        self,
        *,
        model_size: str = "small",
        compute_type: str = "int8",
        device: str = "auto",
        language: str = "en",
        vocabulary_hint: str | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        self._model_size = model_size
        self._compute_type = compute_type
        self._device = device
        self._language = language
        self._vocabulary_hint = vocabulary_hint
        # Read by the runner: CPU inference plus a first-run download outlasts a cloud timeout.
        self.timeout_seconds = timeout_seconds
        self._model: Any = None
        self._lock = threading.Lock()

    def _load_model(self) -> Any:
        with self._lock:
            if self._model is None:
                try:
                    from faster_whisper import WhisperModel
                except ImportError as exc:
                    raise TranscriptionError(self.name, INSTALL_HINT) from exc
                self._model = WhisperModel(
                    self._model_size, device=self._device, compute_type=self._compute_type
                )
            return self._model

    def _transcribe_sync(self, audio: AudioFile) -> str:
        model = self._load_model()
        # The generator is lazy; decoding happens while joining the segments.
        segments, _info = model.transcribe(
            str(audio.path),
            language=self._language,
            initial_prompt=self._vocabulary_hint or None,
            beam_size=5,
        )
        return " ".join(segment.text.strip() for segment in segments).strip()

    async def transcribe(self, audio: AudioFile) -> Transcript:
        try:
            text = await asyncio.to_thread(self._transcribe_sync, audio)
        except TranscriptionError:
            raise
        except Exception as exc:
            raise TranscriptionError(self.name, f"{type(exc).__name__}: {exc}") from exc
        return Transcript(provider=self.name, text=text)

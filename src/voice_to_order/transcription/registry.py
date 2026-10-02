"""Build the configured providers, skipping any without credentials."""

import logging
from collections.abc import Sequence

import httpx

from voice_to_order.config import Settings
from voice_to_order.transcription.base import TranscriptionProvider
from voice_to_order.transcription.providers import DeepgramProvider, WhisperProvider

logger = logging.getLogger(__name__)


def build_providers(
    settings: Settings, client: httpx.AsyncClient, *, vocabulary: Sequence[str] = ()
) -> list[TranscriptionProvider]:
    providers: list[TranscriptionProvider] = []
    for name in settings.transcription_providers:
        if name == "deepgram" and settings.deepgram_api_key:
            providers.append(
                DeepgramProvider(
                    settings.deepgram_api_key.get_secret_value(),
                    client,
                    model=settings.deepgram_model,
                    keyterms=vocabulary,
                )
            )
        elif name == "whisper" and settings.openai_api_key:
            providers.append(
                WhisperProvider(
                    settings.openai_api_key.get_secret_value(),
                    client,
                    model=settings.whisper_model,
                    vocabulary_hint=", ".join(vocabulary) or None,
                )
            )
        elif name in {"deepgram", "whisper"}:
            logger.warning("skipping %s: no API key configured", name)
        else:
            logger.warning("skipping unknown transcription provider %r", name)
    return providers

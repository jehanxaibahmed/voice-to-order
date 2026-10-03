"""Build the production pipeline from settings."""

from pathlib import Path

import httpx

from voice_to_order.audio import AudioIngestor, FFmpeg
from voice_to_order.config import Settings
from voice_to_order.consensus import LLMReconciler
from voice_to_order.extraction import Catalog, OrderExtractor
from voice_to_order.llm import LiteLLMClient, LLMClient
from voice_to_order.pipeline import VoiceToOrderPipeline
from voice_to_order.transcription import TranscriptionRunner, build_providers

DEFAULT_CATALOG = Path("samples/catalog.json")


def build_llm(settings: Settings) -> LLMClient:
    return LiteLLMClient(
        model=settings.llm_model,
        api_key=settings.llm_api_key.get_secret_value() if settings.llm_api_key else None,
        base_url=settings.llm_base_url,
        timeout=settings.llm_timeout_seconds,
    )


def build_pipeline(
    settings: Settings, http_client: httpx.AsyncClient, *, catalog_path: Path = DEFAULT_CATALOG
) -> VoiceToOrderPipeline:
    catalog = Catalog.load(catalog_path)
    providers = build_providers(settings, http_client, vocabulary=catalog.keyterms())
    if not providers:
        raise RuntimeError(
            "no transcription providers configured: set at least one of "
            "VTO_DEEPGRAM_API_KEY, VTO_OPENAI_API_KEY, VTO_ASSEMBLYAI_API_KEY, "
            "or list whisper-local in VTO_TRANSCRIPTION_PROVIDERS"
        )
    llm = build_llm(settings)
    return VoiceToOrderPipeline(
        ingestor=AudioIngestor(
            settings.work_dir,
            FFmpeg(settings.ffmpeg_binary, settings.ffprobe_binary),
            max_bytes=settings.max_upload_bytes,
        ),
        runner=TranscriptionRunner(
            providers, timeout_seconds=settings.transcription_timeout_seconds
        ),
        reconciler=LLMReconciler(llm),
        extractor=OrderExtractor(llm, catalog),
        catalog=catalog,
    )

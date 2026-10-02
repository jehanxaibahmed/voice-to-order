"""Build the production pipeline from settings."""

from pathlib import Path

import anthropic
import httpx

from voice_to_order.audio import AudioIngestor, FFmpeg
from voice_to_order.config import Settings
from voice_to_order.consensus import LLMReconciler
from voice_to_order.extraction import Catalog, OrderExtractor
from voice_to_order.llm import ClaudeClient
from voice_to_order.llm.claude import Effort
from voice_to_order.pipeline import VoiceToOrderPipeline
from voice_to_order.transcription import TranscriptionRunner, build_providers

DEFAULT_CATALOG = Path("samples/catalog.json")


def build_llm(settings: Settings) -> ClaudeClient:
    # With no explicit key the SDK falls back to ANTHROPIC_API_KEY or an `ant auth login` profile.
    anthropic_client = anthropic.AsyncAnthropic(
        api_key=settings.anthropic_api_key.get_secret_value()
        if settings.anthropic_api_key
        else None
    )
    effort: Effort = settings.llm_effort  # type: ignore[assignment]
    return ClaudeClient(anthropic_client, model=settings.llm_model, effort=effort)


def build_pipeline(
    settings: Settings, http_client: httpx.AsyncClient, *, catalog_path: Path = DEFAULT_CATALOG
) -> VoiceToOrderPipeline:
    catalog = Catalog.load(catalog_path)
    providers = build_providers(settings, http_client, vocabulary=catalog.keyterms())
    if not providers:
        raise RuntimeError(
            "no transcription providers configured: set at least one of "
            "VTO_DEEPGRAM_API_KEY, VTO_OPENAI_API_KEY, VTO_ASSEMBLYAI_API_KEY"
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

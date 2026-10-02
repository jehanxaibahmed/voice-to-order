from pathlib import Path

import httpx
import pytest

from voice_to_order.bootstrap import build_pipeline
from voice_to_order.config import Settings


async def test_build_pipeline_needs_a_provider(samples_dir: Path) -> None:
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    async with httpx.AsyncClient() as client:
        with pytest.raises(RuntimeError, match="VTO_DEEPGRAM_API_KEY"):
            build_pipeline(settings, client, catalog_path=samples_dir / "catalog.json")


async def test_build_pipeline_with_keys(samples_dir: Path) -> None:
    settings = Settings(_env_file=None, deepgram_api_key="d", anthropic_api_key="a")  # type: ignore[call-arg]
    async with httpx.AsyncClient() as client:
        pipeline = build_pipeline(settings, client, catalog_path=samples_dir / "catalog.json")
    assert pipeline is not None

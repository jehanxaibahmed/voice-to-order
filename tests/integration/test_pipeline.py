"""End to end: real ffmpeg, replayed transcripts, scripted LLM."""

import datetime as dt
import shutil
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import BaseModel

from tests.helpers import make_tone
from voice_to_order.audio import AudioIngestor
from voice_to_order.consensus import LLMReconciler
from voice_to_order.consensus.llm import SYSTEM_PROMPT as CONSENSUS_PROMPT
from voice_to_order.domain import Voicemail
from voice_to_order.evaluation import load_samples
from voice_to_order.extraction import Catalog, OrderExtractor
from voice_to_order.llm import ScriptedLLM
from voice_to_order.pipeline import VoiceToOrderPipeline
from voice_to_order.transcription import TranscriptionRunner
from voice_to_order.transcription.providers import ReplayProvider

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def scripted_llm(consensus_text: str, extraction: dict[str, object]) -> ScriptedLLM:
    def respond(system: str, prompt: str, schema: type[BaseModel]) -> dict[str, object]:
        if system == CONSENSUS_PROMPT:
            return {"text": consensus_text, "uncertain_spans": []}
        return extraction

    return ScriptedLLM(respond)


async def test_voicemail_becomes_reviewed_order(samples_dir: Path, tmp_path: Path) -> None:
    sample = next(s for s in load_samples(samples_dir / "voicemails") if s.id == "vm-001")
    catalog = Catalog.load(samples_dir / "catalog.json")
    providers = [
        ReplayProvider(name, {sample.id: text}) for name, text in sample.transcripts.items()
    ]
    llm = scripted_llm(
        sample.reference,
        {
            "customer": {"name": "Sam", "company": "Corner Deli", "account_number": "4471",
                         "phone": None},
            "lines": [
                {"description": "oat milk", "quantity": 4, "unit": "cases", "product_code": None},
                {"description": "croissants", "quantity": 2, "unit": "trays",
                 "product_code": None},
            ],
            "notes": None,
        },
    )  # fmt: skip
    pipeline = VoiceToOrderPipeline(
        ingestor=AudioIngestor(tmp_path / "work"),
        runner=TranscriptionRunner(providers),
        reconciler=LLMReconciler(llm),
        extractor=OrderExtractor(llm, catalog),
        catalog=catalog,
    )
    voicemail = Voicemail(
        id=sample.id,
        received_at=sample.received_at,
        caller_number=sample.caller_number,
        source_path=make_tone(tmp_path / "vm-001.mp3", seconds=2),
    )

    result = await pipeline.process(voicemail)

    assert result.audio.sample_rate == 16_000
    assert len(result.transcription.transcripts) == 3
    assert result.consensus.method == "llm"
    order = result.order
    assert [(line.matched_catalog_sku, line.quantity) for line in order.lines] == [
        ("OM-12", Decimal(4)),
        ("CR-24", Decimal(2)),
    ]
    assert order.delivery is not None
    assert order.delivery.date == sample.expected.delivery_date == dt.date(2026, 10, 6)
    assert order.delivery.votes["whisper"] == dt.date(2026, 10, 8)  # whisper heard "Thursday"
    assert order.delivery.agreement == 0.75
    assert not order.needs_review, order.review_reasons
    # the catalogue vocabulary reaches the consensus prompt
    assert "OM-12 Barista oat milk 1L" in llm.calls[0].prompt


async def test_single_provider_is_flagged(samples_dir: Path, tmp_path: Path) -> None:
    catalog = Catalog.load(samples_dir / "catalog.json")
    llm = scripted_llm(
        "",
        {
            "customer": {"name": None, "company": "Deli", "account_number": None, "phone": None},
            "lines": [{"description": "eggs", "quantity": 1, "unit": None, "product_code": None}],
            "notes": None,
        },
    )
    pipeline = VoiceToOrderPipeline(
        ingestor=AudioIngestor(tmp_path / "work"),
        runner=TranscriptionRunner(
            [
                ReplayProvider("only", {"vm-x": "one tray of eggs tomorrow"}),
                ReplayProvider("broken", {}),
            ]
        ),
        reconciler=LLMReconciler(llm),
        extractor=OrderExtractor(llm, catalog),
        catalog=catalog,
    )
    voicemail = Voicemail(
        id="vm-x",
        received_at=dt.datetime(2026, 10, 2, 9, tzinfo=dt.UTC),
        source_path=make_tone(tmp_path / "in.wav", seconds=1),
    )
    order = (await pipeline.process(voicemail)).order
    assert order.needs_review
    assert "only one transcription provider succeeded" in order.review_reasons
    assert order.delivery is not None and order.delivery.date == dt.date(2026, 10, 3)

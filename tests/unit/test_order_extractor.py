from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from voice_to_order.domain import ConsensusTranscript, ExtractionError, LLMError, Voicemail
from voice_to_order.extraction import Catalog, OrderExtractor
from voice_to_order.llm import ScriptedLLM

VOICEMAIL = Voicemail(
    id="vm-1",
    received_at=datetime(2026, 10, 2, 9, tzinfo=UTC),
    caller_number="+44 7700 900123",
    source_path=Path("vm-1.wav"),
)
TRANSCRIPT = ConsensusTranscript(text="...", method="llm", sources=["a", "b"])


def answer(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "customer": {"name": "Sam", "company": "Corner Deli", "account_number": "4471",
                     "phone": None},
        "lines": [
            {"description": "oat milk", "quantity": 4, "unit": "cases", "product_code": None},
            {"description": "croissants", "quantity": 2, "unit": "trays", "product_code": "CR-24"},
        ],
        "notes": "Leave at the back door",
    }  # fmt: skip
    base.update(overrides)
    return base


@pytest.fixture
def catalog(samples_dir: Path) -> Catalog:
    return Catalog.load(samples_dir / "catalog.json")


async def test_extracts_order_and_matches_catalogue(catalog: Catalog) -> None:
    llm = ScriptedLLM(lambda *_: answer())
    order = await OrderExtractor(llm, catalog).extract(VOICEMAIL, TRANSCRIPT)

    assert order.customer.company == "Corner Deli"
    assert order.customer.phone == "+44 7700 900123"  # falls back to caller id
    assert [(line.matched_catalog_sku, line.quantity) for line in order.lines] == [
        ("OM-12", Decimal(4)),
        ("CR-24", Decimal(2)),
    ]
    assert order.notes == "Leave at the back door"
    assert not order.needs_review
    assert (
        "<catalogue>" in llm.calls[0].prompt and "OM-12: Barista oat milk 1L" in llm.calls[0].prompt
    )


async def test_flags_unmatched_lines_and_unknown_caller(catalog: Catalog) -> None:
    llm = ScriptedLLM(
        lambda *_: answer(
            customer={"name": None, "company": None, "account_number": None, "phone": None},
            lines=[{"description": "smoked salmon", "quantity": 1, "unit": None,
                    "product_code": None}],
        )
    )  # fmt: skip
    transcript = TRANSCRIPT.model_copy(update={"uncertain_spans": ["salmon"]})
    order = await OrderExtractor(llm, catalog).extract(VOICEMAIL, transcript)

    assert order.needs_review
    assert order.review_reasons == [
        "no catalogue match for 'smoked salmon'",
        "caller could not be identified",
        "uncertain transcript: 'salmon'",
    ]


async def test_drops_non_positive_quantities(catalog: Catalog) -> None:
    llm = ScriptedLLM(
        lambda *_: answer(lines=[{"description": "eggs", "quantity": 0, "unit": None,
                                  "product_code": None}])
    )  # fmt: skip
    order = await OrderExtractor(llm, catalog).extract(VOICEMAIL, TRANSCRIPT)
    assert order.lines == []
    assert "no order lines found" in order.review_reasons


async def test_llm_failure_raises_extraction_error(catalog: Catalog) -> None:
    llm = ScriptedLLM(lambda *_: LLMError("refused"))
    with pytest.raises(ExtractionError, match="refused"):
        await OrderExtractor(llm, catalog).extract(VOICEMAIL, TRANSCRIPT)

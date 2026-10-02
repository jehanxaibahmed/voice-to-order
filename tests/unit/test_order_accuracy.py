import datetime as dt
from decimal import Decimal
from pathlib import Path

from pydantic import BaseModel

from voice_to_order.consensus import LLMReconciler
from voice_to_order.consensus.llm import SYSTEM_PROMPT as CONSENSUS_PROMPT
from voice_to_order.domain import Customer, DeliveryDateDecision, LLMError, Order, OrderLine
from voice_to_order.evaluation import ExpectedOrder, load_samples
from voice_to_order.evaluation.orders import compare, evaluate_orders
from voice_to_order.extraction import Catalog, OrderExtractor
from voice_to_order.llm import ScriptedLLM

EXPECTED = ExpectedOrder(
    account_number="4471",
    delivery_date=dt.date(2026, 10, 6),
    lines=[("OM-12", Decimal(4)), ("CR-24", Decimal(2))],
)


def order(account: str | None, lines: list[tuple[str, int]], day: int = 6) -> Order:
    return Order(
        voicemail_id="vm",
        customer=Customer(account_number=account),
        lines=[
            OrderLine(description=sku, quantity=Decimal(q), matched_catalog_sku=sku)
            for sku, q in lines
        ],
        delivery=DeliveryDateDecision(date=dt.date(2026, 10, day), votes={}, agreement=1),
    )


def test_exact_order() -> None:
    result = compare("vm", order("4471", [("CR-24", 2), ("OM-12", 4)]), EXPECTED)
    assert result.exact
    assert (result.true_positive_lines, result.extracted_lines, result.expected_lines) == (2, 2, 2)


def test_wrong_quantity_extra_line_and_wrong_date() -> None:
    result = compare(
        "vm", order("4471", [("OM-12", 3), ("CR-24", 2), ("EG-30", 1)], day=8), EXPECTED
    )
    assert not result.exact
    assert not result.date_correct
    assert (result.true_positive_lines, result.extracted_lines) == (1, 3)


async def test_evaluate_orders_end_to_end_with_scripted_llm(samples_dir: Path) -> None:
    samples = load_samples(samples_dir / "voicemails")[:3]
    by_reference = {s.reference: s for s in samples}

    def respond(system: str, prompt: str, schema: type[BaseModel]) -> object:
        sample = (
            next(s for s in samples if s.transcripts["deepgram"] in prompt)
            if (system == CONSENSUS_PROMPT)
            else next(s for ref, s in by_reference.items() if ref in prompt)
        )
        if system == CONSENSUS_PROMPT:
            return {"text": sample.reference, "uncertain_spans": []}
        if sample.id == "vm-003":
            return LLMError("refused")
        return {
            "customer": {"name": None, "company": None,
                         "account_number": sample.expected.account_number, "phone": None},
            "lines": [{"description": sku, "quantity": float(q), "unit": None, "product_code": sku}
                      for sku, q in sample.expected.lines],
            "notes": None,
        }  # fmt: skip

    llm = ScriptedLLM(respond)
    catalog = Catalog.load(samples_dir / "catalog.json")
    report = await evaluate_orders(
        samples,
        reconciler=LLMReconciler(llm),
        extractor=OrderExtractor(llm, catalog),
        catalog=catalog,
    )

    assert report.samples == 3
    assert report.failed == 1
    assert report.exact_order_rate == round(2 / 3, 4)
    assert report.line_precision == 1.0
    assert report.line_recall == round(5 / 8, 4)  # vm-003's 3 lines were never extracted
    assert "| Orders exactly right | 67% |" in report.to_markdown()

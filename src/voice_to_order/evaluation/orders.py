"""Score extracted orders against the expected orders: the metric the business cares about."""

from collections import Counter
from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path

from pydantic import BaseModel

from voice_to_order.consensus import Reconciler
from voice_to_order.domain import Order, Voicemail, VoiceToOrderError
from voice_to_order.evaluation.dataset import ExpectedOrder, Sample, TranscriptSource
from voice_to_order.extraction import Catalog, OrderExtractor, vote_delivery_date


class OrderComparison(BaseModel):
    sample_id: str
    account_correct: bool
    date_correct: bool
    true_positive_lines: int
    extracted_lines: int
    expected_lines: int
    exact: bool
    needs_review: bool
    error: str | None = None


class OrderAccuracyReport(BaseModel):
    samples: int
    account_accuracy: float
    date_accuracy: float
    line_precision: float
    line_recall: float
    exact_order_rate: float
    flagged_for_review: int
    failed: int
    comparisons: list[OrderComparison]

    def to_markdown(self) -> str:
        return "\n".join(
            [
                "| Metric | Value |",
                "|---|---:|",
                f"| Orders scored | {self.samples} |",
                f"| Account number correct | {self.account_accuracy:.0%} |",
                f"| Delivery date correct | {self.date_accuracy:.0%} |",
                f"| Line precision (SKU + quantity) | {self.line_precision:.0%} |",
                f"| Line recall (SKU + quantity) | {self.line_recall:.0%} |",
                f"| Orders exactly right | {self.exact_order_rate:.0%} |",
                f"| Flagged for review | {self.flagged_for_review} |",
                f"| Extraction failures | {self.failed} |",
            ]
        )


def compare(sample_id: str, order: Order, expected: ExpectedOrder) -> OrderComparison:
    got = Counter(
        (line.matched_catalog_sku, Decimal(line.quantity).normalize()) for line in order.lines
    )
    want: Counter[tuple[str | None, Decimal]] = Counter(
        (sku, Decimal(qty).normalize()) for sku, qty in expected.lines
    )
    true_positives = sum((got & want).values())
    account_correct = (order.customer.account_number or None) == expected.account_number
    date_correct = (order.delivery.date if order.delivery else None) == expected.delivery_date
    return OrderComparison(
        sample_id=sample_id,
        account_correct=account_correct,
        date_correct=date_correct,
        true_positive_lines=true_positives,
        extracted_lines=sum(got.values()),
        expected_lines=sum(want.values()),
        exact=account_correct and date_correct and got == want,
        needs_review=order.needs_review,
    )


async def evaluate_orders(
    samples: Sequence[Sample],
    *,
    reconciler: Reconciler,
    extractor: OrderExtractor,
    catalog: Catalog,
    source: TranscriptSource = "synthetic",
) -> OrderAccuracyReport:
    comparisons: list[OrderComparison] = []
    for sample in samples:
        result = sample.transcription_result(source)
        if not result.transcripts:
            continue
        voicemail = Voicemail(
            id=sample.id,
            received_at=sample.received_at,
            caller_number=sample.caller_number,
            source_path=Path(f"{sample.id}.wav"),
        )
        try:
            consensus = await reconciler.reconcile(result, catalog.vocabulary())
            order = await extractor.extract(voicemail, consensus)
        except VoiceToOrderError as exc:
            comparisons.append(
                OrderComparison(
                    sample_id=sample.id,
                    account_correct=False,
                    date_correct=False,
                    true_positive_lines=0,
                    extracted_lines=0,
                    expected_lines=len(sample.expected.lines),
                    exact=False,
                    needs_review=True,
                    error=str(exc),
                )
            )
            continue
        delivery = vote_delivery_date(result, consensus, sample.received_at.date())
        order = order.model_copy(update={"delivery": delivery})
        comparisons.append(compare(sample.id, order, sample.expected))

    if not comparisons:
        raise ValueError(f"no samples have {source} transcripts")
    n = len(comparisons)
    extracted = sum(c.extracted_lines for c in comparisons)
    expected = sum(c.expected_lines for c in comparisons)
    hits = sum(c.true_positive_lines for c in comparisons)
    return OrderAccuracyReport(
        samples=n,
        account_accuracy=round(sum(c.account_correct for c in comparisons) / n, 4),
        date_accuracy=round(sum(c.date_correct for c in comparisons) / n, 4),
        line_precision=round(hits / extracted, 4) if extracted else 0.0,
        line_recall=round(hits / expected, 4) if expected else 0.0,
        exact_order_rate=round(sum(c.exact for c in comparisons) / n, 4),
        flagged_for_review=sum(c.needs_review for c in comparisons),
        failed=sum(c.error is not None for c in comparisons),
        comparisons=comparisons,
    )

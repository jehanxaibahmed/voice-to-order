"""Extract a structured order from the consensus transcript."""

from decimal import Decimal

from pydantic import BaseModel, Field

from voice_to_order.domain import (
    ConsensusTranscript,
    Customer,
    ExtractionError,
    LLMError,
    Order,
    OrderLine,
    Voicemail,
)
from voice_to_order.extraction.catalog import Catalog
from voice_to_order.llm import LLMClient

SYSTEM_PROMPT = """\
You turn the transcript of a customer voicemail into a wholesale order. Extract only what \
the caller actually said.

- customer: the caller's name, company, account number and call-back phone number, each \
null if not mentioned.
- lines: one entry per product the caller wants. Use the caller's own words for the \
description. quantity is a number ("a dozen" is 12, "half a case" is 0.5). unit is the \
packaging they asked for ("cases", "trays"), null if they gave none. product_code is the \
catalogue SKU if the caller said a code or the product clearly matches exactly one \
catalogue entry, otherwise null.
- If the caller changes their mind ("actually make that three"), use the final figure.
- Ignore products the caller says they do not need.
- notes: any delivery instructions or requests that are not order lines, else null."""


class _ExtractedCustomer(BaseModel):
    name: str | None
    company: str | None
    account_number: str | None
    phone: str | None


class _ExtractedLine(BaseModel):
    description: str
    quantity: float
    unit: str | None
    product_code: str | None


class _ExtractedOrder(BaseModel):
    customer: _ExtractedCustomer
    lines: list[_ExtractedLine] = Field(description="Products ordered, in the order mentioned.")
    notes: str | None


def build_prompt(transcript: ConsensusTranscript, catalog: Catalog) -> str:
    catalogue = "\n".join(f"{p.sku}: {p.name} ({p.unit})" for p in catalog.products)
    return (
        f"<catalogue>\n{catalogue}\n</catalogue>\n\n"
        f"<transcript>\n{transcript.text}\n</transcript>\n\n"
        "Extract the order from this voicemail transcript."
    )


class OrderExtractor:
    def __init__(self, llm: LLMClient, catalog: Catalog) -> None:
        self._llm = llm
        self._catalog = catalog

    async def extract(self, voicemail: Voicemail, transcript: ConsensusTranscript) -> Order:
        try:
            extracted = await self._llm.structured(
                system=SYSTEM_PROMPT,
                prompt=build_prompt(transcript, self._catalog),
                schema=_ExtractedOrder,
            )
        except LLMError as exc:
            raise ExtractionError(f"could not extract order: {exc}") from exc

        reasons: list[str] = []
        lines: list[OrderLine] = []
        for raw in extracted.lines:
            if raw.quantity <= 0:
                reasons.append(f"non-positive quantity for {raw.description!r}")
                continue
            match = self._catalog.match(raw.description, raw.product_code)
            if match is None:
                reasons.append(f"no catalogue match for {raw.description!r}")
            lines.append(
                OrderLine(
                    description=raw.description,
                    quantity=Decimal(str(raw.quantity)),
                    unit=raw.unit,
                    product_code=raw.product_code,
                    matched_catalog_sku=match.product.sku if match else None,
                )
            )

        customer = Customer(
            name=extracted.customer.name,
            company=extracted.customer.company,
            account_number=extracted.customer.account_number,
            phone=extracted.customer.phone or voicemail.caller_number,
        )
        if not lines:
            reasons.append("no order lines found")
        if not (customer.account_number or customer.company or customer.name):
            reasons.append("caller could not be identified")
        reasons += [f"uncertain transcript: {span!r}" for span in transcript.uncertain_spans]

        return Order(
            voicemail_id=voicemail.id,
            customer=customer,
            lines=lines,
            notes=extracted.notes,
            needs_review=bool(reasons),
            review_reasons=reasons,
        )

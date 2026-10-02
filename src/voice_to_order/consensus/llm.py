"""LLM transcript reconciliation, falling back to ROVER when the LLM is unavailable."""

import logging
from collections.abc import Sequence

from pydantic import BaseModel, Field

from voice_to_order.consensus.rover import RoverReconciler
from voice_to_order.domain import ConsensusTranscript, LLMError, TranscriptionResult
from voice_to_order.llm import LLMClient

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You reconcile speech-to-text transcripts of one customer voicemail into the single most \
likely transcript. The voicemail is a customer placing an order with a wholesale supplier, \
so it usually contains a name or company, an account number, products with quantities, and \
a delivery date.

Each transcript came from a different speech-to-text engine. They make different mistakes: \
misheard product names, merged or split numbers, missing words at the start or end. Work \
word by word:
- Where the engines agree, keep their wording.
- Where they disagree, choose the reading that is best supported by the other engines, the \
surrounding words, and the product vocabulary if one is given.
- Never add information that no transcript contains. If a span is unclear in every \
transcript, keep the most likely reading and list that span in uncertain_spans.
- Write quantities, account numbers and product codes as digits, exactly as spoken."""


class _LLMConsensus(BaseModel):
    text: str = Field(description="The reconciled transcript.")
    uncertain_spans: list[str] = Field(
        description="Short spans of the reconciled text that remain doubtful."
    )


def build_prompt(result: TranscriptionResult, vocabulary: Sequence[str]) -> str:
    parts = []
    for transcript in result.transcripts:
        confidence = (
            f' confidence="{transcript.confidence:.2f}"'
            if transcript.confidence is not None
            else ""
        )
        parts.append(
            f'<transcript engine="{transcript.provider}"{confidence}>\n'
            f"{transcript.text}\n</transcript>"
        )
    if vocabulary:
        parts.append("<product_vocabulary>\n" + "\n".join(vocabulary) + "\n</product_vocabulary>")
    parts.append("Reconcile these transcripts.")
    return "\n\n".join(parts)


class LLMReconciler:
    def __init__(self, llm: LLMClient, fallback: RoverReconciler | None = None) -> None:
        self._llm = llm
        self._fallback = fallback or RoverReconciler()

    async def reconcile(
        self, result: TranscriptionResult, vocabulary: Sequence[str] = ()
    ) -> ConsensusTranscript:
        sources = [t.provider for t in result.transcripts]
        if len(result.transcripts) == 1:
            only = result.transcripts[0]
            return ConsensusTranscript(text=only.text, method="single", sources=sources)
        try:
            answer = await self._llm.structured(
                system=SYSTEM_PROMPT,
                prompt=build_prompt(result, vocabulary),
                schema=_LLMConsensus,
            )
        except LLMError as exc:
            logger.warning("LLM consensus failed, falling back to ROVER: %s", exc)
            return await self._fallback.reconcile(result, vocabulary)
        return ConsensusTranscript(
            text=answer.text.strip(),
            method="llm",
            sources=sources,
            uncertain_spans=answer.uncertain_spans,
        )

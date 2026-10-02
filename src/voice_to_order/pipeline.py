"""Run one voicemail through every stage: ingest, transcribe, reconcile, extract, vote."""

import logging

from pydantic import BaseModel

from voice_to_order.audio import AudioIngestor
from voice_to_order.consensus import Reconciler
from voice_to_order.domain import (
    AudioFile,
    ConsensusTranscript,
    Order,
    TranscriptionResult,
    Voicemail,
)
from voice_to_order.extraction import Catalog, OrderExtractor, review_reason, vote_delivery_date
from voice_to_order.transcription import TranscriptionRunner

logger = logging.getLogger(__name__)


class PipelineResult(BaseModel):
    voicemail: Voicemail
    audio: AudioFile
    transcription: TranscriptionResult
    consensus: ConsensusTranscript
    order: Order


class VoiceToOrderPipeline:
    def __init__(
        self,
        ingestor: AudioIngestor,
        runner: TranscriptionRunner,
        reconciler: Reconciler,
        extractor: OrderExtractor,
        catalog: Catalog,
    ) -> None:
        self._ingestor = ingestor
        self._runner = runner
        self._reconciler = reconciler
        self._extractor = extractor
        self._catalog = catalog

    async def process(self, voicemail: Voicemail) -> PipelineResult:
        logger.info("processing voicemail %s", voicemail.id)
        audio = await self._ingestor.ingest(voicemail)
        transcription = await self._runner.run(audio)
        consensus = await self._reconciler.reconcile(transcription, self._catalog.vocabulary())
        order = await self._extractor.extract(voicemail, consensus)

        delivery = vote_delivery_date(transcription, consensus, voicemail.received_at.date())
        reasons = list(order.review_reasons)
        if reason := review_reason(delivery):
            reasons.append(reason)
        if len(transcription.transcripts) < 2:
            reasons.append("only one transcription provider succeeded")

        order = order.model_copy(
            update={"delivery": delivery, "needs_review": bool(reasons), "review_reasons": reasons}
        )
        logger.info(
            "voicemail %s -> %d lines, needs_review=%s",
            voicemail.id,
            len(order.lines),
            bool(reasons),
        )
        return PipelineResult(
            voicemail=voicemail,
            audio=audio,
            transcription=transcription,
            consensus=consensus,
            order=order,
        )

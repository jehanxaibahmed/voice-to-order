from collections.abc import Sequence
from typing import Protocol

from voice_to_order.domain import ConsensusTranscript, TranscriptionResult


class Reconciler(Protocol):
    async def reconcile(
        self, result: TranscriptionResult, vocabulary: Sequence[str] = ()
    ) -> ConsensusTranscript: ...

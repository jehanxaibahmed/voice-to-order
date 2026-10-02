"""Run the real speech-to-text providers over sample audio and store what they return."""

import logging
import tempfile
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel

from voice_to_order.audio import AudioIngestor
from voice_to_order.domain import Voicemail, VoiceToOrderError
from voice_to_order.evaluation.dataset import Sample, save_sample
from voice_to_order.transcription import TranscriptionRunner

logger = logging.getLogger(__name__)


class RecordingOutcome(BaseModel):
    sample_id: str
    recorded: list[str]
    failed: dict[str, str]


async def record_transcripts(
    samples: Sequence[Sample],
    *,
    audio_dir: Path,
    samples_dir: Path,
    runner: TranscriptionRunner,
    ingestor: AudioIngestor | None = None,
) -> list[RecordingOutcome]:
    """Transcribe ``audio_dir/<id>.wav`` for each sample and save into ``recorded_transcripts``.

    Results are merged per provider, so providers can be recorded in separate runs.
    """
    outcomes: list[RecordingOutcome] = []
    with tempfile.TemporaryDirectory() as work:
        ingestor = ingestor or AudioIngestor(Path(work))
        for sample in samples:
            audio_path = audio_dir / f"{sample.id}.wav"
            if not audio_path.exists():
                logger.warning("no audio for %s at %s", sample.id, audio_path)
                continue
            voicemail = Voicemail(
                id=sample.id, received_at=sample.received_at, source_path=audio_path
            )
            try:
                audio = await ingestor.ingest(voicemail)
                result = await runner.run(audio)
            except VoiceToOrderError as exc:
                outcomes.append(
                    RecordingOutcome(
                        sample_id=sample.id,
                        recorded=[],
                        failed={name: str(exc) for name in runner.provider_names},
                    )
                )
                continue
            sample.recorded_transcripts |= {t.provider: t.text for t in result.transcripts}
            save_sample(samples_dir, sample)
            outcomes.append(
                RecordingOutcome(
                    sample_id=sample.id,
                    recorded=[t.provider for t in result.transcripts],
                    failed={f.provider: f.error for f in result.failures},
                )
            )
    return outcomes

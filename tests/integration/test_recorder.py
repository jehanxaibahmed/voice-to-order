import shutil
from pathlib import Path

import pytest

from tests.helpers import make_tone
from voice_to_order.domain import AudioFile, Transcript, TranscriptionError
from voice_to_order.evaluation import evaluate, load_samples
from voice_to_order.evaluation.recorder import record_transcripts
from voice_to_order.transcription import TranscriptionRunner

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


class EchoProvider:
    """Pretends to transcribe by returning the sample's reference text, minus one word."""

    name = "echo"

    def __init__(self, references: dict[str, str]) -> None:
        self._references = references

    async def transcribe(self, audio: AudioFile) -> Transcript:
        text = self._references[audio.path.parent.name]
        return Transcript(provider=self.name, text=text.split(" ", 1)[1])


class BrokenProvider:
    name = "broken"

    async def transcribe(self, audio: AudioFile) -> Transcript:
        raise TranscriptionError(self.name, "HTTP 401 Unauthorized")


async def test_record_saves_provider_output_and_evaluates(
    samples_dir: Path, tmp_path: Path
) -> None:
    sample_dir = tmp_path / "voicemails"
    shutil.copytree(samples_dir / "voicemails", sample_dir)
    samples = load_samples(sample_dir)[:2]
    for sample in samples:
        sample.recorded_transcripts = {}  # start from a clean slate
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    make_tone(audio_dir / f"{samples[0].id}.wav", seconds=1)  # second sample has no audio

    runner = TranscriptionRunner(
        [EchoProvider({s.id: s.reference for s in samples}), BrokenProvider()]
    )
    outcomes = await record_transcripts(
        samples, audio_dir=audio_dir, samples_dir=sample_dir, runner=runner
    )

    assert len(outcomes) == 1
    assert outcomes[0].recorded == ["echo"]
    assert outcomes[0].failed == {"broken": "HTTP 401 Unauthorized"}
    reloaded = load_samples(sample_dir)[:2]
    assert reloaded[0].recorded_transcripts["echo"] == samples[0].reference.split(" ", 1)[1]
    assert reloaded[0].transcripts == samples[0].transcripts  # synthetic ones untouched
    assert "echo" not in reloaded[1].recorded_transcripts  # no audio, nothing recorded

    report = evaluate(reloaded[:1], "recorded")
    assert [s.name for s in report.scores] == ["echo"]  # no ROVER row for a single engine
    assert report.scores[0].samples == 1
    assert report.scores[0].wer > 0

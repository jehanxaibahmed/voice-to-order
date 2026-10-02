import shutil
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tests.helpers import make_tone
from voice_to_order.audio import AudioIngestor, FFmpeg
from voice_to_order.domain import AudioError, Voicemail

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def voicemail(path: Path) -> Voicemail:
    return Voicemail(id="vm-1", received_at=datetime.now(UTC), source_path=path)


async def test_ingest_normalises_to_16k_mono_wav(tmp_path: Path) -> None:
    source = make_tone(tmp_path / "in.mp3", seconds=2)
    audio = await AudioIngestor(tmp_path / "work").ingest(voicemail(source))

    assert audio.path == tmp_path / "work" / "vm-1" / "normalised.wav"
    assert audio.path.exists()
    assert (audio.sample_rate, audio.channels) == (16_000, 1)
    assert audio.duration_seconds == pytest.approx(2, abs=0.1)


async def test_ingest_can_produce_mp3(tmp_path: Path) -> None:
    source = make_tone(tmp_path / "in.wav", seconds=1)
    audio = await AudioIngestor(tmp_path / "work").ingest(voicemail(source), output_format="mp3")
    assert audio.format == "mp3"


async def test_rejects_unsupported_extension(tmp_path: Path) -> None:
    source = tmp_path / "notes.txt"
    source.write_text("hello")
    with pytest.raises(AudioError, match="unsupported"):
        await AudioIngestor(tmp_path).ingest(voicemail(source))


async def test_rejects_too_short_audio(tmp_path: Path) -> None:
    source = make_tone(tmp_path / "blip.wav", seconds=0.2)
    with pytest.raises(AudioError, match="too short"):
        await AudioIngestor(tmp_path).ingest(voicemail(source))


async def test_rejects_oversized_file(tmp_path: Path) -> None:
    source = make_tone(tmp_path / "in.wav", seconds=1)
    with pytest.raises(AudioError, match="larger than"):
        await AudioIngestor(tmp_path, max_bytes=100).ingest(voicemail(source))


async def test_corrupt_audio_raises_audio_error(tmp_path: Path) -> None:
    source = tmp_path / "broken.wav"
    source.write_bytes(b"RIFF not really a wav file")
    with pytest.raises(AudioError):
        await AudioIngestor(tmp_path).ingest(voicemail(source))


async def test_missing_binary_raises_audio_error(tmp_path: Path) -> None:
    source = make_tone(tmp_path / "in.wav", seconds=1)
    ffmpeg = FFmpeg(ffprobe_binary="definitely-not-ffprobe")
    with pytest.raises(AudioError, match="not found"):
        await AudioIngestor(tmp_path, ffmpeg).ingest(voicemail(source))

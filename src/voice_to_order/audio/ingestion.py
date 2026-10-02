"""Validate an uploaded voicemail and normalise it for the speech-to-text providers."""

from pathlib import Path

from voice_to_order.audio.ffmpeg import FFmpeg, OutputFormat
from voice_to_order.domain import AudioError, AudioFile, Voicemail

SUPPORTED_EXTENSIONS = frozenset({".wav", ".mp3", ".m4a", ".ogg", ".oga", ".opus", ".flac", ".amr"})
MIN_DURATION_SECONDS = 0.5


class AudioIngestor:
    def __init__(
        self,
        work_dir: Path,
        ffmpeg: FFmpeg | None = None,
        *,
        max_bytes: int = 25 * 1024 * 1024,
        max_duration_seconds: float = 600.0,
    ) -> None:
        self._work_dir = work_dir
        self._ffmpeg = ffmpeg or FFmpeg()
        self._max_bytes = max_bytes
        self._max_duration = max_duration_seconds

    async def ingest(self, voicemail: Voicemail, output_format: OutputFormat = "wav") -> AudioFile:
        source = voicemail.source_path
        self._validate_file(source)
        original = await self._ffmpeg.probe(source)
        if original.duration_seconds < MIN_DURATION_SECONDS:
            raise AudioError(f"{source.name} is too short ({original.duration_seconds:.2f}s)")
        if original.duration_seconds > self._max_duration:
            raise AudioError(f"{source.name} is longer than {self._max_duration:.0f}s")
        destination = self._work_dir / voicemail.id / f"normalised.{output_format}"
        return await self._ffmpeg.convert(source, destination, output_format=output_format)

    def _validate_file(self, path: Path) -> None:
        if not path.is_file():
            raise AudioError(f"{path} does not exist")
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise AudioError(f"unsupported audio type {path.suffix!r}")
        size = path.stat().st_size
        if size == 0:
            raise AudioError(f"{path.name} is empty")
        if size > self._max_bytes:
            raise AudioError(f"{path.name} is larger than {self._max_bytes} bytes")

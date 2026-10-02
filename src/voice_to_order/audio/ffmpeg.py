"""Thin async wrapper around the ffmpeg and ffprobe binaries."""

import asyncio
import json
import shutil
from pathlib import Path
from typing import Literal

from voice_to_order.domain import AudioError, AudioFile

OutputFormat = Literal["wav", "mp3"]

_CODEC_ARGS: dict[OutputFormat, list[str]] = {
    "wav": ["-c:a", "pcm_s16le"],
    "mp3": ["-c:a", "libmp3lame", "-q:a", "4"],
}


class FFmpeg:
    def __init__(self, ffmpeg_binary: str = "ffmpeg", ffprobe_binary: str = "ffprobe") -> None:
        self._ffmpeg = ffmpeg_binary
        self._ffprobe = ffprobe_binary

    def is_available(self) -> bool:
        return shutil.which(self._ffmpeg) is not None and shutil.which(self._ffprobe) is not None

    async def probe(self, path: Path) -> AudioFile:
        stdout = await self._run(
            self._ffprobe,
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=sample_rate,channels:format=duration,format_name",
            "-of",
            "json",
            str(path),
        )
        try:
            info = json.loads(stdout)
            stream = info["streams"][0]
            fmt = info["format"]
            return AudioFile(
                path=path,
                format=fmt["format_name"].split(",")[0],
                sample_rate=int(stream["sample_rate"]),
                channels=int(stream["channels"]),
                duration_seconds=float(fmt["duration"]),
            )
        except (KeyError, IndexError, ValueError, json.JSONDecodeError) as exc:
            raise AudioError(f"{path.name} has no readable audio stream") from exc

    async def convert(
        self,
        source: Path,
        destination: Path,
        *,
        output_format: OutputFormat = "wav",
        sample_rate: int = 16_000,
        channels: int = 1,
    ) -> AudioFile:
        """Convert ``source`` to a mono, resampled file (speech models expect 16 kHz mono)."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        await self._run(
            self._ffmpeg,
            "-nostdin",
            "-y",
            "-v",
            "error",
            "-i",
            str(source),
            "-vn",
            "-ac",
            str(channels),
            "-ar",
            str(sample_rate),
            *_CODEC_ARGS[output_format],
            str(destination),
        )
        return await self.probe(destination)

    @staticmethod
    async def _run(binary: str, *args: str) -> str:
        try:
            process = await asyncio.create_subprocess_exec(
                binary,
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise AudioError(f"{binary} not found on PATH") from exc
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            message = stderr.decode(errors="replace").strip().splitlines()
            raise AudioError(f"{binary} failed: {message[-1] if message else 'unknown error'}")
        return stdout.decode()

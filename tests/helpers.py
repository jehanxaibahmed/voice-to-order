"""Shared test helpers."""

import subprocess
from pathlib import Path


def make_tone(path: Path, *, seconds: float, sample_rate: int = 44_100, channels: int = 2) -> Path:
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-y", "-v", "error",
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}:sample_rate={sample_rate}",
            "-ac", str(channels), str(path),
        ],
        check=True,
    )  # fmt: skip
    return path

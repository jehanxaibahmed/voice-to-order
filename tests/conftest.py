from pathlib import Path

import pytest

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"

_CREDENTIAL_VARS = [
    f"{prefix}{name}_API_KEY"
    for prefix in ("", "VTO_")
    for name in ("ANTHROPIC", "DEEPGRAM", "OPENAI", "ASSEMBLYAI")
]


@pytest.fixture(autouse=True)
def _no_real_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep real API keys from the developer's shell out of every test."""
    for var in _CREDENTIAL_VARS:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def samples_dir() -> Path:
    return SAMPLES_DIR

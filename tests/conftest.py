from pathlib import Path

import pytest

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"


@pytest.fixture
def samples_dir() -> Path:
    return SAMPLES_DIR

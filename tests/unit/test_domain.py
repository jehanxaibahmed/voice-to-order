from decimal import Decimal

import pytest
from pydantic import ValidationError

from voice_to_order.config import Settings
from voice_to_order.domain import OrderLine, Transcript, TranscriptionResult


def test_transcription_result_indexes_by_provider() -> None:
    result = TranscriptionResult(
        transcripts=[Transcript(provider="a", text="x"), Transcript(provider="b", text="y")]
    )
    assert result.by_provider()["b"].text == "y"


def test_order_line_rejects_non_positive_quantity() -> None:
    with pytest.raises(ValidationError):
        OrderLine(description="milk", quantity=Decimal(0))


def test_confidence_must_be_probability() -> None:
    with pytest.raises(ValidationError):
        Transcript(provider="a", text="x", confidence=1.5)


def test_settings_split_provider_csv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VTO_TRANSCRIPTION_PROVIDERS", "deepgram, whisper ,")
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.transcription_providers == ["deepgram", "whisper"]


def test_settings_accept_standard_key_names(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "standard")
    monkeypatch.setenv("VTO_DEEPGRAM_API_KEY", "prefixed")
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.openai_api_key is not None
    assert settings.openai_api_key.get_secret_value() == "standard"
    assert settings.deepgram_api_key is not None
    assert settings.deepgram_api_key.get_secret_value() == "prefixed"

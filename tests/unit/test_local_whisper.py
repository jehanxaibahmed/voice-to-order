import sys
import types
from pathlib import Path
from typing import Any

import httpx
import pytest

from voice_to_order.config import Settings
from voice_to_order.domain import AudioFile, TranscriptionError
from voice_to_order.transcription import TranscriptionRunner, build_providers
from voice_to_order.transcription.providers import LocalWhisperProvider


@pytest.fixture
def audio(tmp_path: Path) -> AudioFile:
    path = tmp_path / "a.wav"
    path.write_bytes(b"RIFF")
    return AudioFile(path=path, format="wav", sample_rate=16000, channels=1, duration_seconds=1.0)


class Segment:
    def __init__(self, text: str) -> None:
        self.text = text


def install_fake(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    class WhisperModel:
        def __init__(self, size: str, **kwargs: Any) -> None:
            calls.append({"init": size, **kwargs})

        def transcribe(self, path: str, **kwargs: Any) -> tuple[Any, None]:
            calls.append({"path": path, **kwargs})
            return iter([Segment(" two cases "), Segment("of Oatly ")]), None

    module = types.ModuleType("faster_whisper")
    module.WhisperModel = WhisperModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    return calls


async def test_transcribes_with_vocabulary_prompt(
    monkeypatch: pytest.MonkeyPatch, audio: AudioFile
) -> None:
    calls = install_fake(monkeypatch)
    provider = LocalWhisperProvider(
        model_size="base", compute_type="int8", vocabulary_hint="Oatly, OM-12"
    )
    transcript = await provider.transcribe(audio)
    await provider.transcribe(audio)

    assert transcript.provider == "whisper-local"
    assert transcript.text == "two cases of Oatly"
    assert calls[0] == {"init": "base", "device": "auto", "compute_type": "int8"}
    assert calls[1]["initial_prompt"] == "Oatly, OM-12"
    assert calls[1]["path"] == str(audio.path)
    assert sum("init" in c for c in calls) == 1  # model loaded once


async def test_missing_dependency_is_a_clear_error(
    monkeypatch: pytest.MonkeyPatch, audio: AudioFile
) -> None:
    monkeypatch.setitem(sys.modules, "faster_whisper", None)  # makes the import fail
    with pytest.raises(TranscriptionError, match=r"pip install -e '\.\[local\]'"):
        await LocalWhisperProvider().transcribe(audio)


async def test_inference_failure_is_wrapped(
    monkeypatch: pytest.MonkeyPatch, audio: AudioFile
) -> None:
    install_fake(monkeypatch)
    audio.path.unlink()  # fake ignores it, so break decoding differently
    provider = LocalWhisperProvider()
    monkeypatch.setattr(provider, "_load_model", lambda: (_ for _ in ()).throw(OSError("bad")))
    with pytest.raises(TranscriptionError, match="OSError: bad"):
        await provider.transcribe(audio)


async def test_registry_enables_without_api_key() -> None:
    settings = Settings(_env_file=None, transcription_providers="whisper-local,whisper,deepgram")  # type: ignore[call-arg]
    async with httpx.AsyncClient() as client:
        providers = build_providers(settings, client, vocabulary=["Oatly", "OM-12"])
    assert [p.name for p in providers] == ["whisper-local"]


async def test_runs_beside_cloud_provider_with_own_timeout(
    monkeypatch: pytest.MonkeyPatch, audio: AudioFile
) -> None:
    install_fake(monkeypatch)
    settings = Settings(  # type: ignore[call-arg]
        _env_file=None,
        transcription_providers="whisper-local,deepgram",
        deepgram_api_key="d",
        local_whisper_timeout_seconds=123,
    )
    async with httpx.AsyncClient() as client:
        providers = build_providers(settings, client)
    assert [p.name for p in providers] == ["whisper-local", "deepgram"]
    assert getattr(providers[0], "timeout_seconds") == 123  # noqa: B009

    runner = TranscriptionRunner([providers[0]], timeout_seconds=0.001)
    result = await runner.run(audio)  # the provider's own timeout overrides the runner default
    assert result.transcripts[0].text == "two cases of Oatly"

import json
from pathlib import Path

import httpx
import pytest

from voice_to_order.config import Settings
from voice_to_order.domain import AudioFile, TranscriptionError
from voice_to_order.transcription import build_providers
from voice_to_order.transcription.providers import (
    AssemblyAIProvider,
    DeepgramProvider,
    ReplayProvider,
    WhisperProvider,
)


@pytest.fixture
def audio(tmp_path: Path) -> AudioFile:
    path = tmp_path / "vm-7" / "normalised.wav"
    path.parent.mkdir()
    path.write_bytes(b"RIFF....fake")
    return AudioFile(path=path, format="wav", sample_rate=16_000, channels=1, duration_seconds=3)


def client_returning(handler: httpx.MockTransport) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=handler)


async def test_deepgram_sends_audio_and_keyterms(audio: AudioFile) -> None:
    seen: dict[str, httpx.Request] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["request"] = request
        body = {"results": {"channels": [{"alternatives": [
            {"transcript": " two cases of oat milk ", "confidence": 0.93}
        ]}]}}  # fmt: skip
        return httpx.Response(200, json=body)

    async with client_returning(httpx.MockTransport(handler)) as client:
        provider = DeepgramProvider("dg-key", client, keyterms=["OM-12", "Oatly"])
        transcript = await provider.transcribe(audio)

    request = seen["request"]
    assert request.headers["Authorization"] == "Token dg-key"
    assert request.headers["Content-Type"] == "audio/wav"
    assert request.url.params.get_list("keyterm") == ["OM-12", "Oatly"]
    assert request.content == b"RIFF....fake"
    assert transcript.text == "two cases of oat milk"
    assert transcript.confidence == pytest.approx(0.93)


async def test_deepgram_http_error_becomes_transcription_error(audio: AudioFile) -> None:
    transport = httpx.MockTransport(lambda _: httpx.Response(401, json={"err": "bad key"}))
    async with client_returning(transport) as client:
        with pytest.raises(TranscriptionError, match="deepgram"):
            await DeepgramProvider("k", client).transcribe(audio)


async def test_deepgram_unexpected_shape(audio: AudioFile) -> None:
    transport = httpx.MockTransport(lambda _: httpx.Response(200, json={"results": {}}))
    async with client_returning(transport) as client:
        with pytest.raises(TranscriptionError, match="unexpected response"):
            await DeepgramProvider("k", client).transcribe(audio)


async def test_whisper_posts_multipart_with_prompt(audio: AudioFile) -> None:
    seen: dict[str, httpx.Request] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["request"] = request
        return httpx.Response(200, json={"text": "Two cases of oat milk."})

    async with client_returning(httpx.MockTransport(handler)) as client:
        provider = WhisperProvider("oa-key", client, vocabulary_hint="Oatly, OM-12")
        transcript = await provider.transcribe(audio)

    request = seen["request"]
    body = request.content.decode(errors="replace")
    assert request.headers["Authorization"] == "Bearer oa-key"
    assert 'name="prompt"' in body and "Oatly, OM-12" in body
    assert 'filename="normalised.wav"' in body
    assert transcript.text == "Two cases of oat milk."
    assert transcript.confidence is None


async def test_whisper_network_error(audio: AudioFile) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom", request=request)

    async with client_returning(httpx.MockTransport(handler)) as client:
        with pytest.raises(TranscriptionError, match="whisper"):
            await WhisperProvider("k", client).transcribe(audio)


async def test_replay_provider_looks_up_by_voicemail_dir(audio: AudioFile) -> None:
    provider = ReplayProvider("recorded", {"vm-7": "hello"})
    assert (await provider.transcribe(audio)).text == "hello"
    with pytest.raises(TranscriptionError):
        await ReplayProvider("recorded", {}).transcribe(audio)


async def test_registry_skips_providers_without_keys() -> None:
    settings = Settings(_env_file=None, deepgram_api_key="dg")  # type: ignore[call-arg]
    async with httpx.AsyncClient() as client:
        providers = build_providers(settings, client, vocabulary=["OM-12"])
    assert [p.name for p in providers] == ["deepgram"]


async def test_assemblyai_uploads_creates_job_and_polls(audio: AudioFile) -> None:
    polls = iter(["queued", "processing", "completed"])
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path == "/v2/upload":
            return httpx.Response(200, json={"upload_url": "https://cdn/x"})
        if request.method == "POST":
            return httpx.Response(200, json={"id": "t1", "status": "queued"})
        status = next(polls)
        body = {"id": "t1", "status": status, "text": " Four cases. ", "confidence": 0.88}
        return httpx.Response(200, json=body)

    async with client_returning(httpx.MockTransport(handler)) as client:
        provider = AssemblyAIProvider(
            "aai-key", client, word_boost=["OM-12"], poll_interval_seconds=0
        )
        transcript = await provider.transcribe(audio)

    assert transcript.text == "Four cases."
    assert transcript.confidence == pytest.approx(0.88)
    assert seen[0].content == b"RIFF....fake"
    assert seen[0].headers["Authorization"] == "aai-key"
    job = json.loads(seen[1].content)
    assert job["audio_url"] == "https://cdn/x"
    assert job["word_boost"] == ["OM-12"]
    assert [r.url.path for r in seen[2:]] == ["/v2/transcript/t1"] * 3


async def test_assemblyai_job_error(audio: AudioFile) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v2/upload":
            return httpx.Response(200, json={"upload_url": "u"})
        if request.method == "POST":
            return httpx.Response(200, json={"id": "t1"})
        return httpx.Response(200, json={"status": "error", "error": "audio too short"})

    async with client_returning(httpx.MockTransport(handler)) as client:
        with pytest.raises(TranscriptionError, match="audio too short"):
            await AssemblyAIProvider("k", client, poll_interval_seconds=0).transcribe(audio)


async def test_assemblyai_missing_field(audio: AudioFile) -> None:
    transport = httpx.MockTransport(lambda _: httpx.Response(200, json={}))
    async with client_returning(transport) as client:
        with pytest.raises(TranscriptionError, match="upload_url"):
            await AssemblyAIProvider("k", client).transcribe(audio)


async def test_http_errors_are_short_and_omit_the_url(audio: AudioFile) -> None:
    transport = httpx.MockTransport(lambda _: httpx.Response(401))
    async with client_returning(transport) as client:
        with pytest.raises(TranscriptionError) as caught:
            await DeepgramProvider("k", client, keyterms=["OM-12"]).transcribe(audio)
    assert caught.value.message == "HTTP 401 Unauthorized"
    assert "keyterm" not in str(caught.value)

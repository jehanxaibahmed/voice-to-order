import asyncio
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from tests.unit.test_worker import FakeProcessor
from voice_to_order.api import create_app
from voice_to_order.config import Settings
from voice_to_order.domain import Voicemail
from voice_to_order.pipeline import PipelineResult
from voice_to_order.worker import Processor


class GatedProcessor(FakeProcessor):
    """Holds jobs until released, so tests can observe the queued state."""

    def __init__(self) -> None:
        self.release = asyncio.Event()

    async def process(self, vm: Voicemail) -> PipelineResult:
        await self.release.wait()
        return await super().process(vm)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(_env_file=None, work_dir=tmp_path, max_upload_bytes=1000)  # type: ignore[call-arg]


def client_with(settings: Settings, processor: Processor) -> Iterator[TestClient]:
    def factory(_: Settings, __: httpx.AsyncClient) -> Processor:
        return processor

    with TestClient(create_app(settings, factory)) as client:
        yield client


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    yield from client_with(settings, FakeProcessor())


def wait_for_status(client: TestClient, job_id: str, wanted: str) -> dict[str, object]:
    for _ in range(100):
        body: dict[str, object] = client.get(f"/jobs/{job_id}").json()
        if body["status"] == wanted:
            return body
        client.portal.call(asyncio.sleep, 0.01)  # type: ignore[union-attr]
    raise AssertionError(f"job never reached {wanted}")


def test_health(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_upload_then_fetch_order(client: TestClient, settings: Settings) -> None:
    response = client.post(
        "/voicemails",
        files={"file": ("message.mp3", b"ID3fake-audio", "audio/mpeg")},
        data={"caller_number": "+44 7700 900999", "received_at": "2026-10-02T09:00:00Z"},
    )
    assert response.status_code == 202
    accepted = response.json()
    assert accepted["status_url"] == f"/jobs/{accepted['job_id']}"

    job = wait_for_status(client, accepted["job_id"], "completed")
    voicemail = job["voicemail"]
    assert isinstance(voicemail, dict)
    assert voicemail["caller_number"] == "+44 7700 900999"
    assert Path(voicemail["source_path"]).parent == settings.work_dir / "uploads"

    order = client.get(f"/jobs/{accepted['job_id']}/order").json()
    assert order["voicemail_id"] == voicemail["id"]


def test_order_not_ready_is_409(settings: Settings) -> None:
    processor = GatedProcessor()
    for client in client_with(settings, processor):
        job_id = client.post("/voicemails", files={"file": ("a.wav", b"RIFF", "audio/wav")}).json()[
            "job_id"
        ]
        assert client.get(f"/jobs/{job_id}/order").status_code == 409
        client.portal.call(processor.release.set)  # type: ignore[union-attr]
        wait_for_status(client, job_id, "completed")


def test_rejects_unsupported_type(client: TestClient) -> None:
    response = client.post("/voicemails", files={"file": ("notes.txt", b"hi", "text/plain")})
    assert response.status_code == 415


def test_rejects_oversized_and_empty_uploads(client: TestClient, settings: Settings) -> None:
    big = client.post("/voicemails", files={"file": ("a.wav", b"x" * 1001, "audio/wav")})
    empty = client.post("/voicemails", files={"file": ("a.wav", b"", "audio/wav")})
    assert (big.status_code, empty.status_code) == (413, 400)
    assert list((settings.work_dir / "uploads").iterdir()) == []


def test_unknown_job_is_404(client: TestClient) -> None:
    assert client.get("/jobs/nope").status_code == 404
    assert client.get("/jobs/nope/order").status_code == 404

import datetime as dt
from pathlib import Path

from voice_to_order.domain import (
    AudioError,
    AudioFile,
    ConsensusTranscript,
    Customer,
    Order,
    Transcript,
    TranscriptionResult,
    Voicemail,
)
from voice_to_order.pipeline import PipelineResult
from voice_to_order.worker import InMemoryJobStore, JobStatus, Worker


def voicemail(vm_id: str) -> Voicemail:
    return Voicemail(id=vm_id, received_at=dt.datetime.now(dt.UTC), source_path=Path(vm_id))


def fake_result(vm: Voicemail) -> PipelineResult:
    return PipelineResult(
        voicemail=vm,
        audio=AudioFile(
            path=vm.source_path, format="wav", sample_rate=16_000, channels=1, duration_seconds=1
        ),
        transcription=TranscriptionResult(transcripts=[Transcript(provider="a", text="hi")]),
        consensus=ConsensusTranscript(text="hi", method="single", sources=["a"]),
        order=Order(voicemail_id=vm.id, customer=Customer(), lines=[]),
    )


class FakeProcessor:
    async def process(self, vm: Voicemail) -> PipelineResult:
        if vm.id == "bad-audio":
            raise AudioError("corrupt")
        if vm.id == "crash":
            raise RuntimeError("secret internals")
        return fake_result(vm)


async def test_worker_processes_jobs_and_records_failures() -> None:
    store = InMemoryJobStore()
    worker = Worker(FakeProcessor(), store, concurrency=2)
    await worker.start()
    try:
        jobs = [await worker.submit(voicemail(i)) for i in ("ok", "bad-audio", "crash")]
        await worker.join()
    finally:
        await worker.stop()

    ok, bad, crash = [await store.get(job.id) for job in jobs]
    assert ok is not None and ok.status is JobStatus.COMPLETED
    assert ok.result is not None and ok.result.order.voicemail_id == "ok"
    assert bad is not None and (bad.status, bad.error) == (JobStatus.FAILED, "corrupt")
    # unexpected exceptions are not leaked to API clients
    assert crash is not None and (crash.status, crash.error) == (JobStatus.FAILED, "internal error")


async def test_store_returns_copies() -> None:
    store = InMemoryJobStore()
    worker = Worker(FakeProcessor(), store)
    job = await worker.submit(voicemail("x"))
    fetched = await store.get(job.id)
    assert fetched is not None
    fetched.status = JobStatus.FAILED
    again = await store.get(job.id)
    assert again is not None and again.status is JobStatus.QUEUED

"""An asyncio worker pool that runs the pipeline for queued voicemails."""

import asyncio
import logging
from typing import Protocol

from voice_to_order.domain import Voicemail, VoiceToOrderError
from voice_to_order.pipeline import PipelineResult
from voice_to_order.worker.jobs import Job, JobStatus, JobStore

logger = logging.getLogger(__name__)


class Processor(Protocol):
    async def process(self, voicemail: Voicemail) -> PipelineResult: ...


class Worker:
    def __init__(self, processor: Processor, store: JobStore, *, concurrency: int = 2) -> None:
        self._processor = processor
        self._store = store
        self._concurrency = concurrency
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._tasks: list[asyncio.Task[None]] = []

    async def start(self) -> None:
        self._tasks = [
            asyncio.create_task(self._loop(), name=f"worker-{i}") for i in range(self._concurrency)
        ]

    async def stop(self) -> None:
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []

    async def submit(self, voicemail: Voicemail) -> Job:
        job = Job(voicemail=voicemail)
        await self._store.add(job)
        await self._queue.put(job.id)
        return job

    async def join(self) -> None:
        """Wait until every submitted job has finished."""
        await self._queue.join()

    async def _loop(self) -> None:
        while True:
            job_id = await self._queue.get()
            try:
                await self._run(job_id)
            finally:
                self._queue.task_done()

    async def _run(self, job_id: str) -> None:
        job = await self._store.get(job_id)
        if job is None:
            return
        job.status = JobStatus.PROCESSING
        await self._store.update(job)
        try:
            job.result = await self._processor.process(job.voicemail)
            job.status = JobStatus.COMPLETED
        except VoiceToOrderError as exc:
            logger.warning("job %s failed: %s", job_id, exc)
            job.status, job.error = JobStatus.FAILED, str(exc)
        except Exception:
            logger.exception("job %s crashed", job_id)
            job.status, job.error = JobStatus.FAILED, "internal error"
        await self._store.update(job)

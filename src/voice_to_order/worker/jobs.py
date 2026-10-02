"""Job records and storage."""

import asyncio
import datetime as dt
import uuid
from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, Field

from voice_to_order.domain import Voicemail
from voice_to_order.pipeline import PipelineResult


def _now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


class JobStatus(StrEnum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Job(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    voicemail: Voicemail
    status: JobStatus = JobStatus.QUEUED
    created_at: dt.datetime = Field(default_factory=_now)
    updated_at: dt.datetime = Field(default_factory=_now)
    result: PipelineResult | None = None
    error: str | None = None


class JobStore(Protocol):
    async def add(self, job: Job) -> None: ...
    async def get(self, job_id: str) -> Job | None: ...
    async def update(self, job: Job) -> None: ...


class InMemoryJobStore:
    """Process-local store. Swap for Redis or Postgres when running several API replicas."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = asyncio.Lock()

    async def add(self, job: Job) -> None:
        async with self._lock:
            self._jobs[job.id] = job

    async def get(self, job_id: str) -> Job | None:
        async with self._lock:
            job = self._jobs.get(job_id)
            return job.model_copy() if job else None

    async def update(self, job: Job) -> None:
        async with self._lock:
            self._jobs[job.id] = job.model_copy(update={"updated_at": _now()})

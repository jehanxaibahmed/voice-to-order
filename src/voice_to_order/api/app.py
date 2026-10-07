"""FastAPI app: upload a voicemail, poll the job, fetch the order."""

import datetime as dt
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import httpx
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Security, UploadFile, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

from voice_to_order.audio import SUPPORTED_EXTENSIONS
from voice_to_order.config import Settings, get_settings
from voice_to_order.domain import Order, Voicemail
from voice_to_order.worker import InMemoryJobStore, Job, JobStatus, JobStore, Processor, Worker

ProcessorFactory = Callable[[Settings, httpx.AsyncClient], Processor]

_CHUNK = 1024 * 1024


class JobAccepted(BaseModel):
    job_id: str
    status: JobStatus
    status_url: str


def _default_processor(settings: Settings, client: httpx.AsyncClient) -> Processor:
    from voice_to_order.bootstrap import build_pipeline

    return build_pipeline(settings, client)


api_key_header = APIKeyHeader(name="X-API-Key")


def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    if api_key != get_settings().api_key_secret:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return api_key


def create_app(
    settings: Settings | None = None,
    processor_factory: ProcessorFactory = _default_processor,
    store: JobStore | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    store = store or InMemoryJobStore()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with httpx.AsyncClient(timeout=settings.transcription_timeout_seconds) as client:
            worker = Worker(processor_factory(settings, client), store)
            await worker.start()
            app.state.worker = worker
            try:
                yield
            finally:
                await worker.stop()

    app = FastAPI(title="Voice to Order", version="0.1.0", lifespan=lifespan)
    
    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/voicemails", status_code=status.HTTP_202_ACCEPTED, dependencies=[Depends(verify_api_key)])
    async def upload_voicemail(
        request: Request,
        file: Annotated[UploadFile, File(description="Voicemail audio")],
        caller_number: Annotated[str | None, Form()] = None,
        received_at: Annotated[dt.datetime | None, Form()] = None,
    ) -> JobAccepted:
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, f"unsupported {suffix!r}")

        voicemail_id = f"vm-{uuid.uuid4().hex[:12]}"
        destination = settings.work_dir / "uploads" / f"{voicemail_id}{suffix}"
        await _save_upload(file, destination, settings.max_upload_bytes)

        worker: Worker = request.app.state.worker
        job = await worker.submit(
            Voicemail(
                id=voicemail_id,
                received_at=received_at or dt.datetime.now(dt.UTC),
                caller_number=caller_number,
                source_path=destination,
            )
        )
        return JobAccepted(job_id=job.id, status=job.status, status_url=f"/jobs/{job.id}")

    @app.get("/jobs/{job_id}", dependencies=[Depends(verify_api_key)])
    async def get_job(job_id: str) -> Job:
        return await _require_job(store, job_id)

    @app.get("/jobs/{job_id}/order", dependencies=[Depends(verify_api_key)])
    async def get_order(job_id: str) -> Order:
        job = await _require_job(store, job_id)
        if job.status is JobStatus.FAILED:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, job.error)
        if job.result is None:
            raise HTTPException(status.HTTP_409_CONFLICT, f"job is {job.status}")
        return job.result.order

    return app


async def _require_job(store: JobStore, job_id: str) -> Job:
    job = await store.get(job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "job not found")
    return job


async def _save_upload(file: UploadFile, destination: Path, max_bytes: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with destination.open("wb") as out:
        while chunk := await file.read(_CHUNK):
            written += len(chunk)
            if written > max_bytes:
                out.close()
                destination.unlink(missing_ok=True)
                raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, "file too large")
            out.write(chunk)
    if written == 0:
        destination.unlink(missing_ok=True)
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "file is empty")

"""Async job pipeline: wraps ClipPipeline so the Flask API can submit a
video link, return immediately with a job id, and let the caller poll for
progress/results instead of blocking on a request for however long
transcription + scoring + cutting takes.

Backend is in-process daemon threads plus an in-memory JobStore -- no new
infra (Redis, Celery, etc) needed to start testing against real videos.
Job state and the run loop are behind small interfaces (JobStore, JobRunner)
specifically so a persistent/distributed backend (RQ, Celery) can replace
them later without changing api.py's calling convention.
"""
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from pipeline import ClipPipeline
from sources import resolve_source


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


# Ordered so the API can report "stage 2 of 5" style progress.
STAGES = ["ingest", "transcribe", "select_clips", "cut", "reformat"]


@dataclass
class Job:
    id: str
    url: str
    vertical: bool
    use_ollama: bool
    multi_speaker: bool
    status: JobStatus = JobStatus.QUEUED
    stage: Optional[str] = None
    error: Optional[str] = None
    output_dir: Optional[str] = None
    clips: List[dict] = field(default_factory=list)
    clip_paths: List[str] = field(default_factory=list)
    vertical_paths: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "url": self.url,
            "status": self.status.value,
            "stage": self.stage,
            "error": self.error,
            "output_dir": self.output_dir,
            "clips": self.clips,
            "clip_paths": self.clip_paths,
            "vertical_paths": self.vertical_paths,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class JobStore:
    """Thread-safe in-memory job registry. Swap for a persistent store
    (SQLite, Redis) later by matching this get/put interface.
    """

    def __init__(self):
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()

    def put(self, job: Job) -> None:
        with self._lock:
            self._jobs[job.id] = job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> List[Job]:
        with self._lock:
            return list(self._jobs.values())

    def update(self, job_id: str, **fields) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            for key, value in fields.items():
                setattr(job, key, value)
            job.updated_at = time.time()


class JobRunner:
    """Runs each job's pipeline stages in its own daemon thread. One runner
    is shared process-wide; submit() is safe to call concurrently.
    """

    def __init__(self, store: JobStore):
        self.store = store

    def submit(
        self,
        url: str,
        vertical: bool = False,
        use_ollama: bool = False,
        multi_speaker: bool = True,
        output_dir: Optional[str] = None,
    ) -> Job:
        job = Job(id=uuid.uuid4().hex, url=url, vertical=vertical, use_ollama=use_ollama, multi_speaker=multi_speaker)
        self.store.put(job)
        thread = threading.Thread(target=self._run, args=(job, output_dir), daemon=True)
        thread.start()
        return job

    def _run(self, job: Job, output_dir: Optional[str]) -> None:
        self.store.update(job.id, status=JobStatus.RUNNING)
        try:
            source = resolve_source(job.url)
            pipeline = ClipPipeline(source)

            self.store.update(job.id, stage="ingest")
            pipeline.ingest()

            self.store.update(job.id, stage="transcribe")
            pipeline.transcribe()

            self.store.update(job.id, stage="select_clips")
            clips = pipeline.select_highlights(use_ollama=job.use_ollama)
            if not clips:
                clips = pipeline.segment_by_topic()
            titles = pipeline.generate_titles(clips)

            self.store.update(job.id, stage="cut")
            resolved_output_dir, clip_paths = pipeline.save_clips(clips, titles, output_dir or "segments")

            vertical_paths = []
            if job.vertical:
                self.store.update(job.id, stage="reformat")
                vertical_paths = pipeline.reformat_all(clips, clip_paths, multi_speaker=job.multi_speaker)

            self.store.update(
                job.id,
                status=JobStatus.DONE,
                stage=None,
                output_dir=resolved_output_dir,
                clips=clips,
                clip_paths=clip_paths,
                vertical_paths=vertical_paths,
            )
        except Exception as exc:
            self.store.update(
                job.id,
                status=JobStatus.FAILED,
                error=f"{exc}\n{traceback.format_exc()}",
            )

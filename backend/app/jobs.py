"""Background job runner.

Fits and backtests run in a bounded ThreadPoolExecutor.  Status/progress is
held in an in-process dict keyed by job id.  Every job gets its own worker
invocation with its own data snapshot, so several planners working on
different series do not block each other beyond the worker pool size.
"""
from __future__ import annotations

import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from .config import settings


@dataclass
class Job:
    id: str
    kind: str            # "fit" | "backtest"
    status: str = "pending"     # pending | running | done | error
    progress: float = 0.0
    stage: str = "排队中"
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    series_id: Optional[int] = None
    created_id: Optional[int] = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def public(self) -> dict:
        with self._lock:
            return {
                "id": self.id,
                "kind": self.kind,
                "status": self.status,
                "progress": round(self.progress, 4),
                "stage": self.stage,
                "result": self.result,
                "error": self.error,
                "series_id": self.series_id,
                "created_id": self.created_id,
            }


class JobManager:
    def __init__(self, workers: int = settings.job_workers) -> None:
        self._pool = ThreadPoolExecutor(max_workers=workers,
                                        thread_name_prefix="hwjob")
        self._jobs: Dict[str, Job] = {}
        self._guard = threading.Lock()

    def submit(self, kind: str, fn: Callable[[Callable[[float, str], None]],
               Dict[str, Any]], series_id: Optional[int] = None) -> Job:
        job_id = uuid.uuid4().hex
        job = Job(id=job_id, kind=kind, series_id=series_id)
        with self._guard:
            self._jobs[job_id] = job

        def task() -> None:
            def report(p: float, stage: str) -> None:
                with job._lock:
                    job.progress = max(0.0, min(1.0, float(p)))
                    job.stage = stage

            with job._lock:
                job.status = "running"
                job.stage = "启动"
            try:
                out = fn(report)
                with job._lock:
                    job.status = "done"
                    job.progress = 1.0
                    job.stage = "完成"
                    if isinstance(out, dict):
                        job.result = out.get("result")
                        job.created_id = out.get("created_id")
                    else:
                        job.result = out
            except Exception as exc:  # noqa: BLE001
                with job._lock:
                    job.status = "error"
                    job.error = str(exc)
                    job.stage = "失败"
                    job.result = None
                traceback.print_exc()

        self._pool.submit(task)
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._guard:
            return self._jobs.get(job_id)


job_manager = JobManager()

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..jobs import job_manager
from ..schemas import JobOut

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


@router.get("/{job_id}", response_model=JobOut)
def status(job_id: str):
    job = job_manager.get(job_id)
    if job is None:
        raise HTTPException(404, "任务不存在")
    return job.public()

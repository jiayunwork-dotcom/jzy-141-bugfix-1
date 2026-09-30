from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..db import get_db
from ..jobs import job_manager
from ..kernels.hw import ModelError, SEASONALS, TRENDS
from ..schemas import FitRequest, JobOut
from .. import service, storage

router = APIRouter(prefix="/api/fits", tags=["fits"])


def _fit_out(f) -> dict:
    return {
        "id": f.id,
        "series_id": f.series_id,
        "created_at": f.created_at.isoformat() if f.created_at else "",
        "label": f.label,
        "auto": f.auto,
        "trend_kind": f.trend_kind,
        "seasonal_kind": f.seasonal_kind,
        "period": f.period,
        "params": f.params,
        "locks": f.locks,
        "sse": f.sse,
        "aic": f.aic,
        "residuals": f.residuals,
        "fitted": f.fitted,
        "forecast": f.forecast,
        "initial_state": f.initial_state,
        "final_state": f.final_state,
        "scores": f.scores or [],
    }


@router.post("", response_model=JobOut)
def fit(req: FitRequest):
    if not req.auto:
        if req.trend_kind not in TRENDS:
            raise HTTPException(400, f"trend_kind 必须是 {TRENDS} 之一")
        if req.seasonal_kind not in SEASONALS:
            raise HTTPException(400, f"seasonal_kind 必须是 {SEASONALS} 之一")
    if req.horizon < 1:
        raise HTTPException(400, "horizon 必须 >= 1")
    if not 0.5 <= req.confidence < 1:
        raise HTTPException(400, "confidence 必须在 [0.5, 1)")
    if req.interval_method not in ("analytic", "simulate"):
        raise HTTPException(400, "interval_method 只能是 analytic/simulate")
    for name in req.locks:
        if name not in ("alpha", "beta", "gamma", "phi"):
            raise HTTPException(400, f"未知参数：{name}")
    job = job_manager.submit(
        "fit",
        lambda report: service.run_fit_job(req.series_id, req, report),
        series_id=req.series_id,
    )
    return job.public()


@router.get("/by-series/{series_id}")
def list_for_series(series_id: int, db: Session = Depends(get_db)):
    return [_fit_out(f) for f in storage.list_fits(db, series_id)]


@router.get("/{fit_id}")
def detail(fit_id: int, db: Session = Depends(get_db)):
    f = storage.get_fit(db, fit_id)
    if f is None:
        raise HTTPException(404, "拟合结果不存在")
    return _fit_out(f)

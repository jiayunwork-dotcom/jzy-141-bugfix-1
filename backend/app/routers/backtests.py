from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..db import get_db
from ..jobs import job_manager
from ..kernels.hw import SEASONALS, TRENDS
from ..schemas import BacktestRequest, JobOut
from .. import service, storage

router = APIRouter(prefix="/api/backtests", tags=["backtests"])


def _bt_out(b) -> dict:
    return {
        "id": b.id,
        "series_id": b.series_id,
        "created_at": b.created_at.isoformat() if b.created_at else "",
        "label": b.label,
        "origin_start": b.origin_start,
        "horizon": b.horizon,
        "stride": b.stride,
        "confidence": b.confidence,
        "interval_method": b.interval_method,
        "auto": b.auto,
        "trend_kind": b.trend_kind,
        "seasonal_kind": b.seasonal_kind,
        "locks": b.locks,
        "result": b.result,
    }


@router.post("", response_model=JobOut)
def run(req: BacktestRequest):
    if not req.auto:
        if req.trend_kind not in TRENDS:
            raise HTTPException(400, f"trend_kind 必须是 {TRENDS} 之一")
        if req.seasonal_kind not in SEASONALS:
            raise HTTPException(400, f"seasonal_kind 必须是 {SEASONALS} 之一")
    if req.horizon < 1 or req.stride < 1:
        raise HTTPException(400, "horizon/stride 必须 >= 1")
    job = job_manager.submit(
        "backtest",
        lambda report: service.run_backtest_job(req.series_id, req, report),
        series_id=req.series_id,
    )
    return job.public()


@router.get("/by-series/{series_id}")
def list_for_series(series_id: int, db: Session = Depends(get_db)):
    return [_bt_out(b) for b in storage.list_backtests(db, series_id)]


@router.get("/{backtest_id}")
def detail(backtest_id: int, db: Session = Depends(get_db)):
    b = storage.get_backtest(db, backtest_id)
    if b is None:
        raise HTTPException(404, "回测结果不存在")
    return _bt_out(b)

from __future__ import annotations

from datetime import datetime
from typing import List

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from ..db import get_db
from ..schemas import SeriesCreate, SeriesOut
from ..series_utils import (
    SeriesValidationError,
    check_weekly_continuity,
    parse_csv,
)
from .. import storage

router = APIRouter(prefix="/api/series", tags=["series"])


def _to_out(s) -> SeriesOut:
    return SeriesOut(
        id=s.id,
        name=s.name,
        period=s.period,
        dates=list(s.dates),
        values=list(s.values),
        missing_dates=list(s.missing_dates or []),
        created_at=s.created_at.isoformat() if s.created_at else "",
    )


@router.get("", response_model=List[SeriesOut])
def list_all(db: Session = Depends(get_db)):
    return [_to_out(s) for s in storage.list_series(db)]


@router.get("/{series_id}", response_model=SeriesOut)
def detail(series_id: int, db: Session = Depends(get_db)):
    s = storage.get_series(db, series_id)
    if s is None:
        raise HTTPException(404, "序列不存在")
    return _to_out(s)


def _create_from_rows(db, name, dates_raw, values_raw, period):
    pairs = sorted(zip(dates_raw, values_raw), key=lambda p: p[0])
    dates = [p[0] for p in pairs]
    values = [float(p[1]) for p in pairs]
    missing = check_weekly_continuity(dates)
    obj = storage.create_series(
        db, name, dates, values, period=int(period), missing_dates=missing
    )
    return _to_out(obj)


@router.post("", response_model=SeriesOut)
def create(payload: SeriesCreate, db: Session = Depends(get_db)):
    try:
        dates = [datetime.fromisoformat(d).date() for d in payload.dates]
    except ValueError as exc:
        raise HTTPException(400, f"日期格式需为 YYYY-MM-DD：{exc}")
    if len(dates) != len(payload.values):
        raise HTTPException(400, "日期与销量数量不一致。")
    try:
        out = _create_from_rows(
            db, payload.name, dates, payload.values, payload.period
        )
        db.commit()
        return out
    except SeriesValidationError as exc:
        raise HTTPException(400, str(exc))


@router.post("/upload", response_model=SeriesOut)
async def upload(
    file: UploadFile = File(...),
    name: str = Form(""),
    period: int = Form(52),
    db: Session = Depends(get_db),
):
    raw = (await file.read()).decode("utf-8-sig")
    try:
        parsed = parse_csv(raw)
    except SeriesValidationError as exc:
        raise HTTPException(400, str(exc))
    series_name = name or file.filename or "上传序列"
    if series_name.lower().endswith(".csv"):
        series_name = series_name[:-4]
    try:
        out = _create_from_rows(
            db, series_name, parsed.dates, parsed.values, period
        )
        db.commit()
        return out
    except SeriesValidationError as exc:
        raise HTTPException(400, str(exc))

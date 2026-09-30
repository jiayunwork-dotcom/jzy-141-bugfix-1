"""Persistence helpers (repository layer) built over SQLAlchemy sessions."""
from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import Backtest, Fit, Series


def _iso(d: date) -> str:
    return d.isoformat() if isinstance(d, date) else str(d)


def create_series(
    db: Session,
    name: str,
    dates: List[date],
    values: List[float],
    period: int,
    missing_dates: Optional[List[date]] = None,
) -> Series:
    s = Series(
        name=name,
        period=period,
        dates=[_iso(d) for d in dates],
        values=[float(v) for v in values],
        missing_dates=[_iso(d) for d in (missing_dates or [])],
    )
    db.add(s)
    db.flush()
    return s


def list_series(db: Session) -> List[Series]:
    return list(db.scalars(select(Series).order_by(Series.created_at.desc())))


def get_series(db: Session, series_id: int) -> Optional[Series]:
    return db.get(Series, series_id)


def create_fit(db: Session, series_id: int, payload: Dict[str, Any]) -> Fit:
    fit = Fit(series_id=series_id, **payload)
    db.add(fit)
    db.flush()
    return fit


def list_fits(db: Session, series_id: int) -> List[Fit]:
    return list(
        db.scalars(
            select(Fit)
            .where(Fit.series_id == series_id)
            .order_by(Fit.created_at.desc())
        )
    )


def get_fit(db: Session, fit_id: int) -> Optional[Fit]:
    return db.get(Fit, fit_id)


def create_backtest(
    db: Session, series_id: int, payload: Dict[str, Any]
) -> Backtest:
    bt = Backtest(series_id=series_id, **payload)
    db.add(bt)
    db.flush()
    return bt


def list_backtests(db: Session, series_id: int) -> List[Backtest]:
    return list(
        db.scalars(
            select(Backtest)
            .where(Backtest.series_id == series_id)
            .order_by(Backtest.created_at.desc())
        )
    )


def get_backtest(db: Session, bt_id: int) -> Optional[Backtest]:
    return db.get(Backtest, bt_id)

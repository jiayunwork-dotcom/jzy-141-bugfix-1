"""Service layer: orchestrates kernels + storage for fits and backtests.

Stored series are sparse (uploaded dates only); the kernels work on the
*calendar grid*.  :func:`build_grid` rebuilds the continuous weekly grid,
placing NaN at every missing week.  Grid position t is exactly the number of
weeks after the series' first date, so it is identical in the full and the
gappy series -- backtest origins and forecast dates stay calendar aligned.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .db import session_scope
from .kernels import fit_hw, forecast, ModelError
from .kernels import selection
from .kernels.backtest import run_backtest
from . import storage

ENGINE_VERSION = "2.0-calendar-grid"
LEGACY_ENGINE_VERSION = "1.0-observed-array"


def build_grid(dates: List[str], values: List[float],
               missing_dates: List[str]) -> Tuple[List[str], np.ndarray]:
    """Rebuild the continuous weekly calendar grid.

    Returns ``(grid_dates, grid_values)`` where ``grid_values`` is NaN at
    every missing week.  The stored lists are already sorted and pass the
    7-day continuity check, so the missing dates slot exactly into the
    calendar spans.
    """
    if not dates:
        return [], np.zeros(0)

    def _d(s: str):
        return datetime.fromisoformat(s).date()

    first = _d(dates[0])
    observed = {_d(d): float(v) for d, v in zip(dates, values)}
    missing = {_d(d) for d in (missing_dates or [])}
    total_weeks = len(dates) + len(missing)
    grid_dates: List[str] = []
    grid_values = np.full(total_weeks, np.nan)
    for t in range(total_weeks):
        cur = first + timedelta(weeks=t)
        grid_dates.append(cur.isoformat())
        if cur in observed:
            grid_values[t] = observed[cur]
        elif cur not in missing:
            # Defensive: an unreported gap.  Still treated as missing.
            missing.add(cur)
    return grid_dates, grid_values


def _fit_payload(series, req, report) -> Dict[str, Any]:
    grid_dates, y = build_grid(
        list(series.dates), list(series.values), list(series.missing_dates)
    )
    m = int(series.period)
    locks = dict(req.locks or {})

    report(0.02, "准备数据（对齐日历网格）")
    if req.auto:
        report(0.05, "自动选型：优化各组合")
        sel = selection.auto_select(
            y, m, locks=locks,
            progress_cb=lambda p: report(0.05 + 0.55 * p, "自动选型中"),
        )
        fr = sel.best
        scores = [s.to_dict() for s in sel.scores]
    else:
        if not req.trend_kind or not req.seasonal_kind:
            raise ModelError("手动模式需要同时指定 trend_kind 和 seasonal_kind。")
        report(0.1, "拟合指定模型")
        fr = selection.fit_one(
            y, req.trend_kind, req.seasonal_kind, m, locks=locks
        )
        scores = []

    report(0.65, f"预测未来 {req.horizon} 步")
    fc = forecast(
        fr.final_state,
        int(req.horizon),
        fr.residuals,
        level=float(req.confidence),
        method=req.interval_method,
    )

    future_dates = _future_dates(grid_dates, int(req.horizon))

    report(0.9, "写入结果")
    with session_scope() as db:
        obj = storage.create_fit(db, series.id, {
            "label": req.label or "",
            "auto": bool(req.auto),
            "engine_version": ENGINE_VERSION,
            "trend_kind": fr.trend_kind,
            "seasonal_kind": fr.seasonal_kind,
            "period": m,
            "params": {
                "alpha": float(fr.params.alpha),
                "beta": float(fr.params.beta),
                "gamma": float(fr.params.gamma),
                "phi": float(fr.params.phi),
            },
            "locks": locks,
            "sse": float(fr.sse),
            "aic": float(fr.aic),
            "n_effective": int(fr.n_effective),
            "n_calendar": int(fr.n_calendar),
            "missing_count": int(fr.missing_count),
            "residuals": [float(v) for v in fr.residuals],
            "fitted": [None if not np.isfinite(v) else float(v)
                       for v in fr.all_fitted],
            "grid_dates": grid_dates,
            "forecast": {
                **fc.to_dict(),
                "horizon": int(req.horizon),
                "future_dates": future_dates,
            },
            "initial_state": {
                "level": fr.initial_level,
                "trend": fr.initial_trend,
                "season": [float(v) for v in fr.initial_season],
            },
            "final_state": fr.final_state.to_dict(),
            "scores": scores,
        })
        fit_id = obj.id
    return {"created_id": fit_id, "result": {"fit_id": fit_id}}


def _future_dates(grid_dates, horizon):
    if not grid_dates:
        return []
    last = datetime.fromisoformat(grid_dates[-1]).date()
    return [(last + timedelta(weeks=k)).isoformat()
            for k in range(1, horizon + 1)]


def run_fit_job(series_id: int, req, report) -> Dict[str, Any]:
    with session_scope() as db:
        series = storage.get_series(db, series_id)
        if series is None:
            raise ModelError(f"序列 {series_id} 不存在。")
        # detach snapshot
        series_values = list(series.values)
        series_dates = list(series.dates)
        series_missing = list(series.missing_dates or [])
        period = series.period

    class _S:
        pass
    snap = _S()
    snap.id = series_id
    snap.values = series_values
    snap.dates = series_dates
    snap.missing_dates = series_missing
    snap.period = period
    return _fit_payload(snap, req, report)


def run_backtest_job(series_id: int, req, report) -> Dict[str, Any]:
    with session_scope() as db:
        series = storage.get_series(db, series_id)
        if series is None:
            raise ModelError(f"序列 {series_id} 不存在。")
        dates = list(series.dates)
        values = list(series.values)
        missing_dates = list(series.missing_dates or [])
        period = series.period

    grid_dates, y = build_grid(dates, values, missing_dates)
    report(0.02, "滚动原点回测（日历网格）")
    bt = run_backtest(
        y,
        period=int(period),
        origin_start=int(req.origin_start),
        horizon=int(req.horizon),
        stride=int(req.stride),
        trend_kind=req.trend_kind if not req.auto else None,
        seasonal_kind=req.seasonal_kind if not req.auto else None,
        locks=dict(req.locks or {}),
        confidence=float(req.confidence),
        interval_method=req.interval_method,
        progress_cb=lambda p: report(0.05 + 0.9 * p, f"回测原点 {p:.0%}"),
    )
    result = bt.to_dict()
    result["grid_dates"] = grid_dates
    report(0.97, "保存回测结果")
    with session_scope() as db:
        obj = storage.create_backtest(db, series_id, {
            "label": req.label or "",
            "engine_version": ENGINE_VERSION,
            "origin_start": int(req.origin_start),
            "horizon": int(req.horizon),
            "stride": int(req.stride),
            "confidence": float(req.confidence),
            "interval_method": req.interval_method,
            "auto": bool(req.auto),
            "trend_kind": req.trend_kind,
            "seasonal_kind": req.seasonal_kind,
            "locks": dict(req.locks or {}),
            "result": result,
        })
        bt_id = obj.id
    return {"created_id": bt_id, "result": {"backtest_id": bt_id}}

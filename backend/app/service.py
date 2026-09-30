"""Service layer: orchestrates kernels + storage for fits and backtests."""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .db import session_scope
from .kernels import fit_hw, forecast, ModelError
from .kernels import selection
from .kernels.backtest import run_backtest
from . import storage


def _fit_payload(series, req, report) -> Dict[str, Any]:
    y = np.asarray(series.values, dtype=float)
    m = int(series.period)
    locks = dict(req.locks or {})

    report(0.02, "准备数据")
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

    report(0.9, "写入结果")
    with session_scope() as db:
        obj = storage.create_fit(db, series.id, {
            "label": req.label or "",
            "auto": bool(req.auto),
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
            "residuals": [float(v) for v in fr.residuals],
            "fitted": [float(v) for v in fr.all_fitted],
            "forecast": {
                **fc.to_dict(),
                "horizon": int(req.horizon),
                "future_dates": _future_dates(series.dates, req.horizon),
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


def _future_dates(dates, horizon):
    from datetime import datetime, timedelta
    if not dates:
        return []
    last = datetime.fromisoformat(dates[-1]).date()
    return [(last + timedelta(weeks=k)).isoformat()
            for k in range(1, horizon + 1)]


def run_fit_job(series_id: int, req, report) -> Dict[str, Any]:
    with session_scope() as db:
        series = storage.get_series(db, series_id)
        if series is None:
            raise ModelError(f"序列 {series_id} 不存在。")
        # detach values snapshot
        series_values = list(series.values)
        series_dates = list(series.dates)
        period = series.period

    class _S:
        pass
    snap = _S()
    snap.id = series_id
    snap.values = series_values
    snap.dates = series_dates
    snap.period = period
    return _fit_payload(snap, req, report)


def run_backtest_job(series_id: int, req, report) -> Dict[str, Any]:
    with session_scope() as db:
        series = storage.get_series(db, series_id)
        if series is None:
            raise ModelError(f"序列 {series_id} 不存在。")
        values = list(series.values)
        period = series.period

    y = np.asarray(values, dtype=float)
    report(0.02, "滚动原点回测")
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
    report(0.97, "保存回测结果")
    with session_scope() as db:
        obj = storage.create_backtest(db, series_id, {
            "label": req.label or "",
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

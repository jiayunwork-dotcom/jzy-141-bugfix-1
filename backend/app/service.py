"""Service layer: orchestrates kernels + storage for fits and backtests.

Sparse uploaded rows (only observed weeks) are expanded onto a dense weekly
grid before any modelling (see ``app.gaps``): gaps carry NaN and a False
mask.  The kernels propagate the state across gaps without a data update,
so every stored fit/backtest is comparable at a *calendar date*.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

import numpy as np

from .db import session_scope
from .gaps import build_dense_grid, future_grid_dates
from .kernels import fit_hw, forecast, ModelError
from .kernels import selection
from .kernels.backtest import run_backtest
from . import storage

# Results produced by the gap-aware dense-grid pipeline carry this marker.
# Rows without it in the database predate missing-week support and must be
# shown as legacy (their indices were positions in the compressed list).
RESULTS_VERSION = 2
LEGACY_REASON = (
    "旧版结果：在缺周序列上按压缩下标建模，季节相位/原点日期可能错位，"
    "不与当前缺周感知结果可比，请重新拟合/回测。"
)


def _load_dense(series):
    dates = [datetime.fromisoformat(d).date() for d in series.dates]
    grid_dates, grid_values, mask = build_dense_grid(dates, series.values)
    return grid_dates, grid_values, mask


def _fit_payload(series, req, report) -> Dict[str, Any]:
    grid_dates, y, mask = _load_dense(series)
    m = int(series.period)
    locks = dict(req.locks or {})

    report(0.02, "准备数据（稠密周网格）")
    if req.auto:
        report(0.05, "自动选型：优化各组合")
        sel = selection.auto_select(
            y, m, locks=locks, mask=mask,
            progress_cb=lambda p: report(0.05 + 0.55 * p, "自动选型中"),
        )
        fr = sel.best
        scores = [s.to_dict() for s in sel.scores]
    else:
        if not req.trend_kind or not req.seasonal_kind:
            raise ModelError("手动模式需要同时指定 trend_kind 和 seasonal_kind。")
        report(0.1, "拟合指定模型")
        fr = selection.fit_one(
            y, req.trend_kind, req.seasonal_kind, m, locks=locks, mask=mask
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
    future_dates = [
        d.isoformat() for d in future_grid_dates(grid_dates[-1],
                                                 int(req.horizon))
    ]
    missing_idx = [int(i) for i in np.flatnonzero(~mask)]

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
            "n_effective": int(fr.n_effective),
            "residuals": [float(v) for v in fr.residuals],
            "fitted": [
                None if not np.isfinite(v) else float(v)
                for v in fr.all_fitted
            ],
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
            "grid_dates": [d.isoformat() for d in grid_dates],
            "missing_indices": missing_idx,
            "results_version": RESULTS_VERSION,
        })
        fit_id = obj.id
    return {"created_id": fit_id, "result": {"fit_id": fit_id}}


def run_fit_job(series_id: int, req, report) -> Dict[str, Any]:
    with session_scope() as db:
        series = storage.get_series(db, series_id)
        if series is None:
            raise ModelError(f"序列 {series_id} 不存在。")
        # detach snapshot
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
        dates = list(series.dates)
        period = series.period

    class _S:
        pass
    snap = _S()
    snap.values = values
    snap.dates = dates
    grid_dates, y, mask = _load_dense(snap)
    grid_iso = [d.isoformat() for d in grid_dates]
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
        mask=mask,
        grid_dates=grid_iso,
    )
    result = bt.to_dict()
    result["results_version"] = RESULTS_VERSION
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

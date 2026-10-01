"""Rolling-origin backtest for Holt-Winters vs seasonal naive.

Everything is indexed on the *calendar grid*: position k means the k-th week
of the grid (k weeks after the first uploaded date), even if some weeks are
missing (NaN).  This is what makes the seasonal-naive benchmark and the
seasonal indices of Holt-Winters land on the right calendar week.

At origin k only the grid prefix ``y[:k]`` is used.  The model is refitted on
that prefix and forecasts 1..h steps ahead.  The seasonal-naive benchmark
forecasts ``yhat_{k+j} = y_{k+j-m}`` -- the value exactly one calendar year
earlier; if that reference week is missing the benchmark has no forecast for
the step and the step is excluded from naive metrics.  Steps whose *actual*
week is missing are excluded from every metric (there is nothing to score).

Metrics
~~~~~~~
* MAE  = mean |e|
* MAPE = mean |e / y|  (only over non-zero actuals; zero actuals are skipped)
* MASE = mean |e| / Q, where Q is the in-sample seasonal-naive MAE of the
         *training prefix*, Q = mean over pairs t,t-m that are both observed
         of |y_t - y_{t-m}| (Hyndman-Koehler scale, recomputed per origin).

All quantities are computed by the backend; the frontend only renders them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .hw import ModelError, TrendKind, SeasonalKind, fit_hw, forecast
from . import optimizer
from .selection import auto_select


@dataclass
class OriginResult:
    origin: int
    train_size: int
    forecast: List[Optional[float]]
    actual: List[Optional[float]]
    naive_forecast: List[Optional[float]]
    errors: List[Optional[float]]
    naive_errors: List[Optional[float]]
    mae: float
    mape: float
    mase: float
    naive_mae: float
    naive_mape: float
    naive_mase: float
    params: Optional[dict]
    aic: Optional[float]
    scale: float
    scored_steps: int = 0       # horizon weeks with an observed actual
    skipped: bool = False       # True if the origin produced no scorable step

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class BacktestResult:
    period: int
    horizon: int
    origins: List[OriginResult]
    model: Dict[str, object]          # aggregate for HW
    naive: Dict[str, object]          # aggregate for seasonal naive
    model_kind: Dict[str, str] = field(default_factory=dict)
    skipped_origins: List[int] = field(default_factory=list)
    calendar_size: int = 0

    def to_dict(self) -> dict:
        return {
            "period": self.period,
            "horizon": self.horizon,
            "origins": [o.to_dict() for o in self.origins],
            "model": self.model,
            "naive": self.naive,
            "model_kind": self.model_kind,
            "skipped_origins": list(self.skipped_origins),
            "calendar_size": self.calendar_size,
        }


def _naive_mae_scale(train: np.ndarray, m: int) -> float:
    """Seasonal-naive in-sample MAE over calendar-aligned observed pairs."""
    if train.size <= m:
        return np.nan
    cur = train[m:]
    ref = train[:-m]
    both = np.isfinite(cur) & np.isfinite(ref)
    if not np.any(both):
        return np.nan
    return float(np.mean(np.abs(cur[both] - ref[both])))


def _agg(errors: np.ndarray, actuals: np.ndarray,
         scales: np.ndarray) -> Dict[str, float]:
    mae = float(np.mean(np.abs(errors)))
    nonzero = actuals != 0
    mape = (float(np.mean(np.abs(errors[nonzero] / actuals[nonzero])))
            if np.any(nonzero) else float("nan"))
    finite = np.isfinite(scales)
    mase = (float(np.mean(np.abs(errors[finite]) / scales[finite]))
            if np.any(finite) else float("nan"))
    return {"mae": mae, "mape": mape, "mase": mase}


def run_backtest(
    y: np.ndarray,
    period: int,
    origin_start: int,
    horizon: int,
    stride: int = 1,
    trend_kind: Optional[TrendKind] = None,
    seasonal_kind: Optional[SeasonalKind] = None,
    locks: Optional[Dict[str, float]] = None,
    confidence: float = 0.95,
    interval_method: str = "analytic",
    progress_cb=None,
) -> BacktestResult:
    """Rolling-origin evaluation on a calendar-aligned grid (NaN = missing)."""
    y = np.asarray(y, dtype=float)
    m = period
    n = y.size
    observed = np.isfinite(y)
    if not np.all(np.isfinite(y) | np.isnan(y)):
        raise ModelError("序列包含无穷大，无法回测。")
    if n < 2 * m + horizon:
        raise ModelError(
            f"回测至少需要跨越 2 个季节 + {horizon} 个日历周。"
        )
    if origin_start < 2 * m:
        raise ModelError(f"首个原点必须 >= 2 个季节（{2 * m}）。")
    if origin_start + horizon > n:
        raise ModelError("最后一个原点之后没有足够的日历周用于比较。")
    if stride < 1:
        raise ModelError("步长必须 >= 1。")

    origins_idx = list(range(origin_start, n - horizon + 1, stride))
    rows: List[OriginResult] = []
    skipped: List[int] = []
    chosen_kinds = {"trend_kind": "", "seasonal_kind": ""}

    for step, k in enumerate(origins_idx):
        train = y[:k].copy()
        actual_raw = y[k : k + horizon].copy()
        actual_mask = np.isfinite(actual_raw)
        if not np.any(actual_mask):
            # Every hold-out week is missing: nothing to score at this origin.
            skipped.append(int(k))
            continue
        # Fitting requires at least one observed week beyond position 2m.
        if train.size <= 2 * m or not np.any(np.isfinite(train[2 * m:])):
            skipped.append(int(k))
            continue

        if trend_kind is None or seasonal_kind is None:
            sel = auto_select(train, m, locks=locks)
            fr = sel.best
        else:
            params, _, _ = optimizer.optimize(
                train, trend_kind, seasonal_kind, m, locks=locks
            )
            fr = fit_hw(train, trend_kind, seasonal_kind, m, params)
        chosen_kinds = {
            "trend_kind": fr.trend_kind,
            "seasonal_kind": fr.seasonal_kind,
        }
        fc = forecast(
            fr.final_state,
            horizon,
            fr.residuals,
            level=confidence,
            method=interval_method,
        )
        pred = np.asarray(fc.point, dtype=float)

        # Seasonal naive on the calendar grid; NaN if reference is missing.
        naive = np.full(horizon, np.nan)
        for j in range(horizon):
            ref_idx = k + j - m
            if ref_idx >= 0 and np.isfinite(y[ref_idx]):
                naive[j] = y[ref_idx]

        # Errors exist only where the actual week was observed.  Naive errors
        # additionally require the reference week to be observed.
        errors = np.full(horizon, np.nan)
        naive_errors = np.full(horizon, np.nan)
        errors[actual_mask] = pred[actual_mask] - actual_raw[actual_mask]
        naive_ok = actual_mask & np.isfinite(naive)
        naive_errors[naive_ok] = naive[naive_ok] - actual_raw[naive_ok]

        q = _naive_mae_scale(train, m)
        q_eff = q if (q and np.isfinite(q) and q > 0) else np.nan

        def block_stats(err: np.ndarray) -> tuple[float, float, float]:
            ok = np.isfinite(err)
            if not np.any(ok):
                return float("nan"), float("nan"), float("nan")
            e = err[ok]
            act = actual_raw[ok]
            mae_b = float(np.mean(np.abs(e)))
            nz = act != 0
            mape_b = (float(np.mean(np.abs(e[nz] / act[nz])))
                      if np.any(nz) else float("nan"))
            mase_b = (float(np.mean(np.abs(e) / q))
                      if np.isfinite(q_eff) else float("nan"))
            return mae_b, mape_b, mase_b

        m_mae, m_mape, m_mase = block_stats(errors)
        n_mae, n_mape, n_mase = block_stats(naive_errors)

        def _lst(a):
            return [None if (v is None or (isinstance(v, float) and
                    not np.isfinite(v))) else float(v) for v in a]

        rows.append(
            OriginResult(
                origin=int(k),
                train_size=int(k),
                forecast=_lst(pred),
                actual=_lst(actual_raw),
                naive_forecast=_lst(naive),
                errors=_lst(errors),
                naive_errors=_lst(naive_errors),
                mae=m_mae,
                mape=m_mape,
                mase=m_mase,
                naive_mae=n_mae,
                naive_mape=n_mape,
                naive_mase=n_mase,
                params={
                    "alpha": fr.params.alpha,
                    "beta": fr.params.beta,
                    "gamma": fr.params.gamma,
                    "phi": fr.params.phi,
                },
                aic=float(fr.aic),
                scale=float(q) if np.isfinite(q) else float("nan"),
                scored_steps=int(actual_mask.sum()),
            )
        )
        if progress_cb is not None:
            progress_cb((step + 1) / len(origins_idx))

    if not rows:
        raise ModelError(
            "所有回测原点的留出周或训练段都是缺周，没有可评分的步骤。"
        )

    def _flat(attr: str) -> np.ndarray:
        out = []
        for r in rows:
            vals = getattr(r, attr)
            for v in vals:
                if v is not None:
                    out.append(v)
        return np.asarray(out, dtype=float)

    all_err = _flat("errors")
    all_naive_err = _flat("naive_errors")

    # Model and naive errors may cover different steps (naive also needs the
    # year-ago week), so actuals and scales are gathered per error list.
    model_act = np.asarray(
        [a for r in rows for e, a in zip(r.errors, r.actual) if e is not None]
    )
    model_scales = np.asarray(
        [r.scale for r in rows for e in r.errors if e is not None]
    )
    naive_act = np.asarray(
        [a for r in rows
         for e, a in zip(r.naive_errors, r.actual) if e is not None]
    )
    naive_scales = np.asarray(
        [r.scale for r in rows for e in r.naive_errors if e is not None]
    )
    model_agg = _agg(all_err, model_act, model_scales)
    naive_agg = _agg(all_naive_err, naive_act, naive_scales)
    return BacktestResult(
        period=m,
        horizon=horizon,
        origins=rows,
        model=model_agg,
        naive=naive_agg,
        model_kind=chosen_kinds,
        skipped_origins=skipped,
        calendar_size=int(n),
    )

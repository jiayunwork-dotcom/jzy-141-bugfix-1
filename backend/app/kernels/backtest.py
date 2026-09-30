"""Rolling-origin backtest for Holt-Winters vs seasonal naive.

At origin k (a series index), only ``y[:k]`` is used.  The model is refitted
on that prefix, forecasts 1..h steps ahead, and errors are compared with the
held-out actuals.  The seasonal-naive benchmark forecasts
``yhat_{k+j} = y_{k+j-m}``.

Metrics
~~~~~~~
* MAE  = mean |e|
* MAPE = mean |e / y|  (only over non-zero actuals; zero actuals are skipped)
* MASE = mean |e| / Q, where Q is the in-sample seasonal-naive MAE of the
         *training prefix*, Q = mean_{t=m..k-1} |y_t - y_{t-m}|
         (Hyndman & Koehler scale, recomputed per origin).

All quantities are computed by the backend; the frontend only renders them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .hw import ModelError, TrendKind, SeasonalKind, fit_hw, forecast
from . import optimizer
from .selection import GRID, auto_select


@dataclass
class OriginResult:
    origin: int
    train_size: int
    forecast: List[float]
    actual: List[float]
    naive_forecast: List[float]
    errors: List[float]
    naive_errors: List[float]
    mae: float
    mape: float
    mase: float
    naive_mae: float
    naive_mape: float
    naive_mase: float
    params: Optional[dict]
    aic: Optional[float]
    scale: float

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

    def to_dict(self) -> dict:
        return {
            "period": self.period,
            "horizon": self.horizon,
            "origins": [o.to_dict() for o in self.origins],
            "model": self.model,
            "naive": self.naive,
            "model_kind": self.model_kind,
        }


def _naive_mae_scale(train: np.ndarray, m: int) -> float:
    if train.size <= m:
        return np.nan
    diffs = np.abs(train[m:] - train[:-m])
    return float(np.mean(diffs))


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
    y = np.asarray(y, dtype=float)
    m = period
    n = y.size
    if y.size < 2 * m + horizon:
        raise ModelError(
            f"回测至少需要 2 个季节 + {horizon} 个留出观测。"
        )
    if origin_start < 2 * m:
        raise ModelError(f"首个原点必须 >= 2 个季节（{2 * m}）。")
    if origin_start + horizon > n:
        raise ModelError("最后一个原点之后没有足够的实际值用于比较。")
    if stride < 1:
        raise ModelError("步长必须 >= 1。")

    origins_idx = list(range(origin_start, n - horizon + 1, stride))
    rows: List[OriginResult] = []
    chosen_kinds = {"trend_kind": "", "seasonal_kind": ""}

    for step, k in enumerate(origins_idx):
        train = y[:k]
        actual = y[k : k + horizon]

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
        errors = pred - actual

        naive = np.array(
            [y[k + j - m] if k + j - m >= 0 else np.nan
             for j in range(horizon)]
        )
        naive_errors = naive - actual

        q = _naive_mae_scale(train, m)
        q_eff = q if (q and np.isfinite(q) and q > 0) else np.nan
        scales = np.full(horizon, q_eff if np.isfinite(q_eff) else np.nan)

        def block_stats(err: np.ndarray) -> tuple[float, float, float]:
            mae_b = float(np.mean(np.abs(err)))
            nz = actual != 0
            mape_b = (float(np.mean(np.abs(err[nz] / actual[nz])))
                      if np.any(nz) else float("nan"))
            mase_b = (float(np.mean(np.abs(err) / q))
                      if np.isfinite(q_eff) else float("nan"))
            return mae_b, mape_b, mase_b

        m_mae, m_mape, m_mase = block_stats(errors)
        n_mae, n_mape, n_mase = block_stats(naive_errors)

        rows.append(
            OriginResult(
                origin=int(k),
                train_size=int(k),
                forecast=[float(v) for v in pred],
                actual=[float(v) for v in actual],
                naive_forecast=[float(v) for v in naive],
                errors=[float(v) for v in errors],
                naive_errors=[float(v) for v in naive_errors],
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
            )
        )
        if progress_cb is not None:
            progress_cb((step + 1) / len(origins_idx))

    all_err = np.array([e for r in rows for e in r.errors])
    all_act = np.array([a for r in rows for a in r.actual])
    all_naive_err = np.array([e for r in rows for e in r.naive_errors])
    scales = np.array(
        [r.scale for r in rows for _ in range(horizon)]
    )

    model_agg = _agg(all_err, all_act, scales)
    naive_agg = _agg(all_naive_err, all_act, scales)
    return BacktestResult(
        period=m,
        horizon=horizon,
        origins=rows,
        model=model_agg,
        naive=naive_agg,
        model_kind=chosen_kinds,
    )

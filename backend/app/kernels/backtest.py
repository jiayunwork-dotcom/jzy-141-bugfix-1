"""Rolling-origin backtest for Holt-Winters vs seasonal naive.

All indices are positions on the dense weekly calendar grid (see
``app.gaps``), never positions in the compressed list of observed rows:
that distinction is exactly what makes seasonal-naive comparisons wrong
when a series has missing weeks.

At origin k (a grid index), only grid positions ``< k`` are used.  The model
is refitted on that prefix (missing weeks inside the prefix are propagated
without state updates), forecasts 1..h steps ahead, and errors are compared
with the held-out actuals.  The seasonal-naive benchmark forecasts
``yhat_{k+j} = y_{k+j-m}`` at the calendar position one season earlier; if
that week itself is missing the naive forecast is undefined and is excluded
from the naive metrics rather than replaced by a guess.

Metrics
~~~~~~~
* MAE  = mean |e| over defined forecasts
* MAPE = mean |e / y|  (only over non-zero actuals; zero actuals are skipped)
* MASE = mean |e| / Q, where Q is the in-sample seasonal-naive MAE of the
         *training prefix* averaged over calendar-season pairs that are
         observed at both ends,
         Q = mean |y_t - y_{t-m}| over observed t >= m in y[:k]
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
    origin_date: Optional[str] = None
    skipped_naive: int = 0  # horizons without a year-ago observation

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
    grid_dates: Optional[List[str]] = None
    skipped_origins: List[Dict[str, object]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "period": self.period,
            "horizon": self.horizon,
            "origins": [o.to_dict() for o in self.origins],
            "model": self.model,
            "naive": self.naive,
            "model_kind": self.model_kind,
            "grid_dates": self.grid_dates,
            "skipped_origins": self.skipped_origins,
        }


def _naive_mae_scale(train: np.ndarray, mask: np.ndarray, m: int) -> float:
    """Mean |y_t - y_{t-m}| over calendar pairs observed at both ends."""
    if train.size <= m:
        return np.nan
    cur = train[m:]
    lag = train[:-m]
    pair_ok = mask[m:] & mask[:-m]
    if not np.any(pair_ok):
        return np.nan
    return float(np.mean(np.abs(cur[pair_ok] - lag[pair_ok])))


def _nanmean_abs(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else float("nan")


def _block_stats(err: np.ndarray, actual: np.ndarray,
                 q: float) -> tuple[float, float, float]:
    finite = np.isfinite(err)
    mae_b = _nanmean_abs(err)
    nz = finite & (actual != 0)
    mape_b = (float(np.mean(np.abs(err[nz] / actual[nz])))
              if np.any(nz) else float("nan"))
    mase_b = (float(np.mean(np.abs(err[finite]) / q))
              if np.isfinite(q) and np.any(finite) else float("nan"))
    return mae_b, mape_b, mase_b


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
    mask: Optional[np.ndarray] = None,
    grid_dates: Optional[List[str]] = None,
) -> BacktestResult:
    y = np.asarray(y, dtype=float)
    m = period
    n = y.size
    if mask is None:
        mask = np.ones(n, dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)
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
    if not np.all(mask[: 2 * m]):
        raise ModelError(
            f"前 {2 * m} 周（两个完整季节）中存在缺周，无法按既定初始规则"
            "初始化回测：请补齐早期数据，或使用缺口之后已积累两个完整"
            "季节的序列段。"
        )

    origins_idx = list(range(origin_start, n - horizon + 1, stride))
    rows: List[OriginResult] = []
    skipped: List[Dict[str, object]] = []
    chosen_kinds = {"trend_kind": "", "seasonal_kind": ""}

    for step, k in enumerate(origins_idx):
        # The origin week itself and every held-out week must be observed:
        # an "actual" that does not exist cannot be scored.
        holdout_ok = mask[k:k + horizon]
        if not mask[k] or not np.all(holdout_ok):
            missing = [
                (grid_dates[k + j] if grid_dates is not None else k + j)
                for j in range(horizon)
                if not mask[k + j]
            ]
            if not mask[k] and grid_dates is not None:
                missing = [grid_dates[k]] + missing
            skipped.append({
                "origin": int(k),
                "origin_date": grid_dates[k] if grid_dates is not None
                else None,
                "reason": "原点或留出窗口内存在缺周，无实际值可评分。",
                "missing": missing,
            })
            if progress_cb is not None:
                progress_cb((step + 1) / len(origins_idx))
            continue

        train = y[:k]
        train_mask = mask[:k]
        actual = y[k:k + horizon]

        if trend_kind is None or seasonal_kind is None:
            sel = auto_select(train, m, locks=locks, mask=train_mask)
            fr = sel.best
        else:
            params, _, _ = optimizer.optimize(
                train, trend_kind, seasonal_kind, m,
                locks=locks, mask=train_mask,
            )
            fr = fit_hw(train, trend_kind, seasonal_kind, m, params,
                        mask=train_mask)
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

        # Seasonal naive on the dense calendar: the same week one season
        # earlier; undefined (NaN) when that week was itself missing.
        naive = np.full(horizon, np.nan)
        for j in range(horizon):
            lag_idx = k + j - m
            if lag_idx >= 0 and mask[lag_idx]:
                naive[j] = y[lag_idx]
        with np.errstate(invalid="ignore"):
            naive_errors = naive - actual
        skipped_naive = int(np.sum(~np.isfinite(naive)))

        q = _naive_mae_scale(train, train_mask, m)
        q_eff = q if (q and np.isfinite(q) and q > 0) else np.nan

        m_mae, m_mape, m_mase = _block_stats(errors, actual, q_eff)
        n_mae, n_mape, n_mase = _block_stats(naive_errors, actual, q_eff)

        rows.append(
            OriginResult(
                origin=int(k),
                train_size=int(train_mask.sum()),
                forecast=[float(v) for v in pred],
                actual=[float(v) for v in actual],
                naive_forecast=[
                    None if not np.isfinite(v) else float(v) for v in naive
                ],
                errors=[float(v) for v in errors],
                naive_errors=[
                    None if not np.isfinite(v) else float(v)
                    for v in naive_errors
                ],
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
                origin_date=grid_dates[k] if grid_dates is not None else None,
                skipped_naive=skipped_naive,
            )
        )
        if progress_cb is not None:
            progress_cb((step + 1) / len(origins_idx))

    if not rows:
        raise ModelError(
            "没有任何可用原点：所有候选原点的原点周或留出窗口都落在缺周上。"
        )

    all_err = np.array([e for r in rows for e in r.errors], dtype=float)
    all_act = np.array([a for r in rows for a in r.actual], dtype=float)
    all_naive_err = np.array(
        [e for r in rows for e in r.naive_errors], dtype=float
    )
    scales = np.array(
        [r.scale for r in rows for _ in range(horizon)], dtype=float
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
        grid_dates=list(grid_dates) if grid_dates is not None else None,
        skipped_origins=skipped,
    )


def _agg(errors: np.ndarray, actuals: np.ndarray,
         scales: np.ndarray) -> Dict[str, float]:
    finite = np.isfinite(errors)
    mae = float(np.mean(np.abs(errors[finite]))) if np.any(finite) \
        else float("nan")
    nonzero = finite & (actuals != 0)
    mape = (float(np.mean(np.abs(errors[nonzero] / actuals[nonzero])))
            if np.any(nonzero) else float("nan"))
    scale_ok = finite & np.isfinite(scales) & (scales > 0)
    mase = (float(np.mean(np.abs(errors[scale_ok]) / scales[scale_ok]))
            if np.any(scale_ok) else float("nan"))
    return {"mae": mae, "mape": mape, "mase": mase}

"""Automatic Holt-Winters model selection over the candidate grid.

Candidates combine trend {none, add, add_damped} with seasonal {add, mul}:

    (none, add) (none, mul)
    (add,  add) (add,  mul)
    (add_damped, add) (add_damped, mul)

Each feasible combination is fitted with optimised parameters and ranked by
AIC.  Multiplicative combinations are *rejected with a recorded reason* for
non-positive series instead of silently dropped, so the API response keeps
the full scorecard.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .hw import (
    FitResult,
    HWParams,
    ModelError,
    SeasonalKind,
    TrendKind,
    fit_hw,
    forecast,
)
from . import optimizer

GRID: List[tuple[TrendKind, SeasonalKind]] = [
    ("none", "add"),
    ("none", "mul"),
    ("add", "add"),
    ("add", "mul"),
    ("add_damped", "add"),
    ("add_damped", "mul"),
]


@dataclass
class CandidateScore:
    trend_kind: TrendKind
    seasonal_kind: SeasonalKind
    feasible: bool
    aic: Optional[float] = None
    sse: Optional[float] = None
    params: Optional[HWParams] = None
    reason: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "trend_kind": self.trend_kind,
            "seasonal_kind": self.seasonal_kind,
            "feasible": self.feasible,
            "aic": None if self.aic is None else float(self.aic),
            "sse": None if self.sse is None else float(self.sse),
            "params": None
            if self.params is None
            else {
                "alpha": self.params.alpha,
                "beta": self.params.beta,
                "gamma": self.params.gamma,
                "phi": self.params.phi,
            },
            "reason": self.reason,
        }


@dataclass
class SelectionResult:
    best: FitResult
    scores: List[CandidateScore]


def fit_one(
    y: np.ndarray,
    trend_kind: TrendKind,
    seasonal_kind: SeasonalKind,
    period: int,
    locks: Optional[Dict[str, float]] = None,
) -> FitResult:
    """Fit a specific combination (used by the manual panel too)."""
    if locks:
        params, sse, aic = optimizer.optimize(
            y, trend_kind, seasonal_kind, period, locks=locks
        )
        return fit_hw(y, trend_kind, seasonal_kind, period, params)
    params, _, _ = optimizer.optimize(
        y, trend_kind, seasonal_kind, period
    )
    return fit_hw(y, trend_kind, seasonal_kind, period, params)


def auto_select(
    y: np.ndarray,
    period: int,
    locks: Optional[Dict[str, float]] = None,
    progress_cb=None,
) -> SelectionResult:
    y = np.asarray(y, dtype=float)
    scores: List[CandidateScore] = []
    fits: List[FitResult] = []
    for i, (tk, sk) in enumerate(GRID):
        try:
            params, sse, aic = optimizer.optimize(
                y, tk, sk, period, locks=locks
            )
            fr = fit_hw(y, tk, sk, period, params)
        except ModelError as exc:
            scores.append(
                CandidateScore(tk, sk, feasible=False, reason=str(exc))
            )
        except Exception as exc:  # pragma: no cover - numerical safety net
            scores.append(
                CandidateScore(tk, sk, feasible=False, reason=f"数值错误：{exc}")
            )
        else:
            scores.append(
                CandidateScore(tk, sk, True, aic=fr.aic, sse=fr.sse,
                               params=params)
            )
            fits.append(fr)
        if progress_cb is not None:
            progress_cb((i + 1) / len(GRID))

    if not fits:
        raise ModelError("没有任何候选组合可用于该序列。")
    best = min(fits, key=lambda f: f.aic)
    return SelectionResult(best=best, scores=scores)

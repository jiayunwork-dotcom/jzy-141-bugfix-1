"""Bounded, deterministic smoothing-parameter optimiser.

We minimise the in-sample one-step-ahead SSE over

    alpha, gamma in [0.001, 0.999]
    beta         in [0.001, alpha]   (trend models only)
    phi          in [0.80, 0.98]     (damped trend only)

(The beta <= alpha restriction is the standard admissibility convention.)

Method: multi-start Nelder-Mead in an unconstrained logit-transformed space,
implemented here from scratch (no scipy).  The starting simplex grid is fixed,
so repeated calls with the same input always produce the same answer.  Any
parameter can be *locked*: locked parameters stay at the supplied value and
are removed from the optimisation vector.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np

from .hw import (
    HWParams,
    ModelError,
    TrendKind,
    SeasonalKind,
    fit_hw,
    validate_series,
)

ALPHA_MIN, ALPHA_MAX = 0.001, 0.999
GAMMA_MIN, GAMMA_MAX = 0.001, 0.999
BETA_MIN = 0.001
PHI_MIN, PHI_MAX = 0.80, 0.98


@dataclass(frozen=True)
class ParamSpec:
    name: str
    low: float
    high: float


def _specs(trend_kind: TrendKind) -> list[ParamSpec]:
    specs = [
        ParamSpec("alpha", ALPHA_MIN, ALPHA_MAX),
        ParamSpec("gamma", GAMMA_MIN, GAMMA_MAX),
    ]
    if trend_kind != "none":
        specs.insert(1, ParamSpec("beta", BETA_MIN, ALPHA_MAX))
    if trend_kind == "add_damped":
        specs.append(ParamSpec("phi", PHI_MIN, PHI_MAX))
    return specs


def _logit(u: np.ndarray) -> np.ndarray:
    return np.log(u / (1.0 - u))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))


def _to_unit(z: np.ndarray, spec: ParamSpec) -> np.ndarray:
    return spec.low + (spec.high - spec.low) * _sigmoid(z)


def _from_unit(u: float, spec: ParamSpec) -> float:
    u = min(max(u, 1e-9), 1 - 1e-9)
    return float(np.log(u / (1 - u)))


def _starts(d: int) -> list[np.ndarray]:
    """Deterministic multi-start grid in unit cube (corners + midpoint etc.)."""
    if d == 2:
        pts = [
            [0.2, 0.2], [0.8, 0.2], [0.2, 0.8], [0.8, 0.8],
            [0.5, 0.5], [0.1, 0.5], [0.5, 0.1], [0.9, 0.9],
        ]
    elif d == 3:
        pts = [
            [0.3, 0.15, 0.3], [0.6, 0.3, 0.6], [0.2, 0.1, 0.2],
            [0.5, 0.5, 0.5], [0.8, 0.4, 0.2], [0.15, 0.05, 0.8],
        ]
    else:  # 4
        pts = [
            [0.3, 0.15, 0.3, 0.95], [0.6, 0.3, 0.6, 0.90],
            [0.2, 0.1, 0.2, 0.98], [0.5, 0.5, 0.5, 0.85],
        ]
    return [np.asarray(p, dtype=float) for p in pts]


def _nelder_mead(
    f, x0: np.ndarray, max_iter: int = 200, tol: float = 1e-8
) -> tuple[np.ndarray, float]:
    """Standard Nelder-Mead in unconstrained space."""
    n = x0.size
    simplex = [x0.copy()]
    for i in range(n):
        v = x0.copy()
        v[i] += 1.2
        simplex.append(v)
    simplex = np.asarray(simplex)
    fvals = np.array([f(v) for v in simplex])

    for _ in range(max_iter):
        order = np.argsort(fvals)
        simplex = simplex[order]
        fvals = fvals[order]
        if np.max(np.abs(fvals[1:] - fvals[0])) < tol and \
                np.max(np.abs(simplex[1:] - simplex[0])) < 1e-6:
            break
        centroid = np.mean(simplex[:-1], axis=0)
        xr = centroid + 1.0 * (centroid - simplex[-1])
        fr = f(xr)
        if fvals[0] <= fr < fvals[-2]:
            simplex[-1] = xr
            fvals[-1] = fr
            continue
        if fr < fvals[0]:
            xe = centroid + 2.0 * (xr - centroid)
            fe = f(xe)
            if fe < fr:
                simplex[-1] = xe
                fvals[-1] = fe
            else:
                simplex[-1] = xr
                fvals[-1] = fr
            continue
        if fr < fvals[-1]:
            xc = centroid + 0.5 * (xr - centroid)
            fc = f(xc)
            if fc < fr:
                simplex[-1] = xc
                fvals[-1] = fc
                continue
        else:
            xc = centroid - 0.5 * (centroid - simplex[-1])
            fc = f(xc)
            if fc < fvals[-1]:
                simplex[-1] = xc
                fvals[-1] = fc
                continue
        # shrink
        for i in range(1, n + 1):
            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
            fvals[i] = f(simplex[i])
    order = np.argsort(fvals)
    return simplex[order[0]], fvals[order[0]]


def optimize(
    y: np.ndarray,
    trend_kind: TrendKind,
    seasonal_kind: SeasonalKind,
    period: int,
    locks: Optional[Dict[str, float]] = None,
    progress_cb=None,
) -> tuple[HWParams, float, float]:
    """Minimise one-step SSE. Returns (params, sse, aic)."""
    y = np.asarray(y, dtype=float)
    validate_series(y, seasonal_kind, period)
    locks = dict(locks or {})
    specs_all = _specs(trend_kind)
    free = [s for s in specs_all if s.name not in locks]
    locked = {s.name: locks[s.name] for s in specs_all if s.name in locks}

    def build_params(unit: np.ndarray) -> HWParams:
        values = dict(locked)
        for s, u in zip(free, unit):
            values[s.name] = float(s.low + (s.high - s.low) * u)
        alpha = values.get("alpha", 0.3)
        beta = values.get("beta", 0.0)
        if trend_kind != "none":
            beta = min(values.get("beta", 0.1), alpha)
        gamma = values.get("gamma", 0.3)
        phi = values.get("phi", 1.0)
        if trend_kind == "add_damped":
            phi = min(max(phi, PHI_MIN), PHI_MAX)
        return HWParams(alpha=alpha, beta=beta, gamma=gamma, phi=phi)

    # objective in logit space of free unit parameters
    def objective(z: np.ndarray) -> float:
        unit = _sigmoid(z)
        p = build_params(unit)
        try:
            res = fit_hw(y, trend_kind, seasonal_kind, period, p)
            return res.sse
        except ModelError:
            return 1e300

    best_val = np.inf
    best_unit: Optional[np.ndarray] = None
    if not free:
        starts = [np.zeros(0)]
    else:
        starts = [s[: len(free)] for s in _starts(len(free))]
    for i, s0 in enumerate(starts):
        z0 = _logit(np.clip(s0, 1e-6, 1 - 1e-6)) if s0.size else s0
        x, val = _nelder_mead(objective, z0)
        if val < best_val:
            best_val = val
            best_unit = _sigmoid(x) if x.size else np.zeros(0)
        if progress_cb is not None:
            progress_cb((i + 1) / max(len(starts), 1))

    if best_unit is None or not np.isfinite(best_val) or best_val >= 1e299:
        raise ModelError("参数优化失败：模型在所有候选参数下均不稳定。")

    params = build_params(best_unit)
    final = fit_hw(y, trend_kind, seasonal_kind, period, params)
    return params, final.sse, final.aic


def default_locks() -> Dict[str, float]:
    return {}

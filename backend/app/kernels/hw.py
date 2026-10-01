"""Holt-Winters (triple exponential smoothing) implementation.

This module is deliberately written from scratch: it contains the recursive
state updates, the documented two-season initialisation rule, forecasting
(analytic and simulated intervals), and the one-step-ahead SSE used by the
optimiser.  No statsmodels / forecast package is used.

Notation
--------
``trend`` is one of ``"add"``, ``"add_damped"`` or ``"none"``.
``seasonal`` is one of ``"add"`` (additive) or ``"mul"`` (multiplicative).

Inputs are aligned to the *calendar grid*.  Weekly data may contain missing
weeks (a store renovation, a data pipeline outage): the caller rebuilds the
continuous week-by-week grid and passes NaN for every missing week.  Grid
position ``t`` is therefore a *time* index -- the seasonal slot is ``t % m``
even when the immediately preceding array element is several weeks older.

Missing weeks (NaN) are treated by **propagation without updating**:

* the seasonal slot advances with the calendar position (``t % m``);
* the state moves forward by the deterministic one/multi-step prediction map
  (level grows by trend, trend is damped, seasonal indices are *not* touched);
* no residual is produced, so missing weeks contribute nothing to the SSE,
  to the AIC sample count, or to the residual variance behind intervals.

This is the state-space equivalent of filling the gap with the model's own
forecast (which carries zero innovation) and then recursing normally; see the
"Missing weeks" section below for the rejected alternatives.

State after observing time t (indices are zero based in code):

    level   l_t
    trend   b_t        (only present when a trend is used)
    season  s_{t-m+1} ... s_t   -- m slots, most recent at slot m-1

Recursions
~~~~~~~~~~
* Additive seasonality:

    yhat_t = l_{t-1} + phi * b_{t-1} + s_{t-m}
    l_t    = alpha * (y_t - s_{t-m}) + (1 - alpha) * (l_{t-1} + phi*b_{t-1})
    b_t    = beta  * (l_t - l_{t-1}) + (1 - beta) * phi * b_{t-1}
    s_t    = gamma * (y_t - l_t)    + (1 - gamma) * s_{t-m}

* Multiplicative seasonality:

    yhat_t = (l_{t-1} + phi * b_{t-1}) * s_{t-m}
    l_t    = alpha * (y_t / s_{t-m}) + (1 - alpha) * (l_{t-1} + phi*b_{t-1})
    b_t    = beta  * (l_t - l_{t-1}) + (1 - beta) * phi * b_{t-1}
    s_t    = gamma * (y_t / l_t)    + (1 - gamma) * s_{t-m}

phi = 1 for the undamped additive trend; b is absent when trend == "none".

Initial state (calendar positions 0..2m-1; gaps allowed)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Let L0 and L1 be the means of the first and the second season.

* initial trend: b_{2m-1} = (L1 - L0) / m
* initial level: l_{2m-1} = L1 + (m-1)/2 * b   (linear extrapolation of the
  two season means back/forward to the centre convention used below)

  Rationale: L0 is the level at the centre of season 1, L1 at the centre of
  season 2, so the per-step slope is (L1-L0)/m and the level at index 2m-1
  (end of season 2) is L1 + (m-1)/2 * b.

* additive initial seasonal indices: s_i = y_i - (L0 + (i - (m-1)/2) * b)
  for i = 0..m-1, then centred so that sum(s) = 0; the same m indices are
  reused for the second season (index i+m uses slot i).
* multiplicative initial seasonal indices: s_i = y_i /
  (L0 + (i - (m-1)/2) * b) for i = 0..m-1, then normalised so that
  mean(s) = 1.

When the first 2m calendar weeks span the full window but contain gaps the
initial components are estimated by least squares over the *observed* weeks:
additive model: OLS of y_q on the calendar position q plus m seasonal
dummies; multiplicative model: OLS of ln y_q on q plus m seasonal dummies,
then exponentiate the seasonal factors and normalise them to mean 1.  A
seasonal phase never observed in the window (both q and q+m missing), or too
few observed weeks to identify the trend, makes the model refuse with a
message naming what is missing.  With a complete window the OLS solution is
bit-for-bit the deterministic rule above, so complete-series numbers do not
change.

The recursive fit therefore starts at calendar position 2m and every
forecast error entering the SSE is a genuine out-of-sample one-step error
(only observed weeks contribute a residual).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np

TrendKind = Literal["none", "add", "add_damped"]
SeasonalKind = Literal["add", "mul"]

TRENDS = ("none", "add", "add_damped")
SEASONALS = ("add", "mul")


class ModelError(ValueError):
    """Raised when a series cannot be handled by the requested model."""


@dataclass
class HWParams:
    alpha: float
    beta: float = 0.0
    gamma: float = 0.0
    phi: float = 1.0  # only meaningful for trend == "add_damped"


@dataclass
class HWState:
    """Final state of the recursion after the last observation."""

    level: float
    trend: Optional[float]
    season: np.ndarray  # shape (m,), oldest slot first, newest last
    trend_kind: TrendKind
    seasonal_kind: SeasonalKind
    phi: float

    def to_dict(self) -> dict:
        return {
            "level": float(self.level),
            "trend": None if self.trend is None else float(self.trend),
            "season": [float(x) for x in self.season],
            "trend_kind": self.trend_kind,
            "seasonal_kind": self.seasonal_kind,
            "phi": float(self.phi),
        }


@dataclass
class FitResult:
    trend_kind: TrendKind
    seasonal_kind: SeasonalKind
    period: int
    params: HWParams
    sse: float
    aic: float
    n_effective: int          # observed weeks that contributed to SSE
    residuals: np.ndarray     # one-step residuals (n_effective values)
    fitted: np.ndarray        # in-sample fitted values, length = n_effective
    fitted_index: np.ndarray  # calendar positions of the fitted values
    initial_level: float
    initial_trend: Optional[float]
    initial_season: np.ndarray
    final_state: HWState
    all_fitted: np.ndarray = field(default_factory=lambda: np.zeros(0))
    # one fitted value per *calendar* position (grid length); missing weeks
    # hold the state-propagated forecast used in place of an observation
    n_calendar: int = 0       # length of the aligned weekly grid
    observed_mask: np.ndarray = field(default_factory=lambda: np.zeros(0,
                                                                        bool))
    # True where the grid has an actual observation
    missing_count: int = 0
    initial_window_complete: bool = True  # first 2m weeks had no gaps
    initial_rule: str = "deterministic"


def validate_series(y: np.ndarray, seasonal_kind: SeasonalKind, period: int) -> np.ndarray:
    """Validate the calendar-aligned grid and return its observation mask.

    ``y`` is a weekly grid: finite values are observations, NaN marks a
    missing week.  +-inf is rejected as data corruption, but NaN gaps are a
    normal, supported case.
    """
    y = np.asarray(y, dtype=float)
    if y.ndim != 1 or y.size == 0:
        raise ModelError("序列为空，无法拟合。")
    mask = np.isfinite(y)
    if np.any(np.isposinf(y) | np.isneginf(y)):
        raise ModelError("序列包含无穷大，请先清理数据。")
    if not np.any(mask):
        raise ModelError("序列没有任何有效观测，无法拟合。")
    if period < 2:
        raise ModelError("季节周期必须 >= 2。")
    if y.size < 2 * period:
        raise ModelError(
            f"至少需要跨越两个完整季节（{2 * period} 个日历周），"
            f"当前网格只有 {y.size} 周。"
        )
    if seasonal_kind == "mul" and np.any(y[mask] <= 0):
        raise ModelError(
            "乘法季节模型要求序列全部为正值；观测中存在 0 或负值，已拒绝。"
        )
    return mask


def _trend_growth(b: float, phi: float, steps: int) -> float:
    """sum_{i=1..steps} phi^i * b  (steps >= 0)."""
    if steps == 0:
        return 0.0
    if abs(phi - 1.0) < 1e-12:
        return steps * b
    return b * phi * (1.0 - phi**steps) / (1.0 - phi)


def initial_components(
    y: np.ndarray, trend_kind: TrendKind, seasonal_kind: SeasonalKind, period: int
) -> tuple[float, Optional[float], np.ndarray, bool, str]:
    """Estimate initial level / trend / seasonal indices from the first
    2m calendar positions.

    Returns ``(l0, b0, seas, window_complete, init_rule)`` where ``l0`` is
    the level at calendar position 2m-1, ``b0`` is the per-week trend
    (None when trend_kind == "none") and ``seas`` has m slots indexed by
    ``q % m``.  ``init_rule`` is "deterministic" for a complete window,
    "saturated" for a gapped window whose year 1 covers every phase, and
    "pooled" when some year-1 phases were filled from year-2 evidence.
    """
    m = period
    win = y[: 2 * m]
    mask = np.isfinite(win)
    complete = bool(np.all(mask))
    if complete:
        l0, b0, seas = _initial_components_complete(
            win, trend_kind, seasonal_kind, m
        )
        rule = "deterministic"
    else:
        l0, b0, seas, rule = _initial_components_gapped(
            win, mask, trend_kind, seasonal_kind, m
        )
    if seasonal_kind == "mul" and np.any(seas <= 0):
        raise ModelError("初始乘法季节指数出现非正值，请改用加法季节模型。")
    return l0, b0, seas, complete, rule


def _initial_components_complete(
    win: np.ndarray, trend_kind: TrendKind,
    seasonal_kind: SeasonalKind, m: int,
) -> tuple[float, Optional[float], np.ndarray]:
    """Documented deterministic rule; identical to OLS on a full window."""
    y0 = win[:m]
    y1 = win[m:]
    L0 = float(np.mean(y0))
    L1 = float(np.mean(y1))

    if trend_kind == "none":
        b0 = None
        l0 = L1
        base = L0
    else:
        b0 = (L1 - L0) / m
        l0 = L1 + (m - 1) / 2.0 * b0
        base = None

    centre = (m - 1) / 2.0
    if seasonal_kind == "add":
        if trend_kind == "none":
            seas = y0 - base
            seas = seas - seas.mean()
        else:
            t = np.arange(m, dtype=float)
            trend_line = L0 + (t - centre) * b0
            seas = y0 - trend_line
            seas = seas - seas.mean()
    else:
        if trend_kind == "none":
            denom = base
            if denom <= 0:
                raise ModelError(
                    "初始季节分母非正，无法为乘法模型构造季节指数（请检查序列）。"
                )
            seas = y0 / denom
        else:
            t = np.arange(m, dtype=float)
            denom = L0 + (t - centre) * b0
            if np.any(denom <= 0):
                raise ModelError(
                    "初始趋势拟合值非正，乘法季节无法处理（请改用加法季节）。"
                )
            seas = y0 / denom
        seas = seas / seas.mean()
    return l0, b0, seas


def _initial_components_gapped(
    win: np.ndarray, mask: np.ndarray, trend_kind: TrendKind,
    seasonal_kind: SeasonalKind, m: int,
) -> tuple[float, Optional[float], np.ndarray, str]:
    """Initial components when the first 2m calendar weeks contain gaps.

    Deterministic two-season rule generalised to missing cells; on a
    complete window the results are bit-for-bit those of
    :func:`_initial_components_complete`.

    * Trend slope b (additive; log-slope for multiplicative): estimate by
      ordinary least squares of z_q (z = y or ln y) on calendar position q
      with a seasonal dummy for every phase, restricted to the observed
      cells.  With all 2m cells present the design is saturated and the
      slope equals (L1-L0)/m exactly.
    * Seasonal slots: deviations from the fitted trend line, year-1 cells
      preferred; a phase missing from year 1 borrows its year-2 cell
      ("pooled" slot).  Additive slots are centred to sum 0 over the m
      phases, multiplicative slots normalised to mean 1.
    * End-of-window level: L1 + (m-1)/2 b, with L1 the mean of year-2
      cells detrended at centre (m-1)/2 -- again identical to the
      deterministic rule on a complete window.

    A phase missing from both years, or too few observations, raises.
    """
    qobs = np.arange(2 * m)[mask]
    if qobs.size < 2:
        raise ModelError(
            f"前两个季节（{2 * m} 个日历周）中只有 {qobs.size} 个观测，"
            "不足以估计初始水平与季节分量，请在缺口两侧补数或更换起点。"
        )
    y1p = mask[:m]
    y2p = mask[m:]
    both_missing = [k for k in range(m) if not y1p[k] and not y2p[k]]
    if both_missing:
        show = ", ".join(str(k + 1) for k in both_missing[:8])
        more = " …" if len(both_missing) > 8 else ""
        raise ModelError(
            "前两个季节内存在整个季节相位两年都未被观测，无法估计季节初值；"
            f"缺失相位（从 1 数）：{show}{more}。"
            "请至少在其中一年补齐该周，或提供更长的历史。"
        )
    if seasonal_kind == "mul" and np.any(win[mask] <= 0):
        raise ModelError(
            "乘法季节模型要求序列全部为正值；前两个季节的观测中存在 0 或负值，"
            "已拒绝。"
        )

    z = win if seasonal_kind == "add" else np.log(win)
    centre = (m - 1) / 2.0
    k_y1 = np.arange(m)[y1p]
    k_y2 = np.arange(m)[y2p]

    # --- slope: OLS with phase dummies (reference phase r = last observed
    # phase; recovered later by centring).  Identical to the deterministic
    # slope on a complete window.
    if trend_kind == "none":
        b = 0.0
        b0 = None
    else:
        if qobs.size < 3:
            raise ModelError(
                f"前两个季节中只有 {qobs.size} 个可用观测，无法同时识别趋势与"
                "季节，请补齐缺口或改用无趋势模型。"
            )
        ref = int(k_y1[-1]) if k_y1.size else int(k_y2[-1])
        active = [k for k in range(m)
                  if k != ref and (y1p[k] or y2p[k])]
        Xcols = [np.ones(qobs.size), qobs.astype(float)]
        for k in active:
            Xcols.append((qobs % m == k).astype(float))
        X = np.column_stack(Xcols)
        rank = int(np.linalg.matrix_rank(X))
        if rank < X.shape[1]:
            raise ModelError(
                "前两个季节的观测不足以同时识别趋势与季节相位，"
                "请补齐缺口或提供更长的历史。"
            )
        beta, *_ = np.linalg.lstsq(X, z[qobs], rcond=None)
        b = float(beta[1])
        b0 = b

    # --- year means.  The deterministic rule detrends at centre (m-1)/2;
    # the seasonal slots use those detrended means, while the end-of-window
    # level is anchored at the *raw* year-2 mean (complete window identity).
    def detr(year_offset, ks):
        return np.array([
            z[year_offset * m + k] - b * (k - centre) for k in ks
        ])

    d0 = detr(0, k_y1) if k_y1.size else detr(1, k_y2) - b * m
    d1 = detr(1, k_y2) if k_y2.size else detr(0, k_y1) + b * m
    L0 = float(np.mean(d0))
    L1 = float(np.mean(d1))
    raw1 = float(np.mean(z[m:][y2p])) if y2p.any() else \
        float(np.mean(z[:m][y1p])) + b * m

    # --- seasonal slots: year-1 deviation, pooled year-2 if year 1 missing
    slots = np.full(m, np.nan)
    for k in k_y1:
        slots[k] = z[k] - (L0 + (k - centre) * b)
    if seasonal_kind == "add":
        shift1 = float(np.nansum(slots) / m)
        for k in k_y2:
            if not y1p[k]:
                slots[k] = z[m + k] - (L1 + (k - centre) * b)
        slots = slots - float(np.nansum(slots) / m)
        l0 = L1 if b0 is None else raw1 + (m - 1) / 2.0 * b
    else:
        shift1 = float(np.nansum(np.exp(slots)) / m)
        for k in k_y2:
            if not y1p[k]:
                slots[k] = (z[m + k] - (L1 + (k - centre) * b)
                            + np.log(shift1))
        raw = np.exp(slots)
        slots = raw / float(np.nansum(raw) / m)
        l0 = np.exp(L1 if b0 is None else raw1 + (m - 1) / 2.0 * b)

    rule = "saturated" if bool(np.all(y1p)) else "pooled"
    return float(l0), b0, slots, rule


def _run_recursion(
    y: np.ndarray,
    trend_kind: TrendKind,
    seasonal_kind: SeasonalKind,
    period: int,
    params: HWParams,
) -> tuple[np.ndarray, np.ndarray, HWState, float, Optional[float],
           np.ndarray, np.ndarray, bool, str]:
    """Run the recursive updates over calendar positions 2m..n-1.

    NaN positions are missing weeks: the state is propagated through them
    (level advances by trend, seasonal slot advances by calendar position)
    without any update or residual.

    Returns ``(fitted_grid, residual_grid, final_state, init_level,
    init_trend, init_season, observed_tail, window_complete)`` where the two
    grids have length n-2m (NaN at missing positions).
    """
    m = period
    l, b, seas, complete, init_rule = initial_components(
        y, trend_kind, seasonal_kind, m
    )
    init_level, init_trend, init_season = l, b, seas.copy()
    phi = params.phi if trend_kind == "add_damped" else 1.0
    alpha, gamma = params.alpha, params.gamma
    beta = params.beta if trend_kind != "none" else 0.0

    n = y.size
    tail = n - 2 * m
    fitted_grid = np.full(tail, np.nan)
    resid_grid = np.full(tail, np.nan)
    observed_tail = np.zeros(tail, dtype=bool)
    for t in range(2 * m, n):
        k = t % m
        observed = np.isfinite(y[t])
        growth = _trend_growth(b, phi, 1) if b is not None else 0.0
        level_plus = l + growth
        if seasonal_kind == "add":
            yhat = level_plus + seas[k]
            if not np.isfinite(yhat):
                raise ModelError("递推过程中拟合值溢出。")
            fitted_grid[t - 2 * m] = yhat
            if not observed:
                # Propagation without updating (zero innovation): the level
                # advances by phi*b and the trend itself decays to phi*b;
                # seasonal indices are left untouched.
                l = level_plus
                if b is not None:
                    b = phi * b
                continue
            e = float(y[t] - yhat)
            l_new = alpha * (y[t] - seas[k]) + (1 - alpha) * level_plus
            seas_new = gamma * (y[t] - l_new) + (1 - gamma) * seas[k]
        else:
            s = seas[k]
            if s <= 0:
                raise ModelError("乘法季节指数在递推中非正，模型失败。")
            yhat = level_plus * s
            if not np.isfinite(yhat) or yhat <= 0:
                raise ModelError("乘法模型拟合值非正或溢出。")
            fitted_grid[t - 2 * m] = yhat
            if not observed:
                l = level_plus
                if b is not None:
                    b = phi * b
                continue
            e = float(y[t] - yhat)
            l_new = alpha * (y[t] / s) + (1 - alpha) * level_plus
            if l_new <= 0:
                raise ModelError("乘法模型水平在递推中非正，模型失败。")
            seas_new = gamma * (y[t] / l_new) + (1 - gamma) * s
            if seas_new <= 0:
                raise ModelError("乘法季节指数在递推中非正，模型失败。")

        resid_grid[t - 2 * m] = e
        observed_tail[t - 2 * m] = True
        if b is not None:
            b = beta * (l_new - l) + (1 - beta) * phi * b
        l = l_new
        seas[k] = seas_new

    # Canonicalise seasonal slots for forecasting: slot r must hold
    # s_{n-1-(m-1)+r}, i.e. slot 0 = s_{n+1-m} (used at horizon 1) and
    # slot m-1 = s_{n-1} (the newest index).
    last_mod = (n - 1) % m
    seas = np.array(
        [seas[(r + last_mod + 1) % m] for r in range(m)]
    )
    state = HWState(
        level=float(l),
        trend=None if b is None else float(b),
        season=np.asarray(seas, dtype=float).copy(),
        trend_kind=trend_kind,
        seasonal_kind=seasonal_kind,
        phi=float(phi),
    )
    return (fitted_grid, resid_grid, state, init_level, init_trend,
            init_season, observed_tail, complete, init_rule)


def _param_count(trend_kind: TrendKind, seasonal_kind: SeasonalKind) -> int:
    """Number of free smoothing parameters (used in AIC)."""
    k = 2 if trend_kind == "none" else 3  # alpha, gamma (+beta)
    if trend_kind == "add_damped":
        k += 1  # phi
    return k


def aic_from_sse(sse: float, n: int, k: int) -> float:
    """AIC under Gaussian one-step errors with SSE-based variance estimate.

    AIC = n * (ln(2*pi) + ln(SSE/n) + 1) + 2(k+1)

    where k is the number of free smoothing parameters and the "+1" is the
    error variance.  (Documented in README.)
    """
    sse = max(float(sse), 1e-12)
    return float(n * (np.log(2 * np.pi) + np.log(sse / n) + 1.0) + 2 * (k + 1))


def fit_hw(
    y: np.ndarray,
    trend_kind: TrendKind,
    seasonal_kind: SeasonalKind,
    period: int,
    params: HWParams,
) -> FitResult:
    """Fit with *given* smoothing parameters and compute fit statistics.

    ``y`` is a calendar-aligned weekly grid with NaN for missing weeks.
    """
    y = np.asarray(y, dtype=float)
    mask = validate_series(y, seasonal_kind, period)
    (fitted_grid, resid_grid, state, l0, b0, s0, observed_tail,
     window_complete, init_rule) = _run_recursion(
        y, trend_kind, seasonal_kind, period, params
    )
    fitted = fitted_grid[observed_tail]
    resid = resid_grid[observed_tail]
    n = resid.size
    if n == 0:
        raise ModelError(
            "前两个季节之后没有任何已观测周，无法计算一步误差与 SSE"
            "（请提供更长的历史）。"
        )
    sse = float(np.sum(resid**2))
    aic = aic_from_sse(sse, n, _param_count(trend_kind, seasonal_kind))

    # One fitted value per *calendar* position (length = grid length).  The
    # first 2m weeks carry the deterministic/OLS in-sample fit; later weeks
    # carry one-step forecasts.  Missing weeks are NaN: they are not a fit.
    all_fitted = np.full(y.size, np.nan)
    if trend_kind == "none":
        for i in range(period):
            if seasonal_kind == "add":
                val = l0 + s0[i]
            else:
                val = l0 * s0[i]
            all_fitted[i] = val
            all_fitted[i + period] = val
    else:
        phi = params.phi if trend_kind == "add_damped" else 1.0
        # initial state is indexed at t = 2m-1; forecast "back" to earlier
        # slots using the same deterministic trend line (phi sums work with
        # negative steps only for phi == 1; with damping we approximate the
        # pre-estimation window linearly, matching initial component design).
        for i in range(2 * period):
            k = i % period
            if abs(phi - 1.0) < 1e-12:
                level_at = l0 + (i - (2 * period - 1)) * b0
            else:
                level_at = l0 + b0 * (i - (2 * period - 1))
            if seasonal_kind == "add":
                all_fitted[i] = level_at + s0[k]
            else:
                all_fitted[i] = level_at * s0[k]
    all_fitted[2 * period :] = fitted_grid
    all_fitted[~mask] = np.nan

    # Shocks used for prediction intervals: raw residuals for the additive
    # error model, relative residuals e/yhat for the multiplicative model.
    if seasonal_kind == "mul":
        shocks = resid / np.maximum(np.abs(fitted), 1e-12)
    else:
        shocks = resid
    object.__setattr__(state, "_shocks", shocks)
    object.__setattr__(state, "_alpha", float(params.alpha))
    object.__setattr__(state, "_beta",
                       float(params.beta) if trend_kind != "none" else 0.0)
    object.__setattr__(state, "_gamma", float(params.gamma))

    return FitResult(
        trend_kind=trend_kind,
        seasonal_kind=seasonal_kind,
        period=period,
        params=params,
        sse=sse,
        aic=aic,
        n_effective=n,
        residuals=resid,
        fitted=fitted,
        fitted_index=np.arange(2 * period, y.size)[observed_tail],
        initial_level=float(l0),
        initial_trend=b0,
        initial_season=s0,
        final_state=state,
        all_fitted=np.asarray(all_fitted, dtype=float),
        n_calendar=int(y.size),
        observed_mask=mask,
        missing_count=int((~mask).sum()),
        initial_window_complete=bool(window_complete),
        initial_rule=str(init_rule),
    )


def forecast(
    state: HWState,
    h: int,
    residuals: np.ndarray,
    level: float = 0.95,
    method: Literal["analytic", "simulate"] = "analytic",
    n_sims: int = 2000,
    rng: Optional[np.random.Generator] = None,
) -> "ForecastResult":
    """Point forecasts and prediction intervals for 1..h steps ahead.

    Analytic intervals follow the additive-error state-space approximation
    (see module docs / README).  Simulated intervals use a residual bootstrap
    with Gaussian replacement when residuals are degenerate.  In both cases
    half-widths are enforced monotone non-decreasing in horizon h (cumulative
    maximum), which is a property of correctly accumulated forecast error
    variance and protects against bootstrap noise.
    """
    if h < 1:
        raise ModelError("预测步长必须 >= 1。")
    m = state.season.size
    phi = state.phi if state.trend_kind == "add_damped" else 1.0
    b = 0.0 if state.trend is None else state.trend
    l = state.level

    point = np.empty(h)
    for j in range(1, h + 1):
        s = state.season[(j - 1) % m]
        growth = _trend_growth(b, phi, j) if state.trend is not None else 0.0
        if state.seasonal_kind == "add":
            point[j - 1] = l + growth + s
        else:
            point[j - 1] = (l + growth) * s
    point = np.maximum(point, 0.0) if state.seasonal_kind == "mul" else point

    shocks = np.asarray(getattr(state, "_shocks", residuals), dtype=float)
    z = _normal_quantile(0.5 + level / 2.0)

    if method == "simulate":
        lower, upper = _simulate_bands(
            state, point, shocks, h, level, n_sims, rng
        )
        sigma = float(np.std(shocks, ddof=1)) if shocks.size > 1 else 0.0
    else:
        lower, upper, sigma = _analytic_bands(state, point, shocks, h, z)

    half = (upper - lower) / 2.0
    half = np.maximum.accumulate(half)
    centre = (upper + lower) / 2.0  # equals point for analytic
    lower_out = centre - half
    upper_out = centre + half
    if state.seasonal_kind == "mul":
        lower_out = np.maximum(lower_out, 0.0)

    return ForecastResult(
        point=point,
        lower=lower_out,
        upper=upper_out,
        level=level,
        method=method,
        residual_std=sigma,
    )


def _normal_quantile(p: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation)."""
    a = [
        -3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
        1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
        6.680131188771972e01, -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
        -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
        3.754408661907416e00,
    ]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = (-2 * np.log(p)) ** 0.5
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5]) * q / \
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    q = (-2 * np.log(1 - p)) ** 0.5
    return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
            ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)


def _psi_weights(state: HWState, h: int, alpha: float, beta: float,
                 gamma: float) -> np.ndarray:
    """Coefficients of past innovations in the h-step forecast error.

    Indexed so that forecast variance at horizon h is
    ``sigma**2 * sum_{i<h} psi[i]**2``.  Using the Gardner equivalent
    error-form recursions (l_t = l_{t-1}+phi*b_{t-1}+alpha*e_t,
    b_t = phi*b_{t-1}+alpha*beta*e_t):

        psi[0]   = 1                                 (the new innovation)
        psi[h-1] = alpha + alpha*beta * PhiSum(h-1)  (h >= 2)
                   + gamma*(1-alpha) * 1{(h-1) mod m == 0}

    with PhiSum(r) = phi+...+phi^r (r when phi = 1).  The seasonal term
    appears once per full season: e_t updates s_t by gamma(1-alpha)e_t and
    that index is consumed exactly m observations later.  These closed
    coefficients were validated by residual-bootstrap simulation (see
    tests/test_hw_properties.py monotonicity + empirical doc in README).
    """
    phi = state.phi if state.trend_kind == "add_damped" else 1.0
    has_trend = state.trend is not None
    m = state.season.size
    psi = np.zeros(h)
    psi[0] = 1.0
    for hh in range(2, h + 1):
        lag = hh - 1
        if has_trend:
            if abs(phi - 1.0) < 1e-12:
                trend_part = alpha + alpha * beta * lag
            else:
                phi_sum = phi * (1.0 - phi**lag) / (1.0 - phi)
                trend_part = alpha + alpha * beta * phi_sum
        else:
            trend_part = alpha
        if lag % m == 0:
            trend_part += gamma * (1.0 - alpha)
        psi[hh - 1] = trend_part
    return psi


def _analytic_bands(
    state: HWState, point: np.ndarray, shocks: np.ndarray, h: int, z: float
) -> tuple[np.ndarray, np.ndarray, float]:
    sigma2 = float(np.var(shocks, ddof=1)) if shocks.size > 1 else 0.0
    alpha = getattr(state, "_alpha", 0.3)
    beta = getattr(state, "_beta", 0.1 if state.trend is not None else 0.0)
    gamma = getattr(state, "_gamma", 0.3)
    psi = _psi_weights(state, h, alpha, beta, gamma)
    cum = np.cumsum(psi**2)
    se = np.sqrt(np.maximum(sigma2 * cum, 0.0))
    if state.seasonal_kind == "mul":
        # shocks are relative -> relative one-step standard errors
        lower = point * (1 - z * se)
        upper = point * (1 + z * se)
    else:
        lower = point - z * se
        upper = point + z * se
    return lower, upper, float(np.sqrt(sigma2))


def _simulate_bands(
    state: HWState,
    point: np.ndarray,
    shocks: np.ndarray,
    h: int,
    level: float,
    n_sims: int,
    rng: Optional[np.random.Generator],
) -> tuple[np.ndarray, np.ndarray]:
    rng = rng or np.random.default_rng(20260929)
    if shocks.size > 1:
        sigma = float(np.std(shocks, ddof=1))
        pool = shocks - shocks.mean()
    else:
        sigma = 0.0
        pool = shocks
    m = state.season.size
    phi = state.phi if state.trend_kind == "add_damped" else 1.0
    alpha = getattr(state, "_alpha", 0.3)
    beta = getattr(state, "_beta", 0.1 if state.trend is not None else 0.0)
    gamma = getattr(state, "_gamma", 0.3)

    paths = np.empty((n_sims, h))
    # Vectorised simulation across paths.
    l = np.full(n_sims, state.level)
    b = (np.zeros(n_sims) if state.trend is None
         else np.full(n_sims, state.trend))
    seas = np.tile(state.season, (n_sims, 1))
    for j in range(1, h + 1):
        k = (j - 1) % m
        growth = _trend_growth_vec(b, phi, j) if state.trend is not None else 0.0
        lp = l + growth
        s = seas[:, k]
        mu = lp + s if state.seasonal_kind == "add" else lp * s
        paths[:, j - 1] = mu
        if sigma > 0:
            e = rng.choice(pool, size=n_sims, replace=True)
        else:
            e = np.zeros(n_sims)
        if state.seasonal_kind == "add":
            y = mu + e
            l_new = alpha * (y - s) + (1 - alpha) * lp
            seas[:, k] = gamma * (y - l_new) + (1 - gamma) * s
        else:
            # e is a *relative* innovation: y = mu * (1 + e)
            y = mu * (1 + e)
            l_new = alpha * (y / np.maximum(s, 1e-12)) + (1 - alpha) * lp
            seas[:, k] = gamma * (y / np.maximum(l_new, 1e-12)) + \
                (1 - gamma) * s
            seas[:, k] = np.maximum(seas[:, k], 1e-9)
        if state.trend is not None:
            b = beta * (l_new - l) + (1 - beta) * phi * b
        l = l_new

    lo_q = (1 - level) / 2.0
    lower = np.quantile(paths, lo_q, axis=0)
    upper = np.quantile(paths, 1 - lo_q, axis=0)
    return lower, upper


def _trend_growth_vec(b: np.ndarray, phi: float, steps: int) -> np.ndarray:
    if steps == 0:
        return np.zeros_like(b)
    if abs(phi - 1.0) < 1e-12:
        return steps * b
    return b * phi * (1.0 - phi**steps) / (1.0 - phi)


@dataclass
class ForecastResult:
    point: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    level: float
    method: str
    residual_std: float

    def to_dict(self) -> dict:
        return {
            "point": [float(x) for x in self.point],
            "lower": [float(x) for x in self.lower],
            "upper": [float(x) for x in self.upper],
            "level": float(self.level),
            "method": self.method,
            "residual_std": float(self.residual_std),
        }


def attach_params_to_state(
    state: HWState, params: HWParams
) -> HWState:
    """Expose smoothing params for interval computation (analytic/simulate)."""
    object.__setattr__(state, "_alpha", float(params.alpha))
    object.__setattr__(state, "_beta", float(params.beta))
    object.__setattr__(state, "_gamma", float(params.gamma))
    return state

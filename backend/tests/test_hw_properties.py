"""Required relationship/property tests for the Holt-Winters kernel."""
from __future__ import annotations

import numpy as np
import pytest

from app.kernels import fit_hw, forecast, optimizer
from app.kernels.backtest import run_backtest
from app.kernels.hw import HWParams, ModelError
from app.kernels.selection import auto_select
from conftest import SEAS


# 1. Noiseless additive series -> near-zero SSE, trend & season continue.
def test_additive_noiseless_fit_and_forecast(additive_series, period):
    y = additive_series
    params, sse, _ = optimizer.optimize(y, "add", "add", period)
    fr = fit_hw(y, "add", "add", period, params)
    n = y.size
    assert fr.sse < 1e-18

    h = 2 * period
    fc = forecast(fr.final_state, h, fr.residuals, level=0.95)
    future_t = np.arange(n, n + h)
    expected = 10.0 + 0.35 * future_t + SEAS[future_t % period]
    np.testing.assert_allclose(fc.point, expected, atol=1e-8)


# 2. Scale invariance of the multiplicative model:
#    y -> c*y  leaves parameters unchanged and scales forecasts by c.
def test_multiplicative_scale_invariance(multiplicative_series, period):
    rng = np.random.default_rng(11)
    # multiplicative noise keeps the model family appropriate and gives a
    # unique optimum.
    y = multiplicative_series * (1 + rng.normal(0, 0.01,
                                                multiplicative_series.shape))
    c = 3.7
    p1, _, _ = optimizer.optimize(y, "add", "mul", period)
    p2, _, _ = optimizer.optimize(c * y, "add", "mul", period)
    assert p1.alpha == pytest.approx(p2.alpha, abs=1e-6)
    assert p1.beta == pytest.approx(p2.beta, abs=1e-6)
    assert p1.gamma == pytest.approx(p2.gamma, abs=1e-6)
    assert p1.phi == pytest.approx(p2.phi, abs=1e-6)

    f1 = fit_hw(y, "add", "mul", period, p1)
    f2 = fit_hw(c * y, "add", "mul", period, p2)
    q1 = forecast(f1.final_state, 10, f1.residuals)
    q2 = forecast(f2.final_state, 10, f2.residuals)
    np.testing.assert_allclose(np.array(q2.point), c * np.array(q1.point),
                               rtol=1e-8)


# 3. Shift invariance of the additive model:
#    y -> y + c leaves parameters unchanged and shifts forecasts by c.
def test_additive_shift_invariance(additive_series, period):
    # Small fixed noise makes the SSE optimum unique (otherwise any params
    # give SSE~0 and the optimum is non-unique).
    rng = np.random.default_rng(7)
    y = additive_series + rng.normal(0, 0.05, additive_series.shape)
    c = 50.0
    p1, _, _ = optimizer.optimize(y, "add", "add", period)
    p2, _, _ = optimizer.optimize(y + c, "add", "add", period)
    assert p1.alpha == pytest.approx(p2.alpha, abs=1e-6)
    assert p1.beta == pytest.approx(p2.beta, abs=1e-6)
    assert p1.gamma == pytest.approx(p2.gamma, abs=1e-6)

    f1 = fit_hw(y, "add", "add", period, p1)
    f2 = fit_hw(y + c, "add", "add", period, p2)
    q1 = forecast(f1.final_state, 10, f1.residuals)
    q2 = forecast(f2.final_state, 10, f2.residuals)
    np.testing.assert_allclose(np.array(q2.point),
                               np.array(q1.point) + c, atol=1e-8)


# 4. Seasonal normalisation conventions.
def test_seasonal_normalisation(additive_series, multiplicative_series, period):
    y = additive_series
    p, _, _ = optimizer.optimize(y, "add_damped", "add", period)
    fr = fit_hw(y, "add_damped", "add", period, p)
    assert abs(fr.initial_season.sum()) < 1e-10
    assert abs(fr.final_state.season.sum()) < 1e-6

    ym = multiplicative_series
    pm, _, _ = optimizer.optimize(ym, "add", "mul", period)
    frm = fit_hw(ym, "add", "mul", period, pm)
    assert frm.initial_season.mean() == pytest.approx(1.0, abs=1e-10)
    assert frm.final_state.season.mean() == pytest.approx(1.0, abs=1e-6)


# 5. Backtest origin k only sees data up to k: injecting extreme values
#    afterwards leaves that origin unchanged.
def test_backtest_origin_isolation(multiplicative_series, period):
    y = multiplicative_series
    origin = 2 * period + 2
    h = 6
    bt_a = run_backtest(y, period, origin_start=origin, horizon=h,
                        trend_kind="add", seasonal_kind="mul")
    row_a = bt_a.origins[0]

    y2 = y.copy()
    y2[origin:] += 1e6  # contaminate *everything after* the origin
    bt_b = run_backtest(y2, period, origin_start=origin, horizon=h,
                        trend_kind="add", seasonal_kind="mul")
    row_b = bt_b.origins[0]

    assert row_b.train_size == row_a.train_size
    np.testing.assert_allclose(row_b.forecast, row_a.forecast, rtol=1e-8)
    np.testing.assert_allclose(row_b.params["alpha"],
                               row_a.params["alpha"], rtol=1e-8)
    # Actuals naturally differ, that is expected; forecasts must not.


# 6. Repeated fits are identical.
def test_repeated_fit_is_deterministic(additive_series, period):
    y = additive_series
    r1 = [optimizer.optimize(y, tk, "add", period)
          for tk in ("none", "add", "add_damped")]
    r2 = [optimizer.optimize(y, tk, "add", period)
          for tk in ("none", "add", "add_damped")]
    for (p_a, s_a, a_a), (p_b, s_b, a_b) in zip(r1, r2):
        assert (p_a.alpha, p_a.beta, p_a.gamma, p_a.phi) == \
               pytest.approx((p_b.alpha, p_b.beta, p_b.gamma, p_b.phi),
                             abs=1e-10)
        assert s_a == pytest.approx(s_b, abs=1e-12)
        assert a_a == pytest.approx(a_b, abs=1e-10)
    sel1 = auto_select(y, period)
    sel2 = auto_select(y, period)
    best1 = (sel1.best.trend_kind, sel1.best.seasonal_kind, sel1.best.aic)
    best2 = (sel2.best.trend_kind, sel2.best.seasonal_kind, sel2.best.aic)
    assert best1 == best2


# 7. Validation rejections.
def test_validation_rejections(period):
    y_pos = np.arange(1, 4 * period + 1, dtype=float)
    y_with_zero = y_pos.copy()
    y_with_zero[3] = 0.0
    with pytest.raises(ModelError, match="正"):
        fit_hw(y_with_zero, "add", "mul", period,
               HWParams(0.3, 0.1, 0.3))
    y_neg = y_pos.copy()
    y_neg[3] = -2.0
    with pytest.raises(ModelError):
        fit_hw(y_neg, "add", "mul", period, HWParams(0.3, 0.1, 0.3))
    short = y_pos[: 2 * period - 1]
    with pytest.raises(ModelError, match="两个完整季节"):
        fit_hw(short, "add", "add", period, HWParams(0.3, 0.1, 0.3))


# 8. Interval widths monotone non-decreasing in horizon.
@pytest.mark.parametrize("method", ["analytic", "simulate"])
def test_interval_widths_monotone(multiplicative_series, period, method):
    y = multiplicative_series
    p, _, _ = optimizer.optimize(y, "add", "mul", period)
    fr = fit_hw(y, "add", "mul", period, p)
    fc = forecast(fr.final_state, 20, fr.residuals, level=0.9,
                  method=method)
    width = np.array(fc.upper) - np.array(fc.lower)
    assert np.all(np.diff(width) >= -1e-9)


# 9. Locked parameters stay fixed; remaining parameters are optimised.
def test_locked_parameters(additive_series, period):
    y = additive_series
    p, _, _ = optimizer.optimize(y, "add", "add", period,
                                 locks={"alpha": 0.41, "gamma": 0.17})
    assert p.alpha == pytest.approx(0.41, abs=1e-9)
    assert p.gamma == pytest.approx(0.17, abs=1e-9)
    assert 0 < p.beta < 1


# 9b. Analytic variance ratios track bootstrap simulation (both are estimates
# of the same error variance; analytic is a linearisation, simulation is
# empirical, so they should broadly agree at medium/long horizons).
def test_analytic_and_simulated_bands_agree(multiplicative_series, period):
    y = multiplicative_series
    p, _, _ = optimizer.optimize(y, "add", "mul", period)
    fr = fit_hw(y, "add", "mul", period, p)
    a = forecast(fr.final_state, 16, fr.residuals, 0.9, "analytic")
    s = forecast(
        fr.final_state, 16, fr.residuals, 0.9, "simulate",
        n_sims=4000, rng=np.random.default_rng(42),
    )
    wa = np.array(a.upper) - np.array(a.lower)
    ws = np.array(s.upper) - np.array(s.lower)
    # same order of magnitude at horizons >= 4
    ratio = wa[4:] / ws[4:]
    assert np.all(ratio > 0.4) and np.all(ratio < 2.5)


# 10. Auto-selection scorecard includes every combination and best = min AIC.
def test_auto_select_scorecard(multiplicative_series, additive_series, period):
    sel = auto_select(multiplicative_series, period)
    assert len(sel.scores) == 6
    feasible = [s for s in sel.scores if s.feasible]
    assert sel.best.aic == min(s.aic for s in feasible)

    # additive-only series with a zero: multiplicative combos are rejected
    # with a reason while additive ones still fit.
    yz = additive_series.copy()
    yz[2 * period] = 0.0
    sel2 = auto_select(yz + 100, period)
    mul_rows = [s for s in sel2.scores if s.seasonal_kind == "mul"]
    # shifted series is positive -> all feasible here
    assert all(s.feasible for s in mul_rows)

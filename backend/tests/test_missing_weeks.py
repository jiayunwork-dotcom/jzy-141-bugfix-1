"""Gap (missing-week) handling: the bug report's pinned numbers.

The series is the one from the incident report: 164 weekly points starting
2022-01-03, y_t = 200 + 0.5 t with a +120 spike at seasonal phase 48 and a
+60 spike at phase 20, no noise.  Weeks 2024-02-26 / 03-04 / 03-11
(calendar positions 112-114) are deleted.  With calendar-aligned handling
the gappy series must fit, select, forecast and backtest like the full one.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from app.kernels import fit_hw, forecast, optimizer
from app.kernels.backtest import run_backtest
from app.kernels.hw import HWParams, ModelError
from app.kernels.selection import auto_select

M = 52
N = 164
D0 = date(2022, 1, 3)
GAP_POSITIONS = [112, 113, 114]          # 2024-02-26, 03-04, 03-11
GAP_DATES = [D0 + timedelta(weeks=t) for t in GAP_POSITIONS]


@pytest.fixture(scope="module")
def full_series():
    t = np.arange(N)
    y = 200.0 + 0.5 * t
    y[t % M == 48] += 120.0
    y[t % M == 20] += 60.0
    return y.astype(float)


@pytest.fixture(scope="module")
def gappy_series(full_series):
    y = full_series.copy()
    y[GAP_POSITIONS] = np.nan
    return y


# ---------------------------------------------------------------- full vs gap
def test_full_series_pinned_numbers(full_series):
    """The complete series' numbers from the report (must never change)."""
    p, sse, _ = optimizer.optimize(full_series, "add", "add", M)
    fr = fit_hw(full_series, "add", "add", M, p)
    fc = forecast(fr.final_state, 12, fr.residuals, level=0.95)

    assert p.alpha == pytest.approx(0.5871, abs=1e-3)
    assert p.beta == pytest.approx(0.1507, abs=1e-3)
    assert p.gamma == pytest.approx(0.3004, abs=1e-3)
    assert fr.sse == pytest.approx(0.0, abs=1e-8)
    assert fr.n_effective == N - 2 * M
    half0 = (fc.upper[0] - fc.lower[0]) / 2
    assert half0 == pytest.approx(0.0, abs=1e-8)


def test_gappy_sse_params_forecast_match_full(full_series, gappy_series):
    """3 deleted bland weeks: same fit, same params, same 12-step forecast."""
    pf, _, _ = optimizer.optimize(full_series, "add", "add", M)
    pg, sse_g, _ = optimizer.optimize(gappy_series, "add", "add", M)
    frf = fit_hw(full_series, "add", "add", M, pf)
    frg = fit_hw(gappy_series, "add", "add", M, pg)
    fcf = forecast(frf.final_state, 12, frf.residuals, level=0.95)
    fcg = forecast(frg.final_state, 12, frg.residuals, level=0.95)

    assert pg.alpha == pytest.approx(pf.alpha, abs=1e-6)
    assert pg.beta == pytest.approx(pf.beta, abs=1e-6)
    assert pg.gamma == pytest.approx(pf.gamma, abs=1e-6)
    assert frg.sse == pytest.approx(0.0, abs=1e-6)
    # the 3 deleted weeks are not observations contributing to SSE/AIC
    assert frg.n_effective == frf.n_effective - 3
    assert frg.missing_count == 3
    assert frg.n_calendar == N
    # first-step interval half width is ~0, and every future date matches
    half0 = (fcg.upper[0] - fcg.lower[0]) / 2
    assert half0 == pytest.approx(0.0, abs=1e-6)
    np.testing.assert_allclose(fcg.point, fcf.point, atol=1e-8)
    # point forecasts continue the clean series: first 12 weeks after the
    # last observed week are plain trend (phases 8..19, no spikes)
    t = np.arange(N, N + 12)
    expected = 200.0 + 0.5 * t
    np.testing.assert_allclose(fcg.point, expected, atol=1e-8)


def test_gappy_auto_select_picks_addadd_with_near_zero_sse(gappy_series):
    """Selection must not get pushed to the parameter boundaries."""
    sel = auto_select(gappy_series, M)
    best = sel.best
    assert (best.trend_kind, best.seasonal_kind) == ("add", "add")
    assert best.sse == pytest.approx(0.0, abs=1e-6)
    assert 0.01 < best.params.alpha < 0.99
    assert 0.01 < best.params.gamma < 0.99
    feasible = [s for s in sel.scores if s.feasible]
    assert len(feasible) == 6


def test_forecast_dates_are_calendar_weeks(gappy_series):
    p, _, _ = optimizer.optimize(gappy_series, "add", "add", M)
    fr = fit_hw(gappy_series, "add", "add", M, p)
    fc = forecast(fr.final_state, 12, fr.residuals, level=0.95)
    last = D0 + timedelta(weeks=N - 1)
    want = [(last + timedelta(weeks=k)).isoformat() for k in range(1, 13)]
    assert len(fc.point) == 12
    assert want[0] == "2025-02-24" and want[-1] == "2025-05-12"


# ------------------------------------------------------------------ backtest
def test_backtest_pinned_numbers_full(full_series):
    bt = run_backtest(full_series, M, origin_start=140, horizon=8, stride=4,
                      trend_kind="add", seasonal_kind="add")
    assert [o.origin for o in bt.origins] == [140, 144, 148, 152, 156]
    assert bt.model["mae"] == pytest.approx(0.0, abs=1e-8)
    assert bt.naive["mae"] == pytest.approx(26.0, abs=1e-8)
    assert bt.naive["mase"] == pytest.approx(1.0, abs=1e-8)


def test_backtest_pinned_numbers_gappy(full_series, gappy_series):
    """The report's gappy backtest: origins and naive MAE restored."""
    bt = run_backtest(gappy_series, M, origin_start=140, horizon=8, stride=4,
                      trend_kind="add", seasonal_kind="add")
    bt_full = run_backtest(full_series, M, origin_start=140, horizon=8,
                           stride=4, trend_kind="add", seasonal_kind="add")
    # calendar origins identical (no longer 4 vs 5 origins)
    assert [o.origin for o in bt.origins] == [140, 144, 148, 152, 156]
    assert bt.skipped_origins == []
    assert bt.model["mae"] == pytest.approx(0.0, abs=1e-8)
    assert bt.naive["mae"] == pytest.approx(26.0, abs=1e-8)
    assert bt.model["mase"] == pytest.approx(0.0, abs=1e-8)
    assert bt.naive["mase"] == pytest.approx(1.0, abs=1e-8)
    # per-origin forecasts identical to the full series on the same dates
    for rg, rf in zip(bt.origins, bt_full.origins):
        np.testing.assert_allclose(rg.forecast, rf.forecast, atol=1e-8)
        assert rg.naive_forecast[0] == pytest.approx(rf.naive_forecast[0])


def test_naive_uses_calendar_year_ago(gappy_series):
    """Seasonal naive at origin 140 copies position 88 (calendar -52 weeks),
    not a position shifted by the 3-week gap."""
    bt = run_backtest(gappy_series, M, origin_start=140, horizon=8, stride=99,
                      trend_kind="add", seasonal_kind="add")
    row = bt.origins[0]
    np.testing.assert_allclose(row.naive_forecast, gappy_series[88:96])


# ------------------------------------------------- gap inside initial 2m window
def test_gap_in_first_two_seasons_is_fit(full_series):
    """Gaps in year 1/2 that avoid the spike phases: the initial state is
    estimated by OLS over observed weeks and the noiseless series is still
    reconstructed almost exactly."""
    y = full_series.copy()
    y[60] = np.nan          # year-2 phase 8 only: year 1 still complete
    p, _, _ = optimizer.optimize(y, "add", "add", M)
    fr = fit_hw(y, "add", "add", M, p)
    fc = forecast(fr.final_state, 12, fr.residuals)
    assert fr.initial_window_complete is False
    assert fr.initial_rule == "saturated"
    assert fr.sse < 0.5
    t = np.arange(N, N + 12)
    # one missing year-2 cell slightly perturbs the OLS slope; the level
    # error stays bounded by one week's trend increment
    np.testing.assert_allclose(fc.point, 200.0 + 0.5 * t, atol=3.0)


def test_gap_on_spike_phase_uses_pooled_year2(full_series):
    """If a spike phase is missing in year 1, its initial slot has to borrow
    year-2 evidence (pooled).  The fit still works; only the first-season
    seasonal slot is approximate until recursion corrects it."""
    y = full_series.copy()
    y[48] = np.nan          # +120 spike phase, year 1 only
    p, _, _ = optimizer.optimize(y, "add", "add", M)
    fr = fit_hw(y, "add", "add", M, p)
    forecast(fr.final_state, 12, fr.residuals)
    assert fr.initial_rule == "pooled"
    # well within the magnitude of the missing spike (120), and far better
    # than the 36141 SSE the old observed-array code produced
    assert fr.sse < 2000


def test_unobserved_seasonal_phase_in_window_is_rejected(full_series):
    y = full_series.copy()
    y[7] = np.nan
    y[7 + M] = np.nan       # phase 7 never observed in either year
    with pytest.raises(ModelError, match="相位"):
        fit_hw(y, "add", "add", M, HWParams(0.3, 0.1, 0.3))


def test_too_few_window_observations_is_rejected(full_series):
    y = full_series.copy()
    y[: 2 * M - 1] = np.nan  # only one observed week in the window
    with pytest.raises(ModelError):
        fit_hw(y, "add", "add", M, HWParams(0.3, 0.1, 0.3))


def test_no_observed_weeks_after_window_is_rejected(full_series):
    y = full_series.copy()
    y[2 * M:] = np.nan
    with pytest.raises(ModelError, match="一步误差"):
        fit_hw(y, "add", "add", M, HWParams(0.3, 0.1, 0.3))


# ----------------------------------------------------------- all missing later
def test_trailing_gap_forecast_matches_full(full_series):
    """A renovation right at the end: last 3 weeks missing.  Forecast starts
    from the calendar week after the gap and still matches the full series."""
    y = full_series.copy()
    y[N - 3:] = np.nan
    p, _, _ = optimizer.optimize(y, "add", "add", M)
    frg = fit_hw(y, "add", "add", M, p)
    fcg = forecast(frg.final_state, 12, frg.residuals)
    pf, _, _ = optimizer.optimize(full_series, "add", "add", M)
    frf = fit_hw(full_series, "add", "add", M, pf)
    fcf = forecast(frf.final_state, 12, frf.residuals)
    np.testing.assert_allclose(fcg.point, fcf.point, atol=1e-6)


def test_backtest_skips_origin_whose_holdout_is_all_missing(full_series):
    y = full_series.copy()
    y[140:148] = np.nan     # first origin's whole holdout window is missing
    bt = run_backtest(y, M, origin_start=140, horizon=8, stride=4,
                      trend_kind="add", seasonal_kind="add")
    assert bt.skipped_origins == [140]
    assert [o.origin for o in bt.origins] == [144, 148, 152, 156]
    assert bt.naive["mae"] == pytest.approx(26.0, abs=1e-8)


def test_inf_is_still_rejected():
    y = np.arange(4 * M, dtype=float)
    y[10] = np.inf
    with pytest.raises(ModelError):
        fit_hw(y, "add", "add", M, HWParams(0.3, 0.1, 0.3))

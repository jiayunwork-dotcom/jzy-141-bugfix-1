"""Regression tests for missing-week handling (store closures / data outages).

These pin the exact before/after numbers reported on the clean synthetic
beverage series: weekly Mondays from 2022-01-03, 164 weeks,

    y_t = 200 + 0.5 t   (+120 in week-of-year slot 48, +60 in slot 20),

no noise.  Removing 2024-02-26 / 2024-03-04 / 2024-03-11 must leave the
dense-grid fit, forecasts and backtest metrics identical to the full series
-- missing weeks are propagated without a state update, never imputed.
"""
from __future__ import annotations

import time
from datetime import date, timedelta

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.gaps import build_dense_grid, future_grid_dates
from app.kernels import fit_hw, forecast, optimizer, ModelError
from app.kernels.backtest import run_backtest

D0 = date(2022, 1, 3)
N = 164
M = 52
GAP_DATES = [date(2024, 2, 26), date(2024, 3, 4), date(2024, 3, 11)]


@pytest.fixture(scope="module")
def series():
    t = np.arange(N, dtype=float)
    y = 200.0 + 0.5 * t
    y[t % M == 48] += 120.0
    y[t % M == 20] += 60.0
    dates = [D0 + timedelta(weeks=int(i)) for i in range(N)]
    return dates, y


@pytest.fixture(scope="module")
def gappy(series):
    dates, y = series
    cut = set(GAP_DATES)
    pairs = [(d, float(v)) for d, v in zip(dates, y) if d not in cut]
    obs_dates = [p[0] for p in pairs]
    obs_values = [p[1] for p in pairs]
    grid_dates, y_dense, mask = build_dense_grid(obs_dates, obs_values)
    return grid_dates, y_dense, mask, obs_dates, obs_values


def _fit(y, mask=None):
    params, sse, aic = optimizer.optimize(y, "add", "add", M, mask=mask)
    return fit_hw(y, "add", "add", M, params, mask=mask), params


# ---------------------------------------------------------------------------
# 1. Full series: the documented baseline numbers never change.
# ---------------------------------------------------------------------------
def test_full_series_baseline(series):
    _, y = series
    fr, p = _fit(y)
    assert fr.sse == pytest.approx(0.0, abs=1e-12)
    assert fr.n_effective == N - 2 * M
    assert (p.alpha, p.beta, p.gamma) == pytest.approx(
        (0.5871, 0.1507, 0.3004), abs=2e-3
    )
    fc = forecast(fr.final_state, 12, fr.residuals, level=0.95)
    half = (np.asarray(fc.upper) - np.asarray(fc.lower)) / 2
    assert half[0] == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# 2. Gaps expand back onto the same 164-week calendar grid.
# ---------------------------------------------------------------------------
def test_dense_grid_restores_calendar(gappy, series):
    grid_dates, y_dense, mask, obs_dates, obs_values = gappy
    dates, _ = series
    assert grid_dates == dates
    assert y_dense.shape == (N,)
    assert mask.sum() == N - 3
    for d in GAP_DATES:
        i = dates.index(d)
        assert mask[i] is False or mask[i] == np.bool_(False)
        assert np.isnan(y_dense[i])
    # observed values survive unchanged
    assert list(obs_dates) == [d for d in dates if d not in set(GAP_DATES)]
    np.testing.assert_allclose(
        y_dense[mask], [v for d, v in zip(dates, series[1])
                        if d not in set(GAP_DATES)]
    )


# ---------------------------------------------------------------------------
# 3. The reported failure numbers are gone: SSE ~ 0, interior params,
#    zero-width first interval.
# ---------------------------------------------------------------------------
def test_gappy_fit_matches_full(series, gappy):
    _, y_full = series
    grid_dates, y_dense, mask, _, _ = gappy

    fr_f, p_f = _fit(y_full)
    fr_g, p_g = _fit(y_dense, mask=mask)

    # the broken run gave SSE 36141.2 and alpha=beta=0.001, gamma=0.999
    assert fr_g.sse == pytest.approx(0.0, abs=1e-10)
    assert p_g.alpha == pytest.approx(p_f.alpha, abs=1e-9)
    assert p_g.beta == pytest.approx(p_f.beta, abs=1e-9)
    assert p_g.gamma == pytest.approx(p_f.gamma, abs=1e-9)
    assert p_g.alpha > 0.1 and p_g.beta > 0.05 and p_g.gamma < 0.9

    # the three gap weeks contribute no residuals -> n is reduced by 3
    assert fr_g.n_effective == fr_f.n_effective - 3


def test_gappy_forecasts_match_full_date_by_date(series, gappy):
    _, y_full = series
    grid_dates, y_dense, mask, _, _ = gappy
    fr_f, _ = _fit(y_full)
    fr_g, _ = _fit(y_dense, mask=mask)

    fc_f = forecast(fr_f.final_state, 12, fr_f.residuals, level=0.95)
    fc_g = forecast(fr_g.final_state, 12, fr_g.residuals, level=0.95)

    # same calendar dates and same point forecasts on each date
    fd_f = [d.isoformat() for d in future_grid_dates(
        date(2022, 1, 3) + timedelta(weeks=N - 1), 12)]
    fd_g = [d.isoformat() for d in future_grid_dates(grid_dates[-1], 12)]
    assert fd_f == fd_g
    np.testing.assert_allclose(fc_g.point, fc_f.point, atol=1e-8)

    # interval half-widths match too (and the first one is ~0, not 49.73)
    half_f = (np.asarray(fc_f.upper) - np.asarray(fc_f.lower)) / 2
    half_g = (np.asarray(fc_g.upper) - np.asarray(fc_g.lower)) / 2
    np.testing.assert_allclose(half_g, half_f, atol=1e-8)
    assert half_g[0] == pytest.approx(0.0, abs=1e-10)

    # and the forecast is the exact deterministic continuation
    t_future = np.arange(N, N + 12)
    expected = 200.0 + 0.5 * t_future
    np.testing.assert_allclose(fc_g.point, expected, atol=1e-8)


def test_gappy_fitted_aligned_to_calendar(series, gappy):
    dates, y_full = series
    grid_dates, y_dense, mask, _, _ = gappy
    fr_g, _ = _fit(y_dense, mask=mask)
    # all_fitted has one entry per grid week; gap positions are NaN, and the
    # fitted value at an observed week matches the full-series one-step fit
    fr_f, _ = _fit(y_full)
    assert fr_g.all_fitted.shape == (N,)
    gap_idx = [dates.index(d) for d in GAP_DATES]
    for i in gap_idx:
        assert np.isnan(fr_g.all_fitted[i])
    obs_tail = np.array([i for i in range(2 * M, N) if mask[i]])
    np.testing.assert_allclose(
        fr_g.all_fitted[obs_tail], fr_f.all_fitted[obs_tail], atol=1e-8
    )


# ---------------------------------------------------------------------------
# 4. Backtest: seasonal naive copies the actual CALENDAR week one year ago,
#    so its MAE stays exactly 26 (a full year of 0.5/week trend) and HW 0.
# ---------------------------------------------------------------------------
def test_backtest_full_series(series):
    _, y = series
    bt = run_backtest(y, M, origin_start=140, horizon=8, stride=4,
                      trend_kind="add", seasonal_kind="add")
    assert [o.origin for o in bt.origins] == [140, 144, 148, 152, 156]
    assert bt.model["mae"] == pytest.approx(0.0, abs=1e-10)
    assert bt.naive["mae"] == pytest.approx(26.0, abs=1e-10)
    assert bt.naive["mase"] == pytest.approx(1.0, abs=1e-10)


def test_backtest_gappy_series(series, gappy):
    dates, y_full = series
    grid_dates, y_dense, mask, _, _ = gappy
    grid_iso = [d.isoformat() for d in grid_dates]
    bt = run_backtest(y_dense, M, origin_start=140, horizon=8, stride=4,
                      trend_kind="add", seasonal_kind="add", mask=mask,
                      grid_dates=grid_iso)

    # broken run gave model MAE 16.7 / naive MAE 39.06; both must be back
    assert [o.origin for o in bt.origins] == [140, 144, 148, 152, 156]
    assert bt.model["mae"] == pytest.approx(0.0, abs=1e-10)
    assert bt.naive["mae"] == pytest.approx(26.0, abs=1e-10)
    assert bt.naive["mase"] == pytest.approx(1.0, abs=1e-10)

    # each naive forecast really is the value 52 calendar weeks earlier and
    # the origin label is a real calendar date (not a compressed index)
    for row in bt.origins:
        k = row.origin
        for j, nv in enumerate(row.naive_forecast):
            assert nv == pytest.approx(y_full[k + j - M])
        assert row.origin_date == dates[k].isoformat()
        assert row.skipped_naive == 0


# ---------------------------------------------------------------------------
# 5. Gaps inside the first two complete seasons: explicit rejection with a
#    reason (either it is computable or we say exactly what is missing).
# ---------------------------------------------------------------------------
def test_gap_inside_first_two_seasons_rejected(series):
    # note: dropping the very first observed row is a left-truncated series
    # (the grid starts at the next observed week), not an internal gap --
    # only missing weeks *between* observed rows can be detected.
    dates, y = series
    for gap_t in (1, 30, 2 * M - 1):
        pairs = [(d, float(v)) for d, v in zip(dates, y)
                 if d != dates[gap_t]]
        _, yd, mask = build_dense_grid([p[0] for p in pairs],
                                       [p[1] for p in pairs])
        assert len([p for p in pairs]) == N - 1
        with pytest.raises(ModelError, match="两个完整季节"):
            optimizer.optimize(yd, "add", "add", M, mask=mask)


def test_gap_exactly_at_recursion_start_is_fine(series):
    # index 2m is already outside the init window
    dates, y = series
    pairs = [(d, float(v)) for d, v in zip(dates, y)
             if d != dates[2 * M]]
    _, yd, mask = build_dense_grid([p[0] for p in pairs],
                                   [p[1] for p in pairs])
    params, sse, _ = optimizer.optimize(yd, "add", "add", M, mask=mask)
    assert sse == pytest.approx(0.0, abs=1e-10)


def test_backtest_gap_in_holdout_skips_origin(series):
    dates, y = series
    cut = {dates[150], dates[151]}
    pairs = [(d, float(v)) for d, v in zip(dates, y) if d not in cut]
    gd, yd, mask = build_dense_grid([p[0] for p in pairs],
                                    [p[1] for p in pairs])
    bt = run_backtest(yd, M, origin_start=140, horizon=8, stride=4,
                      trend_kind="add", seasonal_kind="add", mask=mask,
                      grid_dates=[d.isoformat() for d in gd])
    # origins 144 and 148 see the gap inside their holdout windows
    assert [o.origin for o in bt.origins] == [140, 152, 156]
    skipped = {s["origin"] for s in bt.skipped_origins}
    assert skipped == {144, 148}
    # the remaining origins still score perfectly
    assert bt.model["mae"] == pytest.approx(0.0, abs=1e-10)
    assert bt.naive["mae"] == pytest.approx(26.0, abs=1e-10)


def test_naive_undefined_when_year_ago_missing(series):
    # a gap exactly 52 calendar weeks before holdout step j=2 of origin 156:
    # the model still forecasts, but that naive point is excluded rather
    # than faked.  The gap stays outside the first two init seasons of the
    # origin-156 training prefix.
    dates, y = series
    origin = 156
    lag_missing = origin + 2 - M  # = 106 >= 2m = 104
    pairs = [(d, float(v)) for d, v in zip(dates, y)
             if d != dates[lag_missing]]
    gd, yd, mask = build_dense_grid([p[0] for p in pairs],
                                    [p[1] for p in pairs])
    bt = run_backtest(yd, M, origin_start=origin, horizon=8, stride=4,
                      trend_kind="add", seasonal_kind="add", mask=mask,
                      grid_dates=[d.isoformat() for d in gd])
    assert len(bt.origins) == 1
    row0 = bt.origins[0]
    assert row0.naive_forecast[2] is None
    assert row0.skipped_naive == 1
    assert row0.forecast[2] == pytest.approx(y[origin + 2], abs=1e-8)


# ---------------------------------------------------------------------------
# 6. End-to-end through the API + SQLite.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def client():
    from app.db import init_db, wait_for_db
    from app.jobs import job_manager
    from app.main import app

    wait_for_db()
    init_db()
    with TestClient(app) as c:
        yield c, job_manager


def _wait_job(c, job_id, timeout=120):
    for _ in range(int(timeout / 0.2)):
        j = c.get(f"/api/jobs/{job_id}").json()
        if j["status"] in ("done", "error"):
            return j
        time.sleep(0.2)
    raise AssertionError("job did not finish")


def _csv(series, drop=()):
    dates, y = series
    drop = set(drop)
    lines = ["周起始日期,销量"]
    for d, v in zip(dates, y):
        if d in drop:
            continue
        lines.append(f"{d.isoformat()},{v:.6f}")
    return "\n".join(lines)


def test_api_gappy_fit_and_backtest(client, series):
    c, _ = client
    r = c.post(
        "/api/series/upload",
        files={"file": ("g.csv", _csv(series, GAP_DATES), "text/csv")},
        data={"name": "缺周对照", "period": str(M)},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    sid = body["id"]
    assert body["missing_dates"] == [d.isoformat() for d in GAP_DATES]
    assert len(body["grid_dates"]) == N
    assert body["grid_dates"][0] == D0.isoformat()

    r = c.post("/api/fits", json={
        "series_id": sid, "horizon": 12, "confidence": 0.95,
        "interval_method": "analytic", "auto": False,
        "trend_kind": "add", "seasonal_kind": "add", "locks": {},
    })
    job = _wait_job(c, r.json()["id"])
    assert job["status"] == "done", job
    fit = c.get(f"/api/fits/{job['created_id']}").json()
    assert fit["results_version"] == 2
    assert fit["legacy"] is False
    assert fit["sse"] == pytest.approx(0.0, abs=1e-8)
    assert fit["n_effective"] == N - 3 - 2 * M
    assert len(fit["fitted"]) == N
    assert fit["fitted"][series[0].index(GAP_DATES[0])] is None
    assert fit["forecast"]["future_dates"][0] == "2025-02-24"
    half = [(u - lo) / 2 for lo, u in zip(fit["forecast"]["lower"],
                                          fit["forecast"]["upper"])]
    assert half[0] == pytest.approx(0.0, abs=1e-8)
    t_future = np.arange(N, N + 12)
    np.testing.assert_allclose(
        fit["forecast"]["point"], 200.0 + 0.5 * t_future, atol=1e-6
    )

    r = c.post("/api/backtests", json={
        "series_id": sid, "origin_start": 140, "horizon": 8, "stride": 4,
        "confidence": 0.95, "interval_method": "analytic", "auto": False,
        "trend_kind": "add", "seasonal_kind": "add", "locks": {},
    })
    job = _wait_job(c, r.json()["id"])
    assert job["status"] == "done", job
    bt = c.get(f"/api/backtests/{job['created_id']}").json()
    assert bt["legacy"] is False
    res = bt["result"]
    assert res["model"]["mae"] == pytest.approx(0.0, abs=1e-8)
    assert res["naive"]["mae"] == pytest.approx(26.0, abs=1e-8)
    assert len(res["grid_dates"]) == N


def test_api_early_gap_gives_clear_error(client, series):
    c, _ = client
    r = c.post(
        "/api/series/upload",
        files={"file": ("e.csv", _csv(series, [series[0][30]]),
                        "text/csv")},
        data={"name": "早期缺周", "period": str(M)},
    )
    sid = r.json()["id"]
    r = c.post("/api/fits", json={
        "series_id": sid, "horizon": 12, "confidence": 0.95,
        "interval_method": "analytic", "auto": False,
        "trend_kind": "add", "seasonal_kind": "add", "locks": {},
    })
    job = _wait_job(c, r.json()["id"])
    assert job["status"] == "error"
    assert "两个完整季节" in (job["error"] or "")


def test_legacy_stored_results_are_flagged(client, series):
    """Rows written before the fix must not look as trustworthy as new ones."""
    from app.db import session_scope
    from app import storage

    c, _ = client
    r = c.post(
        "/api/series/upload",
        files={"file": ("old.csv", _csv(series, GAP_DATES), "text/csv")},
        data={"name": "老库序列", "period": str(M)},
    )
    sid = r.json()["id"]

    # Simulate a pre-fix stored fit (no results_version / grid_dates) and a
    # pre-fix stored backtest (no results_version inside result JSON).
    with session_scope() as db:
        storage.create_fit(db, sid, {
            "label": "老拟合", "auto": False,
            "trend_kind": "add", "seasonal_kind": "add", "period": M,
            "params": {"alpha": 0.001, "beta": 0.001, "gamma": 0.999,
                       "phi": 1.0},
            "locks": {}, "sse": 36141.2, "aic": -1.0,
            "residuals": [], "fitted": [],
            "forecast": {"point": [], "lower": [], "upper": [], "level": 0.95,
                         "method": "analytic", "residual_std": 12.0,
                         "horizon": 12, "future_dates": []},
            "initial_state": {"level": 0.0, "trend": 0.0, "season": []},
            "final_state": {"level": 0.0, "trend": 0.0, "season": [],
                            "trend_kind": "add", "seasonal_kind": "add",
                            "phi": 1.0},
            "scores": [],
        })
        storage.create_backtest(db, sid, {
            "label": "老回测", "origin_start": 140, "horizon": 8,
            "stride": 4, "confidence": 0.95, "interval_method": "analytic",
            "auto": False, "trend_kind": "add", "seasonal_kind": "add",
            "locks": {},
            "result": {"period": M, "horizon": 8, "origins": [],
                       "model": {"mae": 16.7}, "naive": {"mae": 39.06},
                       "model_kind": {"trend_kind": "add",
                                      "seasonal_kind": "add"}},
        })

    fits = c.get(f"/api/fits/by-series/{sid}").json()
    legacy_fit = next(f for f in fits if f["label"] == "老拟合")
    assert legacy_fit["legacy"] is True
    assert "缺周" in legacy_fit["legacy_reason"]
    assert legacy_fit["results_version"] is None

    bts = c.get(f"/api/backtests/by-series/{sid}").json()
    legacy_bt = next(b for b in bts if b["label"] == "老回测")
    assert legacy_bt["legacy"] is True
    assert legacy_bt["legacy_reason"]

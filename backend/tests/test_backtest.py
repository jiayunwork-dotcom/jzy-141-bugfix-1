"""Backtest sanity: rolling origins, metrics and naive benchmark."""
from __future__ import annotations

import numpy as np

from app.kernels.backtest import run_backtest
from conftest import SEAS


def test_backtest_metrics_and_naive(additive_series, period):
    y = additive_series
    origin = 2 * period + 2
    h = 6
    bt = run_backtest(y, period, origin_start=origin, horizon=h, stride=2,
                      trend_kind="add", seasonal_kind="add")
    expected_origins = len(list(range(origin, y.size - h + 1, 2)))
    assert len(bt.origins) == expected_origins

    # Noiseless series: both HW and seasonal naive should be near-perfect for
    # 1..h (naive ignores trend, so it errs by h*slope; HW should win).
    assert bt.model["mae"] < bt.naive["mae"]
    for row in bt.origins:
        assert row.train_size <= row.origin
        assert len(row.forecast) == h
        assert np.isfinite(row.mae) and np.isfinite(row.mase)
        # naive forecast equals the value one season earlier
        assert row.naive_forecast[0] == y[row.origin - period]


def test_backtest_origin_uses_only_past(additive_series, period):
    y = additive_series
    origin = 2 * period + 3
    h = 5
    bt1 = run_backtest(y, period, origin_start=origin, horizon=h,
                       trend_kind="add", seasonal_kind="add")
    contaminated = y.copy()
    contaminated[origin + 1 :] += 9e5
    bt2 = run_backtest(contaminated, period, origin_start=origin,
                       horizon=h, trend_kind="add", seasonal_kind="add")
    np.testing.assert_allclose(bt2.origins[0].forecast,
                               bt1.origins[0].forecast, atol=1e-8)

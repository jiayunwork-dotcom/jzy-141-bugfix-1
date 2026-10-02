"""Dense weekly-grid helpers for series with missing weeks.

Uploaded series are sparse: only observed ``(date, value)`` pairs are stored.
The modelling kernels, however, need every calendar position to exist so that

* the seasonal slot ``t % m`` matches the actual calendar week (a store
  closure must not shift seasonality), and
* origins / seasonal-naive lags / forecast dates refer to calendar weeks.

``build_dense_grid`` expands the observed pairs onto a contiguous Monday
grid from the first to the last observed date.  Missing weeks get NaN values
and a ``False`` mask entry; no value is ever fabricated for them.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import List, Sequence, Tuple

import numpy as np

WEEK = timedelta(weeks=1)


def build_dense_grid(
    dates: Sequence[date], values: Sequence[float]
) -> Tuple[List[date], np.ndarray, np.ndarray]:
    """Expand sparse weekly observations onto a dense 7-day grid.

    Returns ``(grid_dates, grid_values, mask)`` with

    * ``grid_dates[i] = dates[0] + i weeks`` for ``i = 0..n_grid-1``;
    * ``grid_values[i]`` the observed value or NaN when the week is missing;
    * ``mask[i]`` True exactly at observed weeks.

    The input must already be sorted, weekly-aligned and duplicate-free (the
    upload layer guarantees this via ``check_weekly_continuity``).
    """
    if not dates:
        raise ValueError("空序列无法构建周网格。")
    by_date = {d: float(v) for d, v in zip(dates, values)}
    start = dates[0]
    end = dates[-1]
    n = int((end - start).days // 7) + 1
    grid_dates = [start + timedelta(weeks=i) for i in range(n)]
    grid_values = np.full(n, np.nan, dtype=float)
    mask = np.zeros(n, dtype=bool)
    for i, d in enumerate(grid_dates):
        if d in by_date:
            grid_values[i] = by_date[d]
            mask[i] = True
    return grid_dates, grid_values, mask


def future_grid_dates(last_grid_date: date, horizon: int) -> List[date]:
    """Calendar dates of the 1..horizon weeks after the grid end."""
    return [last_grid_date + timedelta(weeks=k)
            for k in range(1, horizon + 1)]

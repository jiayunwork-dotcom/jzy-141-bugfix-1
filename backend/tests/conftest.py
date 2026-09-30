"""pytest fixtures: deterministic synthetic series + API DB environment."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

# Must be set before any `app.*` module import (engine is module-level).
_TMP = Path(tempfile.mkdtemp(prefix="hw_api_"))
os.environ.setdefault("DATABASE_URL", f"sqlite+pysqlite:///{_TMP / 'api.db'}")
os.environ.setdefault("JOB_WORKERS", "2")

import numpy as np
import pytest

PERIOD = 12
# A zero-sum seasonal profile, asymmetric on purpose.
RAW_SEAS = np.array(
    [1.2, -0.5, 0.3, 2.0, -1.1, 0.8, -0.7, 0.1, -0.3, 1.5, -2.2, 0.9]
)
SEAS = RAW_SEAS - RAW_SEAS.mean()


@pytest.fixture
def period() -> int:
    return PERIOD


@pytest.fixture
def additive_series():
    """Pure linear trend + additive seasonality, no noise."""
    n = 3 * PERIOD + 8
    t = np.arange(n)
    return 10.0 + 0.35 * t + SEAS[t % PERIOD]


@pytest.fixture
def multiplicative_series():
    """Multiplicative seasonal pattern around a positive trend."""
    n = 3 * PERIOD + 8
    t = np.arange(n)
    pattern = np.array(
        [1.20, 0.90, 1.05, 1.30, 0.82, 1.10, 0.95, 1.00,
         0.92, 1.25, 0.70, 1.06]
    )
    pattern = pattern / pattern.mean()
    return (100.0 + 2.0 * t) * pattern[t % PERIOD]

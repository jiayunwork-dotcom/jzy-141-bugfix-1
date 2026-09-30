"""End-to-end API smoke tests on an in-process SQLite database.

Exercises the full FastAPI stack: upload -> background fit job ->
persisted result -> backtest, plus rejection / missing-week behaviour.
"""
from __future__ import annotations

import time
from datetime import date, timedelta

import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from app.db import init_db, wait_for_db
    from app.jobs import job_manager
    from app.main import app

    wait_for_db()
    init_db()
    with TestClient(app) as c:
        yield c, job_manager


def _make_csv(n: int = 3 * 12 + 10, period: int = 12) -> str:
    seas = np.array([1.2, -0.5, 0.3, 2.0, -1.1, 0.8, -0.7, 0.1,
                     -0.3, 1.5, -2.2, 0.9])
    seas = seas - seas.mean()
    lines = ["周起始日期,销量"]
    d0 = date(2023, 1, 2)
    for t in range(n):
        y = 100 + 2 * t + 18 * seas[t % period] + np.sin(t) * 0.3
        lines.append(f"{(d0 + timedelta(weeks=t)).isoformat()},{y:.3f}")
    return "\n".join(lines)


def _wait_job(c, job_id, timeout=90):
    for _ in range(int(timeout / 0.2)):
        j = c.get(f"/api/jobs/{job_id}").json()
        if j["status"] in ("done", "error"):
            return j
        time.sleep(0.2)
    raise AssertionError("job did not finish")


def test_upload_fit_backtest_flow(client):
    c, _ = client
    r = c.post(
        "/api/series/upload",
        files={"file": ("s.csv", _make_csv(), "text/csv")},
        data={"name": "饮料", "period": "12"},
    )
    assert r.status_code == 200, r.text
    sid = r.json()["id"]
    assert r.json()["missing_dates"] == []

    r = c.post("/api/fits", json={
        "series_id": sid, "horizon": 8, "confidence": 0.95,
        "interval_method": "analytic", "auto": True, "locks": {},
    })
    assert r.status_code == 200, r.text
    job = _wait_job(c, r.json()["id"])
    assert job["status"] == "done", job
    fit_id = job["created_id"]

    fit = c.get(f"/api/fits/{fit_id}").json()
    assert len(fit["forecast"]["point"]) == 8
    widths = np.array(fit["forecast"]["upper"]) - np.array(
        fit["forecast"]["lower"]
    )
    assert np.all(np.diff(widths) >= -1e-9)
    assert len(fit["scores"]) == 6
    # residuals are persisted
    assert len(fit["residuals"]) > 0

    r = c.post("/api/backtests", json={
        "series_id": sid, "origin_start": 2 * 12 + 2, "horizon": 6,
        "stride": 2, "confidence": 0.95, "interval_method": "analytic",
        "auto": True, "locks": {},
    })
    job = _wait_job(c, r.json()["id"])
    assert job["status"] == "done", job
    bt = c.get(f"/api/backtests/{job['created_id']}").json()
    assert len(bt["result"]["origins"]) >= 1
    row0 = bt["result"]["origins"][0]
    assert row0["train_size"] == 2 * 12 + 2
    assert np.isfinite(row0["mase"])


def test_missing_week_upload_is_flagged(client):
    c, _ = client
    csv_text = (
        "周起始日期,销量\n2026-01-05,10\n2026-01-19,12\n2026-01-26,14"
    )
    r = c.post(
        "/api/series/upload",
        files={"file": ("g.csv", csv_text, "text/csv")},
        data={"name": "缺周", "period": "12"},
    )
    assert r.status_code == 200
    assert r.json()["missing_dates"] == ["2026-01-12"]


def test_multiplicative_rejected_for_nonpositive(client):
    c, _ = client
    n = 3 * 12 + 4
    d0 = date(2023, 1, 2)
    dates = [(d0 + timedelta(weeks=i)).isoformat() for i in range(n)]
    values = [float(i + 2) for i in range(n)]
    values[5] = 0.0
    r = c.post("/api/series", json={
        "name": "nonpos", "period": 12, "dates": dates, "values": values,
    })
    assert r.status_code == 200, r.text
    sid = r.json()["id"]

    r = c.post("/api/fits", json={
        "series_id": sid, "horizon": 4, "confidence": 0.95,
        "interval_method": "analytic", "auto": False,
        "trend_kind": "add", "seasonal_kind": "mul", "locks": {},
    })
    job = _wait_job(c, r.json()["id"])
    assert job["status"] == "error"
    assert "正" in (job["error"] or "")

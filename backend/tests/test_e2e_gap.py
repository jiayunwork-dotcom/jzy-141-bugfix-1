"""End-to-end: gappy upload -> fit -> forecast dates align -> backtest."""
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
    wait_for_db(); init_db()
    with TestClient(app) as c:
        yield c, job_manager


M, N = 52, 164
D0 = date(2022, 1, 3)


def _csv(delete=()):
    t = np.arange(N)
    y = 200.0 + 0.5*t
    y[t % M == 48] += 120
    y[t % M == 20] += 60
    lines = ["周起始日期,销量"]
    for i in range(N):
        if i in delete:
            continue
        d = D0 + timedelta(weeks=i)
        lines.append(f"{d.isoformat()},{y[i]:.3f}")
    return "\n".join(lines)


def _wait(c, jid, timeout=120):
    for _ in range(int(timeout/0.2)):
        j = c.get(f"/api/jobs/{jid}").json()
        if j["status"] in ("done", "error"):
            return j
        time.sleep(0.2)
    raise AssertionError("timeout")


def test_gappy_full_flow(client):
    c, _ = client
    # upload gappy series
    r = c.post("/api/series/upload",
               files={"file": ("g.csv", _csv({112,113,114}), "text/csv")},
               data={"name": "停业三周", "period": "52"})
    assert r.status_code == 200, r.text
    sid = r.json()["id"]
    assert r.json()["missing_dates"] == ["2024-02-26", "2024-03-04",
                                         "2024-03-11"]

    # manual fixed add/add fit
    r = c.post("/api/fits", json={"series_id": sid, "horizon": 12,
               "confidence": 0.95, "interval_method": "analytic",
               "auto": False, "trend_kind": "add", "seasonal_kind": "add",
               "locks": {}})
    j = _wait(c, r.json()["id"])
    assert j["status"] == "done", j
    fit = c.get(f"/api/fits/{j['created_id']}").json()
    assert fit["engine_version"] == "2.0-calendar-grid"
    assert fit["missing_count"] == 3
    assert fit["n_calendar"] == 164
    assert fit["n_effective"] == 57
    assert fit["sse"] == pytest.approx(0.0, abs=1e-6)
    # fitted array aligns to grid dates (null at missing weeks)
    assert len(fit["fitted"]) == 164
    assert fit["fitted"][112] is None and fit["fitted"][113] is None
    assert fit["grid_dates"][112] == "2024-02-26"
    fc = fit["forecast"]
    assert fc["future_dates"][0] == "2025-02-24"
    assert fc["point"][0] == pytest.approx(282.0, abs=1e-6)
    half0 = (fc["upper"][0]-fc["lower"][0])/2
    assert half0 == pytest.approx(0.0, abs=1e-6)

    # backtest pinned numbers
    r = c.post("/api/backtests", json={"series_id": sid, "origin_start": 140,
               "horizon": 8, "stride": 4, "confidence": 0.95,
               "interval_method": "analytic", "auto": False,
               "trend_kind": "add", "seasonal_kind": "add", "locks": {}})
    j = _wait(c, r.json()["id"])
    assert j["status"] == "done", j
    bt = c.get(f"/api/backtests/{j['created_id']}").json()
    assert bt["engine_version"] == "2.0-calendar-grid"
    res = bt["result"]
    assert [o["origin"] for o in res["origins"]] == [140,144,148,152,156]
    assert res["model"]["mae"] == pytest.approx(0.0, abs=1e-6)
    assert res["naive"]["mae"] == pytest.approx(26.0, abs=1e-6)
    assert res["naive"]["mase"] == pytest.approx(1.0, abs=1e-6)


def test_full_series_unchanged(client):
    c, _ = client
    r = c.post("/api/series/upload",
               files={"file": ("f.csv", _csv(), "text/csv")},
               data={"name": "完整对照", "period": "52"})
    sid = r.json()["id"]
    r = c.post("/api/fits", json={"series_id": sid, "horizon": 12,
               "confidence": 0.95, "interval_method": "analytic",
               "auto": False, "trend_kind": "add", "seasonal_kind": "add",
               "locks": {}})
    j = _wait(c, r.json()["id"])
    fit = c.get(f"/api/fits/{j['created_id']}").json()
    p = fit["params"]
    assert p["alpha"] == pytest.approx(0.5871, abs=2e-3)
    assert p["beta"] == pytest.approx(0.1507, abs=2e-3)
    assert p["gamma"] == pytest.approx(0.3004, abs=2e-3)
    assert fit["sse"] == pytest.approx(0.0, abs=1e-8)
    assert fit["n_effective"] == 60

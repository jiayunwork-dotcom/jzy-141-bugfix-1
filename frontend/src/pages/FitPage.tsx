import { useCallback, useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { apiClient } from "../api/client";
import type { FitResult, Job, Series } from "../api/types";
import JobProgress from "../components/JobProgress";
import ParamPanel from "../components/ParamPanel";
import Scorecard from "../components/Scorecard";
import SeriesChart from "../components/SeriesChart";
import { useJob } from "../hooks/useJob";
import { fmtNumber } from "../utils/format";

const TREND_KIND_LABEL: Record<string, string> = {
  none: "无趋势",
  add: "加法趋势",
  add_damped: "阻尼趋势",
};

export default function FitPage() {
  const { seriesId } = useParams();
  const id = Number(seriesId);
  const [series, setSeries] = useState<Series | null>(null);
  const [fits, setFits] = useState<FitResult[]>([]);
  const [selectedFitId, setSelectedFitId] = useState<number | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [formError, setFormError] = useState<string | null>(null);

  const [auto, setAuto] = useState(true);
  const [trendKind, setTrendKind] = useState("add");
  const [seasonalKind, setSeasonalKind] = useState("add");
  const [horizon, setHorizon] = useState(12);
  const [confidence, setConfidence] = useState(0.95);
  const [intervalMethod, setIntervalMethod] = useState<
    "analytic" | "simulate"
  >("analytic");
  const [locks, setLocks] = useState<Record<string, number>>({});

  const loadAll = useCallback(async () => {
    const [s, f] = await Promise.all([
      apiClient.getSeries(id),
      apiClient.listFits(id),
    ]);
    setSeries(s);
    setFits(f);
    setSelectedFitId((cur) => cur ?? f[0]?.id ?? null);
  }, [id]);

  useEffect(() => {
    loadAll().catch((e) =>
      setFormError(e instanceof Error ? e.message : String(e))
    );
  }, [loadAll]);

  const onDone = useCallback(
    async (job: Job) => {
      if (job.status === "done") {
        await loadAll();
        if (job.created_id) setSelectedFitId(job.created_id);
      }
    },
    [loadAll]
  );
  const { job } = useJob(jobId, { onDone });

  const selectedFit = useMemo(
    () => fits.find((f) => f.id === selectedFitId) ?? null,
    [fits, selectedFitId]
  );

  function toggleLock(name: string, value: number) {
    setLocks((prev) => {
      const next = { ...prev };
      if (name in next) delete next[name];
      else next[name] = value;
      return next;
    });
  }

  function updateLock(name: string, value: number) {
    setLocks((prev) => ({ ...prev, [name]: value }));
  }

  async function submit() {
    setFormError(null);
    try {
      const j = await apiClient.startFit({
        series_id: id,
        horizon,
        confidence,
        interval_method: intervalMethod,
        auto,
        trend_kind: auto ? null : trendKind,
        seasonal_kind: auto ? null : seasonalKind,
        locks,
      });
      setJobId(j.id);
    } catch (e) {
      setFormError(e instanceof Error ? e.message : String(e));
    }
  }

  if (!series) {
    return <div className="muted">加载中… {formError}</div>;
  }

  return (
    <div>
      <h1 className="page-title">
        拟合与预测 · {series.name}{" "}
        <Link to="/series" className="muted" style={{ fontSize: 13 }}>
          ← 返回列表
        </Link>
      </h1>

      {series.missing_dates.length > 0 && (
        <div className="alert warn">
          该序列有 {series.missing_dates.length} 个缺周标记，拟合前请确认数据已补齐。
        </div>
      )}
      {series.values.some((v) => v <= 0) && (
        <div className="alert warn">
          序列含 0 或负值，乘法季节组合会被后端拒绝并给出原因。
        </div>
      )}
      {formError && <div className="alert error">{formError}</div>}

      <ParamPanel
        auto={auto}
        setAuto={setAuto}
        trendKind={trendKind}
        setTrendKind={setTrendKind}
        seasonalKind={seasonalKind}
        setSeasonalKind={setSeasonalKind}
        horizon={horizon}
        setHorizon={setHorizon}
        confidence={confidence}
        setConfidence={setConfidence}
        intervalMethod={intervalMethod}
        setIntervalMethod={setIntervalMethod}
        locks={locks}
        toggleLock={toggleLock}
        updateLock={updateLock}
        fittedParams={selectedFit?.params}
        onSubmit={submit}
        busy={job?.status === "running" || job?.status === "pending"}
      />

      {jobId && <JobProgress job={job} />}

      {selectedFit && (
        <>
          <div className="panel">
            <h2>拟合统计</h2>
            <div className="kpi-grid">
              <div className="kpi">
                <div className="k-label">模型</div>
                <div className="k-value" style={{ fontSize: 14 }}>
                  {TREND_KIND_LABEL[selectedFit.trend_kind]} +{" "}
                  {selectedFit.seasonal_kind === "add" ? "加法季节" : "乘法季节"}
                </div>
              </div>
              <div className="kpi">
                <div className="k-label">SSE</div>
                <div className="k-value">{fmtNumber(selectedFit.sse, 4)}</div>
              </div>
              <div className="kpi">
                <div className="k-label">AIC</div>
                <div className="k-value">{fmtNumber(selectedFit.aic, 3)}</div>
              </div>
              <div className="kpi">
                <div className="k-label">残差标准差</div>
                <div className="k-value">
                  {fmtNumber(selectedFit.forecast.residual_std, 3)}
                </div>
              </div>
              <div className="kpi">
                <div className="k-label">α / β / γ / φ</div>
                <div className="k-value" style={{ fontSize: 13 }}>
                  {selectedFit.params.alpha.toFixed(3)} /{" "}
                  {selectedFit.params.beta.toFixed(3)} /{" "}
                  {selectedFit.params.gamma.toFixed(3)} /{" "}
                  {selectedFit.params.phi.toFixed(3)}
                </div>
              </div>
            </div>
          </div>

          <div className="panel">
            <h2>原始值 / 拟合 / 预测带</h2>
            <SeriesChart series={series} fit={selectedFit} />
          </div>

          {selectedFit.scores.length > 0 && (
            <Scorecard scores={selectedFit.scores} />
          )}
        </>
      )}

      <div className="panel">
        <h2>历史拟合（同一序列保留多次，可比较）</h2>
        {fits.length === 0 ? (
          <div className="muted">尚无拟合结果。</div>
        ) : (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th></th>
                  <th>时间</th>
                  <th>备注</th>
                  <th>模型</th>
                  <th>AIC</th>
                  <th>SSE</th>
                  <th>α</th>
                  <th>β</th>
                  <th>γ</th>
                  <th>φ</th>
                </tr>
              </thead>
              <tbody>
                {fits.map((f) => (
                  <tr key={f.id}>
                    <td>
                      <input
                        type="radio"
                        name="fitpick"
                        checked={f.id === selectedFitId}
                        onChange={() => setSelectedFitId(f.id)}
                      />
                    </td>
                    <td>{f.created_at.replace("T", " ").slice(0, 19)}</td>
                    <td>{f.label || "—"}</td>
                    <td>
                      {TREND_KIND_LABEL[f.trend_kind]} /{" "}
                      {f.seasonal_kind === "add" ? "加法" : "乘法"}
                    </td>
                    <td>{fmtNumber(f.aic, 2)}</td>
                    <td>{fmtNumber(f.sse, 3)}</td>
                    <td>{f.params.alpha.toFixed(3)}</td>
                    <td>{f.params.beta.toFixed(3)}</td>
                    <td>{f.params.gamma.toFixed(3)}</td>
                    <td>{f.params.phi.toFixed(3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

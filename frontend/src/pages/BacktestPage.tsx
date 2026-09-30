import { useCallback, useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { apiClient } from "../api/client";
import type { BacktestResult, Job, Series } from "../api/types";
import JobProgress from "../components/JobProgress";
import { useJob } from "../hooks/useJob";
import { fmtNumber, fmtPct } from "../utils/format";

export default function BacktestPage() {
  const { seriesId } = useParams();
  const id = Number(seriesId);
  const [series, setSeries] = useState<Series | null>(null);
  const [backtests, setBacktests] = useState<BacktestResult[]>([]);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const period = series?.period ?? 52;
  const [originStart, setOriginStart] = useState(2 * period);
  const [horizon, setHorizon] = useState(8);
  const [stride, setStride] = useState(1);
  const [confidence, setConfidence] = useState(0.95);
  const [intervalMethod, setIntervalMethod] = useState<"analytic" | "simulate">(
    "analytic"
  );
  const [auto, setAuto] = useState(true);
  const [trendKind, setTrendKind] = useState("add");
  const [seasonalKind, setSeasonalKind] = useState("add");

  const load = useCallback(async () => {
    const [s, bts] = await Promise.all([
      apiClient.getSeries(id),
      apiClient.listBacktests(id),
    ]);
    setSeries(s);
    setOriginStart(2 * s.period);
    setBacktests(bts);
    setSelectedId((cur) => cur ?? bts[0]?.id ?? null);
  }, [id]);

  useEffect(() => {
    load().catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, [load]);

  const onDone = useCallback(
    async (job: Job) => {
      if (job.status === "done") {
        await load();
        if (job.created_id) setSelectedId(job.created_id);
      }
    },
    [load]
  );
  const { job } = useJob(jobId, { onDone });

  const selected = useMemo(
    () => backtests.find((b) => b.id === selectedId) ?? null,
    [backtests, selectedId]
  );

  async function start() {
    setError(null);
    try {
      const j = await apiClient.startBacktest({
        series_id: id,
        origin_start: originStart,
        horizon,
        stride,
        confidence,
        interval_method: intervalMethod,
        auto,
        trend_kind: auto ? null : trendKind,
        seasonal_kind: auto ? null : seasonalKind,
        locks: {},
      });
      setJobId(j.id);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  if (!series) return <div className="muted">加载中…</div>;
  const minOrigin = 2 * series.period;

  return (
    <div>
      <h1 className="page-title">
        滚动原点回测 · {series.name}{" "}
        <Link to="/series" className="muted" style={{ fontSize: 13 }}>
          ← 返回列表
        </Link>
      </h1>

      {error && <div className="alert error">{error}</div>}

      <div className="panel">
        <h2>回测设置</h2>
        <div className="row">
          <div className="field">
            <label>首个原点（至少 {minOrigin}）</label>
            <input
              type="number"
              min={minOrigin}
              max={series.dates.length - horizon}
              value={originStart}
              onChange={(e) => setOriginStart(Number(e.target.value))}
            />
          </div>
          <div className="field">
            <label>预测步长 h</label>
            <input
              type="number"
              min={1}
              value={horizon}
              onChange={(e) => setHorizon(Number(e.target.value))}
            />
          </div>
          <div className="field">
            <label>原点步长</label>
            <input
              type="number"
              min={1}
              value={stride}
              onChange={(e) => setStride(Number(e.target.value))}
            />
          </div>
          <div className="field">
            <label>置信水平</label>
            <select
              value={confidence}
              onChange={(e) => setConfidence(Number(e.target.value))}
            >
              <option value={0.8}>80%</option>
              <option value={0.9}>90%</option>
              <option value={0.95}>95%</option>
              <option value={0.99}>99%</option>
            </select>
          </div>
          <div className="field">
            <label>区间方法</label>
            <select
              value={intervalMethod}
              onChange={(e) =>
                setIntervalMethod(e.target.value as "analytic" | "simulate")
              }
            >
              <option value="analytic">解析近似</option>
              <option value="simulate">模拟</option>
            </select>
          </div>
          <div className="field">
            <label>模型</label>
            <select
              value={auto ? "auto" : "manual"}
              onChange={(e) => setAuto(e.target.value === "auto")}
            >
              <option value="auto">每个原点自动选型</option>
              <option value="manual">固定组合</option>
            </select>
          </div>
          {!auto && (
            <>
              <div className="field">
                <label>趋势</label>
                <select
                  value={trendKind}
                  onChange={(e) => setTrendKind(e.target.value)}
                >
                  <option value="none">无</option>
                  <option value="add">加法</option>
                  <option value="add_damped">加法阻尼</option>
                </select>
              </div>
              <div className="field">
                <label>季节</label>
                <select
                  value={seasonalKind}
                  onChange={(e) => setSeasonalKind(e.target.value)}
                >
                  <option value="add">加法</option>
                  <option value="mul">乘法</option>
                </select>
              </div>
            </>
          )}
          <button
            onClick={start}
            disabled={job?.status === "running" || job?.status === "pending"}
          >
            开始回测
          </button>
        </div>
        <div className="muted" style={{ marginTop: 8 }}>
          每个原点只用截至该周的数据重新拟合，再预测未来 {horizon} 周；
          基准为季节朴素法（照抄上一季节同期）。
        </div>
      </div>

      {jobId && <JobProgress job={job} />}

      {selected && (
        <>
          <div className="panel">
            <h2>
              汇总指标 · Holt–Winters vs 季节朴素（模型：
              {selected.result.model_kind.trend_kind}/
              {selected.result.model_kind.seasonal_kind}）
            </h2>
            <table className="compare-table">
              <thead>
                <tr>
                  <th>方法</th>
                  <th>MAE</th>
                  <th>MAPE</th>
                  <th>MASE</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>Holt–Winters</td>
                  <td className={winnerCell(selected, "mae")}>
                    {fmtNumber(selected.result.model.mae, 4)}
                  </td>
                  <td className={winnerCell(selected, "mape")}>
                    {fmtPct(selected.result.model.mape, 2)}
                  </td>
                  <td className={winnerCell(selected, "mase")}>
                    {fmtNumber(selected.result.model.mase, 4)}
                  </td>
                </tr>
                <tr>
                  <td>季节朴素</td>
                  <td className={winnerCell(selected, "mae", true)}>
                    {fmtNumber(selected.result.naive.mae, 4)}
                  </td>
                  <td className={winnerCell(selected, "mape", true)}>
                    {fmtPct(selected.result.naive.mape, 2)}
                  </td>
                  <td className={winnerCell(selected, "mase", true)}>
                    {fmtNumber(selected.result.naive.mase, 4)}
                  </td>
                </tr>
              </tbody>
            </table>
            <div className="muted" style={{ marginTop: 8 }}>
              MASE &lt; 1 表示模型优于该原点训练段上的季节朴素尺度。
            </div>
          </div>

          <div className="panel">
            <h2>各原点误差明细</h2>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>原点序号</th>
                    <th>对应周</th>
                    <th>训练样本数</th>
                    <th>HW MAE</th>
                    <th>朴素 MAE</th>
                    <th>HW MAPE</th>
                    <th>朴素 MAPE</th>
                    <th>HW MASE</th>
                    <th>朴素 MASE</th>
                    <th>AIC</th>
                  </tr>
                </thead>
                <tbody>
                  {selected.result.origins.map((o, idx) => (
                    <tr key={o.origin}>
                      <td>{idx + 1}</td>
                      <td>{series.dates[o.origin]}</td>
                      <td>{o.train_size}</td>
                      <td>{fmtNumber(o.mae, 3)}</td>
                      <td>{fmtNumber(o.naive_mae, 3)}</td>
                      <td>{fmtPct(o.mape, 1)}</td>
                      <td>{fmtPct(o.naive_mape, 1)}</td>
                      <td>{fmtNumber(o.mase, 3)}</td>
                      <td>{fmtNumber(o.naive_mase, 3)}</td>
                      <td>{fmtNumber(o.aic, 1)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <OriginDetailPanel bt={selected} series={series} />
        </>
      )}

      <div className="panel">
        <h2>历史回测</h2>
        {backtests.length === 0 ? (
          <div className="muted">尚无回测结果。</div>
        ) : (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th></th>
                  <th>时间</th>
                  <th>起点</th>
                  <th>h</th>
                  <th>步长</th>
                  <th>HW MAE</th>
                  <th>朴素 MAE</th>
                  <th>HW MASE</th>
                  <th>朴素 MASE</th>
                </tr>
              </thead>
              <tbody>
                {backtests.map((b) => (
                  <tr key={b.id}>
                    <td>
                      <input
                        type="radio"
                        checked={b.id === selectedId}
                        onChange={() => setSelectedId(b.id)}
                      />
                    </td>
                    <td>{b.created_at.replace("T", " ").slice(0, 19)}</td>
                    <td>{b.origin_start}</td>
                    <td>{b.horizon}</td>
                    <td>{b.stride}</td>
                    <td>{fmtNumber(b.result.model.mae, 3)}</td>
                    <td>{fmtNumber(b.result.naive.mae, 3)}</td>
                    <td>{fmtNumber(b.result.model.mase, 3)}</td>
                    <td>{fmtNumber(b.result.naive.mase, 3)}</td>
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

function winnerCell(
  b: BacktestResult,
  metric: "mae" | "mape" | "mase",
  naive = false
): string {
  const m = b.result.model[metric];
  const n = b.result.naive[metric];
  if (!Number.isFinite(m) || !Number.isFinite(n)) return "";
  const modelWins = m < n;
  return (modelWins && !naive) || (!modelWins && naive) ? "winner" : "";
}

function OriginDetailPanel({
  bt,
  series,
}: {
  bt: BacktestResult;
  series: Series;
}) {
  const [idx, setIdx] = useState(0);
  const row = bt.result.origins[Math.min(idx, bt.result.origins.length - 1)];
  if (!row) return null;
  return (
    <div className="panel">
      <h2>单原点预测对照</h2>
      <div className="row" style={{ marginBottom: 10 }}>
        <div className="field">
          <label>选择原点</label>
          <select value={idx} onChange={(e) => setIdx(Number(e.target.value))}>
            {bt.result.origins.map((o, i) => (
              <option key={o.origin} value={i}>
                #{i + 1} · {series.dates[o.origin]}
              </option>
            ))}
          </select>
        </div>
      </div>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>步长</th>
              <th>周</th>
              <th>实际</th>
              <th>HW 预测</th>
              <th>HW 误差</th>
              <th>朴素预测</th>
              <th>朴素误差</th>
            </tr>
          </thead>
          <tbody>
            {row.forecast.map((f, j) => (
              <tr key={j}>
                <td>{j + 1}</td>
                <td>{series.dates[row.origin + j]}</td>
                <td>{fmtNumber(row.actual[j], 2)}</td>
                <td>{fmtNumber(f, 2)}</td>
                <td>{fmtNumber(row.errors[j], 2)}</td>
                <td>{fmtNumber(row.naive_forecast[j], 2)}</td>
                <td>{fmtNumber(row.naive_errors[j], 2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

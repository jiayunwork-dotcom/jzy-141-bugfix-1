import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { apiClient } from "../api/client";
import type { Series } from "../api/types";
import UploadPanel from "../components/UploadPanel";
import { fmtNumber } from "../utils/format";

export default function SeriesListPage() {
  const [series, setSeries] = useState<Series[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  async function load() {
    setLoading(true);
    try {
      setSeries(await apiClient.listSeries());
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load();
  }, []);

  return (
    <div>
      <h1 className="page-title">周销量序列</h1>
      <UploadPanel onCreated={() => load()} />
      <div className="panel">
        <h2>已选序列（{series.length}）</h2>
        {error && <div className="alert error">{error}</div>}
        {loading ? (
          <div className="muted">加载中…</div>
        ) : series.length === 0 ? (
          <div className="muted">还没有序列，请先上传 CSV。</div>
        ) : (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>名称</th>
                  <th>周数</th>
                  <th>周期</th>
                  <th>起始周</th>
                  <th>结束周</th>
                  <th>均值</th>
                  <th>缺周</th>
                  <th>操作</th>
                </tr>
              </thead>
              <tbody>
                {series.map((s) => {
                  const mean =
                    s.values.reduce((a, b) => a + b, 0) / s.values.length;
                  return (
                    <tr key={s.id}>
                      <td>{s.name}</td>
                      <td>{s.dates.length}</td>
                      <td>{s.period}</td>
                      <td>{s.dates[0]}</td>
                      <td>{s.dates[s.dates.length - 1]}</td>
                      <td>{fmtNumber(mean, 2)}</td>
                      <td>
                        {s.missing_dates.length ? (
                          <span className="badge red">
                            {s.missing_dates.length} 个
                          </span>
                        ) : (
                          <span className="badge green">连续</span>
                        )}
                      </td>
                      <td>
                        <Link to={`/series/${s.id}/fit`}>拟合 / 预测</Link>
                        {" · "}
                        <Link to={`/series/${s.id}/backtest`}>回测</Link>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

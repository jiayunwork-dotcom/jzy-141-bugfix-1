import type { CandidateScore } from "../api/types";
import { fmtNumber } from "../utils/format";

const TREND_LABEL: Record<string, string> = {
  none: "无趋势",
  add: "加法趋势",
  add_damped: "加法趋势（阻尼）",
};

export default function Scorecard({ scores }: { scores: CandidateScore[] }) {
  if (!scores.length) return null;
  const feasible = scores.filter((s) => s.feasible);
  const bestAic = feasible.length ? Math.min(...feasible.map((s) => s.aic!)) : null;
  return (
    <div className="panel">
      <h2>自动选型评分（AIC 越小越优）</h2>
      <table>
        <thead>
          <tr>
            <th>趋势</th>
            <th>季节</th>
            <th>状态</th>
            <th>AIC</th>
            <th>SSE</th>
            <th>α</th>
            <th>β</th>
            <th>γ</th>
            <th>φ</th>
            <th>说明</th>
          </tr>
        </thead>
        <tbody>
          {scores.map((s) => (
            <tr
              key={`${s.trend_kind}-${s.seasonal_kind}`}
              style={s.aic === bestAic ? { background: "#eef8f2" } : undefined}
            >
              <td>{TREND_LABEL[s.trend_kind]}</td>
              <td>{s.seasonal_kind === "add" ? "加法" : "乘法"}</td>
              <td>
                {s.feasible ? (
                  <span className="badge green">可行</span>
                ) : (
                  <span className="badge red">拒绝</span>
                )}
              </td>
              <td>{s.aic === null ? "—" : fmtNumber(s.aic, 3)}</td>
              <td>{s.sse === null ? "—" : fmtNumber(s.sse, 4)}</td>
              <td>{s.params?.alpha.toFixed(4) ?? "—"}</td>
              <td>{s.params?.beta.toFixed(4) ?? "—"}</td>
              <td>{s.params?.gamma.toFixed(4) ?? "—"}</td>
              <td>{s.params?.phi.toFixed(4) ?? "—"}</td>
              <td className="muted">{s.reason ?? (s.aic === bestAic ? "★ 最优" : "")}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

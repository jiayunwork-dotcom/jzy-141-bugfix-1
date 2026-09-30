import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { FitResult, Series } from "../api/types";

interface Props {
  series: Series;
  fit: FitResult | null;
}

interface ChartDatum {
  date: string;
  actual?: number;
  fitted?: number;
  forecast?: number;
  lower?: number;
  upper?: number;
}

export default function SeriesChart({ series, fit }: Props) {
  const data: ChartDatum[] = series.dates.map((date, i) => ({
    date,
    actual: series.values[i],
    fitted: fit ? fit.fitted[i] : undefined,
  }));

  if (fit) {
    const fc = fit.forecast;
    fc.future_dates.forEach((date, i) => {
      data.push({
        date,
        forecast: fc.point[i],
        lower: fc.lower[i],
        upper: fc.upper[i],
      });
    });
  }

  return (
    <div>
      <ResponsiveContainer width="100%" height={380}>
        <ComposedChart data={data} margin={{ top: 8, right: 18, bottom: 8, left: 4 }}>
          <CartesianGrid stroke="#eef1f5" />
          <XAxis dataKey="date" minTickGap={48} fontSize={11} />
          <YAxis fontSize={11} domain={["auto", "auto"]} />
          <Tooltip />
          <Area
            type="monotone"
            dataKey="upper"
            name={`预测区间 ${fit ? Math.round(fit.forecast.level * 100) : 95}%`}
            stroke="#9fceb5"
            fill="#1e7f4f"
            fillOpacity={0.12}
            isAnimationActive={false}
            activeDot={false}
            dot={false}
          />
          <Area
            type="monotone"
            dataKey="lower"
            name="区间下限"
            stroke="#9fceb5"
            fill="#f4f6f9"
            fillOpacity={1}
            isAnimationActive={false}
            activeDot={false}
            dot={false}
          />
          <Line
            type="monotone"
            dataKey="actual"
            name="原始销量"
            stroke="#33415c"
            dot={false}
            strokeWidth={1.6}
            isAnimationActive={false}
          />
          <Line
            type="monotone"
            dataKey="fitted"
            name="拟合值"
            stroke="#1f6feb"
            dot={false}
            strokeWidth={1.4}
            isAnimationActive={false}
          />
          <Line
            type="monotone"
            dataKey="forecast"
            name="点预测"
            stroke="#1e7f4f"
            strokeDasharray="6 4"
            dot={false}
            strokeWidth={1.8}
            isAnimationActive={false}
          />
        </ComposedChart>
      </ResponsiveContainer>
      <div className="legend">
        <span style={{ color: "#33415c" }}>
          <span style={{ background: "#33415c" }} />
          原始值
        </span>
        <span style={{ color: "#1f6feb" }}>
          <span style={{ background: "#1f6feb" }} />
          拟合值
        </span>
        <span style={{ color: "#1e7f4f" }}>
          <span style={{ background: "#1e7f4f" }} />
          点预测（虚线）
        </span>
        {fit && <span>预测区间：{Math.round(fit.forecast.level * 100)}%（{fit.forecast.method === "analytic" ? "解析近似" : "模拟"}）</span>}
      </div>
      {fit && <ForecastBandTable data={data.slice(series.dates.length)} />}
    </div>
  );
}

function ForecastBandTable({ data }: { data: ChartDatum[] }) {
  return (
    <div className="table-wrap" style={{ marginTop: 12, maxHeight: 200 }}>
      <table>
        <thead>
          <tr>
            <th>周起始</th>
            <th>下限</th>
            <th>点预测</th>
            <th>上限</th>
          </tr>
        </thead>
        <tbody>
          {data.map((d) => (
            <tr key={d.date}>
              <td>{d.date}</td>
              <td>{d.lower?.toFixed(2)}</td>
              <td>{d.forecast?.toFixed(2)}</td>
              <td>{d.upper?.toFixed(2)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

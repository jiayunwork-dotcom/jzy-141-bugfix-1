import type { FitRequest } from "../api/types";

interface Props {
  auto: boolean;
  setAuto: (v: boolean) => void;
  trendKind: string;
  setTrendKind: (v: string) => void;
  seasonalKind: string;
  setSeasonalKind: (v: string) => void;
  horizon: number;
  setHorizon: (v: number) => void;
  confidence: number;
  setConfidence: (v: number) => void;
  intervalMethod: "analytic" | "simulate";
  setIntervalMethod: (v: "analytic" | "simulate") => void;
  locks: Record<string, number>;
  toggleLock: (name: string, value: number) => void;
  updateLock: (name: string, value: number) => void;
  fittedParams?: { alpha: number; beta: number; gamma: number; phi: number };
  onSubmit: () => void;
  busy: boolean;
}

const PARAM_META = [
  { key: "alpha", label: "α 水平平滑", min: 0.001, max: 0.999, step: 0.01 },
  { key: "beta", label: "β 趋势平滑", min: 0.001, max: 0.999, step: 0.01 },
  { key: "gamma", label: "γ 季节平滑", min: 0.001, max: 0.999, step: 0.01 },
  { key: "phi", label: "φ 阻尼系数", min: 0.8, max: 0.98, step: 0.01 },
];

export default function ParamPanel(props: Props) {
  const damped = props.trendKind === "add_damped";
  const noTrend = props.trendKind === "none";

  const visible = PARAM_META.filter(
    (p) => p.key !== "beta" || !noTrend
  ).filter((p) => p.key !== "phi" || damped);

  return (
    <div className="panel">
      <h2>模型与参数</h2>
      <div className="row" style={{ marginBottom: 14 }}>
        <div className="field">
          <label>选型方式</label>
          <select
            value={props.auto ? "auto" : "manual"}
            onChange={(e) => props.setAuto(e.target.value === "auto")}
          >
            <option value="auto">自动选型（按 AIC）</option>
            <option value="manual">手动指定组合</option>
          </select>
        </div>
        {!props.auto && (
          <>
            <div className="field">
              <label>趋势项</label>
              <select
                value={props.trendKind}
                onChange={(e) => props.setTrendKind(e.target.value)}
              >
                <option value="none">无趋势</option>
                <option value="add">加法趋势</option>
                <option value="add_damped">加法趋势（阻尼）</option>
              </select>
            </div>
            <div className="field">
              <label>季节项</label>
              <select
                value={props.seasonalKind}
                onChange={(e) => props.setSeasonalKind(e.target.value)}
              >
                <option value="add">加法季节</option>
                <option value="mul">乘法季节</option>
              </select>
            </div>
          </>
        )}
        <div className="field">
          <label>预测步长（周）</label>
          <input
            type="number"
            min={1}
            max={104}
            value={props.horizon}
            onChange={(e) => props.setHorizon(Number(e.target.value))}
          />
        </div>
        <div className="field">
          <label>置信水平</label>
          <select
            value={props.confidence}
            onChange={(e) => props.setConfidence(Number(e.target.value))}
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
            value={props.intervalMethod}
            onChange={(e) =>
              props.setIntervalMethod(e.target.value as "analytic" | "simulate")
            }
          >
            <option value="analytic">解析近似</option>
            <option value="simulate">模拟（残差自助）</option>
          </select>
        </div>
      </div>

      <div className="param-grid" style={{ marginBottom: 14 }}>
        {visible.map((p) => {
          const locked = p.key in props.locks;
          const value = locked
            ? props.locks[p.key]
            : props.fittedParams?.[p.key as keyof typeof props.fittedParams];
          return (
            <div className="param-cell" key={p.key}>
              <div className="lock-row">
                <label>
                  <input
                    type="checkbox"
                    checked={locked}
                    onChange={(e) =>
                      props.toggleLock(
                        p.key,
                        e.target.checked
                          ? (value as number) ?? (p.min + p.max) / 2
                          : 0
                      )
                    }
                  />{" "}
                  锁定 {p.label}
                </label>
                {locked ? (
                  <span className="badge red">已锁定</span>
                ) : (
                  <span className="badge gray">自动</span>
                )}
              </div>
              <input
                type="number"
                min={p.min}
                max={p.max}
                step={p.step}
                style={{ width: "100%" }}
                disabled={!locked}
                value={locked ? props.locks[p.key] : value ?? ""}
                onChange={(e) =>
                  props.updateLock(p.key, Number(e.target.value))
                }
              />
              {!locked && value !== undefined && (
                <div className="muted" style={{ marginTop: 4 }}>
                  上次拟合：{Number(value).toFixed(4)}
                </div>
              )}
            </div>
          );
        })}
      </div>

      <button onClick={props.onSubmit} disabled={props.busy}>
        {props.auto ? "自动选型并拟合" : "按指定模型拟合"}
      </button>
    </div>
  );
}

export function buildLocksFromPanel(
  locks: Record<string, number>
): Record<string, number> {
  return { ...locks };
}

export type { FitRequest };

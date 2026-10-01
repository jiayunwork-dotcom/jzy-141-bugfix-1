export interface Series {
  id: number;
  name: string;
  period: number;
  dates: string[];
  values: number[];
  missing_dates: string[];
  created_at: string;
}

export interface HWParams {
  alpha: number;
  beta: number;
  gamma: number;
  phi: number;
}

export interface CandidateScore {
  trend_kind: string;
  seasonal_kind: string;
  feasible: boolean;
  aic: number | null;
  sse: number | null;
  params: HWParams | null;
  reason: string | null;
}

export interface ForecastInfo {
  point: number[];
  lower: number[];
  upper: number[];
  level: number;
  method: string;
  residual_std: number;
  horizon: number;
  future_dates: string[];
}

export interface FitResult {
  id: number;
  series_id: number;
  created_at: string;
  label: string;
  auto: boolean;
  engine_version: string;
  trend_kind: string;
  seasonal_kind: string;
  period: number;
  params: HWParams;
  locks: Record<string, number>;
  sse: number;
  aic: number;
  n_effective: number | null;
  n_calendar: number | null;
  missing_count: number | null;
  residuals: number[];
  /** One entry per *calendar* position; null at missing weeks. */
  fitted: (number | null)[];
  /** Dates of the aligned calendar grid (includes missing weeks). */
  grid_dates: string[];
  forecast: ForecastInfo;
  initial_state: {
    level: number;
    trend: number | null;
    season: number[];
  };
  final_state: {
    level: number;
    trend: number | null;
    season: number[];
    trend_kind: string;
    seasonal_kind: string;
    phi: number;
  };
  scores: CandidateScore[];
}

export interface OriginRow {
  origin: number;
  train_size: number;
  forecast: (number | null)[];
  actual: (number | null)[];
  naive_forecast: (number | null)[];
  errors: (number | null)[];
  naive_errors: (number | null)[];
  mae: number;
  mape: number;
  mase: number;
  naive_mae: number;
  naive_mape: number;
  naive_mase: number;
  params: HWParams;
  aic: number;
  scale: number;
  scored_steps?: number;
  skipped?: boolean;
}

export interface BacktestResult {
  id: number;
  series_id: number;
  created_at: string;
  label: string;
  engine_version: string;
  origin_start: number;
  horizon: number;
  stride: number;
  confidence: number;
  interval_method: string;
  auto: boolean;
  trend_kind: string | null;
  seasonal_kind: string | null;
  locks: Record<string, number>;
  result: {
    period: number;
    horizon: number;
    origins: OriginRow[];
    model: { mae: number; mape: number; mase: number };
    naive: { mae: number; mape: number; mase: number };
    model_kind: { trend_kind: string; seasonal_kind: string };
    skipped_origins?: number[];
    calendar_size?: number;
    grid_dates?: string[];
  };
}

export interface Job {
  id: string;
  kind: "fit" | "backtest";
  status: "pending" | "running" | "done" | "error";
  progress: number;
  stage: string;
  result: { fit_id?: number; backtest_id?: number } | null;
  error: string | null;
  series_id: number | null;
  created_id: number | null;
}

export interface FitRequest {
  series_id: number;
  horizon: number;
  confidence: number;
  interval_method: "analytic" | "simulate";
  auto: boolean;
  trend_kind?: string | null;
  seasonal_kind?: string | null;
  locks: Record<string, number>;
  label?: string;
}

export interface BacktestRequest {
  series_id: number;
  origin_start: number;
  horizon: number;
  stride: number;
  confidence: number;
  interval_method: "analytic" | "simulate";
  auto: boolean;
  trend_kind?: string | null;
  seasonal_kind?: string | null;
  locks: Record<string, number>;
  label?: string;
}

export type Metrics = {
  n_scored: number;
  mae: number;
  rmse: number;
  mape: number;
  pinball: number;
  picp: number;
  interval_width: number;
};

export type ComparisonRow = Metrics & {
  model: string;
  id: string;
  backend: string;
};

export type SeriesPoint = {
  week: string;
  price: number | null;
  intensity: number;
  season: string;
  split: string;
  disruption: boolean;
};

export type LiveForecast = {
  crop: string;
  market: string;
  forecast_week: string;
  cultivation_intensity: number;
  predicted_price: number;
  lower_price: number;
  upper_price: number;
  coverage_level: number;
  commitment_source: string;
  model_version: string;
};

export type StudioData = {
  meta: {
    project: string;
    member: string;
    engine: string;
    version: string;
    generated_at: string;
    seed: number;
    confidence_level: number;
    commitment_source: string;
    panel_grain: string;
    elapsed_s: number;
    n_panel: number;
    n_scored_test: number;
    libraries: Record<string, string | null>;
  };
  coverage: {
    by_crop: Array<Record<string, string | number>>;
    by_series: Array<Record<string, string | number>>;
    by_year: Array<{ year: number; coverage_pct: number }>;
    exclusions: Array<{ item: string; reason: string }>;
  };
  split: Record<string, { n: number; start: string; end: string }>;
  dictionary: Array<{ column: string; dtype: string; role: string; notes: string }>;
  originMap: Record<string, Record<string, string[]>>;
  series: Array<{ crop: string; market: string; points: SeriesPoint[] }>;
  cobweb: Array<{
    crop: string;
    market: string;
    season_t: string;
    intensity_t: number;
    price_t1: number;
    season_t1: string;
  }>;
  comparison: ComparisonRow[];
  ablation: Record<
    string,
    { features: string[]; metrics: Metrics; by_crop: Record<string, Metrics> }
  >;
  weeks: Array<Metrics & { commitment_week: number; mean_intensity: number }>;
  importance: Array<{ feature: string; importance: number; std: number }>;
  live: LiveForecast[];
  models: Record<
    string,
    {
      backend: string;
      qhat: number | null;
      metrics: { overall: Metrics; by_crop: Record<string, Metrics>; by_market?: Record<string, Metrics>; by_season?: Record<string, Metrics> };
      train_curve: Array<Record<string, string | number | null>>;
      note?: string;
    }
  >;
};

// Types mirroring the FastAPI responses (see http://localhost:8000/docs).

export type Spacecraft = "SMAP" | "MSL";
export type Nullable<T> = T | null;

export interface LabelInterval {
  start: number;
  end: number;
  cls: string;
  merged_from?: number[][];
}

export interface ChannelRunRef {
  run_id: number;
  experiment_id: number;
  method: string;
  experiment_name: string;
  has_model: boolean;
}

export interface Channel {
  chan_id: string;
  spacecraft: Spacecraft;
  type_letter: string;
  train_len: number;
  test_len: number;
  n_features: number;
  n_intervals: number;
  labelled_points: number;
  label_fraction: number;
  value_min: number;
  value_max: number;
  train_std: number;
  commands_active_train: number;
  commands_active_test: number;
  label_rows: number;
  issues: string[];
  intervals: LabelInterval[];
  runs?: ChannelRunRef[];
}

export interface Telemetry {
  chan_id: string;
  spacecraft: Spacecraft;
  split: "train" | "test";
  n: number;
  start: number;
  end: number;
  index: number[];
  values: Nullable<number>[];
  command_activity: number[];
  n_features: number;
  labels: LabelInterval[];
  stats: { min: number; max: number; mean: number; std: number };
}

export interface MethodSpec {
  key: string;
  label: string;
  family: string;
  purpose: string;
  trainable: boolean;
  produces_prediction: boolean;
  defaults: Record<string, number>;
  profiles: Record<string, Record<string, number>>;
}

export interface MethodsResponse {
  eval_start: number;
  methods: MethodSpec[];
  threshold_defaults: Record<string, number | string | boolean>;
}

export interface EventMetrics {
  tp: number;
  fp: number;
  fn: number;
  n_labels: number;
  n_detections: number;
  precision: Nullable<number>;
  recall: Nullable<number>;
  f1: Nullable<number>;
  f0_5: Nullable<number>;
  false_alarms_per_10k: Nullable<number>;
  mean_delay: Nullable<number>;
  median_delay: Nullable<number>;
  mean_iou: Nullable<number>;
  macro_f1?: Nullable<number>;
  by_class: Record<string, { labels: number; detected: number; recall: Nullable<number> }>;
  per_label?: Array<{ start: number; end: number; cls: string; detected: boolean; delay?: number; iou?: number; coverage?: number }>;
  false_positive_intervals?: number[][];
}

export interface PointMetrics {
  tp: number;
  fp: number;
  fn: number;
  tn: number;
  precision: Nullable<number>;
  recall: Nullable<number>;
  f1: Nullable<number>;
  f0_5: Nullable<number>;
  tpr: Nullable<number>;
  fpr: Nullable<number>;
  n?: number;
  positives?: number;
}

export interface Curves {
  average_precision: number;
  roc_auc: number;
  prevalence: number;
  pr: { recall: number[]; precision: number[] };
  roc: { fpr: number[]; tpr: number[] };
}

export interface Aggregate {
  n_channels: number;
  n_evaluated: number;
  event: EventMetrics;
  point: PointMetrics;
  mean_average_precision: Nullable<number>;
}

export interface ExperimentMetrics {
  n_channels: number;
  n_completed: number;
  n_failed: number;
  overall: Nullable<Aggregate>;
  by_spacecraft: Record<string, Aggregate>;
  curves: Record<string, Nullable<Curves>>;
}

export interface Headline {
  event_precision: Nullable<number>;
  event_recall: Nullable<number>;
  event_f1: Nullable<number>;
  event_f0_5: Nullable<number>;
  point_f1: Nullable<number>;
  map: Nullable<number>;
  n_detections: Nullable<number>;
}

export type ExperimentStatus = "queued" | "running" | "completed" | "partial" | "failed" | "cancelled";

export interface RunSummary {
  id: number;
  experiment_id: number;
  chan_id: string;
  spacecraft: Nullable<Spacecraft>;
  status: string;
  epochs_run: Nullable<number>;
  best_epoch: Nullable<number>;
  best_val_loss: Nullable<number>;
  train_time_s: Nullable<number>;
  infer_time_s: Nullable<number>;
  n_detections: Nullable<number>;
  has_model: boolean;
  has_arrays: boolean;
  error: Nullable<string>;
  event: Partial<EventMetrics>;
  point: Partial<PointMetrics>;
  average_precision: Nullable<number>;
}

export interface Experiment {
  id: number;
  name: string;
  method: string;
  method_label: string;
  profile: Nullable<string>;
  spacecraft: string;
  channels: string[];
  n_channels: number;
  config: Record<string, number>;
  threshold_config: Record<string, number | string | boolean>;
  status: ExperimentStatus;
  progress: number;
  current_channel: Nullable<string>;
  message: Nullable<string>;
  created_at: Nullable<string>;
  started_at: Nullable<string>;
  finished_at: Nullable<string>;
  device: Nullable<string>;
  seed: number;
  git_commit: Nullable<string>;
  dataset_checksum: Nullable<string>;
  train_time_s: Nullable<number>;
  infer_time_s: Nullable<number>;
  origin: "local" | "reference";
  environment: Nullable<Record<string, unknown>>;
  notes: Nullable<string>;
  headline: Headline;
  metrics?: Nullable<ExperimentMetrics>;
  runs?: RunSummary[];
}

export interface DetectionRef {
  id: Nullable<number>;
  start: number;
  end: number;
  score: number;
  direction: string;
  outcome: "matches_label" | "no_label_overlap";
  threshold: number;
  peak_error: number;
}

export interface EpochLog {
  epoch: number;
  train_loss: number;
  val_loss: Nullable<number>;
  seconds: number;
}

export interface RunDetail extends RunSummary {
  metrics: Nullable<{
    eval_start: number;
    n_evaluated: number;
    unscoreable_label_points: number;
    event: EventMetrics;
    point: PointMetrics;
    curves: Nullable<Curves>;
  }>;
  method: string;
  experiment_name: string;
  config: Record<string, number>;
  threshold_config: Record<string, number | string | boolean>;
  epochs: EpochLog[];
  detections: Array<DetectionRef & { matched_labels: LabelInterval[] }>;
  model_file: Nullable<string>;
  model_available: boolean;
}

export interface Series {
  chan_id: string;
  n: number;
  eval_start: number;
  source: "stored" | "live_inference";
  actual: Nullable<number>[];
  prediction: Nullable<Nullable<number>[]>;
  raw_score: Nullable<number>[];
  smoothed: Nullable<number>[];
  threshold_high: Nullable<number>[];
  threshold_low: Nullable<number>[];
  command_activity: number[];
  detections: DetectionRef[];
  labels: LabelInterval[];
  metrics?: { event: EventMetrics; point: PointMetrics; unscoreable_label_points: number; n_evaluated: number };
  threshold_config?: Record<string, number | string | boolean>;
  effective_threshold?: Record<string, number | string>;
  timing_ms?: { load_model: number; score: number; threshold: number };
  device?: string;
}

export interface DetectionRow {
  id: number;
  experiment_id: number;
  experiment_name: string;
  method: string;
  method_label: string;
  run_id: number;
  chan_id: string;
  start: number;
  end: number;
  length: number;
  score: number;
  peak_error: number;
  threshold: number;
  direction: string;
  outcome: "matches_label" | "no_label_overlap";
  matched_labels: LabelInterval[];
}

export interface DetectionDetail {
  id: number;
  experiment_id: number;
  experiment_name: string;
  method: string;
  method_label: string;
  run_id: number;
  chan_id: string;
  spacecraft: Spacecraft;
  start: number;
  end: number;
  score: number;
  peak_error: number;
  threshold: number;
  direction: string;
  outcome: "matches_label" | "no_label_overlap";
  matched_labels: LabelInterval[];
  window: Record<string, unknown>;
  stats: Record<string, number | null>;
  context: {
    start: number;
    end: number;
    actual: Nullable<number>[];
    prediction: Nullable<Nullable<number>[]>;
    raw_score: Nullable<number>[] | null;
    smoothed: Nullable<number>[] | null;
    threshold_high: Nullable<number>[] | null;
    threshold_low: Nullable<number>[] | null;
  };
  labels_in_context: LabelInterval[];
  confidence_note: string;
}

export interface Overview {
  dataset: { available: boolean; source: Nullable<string>; retrieved_at: Nullable<string>; validation_ok: Nullable<boolean>; checksum: Nullable<string> };
  spacecraft: Record<string, { channels: number; test_values: number; train_values: number; label_sequences: number }>;
  experiments: { by_status: Record<string, number>; total: number };
  completed_channel_runs: number;
  channels_analysed: number;
  detections_total: number;
  recent_experiments: Experiment[];
  leaderboard: Experiment[];
}

export interface DatasetStatus {
  available: boolean;
  acquisition: { running: boolean; message: string; progress: number; result: Nullable<{ ok: boolean; source: Nullable<string>; message: string; attempts: Array<{ source: string; ok: boolean; message: string }> }> };
  snapshot: Nullable<{
    id: number;
    source: string;
    source_url: string;
    retrieved_at: string;
    dataset_checksum: string;
    labels_sha256: string;
    n_files: number;
    total_bytes: number;
    validation_ok: boolean;
    summary: {
      label_rows: number;
      label_sequences_raw: number;
      label_sequences_after_merge: number;
      unique_channels: number;
      by_spacecraft: Record<string, { channels: number; train_values: number; test_values: number; labelled_points: number; sequences: number; features: number[] }>;
    };
    errors: string[];
    warnings: string[];
    provenance: Record<string, string | null>;
  }>;
  sources: Array<{ key: string; title: string; url: string; note: string }>;
  disclosures: string[];
  license_note: string;
}

export interface SystemInfo {
  platform: string;
  python: string;
  torch: string;
  cuda_available: boolean;
  cuda_device: Nullable<string>;
  cpu_count: number;
  torch_threads: number;
  ram_total_gb: Nullable<number>;
  ram_available_gb: Nullable<number>;
  device_preference: string;
  recommendation: string;
}

export interface ReplaySession {
  session_id: string;
  run_id: number;
  experiment_id: number;
  chan_id: string;
  method: string;
  method_label: string;
  n: number;
  eval_start: number;
  threshold_mode: string;
  batch_size: Nullable<number>;
  window_batches: Nullable<number>;
  labels: LabelInterval[];
  produces_prediction: boolean;
  notice: string;
}

export interface ReplayStep {
  start: number;
  end: number;
  cursor: number;
  done: boolean;
  actual?: Nullable<number>[];
  prediction?: Nullable<Nullable<number>[]>;
  raw_score?: Nullable<number>[];
  smoothed?: Nullable<number>[];
  verdict?: { from: number; to: number; threshold_high: Nullable<number>[]; threshold_low: Nullable<number>[]; flags: number[] };
  intervals?: Array<{ start: number; end: number; score: number; direction: string; peak_error: number; threshold: number }>;
  compute_ms?: number;
}

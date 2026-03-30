/**
 * TypeScript types matching backend Pydantic schemas
 * Generated from FastAPI OpenAPI spec
 */

import { z } from 'zod';

// =============================================================================
// Shared schemas
// =============================================================================

export interface PaginationMeta {
  page: number;
  per_page: number;
  total: number;
  total_pages: number;
}

export interface APIResponse<T = unknown> {
  success: boolean;
  data: T | null;
  meta?: PaginationMeta | null;
  errors?: Array<{
    code: string;
    message: string;
    field?: string;
    details?: Record<string, unknown>;
  }>;
}

// =============================================================================
// User schemas
// =============================================================================

export interface User {
  id: string;
  email: string;
  name: string;
  role: 'viewer' | 'editor' | 'admin';
  is_active: boolean;
  created_at: string;
  updated_at: string;
}

export interface UserCreate {
  email: string;
  name: string;
  password: string;
  role?: string;
}

export interface UserUpdate {
  email?: string;
  name?: string;
  role?: string;
}

// =============================================================================
// Model schemas
// =============================================================================

export interface Model {
  id: string;
  name: string;
  checkpoint: string;
  model_type: 'llm' | 'vlm' | 'embedding';
  family?: string | null;
  parameters?: number | null;
  max_context_length?: number | null;
  architecture?: string | null;
  license_type?: string | null;
  hf_repo?: string | null;
  metadata: Record<string, unknown>;
  created_by?: string | null;
  created_at: string;
  updated_at: string;
}

export interface ModelCreate {
  name: string;
  checkpoint: string;
  model_type?: string;
  family?: string | null;
  parameters?: number | null;
  max_context_length?: number | null;
  architecture?: string | null;
  license_type?: string | null;
  hf_repo?: string | null;
  metadata?: Record<string, unknown>;
}

export interface ModelUpdate {
  name?: string;
  model_type?: string;
  family?: string | null;
  parameters?: number | null;
  max_context_length?: number | null;
  architecture?: string | null;
  license_type?: string | null;
  hf_repo?: string | null;
  metadata?: Record<string, unknown>;
}

export interface ModelListResponse {
  success: boolean;
  data: Model[];
  meta?: PaginationMeta | null;
}

// =============================================================================
// Run schemas
// =============================================================================

export type RunStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';

export type RunType = 'benchmark' | 'accuracy-mmlu' | 'accuracy-humaneval' | 'lm-eval' | 'perplexity';

export type DeviceType = 'cpu' | 'gpu' | 'npu' | 'hybrid' | 'igpu';

export type BackendType = 'llamacpp' | 'ort' | 'flm';

export type DtypeType = 'float32' | 'float16' | 'int4' | 'int8';

export interface Run {
  id: string;
  model_id: string;
  user_id?: string | null;
  build_name: string;
  run_type: RunType;
  status: RunStatus;
  status_message?: string | null;
  device?: string | null;
  backend?: string | null;
  dtype?: string | null;
  config: Record<string, unknown>;
  system_info: Record<string, unknown>;
  lemonade_version?: string | null;
  build_uid?: string | null;
  started_at?: string | null;
  completed_at?: string | null;
  duration_seconds?: number | null;
  log_file_path?: string | null;
  error_log?: string | null;
  created_at: string;
  updated_at: string;
}

export interface RunCreate {
  model_id: string;
  build_name: string;
  run_type: string;
  user_id?: string | null;
  status?: string;
  device?: string | null;
  backend?: string | null;
  dtype?: string | null;
  config?: Record<string, unknown>;
  system_info?: Record<string, unknown>;
  lemonade_version?: string | null;
  build_uid?: string | null;
}

export interface RunUpdate {
  status?: string;
  status_message?: string | null;
  device?: string | null;
  backend?: string | null;
  dtype?: string | null;
  config?: Record<string, unknown> | null;
  started_at?: string | null;
  completed_at?: string | null;
  duration_seconds?: number | null;
  system_info?: Record<string, unknown> | null;
  error_log?: string | null;
}

export interface RunListResponse {
  success: boolean;
  data: Run[];
  meta?: PaginationMeta | null;
}

export interface RunDetail extends Run {
  metrics?: Metric[];
}

// =============================================================================
// Metric schemas
// =============================================================================

export type MetricCategory = 'performance' | 'accuracy' | 'efficiency';

export interface Metric {
  id: string;
  run_id: string;
  category: MetricCategory;
  name: string;
  display_name?: string | null;
  value_numeric?: number | null;
  value_text?: string | null;
  unit?: string | null;
  mean_value?: number | null;
  std_dev?: number | null;
  min_value?: number | null;
  max_value?: number | null;
  iteration_values?: unknown[] | null;
  metadata: Record<string, unknown>;
  created_at: string;
}

export interface MetricCreate {
  run_id: string;
  category: string;
  name: string;
  display_name?: string | null;
  value_numeric?: number | null;
  value_text?: string | null;
  unit?: string | null;
  mean_value?: number | null;
  std_dev?: number | null;
  min_value?: number | null;
  max_value?: number | null;
  iteration_values?: unknown[] | null;
  metadata?: Record<string, unknown>;
}

export interface MetricListResponse {
  success: boolean;
  data: Metric[];
  meta?: PaginationMeta | null;
}

export interface MetricTrend {
  run_id: string;
  run_name: string;
  metric_name: string;
  value: number;
  created_at: string;
}

export interface MetricComparison {
  runs: Run[];
  metrics: Record<string, Metric[]>;
  statistics: Record<string, {
    mean: number;
    std_dev: number;
    min: number;
    max: number;
    best_run_id: string;
  }>;
}

// =============================================================================
// Import schemas
// =============================================================================

export interface ImportRequest {
  cache_dir: string;
  skip_duplicates?: boolean;
  dry_run?: boolean;
}

export interface ImportJobStatus {
  job_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  total_files: number;
  processed_files: number;
  imported_runs: number;
  skipped_duplicates: number;
  errors?: string[];
  error?: string;
  created_at?: string | null;
  completed_at?: string | null;
}

export interface ImportScanResult {
  cache_dir: string;
  files_found: number;
  files: string[];
}

// =============================================================================
// WebSocket schemas
// =============================================================================

export interface WSMessage {
  type: string;
  [key: string]: unknown;
}

export interface RunStatusEvent {
  event_type: 'run_status';
  run_id: string;
  status: RunStatus;
  message?: string | null;
  data?: Record<string, unknown>;
}

export interface MetricsStreamEvent {
  event_type: 'metrics_stream';
  run_id: string;
  metrics: Metric[];
}

export interface ProgressEvent {
  event_type: 'progress';
  run_id: string;
  progress: number;
  message?: string | null;
}

export type WSEvent = RunStatusEvent | MetricsStreamEvent | ProgressEvent;

// =============================================================================
// Dashboard/Stats schemas
// =============================================================================

export interface DashboardStats {
  total_models: number;
  total_runs: number;
  total_metrics: number;
  runs_by_status: Record<RunStatus, number>;
  runs_by_type: Record<string, number>;
  recent_runs: Run[];
}

export interface RunStats {
  total_runs: number;
  completed_runs: number;
  failed_runs: number;
  pending_runs: number;
  running_runs: number;
  avg_duration_seconds?: number | null;
}

// =============================================================================
// API Error types
// =============================================================================

export interface ApiError {
  message: string;
  status: number;
  code?: string;
  details?: Record<string, unknown>;
}

export class ApiErrorClass extends Error {
  status: number;
  code?: string;
  details?: Record<string, unknown>;

  constructor(message: string, status: number, code?: string, details?: Record<string, unknown>) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.code = code;
    this.details = details;
  }
}

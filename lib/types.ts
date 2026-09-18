/** Shapes returned by the studio API (ai-services: pipeline/api + studio/api). */

export type JobKind = "scrape" | "label" | "train" | "eval" | "publish";
export type JobStatus = "queued" | "running" | "succeeded" | "failed" | "cancelled";

export interface StageRun {
  stage: string;
  status: string;
  started_at: string;
  finished_at: string | null;
  stats: Record<string, unknown>;
  error: string | null;
}

export interface JobRun {
  id: string;
  kind: JobKind;
  status: JobStatus;
  params: Record<string, unknown>;
  result: Record<string, unknown> | null;
  error: string | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  stages?: StageRun[] | null;
}

export interface ProgressPoint {
  id: number;
  created_at: string;
  phase: "loading" | "training" | "saving" | "done" | "failed";
  step: number | null;
  total_steps: number | null;
  epoch: number | null;
  loss: number | null;
  learning_rate: number | null;
  grad_norm: number | null;
  samples_per_sec: number | null;
  gpu_util: number | null;
  gpu_mem_used_gb: number | null;
  gpu_mem_total_gb: number | null;
  message: string | null;
}

export interface CorpusStatus {
  counts: Record<string, number>;
  recent_stages: StageRun[];
}

export interface DatasetVersion {
  version: string;
  created_at: string;
  fingerprint: string;
  train_count: number;
  eval_count: number;
}

export interface Adapter {
  version: string;
  served_as: string;
  active: boolean;
  metrics: Record<string, number>;
}

export interface DoctorCheck {
  check: string;
  ok: boolean;
  detail: string;
}

export interface Resources {
  functions: Record<string, { backlog: number; num_total_runners: number; num_running_inputs: number }>;
  active_jobs: {
    id: string;
    kind: JobKind;
    status: JobStatus;
    started_at: string | null;
    params: Record<string, unknown>;
    latest: Partial<ProgressPoint> | null;
  }[];
}

export interface Thread {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  artifact_count?: number;
}

export interface ToolCallRecord {
  id?: string;
  name: string;
  args?: Record<string, unknown>;
  result?: string;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  created_at: string;
  tool_calls: ToolCallRecord[];
  asset_ids: string[];
  model: string | null;
}

export interface Asset {
  id: string;
  filename: string;
  content_type: string;
  bytes: number;
  width: number | null;
  height: number | null;
}

export interface PublishedPage {
  slug: string;
  title: string;
  version: number;
  status: "published" | "unpublished";
  artifact_id: string;
  updated_at: string;
  url?: string;
}

export interface Artifact {
  id: string;
  title: string;
  current_version: number;
  updated_at: string;
  versions: { version: number; created_at: string; note: string | null; title: string | null; has_html: boolean }[];
  published: PublishedPage[];
}

export interface AgentRun {
  id: string;
  thread_id: string;
  message_id: string | null;
  status: JobStatus;
  error: string | null;
}

export interface ThreadDetail extends Thread {
  messages: ChatMessage[];
  artifacts: Artifact[];
  assets: Asset[];
  /** A turn still running on its background worker (e.g. after a reload). */
  active_run: AgentRun | null;
}

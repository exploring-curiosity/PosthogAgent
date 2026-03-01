export interface PipelineStep {
  id: string;
  label: string;
  description: string;
  icon: string;
  status: "idle" | "running" | "done" | "error";
  logs: LogEntry[];
  startedAt?: number;
  finishedAt?: number;
  stats?: Record<string, unknown>;
}

export interface LogEntry {
  id: number;
  type: "log" | "error" | "warning" | "progress" | "section" | "status" | "done";
  message: string;
  timestamp: number;
}

export interface AgentData {
  cluster_id: number;
  demographic: string;
  traits: {
    age_group: string;
    country: string;
    nyc_familiarity: string;
  };
  description: string;
  summary: {
    demographic: string;
    total_steps: number;
    total_duration_s: number;
    successful_actions: number;
    failed_actions: number;
    action_history: ActionEntry[];
    impressions: Impression[];
  };
  narrative: string;
}

export interface ActionEntry {
  step: number;
  elapsed_s: number;
  action: string;
  target: string;
  success: boolean;
  error: string;
}

export interface Impression {
  step: number;
  elapsed_s: number;
  context: string;
  sentiment: string;
  url: string;
}

export interface ComparativeData {
  common_friction_points: string[];
  demographic_specific_issues: Array<{
    demographic: string;
    issue: string;
    severity: string;
  }>;
  accessibility_findings: string[];
  engagement_patterns: {
    most_engaged_demographic: string;
    least_engaged_demographic: string;
    engagement_summary: string;
  };
  recommendations: Array<{
    title: string;
    impact: string;
    demographics_affected: string[];
    detail: string;
  }>;
  overall_assessment: string;
}

export interface ReportData {
  generated_at: string;
  target_app: { url: string; name: string };
  agents: AgentData[];
  comparative: ComparativeData;
}

export interface StatusResponse {
  steps: {
    download: { done: boolean; count: number };
    cluster: { done: boolean; count: number; data: unknown };
    training: { done: boolean; count: number };
    finetune: { done: boolean; data: unknown };
    agents: { done: boolean; data: unknown };
  };
  counts: {
    recordings: number;
    parsed: number;
    descriptions: number;
    embeddings: number;
    trainingFiles: number;
    modelFiles: number;
  };
  hasReport: boolean;
}

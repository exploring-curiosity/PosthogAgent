"use client";

import React, { useState, useRef, useEffect } from "react";
import {
  Download,
  GitBranch,
  Database,
  Cpu,
  Play,
  CheckCircle,
  XCircle,
  Loader2,
  Circle,
  ChevronDown,
  ChevronUp,
  Terminal,
  AlertTriangle,
  Rocket,
  ArrowRight,
} from "lucide-react";
import type { PipelineStep, StatusResponse, LogEntry } from "../lib/types";

const STEP_ICONS: Record<string, React.ElementType> = {
  download: Download,
  cluster: GitBranch,
  training: Database,
  finetune: Cpu,
  agents: Play,
};

const STEP_COLORS: Record<string, string> = {
  download: "#06B6D4",
  cluster: "#A855F7",
  training: "#F59E0B",
  finetune: "#EC4899",
  agents: "#22C55E",
};

interface Props {
  steps: PipelineStep[];
  onRunStep: (stepId: string) => void;
  isAnyRunning: boolean;
  status: StatusResponse | null;
}

export default function PipelineRunner({ steps, onRunStep, isAnyRunning, status }: Props) {
  const [expandedStep, setExpandedStep] = useState<string | null>(null);

  // Auto-expand running step
  useEffect(() => {
    const running = steps.find((s) => s.status === "running");
    if (running) setExpandedStep(running.id);
  }, [steps]);

  const completedCount = steps.filter((s) => s.status === "done").length;
  const progress = (completedCount / steps.length) * 100;

  return (
    <div className="p-8 max-w-4xl mx-auto">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center gap-3 mb-2">
          <Rocket className="w-6 h-6" style={{ color: "var(--accent)" }} />
          <h1 className="text-2xl font-bold" style={{ color: "var(--foreground)" }}>
            Pipeline Control
          </h1>
        </div>
        <p className="text-sm" style={{ color: "var(--muted)" }}>
          Execute each stage of the behavioral digital twin pipeline. Steps run sequentially — each builds on the previous.
        </p>
      </div>

      {/* Overall progress bar */}
      <div className="mb-8 rounded-xl p-5" style={{ background: "var(--surface)" }}>
        <div className="flex items-center justify-between mb-3">
          <span className="text-xs font-medium uppercase tracking-wider" style={{ color: "var(--muted)" }}>
            Overall Progress
          </span>
          <span className="text-sm font-bold" style={{ color: "var(--accent)" }}>
            {completedCount}/{steps.length} stages
          </span>
        </div>
        <div className="h-2 rounded-full overflow-hidden" style={{ background: "var(--surface-2)" }}>
          <div
            className="h-full rounded-full transition-all duration-700 ease-out"
            style={{
              width: `${progress}%`,
              background: `linear-gradient(90deg, var(--cyan), var(--accent))`,
            }}
          />
        </div>

        {/* Data counts */}
        {status && (
          <div className="grid grid-cols-3 md:grid-cols-6 gap-3 mt-4">
            <MiniStat label="Recordings" value={status.counts.recordings} />
            <MiniStat label="Parsed" value={status.counts.parsed} />
            <MiniStat label="Described" value={status.counts.descriptions} />
            <MiniStat label="Embedded" value={status.counts.embeddings} />
            <MiniStat label="Training" value={status.counts.trainingFiles} />
            <MiniStat label="Models" value={status.counts.modelFiles} />
          </div>
        )}
      </div>

      {/* Pipeline steps */}
      <div className="space-y-3">
        {steps.map((step, idx) => {
          const Icon = STEP_ICONS[step.id] || Circle;
          const color = STEP_COLORS[step.id] || "#94A3B8";
          const isExpanded = expandedStep === step.id;
          const canRun = !isAnyRunning && step.status !== "running";
          const prevDone = idx === 0 || steps[idx - 1].status === "done";

          return (
            <div
              key={step.id}
              className="rounded-xl border overflow-hidden transition-all duration-300"
              style={{
                background: "var(--surface)",
                borderColor: step.status === "running" ? color : "var(--border)",
                boxShadow: step.status === "running" ? `0 0 20px ${color}20` : "none",
              }}
            >
              {/* Step header */}
              <div
                className="flex items-center gap-4 px-5 py-4 cursor-pointer"
                onClick={() => setExpandedStep(isExpanded ? null : step.id)}
              >
                {/* Step number + icon */}
                <div className="flex items-center gap-3">
                  <div
                    className="w-10 h-10 rounded-lg flex items-center justify-center shrink-0"
                    style={{ background: `${color}15` }}
                  >
                    <Icon className="w-5 h-5" style={{ color }} />
                  </div>
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-mono" style={{ color: "var(--muted)" }}>
                        {String(idx + 1).padStart(2, "0")}
                      </span>
                      <h3 className="font-semibold text-sm" style={{ color: "var(--foreground)" }}>
                        {step.label}
                      </h3>
                    </div>
                    <p className="text-xs mt-0.5" style={{ color: "var(--muted)" }}>
                      {step.description}
                    </p>
                  </div>
                </div>

                {/* Right side: status + actions */}
                <div className="ml-auto flex items-center gap-3">
                  {step.status === "running" && step.startedAt && (
                    <ElapsedTimer startedAt={step.startedAt} color={color} />
                  )}
                  {step.status === "done" && step.startedAt && step.finishedAt && (
                    <span className="text-xs font-mono" style={{ color: "var(--muted)" }}>
                      {((step.finishedAt - step.startedAt) / 1000).toFixed(1)}s
                    </span>
                  )}

                  <StepStatusBadge status={step.status} color={color} />

                  {canRun && prevDone && (
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        onRunStep(step.id);
                      }}
                      className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all cursor-pointer hover:brightness-110"
                      style={{ background: `${color}20`, color }}
                    >
                      <Play className="w-3 h-3" />
                      {step.status === "done" ? "Re-run" : "Run"}
                    </button>
                  )}

                  {isExpanded ? (
                    <ChevronUp className="w-4 h-4" style={{ color: "var(--muted)" }} />
                  ) : (
                    <ChevronDown className="w-4 h-4" style={{ color: "var(--muted)" }} />
                  )}
                </div>
              </div>

              {/* Expanded log panel */}
              {isExpanded && (
                <div
                  className="border-t px-5 py-4"
                  style={{ borderColor: "var(--border)", background: "var(--background)" }}
                >
                  <LogPanel logs={step.logs} isRunning={step.status === "running"} color={color} />
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Run all button */}
      <div className="mt-8 flex justify-center">
        <button
          onClick={() => {
            const nextIdle = steps.find((s) => s.status === "idle" || s.status === "error");
            if (nextIdle) onRunStep(nextIdle.id);
          }}
          disabled={isAnyRunning || completedCount === steps.length}
          className="flex items-center gap-2 px-6 py-3 rounded-xl text-sm font-semibold transition-all cursor-pointer disabled:opacity-30 disabled:cursor-not-allowed"
          style={{
            background: isAnyRunning ? "var(--surface-2)" : "linear-gradient(135deg, var(--cyan), var(--accent))",
            color: "white",
          }}
        >
          {isAnyRunning ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" /> Running...
            </>
          ) : completedCount === steps.length ? (
            <>
              <CheckCircle className="w-4 h-4" /> All Complete
            </>
          ) : (
            <>
              <Rocket className="w-4 h-4" /> Run Next Step
            </>
          )}
        </button>
      </div>
    </div>
  );
}

/* ─── SUB-COMPONENTS ─── */

function MiniStat({ label, value }: { label: string; value: number }) {
  return (
    <div className="text-center py-2 px-1 rounded-lg" style={{ background: "var(--surface-2)" }}>
      <p className="text-lg font-bold" style={{ color: value > 0 ? "var(--foreground)" : "var(--border)" }}>
        {value}
      </p>
      <p className="text-[10px] uppercase tracking-wider" style={{ color: "var(--muted)" }}>
        {label}
      </p>
    </div>
  );
}

function StepStatusBadge({ status, color }: { status: PipelineStep["status"]; color: string }) {
  if (status === "done") {
    return (
      <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium" style={{ background: "rgba(34,197,94,0.15)", color: "var(--accent)" }}>
        <CheckCircle className="w-3 h-3" /> Done
      </div>
    );
  }
  if (status === "error") {
    return (
      <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium" style={{ background: "rgba(239,68,68,0.15)", color: "var(--red)" }}>
        <XCircle className="w-3 h-3" /> Error
      </div>
    );
  }
  if (status === "running") {
    return (
      <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium" style={{ background: `${color}15`, color }}>
        <Loader2 className="w-3 h-3 animate-spin" /> Running
      </div>
    );
  }
  return null;
}

function ElapsedTimer({ startedAt, color }: { startedAt: number; color: string }) {
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setElapsed(Math.floor((Date.now() - startedAt) / 1000));
    }, 1000);
    return () => clearInterval(interval);
  }, [startedAt]);

  const mins = Math.floor(elapsed / 60);
  const secs = elapsed % 60;

  return (
    <span className="text-xs font-mono tabular-nums" style={{ color }}>
      {mins > 0 ? `${mins}m ` : ""}{secs}s
    </span>
  );
}

function LogPanel({ logs, isRunning, color }: { logs: LogEntry[]; isRunning: boolean; color: string }) {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logs]);

  if (logs.length === 0 && !isRunning) {
    return (
      <div className="flex items-center gap-2 py-6 justify-center" style={{ color: "var(--muted)" }}>
        <Terminal className="w-4 h-4" />
        <span className="text-xs">No output yet — click Run to start this step</span>
      </div>
    );
  }

  return (
    <div>
      {/* Terminal header */}
      <div className="flex items-center gap-2 mb-2">
        <Terminal className="w-3.5 h-3.5" style={{ color: "var(--muted)" }} />
        <span className="text-[10px] uppercase tracking-wider font-medium" style={{ color: "var(--muted)" }}>
          Output
        </span>
        <span className="text-[10px]" style={{ color: "var(--border)" }}>
          {logs.length} lines
        </span>
      </div>

      {/* Log lines */}
      <div
        ref={scrollRef}
        className="rounded-lg p-4 font-mono text-xs leading-relaxed max-h-80 overflow-y-auto"
        style={{ background: "var(--surface)" }}
      >
        {logs.map((log) => (
          <LogLine key={log.id} entry={log} />
        ))}

        {/* Blinking cursor */}
        {isRunning && (
          <div className="flex items-center gap-1 mt-1">
            <span className="animate-blink" style={{ color }}>
              _
            </span>
          </div>
        )}
      </div>
    </div>
  );
}

function LogLine({ entry }: { entry: LogEntry }) {
  const colors: Record<string, string> = {
    log: "var(--muted)",
    error: "var(--red)",
    warning: "var(--amber)",
    progress: "var(--cyan)",
    section: "var(--foreground)",
    status: "var(--accent)",
    done: "var(--accent)",
  };

  const icons: Record<string, React.ElementType | null> = {
    error: AlertTriangle,
    warning: AlertTriangle,
    section: ArrowRight,
    status: Loader2,
    done: CheckCircle,
  };

  const IconComp = icons[entry.type] || null;

  return (
    <div className="flex items-start gap-2 py-0.5 animate-slide-in" style={{ color: colors[entry.type] || "var(--muted)" }}>
      {IconComp && <IconComp className="w-3 h-3 mt-0.5 shrink-0" />}
      <span className={entry.type === "section" ? "font-semibold" : ""}>{entry.message}</span>
    </div>
  );
}

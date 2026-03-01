"use client";

import React, { useState, useEffect, useCallback } from "react";
import {
  Zap,
  BarChart3,
  CheckCircle,
  XCircle,
  Loader2,
  Circle,
  Users,
} from "lucide-react";
import PipelineRunner from "./PipelineRunner";
import ResultsView from "./ResultsView";
import type { StatusResponse, PipelineStep } from "../lib/types";

type View = "pipeline" | "results";

const NAV_ITEMS: { id: View; label: string; icon: React.ElementType }[] = [
  { id: "pipeline", label: "Pipeline", icon: Zap },
  { id: "results", label: "Results", icon: BarChart3 },
];

const INITIAL_STEPS: PipelineStep[] = [
  { id: "download", label: "Download Recordings", description: "Fetch session recordings from PostHog API", icon: "Download", status: "idle", logs: [] },
  { id: "cluster", label: "Cluster Demographics", description: "Parse, describe, embed & K-Means cluster", icon: "GitBranch", status: "idle", logs: [] },
  { id: "training", label: "Build Training Data", description: "Generate state→action JSONL pairs", icon: "Database", status: "idle", logs: [] },
  { id: "finetune", label: "Fine-Tune Models", description: "Train 3 Mistral models via API + W&B", icon: "Cpu", status: "idle", logs: [] },
  { id: "agents", label: "Run Agents", description: "Launch 3 demographic agents to explore app", icon: "Play", status: "idle", logs: [] },
];

export default function AppShell() {
  const [view, setView] = useState<View>("pipeline");
  const [steps, setSteps] = useState<PipelineStep[]>(INITIAL_STEPS);
  const [status, setStatus] = useState<StatusResponse | null>(null);

  // Fetch pipeline status on mount
  const fetchStatus = useCallback(async () => {
    try {
      const res = await fetch("/api/status");
      if (res.ok) {
        const data: StatusResponse = await res.json();
        setStatus(data);

        // Update step statuses based on server data
        setSteps((prev) =>
          prev.map((step) => {
            const serverStep = data.steps[step.id as keyof typeof data.steps];
            if (serverStep?.done && step.status === "idle") {
              return { ...step, status: "done" as const };
            }
            return step;
          })
        );
      }
    } catch {
      // Server may not be ready yet
    }
  }, []);

  useEffect(() => {
    fetchStatus();
  }, [fetchStatus]);

  const updateStep = useCallback((stepId: string, update: Partial<PipelineStep>) => {
    setSteps((prev) =>
      prev.map((s) => (s.id === stepId ? { ...s, ...update } : s))
    );
  }, []);

  const runStep = useCallback(
    async (stepId: string) => {
      updateStep(stepId, { status: "running", logs: [], startedAt: Date.now() });

      try {
        const res = await fetch("/api/pipeline", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ step: stepId }),
        });

        if (!res.body) throw new Error("No response body");

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        let logId = 0;

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n\n");
          buffer = lines.pop() || "";

          for (const line of lines) {
            if (!line.startsWith("data: ")) continue;
            try {
              const event = JSON.parse(line.slice(6));

              if (event.type === "done") {
                updateStep(stepId, {
                  status: event.success ? "done" : "error",
                  finishedAt: Date.now(),
                });
                fetchStatus();
              } else {
                const entry = {
                  id: logId++,
                  type: event.type,
                  message: event.message || event.label || "",
                  timestamp: Date.now(),
                };
                setSteps((prev) =>
                  prev.map((s) =>
                    s.id === stepId ? { ...s, logs: [...s.logs, entry] } : s
                  )
                );
              }
            } catch {
              // skip malformed events
            }
          }
        }
      } catch (err) {
        updateStep(stepId, {
          status: "error",
          finishedAt: Date.now(),
          logs: [
            ...steps.find((s) => s.id === stepId)?.logs || [],
            { id: 0, type: "error", message: String(err), timestamp: Date.now() },
          ],
        });
      }
    },
    [updateStep, fetchStatus, steps]
  );

  const isAnyRunning = steps.some((s) => s.status === "running");

  return (
    <div className="flex h-screen overflow-hidden" style={{ background: "var(--background)" }}>
      {/* Sidebar */}
      <aside className="w-64 flex-shrink-0 flex flex-col border-r" style={{ background: "var(--surface)", borderColor: "var(--border)" }}>
        {/* Logo */}
        <div className="px-5 py-5 border-b" style={{ borderColor: "var(--border)" }}>
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg flex items-center justify-center"
              style={{ background: "linear-gradient(135deg, var(--cyan), var(--purple))" }}>
              <Users className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-sm font-bold tracking-tight" style={{ color: "var(--foreground)" }}>
                Agentic World
              </h1>
              <p className="text-[10px]" style={{ color: "var(--muted)" }}>
                Behavioral Digital Twins
              </p>
            </div>
          </div>
        </div>

        {/* Nav */}
        <nav className="flex-1 px-3 py-4 space-y-1">
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            const active = view === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setView(item.id)}
                className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors cursor-pointer"
                style={{
                  background: active ? "var(--accent-dim)" : "transparent",
                  color: active ? "var(--accent)" : "var(--muted)",
                }}
              >
                <Icon className="w-4 h-4" />
                {item.label}
                {item.id === "results" && status?.hasReport && (
                  <span className="ml-auto w-2 h-2 rounded-full" style={{ background: "var(--accent)" }} />
                )}
              </button>
            );
          })}
        </nav>

        {/* Pipeline progress mini */}
        <div className="px-4 py-4 border-t" style={{ borderColor: "var(--border)" }}>
          <p className="text-[10px] uppercase tracking-wider mb-3" style={{ color: "var(--muted)" }}>
            Pipeline Progress
          </p>
          <div className="space-y-2">
            {steps.map((step) => (
              <div key={step.id} className="flex items-center gap-2.5">
                <StepDot status={step.status} />
                <span
                  className="text-xs truncate"
                  style={{ color: step.status === "running" ? "var(--accent)" : step.status === "done" ? "var(--foreground)" : "var(--muted)" }}
                >
                  {step.label}
                </span>
              </div>
            ))}
          </div>
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 overflow-y-auto">
        {view === "pipeline" && (
          <PipelineRunner
            steps={steps}
            onRunStep={runStep}
            isAnyRunning={isAnyRunning}
            status={status}
          />
        )}
        {view === "results" && <ResultsView />}
      </main>
    </div>
  );
}

function StepDot({ status }: { status: PipelineStep["status"] }) {
  if (status === "done") return <CheckCircle className="w-3.5 h-3.5 shrink-0" style={{ color: "var(--accent)" }} />;
  if (status === "error") return <XCircle className="w-3.5 h-3.5 shrink-0" style={{ color: "var(--red)" }} />;
  if (status === "running") return <Loader2 className="w-3.5 h-3.5 shrink-0 animate-spin" style={{ color: "var(--accent)" }} />;
  return <Circle className="w-3.5 h-3.5 shrink-0" style={{ color: "var(--border)" }} />;
}

"use client";

import React from "react";
import reportData from "../data/report.json";
import agentLogData from "../data/agent_log.json";
import HeroStats from "./HeroStats";
import ActionTimeline from "./ActionTimeline";
import TimingComparison from "./TimingComparison";
import FrictionMap from "./FrictionMap";
import QualitativeReport from "./QualitativeReport";
import { Bot } from "lucide-react";

export default function Dashboard() {
  const report = reportData as Record<string, unknown>;
  const agentLog = agentLogData as Record<string, unknown>;

  const metrics = report.metrics as {
    completion: {
      planned_actions: number;
      executed_actions: number;
      successful_actions: number;
      failed_actions: number;
      completion_rate: number;
    };
    timing: {
      agent_duration_s: number;
      real_user_duration_s: number;
      duration_ratio: number;
      per_action_deltas: Record<
        string,
        {
          agent_avg_s: number;
          real_user_avg_s: number;
          delta_s: number;
          ratio: number;
        }
      >;
    };
    errors: {
      total_errors: number;
      total_stuck_events: number;
      errors_by_action: Record<string, number>;
    };
    deviation: {
      match_rate: number;
      deviation_score: number;
      missing_action_types: string[];
      extra_action_types: string[];
    };
    engagement: {
      agent_comments_written: number;
      agent_votes_cast: number;
      real_user_input_events: number;
      real_user_click_events: number;
    };
  };

  const actions = (agentLog.actions as Array<{
    action_name: string;
    relative_s: number;
    duration_s: number;
    success: boolean;
    error: string;
    page_url: string;
    details: Record<string, unknown>;
  }>);

  const summary = agentLog.summary as {
    total_duration_s: number;
    total_actions: number;
    successful_actions: number;
    failed_actions: number;
    avg_action_timings: Record<string, number>;
    action_sequence_actual: string[];
  };

  const qualitative = report.qualitative_feedback as string;
  const sessionId = report.session_id as string;

  return (
    <div className="min-h-screen bg-[#0a0a0a] text-gray-100">
      {/* Header */}
      <header className="border-b border-gray-800 bg-[#0d0d0d]">
        <div className="mx-auto max-w-7xl px-6 py-5 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="flex items-center justify-center w-10 h-10 rounded-lg bg-gradient-to-br from-cyan-500 to-blue-600">
              <Bot className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold tracking-tight">
                PosthogAgent Feedback
              </h1>
              <p className="text-xs text-gray-500 font-mono">
                Session {sessionId.slice(0, 8)}…
              </p>
            </div>
          </div>
          <div className="text-right text-xs text-gray-500">
            <p>Generated {new Date(report.generated_at as string).toLocaleString()}</p>
            <p className="text-cyan-400 font-medium">
              {metrics.completion.successful_actions}/{metrics.completion.planned_actions} actions completed
            </p>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-6 py-8 space-y-10">
        {/* Hero stats */}
        <HeroStats metrics={metrics} summary={summary} />

        {/* Action Timeline */}
        <section>
          <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <span className="inline-block w-1.5 h-5 rounded-full bg-cyan-500" />
            Agent Action Timeline
          </h2>
          <ActionTimeline actions={actions} totalDuration={summary.total_duration_s} />
        </section>

        {/* Timing Comparison */}
        <section>
          <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <span className="inline-block w-1.5 h-5 rounded-full bg-amber-500" />
            Agent vs Real User — Timing Comparison
          </h2>
          <TimingComparison
            perActionDeltas={metrics.timing.per_action_deltas}
            avgTimings={summary.avg_action_timings}
          />
        </section>

        {/* Friction Map */}
        <section>
          <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <span className="inline-block w-1.5 h-5 rounded-full bg-red-500" />
            Friction & Difficulty Map
          </h2>
          <FrictionMap
            actions={actions}
            avgTimings={summary.avg_action_timings}
            perActionDeltas={metrics.timing.per_action_deltas}
            errors={metrics.errors}
          />
        </section>

        {/* Qualitative Report */}
        <section>
          <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <span className="inline-block w-1.5 h-5 rounded-full bg-purple-500" />
            AI-Generated UX Recommendations
          </h2>
          <QualitativeReport markdown={qualitative} />
        </section>
      </main>

      <footer className="border-t border-gray-800 mt-16 py-6 text-center text-xs text-gray-600">
        PosthogAgent — Agentic UX Feedback Pipeline
      </footer>
    </div>
  );
}

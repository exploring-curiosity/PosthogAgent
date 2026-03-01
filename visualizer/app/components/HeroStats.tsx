"use client";

import React from "react";
import {
  CheckCircle,
  XCircle,
  Clock,
  Zap,
  MessageSquare,
  ThumbsUp,
  Target,
  AlertTriangle,
} from "lucide-react";

interface Props {
  metrics: {
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
    };
    errors: {
      total_errors: number;
      total_stuck_events: number;
    };
    deviation: {
      match_rate: number;
      deviation_score: number;
    };
    engagement: {
      agent_comments_written: number;
      agent_votes_cast: number;
      real_user_input_events: number;
      real_user_click_events: number;
    };
  };
  summary: {
    total_duration_s: number;
    total_actions: number;
    successful_actions: number;
    failed_actions: number;
  };
}

function StatCard({
  icon,
  label,
  value,
  subtext,
  accent = "cyan",
}: {
  icon: React.ReactNode;
  label: string;
  value: string | number;
  subtext?: string;
  accent?: "cyan" | "green" | "red" | "amber" | "purple";
}) {
  const accentColors: Record<string, string> = {
    cyan: "from-cyan-500/20 to-cyan-500/5 border-cyan-500/30",
    green: "from-emerald-500/20 to-emerald-500/5 border-emerald-500/30",
    red: "from-red-500/20 to-red-500/5 border-red-500/30",
    amber: "from-amber-500/20 to-amber-500/5 border-amber-500/30",
    purple: "from-purple-500/20 to-purple-500/5 border-purple-500/30",
  };

  return (
    <div
      className={`rounded-xl border bg-gradient-to-br p-5 ${accentColors[accent]}`}
    >
      <div className="flex items-center gap-2 text-gray-400 text-xs uppercase tracking-wider mb-2">
        {icon}
        {label}
      </div>
      <div className="text-2xl font-bold tracking-tight">{value}</div>
      {subtext && (
        <div className="text-xs text-gray-500 mt-1">{subtext}</div>
      )}
    </div>
  );
}

export default function HeroStats({ metrics, summary }: Props) {
  const completionPct = Math.round(metrics.completion.completion_rate * 100);
  const matchPct = Math.round(metrics.deviation.match_rate * 100);

  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-4">
      <StatCard
        icon={<CheckCircle className="w-4 h-4" />}
        label="Actions Completed"
        value={`${summary.successful_actions}/${summary.total_actions}`}
        subtext={`${completionPct}% completion rate`}
        accent="green"
      />
      <StatCard
        icon={<XCircle className="w-4 h-4" />}
        label="Errors"
        value={metrics.errors.total_errors}
        subtext={`${metrics.errors.total_stuck_events} stuck events`}
        accent={metrics.errors.total_errors > 0 ? "red" : "green"}
      />
      <StatCard
        icon={<Clock className="w-4 h-4" />}
        label="Agent Duration"
        value={`${metrics.timing.agent_duration_s.toFixed(1)}s`}
        subtext={`Real user: ${metrics.timing.real_user_duration_s.toFixed(1)}s (${metrics.timing.duration_ratio.toFixed(1)}x ratio)`}
        accent="cyan"
      />
      <StatCard
        icon={<Zap className="w-4 h-4" />}
        label="Speed Ratio"
        value={`${metrics.timing.duration_ratio.toFixed(2)}x`}
        subtext={
          metrics.timing.duration_ratio < 1
            ? "Agent faster than user"
            : "Agent slower than user"
        }
        accent={metrics.timing.duration_ratio > 1.5 ? "amber" : "green"}
      />
      <StatCard
        icon={<MessageSquare className="w-4 h-4" />}
        label="Comments Written"
        value={metrics.engagement.agent_comments_written}
        subtext={`Real user had ${metrics.engagement.real_user_input_events} input events`}
        accent="purple"
      />
      <StatCard
        icon={<ThumbsUp className="w-4 h-4" />}
        label="Votes Cast"
        value={metrics.engagement.agent_votes_cast}
        subtext={`Real user had ${metrics.engagement.real_user_click_events} click events`}
        accent="purple"
      />
      <StatCard
        icon={<Target className="w-4 h-4" />}
        label="Policy Match"
        value={`${matchPct}%`}
        subtext={`Deviation score: ${metrics.deviation.deviation_score.toFixed(2)}`}
        accent="cyan"
      />
      <StatCard
        icon={<AlertTriangle className="w-4 h-4" />}
        label="Stuck Events"
        value={metrics.errors.total_stuck_events}
        subtext="Times agent got stuck on a page"
        accent={metrics.errors.total_stuck_events > 0 ? "red" : "green"}
      />
    </div>
  );
}

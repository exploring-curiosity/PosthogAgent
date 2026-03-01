"use client";

import React from "react";
import { AlertTriangle, Clock, XCircle, Repeat } from "lucide-react";

interface Action {
  action_name: string;
  relative_s: number;
  duration_s: number;
  success: boolean;
  error: string;
  page_url: string;
  details: Record<string, unknown>;
}

interface PerActionDelta {
  agent_avg_s: number;
  real_user_avg_s: number;
  delta_s: number;
  ratio: number;
}

interface Props {
  actions: Action[];
  avgTimings: Record<string, number>;
  perActionDeltas: Record<string, PerActionDelta>;
  errors: {
    total_errors: number;
    total_stuck_events: number;
    errors_by_action: Record<string, number>;
  };
}

interface FrictionItem {
  action: string;
  score: number;
  reasons: string[];
  severity: "critical" | "high" | "medium" | "low";
  avgDuration: number;
  occurrences: number;
  failures: number;
  slowdownRatio: number;
  skippedCount: number;
}

function computeFriction(
  actions: Action[],
  avgTimings: Record<string, number>,
  perActionDeltas: Record<string, PerActionDelta>,
  errors: Props["errors"]
): FrictionItem[] {
  const actionGroups: Record<string, Action[]> = {};
  for (const a of actions) {
    if (!actionGroups[a.action_name]) actionGroups[a.action_name] = [];
    actionGroups[a.action_name].push(a);
  }

  const items: FrictionItem[] = [];

  for (const [actionName, group] of Object.entries(actionGroups)) {
    const reasons: string[] = [];
    let score = 0;

    const failures = group.filter((a) => !a.success).length;
    const skipped = group.filter((a) => a.details.skipped === true).length;
    const avgDur = avgTimings[actionName] || 0;
    const delta = perActionDeltas[actionName];
    const slowdownRatio = delta ? delta.ratio : 0;
    const errorCount = errors.errors_by_action[actionName] || 0;

    // Scoring: higher = more friction
    if (failures > 0) {
      score += failures * 30;
      reasons.push(`${failures} failure(s) — agent could not complete this action`);
    }
    if (errorCount > 0) {
      score += errorCount * 25;
      reasons.push(`${errorCount} error(s) encountered`);
    }
    if (slowdownRatio >= 5) {
      score += 25;
      reasons.push(
        `${slowdownRatio.toFixed(1)}x slower than real user — significant UI friction`
      );
    } else if (slowdownRatio >= 3) {
      score += 15;
      reasons.push(
        `${slowdownRatio.toFixed(1)}x slower than real user — moderate friction`
      );
    } else if (slowdownRatio >= 1.5) {
      score += 5;
      reasons.push(`${slowdownRatio.toFixed(1)}x slower than real user`);
    }
    if (avgDur > 6) {
      score += 10;
      reasons.push(
        `High average duration (${avgDur.toFixed(1)}s) — may indicate loading or rendering delays`
      );
    }
    if (skipped > 0) {
      score += skipped * 5;
      reasons.push(
        `${skipped}x skipped (probability) — agent uncertain about interaction`
      );
    }

    // Only include if there's any friction signal
    if (score > 0 || slowdownRatio > 1) {
      let severity: FrictionItem["severity"] = "low";
      if (score >= 40) severity = "critical";
      else if (score >= 20) severity = "high";
      else if (score >= 10) severity = "medium";

      items.push({
        action: actionName,
        score,
        reasons,
        severity,
        avgDuration: avgDur,
        occurrences: group.length,
        failures,
        slowdownRatio,
        skippedCount: skipped,
      });
    }
  }

  items.sort((a, b) => b.score - a.score);
  return items;
}

const SEVERITY_STYLES: Record<
  string,
  { bg: string; border: string; text: string; icon: string }
> = {
  critical: {
    bg: "bg-red-500/10",
    border: "border-red-500/40",
    text: "text-red-400",
    icon: "text-red-500",
  },
  high: {
    bg: "bg-amber-500/10",
    border: "border-amber-500/40",
    text: "text-amber-400",
    icon: "text-amber-500",
  },
  medium: {
    bg: "bg-yellow-500/10",
    border: "border-yellow-500/30",
    text: "text-yellow-400",
    icon: "text-yellow-500",
  },
  low: {
    bg: "bg-gray-500/10",
    border: "border-gray-500/30",
    text: "text-gray-400",
    icon: "text-gray-500",
  },
};

export default function FrictionMap({
  actions,
  avgTimings,
  perActionDeltas,
  errors,
}: Props) {
  const frictionItems = computeFriction(
    actions,
    avgTimings,
    perActionDeltas,
    errors
  );

  if (frictionItems.length === 0) {
    return (
      <div className="rounded-xl border border-gray-800 bg-[#111] p-8 text-center text-gray-500">
        No significant friction detected — agent completed all actions smoothly.
      </div>
    );
  }

  const maxScore = Math.max(...frictionItems.map((f) => f.score));

  return (
    <div className="space-y-4">
      {/* Friction heatmap grid */}
      <div className="rounded-xl border border-gray-800 bg-[#111] p-5">
        <h3 className="text-sm font-medium text-gray-400 mb-4">
          Friction Heatmap — Action Difficulty Score
        </h3>
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-3">
          {Object.entries(avgTimings).map(([action, avg]) => {
            const item = frictionItems.find((f) => f.action === action);
            const score = item ? item.score : 0;
            const intensity = maxScore > 0 ? score / maxScore : 0;

            // Color interpolation from green → yellow → red
            const r = Math.round(intensity > 0.5 ? 255 : intensity * 2 * 255);
            const g = Math.round(
              intensity > 0.5 ? (1 - (intensity - 0.5) * 2) * 255 : 255
            );
            const b = 50;

            return (
              <div
                key={action}
                className="relative rounded-lg border border-gray-800 p-3 text-center cursor-default group"
                style={{
                  backgroundColor: `rgba(${r}, ${g}, ${b}, ${
                    0.08 + intensity * 0.15
                  })`,
                  borderColor: `rgba(${r}, ${g}, ${b}, ${
                    0.2 + intensity * 0.3
                  })`,
                }}
              >
                <div className="text-xs font-medium text-gray-300 mb-1 truncate">
                  {action.replace(/_/g, " ")}
                </div>
                <div
                  className="text-2xl font-bold"
                  style={{ color: `rgb(${r}, ${g}, ${b})` }}
                >
                  {score}
                </div>
                <div className="text-[10px] text-gray-600 mt-0.5">
                  {avg.toFixed(1)}s avg
                </div>

                {/* Tooltip */}
                {item && (
                  <div className="absolute -top-2 left-1/2 -translate-x-1/2 -translate-y-full w-48 bg-gray-900 border border-gray-700 rounded-lg p-2 text-left opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-20">
                    <div className="text-[10px] text-gray-400 space-y-1">
                      {item.reasons.map((r, i) => (
                        <p key={i}>• {r}</p>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Detailed friction cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {frictionItems.map((item, i) => {
          const style = SEVERITY_STYLES[item.severity];
          return (
            <div
              key={i}
              className={`rounded-xl border ${style.border} ${style.bg} p-5`}
            >
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <AlertTriangle className={`w-4 h-4 ${style.icon}`} />
                  <h4 className="font-semibold text-sm text-gray-200">
                    {item.action.replace(/_/g, " ")}
                  </h4>
                  <span
                    className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase ${style.text} ${style.bg}`}
                  >
                    {item.severity}
                  </span>
                </div>
                <div className={`text-xl font-bold ${style.text}`}>
                  {item.score}
                </div>
              </div>

              {/* Stats row */}
              <div className="flex gap-4 mb-3 text-xs text-gray-500">
                <span className="flex items-center gap-1">
                  <Repeat className="w-3 h-3" />
                  {item.occurrences}x
                </span>
                <span className="flex items-center gap-1">
                  <Clock className="w-3 h-3" />
                  {item.avgDuration.toFixed(1)}s avg
                </span>
                {item.failures > 0 && (
                  <span className="flex items-center gap-1 text-red-400">
                    <XCircle className="w-3 h-3" />
                    {item.failures} failed
                  </span>
                )}
                {item.slowdownRatio > 0 && (
                  <span className="flex items-center gap-1 text-amber-400">
                    {item.slowdownRatio.toFixed(1)}x slower
                  </span>
                )}
              </div>

              {/* Reasons */}
              <ul className="space-y-1">
                {item.reasons.map((reason, j) => (
                  <li
                    key={j}
                    className="text-xs text-gray-400 flex items-start gap-1.5"
                  >
                    <span className="text-gray-600 mt-0.5">→</span>
                    {reason}
                  </li>
                ))}
              </ul>
            </div>
          );
        })}
      </div>
    </div>
  );
}

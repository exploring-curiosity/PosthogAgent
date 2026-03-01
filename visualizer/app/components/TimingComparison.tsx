"use client";

import React from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

interface PerActionDelta {
  agent_avg_s: number;
  real_user_avg_s: number;
  delta_s: number;
  ratio: number;
}

interface Props {
  perActionDeltas: Record<string, PerActionDelta>;
  avgTimings: Record<string, number>;
}

export default function TimingComparison({
  perActionDeltas,
  avgTimings,
}: Props) {
  // Build chart data: combine per-action deltas with avg timings
  const chartData = Object.entries(avgTimings).map(([action, agentAvg]) => {
    const delta = perActionDeltas[action];
    return {
      name: action.replace(/_/g, " "),
      agent: Number(agentAvg.toFixed(2)),
      realUser: delta ? Number(delta.real_user_avg_s.toFixed(2)) : 0,
      ratio: delta ? delta.ratio : 0,
      slowdownLabel: delta
        ? `${delta.ratio.toFixed(1)}x`
        : "N/A",
    };
  });

  // Sort by ratio descending (worst friction first)
  chartData.sort((a, b) => b.ratio - a.ratio);

  const getRatioColor = (ratio: number) => {
    if (ratio >= 5) return "#ef4444"; // red
    if (ratio >= 3) return "#f59e0b"; // amber
    if (ratio >= 1.5) return "#eab308"; // yellow
    return "#22c55e"; // green
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* Bar chart */}
      <div className="rounded-xl border border-gray-800 bg-[#111] p-5">
        <h3 className="text-sm font-medium text-gray-400 mb-4">
          Average Action Duration (seconds)
        </h3>
        <ResponsiveContainer width="100%" height={320}>
          <BarChart
            data={chartData}
            layout="vertical"
            margin={{ top: 0, right: 20, left: 20, bottom: 0 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#1f1f1f" />
            <XAxis type="number" tick={{ fill: "#666", fontSize: 11 }} />
            <YAxis
              dataKey="name"
              type="category"
              tick={{ fill: "#999", fontSize: 11 }}
              width={110}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "#1a1a1a",
                border: "1px solid #333",
                borderRadius: "8px",
                fontSize: "12px",
              }}
              labelStyle={{ color: "#ccc" }}
              // eslint-disable-next-line @typescript-eslint/no-explicit-any
              formatter={((value: any, name: any) => [
                `${Number(value ?? 0).toFixed(2)}s`,
                name === "agent" ? "Agent" : "Real User",
              ]) as any}
            />
            <Legend
              wrapperStyle={{ fontSize: "11px" }}
              formatter={(value: string) =>
                value === "agent" ? "Agent" : "Real User"
              }
            />
            <Bar dataKey="realUser" fill="#06b6d4" radius={[0, 4, 4, 0]} barSize={12} />
            <Bar dataKey="agent" fill="#8b5cf6" radius={[0, 4, 4, 0]} barSize={12} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Slowdown table */}
      <div className="rounded-xl border border-gray-800 bg-[#111] p-5">
        <h3 className="text-sm font-medium text-gray-400 mb-4">
          Slowdown Ratio — Where Agent Struggled Most
        </h3>
        <div className="space-y-3">
          {chartData
            .filter((d) => d.ratio > 0)
            .map((d, i) => (
              <div key={i} className="flex items-center gap-3">
                {/* Rank */}
                <div className="w-6 text-right text-xs font-mono text-gray-600">
                  #{i + 1}
                </div>

                {/* Action name */}
                <div className="w-28 text-sm text-gray-300 truncate">
                  {d.name}
                </div>

                {/* Bar */}
                <div className="flex-1 h-6 bg-gray-800 rounded-md overflow-hidden relative">
                  <div
                    className="h-full rounded-md transition-all duration-500"
                    style={{
                      width: `${Math.min(
                        (d.ratio / Math.max(...chartData.map((x) => x.ratio))) *
                          100,
                        100
                      )}%`,
                      backgroundColor: getRatioColor(d.ratio),
                    }}
                  />
                  <span className="absolute inset-0 flex items-center px-2 text-xs font-medium text-white mix-blend-difference">
                    {d.slowdownLabel} slower
                  </span>
                </div>

                {/* Values */}
                <div className="text-right shrink-0 w-32">
                  <span className="text-xs text-purple-400 font-mono">
                    {d.agent}s
                  </span>
                  <span className="text-xs text-gray-600 mx-1">vs</span>
                  <span className="text-xs text-cyan-400 font-mono">
                    {d.realUser}s
                  </span>
                </div>
              </div>
            ))}

          {/* Actions with no real-user comparison */}
          {chartData
            .filter((d) => d.ratio === 0)
            .map((d, i) => (
              <div key={`na-${i}`} className="flex items-center gap-3">
                <div className="w-6 text-right text-xs font-mono text-gray-600">
                  —
                </div>
                <div className="w-28 text-sm text-gray-500 truncate">
                  {d.name}
                </div>
                <div className="flex-1 text-xs text-gray-600 italic">
                  Agent-only action ({d.agent}s avg)
                </div>
              </div>
            ))}
        </div>
      </div>
    </div>
  );
}

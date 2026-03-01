"use client";

import React, { useState } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
} from "recharts";
import {
  Bot,
  Users,
  AlertTriangle,
  CheckCircle,
  XCircle,
  ChevronDown,
  ChevronUp,
  Globe,
  Clock,
  Zap,
  MessageSquare,
  Target,
  TrendingUp,
} from "lucide-react";

interface AgentData {
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
    action_history: Array<{
      step: number;
      elapsed_s: number;
      action: string;
      target: string;
      success: boolean;
      error: string;
    }>;
    impressions: Array<{
      step: number;
      elapsed_s: number;
      context: string;
      sentiment: string;
      url: string;
    }>;
  };
  narrative: string;
}

interface ComparativeData {
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

interface ReportData {
  generated_at: string;
  target_app: { url: string; name: string };
  agents: AgentData[];
  comparative: ComparativeData;
}

const AGENT_COLORS = ["#06b6d4", "#a855f7", "#f59e0b"];
const AGENT_BG = ["rgba(6,182,212,0.1)", "rgba(168,85,247,0.1)", "rgba(245,158,11,0.1)"];
const SEVERITY_COLORS: Record<string, string> = {
  high: "text-red-400 bg-red-500/10 border-red-500/20",
  medium: "text-amber-400 bg-amber-500/10 border-amber-500/20",
  low: "text-green-400 bg-green-500/10 border-green-500/20",
};
const IMPACT_COLORS: Record<string, string> = {
  high: "bg-red-500",
  medium: "bg-amber-500",
  low: "bg-green-500",
};

export default function ComparativeDashboard({ data }: { data: ReportData }) {
  const [activeTab, setActiveTab] = useState<"comparative" | number>("comparative");
  const [expandedRec, setExpandedRec] = useState<number | null>(null);

  const agents = data.agents;
  const comparative = data.comparative;

  const tabs = [
    { id: "comparative" as const, label: "Comparative", icon: Users },
    ...agents.map((a, i) => ({
      id: i as number,
      label: a.demographic,
      icon: Bot,
    })),
  ];

  return (
    <div className="min-h-screen bg-[#0a0a0a] text-gray-100">
      {/* Header */}
      <header className="border-b border-gray-800 bg-[#0d0d0d]">
        <div className="mx-auto max-w-7xl px-6 py-5 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="flex items-center justify-center w-10 h-10 rounded-lg bg-gradient-to-br from-cyan-500 to-purple-600">
              <Users className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold tracking-tight">
                Agentic World
              </h1>
              <p className="text-xs text-gray-500">
                {data.target_app.name} — {agents.length} Demographics
              </p>
            </div>
          </div>
          <div className="text-right text-xs text-gray-500">
            <p>Generated {new Date(data.generated_at).toLocaleString()}</p>
            <p className="text-cyan-400 font-medium">
              Target: {data.target_app.url}
            </p>
          </div>
        </div>

        {/* Tabs */}
        <div className="mx-auto max-w-7xl px-6">
          <div className="flex gap-1 border-t border-gray-800 pt-2 pb-0 overflow-x-auto">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              const isActive = activeTab === tab.id;
              const colorIdx = typeof tab.id === "number" ? tab.id : -1;
              return (
                <button
                  key={String(tab.id)}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center gap-2 px-4 py-2.5 text-sm font-medium rounded-t-lg transition-colors whitespace-nowrap ${
                    isActive
                      ? "bg-gray-800/80 text-white border-b-2"
                      : "text-gray-500 hover:text-gray-300 hover:bg-gray-800/30"
                  }`}
                  style={
                    isActive && colorIdx >= 0
                      ? { borderColor: AGENT_COLORS[colorIdx] }
                      : isActive
                      ? { borderColor: "#06b6d4" }
                      : {}
                  }
                >
                  <Icon className="w-4 h-4" />
                  {tab.label}
                </button>
              );
            })}
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-6 py-8">
        {activeTab === "comparative" ? (
          <ComparativeView agents={agents} comparative={comparative} />
        ) : (
          <AgentDetailView
            agent={agents[activeTab]}
            colorIdx={activeTab}
          />
        )}
      </main>

      <footer className="border-t border-gray-800 mt-16 py-6 text-center text-xs text-gray-600">
        Agentic World — Behavioral Digital Twin Pipeline
      </footer>
    </div>
  );
}

/* ─── COMPARATIVE VIEW ─── */

function ComparativeView({
  agents,
  comparative,
}: {
  agents: AgentData[];
  comparative: ComparativeData;
}) {
  return (
    <div className="space-y-10">
      {/* Overall Assessment */}
      <div className="rounded-xl bg-gradient-to-r from-cyan-500/10 via-purple-500/10 to-amber-500/10 border border-gray-800 p-6">
        <h2 className="text-lg font-semibold mb-2 flex items-center gap-2">
          <Target className="w-5 h-5 text-cyan-400" />
          Overall Assessment
        </h2>
        <p className="text-gray-300 leading-relaxed">{comparative.overall_assessment}</p>
      </div>

      {/* Agent Comparison Cards */}
      <section>
        <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <span className="inline-block w-1.5 h-5 rounded-full bg-cyan-500" />
          Agent Comparison
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {agents.map((agent, i) => (
            <AgentCard key={i} agent={agent} colorIdx={i} />
          ))}
        </div>
      </section>

      {/* Performance Bar Chart */}
      <section>
        <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <span className="inline-block w-1.5 h-5 rounded-full bg-amber-500" />
          Performance Comparison
        </h2>
        <PerformanceChart agents={agents} />
      </section>

      {/* Common Friction Points */}
      <section>
        <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <span className="inline-block w-1.5 h-5 rounded-full bg-red-500" />
          Common Friction Points
        </h2>
        <div className="space-y-3">
          {comparative.common_friction_points.map((point, i) => (
            <div key={i} className="flex items-start gap-3 rounded-lg bg-red-500/5 border border-red-500/20 p-4">
              <AlertTriangle className="w-5 h-5 text-red-400 mt-0.5 shrink-0" />
              <p className="text-gray-300">{point}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Demographic-Specific Issues */}
      <section>
        <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <span className="inline-block w-1.5 h-5 rounded-full bg-purple-500" />
          Demographic-Specific Issues
        </h2>
        <div className="space-y-2">
          {comparative.demographic_specific_issues.map((issue, i) => (
            <div
              key={i}
              className={`flex items-start gap-3 rounded-lg border p-4 ${SEVERITY_COLORS[issue.severity] || SEVERITY_COLORS.medium}`}
            >
              <span className="text-xs font-bold uppercase tracking-wider mt-0.5 shrink-0 w-16">
                {issue.severity}
              </span>
              <div>
                <span className="text-xs text-gray-500">{issue.demographic}</span>
                <p className="text-gray-300">{issue.issue}</p>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Engagement Patterns */}
      <section>
        <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <span className="inline-block w-1.5 h-5 rounded-full bg-green-500" />
          Engagement Patterns
        </h2>
        <div className="rounded-xl bg-gray-900/60 border border-gray-800 p-6 space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div className="rounded-lg bg-green-500/10 border border-green-500/20 p-4">
              <p className="text-xs text-green-400 mb-1">Most Engaged</p>
              <p className="text-lg font-semibold">{comparative.engagement_patterns.most_engaged_demographic}</p>
            </div>
            <div className="rounded-lg bg-red-500/10 border border-red-500/20 p-4">
              <p className="text-xs text-red-400 mb-1">Least Engaged</p>
              <p className="text-lg font-semibold">{comparative.engagement_patterns.least_engaged_demographic}</p>
            </div>
          </div>
          <p className="text-gray-400 text-sm">{comparative.engagement_patterns.engagement_summary}</p>
        </div>
      </section>

      {/* Recommendations */}
      <section>
        <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <span className="inline-block w-1.5 h-5 rounded-full bg-cyan-500" />
          Prioritized Recommendations
        </h2>
        <div className="space-y-3">
          {comparative.recommendations.map((rec, i) => (
            <RecommendationCard key={i} rec={rec} index={i} />
          ))}
        </div>
      </section>

      {/* Accessibility */}
      {comparative.accessibility_findings.length > 0 && (
        <section>
          <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <span className="inline-block w-1.5 h-5 rounded-full bg-amber-500" />
            Accessibility Findings
          </h2>
          <div className="space-y-2">
            {comparative.accessibility_findings.map((finding, i) => (
              <div key={i} className="flex items-start gap-3 rounded-lg bg-amber-500/5 border border-amber-500/20 p-4">
                <Globe className="w-4 h-4 text-amber-400 mt-0.5 shrink-0" />
                <p className="text-gray-300 text-sm">{finding}</p>
              </div>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}

/* ─── AGENT CARD ─── */

function AgentCard({ agent, colorIdx }: { agent: AgentData; colorIdx: number }) {
  const s = agent.summary;
  const successRate = s.total_steps > 0 ? ((s.successful_actions / s.total_steps) * 100).toFixed(0) : "0";
  const frustrations = s.impressions.filter((imp) => imp.sentiment === "frustration").length;
  const positives = s.impressions.filter((imp) => imp.sentiment === "positive").length;

  return (
    <div
      className="rounded-xl border border-gray-800 p-5 space-y-4"
      style={{ backgroundColor: AGENT_BG[colorIdx] }}
    >
      <div className="flex items-center gap-3">
        <div
          className="w-8 h-8 rounded-lg flex items-center justify-center"
          style={{ backgroundColor: AGENT_COLORS[colorIdx] + "30" }}
        >
          <Bot className="w-4 h-4" style={{ color: AGENT_COLORS[colorIdx] }} />
        </div>
        <div>
          <h3 className="font-semibold text-sm">{agent.demographic}</h3>
          <p className="text-xs text-gray-500">
            {agent.traits.age_group} · {agent.traits.country}
          </p>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <Stat label="Steps" value={s.total_steps} icon={Zap} />
        <Stat label="Duration" value={`${s.total_duration_s.toFixed(0)}s`} icon={Clock} />
        <Stat label="Success" value={`${successRate}%`} icon={CheckCircle} color="text-green-400" />
        <Stat label="Failed" value={s.failed_actions} icon={XCircle} color="text-red-400" />
      </div>

      <div className="flex gap-2 text-xs">
        <span className="px-2 py-1 rounded bg-green-500/10 text-green-400">
          {positives} positive
        </span>
        <span className="px-2 py-1 rounded bg-red-500/10 text-red-400">
          {frustrations} frustration
        </span>
      </div>

      <p className="text-xs text-gray-500 leading-relaxed line-clamp-2">
        {agent.description}
      </p>
    </div>
  );
}

function Stat({
  label,
  value,
  icon: Icon,
  color = "text-gray-100",
}: {
  label: string;
  value: string | number;
  icon: React.ElementType;
  color?: string;
}) {
  return (
    <div className="space-y-0.5">
      <p className="text-[10px] text-gray-500 uppercase tracking-wider">{label}</p>
      <div className="flex items-center gap-1.5">
        <Icon className={`w-3.5 h-3.5 ${color}`} />
        <span className={`text-sm font-semibold ${color}`}>{value}</span>
      </div>
    </div>
  );
}

/* ─── PERFORMANCE CHART ─── */

function PerformanceChart({ agents }: { agents: AgentData[] }) {
  const chartData = agents.map((a, i) => ({
    name: a.demographic.split(" ").slice(0, 2).join(" "),
    steps: a.summary.total_steps,
    success: a.summary.successful_actions,
    failed: a.summary.failed_actions,
    duration: Math.round(a.summary.total_duration_s),
  }));

  return (
    <div className="rounded-xl bg-gray-900/60 border border-gray-800 p-6">
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={chartData} barCategoryGap="20%">
          <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
          <XAxis dataKey="name" tick={{ fill: "#9ca3af", fontSize: 12 }} />
          <YAxis tick={{ fill: "#9ca3af", fontSize: 12 }} />
          <Tooltip
            contentStyle={{
              backgroundColor: "#111827",
              border: "1px solid #374151",
              borderRadius: "8px",
              fontSize: "12px",
            }}
          />
          <Legend wrapperStyle={{ fontSize: "12px" }} />
          <Bar dataKey="success" name="Successful" fill="#10b981" radius={[4, 4, 0, 0]} />
          <Bar dataKey="failed" name="Failed" fill="#ef4444" radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

/* ─── RECOMMENDATION CARD ─── */

function RecommendationCard({
  rec,
  index,
}: {
  rec: ComparativeData["recommendations"][0];
  index: number;
}) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="rounded-xl bg-gray-900/60 border border-gray-800 overflow-hidden">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center gap-4 p-4 text-left hover:bg-gray-800/30 transition-colors"
      >
        <div className="flex items-center gap-3 flex-1 min-w-0">
          <span className="text-lg font-bold text-gray-600">#{index + 1}</span>
          <div
            className={`w-2.5 h-2.5 rounded-full shrink-0 ${IMPACT_COLORS[rec.impact] || IMPACT_COLORS.medium}`}
          />
          <span className="font-medium truncate">{rec.title}</span>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          <span className="text-xs text-gray-500 uppercase">{rec.impact} impact</span>
          {expanded ? (
            <ChevronUp className="w-4 h-4 text-gray-500" />
          ) : (
            <ChevronDown className="w-4 h-4 text-gray-500" />
          )}
        </div>
      </button>
      {expanded && (
        <div className="px-4 pb-4 pt-0 space-y-2">
          <p className="text-sm text-gray-400">{rec.detail}</p>
          <div className="flex gap-2 flex-wrap">
            {rec.demographics_affected.map((d) => (
              <span
                key={d}
                className="text-xs px-2 py-0.5 rounded-full bg-gray-800 text-gray-400"
              >
                {d}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

/* ─── AGENT DETAIL VIEW ─── */

function AgentDetailView({
  agent,
  colorIdx,
}: {
  agent: AgentData;
  colorIdx: number;
}) {
  const s = agent.summary;
  const color = AGENT_COLORS[colorIdx];

  return (
    <div className="space-y-10">
      {/* Agent Header */}
      <div
        className="rounded-xl border border-gray-800 p-6"
        style={{ backgroundColor: AGENT_BG[colorIdx] }}
      >
        <div className="flex items-center gap-4 mb-4">
          <div
            className="w-12 h-12 rounded-xl flex items-center justify-center"
            style={{ backgroundColor: color + "30" }}
          >
            <Bot className="w-6 h-6" style={{ color }} />
          </div>
          <div>
            <h2 className="text-xl font-bold">{agent.demographic}</h2>
            <p className="text-sm text-gray-500">
              {agent.traits.age_group} · {agent.traits.country} · NYC: {agent.traits.nyc_familiarity}
            </p>
          </div>
        </div>
        <p className="text-gray-400">{agent.description}</p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard label="Total Steps" value={s.total_steps} color={color} />
        <StatCard label="Duration" value={`${s.total_duration_s.toFixed(0)}s`} color={color} />
        <StatCard label="Successful" value={s.successful_actions} color="#10b981" />
        <StatCard label="Failed" value={s.failed_actions} color="#ef4444" />
      </div>

      {/* Action Timeline */}
      <section>
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <span className="inline-block w-1.5 h-5 rounded-full" style={{ backgroundColor: color }} />
          Exploration Timeline
        </h3>
        <div className="rounded-xl bg-gray-900/60 border border-gray-800 p-4 space-y-1 max-h-96 overflow-y-auto custom-scrollbar">
          {s.action_history.map((action, i) => (
            <div
              key={i}
              className="flex items-center gap-3 py-2 px-3 rounded-lg hover:bg-gray-800/30 transition-colors"
            >
              <span className="text-xs text-gray-600 font-mono w-12 shrink-0">
                {action.elapsed_s.toFixed(1)}s
              </span>
              <div
                className="w-6 h-6 rounded-full flex items-center justify-center shrink-0"
                style={{
                  backgroundColor: action.success ? "#10b98120" : "#ef444420",
                }}
              >
                {action.success ? (
                  <CheckCircle className="w-3.5 h-3.5 text-green-400" />
                ) : (
                  <XCircle className="w-3.5 h-3.5 text-red-400" />
                )}
              </div>
              <div className="flex-1 min-w-0">
                <span className="text-sm font-medium capitalize">{action.action}</span>
                <span className="text-xs text-gray-500 ml-2 truncate">
                  {action.target.slice(0, 50)}
                </span>
              </div>
              {action.error && (
                <span className="text-xs text-red-400 shrink-0 max-w-48 truncate">
                  {action.error}
                </span>
              )}
            </div>
          ))}
        </div>
      </section>

      {/* Impressions */}
      <section>
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <span className="inline-block w-1.5 h-5 rounded-full" style={{ backgroundColor: color }} />
          Agent Impressions
        </h3>
        <div className="space-y-3">
          {s.impressions.map((imp, i) => {
            const sentimentConfig: Record<string, { color: string; bg: string }> = {
              positive: { color: "text-green-400", bg: "bg-green-500/10 border-green-500/20" },
              frustration: { color: "text-red-400", bg: "bg-red-500/10 border-red-500/20" },
              confusion: { color: "text-amber-400", bg: "bg-amber-500/10 border-amber-500/20" },
              neutral: { color: "text-gray-400", bg: "bg-gray-500/10 border-gray-500/20" },
              summary: { color: "text-cyan-400", bg: "bg-cyan-500/10 border-cyan-500/20" },
            };
            const cfg = sentimentConfig[imp.sentiment] || sentimentConfig.neutral;

            return (
              <div key={i} className={`rounded-lg border p-4 ${cfg.bg}`}>
                <div className="flex items-center gap-2 mb-1">
                  <span className="text-xs font-mono text-gray-500">{imp.elapsed_s.toFixed(1)}s</span>
                  <span className={`text-xs font-semibold uppercase ${cfg.color}`}>
                    {imp.sentiment}
                  </span>
                </div>
                <p className="text-sm text-gray-300">{imp.context}</p>
              </div>
            );
          })}
        </div>
      </section>

      {/* Narrative */}
      <section>
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <span className="inline-block w-1.5 h-5 rounded-full" style={{ backgroundColor: color }} />
          Narrative Report
        </h3>
        <div className="rounded-xl bg-gray-900/60 border border-gray-800 p-6">
          <div className="prose prose-invert prose-sm max-w-none">
            {agent.narrative.split("\n").map((para, i) =>
              para.trim() ? (
                <p key={i} className="text-gray-300 leading-relaxed mb-3">
                  {para.split(/(\*\*.*?\*\*)/).map((part, j) =>
                    part.startsWith("**") && part.endsWith("**") ? (
                      <strong key={j} className="text-gray-100">{part.slice(2, -2)}</strong>
                    ) : (
                      <span key={j}>{part}</span>
                    )
                  )}
                </p>
              ) : null
            )}
          </div>
        </div>
      </section>
    </div>
  );
}

function StatCard({
  label,
  value,
  color,
}: {
  label: string;
  value: string | number;
  color: string;
}) {
  return (
    <div className="rounded-xl bg-gray-900/60 border border-gray-800 p-4">
      <p className="text-xs text-gray-500 uppercase tracking-wider mb-1">{label}</p>
      <p className="text-2xl font-bold" style={{ color }}>
        {value}
      </p>
    </div>
  );
}

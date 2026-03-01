"use client";

import React, { useState, useEffect } from "react";
import {
  BarChart3,
  Users,
  Bot,
  CheckCircle,
  XCircle,
  AlertTriangle,
  Globe,
  Clock,
  Zap,
  Target,
  TrendingUp,
  ChevronDown,
  ChevronUp,
  FileText,
  Loader2,
} from "lucide-react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from "recharts";
import type { ReportData, AgentData, ComparativeData } from "../lib/types";

const AGENT_COLORS = ["#06b6d4", "#a855f7", "#f59e0b"];
const AGENT_BG = ["rgba(6,182,212,0.08)", "rgba(168,85,247,0.08)", "rgba(245,158,11,0.08)"];

const SEVERITY_STYLES: Record<string, string> = {
  high: "border-red-500/20 bg-red-500/5",
  medium: "border-amber-500/20 bg-amber-500/5",
  low: "border-green-500/20 bg-green-500/5",
};

const SEVERITY_TEXT: Record<string, string> = {
  high: "var(--red)",
  medium: "var(--amber)",
  low: "var(--accent)",
};

const IMPACT_BG: Record<string, string> = {
  high: "var(--red)",
  medium: "var(--amber)",
  low: "var(--accent)",
};

export default function ResultsView() {
  const [report, setReport] = useState<ReportData | null>(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<"overview" | number>("overview");

  useEffect(() => {
    async function loadReport() {
      let loaded = false;
      try {
        const res = await fetch("/api/report");
        if (res.ok) {
          setReport(await res.json());
          loaded = true;
        }
      } catch {
        // API not available
      }
      // Fallback to sample data if API didn't return a report
      if (!loaded) {
        try {
          const sample = await import("../data/sample_comparative.json");
          setReport(sample.default as unknown as ReportData);
        } catch {
          // No data available
        }
      }
      setLoading(false);
    }
    loadReport();
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <Loader2 className="w-6 h-6 animate-spin" style={{ color: "var(--muted)" }} />
      </div>
    );
  }

  if (!report) {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-4">
        <BarChart3 className="w-12 h-12" style={{ color: "var(--border)" }} />
        <p className="text-sm" style={{ color: "var(--muted)" }}>
          No results yet. Run the pipeline to generate agent reports.
        </p>
      </div>
    );
  }

  const agents = report.agents;
  const comparative = report.comparative;

  return (
    <div className="p-8 max-w-6xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <div className="flex items-center gap-3 mb-1">
            <BarChart3 className="w-6 h-6" style={{ color: "var(--accent)" }} />
            <h1 className="text-2xl font-bold">Agent Results</h1>
          </div>
          <p className="text-sm" style={{ color: "var(--muted)" }}>
            {report.target_app.name} — {agents.length} demographics analyzed
          </p>
        </div>
        <div className="text-right text-xs" style={{ color: "var(--muted)" }}>
          <p>Generated {new Date(report.generated_at).toLocaleString()}</p>
          <p style={{ color: "var(--cyan)" }}>{report.target_app.url}</p>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex gap-1 mb-8 border-b" style={{ borderColor: "var(--border)" }}>
        <TabButton
          active={activeTab === "overview"}
          onClick={() => setActiveTab("overview")}
          icon={Users}
          label="Comparative"
          color="var(--cyan)"
        />
        {agents.map((a, i) => (
          <TabButton
            key={i}
            active={activeTab === i}
            onClick={() => setActiveTab(i)}
            icon={Bot}
            label={a.demographic}
            color={AGENT_COLORS[i]}
          />
        ))}
      </div>

      {/* Content */}
      {activeTab === "overview" ? (
        <OverviewPanel agents={agents} comparative={comparative} />
      ) : (
        <AgentPanel agent={agents[activeTab]} colorIdx={activeTab} />
      )}
    </div>
  );
}

function TabButton({
  active,
  onClick,
  icon: Icon,
  label,
  color,
}: {
  active: boolean;
  onClick: () => void;
  icon: React.ElementType;
  label: string;
  color: string;
}) {
  return (
    <button
      onClick={onClick}
      className="flex items-center gap-2 px-4 py-2.5 text-xs font-medium rounded-t-lg transition-colors whitespace-nowrap cursor-pointer"
      style={{
        background: active ? "var(--surface)" : "transparent",
        color: active ? color : "var(--muted)",
        borderBottom: active ? `2px solid ${color}` : "2px solid transparent",
      }}
    >
      <Icon className="w-3.5 h-3.5" />
      {label}
    </button>
  );
}

/* ─── OVERVIEW PANEL ─── */

function OverviewPanel({ agents, comparative }: { agents: AgentData[]; comparative: ComparativeData }) {
  const [expandedRec, setExpandedRec] = useState<number | null>(null);

  return (
    <div className="space-y-8">
      {/* Overall Assessment */}
      <div
        className="rounded-xl p-6 border"
        style={{
          background: "linear-gradient(135deg, rgba(6,182,212,0.05), rgba(168,85,247,0.05), rgba(245,158,11,0.05))",
          borderColor: "var(--border)",
        }}
      >
        <div className="flex items-center gap-2 mb-3">
          <Target className="w-5 h-5" style={{ color: "var(--cyan)" }} />
          <h2 className="text-lg font-semibold">Overall Assessment</h2>
        </div>
        <p className="text-sm leading-relaxed" style={{ color: "var(--muted)" }}>
          {comparative.overall_assessment}
        </p>
      </div>

      {/* Agent Cards */}
      <section>
        <SectionHeader icon={Users} label="Agent Comparison" color="var(--cyan)" />
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {agents.map((agent, i) => (
            <AgentCard key={i} agent={agent} colorIdx={i} />
          ))}
        </div>
      </section>

      {/* Performance Chart */}
      <section>
        <SectionHeader icon={TrendingUp} label="Performance Comparison" color="var(--amber)" />
        <div className="rounded-xl p-6 border" style={{ background: "var(--surface)", borderColor: "var(--border)" }}>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart
              data={agents.map((a) => ({
                name: a.demographic.split(" ").slice(0, 2).join(" "),
                success: a.summary.successful_actions,
                failed: a.summary.failed_actions,
              }))}
              barCategoryGap="25%"
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
              <XAxis dataKey="name" tick={{ fill: "#94A3B8", fontSize: 11 }} />
              <YAxis tick={{ fill: "#94A3B8", fontSize: 11 }} />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#0F172A",
                  border: "1px solid #334155",
                  borderRadius: "8px",
                  fontSize: "12px",
                  color: "#F8FAFC",
                }}
              />
              <Legend wrapperStyle={{ fontSize: "11px" }} />
              <Bar dataKey="success" name="Successful" fill="#22C55E" radius={[4, 4, 0, 0]} />
              <Bar dataKey="failed" name="Failed" fill="#EF4444" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>

          {/* Engagement summary */}
          <div className="grid grid-cols-2 gap-3 mt-4">
            <div className="rounded-lg p-3 border" style={{ background: "rgba(34,197,94,0.05)", borderColor: "rgba(34,197,94,0.2)" }}>
              <p className="text-[10px] uppercase tracking-wider" style={{ color: "var(--accent)" }}>Most Engaged</p>
              <p className="text-sm font-semibold mt-0.5">{comparative.engagement_patterns.most_engaged_demographic}</p>
            </div>
            <div className="rounded-lg p-3 border" style={{ background: "rgba(239,68,68,0.05)", borderColor: "rgba(239,68,68,0.2)" }}>
              <p className="text-[10px] uppercase tracking-wider" style={{ color: "var(--red)" }}>Least Engaged</p>
              <p className="text-sm font-semibold mt-0.5">{comparative.engagement_patterns.least_engaged_demographic}</p>
            </div>
          </div>
          <p className="text-xs mt-3" style={{ color: "var(--muted)" }}>
            {comparative.engagement_patterns.engagement_summary}
          </p>
        </div>
      </section>

      {/* Common Friction */}
      <section>
        <SectionHeader icon={AlertTriangle} label="Common Friction Points" color="var(--red)" />
        <div className="space-y-2">
          {comparative.common_friction_points.map((point, i) => (
            <div
              key={i}
              className="flex items-start gap-3 rounded-lg border p-4"
              style={{ background: "rgba(239,68,68,0.03)", borderColor: "rgba(239,68,68,0.15)" }}
            >
              <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" style={{ color: "var(--red)" }} />
              <p className="text-sm" style={{ color: "var(--muted)" }}>{point}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Demographic Issues */}
      <section>
        <SectionHeader icon={Globe} label="Demographic-Specific Issues" color="var(--purple)" />
        <div className="space-y-2">
          {comparative.demographic_specific_issues.map((issue, i) => (
            <div
              key={i}
              className={`flex items-start gap-3 rounded-lg border p-4 ${SEVERITY_STYLES[issue.severity] || ""}`}
            >
              <span
                className="text-[10px] font-bold uppercase tracking-wider mt-0.5 shrink-0 w-14"
                style={{ color: SEVERITY_TEXT[issue.severity] || "var(--muted)" }}
              >
                {issue.severity}
              </span>
              <div>
                <p className="text-[10px] mb-0.5" style={{ color: "var(--muted)" }}>{issue.demographic}</p>
                <p className="text-sm" style={{ color: "var(--foreground)" }}>{issue.issue}</p>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Recommendations */}
      <section>
        <SectionHeader icon={Zap} label="Prioritized Recommendations" color="var(--accent)" />
        <div className="space-y-2">
          {comparative.recommendations.map((rec, i) => (
            <div
              key={i}
              className="rounded-xl border overflow-hidden"
              style={{ background: "var(--surface)", borderColor: "var(--border)" }}
            >
              <button
                onClick={() => setExpandedRec(expandedRec === i ? null : i)}
                className="w-full flex items-center gap-4 px-5 py-3.5 text-left cursor-pointer hover:brightness-110 transition-all"
              >
                <span className="text-base font-bold" style={{ color: "var(--border)" }}>
                  #{i + 1}
                </span>
                <div
                  className="w-2 h-2 rounded-full shrink-0"
                  style={{ background: IMPACT_BG[rec.impact] || "var(--muted)" }}
                />
                <span className="text-sm font-medium flex-1">{rec.title}</span>
                <span className="text-[10px] uppercase" style={{ color: "var(--muted)" }}>
                  {rec.impact}
                </span>
                {expandedRec === i ? (
                  <ChevronUp className="w-4 h-4" style={{ color: "var(--muted)" }} />
                ) : (
                  <ChevronDown className="w-4 h-4" style={{ color: "var(--muted)" }} />
                )}
              </button>
              {expandedRec === i && (
                <div className="px-5 pb-4 border-t" style={{ borderColor: "var(--border)" }}>
                  <p className="text-xs mt-3 leading-relaxed" style={{ color: "var(--muted)" }}>
                    {rec.detail}
                  </p>
                  <div className="flex gap-2 mt-3 flex-wrap">
                    {rec.demographics_affected.map((d) => (
                      <span key={d} className="text-[10px] px-2 py-0.5 rounded-full" style={{ background: "var(--surface-2)", color: "var(--muted)" }}>
                        {d}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      </section>

      {/* Accessibility */}
      {comparative.accessibility_findings?.length > 0 && (
        <section>
          <SectionHeader icon={Globe} label="Accessibility Findings" color="var(--amber)" />
          <div className="space-y-2">
            {comparative.accessibility_findings.map((finding, i) => (
              <div
                key={i}
                className="flex items-start gap-3 rounded-lg border p-4"
                style={{ background: "rgba(245,158,11,0.03)", borderColor: "rgba(245,158,11,0.15)" }}
              >
                <Globe className="w-4 h-4 shrink-0 mt-0.5" style={{ color: "var(--amber)" }} />
                <p className="text-xs" style={{ color: "var(--muted)" }}>{finding}</p>
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
  const color = AGENT_COLORS[colorIdx];
  const successRate = s.total_steps > 0 ? ((s.successful_actions / s.total_steps) * 100).toFixed(0) : "0";
  const frustrations = s.impressions.filter((imp) => imp.sentiment === "frustration").length;
  const positives = s.impressions.filter((imp) => imp.sentiment === "positive").length;

  const pieData = [
    { name: "Success", value: s.successful_actions, fill: "#22C55E" },
    { name: "Failed", value: s.failed_actions, fill: "#EF4444" },
  ];

  return (
    <div className="rounded-xl border p-5 space-y-4" style={{ background: AGENT_BG[colorIdx], borderColor: "var(--border)" }}>
      <div className="flex items-center gap-3">
        <div className="w-8 h-8 rounded-lg flex items-center justify-center" style={{ background: `${color}20` }}>
          <Bot className="w-4 h-4" style={{ color }} />
        </div>
        <div>
          <h3 className="font-semibold text-sm">{agent.demographic}</h3>
          <p className="text-[10px]" style={{ color: "var(--muted)" }}>
            {agent.traits.age_group} · {agent.traits.country}
          </p>
        </div>
      </div>

      {/* Mini pie + stats */}
      <div className="flex items-center gap-4">
        <div className="w-16 h-16">
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie data={pieData} cx="50%" cy="50%" innerRadius={18} outerRadius={28} dataKey="value" strokeWidth={0}>
                {pieData.map((entry, i) => (
                  <Cell key={i} fill={entry.fill} />
                ))}
              </Pie>
            </PieChart>
          </ResponsiveContainer>
        </div>
        <div className="grid grid-cols-2 gap-x-6 gap-y-1 flex-1">
          <MiniStatLine label="Steps" value={s.total_steps} icon={Zap} />
          <MiniStatLine label="Duration" value={`${s.total_duration_s.toFixed(0)}s`} icon={Clock} />
          <MiniStatLine label="Success" value={`${successRate}%`} icon={CheckCircle} color="var(--accent)" />
          <MiniStatLine label="Failed" value={s.failed_actions} icon={XCircle} color="var(--red)" />
        </div>
      </div>

      {/* Sentiment pills */}
      <div className="flex gap-2">
        <span className="text-[10px] px-2 py-0.5 rounded-full" style={{ background: "rgba(34,197,94,0.1)", color: "var(--accent)" }}>
          {positives} positive
        </span>
        <span className="text-[10px] px-2 py-0.5 rounded-full" style={{ background: "rgba(239,68,68,0.1)", color: "var(--red)" }}>
          {frustrations} frustration
        </span>
      </div>

      <p className="text-[10px] leading-relaxed line-clamp-2" style={{ color: "var(--muted)" }}>
        {agent.description}
      </p>
    </div>
  );
}

function MiniStatLine({
  label,
  value,
  icon: Icon,
  color = "var(--foreground)",
}: {
  label: string;
  value: string | number;
  icon: React.ElementType;
  color?: string;
}) {
  return (
    <div className="flex items-center gap-1.5">
      <Icon className="w-3 h-3" style={{ color }} />
      <span className="text-[10px]" style={{ color: "var(--muted)" }}>{label}</span>
      <span className="text-xs font-semibold ml-auto" style={{ color }}>{value}</span>
    </div>
  );
}

/* ─── AGENT DETAIL PANEL ─── */

function AgentPanel({ agent, colorIdx }: { agent: AgentData; colorIdx: number }) {
  const s = agent.summary;
  const color = AGENT_COLORS[colorIdx];

  return (
    <div className="space-y-8">
      {/* Agent Header */}
      <div className="rounded-xl border p-6" style={{ background: AGENT_BG[colorIdx], borderColor: "var(--border)" }}>
        <div className="flex items-center gap-4 mb-3">
          <div className="w-12 h-12 rounded-xl flex items-center justify-center" style={{ background: `${color}20` }}>
            <Bot className="w-6 h-6" style={{ color }} />
          </div>
          <div>
            <h2 className="text-xl font-bold">{agent.demographic}</h2>
            <p className="text-xs" style={{ color: "var(--muted)" }}>
              {agent.traits.age_group} · {agent.traits.country} · NYC: {agent.traits.nyc_familiarity}
            </p>
          </div>
        </div>
        <p className="text-sm" style={{ color: "var(--muted)" }}>{agent.description}</p>
      </div>

      {/* Stats row */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <StatCard label="Steps" value={s.total_steps} color={color} />
        <StatCard label="Duration" value={`${s.total_duration_s.toFixed(0)}s`} color={color} />
        <StatCard label="Success" value={s.successful_actions} color="var(--accent)" />
        <StatCard label="Failed" value={s.failed_actions} color="var(--red)" />
      </div>

      {/* Timeline */}
      <section>
        <SectionHeader icon={Clock} label="Exploration Timeline" color={color} />
        <div
          className="rounded-xl border p-4 max-h-96 overflow-y-auto"
          style={{ background: "var(--surface)", borderColor: "var(--border)" }}
        >
          {s.action_history.map((action, i) => (
            <div
              key={i}
              className="flex items-center gap-3 py-2 px-3 rounded-lg hover:brightness-110 transition-colors"
            >
              <span className="text-[10px] font-mono w-10 shrink-0" style={{ color: "var(--muted)" }}>
                {action.elapsed_s.toFixed(1)}s
              </span>
              <div
                className="w-5 h-5 rounded-full flex items-center justify-center shrink-0"
                style={{ background: action.success ? "rgba(34,197,94,0.15)" : "rgba(239,68,68,0.15)" }}
              >
                {action.success ? (
                  <CheckCircle className="w-3 h-3" style={{ color: "var(--accent)" }} />
                ) : (
                  <XCircle className="w-3 h-3" style={{ color: "var(--red)" }} />
                )}
              </div>
              <div className="flex-1 min-w-0">
                <span className="text-xs font-medium capitalize">{action.action}</span>
                <span className="text-[10px] ml-2 truncate" style={{ color: "var(--muted)" }}>
                  {action.target.slice(0, 50)}
                </span>
              </div>
              {action.error && (
                <span className="text-[10px] shrink-0 max-w-40 truncate" style={{ color: "var(--red)" }}>
                  {action.error}
                </span>
              )}
            </div>
          ))}
        </div>
      </section>

      {/* Impressions */}
      <section>
        <SectionHeader icon={FileText} label="Agent Impressions" color={color} />
        <div className="space-y-2">
          {s.impressions.map((imp, i) => {
            const sentimentColors: Record<string, { text: string; bg: string; border: string }> = {
              positive: { text: "var(--accent)", bg: "rgba(34,197,94,0.05)", border: "rgba(34,197,94,0.15)" },
              frustration: { text: "var(--red)", bg: "rgba(239,68,68,0.05)", border: "rgba(239,68,68,0.15)" },
              confusion: { text: "var(--amber)", bg: "rgba(245,158,11,0.05)", border: "rgba(245,158,11,0.15)" },
              neutral: { text: "var(--muted)", bg: "rgba(148,163,184,0.05)", border: "rgba(148,163,184,0.15)" },
              summary: { text: "var(--cyan)", bg: "rgba(6,182,212,0.05)", border: "rgba(6,182,212,0.15)" },
            };
            const cfg = sentimentColors[imp.sentiment] || sentimentColors.neutral;

            return (
              <div key={i} className="rounded-lg border p-4" style={{ background: cfg.bg, borderColor: cfg.border }}>
                <div className="flex items-center gap-2 mb-1">
                  <span className="text-[10px] font-mono" style={{ color: "var(--muted)" }}>{imp.elapsed_s.toFixed(1)}s</span>
                  <span className="text-[10px] font-bold uppercase" style={{ color: cfg.text }}>{imp.sentiment}</span>
                </div>
                <p className="text-xs" style={{ color: "var(--foreground)" }}>{imp.context}</p>
              </div>
            );
          })}
        </div>
      </section>

      {/* Narrative */}
      <section>
        <SectionHeader icon={FileText} label="Narrative Report" color={color} />
        <div className="rounded-xl border p-6" style={{ background: "var(--surface)", borderColor: "var(--border)" }}>
          {agent.narrative.split("\n").map((para, i) =>
            para.trim() ? (
              <p key={i} className="text-sm leading-relaxed mb-3" style={{ color: "var(--muted)" }}>
                {para.split(/(\*\*.*?\*\*)/).map((part, j) =>
                  part.startsWith("**") && part.endsWith("**") ? (
                    <strong key={j} style={{ color: "var(--foreground)" }}>{part.slice(2, -2)}</strong>
                  ) : (
                    <span key={j}>{part}</span>
                  )
                )}
              </p>
            ) : null
          )}
        </div>
      </section>
    </div>
  );
}

/* ─── SHARED ─── */

function SectionHeader({ icon: Icon, label, color }: { icon: React.ElementType; label: string; color: string }) {
  return (
    <h3 className="text-base font-semibold mb-4 flex items-center gap-2">
      <span className="inline-block w-1 h-5 rounded-full" style={{ background: color }} />
      <Icon className="w-4 h-4" style={{ color }} />
      {label}
    </h3>
  );
}

function StatCard({ label, value, color }: { label: string; value: string | number; color: string }) {
  return (
    <div className="rounded-xl border p-4" style={{ background: "var(--surface)", borderColor: "var(--border)" }}>
      <p className="text-[10px] uppercase tracking-wider mb-1" style={{ color: "var(--muted)" }}>{label}</p>
      <p className="text-2xl font-bold" style={{ color }}>{value}</p>
    </div>
  );
}

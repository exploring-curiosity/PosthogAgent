"use client";

import React from "react";
import {
  LogIn,
  Search,
  FileText,
  MessageSquare,
  ThumbsUp,
  Home,
  Compass,
  TrendingUp,
  PenSquare,
  HelpCircle,
} from "lucide-react";

interface Action {
  action_name: string;
  relative_s: number;
  duration_s: number;
  success: boolean;
  error: string;
  page_url: string;
  details: Record<string, unknown>;
}

interface Props {
  actions: Action[];
  totalDuration: number;
}

const ACTION_ICONS: Record<string, React.ReactNode> = {
  signup: <LogIn className="w-3.5 h-3.5" />,
  login: <LogIn className="w-3.5 h-3.5" />,
  scan_feed: <Search className="w-3.5 h-3.5" />,
  open_post: <FileText className="w-3.5 h-3.5" />,
  write_comment: <MessageSquare className="w-3.5 h-3.5" />,
  vote_on_post: <ThumbsUp className="w-3.5 h-3.5" />,
  return_to_feed: <Home className="w-3.5 h-3.5" />,
  browse_subreddit: <Compass className="w-3.5 h-3.5" />,
  open_related_post: <TrendingUp className="w-3.5 h-3.5" />,
  create_post: <PenSquare className="w-3.5 h-3.5" />,
};

const ACTION_COLORS: Record<string, string> = {
  signup: "bg-emerald-500",
  login: "bg-emerald-500",
  scan_feed: "bg-blue-400",
  open_post: "bg-cyan-500",
  write_comment: "bg-purple-500",
  vote_on_post: "bg-amber-500",
  return_to_feed: "bg-gray-500",
  browse_subreddit: "bg-teal-500",
  open_related_post: "bg-indigo-500",
  create_post: "bg-pink-500",
};

function getActionLabel(action: Action): string {
  const d = action.details;
  switch (action.action_name) {
    case "signup":
      return d.username ? `Signed up as ${d.username}` : "Signed up";
    case "open_post":
      return d.post_title ? `Opened "${d.post_title}"` : "Opened a post";
    case "write_comment":
      return d.skipped
        ? "Comment skipped (probability)"
        : `Wrote comment (${d.comment_length || "?"} chars)`;
    case "vote_on_post":
      return d.skipped ? "Vote skipped (probability)" : "Voted on post";
    case "browse_subreddit":
      return d.subreddit
        ? `Browsed r/${d.subreddit}`
        : "Browsed subreddit";
    case "create_post":
      return "Created a post";
    case "scan_feed":
      return `Scanned feed (${d.pattern || "default"})`;
    case "return_to_feed":
      return "Returned to feed";
    default:
      return action.action_name;
  }
}

export default function ActionTimeline({ actions, totalDuration }: Props) {
  return (
    <div className="rounded-xl border border-gray-800 bg-[#111] overflow-hidden">
      {/* Timeline bar */}
      <div className="relative h-16 bg-[#0d0d0d] border-b border-gray-800 px-4 flex items-center">
        {/* Time markers */}
        {[0, 25, 50, 75, 100].map((pct) => (
          <div
            key={pct}
            className="absolute top-0 h-full border-l border-gray-800"
            style={{ left: `${pct}%` }}
          >
            <span className="absolute -bottom-0 left-1 text-[10px] text-gray-600">
              {Math.round((pct / 100) * totalDuration)}s
            </span>
          </div>
        ))}
        {/* Action blocks */}
        {actions.map((action, i) => {
          const left = (action.relative_s / totalDuration) * 100;
          const width = Math.max(
            ((action.duration_s || 0.5) / totalDuration) * 100,
            0.8
          );
          const color = ACTION_COLORS[action.action_name] || "bg-gray-500";
          const failed = !action.success;

          return (
            <div
              key={i}
              className={`absolute top-3 h-8 rounded-sm ${color} ${
                failed ? "opacity-40 ring-2 ring-red-500" : "opacity-80"
              } hover:opacity-100 transition-opacity cursor-pointer group`}
              style={{
                left: `${left}%`,
                width: `${width}%`,
                minWidth: "6px",
              }}
              title={`${action.action_name} @ ${action.relative_s.toFixed(1)}s (${action.duration_s.toFixed(1)}s)`}
            >
              <div className="absolute -top-10 left-1/2 -translate-x-1/2 whitespace-nowrap bg-gray-900 border border-gray-700 rounded px-2 py-1 text-[10px] text-gray-300 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-10">
                {action.action_name} — {action.duration_s.toFixed(1)}s
                {failed && " ❌"}
              </div>
            </div>
          );
        })}
      </div>

      {/* Legend */}
      <div className="px-4 py-2 flex flex-wrap gap-3 text-[10px] text-gray-500 border-b border-gray-800">
        {Object.entries(ACTION_COLORS).map(([name, color]) => (
          <div key={name} className="flex items-center gap-1">
            <span className={`inline-block w-2.5 h-2.5 rounded-sm ${color}`} />
            {name.replace(/_/g, " ")}
          </div>
        ))}
      </div>

      {/* Action list */}
      <div className="max-h-[420px] overflow-y-auto divide-y divide-gray-800/50">
        {actions.map((action, i) => {
          const icon =
            ACTION_ICONS[action.action_name] || (
              <HelpCircle className="w-3.5 h-3.5" />
            );
          const color = ACTION_COLORS[action.action_name] || "bg-gray-500";
          const failed = !action.success;
          const skipped =
            action.details.skipped === true && action.duration_s === 0;

          return (
            <div
              key={i}
              className={`flex items-center gap-4 px-4 py-3 hover:bg-white/[0.02] transition-colors ${
                failed ? "bg-red-500/[0.04]" : ""
              }`}
            >
              {/* Step number */}
              <div className="w-7 text-right text-xs text-gray-600 font-mono shrink-0">
                {i + 1}
              </div>

              {/* Icon */}
              <div
                className={`w-7 h-7 rounded-md flex items-center justify-center shrink-0 ${color} bg-opacity-20`}
              >
                {icon}
              </div>

              {/* Content */}
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <span className="font-medium text-sm text-gray-200">
                    {action.action_name.replace(/_/g, " ")}
                  </span>
                  {failed && (
                    <span className="px-1.5 py-0.5 text-[10px] rounded bg-red-500/20 text-red-400 font-medium">
                      FAILED
                    </span>
                  )}
                  {skipped && (
                    <span className="px-1.5 py-0.5 text-[10px] rounded bg-gray-500/20 text-gray-400 font-medium">
                      SKIPPED
                    </span>
                  )}
                </div>
                <p className="text-xs text-gray-500 truncate mt-0.5">
                  {getActionLabel(action)}
                </p>
              </div>

              {/* Timing */}
              <div className="text-right shrink-0">
                <div className="text-xs font-mono text-gray-400">
                  {action.duration_s > 0
                    ? `${action.duration_s.toFixed(1)}s`
                    : "—"}
                </div>
                <div className="text-[10px] text-gray-600 font-mono">
                  @{action.relative_s.toFixed(1)}s
                </div>
              </div>

              {/* Duration bar */}
              <div className="w-24 shrink-0">
                <div className="h-1.5 bg-gray-800 rounded-full overflow-hidden">
                  <div
                    className={`h-full rounded-full ${
                      failed ? "bg-red-500" : color
                    }`}
                    style={{
                      width: `${Math.min(
                        (action.duration_s / 10) * 100,
                        100
                      )}%`,
                    }}
                  />
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

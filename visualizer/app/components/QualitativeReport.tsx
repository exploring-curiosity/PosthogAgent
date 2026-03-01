"use client";

import React, { useState } from "react";
import {
  ChevronDown,
  ChevronRight,
  AlertTriangle,
  Accessibility,
  GitBranch,
  Gauge,
  Lightbulb,
} from "lucide-react";

interface Props {
  markdown: string;
}

interface Section {
  title: string;
  icon: React.ReactNode;
  color: string;
  content: string;
}

function parseQualitative(md: string): Section[] {
  const sections: Section[] = [];

  const sectionDefs = [
    {
      pattern: /###?\s*\*?\*?1\.\s*FRICTION\s*POINTS\*?\*?([\s\S]*?)(?=###?\s*\*?\*?2\.|$)/i,
      title: "Friction Points",
      icon: <AlertTriangle className="w-4 h-4" />,
      color: "text-red-400 border-red-500/30 bg-red-500/5",
    },
    {
      pattern: /###?\s*\*?\*?2\.\s*ACCESSIBILITY\s*ISSUES\*?\*?([\s\S]*?)(?=###?\s*\*?\*?3\.|$)/i,
      title: "Accessibility Issues",
      icon: <Accessibility className="w-4 h-4" />,
      color: "text-amber-400 border-amber-500/30 bg-amber-500/5",
    },
    {
      pattern: /###?\s*\*?\*?3\.\s*FLOW\s*PROBLEMS\*?\*?([\s\S]*?)(?=###?\s*\*?\*?4\.|$)/i,
      title: "Flow Problems",
      icon: <GitBranch className="w-4 h-4" />,
      color: "text-yellow-400 border-yellow-500/30 bg-yellow-500/5",
    },
    {
      pattern: /###?\s*\*?\*?4\.\s*PERFORMANCE\s*CONCERNS\*?\*?([\s\S]*?)(?=###?\s*\*?\*?5\.|$)/i,
      title: "Performance Concerns",
      icon: <Gauge className="w-4 h-4" />,
      color: "text-orange-400 border-orange-500/30 bg-orange-500/5",
    },
    {
      pattern: /###?\s*\*?\*?5\.\s*SPECIFIC\s*RECOMMENDATIONS[^*]*\*?\*?([\s\S]*?)(?=###?\s*\*?\*?Key\s*Take|$)/i,
      title: "Specific Recommendations",
      icon: <Lightbulb className="w-4 h-4" />,
      color: "text-cyan-400 border-cyan-500/30 bg-cyan-500/5",
    },
  ];

  for (const def of sectionDefs) {
    const match = md.match(def.pattern);
    if (match) {
      sections.push({
        title: def.title,
        icon: def.icon,
        color: def.color,
        content: match[1].trim(),
      });
    }
  }

  // Key Takeaway
  const takeawayMatch = md.match(
    /###?\s*\*?\*?Key\s*Takeaway\*?\*?([\s\S]*?)$/i
  );
  if (takeawayMatch) {
    sections.push({
      title: "Key Takeaway",
      icon: <Lightbulb className="w-4 h-4" />,
      color: "text-purple-400 border-purple-500/30 bg-purple-500/5",
    content: takeawayMatch[1].trim(),
    });
  }

  return sections;
}

function renderMarkdownContent(content: string) {
  const lines = content.split("\n");
  const elements: React.ReactNode[] = [];

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const trimmed = line.trim();

    if (!trimmed) {
      elements.push(<div key={i} className="h-2" />);
      continue;
    }

    // Sub-headers (####)
    if (trimmed.startsWith("####")) {
      const text = trimmed.replace(/^#{1,4}\s*/, "").replace(/\*\*/g, "");
      elements.push(
        <h4
          key={i}
          className="text-sm font-bold text-gray-200 mt-4 mb-2 flex items-center gap-2"
        >
          <span className="w-1 h-4 rounded-full bg-cyan-500 inline-block" />
          {text}
        </h4>
      );
      continue;
    }

    // Bold sub-points
    if (trimmed.startsWith("- **") || trimmed.startsWith("* **")) {
      const boldMatch = trimmed.match(
        /^[-*]\s*\*\*(.+?)\*\*[:\s]*([\s\S]*)/
      );
      if (boldMatch) {
        elements.push(
          <div key={i} className="flex items-start gap-2 mb-2 ml-2">
            <span className="text-cyan-500 mt-1 shrink-0">▸</span>
            <div>
              <span className="text-sm font-semibold text-gray-200">
                {boldMatch[1]}
              </span>
              {boldMatch[2] && (
                <span className="text-sm text-gray-400">
                  {" "}
                  {boldMatch[2]}
                </span>
              )}
            </div>
          </div>
        );
        continue;
      }
    }

    // Regular list items
    if (trimmed.startsWith("- ") || trimmed.startsWith("* ")) {
      const text = trimmed.replace(/^[-*]\s*/, "");
      elements.push(
        <div key={i} className="flex items-start gap-2 mb-1.5 ml-4">
          <span className="text-gray-600 mt-1 shrink-0">–</span>
          <span className="text-sm text-gray-400">{formatInline(text)}</span>
        </div>
      );
      continue;
    }

    // Indented sub-items
    if (trimmed.startsWith("  - ") || trimmed.startsWith("  * ")) {
      const text = trimmed.replace(/^\s*[-*]\s*/, "");
      elements.push(
        <div key={i} className="flex items-start gap-2 mb-1 ml-8">
          <span className="text-gray-700 mt-1 shrink-0">·</span>
          <span className="text-xs text-gray-500">{formatInline(text)}</span>
        </div>
      );
      continue;
    }

    // Evidence/italics
    if (trimmed.startsWith("*Evidence:*") || trimmed.startsWith("- *Evidence")) {
      elements.push(
        <div
          key={i}
          className="ml-6 mt-1 mb-2 text-xs text-gray-500 italic border-l-2 border-gray-800 pl-3"
        >
          {formatInline(trimmed.replace(/^[-*]\s*/, ""))}
        </div>
      );
      continue;
    }

    // Regular paragraph
    elements.push(
      <p key={i} className="text-sm text-gray-400 mb-1">
        {formatInline(trimmed)}
      </p>
    );
  }

  return elements;
}

function formatInline(text: string): React.ReactNode {
  // Very simple: bold **text** and code `text`
  const parts = text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g);
  return parts.map((part, i) => {
    if (part.startsWith("**") && part.endsWith("**")) {
      return (
        <span key={i} className="font-semibold text-gray-200">
          {part.slice(2, -2)}
        </span>
      );
    }
    if (part.startsWith("`") && part.endsWith("`")) {
      return (
        <code
          key={i}
          className="px-1 py-0.5 rounded bg-gray-800 text-cyan-400 text-xs font-mono"
        >
          {part.slice(1, -1)}
        </code>
      );
    }
    return part;
  });
}

function CollapsibleSection({ section }: { section: Section }) {
  const [open, setOpen] = useState(true);

  return (
    <div className={`rounded-xl border ${section.color} overflow-hidden`}>
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center gap-3 px-5 py-4 text-left hover:bg-white/[0.02] transition-colors"
      >
        {open ? (
          <ChevronDown className="w-4 h-4 text-gray-500 shrink-0" />
        ) : (
          <ChevronRight className="w-4 h-4 text-gray-500 shrink-0" />
        )}
        <span className={section.color.split(" ")[0]}>{section.icon}</span>
        <h3 className="text-sm font-semibold text-gray-200">
          {section.title}
        </h3>
      </button>
      {open && (
        <div className="px-5 pb-5 pt-0">
          {renderMarkdownContent(section.content)}
        </div>
      )}
    </div>
  );
}

export default function QualitativeReport({ markdown }: Props) {
  const sections = parseQualitative(markdown);

  if (sections.length === 0) {
    return (
      <div className="rounded-xl border border-gray-800 bg-[#111] p-6">
        <div className="prose prose-invert prose-sm max-w-none">
          <pre className="whitespace-pre-wrap text-sm text-gray-400">
            {markdown}
          </pre>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {sections.map((section, i) => (
        <CollapsibleSection key={i} section={section} />
      ))}
    </div>
  );
}

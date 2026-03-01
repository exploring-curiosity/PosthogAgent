"""
Stage 6: Feedback Report — Generates a combined quantitative + qualitative
UX feedback report after the agent completes its session.

Quantitative: timing deltas, completion rates, error counts, deviation scores.
Qualitative: LLM-generated UX improvement recommendations based on the agent's experience.
"""

import json
from datetime import datetime
from pathlib import Path
from mistralai import Mistral

from feedback.metrics import compute_metrics


def generate_qualitative_feedback(
    metrics: dict,
    agent_log: dict,
    real_user_description: str,
    api_key: str,
) -> str:
    """
    Use Mistral Large to generate qualitative UX recommendations
    based on the agent's session experience.
    """
    client = Mistral(api_key=api_key)

    # Build a concise summary of what happened
    errors_summary = ""
    for action, errs in metrics["errors"]["errors_by_action"].items():
        errors_summary += f"  - {action}: {', '.join(errs[:3])}\n"

    stuck_summary = ""
    for stuck in metrics["errors"]["stuck_details"]:
        stuck_summary += f"  - Stuck at {stuck.get('action', '?')} ({stuck.get('reason', '?')}) on {stuck.get('page_url', '?')}\n"

    timing_summary = ""
    for action, delta in metrics["timing"]["per_action_deltas"].items():
        ratio = delta.get("ratio", "N/A")
        timing_summary += f"  - {action}: agent {delta['agent_avg_s']}s vs real user {delta['real_user_avg_s']}s (ratio: {ratio}x)\n"

    prompt = f"""You are a UX researcher analyzing the results of an AI agent that mimicked a real user's behavior on FunCity, an NYC community discussion board.

The agent was given a behavioral policy derived from a real user's PostHog session recording and attempted to replicate the same interaction patterns on a sandbox instance of the app.

## REAL USER'S BEHAVIORAL PROFILE:
{real_user_description[:2000]}

## AGENT SESSION METRICS:

### Completion:
- Planned actions: {metrics['completion']['planned_actions']}
- Successfully executed: {metrics['completion']['successful_actions']}
- Failed: {metrics['completion']['failed_actions']}
- Completion rate: {metrics['completion']['completion_rate']}

### Timing:
- Agent session duration: {metrics['timing']['agent_duration_s']}s
- Real user session duration: {metrics['timing']['real_user_duration_s']}s
- Duration ratio: {metrics['timing']['duration_ratio']}x
{timing_summary}

### Errors:
- Total errors: {metrics['errors']['total_errors']}
- Stuck events: {metrics['errors']['total_stuck_events']}
{errors_summary}

### Stuck Events:
{stuck_summary if stuck_summary else "  None"}

### Policy Deviation:
- Match rate: {metrics['deviation']['match_rate'] if isinstance(metrics['deviation'], dict) else 'N/A'}
- Deviation score: {metrics['deviation']['deviation_score'] if isinstance(metrics['deviation'], dict) else 'N/A'}

## AGENT ACTION LOG (last 20 actions):
{_format_action_log(agent_log.get('actions', [])[-20:])}

Based on this analysis, provide UX improvement recommendations for the FunCity web app:

1. **FRICTION POINTS**: Where did the agent struggle? What does this suggest about the UI/UX?
2. **ACCESSIBILITY ISSUES**: Were any elements hard to find or interact with?
3. **FLOW PROBLEMS**: Did the agent's journey match the real user's? Where did it diverge and why?
4. **PERFORMANCE CONCERNS**: Were there timing issues that suggest slow loading or unresponsive UI?
5. **SPECIFIC RECOMMENDATIONS**: List 3-5 concrete, actionable UX improvements ranked by impact.

Be specific. Reference actual actions and metrics. Don't make generic UX advice — ground everything in the data."""

    response = client.chat.complete(
        model="mistral-large-latest",
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message.content


def _format_action_log(actions: list[dict]) -> str:
    """Format action log entries for the LLM prompt."""
    lines = []
    for a in actions:
        status = "OK" if a.get("success") else "FAILED"
        line = f"  [{a.get('relative_s', '?')}s] {a.get('action_name', '?')} - {status} ({a.get('duration_s', '?')}s)"
        if not a.get("success"):
            line += f" ERROR: {a.get('error', '')}"
        if a.get("details"):
            details_str = json.dumps(a["details"])
            if len(details_str) < 100:
                line += f" {details_str}"
        lines.append(line)
    return "\n".join(lines)


def generate_feedback_report(
    agent_log: dict,
    real_session_parsed: dict,
    agent_policy: dict,
    real_user_description: str,
    api_key: str,
    output_dir: str | None = None,
    session_id: str = "",
) -> tuple[dict, str]:
    """
    Generate the complete feedback report (quantitative + qualitative).

    Returns (report_json, report_markdown).
    """
    # ── Quantitative ──
    metrics = compute_metrics(agent_log, real_session_parsed, agent_policy)

    # ── Qualitative ──
    qualitative = generate_qualitative_feedback(
        metrics, agent_log, real_user_description, api_key
    )

    # ── Assemble JSON report ──
    report_json = {
        "session_id": session_id,
        "generated_at": datetime.now().isoformat(),
        "metrics": metrics,
        "qualitative_feedback": qualitative,
    }

    # ── Assemble Markdown report ──
    report_md = _build_markdown_report(report_json, metrics, qualitative, session_id)

    # ── Save if output_dir provided ──
    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        json_path = out / f"feedback_report_{session_id}.json"
        md_path = out / f"feedback_report_{session_id}.md"

        with open(json_path, "w") as f:
            json.dump(report_json, f, indent=2)
        with open(md_path, "w") as f:
            f.write(report_md)

        print(f"  Saved JSON report to {json_path}")
        print(f"  Saved Markdown report to {md_path}")

    return report_json, report_md


def _build_markdown_report(report: dict, metrics: dict, qualitative: str, session_id: str) -> str:
    """Build a human-readable markdown feedback report."""
    lines = [
        f"# FunCity UX Feedback Report",
        f"**Session:** {session_id}",
        f"**Generated:** {report.get('generated_at', 'unknown')}",
        "",
        "---",
        "",
        "## Quantitative Metrics",
        "",
        "### Completion",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Planned actions | {metrics['completion']['planned_actions']} |",
        f"| Executed | {metrics['completion']['executed_actions']} |",
        f"| Successful | {metrics['completion']['successful_actions']} |",
        f"| Failed | {metrics['completion']['failed_actions']} |",
        f"| Completion rate | {metrics['completion']['completion_rate']} |",
        "",
        "### Timing",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Agent duration | {metrics['timing']['agent_duration_s']}s |",
        f"| Real user duration | {metrics['timing']['real_user_duration_s']}s |",
        f"| Duration ratio | {metrics['timing']['duration_ratio']}x |",
        "",
    ]

    if metrics["timing"]["per_action_deltas"]:
        lines.extend([
            "### Per-Action Timing Comparison",
            "| Action | Agent Avg | Real User Avg | Delta | Ratio |",
            "|---|---|---|---|---|",
        ])
        for action, delta in metrics["timing"]["per_action_deltas"].items():
            lines.append(
                f"| {action} | {delta['agent_avg_s']}s | {delta['real_user_avg_s']}s | "
                f"{delta['delta_s']}s | {delta.get('ratio', 'N/A')}x |"
            )
        lines.append("")

    lines.extend([
        "### Errors & Stuck Events",
        f"- **Total errors:** {metrics['errors']['total_errors']}",
        f"- **Stuck events:** {metrics['errors']['total_stuck_events']}",
        "",
    ])

    if metrics["errors"]["errors_by_action"]:
        lines.append("**Errors by action:**")
        for action, errs in metrics["errors"]["errors_by_action"].items():
            lines.append(f"- **{action}**: {', '.join(errs[:3])}")
        lines.append("")

    if metrics["errors"]["stuck_details"]:
        lines.append("**Stuck events:**")
        for stuck in metrics["errors"]["stuck_details"]:
            lines.append(f"- {stuck.get('action', '?')} at {stuck.get('relative_s', '?')}s: {stuck.get('reason', '?')}")
        lines.append("")

    lines.extend([
        "### Policy Deviation",
        f"- **Match rate:** {metrics['deviation'].get('match_rate', 'N/A')}",
        f"- **Deviation score:** {metrics['deviation'].get('deviation_score', 'N/A')}",
    ])

    if metrics["deviation"].get("missing_action_types"):
        lines.append(f"- **Missing actions:** {', '.join(metrics['deviation']['missing_action_types'])}")
    if metrics["deviation"].get("extra_action_types"):
        lines.append(f"- **Extra actions:** {', '.join(metrics['deviation']['extra_action_types'])}")

    lines.extend([
        "",
        "---",
        "",
        "## Qualitative UX Feedback",
        "",
        qualitative,
        "",
        "---",
        "",
        "*Report generated by PosthogAgent feedback pipeline.*",
    ])

    return "\n".join(lines)

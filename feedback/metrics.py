"""
Quantitative Metrics — Compares agent session against the real user session
to surface performance deltas, errors, and friction points.
"""

import json
from typing import Optional


def compute_metrics(
    agent_log: dict,
    real_session_parsed: dict,
    agent_policy: dict,
) -> dict:
    """
    Compute quantitative metrics comparing the agent's session
    against the real user's session and the intended policy.

    Returns a structured metrics dict.
    """
    summary = agent_log.get("summary", {})
    real_actions = real_session_parsed.get("high_level_actions", [])
    real_duration = real_session_parsed.get("session_duration_s", 0)
    policy_sequence = agent_policy.get("action_sequence", [])

    # ── Completion Metrics ──
    total_planned = len(policy_sequence)
    total_executed = summary.get("total_actions", 0)
    successful = summary.get("successful_actions", 0)
    failed = summary.get("failed_actions", 0)

    # ── Timing Metrics ──
    agent_duration = summary.get("total_duration_s", 0)
    duration_ratio = round(agent_duration / real_duration, 2) if real_duration > 0 else 0

    # Build real user timing map (action type -> avg time between actions)
    real_action_times = _extract_real_action_timings(real_actions)
    agent_action_times = summary.get("avg_action_timings", {})

    timing_deltas = {}
    for action_name, agent_avg in agent_action_times.items():
        real_avg = real_action_times.get(action_name)
        if real_avg is not None:
            timing_deltas[action_name] = {
                "agent_avg_s": agent_avg,
                "real_user_avg_s": real_avg,
                "delta_s": round(agent_avg - real_avg, 2),
                "ratio": round(agent_avg / real_avg, 2) if real_avg > 0 else None,
            }

    # ── Error Analysis ──
    errors = agent_log.get("errors", [])
    stuck_events = agent_log.get("stuck_events", [])

    error_by_action = {}
    for err in errors:
        action = err.get("action", "unknown")
        if action not in error_by_action:
            error_by_action[action] = []
        error_by_action[action].append(err.get("error", ""))

    # ── Policy Deviation ──
    actual_sequence = summary.get("action_sequence_actual", [])
    deviation = _compute_sequence_deviation(policy_sequence, actual_sequence)

    # ── Engagement Metrics ──
    real_clicks = sum(1 for a in real_actions if a["action"] == "CLICK")
    real_comments = sum(1 for a in real_actions if a["action"] == "TYPE")
    real_api_calls = sum(1 for a in real_actions if a["action"] == "API_CALL")

    agent_actions_list = agent_log.get("actions", [])
    agent_comments = sum(
        1 for a in agent_actions_list
        if a.get("action_name") == "write_comment"
        and a.get("success")
        and not a.get("details", {}).get("skipped", False)
    )
    agent_votes = sum(
        1 for a in agent_actions_list
        if a.get("action_name") == "vote_on_post"
        and a.get("success")
        and not a.get("details", {}).get("skipped", False)
    )

    return {
        "completion": {
            "planned_actions": total_planned,
            "executed_actions": total_executed,
            "successful_actions": successful,
            "failed_actions": failed,
            "completion_rate": summary.get("completion_rate", 0),
        },
        "timing": {
            "agent_duration_s": agent_duration,
            "real_user_duration_s": real_duration,
            "duration_ratio": duration_ratio,
            "per_action_deltas": timing_deltas,
        },
        "errors": {
            "total_errors": len(errors),
            "total_stuck_events": len(stuck_events),
            "errors_by_action": error_by_action,
            "stuck_details": stuck_events,
        },
        "deviation": deviation,
        "engagement": {
            "agent_comments_written": agent_comments,
            "agent_votes_cast": agent_votes,
            "real_user_input_events": real_comments,
            "real_user_click_events": real_clicks,
        },
    }


def _extract_real_action_timings(real_actions: list[dict]) -> dict:
    """
    Extract average time between similar action types from the real user's session.
    Maps action types to average duration in seconds.
    """
    action_type_map = {
        "CLICK": "open_post",  # approximate mapping
        "TYPE": "write_comment",
        "SCROLL": "scan_feed",
    }

    # Group consecutive same-type actions and compute gaps
    timings: dict[str, list[float]] = {}
    prev_time = 0.0

    for action in real_actions:
        action_type = action.get("action", "")
        mapped = action_type_map.get(action_type)
        current_time = action.get("time", 0)

        if mapped and prev_time > 0:
            gap = current_time - prev_time
            if gap > 0:
                if mapped not in timings:
                    timings[mapped] = []
                timings[mapped].append(gap)

        prev_time = current_time

    return {
        name: round(sum(gaps) / len(gaps), 2)
        for name, gaps in timings.items()
        if gaps
    }


def _compute_sequence_deviation(planned: list[str], actual: list[str]) -> dict:
    """
    Compare planned action sequence against actual execution.
    Returns deviation metrics.
    """
    if not planned:
        return {"deviation_score": 0, "details": "No planned sequence"}

    # Simple edit-distance-like comparison
    matches = 0
    min_len = min(len(planned), len(actual))

    for i in range(min_len):
        if i < len(planned) and i < len(actual) and planned[i] == actual[i]:
            matches += 1

    # Actions in planned but not executed
    planned_set = set(planned)
    actual_set = set(actual)
    missing_actions = list(planned_set - actual_set)
    extra_actions = list(actual_set - planned_set)

    match_rate = round(matches / len(planned), 2) if planned else 0

    return {
        "planned_length": len(planned),
        "actual_length": len(actual),
        "sequential_matches": matches,
        "match_rate": match_rate,
        "deviation_score": round(1 - match_rate, 2),
        "missing_action_types": missing_actions,
        "extra_action_types": extra_actions,
    }

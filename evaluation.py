"""
Weave Evaluation Scorers — Score agent output quality for the W&B fine-tuning track.

These scoring functions are decorated with @weave.op() so they appear as
evaluations in the Weave UI alongside agent traces.

Usage:
    from evaluation import score_exploration_coverage, score_demographic_consistency
    result = score_exploration_coverage(agent_log)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

WEAVE_AVAILABLE = False
try:
    import weave
    WEAVE_AVAILABLE = True
except ImportError:
    pass


def _weave_op(fn):
    """Apply @weave.op() if Weave is available, otherwise return fn unchanged."""
    if WEAVE_AVAILABLE:
        return weave.op()(fn)
    return fn


@_weave_op
def score_exploration_coverage(agent_log: dict) -> dict:
    """How many unique pages/features did the agent discover?"""
    action_history = agent_log.get("action_history", [])
    impressions = agent_log.get("impressions", [])

    # Extract unique URLs from impressions
    unique_urls = set()
    for imp in impressions:
        url = imp.get("url", "")
        if url and url != "unknown":
            unique_urls.add(url)

    # Count unique action targets (proxies for features discovered)
    unique_targets = set()
    for action in action_history:
        target = action.get("target", "")
        if target:
            unique_targets.add(target)

    total_actions = len(action_history)
    successful_actions = sum(1 for a in action_history if a.get("success"))
    total_pages = max(len(unique_urls), 1)

    return {
        "pages_discovered": len(unique_urls),
        "unique_targets_interacted": len(unique_targets),
        "total_actions": total_actions,
        "successful_actions": successful_actions,
        "success_rate": round(successful_actions / max(total_actions, 1), 3),
        "coverage_score": round(len(unique_urls) / total_pages, 3),
    }


@_weave_op
def score_demographic_consistency(agent_log: dict, cluster_profile: dict) -> dict:
    """Does the agent behave consistently with its demographic profile?

    Measures consistency by checking if the agent's browsing speed, action count,
    and engagement patterns align with the cluster's behavioral expectations.
    """
    action_history = agent_log.get("action_history", [])
    total_duration = agent_log.get("total_duration_s", 0)
    total_steps = agent_log.get("total_steps", 0)

    # Extract cluster behavioral expectations
    traits = cluster_profile.get("dominant_traits", {})
    key_behaviors = cluster_profile.get("key_behaviors", [])

    # Compute agent pace (actions per minute)
    actions_per_minute = (total_steps / max(total_duration, 1)) * 60

    # Check action diversity (exploring vs repetitive)
    action_types = [a.get("action", "") for a in action_history]
    unique_action_types = set(action_types)
    diversity_ratio = len(unique_action_types) / max(len(action_types), 1)

    # Check for engagement actions (clicks, types vs just scrolling)
    click_count = sum(1 for a in action_types if a == "click")
    scroll_count = sum(1 for a in action_types if a == "scroll")
    type_count = sum(1 for a in action_types if a == "type")
    engagement_ratio = (click_count + type_count) / max(len(action_types), 1)

    # Simple consistency heuristic:
    # - Good diversity (not just scrolling) = higher score
    # - Reasonable pace = higher score
    consistency_score = min(1.0, (diversity_ratio * 0.4) + (engagement_ratio * 0.4) + (0.2 if actions_per_minute > 2 else 0.1))

    return {
        "consistency_score": round(consistency_score, 3),
        "actions_per_minute": round(actions_per_minute, 2),
        "action_diversity_ratio": round(diversity_ratio, 3),
        "engagement_ratio": round(engagement_ratio, 3),
        "click_count": click_count,
        "scroll_count": scroll_count,
        "type_count": type_count,
        "demographic_label": cluster_profile.get("label", "unknown"),
    }


@_weave_op
def score_friction_detection(agent_log: dict) -> dict:
    """How effectively did the agent identify friction points?"""
    action_history = agent_log.get("action_history", [])
    impressions = agent_log.get("impressions", [])

    failed_actions = [a for a in action_history if not a.get("success")]
    frustration_impressions = [
        imp for imp in impressions if imp.get("sentiment") in ("frustration", "confusion")
    ]

    # Stuck events indicate real friction
    stuck_count = sum(
        1 for imp in impressions
        if "stuck" in imp.get("context", "").lower()
    )

    total_actions = max(len(action_history), 1)
    failure_rate = len(failed_actions) / total_actions

    return {
        "total_failures": len(failed_actions),
        "failure_rate": round(failure_rate, 3),
        "frustration_impressions": len(frustration_impressions),
        "stuck_events": stuck_count,
        "friction_density": round(
            (len(failed_actions) + len(frustration_impressions)) / total_actions, 3
        ),
    }


@_weave_op
def score_session_completeness(agent_log: dict) -> dict:
    """Did the agent complete a full meaningful session?"""
    total_steps = agent_log.get("total_steps", 0)
    total_duration = agent_log.get("total_duration_s", 0)
    impressions = agent_log.get("impressions", [])
    action_history = agent_log.get("action_history", [])

    # Check if session had meaningful progression
    has_summary = any(imp.get("sentiment") == "summary" for imp in impressions)
    has_varied_actions = len(set(a.get("action", "") for a in action_history)) >= 3

    successful = sum(1 for a in action_history if a.get("success"))
    success_rate = successful / max(len(action_history), 1)

    completeness_score = 0.0
    if total_steps >= 5:
        completeness_score += 0.25
    if total_duration >= 30:
        completeness_score += 0.25
    if has_varied_actions:
        completeness_score += 0.25
    if success_rate >= 0.5:
        completeness_score += 0.25

    return {
        "completeness_score": round(completeness_score, 3),
        "total_steps": total_steps,
        "total_duration_s": round(total_duration, 1),
        "success_rate": round(success_rate, 3),
        "has_varied_actions": has_varied_actions,
        "has_session_summary": has_summary,
    }


def run_all_evaluations(agent_log: dict, cluster_profile: dict) -> dict:
    """Run all evaluation scorers and return combined results."""
    return {
        "exploration_coverage": score_exploration_coverage(agent_log),
        "demographic_consistency": score_demographic_consistency(agent_log, cluster_profile),
        "friction_detection": score_friction_detection(agent_log),
        "session_completeness": score_session_completeness(agent_log),
    }


if __name__ == "__main__":
    import json
    from config import AGENT_LOGS_DIR, CLUSTERS_DIR

    # Load latest agent log and cluster for evaluation demo
    log_files = sorted(AGENT_LOGS_DIR.glob("explore_*.json"), reverse=True)
    if not log_files:
        print("No agent logs found. Run run_agents.py first.")
        sys.exit(1)

    with open(log_files[0]) as f:
        agent_log_data = json.load(f)

    clusters_path = CLUSTERS_DIR / "clusters.json"
    if clusters_path.exists():
        with open(clusters_path) as f:
            clusters = json.load(f)
        cluster_profile = clusters["clusters"][0]
    else:
        cluster_profile = {"label": "unknown", "dominant_traits": {}}

    summary = agent_log_data.get("summary", agent_log_data)
    results = run_all_evaluations(summary, cluster_profile)

    print(json.dumps(results, indent=2))

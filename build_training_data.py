"""
Build fine-tuning training data from clustered session recordings.
Converts each session's action sequence into supervised (state → next_action) pairs
in Mistral fine-tuning JSONL format.

Usage:
    python build_training_data.py
    python build_training_data.py --window-size 5
"""

import json
import sys
import argparse
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    PARSED_DIR,
    DESCRIPTIONS_DIR,
    CLUSTERS_DIR,
    TRAINING_DIR,
    ensure_data_dirs,
)


def load_clusters() -> dict:
    """Load cluster assignments."""
    cluster_path = CLUSTERS_DIR / "clusters.json"
    if not cluster_path.exists():
        print("ERROR: clusters.json not found. Run cluster_demographics.py first.")
        sys.exit(1)
    with open(cluster_path) as f:
        return json.load(f)


def build_system_prompt(cluster: dict) -> str:
    """Build the system prompt describing the demographic persona."""
    traits = cluster.get("dominant_traits", {})
    label = cluster.get("label", "Unknown Demographic")
    description = cluster.get("description", "")
    behaviors = cluster.get("key_behaviors", [])

    prompt = f"""You are a digital twin representing the "{label}" user demographic.

Demographic Profile:
- Primary age group: {traits.get('age_group', 'unknown')}
- Primary country: {traits.get('country', 'unknown')}
- NYC familiarity: {traits.get('nyc_familiarity', 'unknown')}
- Average session duration: {cluster.get('avg_duration_s', 0):.0f} seconds
- Average actions per session: {cluster.get('avg_actions', 0):.0f}

Behavioral Description: {description}

Key Behaviors:
{chr(10).join(f'- {b}' for b in behaviors)}

You are exploring a web application. Based on the current page state and your recent actions, decide what to do next. Your response must be valid JSON with:
- "action": the action to take (one of: scroll, click, type, wait, navigate_back)
- "target": what to interact with (element description, scroll direction, or text to type)
- "reasoning": brief explanation of why this demographic would do this
- "hesitation_ms": how long to wait before acting (reflects your demographic's pace)"""

    return prompt


def build_state_description(actions: list[dict], current_idx: int, window_size: int,
                            session_duration: float, profile: dict) -> str:
    """Build the user message describing current page state from action history."""
    # Get recent action window
    start = max(0, current_idx - window_size)
    recent = actions[start:current_idx]
    current = actions[current_idx] if current_idx < len(actions) else None

    # Build context
    lines = []
    lines.append(f"Session progress: {current_idx}/{len(actions)} actions, "
                 f"elapsed: {actions[current_idx-1]['time'] if current_idx > 0 else 0:.1f}s / {session_duration:.0f}s total")

    if recent:
        lines.append("\nRecent actions:")
        for a in recent:
            line = f"  [{a['time']:.1f}s] {a['action']}"
            if a['action'] == 'CLICK':
                target = a.get('target_text', '') or a.get('target_tag', '') or a.get('target_href', '')
                line += f" on {a.get('target_tag', '?')}"
                if target:
                    line += f' "{target[:50]}"'
            elif a['action'] == 'SCROLL':
                line += f" {a.get('direction', '?')} to {a.get('scroll_to', 0)}px ({a.get('duration_s', 0):.1f}s)"
            elif a['action'] == 'TYPE':
                line += f" {a.get('text_length', 0)} chars into {a.get('target_tag', '?')}"
                if a.get('target_placeholder'):
                    line += f' (placeholder: "{a["target_placeholder"]}")'
            elif a['action'] == 'API_CALL':
                line += f" {a.get('path', '?')}"
            elif a['action'] == 'PAGE_LOAD':
                line += f" {a.get('url', '?')}"
            lines.append(line)

    lines.append("\nWhat would you do next?")
    return "\n".join(lines)


def build_action_response(action: dict) -> str:
    """Build the assistant response (the next action the real user took)."""
    action_type = action["action"]
    time = action.get("time", 0)

    if action_type == "CLICK":
        target = action.get("target_text", "") or action.get("target_href", "") or action.get("target_class", "")
        response = {
            "action": "click",
            "target": f"{action.get('target_tag', 'element')} - {target[:60]}".strip(" - "),
            "reasoning": f"Clicking on {action.get('target_tag', 'element')} to explore content",
            "hesitation_ms": 0,
        }
    elif action_type == "SCROLL":
        response = {
            "action": "scroll",
            "target": f"{action.get('direction', 'down')} to {action.get('scroll_to', 0)}px",
            "reasoning": f"Scrolling {action.get('direction', 'down')} to {'discover more content' if action.get('direction') == 'down' else 'review content'}",
            "hesitation_ms": 0,
        }
    elif action_type == "TYPE":
        response = {
            "action": "type",
            "target": f"{action.get('target_tag', 'input')} ({action.get('target_placeholder', 'text field')})",
            "reasoning": "Entering text to engage with the application",
            "hesitation_ms": 0,
        }
    elif action_type == "API_CALL":
        response = {
            "action": "wait",
            "target": f"API response from {action.get('path', '/')}",
            "reasoning": "Waiting for server response",
            "hesitation_ms": 500,
        }
    elif action_type == "PAGE_LOAD":
        response = {
            "action": "navigate_back" if "back" in str(action.get("url", "")).lower() else "click",
            "target": action.get("url", "page"),
            "reasoning": "Navigating to a new page",
            "hesitation_ms": 0,
        }
    else:
        response = {
            "action": action_type.lower(),
            "target": "page",
            "reasoning": "Continuing to browse",
            "hesitation_ms": 0,
        }

    # Calculate hesitation from time gaps
    return json.dumps(response)


def calculate_hesitation(actions: list[dict], idx: int) -> int:
    """Calculate hesitation time based on gap between actions."""
    if idx == 0:
        return int(actions[0].get("time", 0) * 1000)
    gap = actions[idx]["time"] - actions[idx - 1]["time"]
    return max(0, int(gap * 1000))


def build_training_examples(
    parsed_session: dict,
    cluster: dict,
    window_size: int = 5,
) -> list[dict]:
    """Convert one session into training examples."""
    actions = parsed_session.get("high_level_actions", [])
    duration = parsed_session.get("session_duration_s", 0)
    profile = parsed_session.get("user_profile", {})

    # Skip very short sessions
    if len(actions) < 3:
        return []

    system_prompt = build_system_prompt(cluster)
    examples = []

    # Sliding window: for each action, predict it from the previous N actions
    for i in range(1, len(actions)):
        # Skip API_CALL and PAGE_LOAD as prediction targets (they're system events)
        if actions[i]["action"] in ("API_CALL", "PAGE_LOAD", "FULL_SNAPSHOT", "PAGE_META"):
            continue

        state = build_state_description(actions, i, window_size, duration, profile)

        # Build response with calculated hesitation
        response_data = json.loads(build_action_response(actions[i]))
        response_data["hesitation_ms"] = calculate_hesitation(actions, i)
        response = json.dumps(response_data)

        examples.append({
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": state},
                {"role": "assistant", "content": response},
            ]
        })

    return examples


def main():
    parser = argparse.ArgumentParser(description="Build fine-tuning training data from clusters")
    parser.add_argument("--window-size", type=int, default=5, help="Number of recent actions as context")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split ratio")
    args = parser.parse_args()

    ensure_data_dirs()

    # Load clusters
    clusters_data = load_clusters()
    print(f"Loaded {clusters_data['num_clusters']} clusters with {clusters_data['total_sessions']} sessions\n")

    total_train = 0
    total_val = 0

    for cluster in clusters_data["clusters"]:
        cluster_id = cluster["id"]
        label = cluster.get("label", f"Cluster {cluster_id}")
        session_ids = cluster["session_ids"]

        print(f"{'='*50}")
        print(f"Cluster {cluster_id}: {label} ({len(session_ids)} sessions)")
        print(f"{'='*50}")

        all_examples = []

        for sid in session_ids:
            parsed_path = PARSED_DIR / f"parsed_{sid}.json"
            if not parsed_path.exists():
                print(f"  WARNING: parsed data not found for {sid}")
                continue

            with open(parsed_path) as f:
                parsed = json.load(f)

            examples = build_training_examples(parsed, cluster, window_size=args.window_size)
            print(f"  {sid}: {len(examples)} examples")
            all_examples.extend(examples)

        if not all_examples:
            print(f"  WARNING: No training examples for cluster {cluster_id}")
            continue

        # Shuffle and split
        random.seed(42)
        random.shuffle(all_examples)

        val_count = max(1, int(len(all_examples) * args.val_split))
        val_examples = all_examples[:val_count]
        train_examples = all_examples[val_count:]

        # Save JSONL files
        train_path = TRAINING_DIR / f"cluster_{cluster_id}_train.jsonl"
        val_path = TRAINING_DIR / f"cluster_{cluster_id}_val.jsonl"

        with open(train_path, "w") as f:
            for ex in train_examples:
                f.write(json.dumps(ex) + "\n")

        with open(val_path, "w") as f:
            for ex in val_examples:
                f.write(json.dumps(ex) + "\n")

        print(f"\n  Train: {len(train_examples)} examples -> {train_path}")
        print(f"  Val:   {len(val_examples)} examples -> {val_path}")

        total_train += len(train_examples)
        total_val += len(val_examples)

    # Summary
    print(f"\n{'='*60}")
    print("TRAINING DATA GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Total training examples: {total_train}")
    print(f"  Total validation examples: {total_val}")
    print(f"  Output directory: {TRAINING_DIR}")

    # Show file sizes
    for f in sorted(TRAINING_DIR.glob("*.jsonl")):
        size_kb = f.stat().st_size / 1024
        lines = sum(1 for _ in open(f))
        print(f"  {f.name}: {lines} examples ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()

"""
Build fine-tuning training data from clustered session recordings.
Converts each session's action sequence into supervised (state -> next_action)
pairs that teach a model to mimic user *behavior patterns* on a product,
given the product's feature description and the cluster's behavioral persona.

Generates per-cluster training files: cluster_<id>_train.jsonl / val.jsonl
so that each cluster can have its own fine-tuned LoRA adapter.

Requires: clusters.json from process_synthetic_batch.py or cluster_demographics.py

Usage:
    python build_training_data.py
    python build_training_data.py --window-size 5 --app-description "An e-commerce store..."
"""

import json
import sys
import argparse
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    PARSED_DIR,
    TRAINING_DIR,
    CLUSTERS_DIR,
    DESCRIPTIONS_DIR,
    TARGET_APP_DESCRIPTION,
    TARGET_APP_URL,
    TARGET_APP_NAME,
    ensure_data_dirs,
)


# ── System prompt (shared across all clusters) ──────────────────────────────

BASE_SYSTEM_PROMPT = """You are a browser automation agent that mimics realistic human behavior on web applications. You are given a product's feature description, a behavioral persona, and the current page state. Your job is to decide what a real user matching this persona would do next.

Your response must be valid JSON with these fields:
- "action": one of "click", "scroll", "type", "wait", "navigate_back"
- "target": what to interact with — a semantic description of the element (e.g. "login button", "search input", "post title"), a scroll direction ("down 300px"), or text to type
- "reasoning": brief explanation of why this persona would do this given the product features
- "hesitation_ms": milliseconds to wait before acting (reflects this persona's natural pace)

Guidelines for mimicking realistic behavior:
- Users explore features mentioned in the product description
- Users pause to read content before acting
- Users sometimes scroll to discover content before clicking
- Typing has natural hesitation; users don't instantly fill forms
- Users navigate back when content isn't interesting
- Early in a session users explore broadly; later they engage deeper"""


def build_cluster_system_prompt(cluster: dict) -> str:
    """Build a system prompt that includes the cluster's behavioral persona."""
    label = cluster.get("label", f"Cluster {cluster['id']}")
    description = cluster.get("description", "")
    behaviors = cluster.get("key_behaviors", [])

    persona_block = f"\n\nYour persona: \"{label}\""
    if description:
        persona_block += f"\n{description}"
    if behaviors:
        persona_block += "\nKey behaviors:\n" + "\n".join(f"- {b}" for b in behaviors)
    persona_block += f"\nAvg session: {cluster.get('avg_duration_s', 0):.0f}s, {cluster.get('avg_actions', 0):.0f} actions"

    return BASE_SYSTEM_PROMPT + persona_block


def build_product_context(app_name: str, app_url: str, app_description: str) -> str:
    """Build the product feature context that prefixes every user message."""
    lines = [f"Product: {app_name}"]
    if app_url:
        lines.append(f"URL: {app_url}")
    if app_description:
        lines.append(f"Features: {app_description}")
    return "\n".join(lines)


def build_state_description(
    actions: list[dict],
    current_idx: int,
    window_size: int,
    session_duration: float,
    product_context: str,
    error_context: str = "",
) -> str:
    """Build the user message: product context + recent actions + state."""
    start = max(0, current_idx - window_size)
    recent = actions[start:current_idx]

    lines = [product_context, ""]

    elapsed = actions[current_idx - 1]["time"] if current_idx > 0 else 0
    lines.append(f"Session: step {current_idx}, elapsed {elapsed:.1f}s / {session_duration:.0f}s")

    if recent:
        lines.append("\nRecent actions:")
        for a in recent:
            line = f"  [{a['time']:.1f}s] {a['action']}"
            if a["action"] == "CLICK":
                tag = a.get("target_tag", "?")
                text = a.get("target_text", "") or a.get("target_href", "")
                line += f" on {tag}"
                if text:
                    line += f' "{text[:50]}"'
            elif a["action"] == "SCROLL":
                line += f" {a.get('direction', '?')} to {a.get('scroll_to', 0)}px"
            elif a["action"] == "TYPE":
                chars = a.get("text_length", 0)
                tag = a.get("target_tag", "input")
                placeholder = a.get("target_placeholder", "")
                line += f" {chars} chars into {tag}"
                if placeholder:
                    line += f' ("{placeholder}")'
            elif a["action"] == "API_CALL":
                line += f" {a.get('path', '?')}"
            elif a["action"] == "PAGE_LOAD":
                line += f" {a.get('url', '?')}"
            lines.append(line)

    if error_context:
        lines.append(f"\nLast action failed: {error_context}")
        lines.append("Decide how to recover.")
    else:
        lines.append("\nWhat would you do next?")

    return "\n".join(lines)


def build_action_response(action: dict) -> dict:
    """Build the assistant response from a real user action."""
    action_type = action["action"]

    if action_type == "CLICK":
        text = action.get("target_text", "") or action.get("target_href", "") or action.get("target_class", "")
        tag = action.get("target_tag", "element")
        target_desc = f"{tag}"
        if text:
            target_desc += f' "{text[:60]}"'
        return {
            "action": "click",
            "target": target_desc,
            "reasoning": f"Exploring {tag} element to interact with the product",
        }
    elif action_type == "SCROLL":
        direction = action.get("direction", "down")
        pixels = action.get("scroll_to", 0)
        return {
            "action": "scroll",
            "target": f"{direction} to {pixels}px",
            "reasoning": f"Scrolling {direction} to {'discover more content' if direction == 'down' else 'review earlier content'}",
        }
    elif action_type == "TYPE":
        tag = action.get("target_tag", "input")
        placeholder = action.get("target_placeholder", "text field")
        return {
            "action": "type",
            "target": f"{tag} ({placeholder})",
            "reasoning": "Entering text to engage with the application",
        }
    elif action_type == "API_CALL":
        return {
            "action": "wait",
            "target": f"API response from {action.get('path', '/')}",
            "reasoning": "Waiting for the application to respond",
        }
    elif action_type == "PAGE_LOAD":
        return {
            "action": "navigate_back" if "back" in str(action.get("url", "")).lower() else "click",
            "target": action.get("url", "page"),
            "reasoning": "Navigating to a new section of the product",
        }
    else:
        return {
            "action": action_type.lower(),
            "target": "page",
            "reasoning": "Continuing to browse",
        }


def calculate_hesitation(actions: list[dict], idx: int) -> int:
    """Calculate hesitation time (ms) from gap between consecutive actions."""
    if idx == 0:
        return int(actions[0].get("time", 0) * 1000)
    gap = actions[idx]["time"] - actions[idx - 1]["time"]
    return max(0, int(gap * 1000))


def build_training_examples(
    parsed_session: dict,
    cluster: dict,
    product_context: str,
    window_size: int = 5,
) -> list[dict]:
    """Convert one session into step-by-step training examples for a cluster."""
    actions = parsed_session.get("high_level_actions", [])
    duration = parsed_session.get("session_duration_s", 0)

    if len(actions) < 3:
        return []

    system_prompt = build_cluster_system_prompt(cluster)
    examples = []
    skip_types = ("API_CALL", "PAGE_LOAD", "FULL_SNAPSHOT", "PAGE_META")

    for i in range(1, len(actions)):
        if actions[i]["action"] in skip_types:
            continue

        state = build_state_description(
            actions, i, window_size, duration, product_context
        )

        response = build_action_response(actions[i])
        response["hesitation_ms"] = calculate_hesitation(actions, i)

        examples.append({
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": state},
                {"role": "assistant", "content": json.dumps(response)},
            ]
        })

    return examples


def load_clusters() -> dict:
    """Load cluster assignments from clusters.json."""
    cluster_path = CLUSTERS_DIR / "clusters.json"
    if not cluster_path.exists():
        print(f"ERROR: {cluster_path} not found.")
        print("Run: python process_synthetic_batch.py --recordings-dir sessions --clusters 5")
        sys.exit(1)
    with open(cluster_path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Build per-cluster fine-tuning training data")
    parser.add_argument("--window-size", type=int, default=5, help="Recent action window size")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--app-name", type=str, default=None, help="App name (overrides config)")
    parser.add_argument("--app-url", type=str, default=None, help="App URL (overrides config)")
    parser.add_argument("--app-description", type=str, default=None, help="App feature description (overrides config)")
    args = parser.parse_args()

    ensure_data_dirs()

    app_name = args.app_name or TARGET_APP_NAME
    app_url = args.app_url or TARGET_APP_URL
    app_description = args.app_description or TARGET_APP_DESCRIPTION

    if not app_description:
        print("WARNING: No app description set. Set TARGET_APP_DESCRIPTION in .env or use --app-description")
        print("         Training data will lack product context.\n")

    product_context = build_product_context(app_name, app_url, app_description)

    # Load clusters
    clusters_data = load_clusters()
    n_clusters = clusters_data["num_clusters"]
    print(f"Product: {app_name}")
    print(f"URL: {app_url}")
    desc_preview = app_description[:100] + "..." if len(app_description) > 100 else app_description
    print(f"Description: {desc_preview}")
    print(f"Clusters: {n_clusters} ({clusters_data['total_sessions']} total sessions)")
    print(f"Window size: {args.window_size}\n")

    grand_total_train = 0
    grand_total_val = 0

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

            examples = build_training_examples(
                parsed, cluster, product_context, window_size=args.window_size
            )
            n_actions = len(parsed.get("high_level_actions", []))
            print(f"  {sid}: {n_actions} actions -> {len(examples)} pairs")
            all_examples.extend(examples)

        if not all_examples:
            print(f"  WARNING: No training examples for cluster {cluster_id}")
            continue

        # Shuffle and split
        random.seed(42 + cluster_id)
        random.shuffle(all_examples)

        val_count = max(1, int(len(all_examples) * args.val_split))
        val_examples = all_examples[:val_count]
        train_examples = all_examples[val_count:]

        # Save per-cluster JSONL files
        train_path = TRAINING_DIR / f"cluster_{cluster_id}_train.jsonl"
        val_path = TRAINING_DIR / f"cluster_{cluster_id}_val.jsonl"

        with open(train_path, "w") as f:
            for ex in train_examples:
                f.write(json.dumps(ex) + "\n")

        with open(val_path, "w") as f:
            for ex in val_examples:
                f.write(json.dumps(ex) + "\n")

        print(f"  Train: {len(train_examples)} -> {train_path.name}")
        print(f"  Val:   {len(val_examples)} -> {val_path.name}")

        grand_total_train += len(train_examples)
        grand_total_val += len(val_examples)

    print(f"\n{'='*60}")
    print("TRAINING DATA GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Total training examples: {grand_total_train}")
    print(f"  Total validation examples: {grand_total_val}")
    print(f"  Output directory: {TRAINING_DIR}")

    for f in sorted(TRAINING_DIR.glob("cluster_*_*.jsonl")):
        size_kb = f.stat().st_size / 1024
        lines = sum(1 for _ in open(f))
        print(f"  {f.name}: {lines} examples ({size_kb:.1f} KB)")

    print(f"\nReady for: python finetune_job.py --all-clusters")


if __name__ == "__main__":
    main()

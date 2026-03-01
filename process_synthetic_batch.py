"""
Batch-process synthetic session recordings through the full pre-finetuning pipeline:
  1. Parse synthetic flat-event JSON -> parsed_session format
  2. Generate behavioral descriptions via Mistral Large
  3. Generate embeddings via Mistral Embed
  4. Cluster via K-Means (K is configurable)
  5. Generate cluster labels via Mistral
  6. Save KMeans model + clusters.json
  7. Build fine-tuning training data (JSONL)

After this script completes, you can run:
    python fine_tune.py

Usage:
    python process_synthetic_batch.py
    python process_synthetic_batch.py --clusters 3 --min-actions 3
    python process_synthetic_batch.py --recordings-dir data/recordings --clusters 5
"""

import json
import sys
import pickle
import asyncio
import argparse
import time
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.cluster import KMeans
from mistralai import Mistral

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    MISTRAL_API_KEY,
    PARSED_DIR,
    DESCRIPTIONS_DIR,
    EMBEDDINGS_DIR,
    CLUSTERS_DIR,
    TRAINING_DIR,
    ensure_data_dirs,
)
from pipeline.stage2_describe import describe_and_save, describe_and_save_async
from pipeline.stage3_encode import encode_and_save, encode_and_save_async


# ── Step 1: Synthetic adapter ──────────────────────────────────────────────

def parse_synthetic_session(session_path: Path) -> dict:
    """
    Convert a synthetic flat-event JSON file into the same parsed_session
    format that Stage 1 produces from PostHog rrweb data.

    Expected input format:
    {
      "id": "...",
      "events": [
        {"event_type": "click"|"scroll"|"page_load"|..., "timestamp": ..., ...}
      ],
      "duration_ms": ...,
      ...
    }
    """
    with open(session_path) as f:
        session = json.load(f)

    session_id = session["id"]
    events = session.get("events", [])
    duration_ms = session.get("duration_ms", 0)

    if not events:
        return _empty_parsed(session_id, duration_ms)

    ts0 = events[0]["timestamp"]
    high_level_actions = []
    last_scroll_y = 0

    for evt in events:
        etype = evt["event_type"]
        ts = (evt["timestamp"] - ts0) / 1000.0  # relative seconds

        if etype == "page_load":
            vw = evt.get("viewport_width") or 0
            vh = evt.get("viewport_height") or 0
            high_level_actions.append({
                "time": round(ts, 2),
                "action": "PAGE_LOAD",
                "url": evt.get("url", ""),
                "viewport": f"{vw}x{vh}",
            })

        elif etype in ("click", "rage_click", "dead_click"):
            action = {
                "time": round(ts, 2),
                "action": "CLICK",
                "target_tag": evt.get("target_tag", "") or "",
                "target_text": evt.get("target_text", "") or "",
                "target_href": "",
                "target_class": (evt.get("target_selector", "") or "").lstrip("."),
                "coordinates": [
                    int(evt["x"]) if evt.get("x") else 0,
                    int(evt["y"]) if evt.get("y") else 0,
                ],
            }
            if etype == "rage_click":
                action["target_text"] = (action["target_text"] + " [RAGE CLICK]").strip()
            elif etype == "dead_click":
                action["target_text"] = (action["target_text"] + " [DEAD CLICK]").strip()
            high_level_actions.append(action)

        elif etype == "scroll":
            scroll_y = evt.get("scroll_y", 0) or 0
            direction = "down" if scroll_y >= last_scroll_y else "up"
            high_level_actions.append({
                "time": round(ts, 2),
                "action": "SCROLL",
                "scroll_from": int(last_scroll_y),
                "scroll_to": int(scroll_y),
                "max_depth": int(max(scroll_y, last_scroll_y)),
                "duration_s": 0.5,
                "direction": direction,
            })
            last_scroll_y = scroll_y

        elif etype == "hover":
            hover_ms = evt.get("hover_duration_ms", 0) or 0
            high_level_actions.append({
                "time": round(ts, 2),
                "action": "CLICK",
                "target_tag": evt.get("target_tag", "") or "",
                "target_text": f"{evt.get('target_text', '')} [HOVER {hover_ms:.0f}ms]",
                "target_href": "",
                "target_class": (evt.get("target_selector", "") or "").lstrip("."),
                "coordinates": [
                    int(evt["x"]) if evt.get("x") else 0,
                    int(evt["y"]) if evt.get("y") else 0,
                ],
            })

        elif etype == "hesitation":
            hover_ms = evt.get("hover_duration_ms", 0) or 0
            high_level_actions.append({
                "time": round(ts, 2),
                "action": "CLICK",
                "target_tag": "page",
                "target_text": f"[HESITATION {hover_ms:.0f}ms]",
                "target_href": "",
                "target_class": "",
                "coordinates": [0, 0],
            })

        elif etype == "input" or etype == "type":
            high_level_actions.append({
                "time": round(ts, 2),
                "action": "TYPE",
                "target_tag": evt.get("target_tag", "input"),
                "text_length": evt.get("text_length", len(evt.get("text", ""))),
                "is_masked": evt.get("is_masked", False),
                "target_placeholder": evt.get("placeholder", ""),
            })

        # mouse_move is skipped (too granular)

    user_agent = session.get("user_agent", "")
    parsed = {
        "user_profile": {
            "session_id": session_id,
            "age_group": session.get("age_group", "unknown"),
            "country": session.get("country", "unknown"),
            "nyc_familiarity": "unknown",
            "os": "unknown",
            "browser": user_agent if user_agent not in ("imported", "synthetic", "") else "unknown",
            "device_type": session.get("device_type", "desktop"),
            "viewport_width": session.get("viewport_width", 1920),
            "viewport_height": session.get("viewport_height", 1080),
            "screen_width": session.get("viewport_width", 1920),
            "screen_height": session.get("viewport_height", 1080),
        },
        "session_duration_s": round(duration_ms / 1000.0, 1),
        "total_raw_events": len(events),
        "high_level_actions": high_level_actions,
    }

    return parsed


def _empty_parsed(session_id: str, duration_ms: float) -> dict:
    return {
        "user_profile": {"session_id": session_id},
        "session_duration_s": round(duration_ms / 1000.0, 1),
        "total_raw_events": 0,
        "high_level_actions": [],
    }


# ── Step 5: Cluster labels (same logic as cluster_demographics.py) ─────────

def generate_cluster_labels(cluster_data: dict, client: Mistral) -> dict:
    """Use Mistral to generate human-readable labels for each cluster."""
    for cluster in cluster_data["clusters"]:
        excerpts = []
        for sid in cluster["session_ids"][:3]:
            desc_path = DESCRIPTIONS_DIR / f"description_{sid}.txt"
            if desc_path.exists():
                excerpts.append(desc_path.read_text()[:500])

        traits = cluster["dominant_traits"]
        prompt = f"""Based on these user behavioral profiles from a web app, generate a short demographic persona label and description.

CLUSTER STATS:
- {cluster['size']} users in this group
- Average session duration: {cluster['avg_duration_s']}s
- Average actions per session: {cluster['avg_actions']}

SAMPLE BEHAVIORAL DESCRIPTIONS:
{chr(10).join(f'--- User {i+1} ---{chr(10)}{e}' for i, e in enumerate(excerpts))}

Output ONLY valid JSON:
{{
  "label": "<short 2-4 word label like 'Impulsive Explorer' or 'Frustrated Abandoner'>",
  "description": "<1-2 sentence behavioral summary>",
  "key_behaviors": ["<behavior 1>", "<behavior 2>", "<behavior 3>"]
}}"""

        print(f"    Labeling cluster {cluster['id']}...")
        try:
            response = client.chat.complete(
                model="mistral-medium-latest",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            label_data = json.loads(response.choices[0].message.content)
            cluster["label"] = label_data.get("label", f"Cluster {cluster['id']}")
            cluster["description"] = label_data.get("description", "")
            cluster["key_behaviors"] = label_data.get("key_behaviors", [])
            print(f"      -> {cluster['label']}: {cluster['description'][:80]}")
        except Exception as e:
            print(f"      Label generation failed: {e}")
            cluster["label"] = f"Demographic {cluster['id']}"
            cluster["description"] = ""
            cluster["key_behaviors"] = []

    return cluster_data


# ── Step 7: Build training data (inlined from build_training_data.py) ──────

def build_training_data(cluster_data: dict, window_size: int = 5, val_split: float = 0.1):
    """Build fine-tuning JSONL files for each cluster. Same logic as build_training_data.py."""
    import random
    from build_training_data import build_training_examples

    total_train = 0
    total_val = 0

    for cluster in cluster_data["clusters"]:
        cluster_id = cluster["id"]
        label = cluster.get("label", f"Cluster {cluster_id}")
        session_ids = cluster["session_ids"]

        print(f"\n  Cluster {cluster_id} ({label}): {len(session_ids)} sessions")
        all_examples = []

        for sid in session_ids:
            parsed_path = PARSED_DIR / f"parsed_{sid}.json"
            if not parsed_path.exists():
                continue
            with open(parsed_path) as f:
                parsed = json.load(f)
            examples = build_training_examples(parsed, cluster, window_size=window_size)
            print(f"    {sid}: {len(examples)} training examples")
            all_examples.extend(examples)

        if not all_examples:
            print(f"    WARNING: No training examples for cluster {cluster_id}")
            continue

        random.seed(42)
        random.shuffle(all_examples)

        val_count = max(1, int(len(all_examples) * val_split))
        val_examples = all_examples[:val_count]
        train_examples = all_examples[val_count:]

        train_path = TRAINING_DIR / f"cluster_{cluster_id}_train.jsonl"
        val_path = TRAINING_DIR / f"cluster_{cluster_id}_val.jsonl"

        with open(train_path, "w") as f:
            for ex in train_examples:
                f.write(json.dumps(ex) + "\n")
        with open(val_path, "w") as f:
            for ex in val_examples:
                f.write(json.dumps(ex) + "\n")

        print(f"    Train: {len(train_examples)} -> {train_path.name}")
        print(f"    Val:   {len(val_examples)} -> {val_path.name}")
        total_train += len(train_examples)
        total_val += len(val_examples)

    return total_train, total_val


# ── Async PHASE 1 orchestrator ─────────────────────────────────────────────

async def _process_one_session(
    sid: str,
    parsed: dict,
    profile: dict,
    num_actions: int,
    semaphore: asyncio.Semaphore,
    index: int,
    total: int,
) -> dict | None:
    """Describe + embed a single session, respecting the concurrency semaphore."""
    desc_path = DESCRIPTIONS_DIR / f"description_{sid}.txt"
    embed_path = EMBEDDINGS_DIR / f"embedding_{sid}.json"

    # ── Describe ──
    if desc_path.exists():
        description = desc_path.read_text()
    else:
        async with semaphore:
            print(f"  [{index}/{total}] {sid}: describing...")
            try:
                description = await describe_and_save_async(parsed, MISTRAL_API_KEY, str(desc_path))
            except Exception as e:
                print(f"  [{index}/{total}] {sid}: describe FAILED: {e}")
                return None

    # ── Embed ──
    if embed_path.exists():
        with open(embed_path) as f:
            embed_result = json.load(f)
    else:
        async with semaphore:
            print(f"  [{index}/{total}] {sid}: embedding...")
            try:
                embed_result = await encode_and_save_async(description, profile, MISTRAL_API_KEY, str(embed_path))
            except Exception as e:
                print(f"  [{index}/{total}] {sid}: embed FAILED: {e}")
                return None

    print(f"  [{index}/{total}] {sid}: done ({len(description)} chars, {embed_result.get('embedding_dim','?')}d)")

    return {
        "session_id": sid,
        "profile": profile,
        "num_actions": num_actions,
        "duration_s": parsed.get("session_duration_s", 0),
        "description_path": str(desc_path),
        "embedding": embed_result["embedding"],
    }


async def _phase1_async(
    session_files: list[Path],
    min_actions: int,
    concurrency: int,
) -> list[dict]:
    """Parse all sessions (sync), then describe+embed concurrently."""

    # Step 1: Parse all sessions synchronously (fast, no API calls)
    parsed_sessions = []
    for i, sf in enumerate(session_files):
        with open(sf) as f:
            raw = json.load(f)
        sid = raw["id"]

        parsed_path = PARSED_DIR / f"parsed_{sid}.json"
        if parsed_path.exists():
            with open(parsed_path) as f:
                parsed = json.load(f)
        else:
            try:
                parsed = parse_synthetic_session(sf)
                with open(parsed_path, "w") as f:
                    json.dump(parsed, f, indent=2)
            except Exception as e:
                print(f"  {sid}: parse FAILED: {e}")
                continue

        num_actions = len(parsed.get("high_level_actions", []))
        if num_actions < min_actions:
            print(f"  {sid}: skipped ({num_actions} actions < {min_actions})")
            continue

        parsed_sessions.append((sid, parsed, parsed.get("user_profile", {}), num_actions))

    print(f"\n  Parsed {len(parsed_sessions)}/{len(session_files)} sessions (passed min-actions filter)")

    # Step 2: Describe + Embed concurrently
    sem = asyncio.Semaphore(concurrency)
    total = len(parsed_sessions)
    t0 = time.time()

    tasks = [
        _process_one_session(sid, parsed, profile, n_act, sem, i + 1, total)
        for i, (sid, parsed, profile, n_act) in enumerate(parsed_sessions)
    ]

    results = await asyncio.gather(*tasks)
    elapsed = time.time() - t0

    sessions = [r for r in results if r is not None]
    print(f"\n  Completed {len(sessions)}/{total} sessions in {elapsed:.1f}s "
          f"({elapsed / max(len(sessions), 1):.1f}s avg per session)")

    return sessions


# ── Main orchestrator ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Process synthetic recordings: parse → describe → embed → cluster → training data"
    )
    parser.add_argument(
        "--recordings-dir", type=str, default="data/recordings",
        help="Directory containing synthetic session_*.json files"
    )
    parser.add_argument("--clusters", "-k", type=int, default=3, help="Number of K-Means clusters")
    parser.add_argument("--min-actions", type=int, default=3, help="Min high-level actions to include a session")
    parser.add_argument("--window-size", type=int, default=5, help="Sliding window size for training data")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument(
        "--concurrency", "-j", type=int, default=5,
        help="Max concurrent Mistral API calls (default: 5)"
    )
    parser.add_argument(
        "--skip-training-data", action="store_true",
        help="Skip Step 7 (training data generation). Useful if you just want clustering."
    )
    args = parser.parse_args()

    if not MISTRAL_API_KEY:
        print("ERROR: MISTRAL_API_KEY not set. Add it to .env")
        sys.exit(1)

    ensure_data_dirs()
    rec_dir = Path(args.recordings_dir)
    client = Mistral(api_key=MISTRAL_API_KEY)

    # Find all synthetic session files
    session_files = sorted(rec_dir.glob("session_*.json"))
    if not session_files:
        print(f"No session_*.json files found in {rec_dir}")
        print("Place your synthetic session files there and re-run.")
        sys.exit(1)

    print("=" * 60)
    print(f"SYNTHETIC BATCH PIPELINE")
    print(f"  Recordings dir: {rec_dir}")
    print(f"  Sessions found: {len(session_files)}")
    print(f"  K (clusters):   {args.clusters}")
    print(f"  Min actions:    {args.min_actions}")
    print(f"  Concurrency:    {args.concurrency}")
    print("=" * 60)

    # ── PHASE 1: Parse (sync) + Describe + Embed (async concurrent) ──
    print(f"\n{'='*60}")
    print(f"PHASE 1: Parse → Describe → Embed (concurrency={args.concurrency})")
    print("=" * 60)

    sessions = asyncio.run(_phase1_async(
        session_files, args.min_actions, args.concurrency,
    ))

    if not sessions:
        print("\nERROR: No valid sessions after processing. Check your data.")
        sys.exit(1)

    # ── PHASE 2: Cluster ──
    n_clusters = min(args.clusters, len(sessions))
    print(f"\n{'='*60}")
    print(f"PHASE 2: K-Means Clustering (K={n_clusters}, {len(sessions)} sessions)")
    print("=" * 60)

    embeddings = np.array([s["embedding"] for s in sessions])
    print(f"  Embedding matrix: {embeddings.shape}")

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    # Save KMeans model
    kmeans_path = CLUSTERS_DIR / "kmeans_model.pkl"
    with open(kmeans_path, "wb") as f:
        pickle.dump(kmeans, f)
    print(f"  Saved KMeans model -> {kmeans_path}")

    # Build cluster metadata
    clusters = []
    for cid in range(n_clusters):
        members = [sessions[i] for i, l in enumerate(labels) if l == cid]
        avg_dur = np.mean([m["duration_s"] for m in members]) if members else 0
        avg_act = np.mean([m["num_actions"] for m in members]) if members else 0

        clusters.append({
            "id": cid,
            "session_ids": [m["session_id"] for m in members],
            "size": len(members),
            "dominant_traits": {
                "age_group": "synthetic",
                "country": "unknown",
                "nyc_familiarity": "unknown",
                "age_distribution": {},
                "country_distribution": {},
            },
            "avg_duration_s": round(float(avg_dur), 1),
            "avg_actions": round(float(avg_act), 1),
            "centroid": kmeans.cluster_centers_[cid].tolist(),
        })

    cluster_data = {
        "num_clusters": n_clusters,
        "total_sessions": len(sessions),
        "clusters": clusters,
    }

    # ── PHASE 3: Label clusters ──
    print(f"\n{'='*60}")
    print("PHASE 3: Generating cluster labels via Mistral")
    print("=" * 60)

    cluster_data = generate_cluster_labels(cluster_data, client)

    # Save clusters.json
    clusters_path = CLUSTERS_DIR / "clusters.json"
    with open(clusters_path, "w") as f:
        json.dump(cluster_data, f, indent=2)
    print(f"\n  Saved clusters -> {clusters_path}")

    # ── PHASE 4: Build training data ──
    if not args.skip_training_data:
        print(f"\n{'='*60}")
        print("PHASE 4: Building fine-tuning training data (JSONL)")
        print("=" * 60)

        total_train, total_val = build_training_data(
            cluster_data,
            window_size=args.window_size,
            val_split=args.val_split,
        )
    else:
        print(f"\n  Skipping training data generation (--skip-training-data)")
        total_train = total_val = 0

    # ── Summary ──
    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Sessions processed: {len(sessions)}")
    print(f"  Clusters: {n_clusters}")
    for c in cluster_data["clusters"]:
        print(f"    Cluster {c['id']}: {c.get('label', '?')} ({c['size']} sessions, "
              f"avg {c['avg_actions']} actions, avg {c['avg_duration_s']}s)")

    if total_train:
        print(f"\n  Training examples: {total_train}")
        print(f"  Validation examples: {total_val}")
        for f in sorted(TRAINING_DIR.glob("*.jsonl")):
            lines = sum(1 for _ in open(f))
            size_kb = f.stat().st_size / 1024
            print(f"    {f.name}: {lines} examples ({size_kb:.1f} KB)")

    print(f"\n  Next step: python fine_tune.py")
    print("=" * 60)


if __name__ == "__main__":
    main()

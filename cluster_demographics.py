"""
Batch-process all recordings through Stages 1-3 (parse, describe, encode),
then cluster into 3 demographic groups using K-Means on behavioral embeddings.

Usage:
    python cluster_demographics.py
    python cluster_demographics.py --clusters 3
"""

import json
import sys
import argparse
import pickle
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    MISTRAL_API_KEY,
    RECORDINGS_DIR,
    PARSED_DIR,
    DESCRIPTIONS_DIR,
    EMBEDDINGS_DIR,
    CLUSTERS_DIR,
    ensure_data_dirs,
)
from pipeline.stage1_parse import parse_recording
from pipeline.stage2_describe import describe_and_save
from pipeline.stage3_encode import encode_and_save
from mistralai import Mistral


def process_all_recordings(min_actions: int = 5) -> list[dict]:
    """Run stages 1-3 on all recordings, skipping cached results. Returns session metadata."""
    recordings = sorted(RECORDINGS_DIR.glob("export-*-ph-recording.json"))
    print(f"Found {len(recordings)} recordings in {RECORDINGS_DIR}\n")

    sessions = []

    for i, rec_path in enumerate(recordings):
        stem = rec_path.stem
        session_id = stem.replace("export-", "").replace("-ph-recording", "")
        print(f"[{i+1}/{len(recordings)}] {session_id}")

        # Stage 1: Parse
        parsed_path = PARSED_DIR / f"parsed_{session_id}.json"
        if parsed_path.exists():
            print(f"  Stage 1: cached")
            with open(parsed_path) as f:
                parsed = json.load(f)
        else:
            print(f"  Stage 1: parsing...")
            try:
                parsed = parse_recording(str(rec_path), str(parsed_path))
            except Exception as e:
                print(f"  Stage 1 FAILED: {e}")
                continue

        num_actions = len(parsed.get("high_level_actions", []))
        if num_actions < min_actions:
            print(f"  Skipped: only {num_actions} actions (min={min_actions})")
            continue

        profile = parsed.get("user_profile", {})
        print(f"  Actions: {num_actions}, Duration: {parsed.get('session_duration_s', 0):.0f}s, "
              f"Age: {profile.get('age_group', '?')}, Country: {profile.get('country', '?')}")

        # Stage 2: Describe
        desc_path = DESCRIPTIONS_DIR / f"description_{session_id}.txt"
        if desc_path.exists():
            print(f"  Stage 2: cached")
            with open(desc_path) as f:
                description = f.read()
        else:
            print(f"  Stage 2: generating description via Mistral...")
            try:
                description = describe_and_save(parsed, MISTRAL_API_KEY, str(desc_path))
            except Exception as e:
                print(f"  Stage 2 FAILED: {e}")
                continue

        # Stage 3: Encode
        embed_path = EMBEDDINGS_DIR / f"embedding_{session_id}.json"
        if embed_path.exists():
            print(f"  Stage 3: cached")
            with open(embed_path) as f:
                embed_result = json.load(f)
        else:
            print(f"  Stage 3: encoding via Mistral Embed...")
            try:
                embed_result = encode_and_save(description, profile, MISTRAL_API_KEY, str(embed_path))
            except Exception as e:
                print(f"  Stage 3 FAILED: {e}")
                continue

        sessions.append({
            "session_id": session_id,
            "profile": profile,
            "num_actions": num_actions,
            "duration_s": parsed.get("session_duration_s", 0),
            "description_path": str(desc_path),
            "embedding": embed_result["embedding"],
        })

    return sessions


def cluster_sessions(sessions: list[dict], n_clusters: int = 3) -> dict:
    """Cluster sessions by behavioral embedding using K-Means."""
    if len(sessions) < n_clusters:
        print(f"WARNING: Only {len(sessions)} sessions, reducing clusters to {len(sessions)}")
        n_clusters = max(1, len(sessions))

    embeddings = np.array([s["embedding"] for s in sessions])
    print(f"\nClustering {len(sessions)} sessions into {n_clusters} groups...")
    print(f"  Embedding dimensions: {embeddings.shape[1]}")

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    # Persist the fitted model so the online pipeline can classify new sessions
    kmeans_path = CLUSTERS_DIR / "kmeans_model.pkl"
    with open(kmeans_path, "wb") as f:
        pickle.dump(kmeans, f)
    print(f"  Saved KMeans model to {kmeans_path}")

    # Build cluster info
    clusters = []
    for cluster_id in range(n_clusters):
        member_indices = [i for i, l in enumerate(labels) if l == cluster_id]
        members = [sessions[i] for i in member_indices]

        # Dominant traits
        age_groups = Counter(m["profile"].get("age_group") for m in members if m["profile"].get("age_group"))
        countries = Counter(m["profile"].get("country") for m in members if m["profile"].get("country"))
        nyc_fam = Counter(m["profile"].get("nyc_familiarity") for m in members if m["profile"].get("nyc_familiarity"))

        avg_duration = np.mean([m["duration_s"] for m in members]) if members else 0
        avg_actions = np.mean([m["num_actions"] for m in members]) if members else 0

        clusters.append({
            "id": cluster_id,
            "session_ids": [m["session_id"] for m in members],
            "size": len(members),
            "dominant_traits": {
                "age_group": age_groups.most_common(1)[0][0] if age_groups else "unknown",
                "country": countries.most_common(1)[0][0] if countries else "unknown",
                "nyc_familiarity": nyc_fam.most_common(1)[0][0] if nyc_fam else "unknown",
                "age_distribution": dict(age_groups),
                "country_distribution": dict(countries),
            },
            "avg_duration_s": round(avg_duration, 1),
            "avg_actions": round(avg_actions, 1),
            "centroid": kmeans.cluster_centers_[cluster_id].tolist(),
        })

    return {
        "num_clusters": n_clusters,
        "total_sessions": len(sessions),
        "clusters": clusters,
    }


def generate_cluster_labels(cluster_data: dict, sessions: list[dict]) -> dict:
    """Use Mistral to generate human-readable labels for each cluster."""
    client = Mistral(api_key=MISTRAL_API_KEY)

    for cluster in cluster_data["clusters"]:
        # Collect a few description excerpts from cluster members
        excerpts = []
        for sid in cluster["session_ids"][:3]:
            desc_path = DESCRIPTIONS_DIR / f"description_{sid}.txt"
            if desc_path.exists():
                with open(desc_path) as f:
                    text = f.read()
                excerpts.append(text[:500])

        traits = cluster["dominant_traits"]
        prompt = f"""Based on these user behavioral profiles from a web app, generate a short demographic persona label and description.

CLUSTER STATS:
- {cluster['size']} users in this group
- Dominant age group: {traits['age_group']}
- Dominant country: {traits['country']}
- NYC familiarity: {traits.get('nyc_familiarity', 'unknown')}
- Average session duration: {cluster['avg_duration_s']}s
- Average actions per session: {cluster['avg_actions']}
- Age distribution: {traits.get('age_distribution', {})}
- Country distribution: {traits.get('country_distribution', {})}

SAMPLE BEHAVIORAL DESCRIPTIONS:
{chr(10).join(f'--- User {i+1} ---{chr(10)}{e}' for i, e in enumerate(excerpts))}

Output ONLY valid JSON:
{{
  "label": "<short 2-4 word label like 'Young Urban Explorer' or 'Cautious First-Timer'>",
  "description": "<1-2 sentence behavioral summary of this demographic>",
  "key_behaviors": ["<behavior 1>", "<behavior 2>", "<behavior 3>"]
}}"""

        print(f"\n  Generating label for cluster {cluster['id']} ({cluster['size']} members)...")
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
            print(f"    Label: {cluster['label']}")
            print(f"    Description: {cluster['description']}")
        except Exception as e:
            print(f"    Label generation failed: {e}")
            cluster["label"] = f"Demographic {cluster['id']}"
            cluster["description"] = ""
            cluster["key_behaviors"] = []

    return cluster_data


def main():
    parser = argparse.ArgumentParser(description="Process recordings and cluster into demographics")
    parser.add_argument("--clusters", type=int, default=3, help="Number of demographic clusters")
    parser.add_argument("--min-actions", type=int, default=5, help="Min actions to include a recording")
    args = parser.parse_args()

    if not MISTRAL_API_KEY:
        print("ERROR: MISTRAL_API_KEY not set in .env")
        sys.exit(1)

    ensure_data_dirs()

    # Stage 1-3: Process all recordings
    print("=" * 60)
    print("PHASE 1: Processing all recordings (Stages 1-3)")
    print("=" * 60)
    sessions = process_all_recordings(min_actions=args.min_actions)

    if not sessions:
        print("ERROR: No valid sessions to cluster")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"PHASE 2: Clustering {len(sessions)} sessions")
    print(f"{'='*60}")
    cluster_data = cluster_sessions(sessions, n_clusters=args.clusters)

    # Generate labels
    print(f"\n{'='*60}")
    print("PHASE 3: Generating cluster labels")
    print(f"{'='*60}")
    cluster_data = generate_cluster_labels(cluster_data, sessions)

    # Save
    output_path = CLUSTERS_DIR / "clusters.json"
    with open(output_path, "w") as f:
        json.dump(cluster_data, f, indent=2)
    print(f"\nSaved cluster data to {output_path}")

    # Summary
    print(f"\n{'='*60}")
    print("CLUSTERING COMPLETE")
    print(f"{'='*60}")
    for c in cluster_data["clusters"]:
        traits = c["dominant_traits"]
        print(f"\n  Cluster {c['id']}: {c.get('label', '?')}")
        print(f"    Members: {c['size']}")
        print(f"    Age: {traits['age_group']}, Country: {traits['country']}")
        print(f"    Avg duration: {c['avg_duration_s']}s, Avg actions: {c['avg_actions']}")
        print(f"    Sessions: {', '.join(c['session_ids'][:5])}{'...' if len(c['session_ids']) > 5 else ''}")
        if c.get("description"):
            print(f"    {c['description']}")


if __name__ == "__main__":
    main()

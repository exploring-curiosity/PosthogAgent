"""
Online processor — processes a single new PostHog recording through the full pipeline:
  1. Download recording from PostHog API
  2. Parse (Stage 1)
  3. Describe behavior (Stage 2 — Mistral)
  4. Embed (Stage 3 — Mistral Embed)
  5. Classify against existing KMeans clusters
"""

import json
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    MISTRAL_API_KEY,
    POSTHOG_PERSONAL_API_KEY,
    POSTHOG_PROJECT_ID,
    POSTHOG_HOST,
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
from download_recordings import download_snapshots, get_recording_person, save_recording
from online_pipeline.store import add_session, is_session_processed


KMEANS_PATH = CLUSTERS_DIR / "kmeans_model.pkl"
CLUSTERS_JSON = CLUSTERS_DIR / "clusters.json"


class ProcessingResult:
    """Result of processing a single recording."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.status = "pending"
        self.stage = ""
        self.cluster_id: int | None = None
        self.cluster_label: str = ""
        self.num_actions: int = 0
        self.description_preview: str = ""
        self.error: str = ""
        self.started_at = datetime.now().isoformat()
        self.completed_at: str = ""

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "status": self.status,
            "stage": self.stage,
            "cluster_id": self.cluster_id,
            "cluster_label": self.cluster_label,
            "num_actions": self.num_actions,
            "description_preview": self.description_preview,
            "error": self.error,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }

    def fail(self, stage: str, error: str):
        self.status = "failed"
        self.stage = stage
        self.error = error
        self.completed_at = datetime.now().isoformat()

    def complete(self, cluster_id: int, cluster_label: str):
        self.status = "completed"
        self.cluster_id = cluster_id
        self.cluster_label = cluster_label
        self.completed_at = datetime.now().isoformat()


def load_kmeans():
    """Load the saved KMeans model. Returns (kmeans, clusters_data) or (None, None)."""
    if not KMEANS_PATH.exists():
        return None, None
    if not CLUSTERS_JSON.exists():
        return None, None

    with open(KMEANS_PATH, "rb") as f:
        kmeans = pickle.load(f)
    with open(CLUSTERS_JSON) as f:
        clusters_data = json.load(f)

    return kmeans, clusters_data


def classify_embedding(embedding: list[float], kmeans, clusters_data: dict) -> tuple[int, str]:
    """Predict the cluster for a single embedding. Returns (cluster_id, cluster_label)."""
    vec = np.array(embedding).reshape(1, -1)
    cluster_id = int(kmeans.predict(vec)[0])

    # Find label
    label = f"Cluster {cluster_id}"
    for c in clusters_data.get("clusters", []):
        if c["id"] == cluster_id:
            label = c.get("label", label)
            break

    return cluster_id, label


def process_recording(recording_id: str, skip_download: bool = False) -> ProcessingResult:
    """
    Process a single recording end-to-end.

    Args:
        recording_id: The PostHog recording ID (UUID).
        skip_download: If True, assume recording file already exists on disk.

    Returns:
        ProcessingResult with status, cluster assignment, etc.
    """
    result = ProcessingResult(recording_id)
    ensure_data_dirs()

    # Check if already processed
    if is_session_processed(recording_id):
        result.status = "skipped"
        result.stage = "already_processed"
        result.completed_at = datetime.now().isoformat()
        return result

    # Load KMeans model
    kmeans, clusters_data = load_kmeans()
    if kmeans is None:
        result.fail("init", "No KMeans model found. Run cluster_demographics.py first, then save_kmeans().")
        return result

    recording_path = RECORDINGS_DIR / f"export-{recording_id}-ph-recording.json"

    # ── Step 1: Download ──
    result.stage = "download"
    if not skip_download and not recording_path.exists():
        try:
            if not POSTHOG_PERSONAL_API_KEY or not POSTHOG_PROJECT_ID:
                result.fail("download", "PostHog API credentials not configured")
                return result

            person = get_recording_person(
                POSTHOG_HOST, POSTHOG_PROJECT_ID, POSTHOG_PERSONAL_API_KEY, recording_id
            )
            snapshots = download_snapshots(
                POSTHOG_HOST, POSTHOG_PROJECT_ID, POSTHOG_PERSONAL_API_KEY, recording_id
            )
            if not snapshots:
                result.fail("download", "No snapshots returned from PostHog")
                return result

            save_recording({"id": recording_id}, person, snapshots, recording_path)
        except Exception as e:
            result.fail("download", str(e))
            return result

    if not recording_path.exists():
        result.fail("download", f"Recording file not found: {recording_path}")
        return result

    # ── Step 2: Parse (Stage 1) ──
    result.stage = "parse"
    parsed_path = PARSED_DIR / f"parsed_{recording_id}.json"
    try:
        if parsed_path.exists():
            with open(parsed_path) as f:
                parsed = json.load(f)
        else:
            parsed = parse_recording(str(recording_path), str(parsed_path))
    except Exception as e:
        result.fail("parse", str(e))
        return result

    num_actions = len(parsed.get("high_level_actions", []))
    result.num_actions = num_actions
    if num_actions < 3:
        result.fail("parse", f"Too few actions ({num_actions}), skipping")
        return result

    profile = parsed.get("user_profile", {})

    # ── Step 3: Describe (Stage 2) ──
    result.stage = "describe"
    desc_path = DESCRIPTIONS_DIR / f"description_{recording_id}.txt"
    try:
        if desc_path.exists():
            with open(desc_path) as f:
                description = f.read()
        else:
            description = describe_and_save(parsed, MISTRAL_API_KEY, str(desc_path))
    except Exception as e:
        result.fail("describe", str(e))
        return result

    result.description_preview = description[:200]

    # ── Step 4: Embed (Stage 3) ──
    result.stage = "embed"
    embed_path = EMBEDDINGS_DIR / f"embedding_{recording_id}.json"
    try:
        if embed_path.exists():
            with open(embed_path) as f:
                embed_result = json.load(f)
        else:
            embed_result = encode_and_save(description, profile, MISTRAL_API_KEY, str(embed_path))
    except Exception as e:
        result.fail("embed", str(e))
        return result

    # ── Step 5: Classify ──
    result.stage = "classify"
    try:
        cluster_id, cluster_label = classify_embedding(
            embed_result["embedding"], kmeans, clusters_data
        )
    except Exception as e:
        result.fail("classify", str(e))
        return result

    # ── Store result ──
    add_session(
        session_id=recording_id,
        cluster_id=cluster_id,
        recording_path=str(recording_path),
        description_preview=description[:200],
        embedding_dim=embed_result.get("embedding_dim", 0),
    )

    result.complete(cluster_id, cluster_label)
    return result

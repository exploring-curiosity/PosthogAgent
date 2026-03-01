"""
Online retrain — rebuild training data and trigger fine-tuning
for clusters that have accumulated enough new data.

This reuses the existing build_training_data.py and fine_tune.py logic
but can be triggered programmatically from the API.
"""

import json
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CLUSTERS_DIR, TRAINING_DIR, MODELS_DIR, PROJECT_ROOT
from online_pipeline.store import (
    get_status,
    record_retrain,
    update_retrain_status,
)

# Minimum new sessions per cluster before retrain is worthwhile
MIN_NEW_SESSIONS_FOR_RETRAIN = 2

# Track running retrain jobs
_retrain_lock = threading.Lock()
_retrain_running = False


def check_retrain_ready() -> dict:
    """Check which clusters have enough new data to justify retraining."""
    status = get_status()
    counts = status.get("cluster_new_counts", {})

    ready_clusters = []
    for cid_str, count in counts.items():
        if count >= MIN_NEW_SESSIONS_FOR_RETRAIN:
            ready_clusters.append({
                "cluster_id": int(cid_str),
                "new_sessions": count,
            })

    return {
        "ready": len(ready_clusters) > 0,
        "clusters": ready_clusters,
        "threshold": MIN_NEW_SESSIONS_FOR_RETRAIN,
    }


def trigger_retrain(cluster_ids: list[int] | None = None) -> dict:
    """
    Trigger retraining for specified clusters (or all ready clusters).
    Runs build_training_data.py then fine_tune.py in a background thread.

    Returns immediately with a job status dict.
    """
    global _retrain_running

    with _retrain_lock:
        if _retrain_running:
            return {"status": "already_running", "message": "A retrain job is already in progress"}
        _retrain_running = True

    # Determine which clusters to retrain
    if cluster_ids is None:
        readiness = check_retrain_ready()
        cluster_ids = [c["cluster_id"] for c in readiness["clusters"]]

    if not cluster_ids:
        with _retrain_lock:
            _retrain_running = False
        return {"status": "no_data", "message": "No clusters have enough new data to retrain"}

    # Record the retrain event
    entry = record_retrain(cluster_ids, status="running")
    retrain_index = len(get_status()["retrain_history"]) - 1

    # Run in background
    thread = threading.Thread(
        target=_run_retrain,
        args=(cluster_ids, retrain_index),
        daemon=True,
    )
    thread.start()

    return {
        "status": "started",
        "cluster_ids": cluster_ids,
        "triggered_at": entry["triggered_at"],
    }


def _run_retrain(cluster_ids: list[int], retrain_index: int):
    """Background worker: rebuild training data then fine-tune."""
    global _retrain_running
    python = sys.executable

    try:
        # Step 1: Rebuild training data
        update_retrain_status(retrain_index, "building_training_data")
        result = subprocess.run(
            [python, "build_training_data.py"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            update_retrain_status(
                retrain_index, "failed",
                f"build_training_data failed: {result.stderr[-500:]}"
            )
            return

        # Step 2: Fine-tune
        update_retrain_status(retrain_index, "fine_tuning")
        result = subprocess.run(
            [python, "fine_tune.py"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=1800,  # fine-tuning can take a while
        )
        if result.returncode != 0:
            update_retrain_status(
                retrain_index, "failed",
                f"fine_tune failed: {result.stderr[-500:]}"
            )
            return

        update_retrain_status(retrain_index, "completed")

    except subprocess.TimeoutExpired:
        update_retrain_status(retrain_index, "failed", "Timed out")
    except Exception as e:
        update_retrain_status(retrain_index, "failed", str(e))
    finally:
        with _retrain_lock:
            _retrain_running = False


def get_retrain_status() -> dict:
    """Get current retrain job status."""
    global _retrain_running
    status = get_status()
    history = status.get("retrain_history", [])
    latest = history[-1] if history else None

    return {
        "is_running": _retrain_running,
        "latest_job": latest,
        "readiness": check_retrain_ready(),
    }

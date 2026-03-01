"""
JSON-backed store for online pipeline state.

Tracks:
- All processed sessions with their cluster assignment
- Per-cluster counts of new (unretrained) sessions
- Retrain history
"""

import json
import threading
from datetime import datetime
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR

STORE_PATH = DATA_DIR / "online_store.json"


def _default_store() -> dict:
    return {
        "sessions": {},          # session_id -> {cluster_id, processed_at, recording_path, ...}
        "cluster_new_counts": {},  # cluster_id (str) -> count of sessions since last retrain
        "retrain_history": [],   # [{triggered_at, cluster_ids, status, ...}]
    }


_lock = threading.Lock()


def _load() -> dict:
    if STORE_PATH.exists():
        with open(STORE_PATH) as f:
            return json.load(f)
    return _default_store()


def _save(store: dict):
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STORE_PATH, "w") as f:
        json.dump(store, f, indent=2)


def add_session(session_id: str, cluster_id: int, recording_path: str,
                description_preview: str = "", embedding_dim: int = 0) -> dict:
    """Record a newly processed session and increment new-data counter for its cluster."""
    with _lock:
        store = _load()
        entry = {
            "cluster_id": cluster_id,
            "processed_at": datetime.now().isoformat(),
            "recording_path": recording_path,
            "description_preview": description_preview[:200],
            "embedding_dim": embedding_dim,
        }
        store["sessions"][session_id] = entry

        cid = str(cluster_id)
        store["cluster_new_counts"][cid] = store["cluster_new_counts"].get(cid, 0) + 1

        _save(store)
        return entry


def get_status() -> dict:
    """Return current pipeline status: total sessions, per-cluster new counts, retrain history."""
    with _lock:
        store = _load()
        return {
            "total_processed": len(store["sessions"]),
            "cluster_new_counts": store["cluster_new_counts"],
            "retrain_history": store["retrain_history"][-5:],  # last 5
            "recent_sessions": _recent_sessions(store, n=10),
        }


def _recent_sessions(store: dict, n: int = 10) -> list[dict]:
    """Return the N most recently processed sessions."""
    items = [
        {"session_id": sid, **info}
        for sid, info in store["sessions"].items()
    ]
    items.sort(key=lambda x: x.get("processed_at", ""), reverse=True)
    return items[:n]


def get_new_session_ids_for_cluster(cluster_id: int) -> list[str]:
    """Return session IDs assigned to a cluster that haven't been retrained on yet."""
    with _lock:
        store = _load()
        # Find the last retrain time for this cluster
        last_retrain = None
        for entry in reversed(store["retrain_history"]):
            if cluster_id in entry.get("cluster_ids", []):
                last_retrain = entry.get("triggered_at", "")
                break

        result = []
        for sid, info in store["sessions"].items():
            if info["cluster_id"] != cluster_id:
                continue
            if last_retrain and info.get("processed_at", "") <= last_retrain:
                continue
            result.append(sid)
        return result


def get_all_session_ids_for_cluster(cluster_id: int) -> list[str]:
    """Return all session IDs assigned to a given cluster."""
    with _lock:
        store = _load()
        return [
            sid for sid, info in store["sessions"].items()
            if info["cluster_id"] == cluster_id
        ]


def record_retrain(cluster_ids: list[int], status: str = "started") -> dict:
    """Record that a retrain was triggered."""
    with _lock:
        store = _load()
        entry = {
            "triggered_at": datetime.now().isoformat(),
            "cluster_ids": cluster_ids,
            "status": status,
        }
        store["retrain_history"].append(entry)

        # Reset new-data counters for retrained clusters
        if status == "completed":
            for cid in cluster_ids:
                store["cluster_new_counts"][str(cid)] = 0

        _save(store)
        return entry


def update_retrain_status(index: int, status: str, details: str = ""):
    """Update the status of a retrain entry by index."""
    with _lock:
        store = _load()
        if 0 <= index < len(store["retrain_history"]):
            store["retrain_history"][index]["status"] = status
            if details:
                store["retrain_history"][index]["details"] = details
            store["retrain_history"][index]["updated_at"] = datetime.now().isoformat()

            # Reset counters on completion
            if status == "completed":
                for cid in store["retrain_history"][index].get("cluster_ids", []):
                    store["cluster_new_counts"][str(cid)] = 0

            _save(store)


def is_session_processed(session_id: str) -> bool:
    """Check if a session has already been processed."""
    with _lock:
        store = _load()
        return session_id in store["sessions"]

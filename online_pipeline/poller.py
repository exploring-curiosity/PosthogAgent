"""
PostHog recording poller — periodically checks for new recordings
and feeds them into the online pipeline.

Alternative to webhooks for environments where PostHog can't send webhooks.
"""

import json
import time
import threading
from pathlib import Path
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    POSTHOG_PERSONAL_API_KEY,
    POSTHOG_PROJECT_ID,
    POSTHOG_HOST,
    RECORDINGS_DIR,
)
from online_pipeline.store import is_session_processed
from online_pipeline.processor import process_recording

import httpx

_poller_thread: threading.Thread | None = None
_poller_running = False
_poll_status = {
    "running": False,
    "last_poll": None,
    "last_found": 0,
    "last_processed": 0,
    "total_polls": 0,
    "errors": [],
}


def _list_recent_recordings(limit: int = 20) -> list[dict]:
    """Fetch recent recordings from PostHog API."""
    resp = httpx.get(
        f"{POSTHOG_HOST}/api/projects/{POSTHOG_PROJECT_ID}/session_recordings",
        headers={"Authorization": f"Bearer {POSTHOG_PERSONAL_API_KEY}"},
        params={"limit": limit, "order": "-start_time"},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json().get("results", [])


def _poll_once() -> dict:
    """Run a single poll cycle. Returns summary."""
    global _poll_status

    found = 0
    processed = 0
    errors = []

    try:
        recordings = _list_recent_recordings()
        found = len(recordings)

        for rec in recordings:
            rid = rec["id"]
            duration = rec.get("recording_duration", 0)

            if duration < 10:
                continue
            if is_session_processed(rid):
                continue

            # Check if recording file already exists on disk
            recording_path = RECORDINGS_DIR / f"export-{rid}-ph-recording.json"
            skip_download = recording_path.exists()

            try:
                result = process_recording(rid, skip_download=skip_download)
                if result.status == "completed":
                    processed += 1
                    print(f"  [poller] Processed {rid} -> cluster {result.cluster_id} ({result.cluster_label})")
                elif result.status == "failed":
                    errors.append(f"{rid}: {result.error}")
            except Exception as e:
                errors.append(f"{rid}: {str(e)}")

    except Exception as e:
        errors.append(f"Poll failed: {str(e)}")

    _poll_status.update({
        "last_poll": datetime.now().isoformat(),
        "last_found": found,
        "last_processed": processed,
        "total_polls": _poll_status["total_polls"] + 1,
        "errors": errors[-5:],  # keep last 5
    })

    return {"found": found, "processed": processed, "errors": errors}


def _poll_loop(interval_s: int):
    """Background polling loop."""
    global _poller_running, _poll_status
    _poll_status["running"] = True

    while _poller_running:
        try:
            summary = _poll_once()
            if summary["processed"] > 0:
                print(f"  [poller] Processed {summary['processed']} new recordings")
        except Exception as e:
            print(f"  [poller] Error: {e}")

        # Sleep in small increments so we can stop quickly
        for _ in range(interval_s):
            if not _poller_running:
                break
            time.sleep(1)

    _poll_status["running"] = False


def start_poller(interval_s: int = 60) -> dict:
    """Start the background poller. Returns status."""
    global _poller_thread, _poller_running

    if _poller_running:
        return {"status": "already_running", "interval_s": interval_s}

    if not POSTHOG_PERSONAL_API_KEY or not POSTHOG_PROJECT_ID:
        return {"status": "error", "message": "PostHog API credentials not configured"}

    _poller_running = True
    _poller_thread = threading.Thread(target=_poll_loop, args=(interval_s,), daemon=True)
    _poller_thread.start()

    return {"status": "started", "interval_s": interval_s}


def stop_poller() -> dict:
    """Stop the background poller."""
    global _poller_running
    _poller_running = False
    return {"status": "stopped"}


def get_poller_status() -> dict:
    """Return current poller status."""
    return _poll_status.copy()

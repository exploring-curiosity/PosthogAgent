"""
FastAPI server for the online pipeline.

Endpoints:
  POST /webhook/posthog       — PostHog webhook receiver (new recording events)
  POST /process/{recording_id} — Manually trigger processing of a recording
  GET  /status                 — Pipeline status: processed sessions, cluster counts
  GET  /retrain/check          — Check if any clusters are ready for retraining
  POST /retrain/trigger        — Trigger retraining for clusters with new data
  GET  /retrain/status         — Current retrain job status

Usage:
  uvicorn online_pipeline.server:app --port 8100 --reload
"""

import sys
import threading
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent))

from online_pipeline.processor import process_recording, ProcessingResult
from online_pipeline.store import get_status, is_session_processed
from online_pipeline.retrain import (
    check_retrain_ready,
    trigger_retrain,
    get_retrain_status,
)
from online_pipeline.poller import start_poller, stop_poller, get_poller_status

app = FastAPI(
    title="PostHog Agent — Online Pipeline",
    description="Webhook-driven processing of new PostHog recordings",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-flight processing tracker ──
_processing: dict[str, ProcessingResult] = {}
_processing_lock = threading.Lock()


# ── Models ──

class ProcessRequest(BaseModel):
    recording_id: str
    skip_download: bool = False


class RetrainRequest(BaseModel):
    cluster_ids: Optional[list[int]] = None


class WebhookEvent(BaseModel):
    """Minimal PostHog webhook payload for recording events."""
    event: Optional[str] = None
    properties: Optional[dict] = None
    # PostHog sends various shapes; we extract what we need


# ── Background processing ──

def _process_in_background(recording_id: str, skip_download: bool = False):
    """Run the full pipeline for a recording in a background thread."""
    result = process_recording(recording_id, skip_download=skip_download)
    with _processing_lock:
        _processing[recording_id] = result


# ── Endpoints ──

@app.post("/webhook/posthog")
async def posthog_webhook(request: Request, background_tasks: BackgroundTasks):
    """
    Receive PostHog webhook events.

    PostHog can send webhooks on various triggers. We look for recording-related
    events and extract the session_recording_id to process.

    Configure in PostHog:
      Settings → Webhooks → Add webhook URL → https://your-server/webhook/posthog
      Or use Actions → Post to webhook when action matches.
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    # PostHog webhook payloads vary. Try to extract a recording ID.
    recording_id = None

    # Option 1: Direct recording event with session_id in properties
    if isinstance(body, dict):
        props = body.get("properties", {}) or {}
        recording_id = (
            props.get("$session_recording_id")
            or props.get("$session_id")
            or props.get("session_id")
        )

        # Option 2: The event itself might carry the recording ID
        if not recording_id and body.get("event") == "$recording_completed":
            recording_id = props.get("$session_id")

        # Option 3: Batch webhook with a list of events
        if not recording_id and isinstance(body.get("data"), dict):
            recording_id = body["data"].get("session_id")

    if not recording_id:
        return {"status": "ignored", "reason": "No recording ID found in webhook payload"}

    # Skip if already processed
    if is_session_processed(recording_id):
        return {"status": "skipped", "recording_id": recording_id, "reason": "already processed"}

    # Process in background
    background_tasks.add_task(_process_in_background, recording_id)

    return {"status": "accepted", "recording_id": recording_id}


@app.post("/process/{recording_id}")
async def process_single(recording_id: str, background_tasks: BackgroundTasks, skip_download: bool = False):
    """Manually trigger processing of a specific recording."""
    if is_session_processed(recording_id):
        return {"status": "skipped", "recording_id": recording_id, "reason": "already processed"}

    with _processing_lock:
        if recording_id in _processing and _processing[recording_id].status == "pending":
            return {"status": "in_progress", "recording_id": recording_id}

    background_tasks.add_task(_process_in_background, recording_id, skip_download)
    return {"status": "accepted", "recording_id": recording_id}


@app.get("/process/{recording_id}/status")
async def process_status(recording_id: str):
    """Check processing status for a specific recording."""
    with _processing_lock:
        if recording_id in _processing:
            return _processing[recording_id].to_dict()

    if is_session_processed(recording_id):
        # Pull details from the store
        status = get_status()
        for s in status.get("recent_sessions", []):
            if s["session_id"] == recording_id:
                return {**s, "status": "completed"}
        return {"session_id": recording_id, "status": "completed"}

    return {"session_id": recording_id, "status": "unknown"}


@app.get("/status")
async def pipeline_status():
    """Get overall pipeline status: total processed, per-cluster counts, recent sessions."""
    return get_status()


@app.get("/retrain/check")
async def retrain_check():
    """Check if any clusters have enough new data to justify retraining."""
    return check_retrain_ready()


@app.post("/retrain/trigger")
async def retrain_trigger(body: RetrainRequest = RetrainRequest()):
    """Trigger retraining. Optionally specify cluster IDs, or retrain all ready clusters."""
    result = trigger_retrain(cluster_ids=body.cluster_ids)
    if result["status"] == "already_running":
        raise HTTPException(status_code=409, detail=result["message"])
    return result


@app.get("/retrain/status")
async def retrain_status():
    """Get current retrain job status."""
    return get_retrain_status()


@app.post("/poller/start")
async def poller_start(interval_s: int = 60):
    """Start background polling for new PostHog recordings."""
    return start_poller(interval_s=interval_s)


@app.post("/poller/stop")
async def poller_stop():
    """Stop background polling."""
    return stop_poller()


@app.get("/poller/status")
async def poller_status():
    """Get poller status."""
    return get_poller_status()


@app.get("/health")
async def health():
    """Health check."""
    return {"status": "ok", "service": "online-pipeline"}

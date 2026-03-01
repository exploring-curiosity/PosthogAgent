"""
============================================================
Agentic Evaluation Service (FastAPI)
============================================================
Persistent HTTP service that evaluates a target website using
all trained cluster models sequentially.

Endpoints:
    POST /evaluate       — Start evaluation (runs all clusters against URL)
    GET  /status/{id}    — Check job status
    GET  /results/{id}   — Get full results
    GET  /clusters       — List available clusters
    GET  /health         — Health check

Usage:
    uvicorn service:app --host 0.0.0.0 --port 8000
    # or
    python service.py
============================================================
"""

import os
import sys
import uuid
import time
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import uvicorn

sys.path.insert(0, str(Path(__file__).parent))

from config import ensure_data_dirs, TARGET_APP_NAME, TARGET_APP_DESCRIPTION
from agent_runner import (
    ModelManager,
    AgentRunner,
    get_available_cluster_ids,
    get_cluster_meta,
    load_all_cluster_metas,
)

# ============================================================
# APP + STATE
# ============================================================

app = FastAPI(
    title="Agentic Evaluation Service",
    description="Evaluate websites using per-cluster fine-tuned behavioral models",
    version="1.0.0",
)

ensure_data_dirs()

# Global model manager — loaded once at startup
model_manager: ModelManager | None = None
model_lock = threading.Lock()

# Job store (in-memory)
jobs: dict[str, dict] = {}


# ============================================================
# STARTUP — Load model
# ============================================================

@app.on_event("startup")
def startup_load_model():
    global model_manager
    model_name = os.environ.get("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.3")
    print(f"\n{'='*60}")
    print("AGENTIC EVALUATION SERVICE — STARTING")
    print(f"{'='*60}")
    model_manager = ModelManager(model_name=model_name)
    clusters = get_available_cluster_ids()
    print(f"Available cluster adapters: {clusters}")
    print(f"{'='*60}\n")


# ============================================================
# REQUEST / RESPONSE MODELS
# ============================================================

class EvaluateRequest(BaseModel):
    url: str = Field(..., description="Target URL to evaluate")
    app_name: str = Field(default="Web App", description="Application name")
    app_description: str = Field(..., description="Description of the app and features to test")
    cluster_ids: Optional[list[int]] = Field(
        default=None,
        description="Specific cluster IDs to run. If null, runs all available clusters."
    )
    max_steps: int = Field(default=20, ge=1, le=100, description="Max actions per cluster session")
    max_duration: int = Field(default=300, ge=30, le=600, description="Max seconds per cluster session")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    headless: bool = Field(default=True, description="Run browser in headless mode")


class EvaluateResponse(BaseModel):
    job_id: str
    status: str
    message: str


class JobStatus(BaseModel):
    job_id: str
    status: str  # "queued", "running", "completed", "failed"
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    current_cluster: Optional[int] = None
    clusters_completed: int = 0
    clusters_total: int = 0
    message: str = ""


class ClusterInfo(BaseModel):
    id: int
    label: str
    has_adapter: bool
    description: str = ""
    key_behaviors: list[str] = []


# ============================================================
# BACKGROUND WORKER — runs evaluation across clusters
# ============================================================

def _run_evaluation(job_id: str, request: EvaluateRequest):
    """Background task: run all cluster models against the target URL."""
    global model_manager

    job = jobs[job_id]
    job["status"] = "running"
    job["started_at"] = datetime.now().isoformat()

    available = get_available_cluster_ids()
    cluster_ids = request.cluster_ids or available

    # Filter to only clusters that have adapters
    valid_ids = [cid for cid in cluster_ids if cid in available]
    if not valid_ids:
        job["status"] = "failed"
        job["error"] = f"No valid cluster adapters found. Available: {available}, Requested: {cluster_ids}"
        job["completed_at"] = datetime.now().isoformat()
        return

    job["clusters_total"] = len(valid_ids)
    job["cluster_results"] = {}

    runner = AgentRunner(
        model_manager=model_manager,
        max_steps=request.max_steps,
        max_duration=request.max_duration,
        temperature=request.temperature,
        headless=request.headless,
    )

    print(f"\n[Job {job_id[:8]}] Starting evaluation of {request.url}")
    print(f"  Clusters: {valid_ids}")
    print(f"  Max steps: {request.max_steps}, Max duration: {request.max_duration}s\n")

    for i, cid in enumerate(valid_ids):
        job["current_cluster"] = cid
        job["clusters_completed"] = i
        job["message"] = f"Running cluster {cid} ({i+1}/{len(valid_ids)})"

        meta = get_cluster_meta(cid)
        print(f"[Job {job_id[:8]}] Cluster {cid}: {meta.get('label', '?')}")

        try:
            with model_lock:
                result = runner.run_session(
                    url=request.url,
                    app_name=request.app_name,
                    app_description=request.app_description,
                    cluster_id=cid,
                    cluster_meta=meta,
                )
            job["cluster_results"][cid] = result
            print(f"[Job {job_id[:8]}] Cluster {cid}: {result['status']} — "
                  f"{result.get('total_steps', 0)} steps, "
                  f"{result.get('completion_rate', 0)*100:.0f}% success")
        except Exception as e:
            print(f"[Job {job_id[:8]}] Cluster {cid}: ERROR — {e}")
            job["cluster_results"][cid] = {
                "cluster_id": cid,
                "cluster_label": meta.get("label", "?"),
                "status": "error",
                "error": str(e),
                "actions": [],
            }

    job["clusters_completed"] = len(valid_ids)
    job["current_cluster"] = None
    job["status"] = "completed"
    job["completed_at"] = datetime.now().isoformat()
    job["message"] = f"Evaluation complete: {len(valid_ids)} clusters tested"

    # Build summary
    total_actions = 0
    total_success = 0
    total_failed = 0
    for r in job["cluster_results"].values():
        total_actions += r.get("total_steps", 0)
        total_success += r.get("successful_actions", 0)
        total_failed += r.get("failed_actions", 0)

    job["summary"] = {
        "url": request.url,
        "clusters_tested": len(valid_ids),
        "total_actions": total_actions,
        "total_successful": total_success,
        "total_failed": total_failed,
        "overall_success_rate": round(total_success / total_actions, 2) if total_actions > 0 else 0,
    }

    print(f"\n[Job {job_id[:8]}] COMPLETE — {total_actions} actions, "
          f"{total_success} ok, {total_failed} failed\n")


# ============================================================
# ENDPOINTS
# ============================================================

@app.post("/evaluate", response_model=EvaluateResponse)
def evaluate(request: EvaluateRequest, background_tasks: BackgroundTasks):
    """Start an evaluation job. Runs all cluster models against the target URL."""
    if model_manager is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet. Wait for startup.")

    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "request": request.model_dump(),
        "started_at": None,
        "completed_at": None,
        "current_cluster": None,
        "clusters_completed": 0,
        "clusters_total": 0,
        "cluster_results": {},
        "message": "Queued",
    }

    background_tasks.add_task(_run_evaluation, job_id, request)

    return EvaluateResponse(
        job_id=job_id,
        status="queued",
        message=f"Evaluation queued. Track with GET /status/{job_id}",
    )


@app.get("/status/{job_id}", response_model=JobStatus)
def get_status(job_id: str):
    """Check the status of an evaluation job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    return JobStatus(
        job_id=job_id,
        status=job["status"],
        started_at=job.get("started_at"),
        completed_at=job.get("completed_at"),
        current_cluster=job.get("current_cluster"),
        clusters_completed=job.get("clusters_completed", 0),
        clusters_total=job.get("clusters_total", 0),
        message=job.get("message", ""),
    )


@app.get("/results/{job_id}")
def get_results(job_id: str):
    """Get full results for a completed evaluation job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]

    if job["status"] not in ("completed", "failed"):
        return {
            "job_id": job_id,
            "status": job["status"],
            "message": "Job still in progress. Check /status for updates.",
        }

    return {
        "job_id": job_id,
        "status": job["status"],
        "request": job.get("request"),
        "started_at": job.get("started_at"),
        "completed_at": job.get("completed_at"),
        "summary": job.get("summary", {}),
        "cluster_results": {
            str(k): {
                "cluster_id": v.get("cluster_id"),
                "cluster_label": v.get("cluster_label"),
                "status": v.get("status"),
                "total_steps": v.get("total_steps", 0),
                "successful_actions": v.get("successful_actions", 0),
                "failed_actions": v.get("failed_actions", 0),
                "completion_rate": v.get("completion_rate", 0),
                "duration_s": v.get("duration_s", 0),
                "stuck_events": v.get("stuck_events", 0),
                "actions": v.get("actions", []),
                "log_path": v.get("log_path", ""),
                "error": v.get("error", ""),
            }
            for k, v in job.get("cluster_results", {}).items()
        },
    }


@app.get("/clusters", response_model=list[ClusterInfo])
def list_clusters():
    """List all available clusters and their adapter status."""
    available = set(get_available_cluster_ids())
    all_metas = load_all_cluster_metas()

    result = []
    seen = set()

    for meta in all_metas:
        cid = meta["id"]
        seen.add(cid)
        result.append(ClusterInfo(
            id=cid,
            label=meta.get("label", f"Cluster {cid}"),
            has_adapter=cid in available,
            description=meta.get("description", ""),
            key_behaviors=meta.get("key_behaviors", []),
        ))

    # Include any adapters not in clusters.json
    for cid in available:
        if cid not in seen:
            result.append(ClusterInfo(
                id=cid,
                label=f"Cluster {cid}",
                has_adapter=True,
            ))

    return sorted(result, key=lambda x: x.id)


@app.get("/health")
def health():
    """Health check."""
    return {
        "status": "ok",
        "model_loaded": model_manager is not None,
        "available_clusters": get_available_cluster_ids(),
        "active_jobs": sum(1 for j in jobs.values() if j["status"] == "running"),
        "total_jobs": len(jobs),
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    uvicorn.run(
        "service:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,  # single worker — model is in GPU memory
    )

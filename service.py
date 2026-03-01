"""
============================================================
GPU Inference Service (FastAPI) — runs on VM with A100
============================================================
Pure inference API. No browser. Models stay loaded in GPU.

The local client (local_client.py) runs the browser via AgentQL
and calls this service for action predictions.

Endpoints:
    POST /predict         — Predict next action for a given cluster + page state
    POST /predict/batch   — Predict for ALL clusters given the same page state
    GET  /clusters        — List available clusters + personas
    POST /switch/{id}     — Pre-load a specific cluster adapter
    GET  /health          — Health check

Usage (on VM):
    python service.py
    # Listens on 0.0.0.0:8000
============================================================
"""

import os
import sys
import json
import gc
import threading
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

sys.path.insert(0, str(Path(__file__).parent))

from config import MODELS_DIR, CLUSTERS_DIR, ensure_data_dirs
from build_training_data import BASE_SYSTEM_PROMPT, build_cluster_system_prompt, build_product_context
from agent_runner import ModelManager, get_available_cluster_ids, get_cluster_meta, load_all_cluster_metas

# ============================================================
# APP + STATE
# ============================================================

app = FastAPI(
    title="Agentic GPU Inference Service",
    description="Predict browser actions using per-cluster fine-tuned models. Models stay loaded in GPU.",
    version="2.0.0",
)

ensure_data_dirs()

model_manager: ModelManager | None = None
model_lock = threading.Lock()
_current_cluster_id: int | None = None


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
def startup_load_model():
    global model_manager
    model_name = os.environ.get("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.3")
    print(f"\n{'='*60}")
    print("GPU INFERENCE SERVICE — STARTING")
    print(f"{'='*60}")
    model_manager = ModelManager(model_name=model_name)
    clusters = get_available_cluster_ids()
    print(f"Available cluster adapters: {clusters}")
    if clusters:
        _switch_cluster(clusters[0])
        print(f"Pre-loaded cluster {clusters[0]}")
    print(f"Service ready on port 8000")
    print(f"{'='*60}\n")


def _switch_cluster(cluster_id: int) -> dict:
    """Switch the active LoRA adapter. Returns cluster meta."""
    global _current_cluster_id
    adapter_path = str(MODELS_DIR / f"cluster_{cluster_id}_lora")
    with model_lock:
        success = model_manager.load_adapter(adapter_path)
    if success:
        _current_cluster_id = cluster_id
    meta = get_cluster_meta(cluster_id)
    return meta


# ============================================================
# REQUEST / RESPONSE MODELS
# ============================================================

class PredictRequest(BaseModel):
    cluster_id: int = Field(..., description="Which cluster model to use")
    page_state: str = Field(..., description="Current page observation (URL, title, elements)")
    app_name: str = Field(default="Web App")
    app_url: str = Field(default="")
    app_description: str = Field(default="", description="App feature description")
    action_history: list[dict] = Field(default_factory=list, description="Recent action history")
    error_context: str = Field(default="", description="Error from last failed action (for replanning)")
    step_number: int = Field(default=0)
    elapsed_s: float = Field(default=0.0)
    max_duration_s: float = Field(default=300.0)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    window_size: int = Field(default=5, description="How many recent actions to include in prompt")


class PredictResponse(BaseModel):
    cluster_id: int
    cluster_label: str
    action: str
    target: str
    reasoning: str
    hesitation_ms: int
    raw_output: str = ""


class BatchPredictRequest(BaseModel):
    page_state: str = Field(..., description="Current page observation")
    app_name: str = Field(default="Web App")
    app_url: str = Field(default="")
    app_description: str = Field(default="")
    action_history: list[dict] = Field(default_factory=list)
    error_context: str = Field(default="")
    step_number: int = Field(default=0)
    elapsed_s: float = Field(default=0.0)
    max_duration_s: float = Field(default=300.0)
    temperature: float = Field(default=0.7)
    window_size: int = Field(default=5)
    cluster_ids: Optional[list[int]] = Field(default=None, description="If null, runs all available")


class ClusterInfo(BaseModel):
    id: int
    label: str
    has_adapter: bool
    description: str = ""
    key_behaviors: list[str] = []
    persona_prompt: str = ""


# ============================================================
# CORE INFERENCE
# ============================================================

def _build_prompt(cluster_meta: dict, req: PredictRequest) -> list[dict]:
    """Build the chat messages for the model."""
    system_prompt = build_cluster_system_prompt(cluster_meta)
    product_context = build_product_context(req.app_name, req.app_url, req.app_description)

    lines = [product_context, ""]
    lines.append(f"Session: step {req.step_number}, elapsed {req.elapsed_s:.1f}s / {req.max_duration_s:.0f}s")
    lines.append(f"\n{req.page_state}")

    window = req.action_history[-req.window_size:]
    if window:
        lines.append("\nRecent actions:")
        for a in window:
            status = "" if a.get("success", True) else " [FAILED]"
            line = f"  [{a.get('elapsed', 0):.1f}s] {a.get('action', '?')} -> {a.get('target', '?')[:50]}{status}"
            if not a.get("success", True) and a.get("error"):
                line += f" ({a['error'][:40]})"
            lines.append(line)

    if req.error_context:
        lines.append(f"\nLast action failed: {req.error_context}")
        lines.append("Decide how to recover or try an alternative approach.")
    else:
        lines.append("\nWhat would you do next?")

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "\n".join(lines)},
    ]


def _parse_action(raw: str) -> dict:
    """Parse model JSON output robustly."""
    import re
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r'\{[^{}]*\}', raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return {
        "action": "scroll", "target": "down 300px",
        "reasoning": "fallback — unparseable model output",
        "hesitation_ms": 500,
    }


def _predict_single(cluster_id: int, req: PredictRequest) -> PredictResponse:
    """Run inference for one cluster."""
    global _current_cluster_id

    if _current_cluster_id != cluster_id:
        meta = _switch_cluster(cluster_id)
    else:
        meta = get_cluster_meta(cluster_id)

    messages = _build_prompt(meta, req)

    with model_lock:
        raw = model_manager.generate(messages, temperature=req.temperature)

    action_data = _parse_action(raw)

    return PredictResponse(
        cluster_id=cluster_id,
        cluster_label=meta.get("label", f"Cluster {cluster_id}"),
        action=action_data.get("action", "wait"),
        target=action_data.get("target", ""),
        reasoning=action_data.get("reasoning", ""),
        hesitation_ms=action_data.get("hesitation_ms", 0),
        raw_output=raw[:500],
    )


# ============================================================
# ENDPOINTS
# ============================================================

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """Predict the next browser action for a specific cluster model."""
    if model_manager is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    available = get_available_cluster_ids()
    if req.cluster_id not in available:
        raise HTTPException(
            status_code=404,
            detail=f"Cluster {req.cluster_id} not found. Available: {available}",
        )

    return _predict_single(req.cluster_id, req)


@app.post("/predict/batch", response_model=list[PredictResponse])
def predict_batch(req: BatchPredictRequest):
    """Predict next action from ALL cluster models (or specified subset)."""
    if model_manager is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    available = get_available_cluster_ids()
    cluster_ids = req.cluster_ids or available
    valid_ids = [cid for cid in cluster_ids if cid in available]

    if not valid_ids:
        raise HTTPException(status_code=404, detail=f"No valid clusters. Available: {available}")

    results = []
    for cid in valid_ids:
        single_req = PredictRequest(
            cluster_id=cid,
            page_state=req.page_state,
            app_name=req.app_name,
            app_url=req.app_url,
            app_description=req.app_description,
            action_history=req.action_history,
            error_context=req.error_context,
            step_number=req.step_number,
            elapsed_s=req.elapsed_s,
            max_duration_s=req.max_duration_s,
            temperature=req.temperature,
            window_size=req.window_size,
        )
        try:
            result = _predict_single(cid, single_req)
            results.append(result)
        except Exception as e:
            results.append(PredictResponse(
                cluster_id=cid,
                cluster_label=get_cluster_meta(cid).get("label", "?"),
                action="wait",
                target="",
                reasoning=f"Error: {e}",
                hesitation_ms=0,
            ))

    return results


@app.post("/switch/{cluster_id}")
def switch_cluster(cluster_id: int):
    """Pre-load a specific cluster's adapter into GPU."""
    if model_manager is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    available = get_available_cluster_ids()
    if cluster_id not in available:
        raise HTTPException(status_code=404, detail=f"Cluster {cluster_id} not found. Available: {available}")

    meta = _switch_cluster(cluster_id)
    return {
        "status": "ok",
        "cluster_id": cluster_id,
        "cluster_label": meta.get("label", "?"),
        "message": f"Adapter for cluster {cluster_id} loaded into GPU",
    }


@app.get("/clusters", response_model=list[ClusterInfo])
def list_clusters():
    """List all available clusters, their personas, and adapter status."""
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
            persona_prompt=build_cluster_system_prompt(meta)[:200] + "..." if meta.get("description") else "",
        ))

    for cid in available:
        if cid not in seen:
            result.append(ClusterInfo(id=cid, label=f"Cluster {cid}", has_adapter=True))

    return sorted(result, key=lambda x: x.id)


@app.get("/health")
def health():
    """Health check."""
    return {
        "status": "ok",
        "model_loaded": model_manager is not None,
        "current_cluster": _current_cluster_id,
        "available_clusters": get_available_cluster_ids(),
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
        workers=1,
    )

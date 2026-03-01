#!/usr/bin/env bash
# ============================================================
# Full Pipeline: Parse → Describe → Cluster → Train → Infer
# ============================================================
#
# Usage:
#   chmod +x run_pipeline.sh
#   ./run_pipeline.sh                          # run everything
#   ./run_pipeline.sh --skip-describe          # skip description generation (use cached)
#   ./run_pipeline.sh --clusters 5             # override cluster count
#   ./run_pipeline.sh --skip-train             # skip training, just cluster + build data
#   ./run_pipeline.sh --infer-only             # skip everything, just run inference
#
# Prerequisites:
#   - .env with MISTRAL_API_KEY, AGENTQL_API_KEY, TARGET_APP_DESCRIPTION, etc.
#   - sessions/ directory with session_*.json files
#   - descriptions/ directory (partially filled is OK)
#   - pip install -r requirements.txt
# ============================================================

set -euo pipefail

cd "$(dirname "$0")"

# Defaults
CLUSTERS=5
MIN_ACTIONS=3
WINDOW_SIZE=5
CONCURRENCY=5
SKIP_DESCRIBE=false
SKIP_TRAIN=false
INFER_ONLY=false
EPOCHS=5
LR="2e-4"
LORA_RANK=16
MAX_STEPS=20
INFER_URL=""
APP_DESC=""
NO_WANDB="--no-wandb"

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --clusters)       CLUSTERS="$2"; shift 2;;
        --min-actions)    MIN_ACTIONS="$2"; shift 2;;
        --window-size)    WINDOW_SIZE="$2"; shift 2;;
        --concurrency)    CONCURRENCY="$2"; shift 2;;
        --skip-describe)  SKIP_DESCRIBE=true; shift;;
        --skip-train)     SKIP_TRAIN=true; shift;;
        --infer-only)     INFER_ONLY=true; shift;;
        --epochs)         EPOCHS="$2"; shift 2;;
        --lr)             LR="$2"; shift 2;;
        --lora-rank)      LORA_RANK="$2"; shift 2;;
        --max-steps)      MAX_STEPS="$2"; shift 2;;
        --url)            INFER_URL="$2"; shift 2;;
        --app-description) APP_DESC="$2"; shift 2;;
        --wandb)          NO_WANDB=""; shift;;
        *)                echo "Unknown option: $1"; exit 1;;
    esac
done

echo "============================================================"
echo "AGENTIC WORLD — FULL PIPELINE"
echo "============================================================"
echo "  Clusters:     $CLUSTERS"
echo "  Min actions:  $MIN_ACTIONS"
echo "  Window size:  $WINDOW_SIZE"
echo "  Epochs:       $EPOCHS"
echo "  LoRA rank:    $LORA_RANK"
echo "  Max steps:    $MAX_STEPS"
echo "============================================================"

# ── STEP 0: Verify environment ──────────────────────────────

echo ""
echo "Step 0: Checking environment..."

if [ ! -f .env ]; then
    echo "ERROR: .env file not found. Copy .env.example and fill in your keys."
    exit 1
fi

SESSION_COUNT=$(ls sessions/session_*.json 2>/dev/null | wc -l | tr -d ' ')
DESC_COUNT=$(ls descriptions/description_*.txt 2>/dev/null | wc -l | tr -d ' ')
echo "  Sessions:     $SESSION_COUNT"
echo "  Descriptions: $DESC_COUNT (cached)"

if [ "$SESSION_COUNT" -eq 0 ]; then
    echo "ERROR: No session_*.json files in sessions/"
    exit 1
fi

# ── STEP 1: Parse + Describe + Embed + Cluster ──────────────

if [ "$INFER_ONLY" = false ]; then
    echo ""
    echo "============================================================"
    echo "Step 1: Parse → Describe → Embed → Cluster (K=$CLUSTERS)"
    echo "============================================================"
    echo "  (Cached descriptions will be skipped automatically)"
    echo ""

    python process_synthetic_batch.py \
        --recordings-dir sessions \
        --clusters "$CLUSTERS" \
        --min-actions "$MIN_ACTIONS" \
        --concurrency "$CONCURRENCY" \
        --skip-training-data

    echo ""
    echo "Step 1 complete. Clusters saved to data/clusters/clusters.json"

    # ── STEP 2: Build per-cluster training data ──────────────

    echo ""
    echo "============================================================"
    echo "Step 2: Build per-cluster training data"
    echo "============================================================"
    echo ""

    python build_training_data.py \
        --window-size "$WINDOW_SIZE"

    echo ""
    echo "Step 2 complete. Training data in data/training/"

    # ── STEP 3: Fine-tune per-cluster models ─────────────────

    if [ "$SKIP_TRAIN" = false ]; then
        echo ""
        echo "============================================================"
        echo "Step 3: Fine-tune $CLUSTERS LoRA adapters"
        echo "============================================================"
        echo ""

        python finetune_job.py \
            --all-clusters \
            --epochs "$EPOCHS" \
            --learning-rate "$LR" \
            --lora-rank "$LORA_RANK" \
            --skip-inference \
            $NO_WANDB

        echo ""
        echo "Step 3 complete. Adapters in data/models/cluster_*_lora/"
    else
        echo ""
        echo "Step 3: SKIPPED (--skip-train)"
    fi
fi

# ── STEP 4: Run agentic inference loop ───────────────────────

echo ""
echo "============================================================"
echo "Step 4: Agentic inference loop"
echo "============================================================"

if [ -z "$INFER_URL" ]; then
    # Try to get from .env
    INFER_URL=$(grep -E '^TARGET_APP_URL=' .env | cut -d'=' -f2- | tr -d '"' || true)
    if [ -z "$INFER_URL" ]; then
        INFER_URL=$(grep -E '^SANDBOX_URL=' .env | cut -d'=' -f2- | tr -d '"' || true)
    fi
fi

if [ -z "$INFER_URL" ]; then
    echo "WARNING: No inference URL. Set --url or TARGET_APP_URL in .env"
    echo "  To run manually:"
    echo "    python agentic_loop.py --url http://localhost:3000 --max-steps $MAX_STEPS"
else
    echo "  URL: $INFER_URL"
    echo "  Steps: $MAX_STEPS"
    echo ""

    DESC_FLAG=""
    if [ -n "$APP_DESC" ]; then
        DESC_FLAG="--app-description \"$APP_DESC\""
    fi

    python agentic_loop.py \
        --url "$INFER_URL" \
        --max-steps "$MAX_STEPS" \
        $DESC_FLAG

    echo ""
    echo "Step 4 complete. Logs in data/agent_logs/"
fi

echo ""
echo "============================================================"
echo "PIPELINE COMPLETE"
echo "============================================================"
echo ""
echo "Artifacts:"
echo "  Clusters:       data/clusters/clusters.json"
echo "  Training data:  data/training/cluster_*_{train,val}.jsonl"
echo "  Adapters:       data/models/cluster_*_lora/"
echo "  Session logs:   data/agent_logs/"
echo "============================================================"

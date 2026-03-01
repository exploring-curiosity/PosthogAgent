#!/usr/bin/env bash
# ============================================================
# serve_vllm.sh — Launch vLLM with multi-LoRA on port 8001
# ============================================================
# Auto-discovers trained LoRA adapters in data/models/ and
# serves them all via vLLM's multi-LoRA support.
#
# Usage:
#   bash serve_vllm.sh
#   bash serve_vllm.sh --port 8080
#   BASE_MODEL=mistralai/Mistral-7B-Instruct-v0.3 bash serve_vllm.sh
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="${SCRIPT_DIR}/data/models"

# Defaults (override via env vars or CLI)
BASE_MODEL="${BASE_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"
PORT="${PORT:-8001}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
MAX_LORA_RANK="${MAX_LORA_RANK:-32}"
DTYPE="${DTYPE:-bfloat16}"

# Parse CLI args
while [[ $# -gt 0 ]]; do
    case $1 in
        --port) PORT="$2"; shift 2 ;;
        --model) BASE_MODEL="$2"; shift 2 ;;
        --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
        --max-lora-rank) MAX_LORA_RANK="$2"; shift 2 ;;
        --dtype) DTYPE="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── Discover LoRA adapters ──

LORA_MODULES=""
ADAPTER_COUNT=0

if [ -d "$MODELS_DIR" ]; then
    # Prefer real_cluster_*_lora, fall back to cluster_*_lora
    FOUND_REAL=0
    for adapter_dir in "$MODELS_DIR"/real_cluster_*_lora; do
        [ -d "$adapter_dir" ] || continue
        FOUND_REAL=1
        break
    done

    if [ "$FOUND_REAL" -eq 1 ]; then
        PATTERN="real_cluster_*_lora"
        PREFIX="real_cluster_"
    else
        PATTERN="cluster_*_lora"
        PREFIX="cluster_"
    fi

    for adapter_dir in "$MODELS_DIR"/$PATTERN; do
        [ -d "$adapter_dir" ] || continue

        # Extract cluster ID from dir name
        dir_name="$(basename "$adapter_dir")"
        cluster_id="${dir_name#$PREFIX}"
        cluster_id="${cluster_id%_lora}"

        # Validate it has adapter_config.json
        if [ ! -f "$adapter_dir/adapter_config.json" ]; then
            echo "  SKIP: $dir_name (no adapter_config.json)"
            continue
        fi

        module_name="cluster${cluster_id}"
        if [ -n "$LORA_MODULES" ]; then
            LORA_MODULES="$LORA_MODULES "
        fi
        LORA_MODULES="${LORA_MODULES}${module_name}=${adapter_dir}"
        ADAPTER_COUNT=$((ADAPTER_COUNT + 1))
        echo "  Found adapter: ${module_name} → ${adapter_dir}"
    done
fi

if [ "$ADAPTER_COUNT" -eq 0 ]; then
    echo ""
    echo "ERROR: No LoRA adapters found in $MODELS_DIR"
    echo "  Expected directories like: cluster_0_lora/ or real_cluster_0_lora/"
    echo "  Each must contain adapter_config.json"
    echo ""
    echo "  Run fine-tuning first:"
    echo "    python3 finetune_job.py --all-clusters --skip-inference"
    exit 1
fi

# ── Launch vLLM ──

echo ""
echo "============================================================"
echo "  vLLM Multi-LoRA Server"
echo "============================================================"
echo "  Base model:     $BASE_MODEL"
echo "  Adapters:       $ADAPTER_COUNT"
echo "  Port:           $PORT"
echo "  Max model len:  $MAX_MODEL_LEN"
echo "  Max LoRA rank:  $MAX_LORA_RANK"
echo "  Dtype:          $DTYPE"
echo "============================================================"
echo ""
echo "  Frontend endpoint: http://0.0.0.0:${PORT}/v1/chat/completions"
echo "  Model names:       cluster0, cluster1, cluster2, ..."
echo ""
echo "  Example request:"
echo "    curl http://localhost:${PORT}/v1/chat/completions \\"
echo "      -H 'Content-Type: application/json' \\"
echo "      -d '{\"model\": \"cluster0\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello\"}]}'"
echo ""
echo "============================================================"
echo ""

exec vllm serve "$BASE_MODEL" \
    --enable-lora \
    --lora-modules $LORA_MODULES \
    --host 0.0.0.0 \
    --port "$PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-lora-rank "$MAX_LORA_RANK" \
    --dtype "$DTYPE"

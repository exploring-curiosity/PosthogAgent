"""
============================================================
Real Data Pipeline — Cluster + Build Training Data + Fine-Tune
============================================================
Uses ONLY real PostHog session recordings (export-* files).
Ignores all synthetic data.

Steps:
  1. Identify real session IDs from export-* files in data/recordings/
  2. Load their embeddings (1024-dim from Mistral Embed)
  3. Balanced KMeans into 5 clusters (enforces near-equal sizing)
  4. Label clusters using behavioral descriptions
  5. Build per-cluster training JSONL from high_level_actions
  6. Fine-tune 5 LoRA adapters (saves to data/models/real_cluster_<id>_lora/)

Usage:
    # Full pipeline: cluster + build data + fine-tune
    python real_data_pipeline.py

    # Just cluster + build data (no fine-tuning, for local machine)
    python real_data_pipeline.py --skip-finetune

    # Just fine-tune (assumes clustering + data already done)
    python real_data_pipeline.py --finetune-only

    # Override hyperparams
    python real_data_pipeline.py --epochs 5 --lora-rank 32 --batch-size 4
============================================================
"""

import os
import sys
import json
import gc
import random
import argparse
import pickle
import numpy as np
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    RECORDINGS_DIR, PARSED_DIR, DESCRIPTIONS_DIR, EMBEDDINGS_DIR,
    TRAINING_DIR, MODELS_DIR, CLUSTERS_DIR,
    TARGET_APP_NAME, TARGET_APP_URL, TARGET_APP_DESCRIPTION,
    ensure_data_dirs,
)
from build_training_data import (
    BASE_SYSTEM_PROMPT,
    build_product_context,
    build_training_examples,
)


# ============================================================
# ARGS
# ============================================================

parser = argparse.ArgumentParser(description="Real data pipeline: cluster → train data → fine-tune")
parser.add_argument("--n-clusters", type=int, default=5, help="Number of clusters")
parser.add_argument("--skip-finetune", action="store_true", help="Only cluster + build data, no fine-tuning")
parser.add_argument("--finetune-only", action="store_true", help="Skip clustering, just fine-tune existing data")
parser.add_argument("--window-size", type=int, default=5, help="Action history window for training pairs")
parser.add_argument("--val-split", type=float, default=0.1, help="Validation split ratio")

# Fine-tuning args
parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-Instruct-v0.3")
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--batch-size", type=int, default=4)
parser.add_argument("--gradient-accumulation", type=int, default=4)
parser.add_argument("--learning-rate", type=float, default=2e-4)
parser.add_argument("--lora-rank", type=int, default=32)
parser.add_argument("--lora-alpha", type=int, default=64)
parser.add_argument("--max-seq-length", type=int, default=2048)
parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
args = parser.parse_args()

ensure_data_dirs()

# Output dirs for real-only data
REAL_CLUSTERS_FILE = CLUSTERS_DIR / "real_clusters.json"
REAL_KMEANS_FILE = CLUSTERS_DIR / "real_kmeans_model.pkl"
REAL_TRAINING_PREFIX = "real_cluster"
REAL_MODEL_PREFIX = "real_cluster"


# ============================================================
# STEP 1: Identify real sessions
# ============================================================

def get_real_session_ids() -> list[str]:
    """Extract session IDs from export-* recording files."""
    ids = []
    for f in sorted(RECORDINGS_DIR.glob("export-*-ph-recording.json")):
        # export-019ca647-dfae-7dd8-bcfd-53af7ca246b1-ph-recording.json
        name = f.stem  # without .json
        sid = name.replace("export-", "").replace("-ph-recording", "")
        ids.append(sid)
    return ids


def filter_sessions_with_data(session_ids: list[str]) -> list[dict]:
    """Keep only sessions that have parsed data, descriptions, and embeddings."""
    valid = []
    for sid in session_ids:
        parsed = PARSED_DIR / f"parsed_{sid}.json"
        embedding = EMBEDDINGS_DIR / f"embedding_{sid}.json"
        description = DESCRIPTIONS_DIR / f"description_{sid}.txt"

        if not parsed.exists() or not embedding.exists():
            continue

        with open(parsed) as f:
            data = json.load(f)
        actions = data.get("high_level_actions", [])

        # Skip sessions with too few real actions
        real_actions = [a for a in actions if a.get("action") not in
                        ("API_CALL", "PAGE_LOAD", "FULL_SNAPSHOT", "PAGE_META")]
        if len(real_actions) < 3:
            continue

        with open(embedding) as f:
            emb_data = json.load(f)

        desc_text = ""
        if description.exists():
            desc_text = description.read_text()

        valid.append({
            "session_id": sid,
            "embedding": emb_data["embedding"],
            "n_actions": len(actions),
            "n_real_actions": len(real_actions),
            "duration": data.get("session_duration_s", 0),
            "description": desc_text[:500],
        })
    return valid


# ============================================================
# STEP 2: Balanced KMeans clustering
# ============================================================

def balanced_kmeans(embeddings: np.ndarray, n_clusters: int, seed: int = 42) -> np.ndarray:
    """KMeans with post-hoc balancing to enforce near-equal cluster sizes.

    1. Run standard KMeans
    2. Sort all points by distance to their assigned centroid
    3. Greedily reassign overflow points to the nearest under-filled cluster
    """
    from sklearn.cluster import KMeans

    n = len(embeddings)
    target_size = n // n_clusters
    max_size = target_size + (1 if n % n_clusters else 0)

    # Standard KMeans first
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    km.fit(embeddings)
    labels = km.labels_.copy()
    centroids = km.cluster_centers_

    # Compute distances to all centroids
    # distances[i][j] = distance from point i to centroid j
    distances = np.linalg.norm(
        embeddings[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2
    )

    # Sort points by distance to assigned centroid (farthest first → they get moved)
    assigned_dist = distances[np.arange(n), labels]
    order = np.argsort(-assigned_dist)  # farthest first

    # Greedy balanced assignment
    balanced_labels = np.full(n, -1, dtype=int)
    cluster_counts = np.zeros(n_clusters, dtype=int)

    for idx in order:
        # Try clusters in order of distance (nearest first)
        preferred = np.argsort(distances[idx])
        for c in preferred:
            if cluster_counts[c] < max_size:
                balanced_labels[idx] = c
                cluster_counts[c] += 1
                break

    return balanced_labels, km


def build_cluster_label(sessions: list[dict], cluster_id: int) -> dict:
    """Build cluster metadata from session descriptions."""
    n = len(sessions)
    durations = [s["duration"] for s in sessions]
    action_counts = [s["n_real_actions"] for s in sessions]

    avg_dur = sum(durations) / max(n, 1)
    avg_actions = sum(action_counts) / max(n, 1)

    # Extract key behaviors from descriptions
    all_desc = " ".join(s["description"] for s in sessions).lower()

    # Simple keyword-based behavior detection
    behaviors = []
    if "click" in all_desc and "sign up" in all_desc:
        behaviors.append("Attempts sign-up flow")
    if "scroll" in all_desc:
        behaviors.append("Scrolls to explore content")
    if "comment" in all_desc or "type" in all_desc:
        behaviors.append("Types and engages with forms")
    if "rage" in all_desc:
        behaviors.append("Shows frustration (rage clicks)")
    if "linear" in all_desc:
        behaviors.append("Linear navigation pattern")
    if "explor" in all_desc:
        behaviors.append("Exploratory browsing")
    if "quick" in all_desc or "rapid" in all_desc or "fast" in all_desc:
        behaviors.append("Fast-paced interaction")
    if "slow" in all_desc or "cautious" in all_desc or "deliberate" in all_desc:
        behaviors.append("Careful, deliberate actions")

    if not behaviors:
        behaviors.append("General browsing behavior")

    # Generate label based on avg behavior
    if avg_dur < 60 and avg_actions > 20:
        label = f"Active Browser {cluster_id}"
    elif avg_dur > 200:
        label = f"Deep Explorer {cluster_id}"
    elif avg_actions < 10:
        label = f"Minimal User {cluster_id}"
    else:
        label = f"Engaged User {cluster_id}"

    return {
        "id": cluster_id,
        "label": label,
        "session_ids": [s["session_id"] for s in sessions],
        "description": f"Cluster of {n} real user sessions with avg {avg_dur:.0f}s duration and {avg_actions:.0f} actions",
        "key_behaviors": behaviors[:5],
        "avg_duration_s": avg_dur,
        "avg_actions": avg_actions,
        "size": n,
    }


def run_clustering(sessions: list[dict], n_clusters: int):
    """Run balanced clustering and save results."""
    print(f"\n{'='*60}")
    print("STEP 2: BALANCED CLUSTERING")
    print(f"{'='*60}")

    embeddings = np.array([s["embedding"] for s in sessions], dtype=np.float32)
    print(f"  Embedding matrix: {embeddings.shape}")

    labels, km = balanced_kmeans(embeddings, n_clusters)

    # Group sessions by cluster
    clusters_data = {"num_clusters": n_clusters, "total_sessions": len(sessions), "data_source": "real_posthog_only", "clusters": []}

    for cid in range(n_clusters):
        cluster_sessions = [sessions[i] for i in range(len(sessions)) if labels[i] == cid]
        meta = build_cluster_label(cluster_sessions, cid)
        clusters_data["clusters"].append(meta)
        print(f"  Cluster {cid}: {meta['label']} — {meta['size']} sessions, "
              f"avg {meta['avg_duration_s']:.0f}s, {meta['avg_actions']:.0f} actions")

    # Save
    with open(REAL_CLUSTERS_FILE, "w") as f:
        json.dump(clusters_data, f, indent=2)
    with open(REAL_KMEANS_FILE, "wb") as f:
        pickle.dump(km, f)

    print(f"\n  Saved: {REAL_CLUSTERS_FILE}")
    print(f"  Saved: {REAL_KMEANS_FILE}")
    return clusters_data


# ============================================================
# STEP 3: Build training data
# ============================================================

def build_real_training_data(clusters_data: dict, window_size: int, val_split: float):
    """Build per-cluster training JSONL from real session data."""
    print(f"\n{'='*60}")
    print("STEP 3: BUILD TRAINING DATA")
    print(f"{'='*60}")

    app_name = TARGET_APP_NAME or "FunCity"
    app_url = TARGET_APP_URL or "https://fun-city-xi.vercel.app"
    app_description = TARGET_APP_DESCRIPTION or "A social platform for discovering and sharing experiences in New York City"
    product_context = build_product_context(app_name, app_url, app_description)

    print(f"  App: {app_name}")
    print(f"  URL: {app_url}")

    total_train = 0
    total_val = 0

    for cluster in clusters_data["clusters"]:
        cid = cluster["id"]
        label = cluster["label"]
        session_ids = cluster["session_ids"]

        print(f"\n  Cluster {cid}: {label} ({len(session_ids)} sessions)")

        all_examples = []
        for sid in session_ids:
            parsed_path = PARSED_DIR / f"parsed_{sid}.json"
            if not parsed_path.exists():
                continue

            with open(parsed_path) as f:
                parsed = json.load(f)

            examples = build_training_examples(
                parsed, cluster, product_context, window_size=window_size
            )
            all_examples.extend(examples)

        if not all_examples:
            print(f"    WARNING: No training examples")
            continue

        # Shuffle and split
        random.seed(42 + cid)
        random.shuffle(all_examples)

        val_count = max(1, int(len(all_examples) * val_split))
        val_examples = all_examples[:val_count]
        train_examples = all_examples[val_count:]

        # Save with real_ prefix
        train_path = TRAINING_DIR / f"{REAL_TRAINING_PREFIX}_{cid}_train.jsonl"
        val_path = TRAINING_DIR / f"{REAL_TRAINING_PREFIX}_{cid}_val.jsonl"

        with open(train_path, "w") as f:
            for ex in train_examples:
                f.write(json.dumps(ex) + "\n")
        with open(val_path, "w") as f:
            for ex in val_examples:
                f.write(json.dumps(ex) + "\n")

        print(f"    Train: {len(train_examples)} examples → {train_path.name}")
        print(f"    Val:   {len(val_examples)} examples → {val_path.name}")

        total_train += len(train_examples)
        total_val += len(val_examples)

    print(f"\n  Total: {total_train} train + {total_val} val examples")
    return total_train, total_val


# ============================================================
# STEP 4: Fine-tune LoRA adapters
# ============================================================

def finetune_all_clusters(clusters_data: dict):
    """Fine-tune one LoRA adapter per cluster."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    from trl import SFTTrainer, SFTConfig
    from datasets import load_dataset, Dataset as HFDataset

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    USE_WANDB = not args.no_wandb
    if USE_WANDB:
        import wandb

    print(f"\n{'='*60}")
    print("STEP 4: FINE-TUNE LORA ADAPTERS")
    print(f"{'='*60}")
    print(f"  Model:  {args.model}")
    print(f"  LoRA:   rank={args.lora_rank}, alpha={args.lora_alpha}")
    print(f"  Epochs: {args.epochs}, LR: {args.learning_rate}")
    print(f"  Batch:  {args.batch_size} x {args.gradient_accumulation} = {args.batch_size * args.gradient_accumulation}")

    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
        print(f"  Attn:   Flash Attention 2")
    except ImportError:
        attn_impl = "sdpa"
        print(f"  Attn:   SDPA")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    def to_text(messages):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    results = []

    for cluster in clusters_data["clusters"]:
        cid = cluster["id"]
        label = cluster["label"]

        train_file = str(TRAINING_DIR / f"{REAL_TRAINING_PREFIX}_{cid}_train.jsonl")
        val_file = str(TRAINING_DIR / f"{REAL_TRAINING_PREFIX}_{cid}_val.jsonl")
        output_dir = str(MODELS_DIR / f"{REAL_MODEL_PREFIX}_{cid}_lora")

        if not Path(train_file).exists() or not Path(val_file).exists():
            print(f"\n  SKIP cluster {cid}: training data not found")
            results.append({"cluster_id": cid, "status": "skipped"})
            continue

        print(f"\n  {'='*50}")
        print(f"  TRAINING CLUSTER {cid}: {label}")
        print(f"  {'='*50}")

        # W&B
        run = None
        if USE_WANDB:
            run = wandb.init(
                project=os.environ.get("WANDB_PROJECT", "agentic-world"),
                job_type="finetune",
                name=f"real_cluster_{cid}_{label.replace(' ', '_')}",
                config={
                    "cluster_id": cid,
                    "cluster_label": label,
                    "model": args.model,
                    "lora_rank": args.lora_rank,
                    "data_source": "real_posthog_only",
                },
                tags=["hackathon", "mistral-worldwide", "w&b-finetuning-track", "real-data", "lora"],
                reinit=True,
            )

        # Fresh base model
        print(f"    Loading base model (bf16)...")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_impl,
            trust_remote_code=True,
        )
        base_model.enable_input_require_grads()

        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base_model, lora_config)
        model.print_trainable_parameters()

        # Load data
        dataset = load_dataset("json", data_files={"train": train_file, "eval": val_file})
        print(f"    Train: {len(dataset['train'])} examples, Val: {len(dataset['eval'])} examples")

        train_texts = [to_text(ex["messages"]) for ex in dataset["train"]]
        eval_texts = [to_text(ex["messages"]) for ex in dataset["eval"]]
        train_dataset = HFDataset.from_dict({"text": train_texts})
        eval_dataset = HFDataset.from_dict({"text": eval_texts})

        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=SFTConfig(
                output_dir=output_dir,
                per_device_train_batch_size=args.batch_size,
                gradient_accumulation_steps=args.gradient_accumulation,
                num_train_epochs=args.epochs,
                learning_rate=args.learning_rate,
                warmup_ratio=0.03,
                weight_decay=0.01,
                lr_scheduler_type="cosine",
                seed=42,
                bf16=True,
                tf32=True,
                logging_steps=5,
                eval_strategy="epoch",
                save_strategy="epoch",
                save_total_limit=1,
                load_best_model_at_end=False,
                report_to="wandb" if USE_WANDB else "none",
                run_name=f"real_cluster_{cid}" if USE_WANDB else None,
                max_length=args.max_seq_length,
                dataset_text_field="text",
                packing=True,
                dataset_num_proc=4,
                dataloader_num_workers=4,
                dataloader_pin_memory=True,
                gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False},
                optim="adamw_torch_fused",
            ),
        )

        gpu_stats = torch.cuda.get_device_properties(0)
        print(f"    GPU: {gpu_stats.name} ({gpu_stats.total_memory / 1024**3:.1f} GB)")

        train_result = trainer.train()
        peak_memory = torch.cuda.max_memory_reserved() / 1024**3
        eval_results = trainer.evaluate()

        print(f"    Train loss: {train_result.training_loss:.4f}")
        print(f"    Eval loss:  {eval_results['eval_loss']:.4f}")
        print(f"    Time:       {train_result.metrics['train_runtime']:.1f}s")
        print(f"    Peak VRAM:  {peak_memory:.1f} GB")

        # Save adapter
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        config_path = os.path.join(output_dir, "training_config.json")
        with open(config_path, "w") as f:
            json.dump({
                "cluster_id": cid,
                "cluster_label": label,
                "base_model": args.model,
                "data_source": "real_posthog_only",
                "lora_rank": args.lora_rank,
                "lora_alpha": args.lora_alpha,
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "train_examples": len(train_dataset),
                "eval_examples": len(eval_dataset),
                "final_train_loss": train_result.training_loss,
                "final_eval_loss": eval_results["eval_loss"],
                "training_time_s": train_result.metrics["train_runtime"],
                "peak_vram_gb": peak_memory,
            }, f, indent=2)

        print(f"    Adapter saved to {output_dir}")

        if USE_WANDB and run:
            wandb.log({
                "final_train_loss": train_result.training_loss,
                "final_eval_loss": eval_results["eval_loss"],
                "training_time_s": train_result.metrics["train_runtime"],
                "peak_vram_gb": peak_memory,
            })
            wandb.summary["status"] = "success"
            wandb.finish()

        results.append({
            "cluster_id": cid,
            "label": label,
            "status": "completed",
            "train_loss": train_result.training_loss,
            "eval_loss": eval_results["eval_loss"],
            "train_examples": len(train_dataset),
            "time_s": train_result.metrics["train_runtime"],
            "adapter_path": output_dir,
        })

        # Cleanup
        del model, base_model, trainer, train_dataset, eval_dataset
        gc.collect()
        torch.cuda.empty_cache()

    # Summary
    summary_path = MODELS_DIR / "real_training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("FINE-TUNING COMPLETE")
    print(f"{'='*60}")
    for r in results:
        if r["status"] == "skipped":
            print(f"  Cluster {r['cluster_id']}: SKIPPED")
        else:
            print(f"  Cluster {r['cluster_id']} ({r['label']}): "
                  f"loss={r['train_loss']:.4f}/{r['eval_loss']:.4f}, "
                  f"{r['train_examples']} examples, {r['time_s']:.0f}s")
    print(f"  Summary: {summary_path}")

    return results


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("REAL DATA PIPELINE")
    print("=" * 60)

    # ── Step 1: Identify real sessions ──
    if not args.finetune_only:
        print(f"\n{'='*60}")
        print("STEP 1: IDENTIFY REAL SESSIONS")
        print(f"{'='*60}")

        all_ids = get_real_session_ids()
        print(f"  Found {len(all_ids)} export-* recordings")

        sessions = filter_sessions_with_data(all_ids)
        print(f"  Valid sessions (parsed + embedding + ≥3 actions): {len(sessions)}")

        if len(sessions) < args.n_clusters:
            print(f"  ERROR: Need at least {args.n_clusters} valid sessions, got {len(sessions)}")
            sys.exit(1)

        # Show action stats
        total_actions = sum(s["n_real_actions"] for s in sessions)
        print(f"  Total real actions: {total_actions}")
        print(f"  Avg actions/session: {total_actions / len(sessions):.0f}")

        # ── Step 2: Cluster ──
        clusters_data = run_clustering(sessions, args.n_clusters)

        # ── Step 3: Build training data ──
        total_train, total_val = build_real_training_data(
            clusters_data, args.window_size, args.val_split
        )

        if total_train == 0:
            print("\nERROR: No training examples generated. Check parsed data.")
            sys.exit(1)

    else:
        # Load existing clustering
        if not REAL_CLUSTERS_FILE.exists():
            print(f"ERROR: {REAL_CLUSTERS_FILE} not found. Run without --finetune-only first.")
            sys.exit(1)
        with open(REAL_CLUSTERS_FILE) as f:
            clusters_data = json.load(f)
        print(f"  Loaded existing clustering: {len(clusters_data['clusters'])} clusters")

    # ── Step 4: Fine-tune ──
    if not args.skip_finetune:
        finetune_all_clusters(clusters_data)
    else:
        print(f"\n  Skipping fine-tuning (--skip-finetune)")
        print(f"  Training data ready in {TRAINING_DIR}")
        print(f"  To fine-tune on VM:")
        print(f"    python real_data_pipeline.py --finetune-only")

    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"  Clusters:      {REAL_CLUSTERS_FILE}")
    print(f"  Training data: {TRAINING_DIR}/{REAL_TRAINING_PREFIX}_*")
    print(f"  Models:        {MODELS_DIR}/{REAL_MODEL_PREFIX}_*_lora/")
    print(f"\n  To serve: update service.py to use real_clusters.json + real_cluster_*_lora models")

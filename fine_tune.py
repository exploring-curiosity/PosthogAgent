"""
Fine-tune 3 Mistral models (one per demographic cluster) using the Mistral API.
Tracks experiments with Weights & Biases.

Usage:
    python fine_tune.py
    python fine_tune.py --base-model open-mistral-nemo
"""

import json
import sys
import time
import argparse
from pathlib import Path

from mistralai import Mistral

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    MISTRAL_API_KEY,
    WANDB_API_KEY,
    TRAINING_DIR,
    MODELS_DIR,
    CLUSTERS_DIR,
    ensure_data_dirs,
)

WANDB_AVAILABLE = False
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    pass


def init_wandb(cluster_id: int, cluster_label: str, base_model: str):
    """Initialize W&B run for a fine-tuning job. Returns None if W&B not configured."""
    if not WANDB_API_KEY:
        print("  W&B: No API key, logging to console only")
        return None

    try:
        import wandb
        wandb.login(key=WANDB_API_KEY)
        run = wandb.init(
            project="agentic-world",
            name=f"finetune-cluster-{cluster_id}-{cluster_label.lower().replace(' ', '-')}",
            config={
                "cluster_id": cluster_id,
                "cluster_label": cluster_label,
                "base_model": base_model,
                "method": "Mistral Fine-Tuning API",
                "num_clusters": 3,
                "training_data_source": "PostHog session recordings",
            },
            tags=["hackathon", "mistral-worldwide", "w&b-finetuning-track", "behavioral-finetuning"],
        )
        return run
    except Exception as e:
        print(f"  W&B init failed: {e}")
        return None


def upload_training_file(client: Mistral, file_path: Path) -> str:
    """Upload a JSONL file to Mistral and return the file ID."""
    with open(file_path, "rb") as f:
        uploaded = client.files.upload(
            file={
                "file_name": file_path.name,
                "content": f.read(),
            },
            purpose="fine-tune",
        )
    return uploaded.id


def create_fine_tuning_job(client: Mistral, model: str, train_file_id: str,
                           val_file_id: str, suffix: str) -> str:
    """Create a fine-tuning job and return the job ID."""
    job = client.fine_tuning.jobs.create(
        model=model,
        training_files=[{"file_id": train_file_id, "weight": 1}],
        validation_files=[val_file_id],
        suffix=suffix,
        hyperparameters={
            "training_steps": 100,
            "learning_rate": 1e-5,
        },
    )
    return job.id


def poll_job(client: Mistral, job_id: str, wandb_run=None, poll_interval: int = 30) -> dict:
    """Poll a fine-tuning job until completion. Log metrics to W&B."""
    print(f"  Polling job {job_id}...")
    last_status = None

    while True:
        job = client.fine_tuning.jobs.get(job_id=job_id)
        status = job.status

        if status != last_status:
            print(f"  Status: {status}")
            last_status = status

        # Log to W&B if available
        if wandb_run and hasattr(job, "metrics") and job.metrics:
            try:
                import wandb
                metrics = {}
                if hasattr(job.metrics, "train_loss"):
                    metrics["train_loss"] = job.metrics.train_loss
                if hasattr(job.metrics, "val_loss"):
                    metrics["val_loss"] = job.metrics.val_loss
                if hasattr(job.metrics, "train_steps"):
                    metrics["train_steps"] = job.metrics.train_steps
                if metrics:
                    wandb.log(metrics)
            except Exception:
                pass

        if status in ("SUCCESS", "SUCCEEDED"):
            return {
                "status": "success",
                "model_id": job.fine_tuned_model,
                "job_id": job_id,
            }
        elif status in ("FAILED", "CANCELLED", "CANCELED"):
            error_msg = ""
            if hasattr(job, "error") and job.error:
                error_msg = str(job.error)
            return {
                "status": "failed",
                "error": error_msg or f"Job {status}",
                "job_id": job_id,
            }

        time.sleep(poll_interval)


def fine_tune_cluster(client: Mistral, cluster: dict, base_model: str) -> dict:
    """Fine-tune a model for one cluster. Returns model info."""
    cluster_id = cluster["id"]
    label = cluster.get("label", f"Cluster {cluster_id}")

    train_path = TRAINING_DIR / f"cluster_{cluster_id}_train.jsonl"
    val_path = TRAINING_DIR / f"cluster_{cluster_id}_val.jsonl"

    if not train_path.exists():
        return {"status": "skipped", "error": f"Training file not found: {train_path}"}

    # Count examples
    train_count = sum(1 for _ in open(train_path))
    val_count = sum(1 for _ in open(val_path)) if val_path.exists() else 0

    print(f"\n  Training: {train_count} examples, Validation: {val_count} examples")

    if train_count < 10:
        return {"status": "skipped", "error": f"Too few training examples ({train_count})"}

    # Init W&B
    wandb_run = init_wandb(cluster_id, label, base_model)

    try:
        # Upload files
        print(f"  Uploading training data...")
        train_file_id = upload_training_file(client, train_path)
        print(f"    Train file: {train_file_id}")

        val_file_id = None
        if val_path.exists() and val_count > 0:
            val_file_id = upload_training_file(client, val_path)
            print(f"    Val file: {val_file_id}")

        # Log to W&B
        if wandb_run:
            import wandb
            wandb.config.update({
                "train_examples": train_count,
                "val_examples": val_count,
                "train_file_id": train_file_id,
            })

        # Create fine-tuning job
        suffix = f"agentic-world-c{cluster_id}"
        print(f"  Creating fine-tuning job (base: {base_model}, suffix: {suffix})...")
        job_id = create_fine_tuning_job(
            client, base_model, train_file_id, val_file_id, suffix
        )
        print(f"  Job ID: {job_id}")

        # Poll until done
        result = poll_job(client, job_id, wandb_run)

        if result["status"] == "success":
            print(f"  Fine-tuning SUCCEEDED: {result['model_id']}")
            if wandb_run:
                import wandb
                wandb.summary["model_id"] = result["model_id"]
                wandb.summary["status"] = "success"
        else:
            print(f"  Fine-tuning FAILED: {result.get('error', 'unknown')}")
            if wandb_run:
                import wandb
                wandb.summary["status"] = "failed"
                wandb.summary["error"] = result.get("error", "")

        result["label"] = label
        result["training_examples"] = train_count
        result["validation_examples"] = val_count
        return result

    except Exception as e:
        print(f"  ERROR: {e}")
        if wandb_run:
            import wandb
            wandb.summary["status"] = "error"
            wandb.summary["error"] = str(e)
        return {"status": "error", "error": str(e), "label": label}

    finally:
        if wandb_run:
            import wandb
            wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Mistral models per demographic")
    parser.add_argument("--base-model", type=str, default="open-mistral-nemo",
                        help="Base model for fine-tuning")
    parser.add_argument("--poll-interval", type=int, default=30,
                        help="Seconds between status polls")
    args = parser.parse_args()

    if not MISTRAL_API_KEY:
        print("ERROR: MISTRAL_API_KEY not set in .env")
        sys.exit(1)

    ensure_data_dirs()

    # Load clusters
    clusters_path = CLUSTERS_DIR / "clusters.json"
    if not clusters_path.exists():
        print("ERROR: clusters.json not found. Run cluster_demographics.py first.")
        sys.exit(1)

    with open(clusters_path) as f:
        clusters_data = json.load(f)

    client = Mistral(api_key=MISTRAL_API_KEY)

    print("=" * 60)
    print("FINE-TUNING DEMOGRAPHIC MODELS")
    print(f"Base model: {args.base_model}")
    print(f"Clusters: {clusters_data['num_clusters']}")
    print("=" * 60)

    models = {}

    for cluster in clusters_data["clusters"]:
        cluster_id = cluster["id"]
        label = cluster.get("label", f"Cluster {cluster_id}")

        print(f"\n{'='*50}")
        print(f"Cluster {cluster_id}: {label}")
        print(f"{'='*50}")

        result = fine_tune_cluster(client, cluster, args.base_model)
        models[f"cluster_{cluster_id}"] = result

    # Save model registry
    models_path = MODELS_DIR / "models.json"
    with open(models_path, "w") as f:
        json.dump(models, f, indent=2)
    print(f"\nSaved model registry to {models_path}")

    # ── Log W&B Artifacts (training data + model registry) ──
    if WANDB_API_KEY and WANDB_AVAILABLE:
        try:
            wandb.login(key=WANDB_API_KEY)
            artifact_run = wandb.init(
                project="agentic-world",
                name="artifact-upload",
                job_type="artifact-logging",
                tags=["hackathon", "mistral-worldwide", "w&b-finetuning-track"],
            )

            # Log training data as artifact
            data_artifact = wandb.Artifact(
                "behavioral-training-data", type="dataset",
                description="PostHog-derived behavioral session data, clustered into 3 demographics",
            )
            for jsonl_file in TRAINING_DIR.glob("*.jsonl"):
                data_artifact.add_file(str(jsonl_file))
            wandb.log_artifact(data_artifact)
            print("  W&B: Logged training data artifact")

            # Log model registry as artifact
            model_ids = [
                info.get("model_id") for info in models.values()
                if info.get("status") == "success" and info.get("model_id")
            ]
            model_artifact = wandb.Artifact(
                "mistral-nemo-behavioral-ft", type="model",
                description="3 demographic-specific Mistral models fine-tuned on real user behavior",
                metadata={
                    "base_model": args.base_model,
                    "fine_tuned_model_ids": model_ids,
                    "num_clusters": len(clusters_data["clusters"]),
                },
            )
            model_artifact.add_file(str(models_path))
            wandb.log_artifact(model_artifact)
            print("  W&B: Logged model registry artifact")

            wandb.finish()
        except Exception as e:
            print(f"  W&B artifact logging failed: {e}")

    # Summary
    print(f"\n{'='*60}")
    print("FINE-TUNING COMPLETE")
    print(f"{'='*60}")
    for key, info in models.items():
        status = info.get("status", "?")
        label = info.get("label", key)
        model_id = info.get("model_id", "N/A")
        examples = info.get("training_examples", "?")
        print(f"  {label}: {status} | model={model_id} | examples={examples}")


if __name__ == "__main__":
    main()

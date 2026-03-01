# Shadow Verse
## Agentic World — Behavioral Digital Twins from Real User Data

Train AI agents on real user behavior from PostHog session recordings — not LLM persona prompts — to create persistent digital twins of user demographics. These agents learn how different people actually interact with web products, then autonomously explore and evaluate any web application from their demographic's perspective.

**Hugging Face Artifacts:** [amaan784/agentic-world-behavioral](https://huggingface.co/amaan784/agentic-world-behavioral)

**Weights and Biases Finetuning Report:** https://api.wandb.ai/links/amaan784-columbia-university/5kjdhb4e

**Youtube demo link:** https://youtu.be/XkuUeHlwi5k

**Google drive link:** https://drive.google.com/drive/folders/1jrB0kd2uz1yg_ZYN0xU1qZ-EqIuYTakW?usp=drive_link

## Core Differentiator

Every competitor describes personas with prompts. **We clone them from real behavioral data.**

Agents learn scrolling speed, click patterns, hesitation moments, feature discovery order, and engagement style from actual PostHog recordings. They are then fine-tuned into demographic-specific models that can explore unfamiliar applications the way real users from that demographic would.

## Architecture

```
                         ┌─────────────────────────────────────────────────────────┐
                         │                   DATA PIPELINE                         │
                         │                                                         │
  PostHog Recordings ──► │  Parse ──► Describe ──► Encode ──► Cluster ──► Train    │
       (15+ users)       │  (S1)      (S2)         (S3)       K-Means     JSONL    │
                         │  Python    Mistral      Mistral    scikit-     Training  │
                         │            Medium       Embed      learn       Data      │
                         └──────────────────────────────────────┬──────────────────┘
                                                                │
                         ┌──────────────────────────────────────▼──────────────────┐
                         │                   FINE-TUNING                            │
                         │                                                         │
                         │  Option A: Mistral API (cloud) ── W&B tracking          │
                         │  Option B: Local LoRA on Mistral-7B (A100 GPU)          │
                         │  Option C: vLLM multi-LoRA serving                      │
                         └──────────────────────────────────────┬──────────────────┘
                                                                │
                         ┌──────────────────────────────────────▼──────────────────┐
                         │                  AGENT EXECUTION                         │
                         │                                                         │
                         │  Autonomous exploration via AgentQL + Playwright         │
                         │  Per-demographic decision model ── Weave tracing        │
                         │  Stuck detection ── Session logging                     │
                         └──────────────────────────────────────┬──────────────────┘
                                                                │
                         ┌──────────────────────────────────────▼──────────────────┐
                         │              EVALUATION & REPORTING                     │
                         │                                                         │
                         │  Quantitative metrics ── Qualitative LLM feedback       │
                         │  Weave evaluations ── Next.js comparative dashboard     │
                         └─────────────────────────────────────────────────────────┘
```

## How It Works

1. **Download** — Bulk-fetch all session recordings from PostHog API
2. **Parse** — Extract structured events (clicks, scrolls, inputs, API calls) from rrweb data
3. **Describe** — Mistral Medium generates natural language behavioral profiles
4. **Encode** — Mistral Embed creates 768-D vector embeddings for each session
5. **Cluster** — K-Means groups sessions into 3 natural demographic archetypes
6. **Train** — Convert action sequences into supervised (state -> next_action) training pairs
7. **Fine-Tune** — Create 3 demographic-specific Mistral models (cloud API or local LoRA)
8. **Explore** — Each agent autonomously explores the target app using its fine-tuned decision model
9. **Report** — Comparative dashboard + per-demographic narrative reports with Weave evaluations

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt
playwright install chromium

# 2. Configure environment
cp .env.example .env
# Fill in: MISTRAL_API_KEY, AGENTQL_API_KEY, POSTHOG_PERSONAL_API_KEY,
#          POSTHOG_PROJECT_ID, WANDB_API_KEY, TARGET_APP_URL

# 3. Download all recordings from PostHog
python download_recordings.py

# 4. Process recordings and cluster into 3 demographics
python cluster_demographics.py

# 5. Build training data from clustered sessions
python build_training_data.py

# 6. Fine-tune 3 Mistral models (one per demographic)
#    Option A: Mistral API fine-tuning (cloud)
python fine_tune.py
#    Option B: Local LoRA fine-tuning (requires GPU, e.g. A100)
python finetune_job.py --all-clusters --skip-inference

# 7. Run all 3 agents against the target app
python run_agents.py

# 8. View results in the comparative dashboard
cd visualizer && npm install && npx next dev -p 3333
```

### Linux / GPU Server Setup

If running on a remote GPU server (e.g. Shadeform, Lambda, Brev):

```bash
# Create a virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
playwright install chromium

# Run local LoRA fine-tuning
python3 finetune_job.py --all-clusters --skip-inference
```

> **Note:** Always use `python3 -m pip install` (not bare `pip install`) to ensure packages install into the correct interpreter.

## Prerequisites

- **Python 3.10+**
- **Mistral API key** — [console.mistral.ai](https://console.mistral.ai/) (for description, embeddings, fine-tuning, and reports)
- **AgentQL API key** — [agentql.com](https://www.agentql.com/) (for semantic DOM querying)
- **PostHog Personal API key** — [posthog.com](https://posthog.com/) (for downloading recordings)
- **W&B API key** — [wandb.ai](https://wandb.ai/) (for fine-tuning experiment tracking)
- **Target web app** — Any web application the agents should explore

## Safety

The agent **refuses to run against production URLs** listed in `BLOCKED_HOSTS`. The `config.py` module validates the target URL at startup and exits immediately if it matches any blocked hostname.

## Project Structure

```
ShadowVerse/
├── config.py                       # Env vars, URL validation, data paths
├── .env.example                    # Template for all API keys and config
├── requirements.txt                # Python dependencies
│
├── CORE PIPELINE
├── download_recordings.py          # Fetch all recordings from PostHog API
├── cluster_demographics.py         # Process recordings + K-Means clustering
├── build_training_data.py          # Generate fine-tuning JSONL from clusters
├── fine_tune.py                    # Mistral API fine-tuning + W&B tracking
├── finetune_job.py                 # Local LoRA fine-tuning (A100 optimized)
├── run_agents.py                   # Launch 3 demographic agents
├── run_pipeline.py                 # End-to-end single-session pipeline
│
├── AGENT EXECUTION
├── agent_runner.py                 # Reusable agentic loop engine
├── agentic_loop.py                 # Local model agentic loop (LoRA)
├── local_client.py                 # Browser client for remote GPU service
├── service.py                      # Unified GPU inference + online pipeline (FastAPI)
├── serve_vllm.sh                   # vLLM multi-LoRA server launcher
│
├── EVALUATION & BATCH
├── evaluation.py                   # Weave evaluation scorers
├── process_synthetic_batch.py      # Batch-process synthetic sessions
├── real_data_pipeline.py           # Real-session-only processing pipeline
├── generate_all_policies.py        # Batch-generate behavioral policies
│
├── pipeline/                       # Core processing stages
│   ├── stage1_parse.py             # PostHog JSON → structured events
│   ├── stage2_describe.py          # Events → behavioral narrative (Mistral)
│   ├── stage3_encode.py            # Narrative → vector embedding (Mistral Embed)
│   ├── stage4_policy.py            # Description → agent policy JSON
│   ├── stage5_execute.py           # Legacy: fixed-sequence agent
│   └── stage5_explore.py           # Autonomous exploratory agent
│
├── feedback/                       # UX metrics and report generation
│   ├── session_logger.py           # Records agent actions + stuck detection
│   ├── metrics.py                  # Quantitative agent vs real-user metrics
│   └── stage6_report.py            # Qualitative + quantitative UX report
│
├── online_pipeline/                # FastAPI server for continuous processing
│   ├── server.py                   # Webhook receiver + REST API
│   ├── poller.py                   # Background PostHog polling
│   ├── processor.py                # Single-recording pipeline processor
│   ├── store.py                    # JSON-backed state store
│   └── retrain.py                  # Auto-retrain when new data arrives
│
├── visualizer/                     # Next.js comparative dashboard
│   ├── app/
│   │   ├── page.tsx                # Main dashboard page
│   │   ├── components/
│   │   │   ├── ComparativeDashboard.tsx   # Multi-agent comparison view
│   │   │   ├── ActionTimeline.tsx         # Agent action timeline
│   │   │   ├── FrictionMap.tsx            # UX friction visualization
│   │   │   ├── HeroStats.tsx              # Key metric cards
│   │   │   ├── TimingComparison.tsx       # Timing analysis charts
│   │   │   ├── QualitativeReport.tsx      # Narrative report display
│   │   │   ├── PipelineRunner.tsx         # Pipeline control UI
│   │   │   └── ResultsView.tsx            # Per-agent results
│   │   ├── api/                           # Next.js API routes
│   │   └── data/                          # Sample data for development
│   └── package.json                # React 19, Next.js 16, Recharts, Tailwind 4
│
└── data/                           # All generated artifacts
    ├── recordings/                 # Raw PostHog exports
    ├── parsed/                     # Structured event sequences
    ├── descriptions/               # Behavioral narratives
    ├── embeddings/                 # Vector embeddings
    ├── clusters/                   # Demographic cluster assignments
    ├── training/                   # Fine-tuning JSONL data
    ├── models/                     # Fine-tuned model registry + LoRA adapters
    ├── agent_logs/                 # Exploration session logs
    └── reports/                    # Comparative + per-agent reports
```

## Running Individual Steps

```bash
# Download recordings (skip already downloaded)
python download_recordings.py --min-duration 30

# Cluster with custom count
python cluster_demographics.py --clusters 3

# Build training data with larger context window
python build_training_data.py --window-size 7

# Fine-tune via Mistral API
python fine_tune.py --base-model open-mistral-nemo

# Fine-tune locally with LoRA (GPU required)
python finetune_job.py --all-clusters                    # Train all clusters
python finetune_job.py --cluster 0 --epochs 3            # Train single cluster
python finetune_job.py --all-clusters --lora-rank 64     # Higher LoRA rank
python finetune_job.py --all-clusters --no-wandb         # Disable W&B tracking

# Run only one demographic
python run_agents.py --cluster 0 --max-steps 20 --max-duration 120

# Run the agentic loop with local LoRA model
python agentic_loop.py --url http://localhost:3000 --app-description "..."

# Run end-to-end pipeline on a single recording
python run_pipeline.py <recording.json>

# Process synthetic sessions (flat-event JSON format)
python process_synthetic_batch.py --clusters 3 --concurrency 5

# Generate policies for all described sessions
python generate_all_policies.py

# Run evaluation scorers
python evaluation.py
```

## Fine-Tuning Options

| Approach | Script | Where | Requirements |
|---|---|---|---|
| **Mistral API** | `fine_tune.py` | Cloud (Mistral servers) | `MISTRAL_API_KEY` |
| **Local LoRA** | `finetune_job.py` | Your GPU server | A100 80GB recommended, PyTorch, PEFT, TRL |
| **vLLM Serving** | `serve_vllm.sh` | Your GPU server | vLLM, trained LoRA adapters |

The local LoRA approach fine-tunes **Mistral-7B-Instruct-v0.3** with per-cluster LoRA adapters. It supports Flash Attention 2, fused AdamW, gradient checkpointing, and packing for efficient A100 training.

### Experiment Tracking with Weights & Biases

All fine-tuning runs are tracked in **Weights & Biases** for full experiment observability. Both `fine_tune.py` (cloud) and `finetune_job.py` (local LoRA) integrate with W&B:

- **Real-time metrics** — Training loss, eval loss, and learning rate logged per step
- **Run config** — Cluster ID, demographic label, base model, hyperparameters (LoRA rank, epochs, batch size)
- **Artifacts** — Training JSONL data and trained LoRA adapters are logged as versioned W&B Artifacts
- **Model registry** — Fine-tuned model IDs (cloud) or adapter paths (local) stored in run summary
- **Run tags** — Runs tagged with `behavioral-finetuning`, `lora`, and cluster identifiers for filtering

Each cluster gets its own W&B run under the `agentic-world` project, making it easy to compare training dynamics across demographics. Disable tracking with `--no-wandb` if needed.

### Hugging Face Artifacts

Trained LoRA adapters, training data, and model artifacts are uploaded to Hugging Face for reproducibility and sharing. The published repository is available at [amaan784/agentic-world-behavioral](https://huggingface.co/amaan784/agentic-world-behavioral).

### vLLM Multi-LoRA Serving

After local fine-tuning, serve all cluster adapters via vLLM:

```bash
bash serve_vllm.sh                        # Default: port 8001
bash serve_vllm.sh --port 8080            # Custom port
BASE_MODEL=mistralai/Mistral-7B-Instruct-v0.3 bash serve_vllm.sh
```

The script auto-discovers trained LoRA adapters in `data/models/` and serves them as named models (`cluster0`, `cluster1`, `cluster2`, ...) via the OpenAI-compatible `/v1/chat/completions` endpoint.

## GPU Inference Service

The `service.py` module is a unified FastAPI server that combines GPU inference with the online data pipeline. It runs on a VM with a GPU and exposes:

**Inference Endpoints:**
| Endpoint | Method | Description |
|---|---|---|
| `/predict` | POST | Predict next action for a cluster + page state |
| `/predict/batch` | POST | Predict for all clusters at once |
| `/clusters` | GET | List clusters and their personas |
| `/switch/{id}` | POST | Pre-load a specific cluster adapter |
| `/health` | GET | Health check |

**Online Pipeline Endpoints:**
| Endpoint | Method | Description |
|---|---|---|
| `/webhook/posthog` | POST | Receive PostHog webhook events |
| `/process/{id}` | POST | Manually trigger recording processing |
| `/process/{id}/status` | GET | Check processing status |
| `/pipeline/status` | GET | Pipeline stats and cluster counts |
| `/retrain/check` | GET | Check if retraining is needed |
| `/retrain/trigger` | POST | Trigger async retraining |
| `/retrain/status` | GET | Retrain job status |
| `/reload` | POST | Hot-reload adapters after retraining |
| `/poller/start` | POST | Start background PostHog polling |
| `/poller/stop` | POST | Stop poller |
| `/poller/status` | GET | Poller status |

```bash
# Start the unified service
python service.py

# With ngrok tunnel and API key
NGROK_AUTH_TOKEN=<token> SERVICE_API_KEY=mysecret python service.py
```

Use `local_client.py` to drive browser automation locally against the remote GPU service, keeping the browser on your machine and inference on the GPU.

## Online Pipeline

The `online_pipeline/` module provides a standalone FastAPI server for continuous, webhook-driven processing (alternative to the unified `service.py`):

```bash
uvicorn online_pipeline.server:app --port 8100 --reload
```

- **`POST /webhook/posthog`** — Receive PostHog webhook events for new recordings
- **`POST /process/{recording_id}`** — Manually trigger processing
- **`GET /status`** — Pipeline status and per-cluster counts
- **`POST /retrain/trigger`** — Retrain clusters that have accumulated new data
- **`POST /poller/start`** — Start background polling (alternative to webhooks)

## Evaluation & Tracing

Agent sessions are traced with [Weave](https://wandb.ai/site/weave) for observability. The `evaluation.py` module provides scoring functions decorated with `@weave.op()`:

- **Exploration coverage** — Unique pages/features discovered
- **Demographic consistency** — Whether agent behavior matches cluster profile
- **Friction detection** — Identification of UX pain points

Scores appear in the Weave UI alongside agent traces for debugging and analysis.

## Output

After `run_agents.py` completes, find results in `data/reports/`:

- **`comparative_report_latest.json`** — Multi-agent comparison with recommendations
- **`comparative_report_<timestamp>.json`** — Timestamped report archive

The comparative report includes:
- **Per-agent metrics** — Steps, duration, success/failure rates, impressions
- **Per-agent narrative** — First-person UX report from each demographic
- **Common friction points** — Issues found by multiple demographics
- **Demographic-specific issues** — Problems unique to certain user groups
- **Prioritized recommendations** — Ranked by impact, tagged by affected demographics
- **Engagement patterns** — Which demographics engaged most/least and why

## Visualizer Dashboard

The `visualizer/` directory contains a **Next.js 16** dashboard (React 19, Recharts, Tailwind CSS 4) for interactive comparison of agent results:

```bash
cd visualizer && npm install && npm run dev
```

Components include comparative dashboards, action timelines, friction maps, timing analysis, and qualitative report rendering.

## Tech Stack

| Category | Technologies |
|---|---|
| **Language** | Python 3.10+, TypeScript |
| **LLM** | Mistral-7B-Instruct-v0.3, Mistral API (Medium, Embed) |
| **Fine-Tuning** | PEFT (LoRA), TRL, Flash Attention 2, W&B |
| **Serving** | vLLM (multi-LoRA), FastAPI + Uvicorn |
| **Browser Automation** | Playwright, AgentQL |
| **ML/Clustering** | scikit-learn (K-Means), NumPy |
| **Observability** | Weave (tracing + evaluation), W&B (experiments) |
| **Frontend** | Next.js 16, React 19, Recharts, Tailwind CSS 4 |
| **Data Source** | PostHog session recordings (rrweb) |

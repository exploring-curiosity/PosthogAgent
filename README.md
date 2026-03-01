# Agentic World — Behavioral Digital Twins from Real User Data

Train AI agents on real user behavior from PostHog session recordings — not LLM persona prompts — to create persistent digital twins of user demographics. These agents learn how different people actually interact with web products, then autonomously explore and evaluate any web application from their demographic's perspective.

## Core Differentiator

Every competitor describes personas with prompts. **We clone them from real behavioral data.**

Agents learn scrolling speed, click patterns, hesitation moments, feature discovery order, and engagement style from actual PostHog recordings. They are then fine-tuned into demographic-specific models that can explore unfamiliar applications the way real users from that demographic would.

## How It Works

```
PostHog Recordings ──► Parse ──► Describe ──► Encode ──► Cluster ──► Train ──► Fine-Tune ──► Explore ──► Report
     (15+ users)       S1        S2           S3         K-Means     JSONL     Mistral       AgentQL     Comparative
                       Python    Mistral      Mistral    scikit      Training  Fine-tune     Playwright  Dashboard
                                 Medium       Embed      learn       Data      API + W&B
```

1. **Download** — Bulk-fetch all session recordings from PostHog API
2. **Parse** — Extract structured events (clicks, scrolls, inputs, API calls) from rrweb data
3. **Describe** — Mistral Medium generates natural language behavioral profiles
4. **Encode** — Mistral Embed creates vector embeddings for each session
5. **Cluster** — K-Means groups sessions into 3 natural demographic archetypes
6. **Train** — Convert action sequences into supervised (state → next_action) training pairs
7. **Fine-Tune** — Create 3 demographic-specific Mistral models via fine-tuning API, tracked in W&B
8. **Explore** — Each agent autonomously explores the target app using its fine-tuned decision model
9. **Report** — Comparative dashboard + per-demographic narrative reports

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
python fine_tune.py

# 7. Run all 3 agents against the target app
python run_agents.py

# 8. View results in the comparative dashboard
cd visualizer && npx next dev -p 3333
```

## Prerequisites

- **Python 3.11+**
- **Mistral API key** — [console.mistral.ai](https://console.mistral.ai/) (for description, fine-tuning, and reports)
- **AgentQL API key** — [agentql.com](https://www.agentql.com/) (for semantic DOM querying)
- **PostHog Personal API key** — [posthog.com](https://posthog.com/) (for downloading recordings)
- **W&B API key** — [wandb.ai](https://wandb.ai/) (for fine-tuning experiment tracking)
- **Target web app** — Any web application the agents should explore

## Safety

The agent **refuses to run against production URLs** listed in `BLOCKED_HOSTS`. The `config.py` module validates the target URL at startup.

## Project Structure

```
PosthogAgent/
├── download_recordings.py       # Fetch all recordings from PostHog API
├── cluster_demographics.py      # Process recordings + K-Means clustering
├── build_training_data.py       # Generate fine-tuning JSONL from clusters
├── fine_tune.py                 # Mistral fine-tuning + W&B tracking
├── run_agents.py                # Launch 3 demographic agents
├── config.py                    # Env vars, validation, paths
├── run_pipeline.py              # Legacy single-session pipeline
├── pipeline/
│   ├── stage1_parse.py          # PostHog JSON → structured events
│   ├── stage2_describe.py       # Events → behavioral narrative
│   ├── stage3_encode.py         # Narrative → vector embedding
│   ├── stage4_policy.py         # Description → agent policy JSON
│   ├── stage5_execute.py        # Legacy: fixed-sequence agent
│   └── stage5_explore.py        # NEW: autonomous exploratory agent
├── feedback/
│   ├── session_logger.py        # Records agent actions
│   ├── metrics.py               # Quantitative metrics
│   └── stage6_report.py         # Feedback report generation
├── visualizer/                  # Next.js comparative dashboard
│   └── app/components/
│       ├── ComparativeDashboard.tsx  # Multi-agent comparison view
│       └── ...                       # Per-agent detail components
├── data/
│   ├── recordings/              # Raw PostHog exports
│   ├── parsed/                  # Structured event sequences
│   ├── descriptions/            # Behavioral narratives
│   ├── embeddings/              # Vector embeddings
│   ├── clusters/                # Demographic cluster assignments
│   ├── training/                # Fine-tuning JSONL data
│   ├── models/                  # Fine-tuned model registry
│   ├── agent_logs/              # Exploration session logs
│   └── reports/                 # Comparative + per-agent reports
├── .env.example
└── requirements.txt
```

## Running Individual Steps

```bash
# Download recordings (skip already downloaded)
python download_recordings.py --min-duration 30

# Cluster with custom count
python cluster_demographics.py --clusters 3

# Build training data with larger context window
python build_training_data.py --window-size 7

# Fine-tune with a specific base model
python fine_tune.py --base-model open-mistral-nemo

# Run only one demographic
python run_agents.py --cluster 0 --max-steps 20 --max-duration 120
```

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

"""
Run all 3 demographic agents against the target application.
Each agent autonomously explores using its fine-tuned model, then generates
a narrative report from its demographic perspective.

Usage:
    python run_agents.py
    python run_agents.py --max-steps 20 --max-duration 120
    python run_agents.py --cluster 0   # run only one cluster
"""

import json
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    MISTRAL_API_KEY,
    TARGET_APP_URL,
    TARGET_APP_NAME,
    AGENT_LOGS_DIR,
    MODELS_DIR,
    CLUSTERS_DIR,
    REPORTS_DIR,
    ensure_data_dirs,
    validate_sandbox_url,
)
from pipeline.stage5_explore import ExploratoryAgent
from feedback.session_logger import SessionLogger


def load_config() -> tuple[dict, dict]:
    """Load models and clusters config."""
    models_path = MODELS_DIR / "models.json"
    clusters_path = CLUSTERS_DIR / "clusters.json"

    if not models_path.exists():
        print("ERROR: models.json not found. Run fine_tune.py first.")
        sys.exit(1)
    if not clusters_path.exists():
        print("ERROR: clusters.json not found. Run cluster_demographics.py first.")
        sys.exit(1)

    with open(models_path) as f:
        models = json.load(f)
    with open(clusters_path) as f:
        clusters = json.load(f)

    return models, clusters


def run_single_agent(cluster: dict, model_id: str, target_url: str,
                     max_steps: int, max_duration: int) -> tuple[dict, str, dict]:
    """Run one agent and return (summary, narrative, logger_dict)."""
    cluster_id = cluster["id"]
    label = cluster.get("label", f"Cluster {cluster_id}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger = SessionLogger(session_id=f"explore_{cluster_id}_{timestamp}")

    agent = ExploratoryAgent(
        model_id=model_id,
        demographic=cluster,
        target_url=target_url,
        mistral_api_key=MISTRAL_API_KEY,
        session_logger=logger,
        max_steps=max_steps,
        max_duration_s=max_duration,
    )

    # Run exploration
    summary = agent.run()

    # Generate narrative
    print(f"\n  Generating narrative report for {label}...")
    narrative = agent.generate_narrative(summary)

    # Save agent log
    log_path = AGENT_LOGS_DIR / f"explore_{cluster_id}_{timestamp}.json"
    logger.save(str(log_path))
    print(f"  Agent log saved to {log_path}")

    return summary, narrative, logger.to_dict()


def generate_comparative_report(
    all_summaries: list[dict],
    all_narratives: list[str],
    all_clusters: list[dict],
    target_url: str,
) -> dict:
    """Generate comparative analysis across all agents."""
    from mistralai import Mistral
    client = Mistral(api_key=MISTRAL_API_KEY)

    # Build per-agent summaries for the prompt
    agent_sections = ""
    for i, (summary, narrative, cluster) in enumerate(zip(all_summaries, all_narratives, all_clusters)):
        label = cluster.get("label", f"Agent {i}")
        agent_sections += f"""
--- Agent {i+1}: {label} ---
Steps: {summary.get('total_steps', 0)}, Duration: {summary.get('total_duration_s', 0):.0f}s
Successful: {summary.get('successful_actions', 0)}, Failed: {summary.get('failed_actions', 0)}
Impressions: {len(summary.get('impressions', []))}

Narrative excerpt:
{narrative[:800]}

Key impressions:
"""
        for imp in summary.get("impressions", [])[:5]:
            agent_sections += f"  [{imp.get('sentiment', '?')}] {imp.get('context', '')[:80]}\n"

    prompt = f"""You are analyzing the results of 3 AI agents that explored a web application ({target_url}) from different demographic perspectives.

{agent_sections}

Generate a comprehensive comparative analysis in JSON format:
{{
  "common_friction_points": ["<issue found by multiple demographics>", ...],
  "demographic_specific_issues": [
    {{"demographic": "<label>", "issue": "<specific issue>", "severity": "high|medium|low"}},
    ...
  ],
  "accessibility_findings": ["<finding>", ...],
  "engagement_patterns": {{
    "most_engaged_demographic": "<label>",
    "least_engaged_demographic": "<label>",
    "engagement_summary": "<brief comparison>"
  }},
  "recommendations": [
    {{"title": "<recommendation>", "impact": "high|medium|low", "demographics_affected": ["<label>", ...], "detail": "<explanation>"}},
    ...
  ],
  "overall_assessment": "<2-3 sentence overall UX assessment>"
}}"""

    try:
        response = client.chat.complete(
            model="mistral-medium-latest",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"  Comparative analysis failed: {e}")
        return {"error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Run demographic agents against target app")
    parser.add_argument("--max-steps", type=int, default=25, help="Max steps per agent")
    parser.add_argument("--max-duration", type=int, default=180, help="Max duration per agent (seconds)")
    parser.add_argument("--cluster", type=int, default=None, help="Run only this cluster ID")
    args = parser.parse_args()

    if not MISTRAL_API_KEY:
        print("ERROR: MISTRAL_API_KEY not set in .env")
        sys.exit(1)

    ensure_data_dirs()

    # Validate target URL
    target_url = validate_sandbox_url(TARGET_APP_URL)
    print(f"Target app: {target_url} ({TARGET_APP_NAME})")

    # Load config
    models, clusters_data = load_config()

    # Determine which clusters to run
    all_clusters = clusters_data["clusters"]
    if args.cluster is not None:
        all_clusters = [c for c in all_clusters if c["id"] == args.cluster]
        if not all_clusters:
            print(f"ERROR: Cluster {args.cluster} not found")
            sys.exit(1)

    print(f"\n{'='*60}")
    print(f"AGENTIC WORLD — Multi-Agent Exploration")
    print(f"{'='*60}")
    print(f"Target: {target_url}")
    print(f"Agents: {len(all_clusters)}")
    print(f"Budget: {args.max_steps} steps / {args.max_duration}s per agent")
    print(f"{'='*60}\n")

    all_summaries = []
    all_narratives = []
    all_logs = []
    run_clusters = []

    for cluster in all_clusters:
        cluster_id = cluster["id"]
        label = cluster.get("label", f"Cluster {cluster_id}")
        model_info = models.get(f"cluster_{cluster_id}", {})
        model_id = model_info.get("model_id")

        if not model_id or model_info.get("status") != "success":
            print(f"\nSKIPPING cluster {cluster_id} ({label}): no fine-tuned model available")
            print(f"  Run fine_tune.py first to create a model for this cluster")
            continue

        print(f"\n{'='*50}")
        print(f"RUNNING AGENT: {label} (cluster {cluster_id})")
        print(f"Model: {model_id}")
        print(f"{'='*50}")

        try:
            summary, narrative, log_dict = run_single_agent(
                cluster, model_id, target_url, args.max_steps, args.max_duration
            )
            all_summaries.append(summary)
            all_narratives.append(narrative)
            all_logs.append(log_dict)
            run_clusters.append(cluster)
        except Exception as e:
            print(f"\n  AGENT FAILED: {e}")
            all_summaries.append({"demographic": label, "error": str(e)})
            all_narratives.append(f"Agent failed: {e}")
            all_logs.append({})
            run_clusters.append(cluster)

    # Generate comparative report
    print(f"\n{'='*60}")
    print("GENERATING COMPARATIVE ANALYSIS")
    print(f"{'='*60}")

    comparative = generate_comparative_report(
        all_summaries, all_narratives, run_clusters, target_url
    )

    # Assemble final report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {
        "generated_at": datetime.now().isoformat(),
        "target_app": {
            "url": target_url,
            "name": TARGET_APP_NAME,
        },
        "agents": [],
        "comparative": comparative,
    }

    for i, (summary, narrative, cluster) in enumerate(zip(all_summaries, all_narratives, run_clusters)):
        report["agents"].append({
            "cluster_id": cluster["id"],
            "demographic": cluster.get("label", f"Cluster {cluster['id']}"),
            "traits": cluster.get("dominant_traits", {}),
            "description": cluster.get("description", ""),
            "summary": summary,
            "narrative": narrative,
        })

    # Save report
    report_path = REPORTS_DIR / f"comparative_report_{timestamp}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nComparative report saved to {report_path}")

    # Also save a latest symlink/copy for the visualizer
    latest_path = REPORTS_DIR / "comparative_report_latest.json"
    with open(latest_path, "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print("ALL AGENTS COMPLETE")
    print(f"{'='*60}")

    for agent_data in report["agents"]:
        label = agent_data["demographic"]
        summary = agent_data["summary"]
        print(f"\n  {label}:")
        print(f"    Steps: {summary.get('total_steps', '?')}, "
              f"Duration: {summary.get('total_duration_s', '?')}s")
        print(f"    Success: {summary.get('successful_actions', '?')}, "
              f"Failed: {summary.get('failed_actions', '?')}")

    if comparative.get("overall_assessment"):
        print(f"\n  Overall Assessment:")
        print(f"    {comparative['overall_assessment']}")

    if comparative.get("recommendations"):
        print(f"\n  Top Recommendations:")
        for rec in comparative["recommendations"][:3]:
            print(f"    [{rec.get('impact', '?')}] {rec.get('title', '?')}")

    print(f"\n  Full report: {report_path}")


if __name__ == "__main__":
    main()

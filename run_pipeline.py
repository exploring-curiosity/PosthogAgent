"""
End-to-End Pipeline Orchestrator
PostHog Session Recording -> Behavioral Agent -> UX Feedback Report

Usage:
    python run_pipeline.py <recording.json>
    python run_pipeline.py  # uses default recording in project root
"""

import json
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    MISTRAL_API_KEY,
    SANDBOX_URL,
    validate_sandbox_url,
    validate_api_keys,
    ensure_data_dirs,
    PARSED_DIR,
    DESCRIPTIONS_DIR,
    EMBEDDINGS_DIR,
    POLICIES_DIR,
    AGENT_LOGS_DIR,
    REPORTS_DIR,
)
from pipeline.stage1_parse import parse_recording
from pipeline.stage2_describe import describe_and_save
from pipeline.stage3_encode import encode_and_save
from pipeline.stage4_policy import generate_and_save
from pipeline.stage5_execute import BehavioralAgent
from feedback.session_logger import SessionLogger
from feedback.stage6_report import generate_feedback_report


def main(recording_path: str):
    # ── VALIDATION ──
    print("=" * 60)
    print("PosthogAgent — End-to-End Pipeline")
    print("=" * 60)

    validate_api_keys()
    sandbox_url = validate_sandbox_url()
    ensure_data_dirs()

    print(f"  Recording: {recording_path}")
    print(f"  Sandbox URL: {sandbox_url}")
    print()

    # Derive session ID from filename
    stem = Path(recording_path).stem
    session_id = stem.replace("export-", "").replace("-ph-recording", "")

    # ── STAGE 1: PARSE ──
    parsed_path = str(PARSED_DIR / f"parsed_{session_id}.json")
    if Path(parsed_path).exists():
        print("STAGE 1: Using cached parsed session...")
        with open(parsed_path) as f:
            parsed = json.load(f)
    else:
        print("STAGE 1: Parsing session recording...")
        parsed = parse_recording(recording_path, parsed_path)

    print(f"  Parsed {parsed['total_raw_events']} raw events -> "
          f"{len(parsed['high_level_actions'])} high-level actions")
    print(f"  User: {parsed['user_profile']['age_group']}, "
          f"{parsed['user_profile']['country']}, "
          f"NYC familiarity: {parsed['user_profile']['nyc_familiarity']}")
    print(f"  Duration: {parsed['session_duration_s']}s")

    # ── STAGE 2: DESCRIBE ──
    desc_path = str(DESCRIPTIONS_DIR / f"description_{session_id}.txt")
    if Path(desc_path).exists():
        print("\nSTAGE 2: Using cached behavioral description...")
        with open(desc_path) as f:
            description = f.read()
    else:
        print("\nSTAGE 2: Generating behavioral description via Mistral Large...")
        description = describe_and_save(parsed, MISTRAL_API_KEY, desc_path)
    print(f"  Generated {len(description)} character behavioral description")

    # ── STAGE 3: ENCODE ──
    embed_path = str(EMBEDDINGS_DIR / f"embedding_{session_id}.json")
    if Path(embed_path).exists():
        print("\nSTAGE 3: Using cached embedding...")
        with open(embed_path) as f:
            embed_result = json.load(f)
    else:
        print("\nSTAGE 3: Encoding behavior into vector via Mistral Embed...")
        embed_result = encode_and_save(
            description, parsed["user_profile"], MISTRAL_API_KEY, embed_path
        )
    print(f"  Generated {embed_result['embedding_dim']}-dimensional embedding")

    # ── STAGE 4: GENERATE POLICY ──
    policy_path = str(POLICIES_DIR / f"policy_{session_id}.json")
    if Path(policy_path).exists():
        print("\nSTAGE 4: Using cached policy...")
        with open(policy_path) as f:
            policy = json.load(f)
    else:
        print("\nSTAGE 4: Generating agent behavioral policy via Mistral Large...")
        policy = generate_and_save(
            description, parsed["user_profile"], MISTRAL_API_KEY, policy_path
        )
    print(f"  Generated policy with {len(policy.get('action_sequence', []))} actions")
    print(f"  Style: {policy.get('navigation_style')}, Speed: {policy.get('browsing_speed')}")
    print(f"  Sequence: {policy.get('action_sequence', [])}")

    # ── STAGE 5: EXECUTE ──
    print("\nSTAGE 5: Launching agent on sandbox...")
    print(f"  Target: {sandbox_url}")

    logger = SessionLogger(session_id=session_id)
    agent = BehavioralAgent(
        policy=policy,
        sandbox_url=sandbox_url,
        mistral_api_key=MISTRAL_API_KEY,
        session_logger=logger,
    )
    agent.run()

    # Save agent session log
    log_path = str(AGENT_LOGS_DIR / f"agent_log_{session_id}.json")
    logger.save(log_path)
    agent_log = logger.to_dict()

    summary = logger.get_summary()
    print(f"\n  Agent session summary:")
    print(f"    Duration: {summary['total_duration_s']}s")
    print(f"    Actions: {summary['total_actions']} "
          f"({summary['successful_actions']} ok, {summary['failed_actions']} failed)")
    print(f"    Stuck events: {summary['stuck_events']}")

    # ── STAGE 6: FEEDBACK REPORT ──
    print("\nSTAGE 6: Generating UX feedback report...")
    report_json, report_md = generate_feedback_report(
        agent_log=agent_log,
        real_session_parsed=parsed,
        agent_policy=policy,
        real_user_description=description,
        api_key=MISTRAL_API_KEY,
        output_dir=str(REPORTS_DIR),
        session_id=session_id,
    )

    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"\nOutputs:")
    print(f"  Parsed session:  {parsed_path}")
    print(f"  Description:     {desc_path}")
    print(f"  Embedding:       {embed_path}")
    print(f"  Policy:          {policy_path}")
    print(f"  Agent log:       {log_path}")
    print(f"  Feedback (JSON): {REPORTS_DIR / f'feedback_report_{session_id}.json'}")
    print(f"  Feedback (MD):   {REPORTS_DIR / f'feedback_report_{session_id}.md'}")

    # Print a preview of the qualitative feedback
    qual = report_json.get("qualitative_feedback", "")
    if qual:
        print(f"\n--- Feedback Preview (first 500 chars) ---")
        print(qual[:500])
        print("...")


if __name__ == "__main__":
    # Default to the recording in the project root
    default_recording = str(
        Path(__file__).parent
        / "export-019ca1b0-0b09-7402-890c-e7b3e9f23d25-ph-recording.json"
    )
    recording_path = sys.argv[1] if len(sys.argv) > 1 else default_recording

    if not Path(recording_path).exists():
        print(f"ERROR: Recording file not found: {recording_path}")
        sys.exit(1)

    main(recording_path)

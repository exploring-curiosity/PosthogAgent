"""
Batch-generate policies for all sessions that have descriptions but no policies yet.
Uses stage4_policy.py to generate structured behavioral policies via Mistral Large.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import MISTRAL_API_KEY, DESCRIPTIONS_DIR, POLICIES_DIR, PARSED_DIR, ensure_data_dirs, validate_api_keys
from pipeline.stage4_policy import generate_and_save

validate_api_keys()
ensure_data_dirs()

description_files = sorted(DESCRIPTIONS_DIR.glob("description_*.txt"))
print(f"Found {len(description_files)} descriptions")

generated = 0
skipped = 0
failed = 0

for desc_file in description_files:
    session_id = desc_file.stem.replace("description_", "")
    policy_path = POLICIES_DIR / f"policy_{session_id}.json"
    parsed_path = PARSED_DIR / f"parsed_{session_id}.json"

    if policy_path.exists():
        print(f"  SKIP {session_id} (policy exists)")
        skipped += 1
        continue

    # Load description
    description = desc_file.read_text().strip()
    if len(description) < 100:
        print(f"  SKIP {session_id} (description too short)")
        skipped += 1
        continue

    # Load user profile from parsed session
    user_profile = {}
    if parsed_path.exists():
        with open(parsed_path) as f:
            parsed = json.load(f)
        user_profile = parsed.get("user_profile", {})

    print(f"  GENERATING policy for {session_id}...")
    try:
        policy = generate_and_save(
            description, user_profile, MISTRAL_API_KEY, str(policy_path)
        )
        n_actions = len(policy.get("action_sequence", []))
        print(f"    -> {n_actions} actions, style={policy.get('navigation_style')}")
        generated += 1
    except Exception as e:
        print(f"    FAILED: {e}")
        failed += 1

print(f"\nDone: {generated} generated, {skipped} skipped, {failed} failed")
print(f"Total policies: {len(list(POLICIES_DIR.glob('policy_*.json')))}")

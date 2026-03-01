"""
Download all session recordings from PostHog API.
Skips recordings already downloaded. Saves in the same format as manual exports.

Usage:
    python download_recordings.py
    python download_recordings.py --min-duration 30   # skip recordings shorter than 30s
"""

import json
import sys
import time
import argparse
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    POSTHOG_PERSONAL_API_KEY,
    POSTHOG_PROJECT_ID,
    POSTHOG_HOST,
    RECORDINGS_DIR,
    ensure_data_dirs,
)


def list_all_recordings(host: str, project_id: str, api_key: str) -> list[dict]:
    """Paginate through all session recordings in the project."""
    all_recordings = []
    cursor = None

    while True:
        params = {"limit": 50}
        if cursor:
            params["cursor"] = cursor

        resp = httpx.get(
            f"{host}/api/projects/{project_id}/session_recordings",
            headers={"Authorization": f"Bearer {api_key}"},
            params=params,
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()

        all_recordings.extend(data.get("results", []))

        if data.get("has_next") and data.get("next_cursor"):
            cursor = data["next_cursor"]
        else:
            break

    return all_recordings


def get_recording_person(host: str, project_id: str, api_key: str, recording_id: str) -> dict:
    """Fetch person details for a recording."""
    resp = httpx.get(
        f"{host}/api/projects/{project_id}/session_recordings/{recording_id}",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("person", {})


def download_snapshots(host: str, project_id: str, api_key: str, recording_id: str) -> list[dict]:
    """Download all snapshot blobs for a recording and return as a list of rrweb events."""
    # First, get the list of sources/blob keys
    resp = httpx.get(
        f"{host}/api/projects/{project_id}/session_recordings/{recording_id}/snapshots",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=60,
    )
    resp.raise_for_status()
    meta = resp.json()

    sources = meta.get("sources", [])
    if not sources:
        print(f"    Warning: No snapshot sources found for {recording_id}")
        return []

    # Get the min and max blob keys
    blob_keys = [int(s["blob_key"]) for s in sources if s.get("source") == "blob_v2"]
    if not blob_keys:
        print(f"    Warning: No blob_v2 sources for {recording_id}")
        return []

    start_key = str(min(blob_keys))
    end_key = str(max(blob_keys))

    # Fetch all snapshots in one request
    resp = httpx.get(
        f"{host}/api/projects/{project_id}/session_recordings/{recording_id}/snapshots",
        headers={"Authorization": f"Bearer {api_key}"},
        params={"source": "blob_v2", "start_blob_key": start_key, "end_blob_key": end_key},
        timeout=120,
    )
    resp.raise_for_status()

    # Parse JSONL response: each line is [event_id, snapshot_data]
    snapshots = []
    for line in resp.text.strip().split("\n"):
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
            if isinstance(parsed, list) and len(parsed) >= 2:
                event_data = parsed[1]
                # Add windowId from the event_id prefix if not present
                if "windowId" not in event_data:
                    event_data["windowId"] = parsed[0]
                snapshots.append(event_data)
            elif isinstance(parsed, dict):
                snapshots.append(parsed)
        except json.JSONDecodeError:
            continue

    return snapshots


def save_recording(recording_meta: dict, person: dict, snapshots: list[dict], output_path: Path):
    """Save in the same format as PostHog manual exports."""
    export = {
        "version": "2024-04-30",
        "data": {
            "id": recording_meta["id"],
            "person": person,
            "snapshots": snapshots,
        },
    }

    with open(output_path, "w") as f:
        json.dump(export, f)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"    Saved {len(snapshots)} snapshots ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Download PostHog session recordings")
    parser.add_argument("--min-duration", type=int, default=10,
                        help="Skip recordings shorter than this (seconds)")
    args = parser.parse_args()

    # Validate config
    if not POSTHOG_PERSONAL_API_KEY:
        print("ERROR: POSTHOG_PERSONAL_API_KEY not set in .env")
        sys.exit(1)
    if not POSTHOG_PROJECT_ID:
        print("ERROR: POSTHOG_PROJECT_ID not set in .env")
        sys.exit(1)

    ensure_data_dirs()

    # Find already-downloaded recordings
    existing = set()
    for f in RECORDINGS_DIR.glob("export-*-ph-recording.json"):
        session_id = f.stem.replace("export-", "").replace("-ph-recording", "")
        existing.add(session_id)

    print(f"Already downloaded: {len(existing)} recordings")

    # List all recordings from PostHog
    print(f"Fetching recording list from {POSTHOG_HOST}...")
    all_recordings = list_all_recordings(POSTHOG_HOST, POSTHOG_PROJECT_ID, POSTHOG_PERSONAL_API_KEY)
    print(f"Found {len(all_recordings)} total recordings in project {POSTHOG_PROJECT_ID}")

    # Filter
    to_download = []
    skipped_short = 0
    skipped_existing = 0

    for rec in all_recordings:
        rid = rec["id"]
        duration = rec.get("recording_duration", 0)

        if rid in existing:
            skipped_existing += 1
            continue
        if duration < args.min_duration:
            skipped_short += 1
            continue
        to_download.append(rec)

    print(f"\nSkipped: {skipped_existing} already downloaded, {skipped_short} too short (<{args.min_duration}s)")
    print(f"To download: {len(to_download)} recordings\n")

    # Download each
    downloaded = 0
    failed = 0

    for i, rec in enumerate(to_download):
        rid = rec["id"]
        duration = rec.get("recording_duration", 0)
        person_meta = rec.get("person", {}) or {}
        props = person_meta.get("properties", {}) or {}
        age = props.get("age_group", "?")
        country = props.get("country", "?")

        print(f"[{i+1}/{len(to_download)}] {rid} ({duration:.0f}s, age={age}, country={country})")

        try:
            # Get full person data
            person = get_recording_person(POSTHOG_HOST, POSTHOG_PROJECT_ID, POSTHOG_PERSONAL_API_KEY, rid)

            # Download snapshots
            snapshots = download_snapshots(POSTHOG_HOST, POSTHOG_PROJECT_ID, POSTHOG_PERSONAL_API_KEY, rid)

            if not snapshots:
                print(f"    Skipped: no snapshots")
                failed += 1
                continue

            # Save
            output_path = RECORDINGS_DIR / f"export-{rid}-ph-recording.json"
            save_recording(rec, person, snapshots, output_path)
            downloaded += 1

            # Brief pause to avoid rate limiting
            time.sleep(0.5)

        except Exception as e:
            print(f"    ERROR: {e}")
            failed += 1

    # Summary
    print(f"\n{'='*60}")
    print(f"DOWNLOAD COMPLETE")
    print(f"{'='*60}")
    print(f"  Downloaded: {downloaded}")
    print(f"  Failed: {failed}")
    print(f"  Previously existing: {skipped_existing}")
    print(f"  Total on disk: {downloaded + len(existing)}")

    # Show demographics summary
    all_on_disk = []
    for f in sorted(RECORDINGS_DIR.glob("export-*-ph-recording.json")):
        with open(f) as fh:
            data = json.load(fh)
        person = data.get("data", {}).get("person", {})
        props = person.get("properties", {}) or {}
        all_on_disk.append({
            "id": data["data"]["id"],
            "age_group": props.get("age_group", "?"),
            "country": props.get("country", "?"),
            "nyc_familiarity": props.get("nyc_familiarity", "?"),
            "snapshots": len(data["data"].get("snapshots", [])),
        })

    print(f"\nDemographics summary ({len(all_on_disk)} recordings):")
    ages = {}
    countries = {}
    for r in all_on_disk:
        ages[r["age_group"]] = ages.get(r["age_group"], 0) + 1
        countries[r["country"]] = countries.get(r["country"], 0) + 1

    print(f"  Age groups: {dict(sorted(ages.items(), key=lambda x: -x[1]))}")
    print(f"  Countries: {dict(sorted(countries.items(), key=lambda x: -x[1]))}")


if __name__ == "__main__":
    main()

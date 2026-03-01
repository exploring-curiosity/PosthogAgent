"""
Stage 2: Describe — Convert structured events into a natural language
behavioral narrative using Mistral Large.

Takes the parsed session output from Stage 1 and sends it to Mistral Large
to produce a detailed behavioral profile of the user's session.
"""

import json
from pathlib import Path
from mistralai import Mistral


def build_action_log(actions: list[dict]) -> str:
    """Convert high-level actions into a human-readable action log."""
    lines = []
    for a in actions:
        if a["action"] == "CLICK":
            line = f"[{a['time']}s] Clicked {a['target_tag']}"
            if a.get("target_href"):
                line += f" (link to: {a['target_href']})"
            if a.get("target_text"):
                line += f' text="{a["target_text"]}"'
            line += f" at ({a['coordinates'][0]}, {a['coordinates'][1]})"
            lines.append(line)

        elif a["action"] == "SCROLL":
            lines.append(
                f"[{a['time']}s] Scrolled {a['direction']} "
                f"from {a['scroll_from']}px to {a['scroll_to']}px "
                f"(max depth: {a['max_depth']}px) over {a['duration_s']}s"
            )

        elif a["action"] == "TYPE":
            line = (
                f"[{a['time']}s] Typed {a['text_length']} characters "
                f"into {a['target_tag']}"
            )
            if a.get("target_placeholder"):
                line += f' (placeholder: "{a["target_placeholder"]}")'
            lines.append(line)

        elif a["action"] == "API_CALL":
            lines.append(f"[{a['time']}s] API: {a['full_path']}")

        elif a["action"] == "PAGE_LOAD":
            lines.append(f"[{a['time']}s] Page loaded: {a['url']} ({a['viewport']})")

    return "\n".join(lines)


def describe_session(parsed_session: dict, api_key: str) -> str:
    """
    Send parsed session data to Mistral Large to get a
    natural language behavioral description.
    """
    client = Mistral(api_key=api_key)

    profile = parsed_session["user_profile"]
    actions = parsed_session["high_level_actions"]
    duration = parsed_session["session_duration_s"]

    action_log = build_action_log(actions)

    prompt = f"""You are analyzing a web session recording from a web application.
The session was recorded via PostHog and parsed into a structured action log.

USER PROFILE:
- Age group: {profile.get('age_group', 'unknown')}
- Country: {profile.get('country', 'unknown')}
- NYC familiarity: {profile.get('nyc_familiarity', 'unknown')}
- Device: {profile.get('device_type', 'unknown')}, {profile.get('browser', 'unknown')} on {profile.get('os', 'unknown')}
- Viewport: {profile.get('viewport_width')}x{profile.get('viewport_height')}

SESSION DURATION: {duration} seconds

ACTION LOG:
{action_log}

Based on this action log, produce a detailed behavioral description covering:

1. NAVIGATION PATTERN: How does this user move through the site? (linear, exploratory, focused, scattered?)
2. READING BEHAVIOR: Do they read deeply or skim? How do scroll patterns indicate this?
3. ENGAGEMENT STYLE: Do they vote, comment, or both? How quickly? What's their vote-to-comment ratio?
4. INTERACTION SPEED: Are they fast or deliberate? How long between actions?
5. CONTENT PREFERENCES: What types of content/subreddits did they gravitate toward?
6. TYPING BEHAVIOR: How fast do they type? Did they make corrections? How long are their inputs?
7. FEATURE DISCOVERY: Which features did they find and use? Which did they ignore?
8. SESSION FLOW: Describe the overall arc of the session (exploration -> engagement -> browsing -> exit)

Be specific and quantitative. Reference actual timestamps and actions.
Do NOT speculate beyond what the data shows.
Output as a structured behavioral profile in plain text."""

    response = client.chat.complete(
        model="mistral-large-latest",
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message.content


def describe_and_save(parsed_session: dict, api_key: str, output_path: str | None = None) -> str:
    """Run Stage 2 and optionally save the output."""
    description = describe_session(parsed_session, api_key)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(description)
        print(f"  Saved behavioral description to {output_path}")

    return description


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import MISTRAL_API_KEY, PARSED_DIR, DESCRIPTIONS_DIR, ensure_data_dirs, validate_api_keys

    validate_api_keys()
    ensure_data_dirs()

    if len(sys.argv) < 2:
        print("Usage: python -m pipeline.stage2_describe <parsed_session.json>")
        sys.exit(1)

    with open(sys.argv[1]) as f:
        parsed = json.load(f)

    session_id = parsed["user_profile"]["session_id"]
    output_path = str(DESCRIPTIONS_DIR / f"description_{session_id}.txt")

    description = describe_and_save(parsed, MISTRAL_API_KEY, output_path)
    print(f"\nGenerated {len(description)} character behavioral description")
    print("\n--- Preview (first 500 chars) ---")
    print(description[:500])

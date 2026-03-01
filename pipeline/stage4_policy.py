"""
Stage 4: Generate Policy — Convert behavioral description into a structured
JSON policy that an autonomous agent can follow.

The policy captures behavioral patterns (not specific content) so the agent
acts like the real user.
"""

import json
from pathlib import Path
from mistralai import Mistral


def generate_agent_policy(behavioral_description: str, user_profile: dict, api_key: str) -> dict:
    """
    Generate a structured behavioral policy from a session description.
    """
    client = Mistral(api_key=api_key)

    prompt = f"""You are creating a behavioral policy for an AI agent that will
simulate a real user's behavior on a Reddit-style NYC community platform called FunCity.

USER PROFILE:
{json.dumps(user_profile, indent=2)}

OBSERVED BEHAVIORAL DESCRIPTION:
{behavioral_description}

Based on this observed behavior, generate a structured JSON policy that an
autonomous browser agent can follow. The policy should capture the behavioral
patterns, NOT the specific content.

Output ONLY valid JSON with this exact structure:

{{
  "session_target_duration_s": <number>,
  "navigation_style": "exploratory" | "focused" | "scattered" | "linear",
  "browsing_speed": "fast" | "medium" | "slow",

  "feed_behavior": {{
    "initial_scroll_pattern": "quick_scan" | "slow_read" | "no_scroll",
    "sort_preference": "hot" | "new" | "top" | "controversial",
    "max_posts_to_visit": <number>,
    "scroll_between_posts": true | false
  }},

  "post_interaction": {{
    "comment_probability": <0-1>,
    "vote_probability": <0-1>,
    "vote_then_comment_order": "comment_first" | "vote_first" | "random",
    "avg_comment_length_chars": <number>,
    "typing_speed": "fast" | "medium" | "slow",
    "makes_typos": true | false,
    "read_before_engaging_s": <seconds to pause before acting>
  }},

  "subreddit_exploration": {{
    "explores_subreddits": true | false,
    "preferred_subreddits": [<list of subreddit slugs if observed>],
    "subreddit_browse_depth": "surface" | "deep"
  }},

  "auth_behavior": {{
    "needs_auth": true | false,
    "auth_method": "signup" | "login",
    "auth_timing": "immediately" | "before_first_action" | "when_prompted"
  }},

  "engagement_arc": {{
    "early_session": "exploring" | "engaging" | "lurking",
    "mid_session": "engaging" | "browsing" | "deep_reading",
    "late_session": "browsing" | "disengaging" | "still_active"
  }},

  "action_sequence": [
    <ordered list of high-level actions the agent should take>
  ]
}}

Valid action names for the action_sequence:
- scan_feed: scroll through the homepage feed
- open_post: click on a post to view its detail page
- signup: create a new account (use this if the real user signed up during the session)
- login: log in with existing credentials (use this if the real user logged in during the session)
- write_comment: write and submit a comment on the current post
- vote_on_post: upvote or downvote the current post
- return_to_feed: navigate back to the homepage
- browse_subreddit: click on a subreddit link to explore it
- open_related_post: click on a trending or related post
- create_post: create a new post in a subreddit

IMPORTANT: If the behavioral description mentions the user signing up or creating an account,
use "signup" in the action_sequence, NOT "login". Only use "login" if the user logged into
an existing account."""

    response = client.chat.complete(
        model="mistral-large-latest",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )

    return json.loads(response.choices[0].message.content)


def generate_and_save(
    behavioral_description: str,
    user_profile: dict,
    api_key: str,
    output_path: str | None = None,
) -> dict:
    """Run Stage 4 and optionally save the output."""
    policy = generate_agent_policy(behavioral_description, user_profile, api_key)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(policy, f, indent=2)
        print(f"  Saved agent policy to {output_path}")

    return policy


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import MISTRAL_API_KEY, DESCRIPTIONS_DIR, POLICIES_DIR, ensure_data_dirs, validate_api_keys

    validate_api_keys()
    ensure_data_dirs()

    if len(sys.argv) < 2:
        print("Usage: python -m pipeline.stage4_policy <description.txt>")
        sys.exit(1)

    with open(sys.argv[1]) as f:
        description = f.read()

    stem = Path(sys.argv[1]).stem
    session_id = stem.replace("description_", "")

    output_path = str(POLICIES_DIR / f"policy_{session_id}.json")
    policy = generate_and_save(
        description,
        {"session_id": session_id},
        MISTRAL_API_KEY,
        output_path,
    )
    print(f"Generated policy with {len(policy.get('action_sequence', []))} actions")
    print(f"Navigation style: {policy.get('navigation_style')}")
    print(f"Browsing speed: {policy.get('browsing_speed')}")

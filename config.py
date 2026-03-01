"""
Configuration loader and sandbox URL validation.
All API keys and URLs are loaded from environment variables / .env file.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the project root
_env_path = Path(__file__).parent / ".env"
load_dotenv(_env_path)

# ── API Keys ──
MISTRAL_API_KEY: str = os.environ.get("MISTRAL_API_KEY", "")
AGENTQL_API_KEY: str = os.environ.get("AGENTQL_API_KEY", "")
WANDB_API_KEY: str = os.environ.get("WANDB_API_KEY", "")
NVIDIA_API_KEY: str = os.environ.get("NVIDIA_API_KEY", "")

# ── WhiteCircle ──
WHITECIRCLE_API_KEY: str = os.environ.get("WHITECIRCLE_API_KEY", "")
WHITECIRCLE_API_URL: str = os.environ.get("WHITECIRCLE_API_URL", "https://us.whitecircle.ai/api/v1")
WHITECIRCLE_DEPLOYMENT_ID: str = os.environ.get("WHITECIRCLE_DEPLOYMENT_ID", "")

# ── PostHog ──
POSTHOG_PERSONAL_API_KEY: str = os.environ.get("POSTHOG_PERSONAL_API_KEY", "")
POSTHOG_PROJECT_ID: str = os.environ.get("POSTHOG_PROJECT_ID", "")
POSTHOG_HOST: str = os.environ.get("POSTHOG_HOST", "https://us.posthog.com")

# ── URLs ──
TARGET_APP_URL: str = os.environ.get("TARGET_APP_URL", os.environ.get("SANDBOX_URL", "http://localhost:3000"))
TARGET_APP_NAME: str = os.environ.get("TARGET_APP_NAME", "Web App")
TARGET_APP_DESCRIPTION: str = os.environ.get("TARGET_APP_DESCRIPTION", "")
SANDBOX_URL: str = os.environ.get("SANDBOX_URL", "http://localhost:3000")
PRODUCTION_URL: str = os.environ.get("PRODUCTION_URL", "https://fun-city-xi.vercel.app")

# ── Blocked production hostnames (agent must never target these) ──
BLOCKED_HOSTS = [
    "fun-city-xi.vercel.app",
    "fun-city.vercel.app",
]

# ── Project paths ──
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RECORDINGS_DIR = DATA_DIR / "recordings"
PARSED_DIR = DATA_DIR / "parsed"
DESCRIPTIONS_DIR = DATA_DIR / "descriptions"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
POLICIES_DIR = DATA_DIR / "policies"
AGENT_LOGS_DIR = DATA_DIR / "agent_logs"
REPORTS_DIR = DATA_DIR / "reports"
TRAINING_DIR = DATA_DIR / "training"
MODELS_DIR = DATA_DIR / "models"
CLUSTERS_DIR = DATA_DIR / "clusters"


def ensure_data_dirs():
    """Create all data subdirectories if they don't exist."""
    for d in [
        RECORDINGS_DIR, PARSED_DIR, DESCRIPTIONS_DIR,
        EMBEDDINGS_DIR, POLICIES_DIR, AGENT_LOGS_DIR, REPORTS_DIR,
        TRAINING_DIR, MODELS_DIR, CLUSTERS_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)


def validate_sandbox_url(url: str | None = None) -> str:
    """
    Validate that the target URL is NOT a production URL.
    Returns the validated URL. Exits with error if the URL is blocked.
    """
    target = url or SANDBOX_URL

    from urllib.parse import urlparse
    parsed = urlparse(target)
    hostname = parsed.hostname or ""

    for blocked in BLOCKED_HOSTS:
        if blocked in hostname:
            print(f"\n{'='*60}")
            print("🚫 SAFETY BLOCK: Refusing to run agent against production!")
            print(f"   Target URL: {target}")
            print(f"   Blocked host: {blocked}")
            print(f"   Set SANDBOX_URL in .env to your staging instance.")
            print(f"{'='*60}\n")
            sys.exit(1)

    if not target:
        print("ERROR: SANDBOX_URL is not set. Configure it in .env")
        sys.exit(1)

    return target


def validate_api_keys():
    """Check that required API keys are present."""
    missing = []
    if not MISTRAL_API_KEY:
        missing.append("MISTRAL_API_KEY")
    if not AGENTQL_API_KEY:
        missing.append("AGENTQL_API_KEY")

    if missing:
        print(f"ERROR: Missing API keys: {', '.join(missing)}")
        print("Set them in .env (see .env.example)")
        sys.exit(1)

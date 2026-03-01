"""
============================================================
Agentic Loop — Plan → Observe → Act → Replan → Repeat
============================================================
Drives a browser session using a per-cluster finetuned model.
Input: URL + app description.

Flow:
  1. Generate a behavioral description of the target app
  2. Embed it and classify into the nearest cluster via KMeans
  3. Load the matching per-cluster LoRA adapter
  4. Loop:
     a. OBSERVE — read current page state (URL, title, visible elements)
     b. PLAN   — model predicts next action JSON given product + history + state
     c. ACT    — execute action via AgentQL / Playwright
     d. REPLAN — if error, feed error context back; model picks recovery action
     e. Repeat until max_steps or timeout

Usage:
    python agentic_loop.py --url http://localhost:3000 \
        --app-description "A social forum with posts, voting, and comments"

    python agentic_loop.py --url http://localhost:3000 \
        --cluster 2 --max-steps 30

    python agentic_loop.py --url http://localhost:3000 \
        --adapter data/models/cluster_0_lora --no-classify
============================================================
"""

import os
import sys
import json
import time
import random
import re
import pickle
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    MISTRAL_API_KEY,
    TARGET_APP_URL,
    TARGET_APP_NAME,
    TARGET_APP_DESCRIPTION,
    MODELS_DIR,
    CLUSTERS_DIR,
    AGENT_LOGS_DIR,
    ensure_data_dirs,
    validate_sandbox_url,
)
from build_training_data import (
    BASE_SYSTEM_PROMPT,
    build_cluster_system_prompt,
    build_product_context,
)
from feedback.session_logger import SessionLogger, StuckDetector

# ============================================================
# PARSE ARGS
# ============================================================

parser = argparse.ArgumentParser(description="Agentic browser loop: plan → observe → act → replan")
parser.add_argument("--url", type=str, required=True, help="Target URL to interact with")
parser.add_argument("--app-name", type=str, default=None, help="App name")
parser.add_argument("--app-description", type=str, default=None, help="App feature description")
parser.add_argument("--cluster", type=int, default=None, help="Force a specific cluster ID (skip classification)")
parser.add_argument("--adapter", type=str, default=None, help="Force a specific adapter path")
parser.add_argument("--no-classify", action="store_true", help="Skip cluster classification (use --adapter)")
parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-Instruct-v0.3")
parser.add_argument("--max-steps", type=int, default=20, help="Max actions per session")
parser.add_argument("--max-duration", type=int, default=300, help="Max session duration (seconds)")
parser.add_argument("--temperature", type=float, default=0.7)
parser.add_argument("--headless", action="store_true", help="Run browser headless")
parser.add_argument("--window-size", type=int, default=5, help="Recent action history window")
parser.add_argument("--output", type=str, default=None, help="Session log output path")
args = parser.parse_args()

ensure_data_dirs()

target_url = args.url
app_name = args.app_name or TARGET_APP_NAME or "Web App"
app_description = args.app_description or TARGET_APP_DESCRIPTION

if not app_description:
    print("ERROR: --app-description is required (or set TARGET_APP_DESCRIPTION in .env)")
    sys.exit(1)


# ============================================================
# STEP 1: CLASSIFY INTO CLUSTER
# ============================================================

def classify_into_cluster(description: str) -> tuple[int, dict]:
    """Embed the app description and find the nearest cluster via KMeans."""
    from mistralai import Mistral

    kmeans_path = CLUSTERS_DIR / "kmeans_model.pkl"
    clusters_path = CLUSTERS_DIR / "clusters.json"

    if not kmeans_path.exists() or not clusters_path.exists():
        print("ERROR: No KMeans model or clusters.json found.")
        print("Run: python process_synthetic_batch.py --recordings-dir sessions --clusters 5")
        sys.exit(1)

    # Load KMeans
    with open(kmeans_path, "rb") as f:
        kmeans = pickle.load(f)

    with open(clusters_path) as f:
        clusters_data = json.load(f)

    # Embed the description
    print("  Embedding app description for cluster classification...")
    client = Mistral(api_key=MISTRAL_API_KEY)
    response = client.embeddings.create(
        model="mistral-embed",
        inputs=[description],
    )
    embedding = np.array(response.data[0].embedding).reshape(1, -1)

    # Predict cluster
    cluster_id = int(kmeans.predict(embedding)[0])

    # Find cluster metadata
    cluster_meta = {"id": cluster_id, "label": f"Cluster {cluster_id}"}
    for c in clusters_data.get("clusters", []):
        if c["id"] == cluster_id:
            cluster_meta = c
            break

    return cluster_id, cluster_meta


def get_cluster_meta(cluster_id: int) -> dict:
    """Load cluster metadata by ID."""
    clusters_path = CLUSTERS_DIR / "clusters.json"
    if clusters_path.exists():
        with open(clusters_path) as f:
            data = json.load(f)
        for c in data.get("clusters", []):
            if c["id"] == cluster_id:
                return c
    return {"id": cluster_id, "label": f"Cluster {cluster_id}"}


print("=" * 60)
print("AGENTIC LOOP — PLAN → OBSERVE → ACT → REPLAN")
print("=" * 60)
print(f"Target:      {target_url}")
print(f"Product:     {app_name}")
print(f"Description: {app_description[:80]}...")
print(f"Max steps:   {args.max_steps}, Timeout: {args.max_duration}s")

# Determine cluster and adapter
if args.cluster is not None:
    cluster_id = args.cluster
    cluster_meta = get_cluster_meta(cluster_id)
    print(f"Cluster:     {cluster_id} (forced) — {cluster_meta.get('label', '?')}")
elif args.no_classify and args.adapter:
    cluster_id = -1
    cluster_meta = {"id": -1, "label": "Custom", "key_behaviors": [], "description": ""}
    print(f"Cluster:     custom adapter (no classification)")
else:
    cluster_id, cluster_meta = classify_into_cluster(app_description)
    print(f"Cluster:     {cluster_id} — {cluster_meta.get('label', '?')}")

adapter_path = args.adapter or str(MODELS_DIR / f"cluster_{cluster_id}_lora")
print(f"Adapter:     {adapter_path}")
print("=" * 60)


# ============================================================
# STEP 2: LOAD MODEL + ADAPTER
# ============================================================

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

print(f"\nLoading model: {args.model}")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    bnb_4bit_use_double_quant=True,
)

try:
    import flash_attn  # noqa: F401
    attn_impl = "flash_attention_2"
except ImportError:
    attn_impl = "sdpa"

model = AutoModelForCausalLM.from_pretrained(
    args.model,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    attn_implementation=attn_impl,
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

if os.path.exists(adapter_path):
    print(f"Loading LoRA adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()
    print("  Adapter merged successfully")
else:
    print(f"WARNING: No adapter at {adapter_path}, using base model")

model.eval()
print("Model ready\n")


# ============================================================
# STEP 3: BROWSER SETUP
# ============================================================

import agentql
from playwright.sync_api import sync_playwright

print("Launching browser...")
pw = sync_playwright().start()
browser = pw.chromium.launch(headless=args.headless)
context = browser.new_context(viewport={"width": 1512, "height": 823})
page = agentql.wrap(context.new_page())
page.goto(target_url)
time.sleep(2)
print(f"Navigated to {target_url}\n")


# ============================================================
# STEP 4: PROMPT BUILDING + INFERENCE
# ============================================================

product_context = build_product_context(app_name, target_url, app_description)
system_prompt = build_cluster_system_prompt(cluster_meta)
action_history = []  # [{elapsed, action, target, success, error, page_url}]
current_plan = ""  # Tracks the model's current high-level plan


def observe_page() -> str:
    """Get current page state via AgentQL and Playwright."""
    try:
        url = page.url
        title = page.title()

        # Try to get visible interactive elements via AgentQL
        try:
            query = """{
                navigation_links[]
                buttons[]
                input_fields[]
                main_content_headings[]
            }"""
            elements = page.query_elements(query)

            parts = [f"Page: {url} — {title}"]
            if hasattr(elements, 'navigation_links') and elements.navigation_links:
                nav_texts = []
                for el in elements.navigation_links[:5]:
                    try:
                        nav_texts.append(el.inner_text()[:30])
                    except Exception:
                        pass
                if nav_texts:
                    parts.append(f"Nav: {', '.join(nav_texts)}")

            if hasattr(elements, 'buttons') and elements.buttons:
                btn_texts = []
                for el in elements.buttons[:5]:
                    try:
                        btn_texts.append(el.inner_text()[:30])
                    except Exception:
                        pass
                if btn_texts:
                    parts.append(f"Buttons: {', '.join(btn_texts)}")

            if hasattr(elements, 'input_fields') and elements.input_fields:
                parts.append(f"Inputs: {len(elements.input_fields)} visible")

            if hasattr(elements, 'main_content_headings') and elements.main_content_headings:
                heading_texts = []
                for el in elements.main_content_headings[:3]:
                    try:
                        heading_texts.append(el.inner_text()[:40])
                    except Exception:
                        pass
                if heading_texts:
                    parts.append(f"Content: {', '.join(heading_texts)}")

            return "\n".join(parts)
        except Exception:
            return f"Page: {url} — {title}"

    except Exception:
        return "Page: unknown"


def build_prompt(error_context: str = "") -> list[dict]:
    """Build chat messages: system (persona) + user (product + state + history)."""
    lines = [product_context, ""]

    elapsed = time.time() - session_start
    lines.append(f"Session: step {len(action_history)}, elapsed {elapsed:.1f}s / {args.max_duration}s")

    # Current page observation
    page_state = observe_page()
    lines.append(f"\n{page_state}")

    # Action history window
    window = action_history[-args.window_size:]
    if window:
        lines.append("\nRecent actions:")
        for a in window:
            status = "" if a.get("success", True) else " [FAILED]"
            line = f"  [{a['elapsed']:.1f}s] {a['action']} -> {a['target']}{status}"
            if not a.get("success", True) and a.get("error"):
                line += f" ({a['error'][:40]})"
            lines.append(line)

    # Current plan context
    if current_plan:
        lines.append(f"\nCurrent plan: {current_plan}")

    if error_context:
        lines.append(f"\nLast action failed: {error_context}")
        lines.append("Decide how to recover or try an alternative approach.")
    else:
        lines.append("\nWhat would you do next?")

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "\n".join(lines)},
    ]


def predict_action(error_context: str = "") -> dict:
    """Call model to predict next action."""
    messages = build_prompt(error_context)

    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt",
    ).to(model.device)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=256,
            temperature=args.temperature,
            top_p=0.9,
            do_sample=True,
        )

    raw = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)

    # Parse JSON from response
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try to find JSON object in response
        match = re.search(r'\{[^{}]*\}', raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        print(f"  WARNING: Could not parse model output: {raw[:200]}")
        return {"action": "scroll", "target": "down 300px", "reasoning": "fallback — unparseable output", "hesitation_ms": 500}


# ============================================================
# STEP 5: ACTION EXECUTION
# ============================================================

def execute_action(action_data: dict) -> tuple[bool, str]:
    """Execute a predicted action in the browser. Returns (success, error_msg)."""
    action = action_data.get("action", "wait")
    target = action_data.get("target", "")
    hesitation = action_data.get("hesitation_ms", 0)

    # Apply hesitation (capped at 5s)
    if hesitation > 0:
        time.sleep(min(hesitation / 1000.0, 5.0))

    try:
        if action == "click":
            return _execute_click(target)
        elif action == "scroll":
            return _execute_scroll(target)
        elif action == "type":
            return _execute_type(target)
        elif action == "wait":
            time.sleep(random.uniform(0.5, 2.0))
            return True, ""
        elif action == "navigate_back":
            page.go_back()
            time.sleep(1)
            return True, ""
        else:
            return False, f"Unknown action: {action}"
    except Exception as e:
        return False, str(e)


def _execute_click(target: str) -> tuple[bool, str]:
    """Click via AgentQL semantic query, then Playwright text fallback."""
    semantic = target.strip()

    # 1. AgentQL semantic query
    try:
        escaped = semantic[:100].replace('"', '\\"')
        query = f"""{{ target_element(description: "{escaped}") }}"""
        response = page.query_elements(query)
        if response.target_element:
            response.target_element.click()
            time.sleep(0.5)
            return True, ""
    except Exception:
        pass

    # 2. Playwright text-based locator
    try:
        text_match = re.search(r'"([^"]+)"', semantic)
        if text_match:
            text = text_match.group(1)
            loc = page.locator(f"*:has-text('{text}')").first
            if loc.is_visible(timeout=3000):
                loc.click()
                time.sleep(0.5)
                return True, ""
    except Exception:
        pass

    # 3. Try tag-based locator
    try:
        tag_match = re.match(r'^(\w+)', semantic)
        if tag_match:
            tag = tag_match.group(1).lower()
            if tag in ("button", "a", "link", "input", "span", "div", "h1", "h2", "h3"):
                loc = page.locator(f"{tag}:visible").first
                if loc.is_visible(timeout=2000):
                    loc.click()
                    time.sleep(0.5)
                    return True, ""
    except Exception:
        pass

    return False, f"Could not find element: {semantic[:60]}"


def _execute_scroll(target: str) -> tuple[bool, str]:
    """Scroll the page."""
    direction = "down"
    pixels = 300

    if "up" in target.lower():
        direction = "up"
    match = re.search(r'(\d+)', target)
    if match:
        pixels = min(int(match.group(1)), 2000)

    delta = pixels if direction == "down" else -pixels
    page.mouse.wheel(0, delta)
    time.sleep(0.3)
    return True, ""


def _execute_type(target: str) -> tuple[bool, str]:
    """Type into an input found via AgentQL or Playwright."""
    semantic = target.strip()

    # 1. AgentQL
    try:
        escaped = semantic[:100].replace('"', '\\"')
        query = f"""{{ input_field(description: "{escaped}") }}"""
        response = page.query_elements(query)
        if response.input_field:
            response.input_field.click()
            sample_texts = [
                "Hello world", "Testing 123", "Great post!",
                "This is interesting", "Love this place", "Check this out",
            ]
            response.input_field.type(random.choice(sample_texts), delay=random.randint(50, 120))
            time.sleep(0.3)
            return True, ""
    except Exception:
        pass

    # 2. Playwright fallback
    try:
        inputs = page.locator("input:visible, textarea:visible")
        if inputs.count() > 0:
            inp = inputs.first
            inp.click()
            inp.type("Test input", delay=80)
            time.sleep(0.3)
            return True, ""
    except Exception as e:
        return False, f"Type failed: {e}"

    return False, "No input field found"


# ============================================================
# STEP 6: MAIN LOOP
# ============================================================

session_id = f"agentic_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
logger = SessionLogger(session_id=session_id)
stuck_detector = StuckDetector(timeout_s=15)
session_start = time.time()

print(f"\n{'='*60}")
print(f"STARTING AGENTIC LOOP")
print(f"  Cluster: {cluster_id} — {cluster_meta.get('label', '?')}")
print(f"  Max steps: {args.max_steps}, Timeout: {args.max_duration}s")
print(f"{'='*60}\n")

consecutive_errors = 0
MAX_CONSECUTIVE_ERRORS = 3

for step in range(args.max_steps):
    elapsed = time.time() - session_start
    if elapsed > args.max_duration:
        print(f"\n  SESSION TIMEOUT ({args.max_duration}s)")
        break

    # Check browser alive
    try:
        current_url = page.url
    except Exception:
        print("  BROWSER CLOSED — aborting")
        break

    # Stuck detection
    if stuck_detector.check(current_url):
        print("  STUCK DETECTED — navigating home")
        logger.log_stuck_event("stuck", "Timeout", current_url)
        page.goto(target_url)
        time.sleep(2)
        stuck_detector.reset()
        consecutive_errors = 0
        continue

    # Build error context if recovering from failure
    error_ctx = ""
    if consecutive_errors > 0 and action_history:
        last = action_history[-1]
        error_ctx = last.get("error", "Previous action failed")

    print(f"\n--- Step {step + 1}/{args.max_steps} (elapsed: {elapsed:.1f}s) ---")

    # PLAN + PREDICT
    action_data = predict_action(error_context=error_ctx)
    action_name = action_data.get("action", "wait")
    target = action_data.get("target", "")
    reasoning = action_data.get("reasoning", "")

    # Update plan tracking
    current_plan = reasoning

    print(f"  Action:    {action_name} -> {target[:60]}")
    print(f"  Reasoning: {reasoning[:80]}")
    print(f"  Page:      {current_url[:60]}")

    logger.begin_action(action_name, page_url=current_url)

    # ACT
    success, error_msg = execute_action(action_data)

    if success:
        print(f"  Result:    OK")
        consecutive_errors = 0
        logger.end_action(success=True, details={"target": target, "reasoning": reasoning})
    else:
        print(f"  Result:    FAILED — {error_msg[:80]}")
        consecutive_errors += 1
        logger.end_action(success=False, error=error_msg, details={"target": target, "reasoning": reasoning})

    # Record in history
    action_history.append({
        "elapsed": round(elapsed, 1),
        "action": action_name,
        "target": target[:60],
        "success": success,
        "error": error_msg if not success else "",
        "page_url": current_url,
        "reasoning": reasoning,
    })

    stuck_detector.reset()

    # Recovery: too many consecutive errors → navigate home
    if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
        print(f"\n  {MAX_CONSECUTIVE_ERRORS} consecutive errors — recovering to homepage")
        page.goto(target_url)
        time.sleep(2)
        consecutive_errors = 0
        current_plan = ""


# ============================================================
# STEP 7: CLEANUP + REPORT
# ============================================================

total_elapsed = time.time() - session_start

print(f"\n{'='*60}")
print("AGENTIC LOOP COMPLETE")
print("=" * 60)

summary = logger.get_summary()
print(f"  Cluster:     {cluster_id} — {cluster_meta.get('label', '?')}")
print(f"  Duration:    {summary['total_duration_s']}s")
print(f"  Actions:     {summary['total_actions']} ({summary['successful_actions']} ok, {summary['failed_actions']} failed)")
print(f"  Completion:  {summary['completion_rate']*100:.0f}%")
print(f"  Stuck:       {summary['stuck_events']}")

# Save session log
log_path = args.output or str(AGENT_LOGS_DIR / f"{session_id}_log.json")
log_data = logger.to_dict()
log_data["cluster_id"] = cluster_id
log_data["cluster_label"] = cluster_meta.get("label", "?")
log_data["target_url"] = target_url
log_data["app_description"] = app_description
log_data["adapter_path"] = adapter_path

Path(log_path).parent.mkdir(parents=True, exist_ok=True)
with open(log_path, "w") as f:
    json.dump(log_data, f, indent=2)
print(f"  Log: {log_path}")

# Save action history
history_path = log_path.replace("_log.json", "_history.json")
with open(history_path, "w") as f:
    json.dump(action_history, f, indent=2)
print(f"  History: {history_path}")

browser.close()
pw.stop()
print("Browser closed.")

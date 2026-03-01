"""
============================================================
Agentic Loop — Observe → Predict → Execute → Repeat
============================================================
Uses the finetuned model to drive a browser session step-by-step.

Loop:
  1. Observe current page state (URL, visible elements via AgentQL)
  2. Build prompt: product features + action history + page state
  3. Model predicts next action JSON
  4. Execute action via AgentQL/Playwright
  5. If error → feed error back to model → model picks recovery action
  6. Repeat until max_steps reached or session_duration exceeded

Usage:
    # Run with finetuned adapter
    python agentic_loop.py --url http://localhost:3000 --max-steps 20

    # Run with base model (no adapter)
    python agentic_loop.py --url http://localhost:3000 --no-adapter

    # Override app description
    python agentic_loop.py --url http://localhost:3000 \
        --app-description "An e-commerce store with product grid, cart, and checkout"
============================================================
"""

import os
import sys
import json
import time
import random
import re
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    MISTRAL_API_KEY,
    TARGET_APP_URL,
    TARGET_APP_NAME,
    TARGET_APP_DESCRIPTION,
    MODELS_DIR,
    AGENT_LOGS_DIR,
    ensure_data_dirs,
    validate_sandbox_url,
)
from build_training_data import SYSTEM_PROMPT, build_product_context
from feedback.session_logger import SessionLogger, StuckDetector

# ============================================================
# PARSE ARGS
# ============================================================

parser = argparse.ArgumentParser(description="Run agentic browser loop with finetuned model")
parser.add_argument("--url", type=str, default=None, help="Target URL (overrides config)")
parser.add_argument("--app-name", type=str, default=None)
parser.add_argument("--app-description", type=str, default=None)
parser.add_argument("--adapter", type=str, default=None, help="LoRA adapter path (default: data/models/behavioral-lora)")
parser.add_argument("--no-adapter", action="store_true", help="Use base model without adapter")
parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-Instruct-v0.3")
parser.add_argument("--max-steps", type=int, default=20, help="Max actions to execute")
parser.add_argument("--max-duration", type=int, default=300, help="Max session duration in seconds")
parser.add_argument("--temperature", type=float, default=0.7)
parser.add_argument("--headless", action="store_true", help="Run browser headless")
parser.add_argument("--window-size", type=int, default=5, help="Recent action history window")
parser.add_argument("--output", type=str, default=None, help="Save session log to file")
args = parser.parse_args()

ensure_data_dirs()

target_url = args.url or TARGET_APP_URL
app_name = args.app_name or TARGET_APP_NAME
app_description = args.app_description or TARGET_APP_DESCRIPTION
adapter_path = args.adapter or str(MODELS_DIR / "behavioral-lora")

if not target_url:
    print("ERROR: No target URL. Set --url or TARGET_APP_URL in .env")
    sys.exit(1)

# ============================================================
# LOAD MODEL
# ============================================================

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

print("=" * 60)
print("AGENTIC LOOP — BEHAVIORAL BROWSER AGENT")
print("=" * 60)
print(f"Target:  {target_url}")
print(f"Product: {app_name}")
print(f"Steps:   {args.max_steps}, Duration: {args.max_duration}s")
print(f"Model:   {args.model}")
print(f"Adapter: {'none' if args.no_adapter else adapter_path}")
print("=" * 60)

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

if not args.no_adapter and os.path.exists(adapter_path):
    print(f"Loading LoRA adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()
    print("Adapter merged")
elif not args.no_adapter:
    print(f"WARNING: No adapter at {adapter_path}, using base model")

model.eval()
print("Model ready\n")

# ============================================================
# BROWSER SETUP
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
# HELPER FUNCTIONS
# ============================================================

product_context = build_product_context(app_name, target_url, app_description)
action_history = []  # list of dicts matching training data format


def observe_page() -> str:
    """Get current page state via AgentQL."""
    try:
        url = page.url
        title = page.title()
        return f"Current page: {url} ({title})"
    except Exception:
        return "Current page: unknown"


def build_prompt(error_context: str = "") -> list[dict]:
    """Build the chat messages for the model."""
    lines = [product_context, ""]

    elapsed = time.time() - session_start
    lines.append(f"Session: step {len(action_history)}, elapsed {elapsed:.1f}s / {args.max_duration}s")
    lines.append(observe_page())

    # Recent action history window
    window = action_history[-args.window_size:]
    if window:
        lines.append("\nRecent actions:")
        for a in window:
            status = "" if a.get("success", True) else " [FAILED]"
            line = f"  [{a['elapsed']:.1f}s] {a['action']} -> {a['target']}{status}"
            lines.append(line)

    if error_context:
        lines.append(f"\nLast action failed: {error_context}")
        lines.append("Decide how to recover.")
    else:
        lines.append("\nWhat would you do next?")

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(lines)},
    ]


def predict_action(error_context: str = "") -> dict:
    """Call the model to predict the next action."""
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
        match = re.search(r'\{[^}]+\}', raw)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        # Fallback: scroll down (safe default)
        print(f"  WARNING: Could not parse model output: {raw[:200]}")
        return {"action": "scroll", "target": "down 300px", "reasoning": "fallback", "hesitation_ms": 500}


def execute_action(action_data: dict) -> tuple[bool, str]:
    """Execute a predicted action in the browser. Returns (success, error_msg)."""
    action = action_data.get("action", "wait")
    target = action_data.get("target", "")
    hesitation = action_data.get("hesitation_ms", 0)

    # Apply hesitation
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
            print(f"  Unknown action: {action}")
            return False, f"Unknown action: {action}"
    except Exception as e:
        return False, str(e)


def _execute_click(target: str) -> tuple[bool, str]:
    """Execute a click action using AgentQL to find the element."""
    # Extract semantic description from target
    # Target format: 'tag "text"' or 'tag (class)' or just 'element description'
    semantic = target.strip()

    # Try AgentQL semantic query first
    try:
        query = f"""{{ target_element(description: "{semantic[:100]}") }}"""
        response = page.query_elements(query)
        if response.target_element:
            response.target_element.click()
            time.sleep(0.5)
            return True, ""
    except Exception:
        pass

    # Fallback: try Playwright text-based locator
    try:
        # Extract text from quotes if present
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

    # Final fallback: click at a reasonable position on the page
    try:
        page.mouse.click(random.randint(200, 800), random.randint(200, 600))
        time.sleep(0.5)
        return True, ""
    except Exception as e:
        return False, f"Click failed: {e}"


def _execute_scroll(target: str) -> tuple[bool, str]:
    """Execute a scroll action."""
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
    """Execute a type action using AgentQL to find the input."""
    semantic = target.strip()

    # Try AgentQL to find input
    try:
        query = f"""{{ input_field(description: "{semantic[:100]}") }}"""
        response = page.query_elements(query)
        if response.input_field:
            response.input_field.click()
            # Type some placeholder text
            sample_texts = [
                "Hello world", "Testing 123", "Great post!",
                "NYC is amazing", "Love this place", "Interesting...",
            ]
            response.input_field.type(random.choice(sample_texts), delay=random.randint(50, 120))
            time.sleep(0.3)
            return True, ""
    except Exception:
        pass

    # Fallback: find any visible input
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
# MAIN LOOP
# ============================================================

session_id = f"agentic_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
logger = SessionLogger(session_id=session_id)
stuck_detector = StuckDetector(timeout_s=15)
session_start = time.time()

print(f"\n{'='*60}")
print(f"STARTING AGENTIC LOOP — max {args.max_steps} steps, {args.max_duration}s")
print(f"{'='*60}\n")

consecutive_errors = 0
MAX_CONSECUTIVE_ERRORS = 3

for step in range(args.max_steps):
    elapsed = time.time() - session_start
    if elapsed > args.max_duration:
        print(f"\n  SESSION TIMEOUT ({args.max_duration}s)")
        break

    # Check stuck
    try:
        current_url = page.url
    except Exception:
        print("  BROWSER CLOSED — aborting")
        break

    if stuck_detector.check(current_url):
        print("  STUCK DETECTED — navigating home")
        logger.log_stuck_event("stuck", "Timeout", current_url)
        page.goto(target_url)
        time.sleep(2)
        stuck_detector.reset()
        consecutive_errors = 0
        continue

    # Predict next action
    error_ctx = ""
    if consecutive_errors > 0 and action_history:
        last = action_history[-1]
        error_ctx = last.get("error", "Previous action failed")

    print(f"\n--- Step {step + 1}/{args.max_steps} (elapsed: {elapsed:.1f}s) ---")

    action_data = predict_action(error_context=error_ctx)
    action_name = action_data.get("action", "wait")
    target = action_data.get("target", "")
    reasoning = action_data.get("reasoning", "")

    print(f"  Predicted: {action_name} -> {target[:60]}")
    print(f"  Reasoning: {reasoning[:80]}")

    logger.begin_action(action_name, page_url=current_url)

    # Execute action
    success, error_msg = execute_action(action_data)

    if success:
        print(f"  OK")
        consecutive_errors = 0
        logger.end_action(success=True, details={"target": target})
    else:
        print(f"  FAILED: {error_msg[:80]}")
        consecutive_errors += 1
        logger.end_action(success=False, error=error_msg, details={"target": target})

    # Record in history
    action_history.append({
        "elapsed": round(elapsed, 1),
        "action": action_name,
        "target": target[:60],
        "success": success,
        "error": error_msg if not success else "",
    })

    stuck_detector.reset()

    # Bail if too many consecutive errors
    if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
        print(f"\n  {MAX_CONSECUTIVE_ERRORS} consecutive errors — recovering to homepage")
        page.goto(target_url)
        time.sleep(2)
        consecutive_errors = 0

# ============================================================
# CLEANUP
# ============================================================

print(f"\n{'='*60}")
print("AGENTIC LOOP COMPLETE")
print("=" * 60)

summary = logger.get_summary()
print(f"  Duration: {summary['total_duration_s']}s")
print(f"  Actions:  {summary['total_actions']} ({summary['successful_actions']} ok, {summary['failed_actions']} failed)")
print(f"  Stuck events: {summary['stuck_events']}")
print(f"  Sequence: {summary['action_sequence_actual']}")

# Save log
log_path = args.output or str(AGENT_LOGS_DIR / f"{session_id}_log.json")
logger.save(log_path)

# Also save action history
history_path = log_path.replace("_log.json", "_history.json")
with open(history_path, "w") as f:
    json.dump(action_history, f, indent=2)
print(f"  History: {history_path}")

browser.close()
pw.stop()
print("Browser closed.")

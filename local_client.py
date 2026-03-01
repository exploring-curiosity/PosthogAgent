"""
============================================================
Local Browser Client — runs on your machine (no GPU needed)
============================================================
Drives a browser via AgentQL + Playwright against a target URL.
Calls the remote GPU service (service.py on VM) for action predictions.
Runs ALL cluster models sequentially, each gets its own browser session.

Closed feedback loop:
  1. Observe page state (AgentQL)
  2. Send state to VM → get predicted action
  3. Execute action in browser
  4. If error → send error context back to VM → get recovery action
  5. Retry up to N times per action before moving on
  6. Repeat until max_steps or task completed

Usage:
    python local_client.py \
        --vm-url http://<VM_IP>:8000 \
        --target-url https://fun-city-xi.vercel.app \
        --description "Sign up for an account and comment on the top thread" \
        --max-steps 20

    # Run only specific clusters
    python local_client.py \
        --vm-url http://<VM_IP>:8000 \
        --target-url https://fun-city-xi.vercel.app \
        --description "Browse posts and upvote content" \
        --clusters 0 2 4
============================================================
"""

import argparse
import json
import os
import re
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

# ============================================================
# ARGS
# ============================================================

parser = argparse.ArgumentParser(description="Local browser client — calls VM GPU service for action predictions")
parser.add_argument("--vm-url", type=str, required=True, help="URL of the GPU inference service (e.g. http://VM_IP:8000)")
parser.add_argument("--target-url", type=str, required=True, help="Target website URL to evaluate")
parser.add_argument("--app-name", type=str, default="FunCity", help="App name")
parser.add_argument("--description", type=str, required=True, help="Task description: what the agent should do")
parser.add_argument("--clusters", type=int, nargs="+", default=None, help="Specific cluster IDs (default: all)")
parser.add_argument("--max-steps", type=int, default=20, help="Max actions per cluster session")
parser.add_argument("--max-duration", type=int, default=300, help="Max session duration in seconds")
parser.add_argument("--max-retries", type=int, default=3, help="Max retries per failed action")
parser.add_argument("--temperature", type=float, default=0.7)
parser.add_argument("--headless", action="store_true", help="Run browser headless")
parser.add_argument("--api-key", type=str, default="", help="API key for VM service (if SERVICE_API_KEY is set on VM)")
parser.add_argument("--output-dir", type=str, default="eval_results", help="Directory for result files")
args = parser.parse_args()

VM_URL = args.vm_url.rstrip("/")
OUTPUT_DIR = Path(args.output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Build auth headers
VM_HEADERS = {"Content-Type": "application/json"}
if args.api_key:
    VM_HEADERS["Authorization"] = f"Bearer {args.api_key}"


# ============================================================
# VM API CLIENT
# ============================================================

def vm_health() -> dict:
    """Check VM service health."""
    resp = requests.get(f"{VM_URL}/health", headers=VM_HEADERS, timeout=10)
    resp.raise_for_status()
    return resp.json()


def vm_get_clusters() -> list[dict]:
    """Get available clusters from VM."""
    resp = requests.get(f"{VM_URL}/clusters", headers=VM_HEADERS, timeout=10)
    resp.raise_for_status()
    return resp.json()


def vm_predict(cluster_id: int, page_state: str, action_history: list,
               error_context: str, step_number: int, elapsed_s: float) -> dict:
    """Call VM to predict next action."""
    payload = {
        "cluster_id": cluster_id,
        "page_state": page_state,
        "app_name": args.app_name,
        "app_url": args.target_url,
        "app_description": args.description,
        "action_history": action_history[-5:],  # last 5 actions
        "error_context": error_context,
        "step_number": step_number,
        "elapsed_s": elapsed_s,
        "max_duration_s": float(args.max_duration),
        "temperature": args.temperature,
        "window_size": 5,
    }
    resp = requests.post(f"{VM_URL}/predict", json=payload, headers=VM_HEADERS, timeout=120)
    resp.raise_for_status()
    return resp.json()


# ============================================================
# PAGE OBSERVATION (AgentQL)
# ============================================================

def observe_page(page) -> str:
    """Get current page state via AgentQL."""
    try:
        url = page.url
        title = page.title()

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


# ============================================================
# ACTION EXECUTION
# ============================================================

def execute_action(page, action_data: dict, home_url: str) -> tuple[bool, str]:
    """Execute a predicted action in the browser. Returns (success, error_msg)."""
    action = action_data.get("action", "wait")
    target = action_data.get("target", "")
    hesitation = action_data.get("hesitation_ms", 0)

    if hesitation > 0:
        time.sleep(min(hesitation / 1000.0, 5.0))

    try:
        if action == "click":
            return _click(page, target)
        elif action == "scroll":
            return _scroll(page, target)
        elif action == "type":
            return _type(page, target)
        elif action == "wait":
            time.sleep(random.uniform(0.5, 2.0))
            return True, ""
        elif action == "navigate_back":
            page.go_back()
            time.sleep(1)
            return True, ""
        elif action == "navigate_to":
            return _navigate_to(page, target, home_url)
        elif action == "press_enter":
            page.keyboard.press("Enter")
            time.sleep(1)
            return True, ""
        elif action == "select":
            return _click(page, target)  # treat select as click
        else:
            return False, f"Unknown action: {action}"
    except Exception as e:
        return False, str(e)


def _click(page, target: str) -> tuple[bool, str]:
    semantic = target.strip()

    # AgentQL semantic query
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

    # Playwright text fallback
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

    # Tag-based fallback
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


def _scroll(page, target: str) -> tuple[bool, str]:
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


def _navigate_to(page, target: str, home_url: str) -> tuple[bool, str]:
    """Navigate to a URL, but only if it's on the same domain as the target app."""
    from urllib.parse import urlparse
    url = target.strip()

    # If it's a relative path, make it absolute
    if url.startswith("/"):
        parsed_home = urlparse(home_url)
        url = f"{parsed_home.scheme}://{parsed_home.netloc}{url}"

    # Only allow same-domain navigation
    try:
        parsed_target = urlparse(url)
        parsed_home = urlparse(home_url)
        if parsed_target.netloc and parsed_target.netloc != parsed_home.netloc:
            # Redirect to home instead of external URL
            print(f"    Blocked external nav to {parsed_target.netloc}, going home instead")
            page.goto(home_url)
            time.sleep(2)
            return True, ""
    except Exception:
        pass

    try:
        page.goto(url)
        time.sleep(2)
        return True, ""
    except Exception as e:
        return False, f"Navigate failed: {e}"


def _type(page, target: str) -> tuple[bool, str]:
    semantic = target.strip()

    # Extract text to type if format is "field description: text to type"
    text_to_type = None
    if ":" in semantic:
        parts = semantic.split(":", 1)
        semantic = parts[0].strip()
        text_to_type = parts[1].strip()

    # AgentQL
    try:
        escaped = semantic[:100].replace('"', '\\"')
        query = f"""{{ input_field(description: "{escaped}") }}"""
        response = page.query_elements(query)
        if response.input_field:
            response.input_field.click()
            if text_to_type:
                response.input_field.type(text_to_type, delay=random.randint(50, 120))
            else:
                sample_texts = [
                    "Hello world", "Testing 123", "Great post!",
                    "This is interesting", "Love this place", "Check this out",
                ]
                response.input_field.type(random.choice(sample_texts), delay=random.randint(50, 120))
            time.sleep(0.3)
            return True, ""
    except Exception:
        pass

    # Playwright fallback
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
# RUN ONE CLUSTER SESSION
# ============================================================

def run_cluster_session(cluster_id: int, cluster_label: str) -> dict:
    """Run a full agentic session for one cluster model."""
    import agentql
    from playwright.sync_api import sync_playwright

    print(f"\n{'='*60}")
    print(f"CLUSTER {cluster_id}: {cluster_label}")
    print(f"  Target: {args.target_url}")
    print(f"  Task:   {args.description[:80]}")
    print(f"  Max steps: {args.max_steps}, Retries: {args.max_retries}")
    print(f"{'='*60}")

    # Launch browser
    pw = sync_playwright().start()
    browser = pw.chromium.launch(headless=args.headless)
    context = browser.new_context(viewport={"width": 1512, "height": 823})
    page = agentql.wrap(context.new_page())

    try:
        page.goto(args.target_url)
        time.sleep(2)
    except Exception as e:
        browser.close()
        pw.stop()
        return {
            "cluster_id": cluster_id,
            "cluster_label": cluster_label,
            "status": "error",
            "error": f"Failed to navigate: {e}",
            "actions": [],
        }

    action_history = []
    session_start = time.time()
    last_url = args.target_url
    consecutive_errors = 0

    for step in range(args.max_steps):
        elapsed = time.time() - session_start
        if elapsed > args.max_duration:
            print(f"\n  TIMEOUT ({args.max_duration}s)")
            break

        # Check browser alive
        try:
            current_url = page.url
        except Exception:
            print(f"\n  BROWSER CLOSED")
            break

        # Observe
        page_state = observe_page(page)

        # Build error context from last failure
        error_ctx = ""
        if consecutive_errors > 0 and action_history:
            error_ctx = action_history[-1].get("error", "Previous action failed")

        # Call VM for prediction
        print(f"\n  Step {step+1}/{args.max_steps} (elapsed: {elapsed:.1f}s)")
        try:
            prediction = vm_predict(
                cluster_id=cluster_id,
                page_state=page_state,
                action_history=action_history,
                error_context=error_ctx,
                step_number=step,
                elapsed_s=elapsed,
            )
        except requests.exceptions.HTTPError as e:
            detail = ""
            try:
                detail = e.response.json().get("detail", "")
            except Exception:
                detail = e.response.text[:200] if e.response else ""
            print(f"    VM ERROR: {e}")
            if detail:
                print(f"    Detail:  {detail[:120]}")
            action_history.append({
                "elapsed": round(elapsed, 1),
                "action": "vm_error",
                "target": "",
                "success": False,
                "error": str(e),
                "page_url": current_url,
                "reasoning": "",
            })
            consecutive_errors += 1
            if consecutive_errors >= args.max_retries:
                print(f"    {args.max_retries} consecutive VM errors — aborting cluster")
                break
            continue

        action_name = prediction.get("action", "wait")
        target = prediction.get("target", "")
        reasoning = prediction.get("reasoning", "")
        hesitation = prediction.get("hesitation_ms", 0)

        print(f"    Action:    {action_name} → {target[:50]}")
        print(f"    Reasoning: {reasoning[:60]}")

        # Execute with retry loop
        success = False
        error_msg = ""

        for attempt in range(args.max_retries):
            success, error_msg = execute_action(page, prediction, args.target_url)

            if success:
                print(f"    Result:    OK" + (f" (attempt {attempt+1})" if attempt > 0 else ""))
                consecutive_errors = 0
                break
            else:
                print(f"    FAILED (attempt {attempt+1}/{args.max_retries}): {error_msg[:50]}")

                if attempt < args.max_retries - 1:
                    # Ask VM for recovery action with error context
                    try:
                        recovery = vm_predict(
                            cluster_id=cluster_id,
                            page_state=observe_page(page),
                            action_history=action_history,
                            error_context=f"{action_name} on '{target[:40]}' failed: {error_msg[:60]}",
                            step_number=step,
                            elapsed_s=time.time() - session_start,
                        )
                        prediction = recovery
                        action_name = recovery.get("action", "wait")
                        target = recovery.get("target", "")
                        reasoning = recovery.get("reasoning", "")
                        print(f"    Recovery:  {action_name} → {target[:50]} ({reasoning[:40]})")
                    except Exception as e:
                        print(f"    Recovery VM error: {e}")
                        break

        if not success:
            consecutive_errors += 1
        else:
            consecutive_errors = 0

        action_history.append({
            "elapsed": round(elapsed, 1),
            "action": action_name,
            "target": target[:60],
            "success": success,
            "error": error_msg if not success else "",
            "page_url": current_url,
            "reasoning": reasoning,
        })

        # Too many consecutive failures → navigate home
        if consecutive_errors >= args.max_retries:
            print(f"\n    {args.max_retries} consecutive failures — navigating home")
            try:
                page.goto(args.target_url)
                time.sleep(2)
            except Exception:
                break
            consecutive_errors = 0

    # Cleanup
    total_elapsed = time.time() - session_start
    try:
        browser.close()
        pw.stop()
    except Exception:
        pass

    total_actions = len(action_history)
    successful = sum(1 for a in action_history if a.get("success"))
    failed = total_actions - successful

    result = {
        "cluster_id": cluster_id,
        "cluster_label": cluster_label,
        "status": "completed",
        "total_steps": total_actions,
        "successful_actions": successful,
        "failed_actions": failed,
        "completion_rate": round(successful / total_actions, 2) if total_actions > 0 else 0,
        "duration_s": round(total_elapsed, 1),
        "actions": action_history,
    }

    print(f"\n  Cluster {cluster_id} done: {successful}/{total_actions} actions ok, {total_elapsed:.0f}s")
    return result


# ============================================================
# MAIN
# ============================================================

def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 1. Check VM is alive
    print(f"\n{'='*60}")
    print("AGENTIC EVALUATION CLIENT")
    print(f"{'='*60}")
    print(f"VM:      {VM_URL}")
    print(f"Target:  {args.target_url}")
    print(f"Task:    {args.description}")
    print(f"{'='*60}\n")

    print("Checking VM health...")
    try:
        health = vm_health()
        print(f"  VM status: {health['status']}")
        print(f"  Model loaded: {health['model_loaded']}")
        print(f"  Available clusters: {health['available_clusters']}")
    except Exception as e:
        print(f"  ERROR: Cannot reach VM at {VM_URL}")
        print(f"  {e}")
        print(f"\n  Make sure service.py is running on the VM:")
        print(f"    python service.py")
        sys.exit(1)

    # 2. Get cluster list
    try:
        clusters = vm_get_clusters()
    except Exception as e:
        print(f"  ERROR: Cannot get clusters: {e}")
        sys.exit(1)

    available = [c for c in clusters if c.get("has_adapter")]
    if args.clusters:
        available = [c for c in available if c["id"] in args.clusters]

    if not available:
        print("ERROR: No clusters with trained adapters found")
        sys.exit(1)

    print(f"\nRunning {len(available)} cluster(s):")
    for c in available:
        print(f"  Cluster {c['id']}: {c['label']}")

    # 3. Run each cluster sequentially
    all_results = {}

    for cluster_info in available:
        cid = cluster_info["id"]
        label = cluster_info.get("label", f"Cluster {cid}")

        result = run_cluster_session(cid, label)
        all_results[cid] = result

    # 4. Summary
    print(f"\n\n{'='*60}")
    print("EVALUATION COMPLETE")
    print(f"{'='*60}")

    total_actions = 0
    total_success = 0
    for cid, r in all_results.items():
        total_actions += r.get("total_steps", 0)
        total_success += r.get("successful_actions", 0)
        rate = r.get("completion_rate", 0)
        print(f"  Cluster {cid} ({r.get('cluster_label', '?')}): "
              f"{r.get('successful_actions', 0)}/{r.get('total_steps', 0)} ok "
              f"({rate*100:.0f}%), {r.get('duration_s', 0):.0f}s")

    overall_rate = round(total_success / total_actions, 2) if total_actions > 0 else 0
    print(f"\n  Overall: {total_success}/{total_actions} ({overall_rate*100:.0f}%)")

    # 5. Save results
    result_path = OUTPUT_DIR / f"eval_{timestamp}.json"
    full_report = {
        "timestamp": timestamp,
        "target_url": args.target_url,
        "task_description": args.description,
        "app_name": args.app_name,
        "vm_url": VM_URL,
        "max_steps": args.max_steps,
        "max_retries": args.max_retries,
        "overall": {
            "total_actions": total_actions,
            "total_successful": total_success,
            "success_rate": overall_rate,
        },
        "cluster_results": {str(k): v for k, v in all_results.items()},
    }

    with open(result_path, "w") as f:
        json.dump(full_report, f, indent=2)
    print(f"\n  Results saved to {result_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

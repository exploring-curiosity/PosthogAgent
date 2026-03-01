"""
============================================================
AgentRunner — Reusable agentic loop engine
============================================================
Encapsulates the plan → observe → act → replan loop into a class
that can be called programmatically (from a service, script, etc.)

Loads the base model once, swaps LoRA adapters per cluster,
and runs browser sessions via AgentQL + Playwright.
============================================================
"""

import os
import sys
import json
import time
import random
import re
import gc
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    MISTRAL_API_KEY,
    MODELS_DIR,
    CLUSTERS_DIR,
    AGENT_LOGS_DIR,
    ensure_data_dirs,
)
from build_training_data import (
    BASE_SYSTEM_PROMPT,
    build_cluster_system_prompt,
    build_product_context,
)
from feedback.session_logger import SessionLogger, StuckDetector


# ============================================================
# MODEL MANAGER — loads base model once, swaps adapters
# ============================================================

class ModelManager:
    """Loads the base LLM once and can hot-swap LoRA adapters per cluster."""

    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.3"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.torch = torch
        self._current_adapter: str | None = None

        print(f"[ModelManager] Loading base model: {model_name}")
        print(f"  Mode: Full bf16 (A100 optimized)")

        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
            print(f"  Using Flash Attention 2")
        except ImportError:
            attn_impl = "sdpa"
            print(f"  Using SDPA")

        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_impl,
            trust_remote_code=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = self.base_model
        self.model.eval()
        print(f"[ModelManager] Base model ready")

    def load_adapter(self, adapter_path: str) -> bool:
        """Load a LoRA adapter on top of the base model. Returns True if successful.
        
        Uses PeftModel directly (no merge_and_unload) to preserve base model
        weights for subsequent adapter swaps.
        """
        from peft import PeftModel

        if self._current_adapter == adapter_path:
            return True

        if not os.path.exists(adapter_path):
            print(f"[ModelManager] WARNING: adapter not found at {adapter_path}")
            self.model = self.base_model
            self._current_adapter = None
            return False

        print(f"[ModelManager] Loading adapter: {adapter_path}")

        # Unload previous adapter — release PeftModel, revert to base
        if self._current_adapter is not None and self.model is not self.base_model:
            del self.model
            gc.collect()
            self.torch.cuda.empty_cache()

        # Wrap base model with new adapter (do NOT merge — preserves base weights)
        self.model = PeftModel.from_pretrained(
            self.base_model, adapter_path, torch_dtype=self.torch.bfloat16
        )
        self.model.eval()
        self._current_adapter = adapter_path
        print(f"[ModelManager] Adapter loaded (PeftModel, no merge)")
        return True

    def generate(self, messages: list[dict], max_new_tokens: int = 256,
                 temperature: float = 0.7) -> str:
        """Generate text from chat messages."""
        input_ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt",
        ).to(self.model.device)
        attention_mask = self.torch.ones_like(input_ids)

        with self.torch.no_grad():
            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pad_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
            )

        return self.tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)


# ============================================================
# CLUSTER UTILITIES
# ============================================================

def load_all_cluster_metas() -> list[dict]:
    """Load all cluster metadata from clusters.json."""
    clusters_path = CLUSTERS_DIR / "clusters.json"
    if not clusters_path.exists():
        return []
    with open(clusters_path) as f:
        data = json.load(f)
    return data.get("clusters", [])


def get_cluster_meta(cluster_id: int) -> dict:
    """Load a single cluster's metadata."""
    for c in load_all_cluster_metas():
        if c["id"] == cluster_id:
            return c
    return {"id": cluster_id, "label": f"Cluster {cluster_id}"}


def get_available_cluster_ids() -> list[int]:
    """Return cluster IDs that have a trained adapter."""
    ids = []
    for d in sorted(MODELS_DIR.iterdir()):
        if d.is_dir() and d.name.startswith("cluster_") and d.name.endswith("_lora"):
            try:
                cid = int(d.name.split("_")[1])
                ids.append(cid)
            except (IndexError, ValueError):
                pass
    return ids


# ============================================================
# AGENT RUNNER — runs one agentic session for a given cluster
# ============================================================

class AgentRunner:
    """
    Runs one agentic browser session using a specific cluster's model.
    
    Flow per session:
      1. Load cluster adapter
      2. Launch browser → navigate to URL
      3. Loop: observe → plan → act → replan on error
      4. Return results dict with action log, success rate, timings
    """

    def __init__(self, model_manager: ModelManager,
                 max_steps: int = 20, max_duration: int = 300,
                 temperature: float = 0.7, window_size: int = 5,
                 headless: bool = True):
        self.mm = model_manager
        self.max_steps = max_steps
        self.max_duration = max_duration
        self.temperature = temperature
        self.window_size = window_size
        self.headless = headless

    def run_session(self, url: str, app_name: str, app_description: str,
                    cluster_id: int, cluster_meta: dict) -> dict:
        """Run a full agentic session for one cluster. Returns results dict."""
        import agentql
        from playwright.sync_api import sync_playwright

        adapter_path = str(MODELS_DIR / f"cluster_{cluster_id}_lora")
        self.mm.load_adapter(adapter_path)

        product_context = build_product_context(app_name, url, app_description)
        system_prompt = build_cluster_system_prompt(cluster_meta)

        session_id = f"eval_c{cluster_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger = SessionLogger(session_id=session_id)
        stuck_detector = StuckDetector(timeout_s=15)
        action_history = []
        consecutive_errors = 0
        MAX_CONSECUTIVE_ERRORS = 3

        # Launch browser
        print(f"\n  [Cluster {cluster_id}] Launching browser → {url}")
        pw = sync_playwright().start()
        browser = pw.chromium.launch(headless=self.headless)
        context = browser.new_context(viewport={"width": 1512, "height": 823})
        page = agentql.wrap(context.new_page())

        try:
            page.goto(url)
            time.sleep(2)
        except Exception as e:
            browser.close()
            pw.stop()
            return {
                "cluster_id": cluster_id,
                "cluster_label": cluster_meta.get("label", "?"),
                "status": "error",
                "error": f"Failed to navigate: {e}",
                "actions": [],
                "summary": {},
            }

        session_start = time.time()

        for step in range(self.max_steps):
            elapsed = time.time() - session_start
            if elapsed > self.max_duration:
                print(f"    Step {step+1}: TIMEOUT ({self.max_duration}s)")
                break

            try:
                current_url = page.url
            except Exception:
                print(f"    Step {step+1}: BROWSER CLOSED")
                break

            # Stuck detection
            if stuck_detector.check(current_url):
                print(f"    Step {step+1}: STUCK → home")
                logger.log_stuck_event("stuck", "Timeout", current_url)
                try:
                    page.goto(url)
                    time.sleep(2)
                except Exception:
                    break
                stuck_detector.reset()
                consecutive_errors = 0
                continue

            # Build error context
            error_ctx = ""
            if consecutive_errors > 0 and action_history:
                error_ctx = action_history[-1].get("error", "Previous action failed")

            # Observe
            page_state = self._observe_page(page)

            # Build prompt
            messages = self._build_prompt(
                system_prompt, product_context, page_state,
                action_history, elapsed, error_ctx
            )

            # Predict
            raw = self.mm.generate(messages, temperature=self.temperature)
            action_data = self._parse_action(raw)

            action_name = action_data.get("action", "wait")
            target = action_data.get("target", "")
            reasoning = action_data.get("reasoning", "")

            print(f"    Step {step+1}: {action_name} → {target[:50]}  ({reasoning[:50]})")

            logger.begin_action(action_name, page_url=current_url)

            # Execute
            success, error_msg = self._execute_action(page, action_data, url)

            if success:
                consecutive_errors = 0
                logger.end_action(success=True, details={"target": target, "reasoning": reasoning})
            else:
                consecutive_errors += 1
                logger.end_action(success=False, error=error_msg, details={"target": target, "reasoning": reasoning})
                print(f"      FAILED: {error_msg[:60]}")

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

            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                print(f"    {MAX_CONSECUTIVE_ERRORS} consecutive errors → home")
                try:
                    page.goto(url)
                    time.sleep(2)
                except Exception:
                    break
                consecutive_errors = 0

        # Cleanup
        try:
            browser.close()
            pw.stop()
        except Exception:
            pass

        summary = logger.get_summary()

        # Save log
        ensure_data_dirs()
        log_path = str(AGENT_LOGS_DIR / f"{session_id}_log.json")
        log_data = logger.to_dict()
        log_data["cluster_id"] = cluster_id
        log_data["cluster_label"] = cluster_meta.get("label", "?")
        log_data["target_url"] = url
        log_data["app_description"] = app_description

        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=2)

        history_path = log_path.replace("_log.json", "_history.json")
        with open(history_path, "w") as f:
            json.dump(action_history, f, indent=2)

        return {
            "cluster_id": cluster_id,
            "cluster_label": cluster_meta.get("label", "?"),
            "status": "completed",
            "session_id": session_id,
            "total_steps": len(action_history),
            "successful_actions": summary.get("successful_actions", 0),
            "failed_actions": summary.get("failed_actions", 0),
            "completion_rate": summary.get("completion_rate", 0),
            "duration_s": summary.get("total_duration_s", 0),
            "stuck_events": summary.get("stuck_events", 0),
            "actions": action_history,
            "log_path": log_path,
        }

    # ── Page observation ──────────────────────────────────────

    def _observe_page(self, page) -> str:
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

    # ── Prompt building ───────────────────────────────────────

    def _build_prompt(self, system_prompt: str, product_context: str,
                      page_state: str, action_history: list,
                      elapsed: float, error_context: str) -> list[dict]:
        """Build chat messages for the model."""
        lines = [product_context, ""]
        lines.append(f"Session: step {len(action_history)}, elapsed {elapsed:.1f}s / {self.max_duration}s")
        lines.append(f"\n{page_state}")

        window = action_history[-self.window_size:]
        if window:
            lines.append("\nRecent actions:")
            for a in window:
                status = "" if a.get("success", True) else " [FAILED]"
                line = f"  [{a['elapsed']:.1f}s] {a['action']} -> {a['target']}{status}"
                if not a.get("success", True) and a.get("error"):
                    line += f" ({a['error'][:40]})"
                lines.append(line)

        if error_context:
            lines.append(f"\nLast action failed: {error_context}")
            lines.append("Decide how to recover or try an alternative approach.")
        else:
            lines.append("\nWhat would you do next?")

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "\n".join(lines)},
        ]

    # ── JSON parsing ──────────────────────────────────────────

    def _parse_action(self, raw: str) -> dict:
        """Parse model output into action dict."""
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r'\{[^{}]*\}', raw, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
            return {
                "action": "scroll", "target": "down 300px",
                "reasoning": "fallback — unparseable output",
                "hesitation_ms": 500,
            }

    # ── Action execution ──────────────────────────────────────

    def _execute_action(self, page, action_data: dict, home_url: str) -> tuple[bool, str]:
        """Execute a predicted action. Returns (success, error_msg)."""
        action = action_data.get("action", "wait")
        target = action_data.get("target", "")
        hesitation = action_data.get("hesitation_ms", 0)

        if hesitation > 0:
            time.sleep(min(hesitation / 1000.0, 5.0))

        try:
            if action == "click":
                return self._click(page, target)
            elif action == "scroll":
                return self._scroll(page, target)
            elif action == "type":
                return self._type(page, target)
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

    def _click(self, page, target: str) -> tuple[bool, str]:
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

    def _scroll(self, page, target: str) -> tuple[bool, str]:
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

    def _type(self, page, target: str) -> tuple[bool, str]:
        semantic = target.strip()

        # AgentQL
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

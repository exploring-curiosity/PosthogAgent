"""
Stage 5 (Explore): Autonomous Exploratory Agent — Uses a fine-tuned Mistral model
to make real-time decisions about what to do next on any web application.

Unlike stage5_execute.py which follows a fixed action sequence, this agent
uses its demographic-specific fine-tuned model to decide each action based
on the current page state.

SAFETY: Inherits production URL blocking from config.py.
"""

import json
import random
import time
from pathlib import Path
from typing import Optional

import agentql
from playwright.sync_api import sync_playwright
from mistralai import Mistral


class ExploratoryAgent:
    def __init__(
        self,
        model_id: str,
        demographic: dict,
        target_url: str,
        mistral_api_key: str,
        session_logger=None,
        max_steps: int = 25,
        max_duration_s: int = 180,
    ):
        self.model_id = model_id
        self.demographic = demographic
        self.target_url = target_url
        self.mistral_api_key = mistral_api_key
        self.logger = session_logger
        self.max_steps = max_steps
        self.max_duration_s = max_duration_s

        self.client = Mistral(api_key=mistral_api_key)
        self.p = None
        self.browser = None
        self.page = None

        self.action_history: list[dict] = []
        self.impressions: list[dict] = []
        self.start_time: float = 0
        self.step_count: int = 0

    # ─── BROWSER LIFECYCLE ───

    def start(self):
        """Launch browser and navigate to the target app."""
        self.p = sync_playwright().start()
        self.browser = self.p.chromium.launch(headless=False)
        context = self.browser.new_context(
            viewport={"width": 1512, "height": 823},
        )
        self.page = agentql.wrap(context.new_page())
        self.page.goto(self.target_url)
        self.start_time = time.time()
        self._wait(2)

    def stop(self):
        """Close the browser."""
        if self.browser:
            try:
                self.browser.close()
            except Exception:
                pass
        if self.p:
            try:
                self.p.stop()
            except Exception:
                pass

    def _is_browser_alive(self) -> bool:
        """Check if the browser/page is still usable."""
        try:
            self.page.url
            return True
        except Exception:
            return False

    def _wait(self, base_seconds: float):
        """Wait with slight randomization."""
        actual = base_seconds * random.uniform(0.8, 1.2)
        time.sleep(actual)

    # ─── STATE OBSERVATION ───

    def _observe_page(self) -> dict:
        """Capture the current state of the page for the model."""
        try:
            url = self.page.url
        except Exception:
            return {"url": "unknown", "error": "page not accessible"}

        state = {
            "url": url,
            "title": "",
            "visible_elements": [],
            "scroll_position": 0,
        }

        try:
            state["title"] = self.page.title()
        except Exception:
            pass

        # Get visible interactive elements via AgentQL
        try:
            response = self.page.query_elements("""
            {
                navigation_links[] {
                    link_text
                }
                buttons[] {
                    button_text
                }
                post_titles[] {
                    title_text
                }
                input_fields[] {
                    placeholder_text
                }
            }
            """)

            if response.navigation_links:
                for link in response.navigation_links[:10]:
                    try:
                        text = link.link_text.text_content() if link.link_text else ""
                        if text.strip():
                            state["visible_elements"].append(f"link: {text.strip()[:50]}")
                    except Exception:
                        pass

            if response.buttons:
                for btn in response.buttons[:10]:
                    try:
                        text = btn.button_text.text_content() if btn.button_text else ""
                        if text.strip():
                            state["visible_elements"].append(f"button: {text.strip()[:50]}")
                    except Exception:
                        pass

            if response.post_titles:
                for post in response.post_titles[:8]:
                    try:
                        text = post.title_text.text_content() if post.title_text else ""
                        if text.strip():
                            state["visible_elements"].append(f"post: {text.strip()[:60]}")
                    except Exception:
                        pass

            if response.input_fields:
                for inp in response.input_fields[:5]:
                    try:
                        text = inp.placeholder_text.text_content() if inp.placeholder_text else ""
                        if text.strip():
                            state["visible_elements"].append(f"input: {text.strip()[:50]}")
                    except Exception:
                        pass

        except Exception as e:
            state["observation_error"] = str(e)[:100]

        # Get scroll position
        try:
            state["scroll_position"] = self.page.evaluate("window.scrollY")
        except Exception:
            pass

        return state

    def _build_context(self, page_state: dict) -> str:
        """Build the user message for the fine-tuned model."""
        elapsed = time.time() - self.start_time
        lines = []

        lines.append(f"Session progress: {self.step_count}/{self.max_steps} steps, "
                     f"elapsed: {elapsed:.1f}s / {self.max_duration_s}s total")
        lines.append(f"Current page: {page_state.get('url', 'unknown')}")
        lines.append(f"Page title: {page_state.get('title', 'unknown')}")

        elements = page_state.get("visible_elements", [])
        if elements:
            lines.append(f"\nVisible elements ({len(elements)}):")
            for el in elements[:15]:
                lines.append(f"  - {el}")

        if self.action_history:
            recent = self.action_history[-5:]
            lines.append(f"\nRecent actions:")
            for a in recent:
                lines.append(f"  [{a.get('elapsed_s', 0):.1f}s] {a.get('action', '?')} on {a.get('target', '?')}")

        lines.append("\nWhat would you do next?")
        return "\n".join(lines)

    # ─── DECISION MAKING ───

    def _decide_next_action(self, page_state: dict) -> dict:
        """Ask the fine-tuned model what to do next."""
        system_prompt = self._build_system_prompt()
        user_message = self._build_context(page_state)

        try:
            response = self.client.chat.complete(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                response_format={"type": "json_object"},
            )

            decision = json.loads(response.choices[0].message.content)

            # Validate required fields
            if "action" not in decision:
                decision["action"] = "scroll"
            if "target" not in decision:
                decision["target"] = "down"

            return decision

        except Exception as e:
            print(f"  Model decision failed (attempt 1): {e}")
            # Retry once with the fine-tuned model
            try:
                response = self.client.chat.complete(
                    model=self.model_id,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                )
                decision = json.loads(response.choices[0].message.content)
                if "action" not in decision:
                    decision["action"] = "scroll"
                if "target" not in decision:
                    decision["target"] = "down"
                return decision
            except Exception as e2:
                print(f"  Model decision failed (attempt 2): {e2}")
                return {
                    "action": "scroll",
                    "target": "down",
                    "reasoning": f"model_error: {str(e2)[:100]}",
                    "hesitation_ms": 500,
                    "_model_failed": True,
                }

    def _build_system_prompt(self) -> str:
        """Build the system prompt from demographic info."""
        label = self.demographic.get("label", "Unknown")
        description = self.demographic.get("description", "")
        traits = self.demographic.get("dominant_traits", {})
        behaviors = self.demographic.get("key_behaviors", [])

        return f"""You are a digital twin representing the "{label}" user demographic.

Demographic Profile:
- Primary age group: {traits.get('age_group', 'unknown')}
- Primary country: {traits.get('country', 'unknown')}
- NYC familiarity: {traits.get('nyc_familiarity', 'unknown')}

Behavioral Description: {description}

Key Behaviors:
{chr(10).join(f'- {b}' for b in behaviors)}

You are exploring a web application. Based on the current page state and your recent actions, decide what to do next. Your response must be valid JSON with:
- "action": the action to take (one of: scroll, click, type, wait, navigate_back)
- "target": what to interact with (element description, scroll direction, or text to type)
- "reasoning": brief explanation of why this demographic would do this
- "hesitation_ms": how long to wait before acting (reflects your demographic's pace)"""

    # ─── ACTION EXECUTION ───

    def _execute_action(self, decision: dict) -> dict:
        """Execute the model's decision on the page. Returns execution result."""
        action = decision.get("action", "scroll")
        target = decision.get("target", "")
        hesitation = decision.get("hesitation_ms", 0)
        reasoning = decision.get("reasoning", "")

        # Apply hesitation
        if hesitation > 0:
            time.sleep(min(hesitation / 1000, 5))

        result = {"action": action, "target": target, "reasoning": reasoning, "success": False, "error": ""}

        try:
            if action == "scroll":
                self._execute_scroll(target)
                result["success"] = True

            elif action == "click":
                self._execute_click(target)
                result["success"] = True

            elif action == "type":
                self._execute_type(target)
                result["success"] = True

            elif action == "wait":
                self._wait(random.uniform(1, 3))
                result["success"] = True

            elif action == "navigate_back":
                self.page.go_back()
                self._wait(1.5)
                result["success"] = True

            else:
                # Unknown action, treat as scroll
                self.page.mouse.wheel(0, 300)
                self._wait(0.5)
                result["success"] = True
                result["action"] = "scroll (fallback)"

        except Exception as e:
            result["error"] = str(e)[:200]
            print(f"  Action failed: {e}")

        return result

    def _execute_scroll(self, target: str):
        """Execute a scroll action."""
        direction = "down" if "down" in str(target).lower() else "up"
        distance = 400 if direction == "down" else -400

        # Try to extract distance from target
        try:
            if "px" in str(target):
                import re
                nums = re.findall(r'(\d+)px', str(target))
                if nums:
                    distance = int(nums[0]) if direction == "down" else -int(nums[0])
        except Exception:
            pass

        self.page.mouse.wheel(0, distance)
        self._wait(0.8)

    def _execute_click(self, target: str):
        """Execute a click action by finding the best matching element."""
        # Clean up target description
        clean_target = target.replace("post: ", "").replace("link: ", "").replace("button: ", "").strip()

        if not clean_target:
            # Click somewhere on the page
            self.page.mouse.click(400, 400)
            self._wait(1)
            return

        # Try AgentQL semantic query
        try:
            query = f"""{{
                target_element(description: "{clean_target[:80]}")
            }}"""
            response = self.page.query_elements(query)
            if response.target_element:
                response.target_element.click()
                self._wait(1.5)
                return
        except Exception:
            pass

        # Fallback: try Playwright text selector
        try:
            locator = self.page.locator(f"text={clean_target[:50]}").first
            if locator.is_visible(timeout=2000):
                locator.click()
                self._wait(1.5)
                return
        except Exception:
            pass

        # Last resort: try partial text match
        try:
            words = clean_target.split()[:3]
            for word in words:
                if len(word) > 3:
                    locator = self.page.locator(f"text={word}").first
                    if locator.is_visible(timeout=1000):
                        locator.click()
                        self._wait(1)
                        return
        except Exception:
            pass

        print(f"  Could not find element: {clean_target[:60]}")
        raise Exception(f"Element not found: {clean_target[:60]}")

    def _execute_type(self, target: str):
        """Execute a type action."""
        # Try to find a focused or visible input
        try:
            # Check if there's an active/focused input
            active = self.page.evaluate("document.activeElement?.tagName")
            if active and active.lower() in ("input", "textarea"):
                self.page.keyboard.type(target[:100], delay=80)
                self._wait(0.5)
                return
        except Exception:
            pass

        # Try to find an input field
        try:
            response = self.page.query_elements("""
            {
                text_input_field
            }
            """)
            if response.text_input_field:
                response.text_input_field.click()
                self._wait(0.3)
                response.text_input_field.type(target[:100], delay=80)
                self._wait(0.5)
                return
        except Exception:
            pass

        print(f"  No input field found for typing")

    # ─── IMPRESSIONS ───

    def _record_impression(self, context: str, sentiment: str):
        """Record an impression about the app from this demographic's perspective."""
        self.impressions.append({
            "step": self.step_count,
            "elapsed_s": round(time.time() - self.start_time, 1),
            "context": context,
            "sentiment": sentiment,
            "url": self.page.url if self._is_browser_alive() else "unknown",
        })

    def _check_for_impressions(self, decision: dict, result: dict):
        """Check if this action warrants recording an impression."""
        if not result.get("success"):
            self._record_impression(
                f"Failed to {decision.get('action', '?')} on {decision.get('target', '?')[:50]}: {result.get('error', '')[:100]}",
                "frustration"
            )
        elif self.step_count % 5 == 0:
            # Periodic impression
            self._record_impression(
                f"After {self.step_count} actions, currently at {self.page.url if self._is_browser_alive() else 'unknown'}",
                "neutral"
            )

    # ─── LOGGING ───

    def _log_action(self, action_name: str, success: bool, error: str = "", details: dict = None):
        """Log action to session logger if available."""
        if self.logger:
            self.logger.end_action(success=success, error=error, details=details)

    def _begin_action(self, action_name: str):
        """Begin logging an action."""
        if self.logger:
            url = self.page.url if self._is_browser_alive() else ""
            self.logger.begin_action(action_name, page_url=url)

    # ─── MAIN EXECUTION LOOP ───

    def run(self) -> dict:
        """Autonomous exploration loop. Returns session summary."""
        from feedback.session_logger import StuckDetector

        self.start()
        stuck_detector = StuckDetector(timeout_s=20)

        label = self.demographic.get("label", "Unknown")
        print(f"\n{'='*50}")
        print(f"EXPLORATORY AGENT: {label}")
        print(f"Target: {self.target_url}")
        print(f"Model: {self.model_id}")
        print(f"Budget: {self.max_steps} steps / {self.max_duration_s}s")
        print(f"{'='*50}\n")

        while self.step_count < self.max_steps:
            elapsed = time.time() - self.start_time
            if elapsed >= self.max_duration_s:
                print(f"  Time budget exhausted ({elapsed:.0f}s)")
                break

            if not self._is_browser_alive():
                print("  BROWSER CLOSED — stopping")
                break

            # Check stuck
            try:
                current_url = self.page.url
            except Exception:
                break

            if stuck_detector.check(current_url):
                print(f"  STUCK DETECTED at {current_url}")
                self._record_impression(f"Got stuck at {current_url}", "confusion")
                if self.logger:
                    self.logger.log_stuck_event("explore", "Timeout", current_url)
                self.page.goto(self.target_url)
                self._wait(2)
                stuck_detector.reset()
                continue

            self.step_count += 1
            print(f"\n--- Step {self.step_count}/{self.max_steps} ({elapsed:.0f}s) ---")

            # 1. Observe
            page_state = self._observe_page()
            print(f"  Page: {page_state.get('url', '?')[:60]}")
            print(f"  Elements: {len(page_state.get('visible_elements', []))}")

            # 2. Decide
            self._begin_action("explore_step")
            decision = self._decide_next_action(page_state)
            print(f"  Decision: {decision.get('action', '?')} -> {decision.get('target', '?')[:50]}")
            if decision.get("reasoning"):
                print(f"  Reasoning: {decision['reasoning'][:80]}")

            # 3. Execute
            result = self._execute_action(decision)

            # 4. Log
            self._log_action(
                f"explore_{decision.get('action', 'unknown')}",
                success=result["success"],
                error=result.get("error", ""),
                details={
                    "action": decision.get("action"),
                    "target": decision.get("target", "")[:100],
                    "reasoning": decision.get("reasoning", "")[:200],
                }
            )

            # 5. Record action history
            self.action_history.append({
                "step": self.step_count,
                "elapsed_s": round(time.time() - self.start_time, 1),
                "action": decision.get("action"),
                "target": decision.get("target", "")[:100],
                "success": result["success"],
                "error": result.get("error", ""),
            })

            # 6. Check for impressions
            self._check_for_impressions(decision, result)

            stuck_detector.reset()

        # Final impression
        self._record_impression(
            f"Session complete after {self.step_count} steps, {time.time() - self.start_time:.0f}s",
            "summary"
        )

        total_time = time.time() - self.start_time
        successes = sum(1 for a in self.action_history if a["success"])
        failures = sum(1 for a in self.action_history if not a["success"])

        print(f"\n{'='*50}")
        print(f"AGENT FINISHED: {label}")
        print(f"Steps: {self.step_count}, Duration: {total_time:.0f}s")
        print(f"Success: {successes}, Failed: {failures}")
        print(f"Impressions: {len(self.impressions)}")
        print(f"{'='*50}")

        self.stop()

        return {
            "demographic": label,
            "model_id": self.model_id,
            "target_url": self.target_url,
            "total_steps": self.step_count,
            "total_duration_s": round(total_time, 1),
            "successful_actions": successes,
            "failed_actions": failures,
            "action_history": self.action_history,
            "impressions": self.impressions,
        }

    # ─── NARRATIVE GENERATION ───

    def generate_narrative(self, session_summary: dict) -> str:
        """Generate a first-person narrative report from this demographic's perspective."""
        label = self.demographic.get("label", "Unknown")
        description = self.demographic.get("description", "")
        traits = self.demographic.get("dominant_traits", {})

        # Summarize actions
        action_summary = ""
        for a in session_summary.get("action_history", [])[:20]:
            status = "OK" if a["success"] else f"FAILED: {a.get('error', '')[:50]}"
            action_summary += f"  [{a['elapsed_s']}s] {a['action']} on {a['target'][:40]} - {status}\n"

        # Summarize impressions
        impression_summary = ""
        for imp in session_summary.get("impressions", []):
            impression_summary += f"  [{imp['elapsed_s']}s] ({imp['sentiment']}) {imp['context'][:80]}\n"

        prompt = f"""You are writing a first-person UX report from the perspective of the "{label}" demographic.

DEMOGRAPHIC PROFILE:
- Age group: {traits.get('age_group', 'unknown')}
- Country: {traits.get('country', 'unknown')}
- NYC familiarity: {traits.get('nyc_familiarity', 'unknown')}
- Behavioral style: {description}

SESSION SUMMARY:
- Target app: {session_summary.get('target_url', '?')}
- Duration: {session_summary.get('total_duration_s', 0):.0f}s
- Steps taken: {session_summary.get('total_steps', 0)}
- Successful: {session_summary.get('successful_actions', 0)}
- Failed: {session_summary.get('failed_actions', 0)}

ACTION LOG:
{action_summary}

IMPRESSIONS:
{impression_summary}

Write a 300-500 word first-person narrative report covering:

1. **First Impressions**: What was the landing experience like for someone from my demographic?
2. **Navigation Experience**: How easy/hard was it to find things? Did the layout make sense?
3. **Engagement Barriers**: What stopped me from engaging more? What was confusing?
4. **What Worked Well**: What parts of the app felt natural and intuitive?
5. **Recommendations**: 3 specific improvements that would help people like me.

Write in first person as if you ARE this user demographic. Be specific and reference actual actions from the log.
Do not be generic — ground everything in the actual session data."""

        try:
            response = self.client.chat.complete(
                model="mistral-medium-latest",
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Narrative generation failed: {e}"


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import validate_sandbox_url, MISTRAL_API_KEY, TARGET_APP_URL, AGENT_LOGS_DIR, MODELS_DIR, CLUSTERS_DIR, ensure_data_dirs
    from feedback.session_logger import SessionLogger

    ensure_data_dirs()

    # Load model and cluster info
    models_path = MODELS_DIR / "models.json"
    clusters_path = CLUSTERS_DIR / "clusters.json"

    if not models_path.exists() or not clusters_path.exists():
        print("ERROR: Run fine_tune.py and cluster_demographics.py first")
        sys.exit(1)

    with open(models_path) as f:
        models = json.load(f)
    with open(clusters_path) as f:
        clusters = json.load(f)

    # Run the first cluster's agent as a test
    cluster = clusters["clusters"][0]
    model_info = models.get(f"cluster_{cluster['id']}", {})
    model_id = model_info.get("model_id")

    if not model_id:
        print(f"ERROR: No model found for cluster {cluster['id']}")
        sys.exit(1)

    target = validate_sandbox_url(TARGET_APP_URL)
    logger = SessionLogger(session_id=f"explore_{cluster['id']}")

    agent = ExploratoryAgent(
        model_id=model_id,
        demographic=cluster,
        target_url=target,
        mistral_api_key=MISTRAL_API_KEY,
        session_logger=logger,
        max_steps=15,
        max_duration_s=120,
    )

    summary = agent.run()
    narrative = agent.generate_narrative(summary)

    log_path = str(AGENT_LOGS_DIR / f"explore_{cluster['id']}_log.json")
    logger.save(log_path)

    print(f"\n--- Narrative Report ---")
    print(narrative)

"""
Stage 5: Execute — AgentQL + Playwright runs the behavioral agent against
the SANDBOX FunCity instance.

The agent follows the behavioral policy generated in Stage 4, using AgentQL
for semantic DOM queries and Playwright for browser automation.

SAFETY: This module validates the target URL and refuses to run against production.
"""

import json
import random
import time
from pathlib import Path
from typing import Optional

import agentql
from playwright.sync_api import sync_playwright
from mistralai import Mistral


class BehavioralAgent:
    def __init__(
        self,
        policy: dict,
        sandbox_url: str,
        mistral_api_key: str,
        session_logger=None,
    ):
        self.policy = policy
        self.sandbox_url = sandbox_url
        self.mistral_api_key = mistral_api_key
        self.logger = session_logger
        self.p = None
        self.browser = None
        self.page = None
        self.posts_visited = 0
        self.is_authenticated = False

    def start(self):
        """Launch browser and navigate to the sandbox app."""
        self.p = sync_playwright().start()
        self.browser = self.p.chromium.launch(headless=False)
        context = self.browser.new_context(
            viewport={"width": 1512, "height": 823},
        )
        self.page = agentql.wrap(context.new_page())
        self.page.goto(self.sandbox_url)
        self._wait(2)  # Let page load

    def stop(self):
        """Close the browser."""
        if self.browser:
            self.browser.close()
        if self.p:
            self.p.stop()

    def _wait(self, base_seconds: float):
        """Wait with randomized timing based on browsing speed."""
        speed_multiplier = {
            "fast": 0.5,
            "medium": 1.0,
            "slow": 2.0,
        }.get(self.policy.get("browsing_speed", "medium"), 1.0)

        actual_wait = base_seconds * speed_multiplier * random.uniform(0.7, 1.3)
        time.sleep(actual_wait)

    def _type_like_human(self, element, text: str):
        """Type text with human-like speed and optional typos."""
        for i, char in enumerate(text):
            element.type(char, delay=random.randint(50, 150))

            # Simulate typo and correction
            if self.policy["post_interaction"]["makes_typos"] and random.random() < 0.05:
                wrong_char = random.choice("abcdefghijklmnopqrstuvwxyz")
                element.type(wrong_char, delay=80)
                time.sleep(0.2)
                self.page.keyboard.press("Backspace")

    def _generate_comment(self, post_context: str = "") -> str:
        """Use Mistral Large to generate a contextually relevant comment."""
        client = Mistral(api_key=self.mistral_api_key)

        target_length = self.policy["post_interaction"]["avg_comment_length_chars"]

        prompt = f"""You are a user on FunCity, an NYC community discussion board.
You are browsing a post and want to leave a comment.

Context about the post (if available): {post_context or 'A post on the NYC community board'}

Generate a single, natural-sounding comment that:
- Is approximately {target_length} characters long
- Sounds like a real person commenting on an NYC community post
- Is casual and conversational
- Does NOT use hashtags or emojis excessively

Output ONLY the comment text, nothing else."""

        try:
            response = client.chat.complete(
                model="mistral-large-latest",
                messages=[{"role": "user", "content": prompt}],
            )
            comment = response.choices[0].message.content.strip()
            # Trim to approximate target length
            if len(comment) > target_length * 1.5:
                comment = comment[:target_length]
            return comment
        except Exception as e:
            print(f"  Warning: Comment generation failed ({e}), using fallback")
            return "This is a great spot, definitely checking it out!"[:target_length]

    def _log_action(self, action_name: str, success: bool = True, error: str = "", details: dict | None = None):
        """Helper to log an action if logger is available."""
        if self.logger:
            self.logger.end_action(success=success, error=error, details=details)

    def _begin_action(self, action_name: str):
        """Helper to begin logging an action."""
        if self.logger:
            url = self.page.url if self.page else ""
            self.logger.begin_action(action_name, page_url=url)

    # ─── ACTION IMPLEMENTATIONS ───

    def scan_feed(self):
        """Scroll through the feed to scan posts."""
        print("[AGENT] Scanning feed...")
        self._begin_action("scan_feed")

        try:
            pattern = self.policy["feed_behavior"]["initial_scroll_pattern"]

            if pattern == "quick_scan":
                self.page.mouse.wheel(0, 600)
                self._wait(0.5)
                self.page.mouse.wheel(0, 400)
                self._wait(0.5)
                self.page.mouse.wheel(0, -1000)
                self._wait(1)
            elif pattern == "slow_read":
                for _ in range(5):
                    self.page.mouse.wheel(0, 200)
                    self._wait(2)
            else:  # no_scroll
                self._wait(2)

            self._log_action("scan_feed", success=True, details={"pattern": pattern})
        except Exception as e:
            print(f"  Error scanning feed: {e}")
            self._log_action("scan_feed", success=False, error=str(e))

    def open_post(self):
        """Find and click on a post in the feed."""
        print("[AGENT] Opening a post...")
        self._begin_action("open_post")

        try:
            response = self.page.query_elements("""
            {
                post_links[] {
                    title_text
                    link_element
                }
            }
            """)

            posts = response.post_links
            if posts and len(posts) > 0:
                target = random.choice(posts[:5])
                title = ""
                try:
                    title = target.title_text.text_content() if target.title_text else "unknown"
                except Exception:
                    title = "unknown"
                print(f"  -> Clicking: {title}")
                target.link_element.click()
                self._wait(self.policy["post_interaction"]["read_before_engaging_s"])
                self.posts_visited += 1
                self._log_action("open_post", success=True, details={"post_title": title})
            else:
                print("  Warning: No posts found on page")
                self._log_action("open_post", success=False, error="No posts found")
        except Exception as e:
            print(f"  Error opening post: {e}")
            self._log_action("open_post", success=False, error=str(e))

    def _is_browser_alive(self) -> bool:
        """Check if the browser/page is still usable."""
        try:
            self.page.url  # Simple check — throws if page is closed
            return True
        except Exception:
            return False

    def _check_auth_status(self) -> bool:
        """Check if the agent is currently logged in.

        The FunCity navbar shows 'Log In' and 'Sign Up' buttons when not
        authenticated.  After auth these disappear and a logout/avatar appears.
        We use plain Playwright locators for speed (no LLM round-trip).
        """
        try:
            # If a "Sign Up" or "Log In" button is visible in the navbar, NOT logged in
            sign_up_btn = self.page.locator("button:has-text('Sign Up')").first
            if sign_up_btn.is_visible(timeout=2000):
                return False
        except Exception:
            pass

        # If we got here, the auth buttons are gone → likely logged in
        return True

    def _open_auth_modal(self, target_tab: str = "signup"):
        """Open the auth modal and switch to the requested tab.

        FunCity auth flow:
        1. Click 'Sign Up' (or 'Log In') button in the navbar → opens a
           modal dialog with Login / Sign Up tabs.
        2. The Login tab is active by default.
        3. To sign up, click the 'Sign Up' tab text inside the modal.
        """
        # Click whichever navbar button is visible to open the modal
        try:
            btn = self.page.locator("button:has-text('Sign Up')").first
            if btn.is_visible(timeout=2000):
                btn.click()
                self._wait(1)
            else:
                btn = self.page.locator("button:has-text('Log In')").first
                if btn.is_visible(timeout=2000):
                    btn.click()
                    self._wait(1)
        except Exception:
            pass

        # Wait for the modal overlay to appear
        self.page.wait_for_selector("div.fixed.inset-0", timeout=5000)
        self._wait(0.5)

        # Now click the correct tab inside the modal
        # The modal has tab buttons; pick the one matching our target
        if target_tab == "signup":
            tab_texts = ["Sign Up", "Sign up", "Signup", "Register"]
        else:
            tab_texts = ["Log In", "Log in", "Login"]

        for txt in tab_texts:
            try:
                # Look for tab buttons inside the modal (not the navbar buttons)
                tab_btn = self.page.locator(f"div.fixed button:has-text('{txt}')").first
                if tab_btn.is_visible(timeout=1000):
                    tab_btn.click()
                    self._wait(1)
                    return True
            except Exception:
                continue

        print(f"  Warning: Could not find '{target_tab}' tab in modal")
        return False

    def signup(self):
        """Create a new account via the FunCity auth modal.

        Flow (from actual recording analysis):
        1. Click 'Sign Up' in navbar → modal opens (Login tab active)
        2. Click 'Sign Up' tab inside the modal
        3. Fill: username (text input), password (password input)
        4. Fill: 3 <select> dropdowns (age group, country, NYC familiarity)
        5. Click submit button
        6. Verify auth succeeded (Sign Up button disappears from navbar)
        """
        print("[AGENT] Signing up...")
        self._begin_action("signup")

        try:
            # Step 1-2: Open modal and switch to Sign Up tab
            if not self._open_auth_modal(target_tab="signup"):
                self._log_action("signup", success=False, error="Could not open signup tab")
                return

            # Generate unique agent credentials
            agent_username = f"agent_{random.randint(10000, 99999)}"
            agent_password = f"Pass{random.randint(1000, 9999)}!"

            # Step 3: Fill text inputs using Playwright locators inside the modal
            modal = self.page.locator("div.fixed.inset-0")

            # Username — first text input in the modal form
            username_input = modal.locator("input[type='text']").first
            username_input.click()
            username_input.fill("")  # Clear any existing text
            self._type_like_human(username_input, agent_username)
            self._wait(0.5)

            # Password — password input in the modal form
            password_input = modal.locator("input[type='password']").first
            password_input.click()
            password_input.fill("")
            self._type_like_human(password_input, agent_password)
            self._wait(0.3)

            # Step 4: Fill the 3 <select> dropdowns by picking a non-first option
            selects = modal.locator("select")
            select_count = selects.count()
            print(f"  Found {select_count} dropdown(s) in signup form")
            for i in range(select_count):
                try:
                    sel = selects.nth(i)
                    # Get all option values
                    options = sel.locator("option").all()
                    if len(options) > 1:
                        # Pick index 1 (skip placeholder at index 0)
                        value = options[1].get_attribute("value")
                        if value:
                            sel.select_option(value=value)
                        else:
                            sel.select_option(index=1)
                        self._wait(0.3)
                except Exception as e:
                    print(f"  Warning: Could not fill select #{i}: {e}")

            # Step 5: Click submit
            submit_btn = modal.locator("button[type='submit']").first
            if not submit_btn.is_visible(timeout=2000):
                # Fallback: find the neon-cyan submit button
                submit_btn = modal.locator("button.bg-neon-cyan").first
            submit_btn.click()
            self._wait(3)

            # Step 6: Verify
            self.is_authenticated = self._check_auth_status()
            if self.is_authenticated:
                print(f"  Signup successful as '{agent_username}'")
                self._log_action("signup", success=True, details={
                    "username": agent_username,
                    "verified": True,
                })
            else:
                print("  Warning: Signup may have failed — auth buttons still visible")
                self._log_action("signup", success=False, error="Auth not detected after signup submit")

        except Exception as e:
            print(f"  Error signing up: {e}")
            self._log_action("signup", success=False, error=str(e))

    def login(self):
        """Log in with existing credentials. Falls back to signup on failure.

        Uses the same tabbed auth modal as signup, but stays on the Login tab.
        """
        print("[AGENT] Logging in...")
        self._begin_action("login")

        try:
            # Open modal and stay on Login tab (default)
            if not self._open_auth_modal(target_tab="login"):
                print("  Warning: Could not open login tab, trying signup")
                self._log_action("login", success=False, error="Could not open login tab")
                self.signup()
                return

            modal = self.page.locator("div.fixed.inset-0")

            # Fill username
            username_input = modal.locator("input[type='text']").first
            username_input.click()
            username_input.fill("")
            self._type_like_human(username_input, "agent_user_01")
            self._wait(0.5)

            # Fill password
            password_input = modal.locator("input[type='password']").first
            password_input.click()
            password_input.fill("")
            self._type_like_human(password_input, "agent_pass")
            self._wait(0.3)

            # Submit
            submit_btn = modal.locator("button[type='submit']").first
            if not submit_btn.is_visible(timeout=2000):
                submit_btn = modal.locator("button.bg-neon-cyan").first
            submit_btn.click()
            self._wait(3)

            # Verify
            self.is_authenticated = self._check_auth_status()
            if self.is_authenticated:
                print("  Login successful")
                self._log_action("login", success=True, details={"verified": True})
            else:
                print("  Login failed — user may not exist. Falling back to signup...")
                self._log_action("login", success=False, error="Auth not detected after login")
                # Close modal if still open, go home, try signup
                try:
                    close_btn = modal.locator("button:has(svg.lucide-x)").first
                    if close_btn.is_visible(timeout=1000):
                        close_btn.click()
                        self._wait(0.5)
                except Exception:
                    pass
                self.page.goto(self.sandbox_url)
                self._wait(2)
                self.signup()

        except Exception as e:
            print(f"  Error logging in: {e}")
            self._log_action("login", success=False, error=str(e))

    def write_comment(self):
        """Find comment box and write a comment."""
        if random.random() > self.policy["post_interaction"]["comment_probability"]:
            print("[AGENT] Skipping comment (probability check)")
            self._begin_action("write_comment")
            self._log_action("write_comment", success=True, details={"skipped": True})
            return

        print("[AGENT] Writing comment...")
        self._begin_action("write_comment")

        if not self.is_authenticated:
            print("  Warning: Not authenticated — comment section may not be available")

        try:
            # Get some page context for comment generation
            page_title = ""
            try:
                page_title = self.page.title()
            except Exception:
                pass

            response = self.page.query_elements("""
            {
                comment_section {
                    comment_input_field
                    submit_comment_button
                }
            }
            """)

            if response.comment_section and response.comment_section.comment_input_field:
                field = response.comment_section.comment_input_field
                field.click()
                self._wait(1)

                comment = self._generate_comment(post_context=page_title)
                self._type_like_human(field, comment)
                self._wait(0.5)

                if response.comment_section.submit_comment_button:
                    response.comment_section.submit_comment_button.click()
                    self._wait(1)

                self._log_action("write_comment", success=True, details={
                    "comment_length": len(comment),
                    "skipped": False,
                })
            else:
                print("  Warning: Comment section not found")
                self._log_action("write_comment", success=False, error="Comment section not found")
        except Exception as e:
            print(f"  Error writing comment: {e}")
            self._log_action("write_comment", success=False, error=str(e))

    def vote_on_post(self):
        """Find and click the upvote button."""
        if random.random() > self.policy["post_interaction"]["vote_probability"]:
            print("[AGENT] Skipping vote (probability check)")
            self._begin_action("vote_on_post")
            self._log_action("vote_on_post", success=True, details={"skipped": True})
            return

        print("[AGENT] Voting on post...")
        self._begin_action("vote_on_post")

        if not self.is_authenticated:
            print("  Warning: Not authenticated — vote buttons may be disabled")

        try:
            response = self.page.query_elements("""
            {
                vote_buttons {
                    upvote_button
                    downvote_button
                }
            }
            """)

            if response.vote_buttons and response.vote_buttons.upvote_button:
                # Check if button is disabled before attempting click
                try:
                    is_disabled = response.vote_buttons.upvote_button.is_disabled()
                except Exception:
                    is_disabled = False

                if is_disabled:
                    print("  Warning: Upvote button is disabled (not logged in?)")
                    self._log_action("vote_on_post", success=False, error="Upvote button disabled — not authenticated")
                    return

                response.vote_buttons.upvote_button.click(timeout=5000)
                self._wait(0.5)
                self._log_action("vote_on_post", success=True, details={"skipped": False})
            else:
                print("  Warning: Vote buttons not found")
                self._log_action("vote_on_post", success=False, error="Vote buttons not found")
        except Exception as e:
            error_msg = str(e)
            if "disabled" in error_msg.lower() or "not enabled" in error_msg.lower():
                print("  Warning: Vote button disabled — likely not authenticated")
                self._log_action("vote_on_post", success=False, error="Vote button disabled — not authenticated")
            else:
                print(f"  Error voting: {e}")
                self._log_action("vote_on_post", success=False, error=error_msg)

    def return_to_feed(self):
        """Click logo/home link to return to the feed."""
        print("[AGENT] Returning to feed...")
        self._begin_action("return_to_feed")

        try:
            response = self.page.query_elements("""
            {
                navigation {
                    home_link
                }
            }
            """)

            if response.navigation and response.navigation.home_link:
                response.navigation.home_link.click()
                self._wait(2)
                self._log_action("return_to_feed", success=True)
            else:
                # Fallback: navigate directly
                self.page.goto(self.sandbox_url)
                self._wait(2)
                self._log_action("return_to_feed", success=True, details={"fallback": True})
        except Exception as e:
            print(f"  Error returning to feed: {e}")
            self._log_action("return_to_feed", success=False, error=str(e))

    def browse_subreddit(self):
        """Navigate to a subreddit from the sidebar."""
        print("[AGENT] Browsing subreddit...")
        self._begin_action("browse_subreddit")

        try:
            preferred = self.policy["subreddit_exploration"].get("preferred_subreddits", [])

            response = self.page.query_elements("""
            {
                sidebar {
                    subreddit_links[] {
                        name_text
                        link_element
                    }
                }
            }
            """)

            if response.sidebar and response.sidebar.subreddit_links:
                links = response.sidebar.subreddit_links

                # Try to find a preferred subreddit
                target = None
                for link in links:
                    try:
                        name = link.name_text.text_content() if link.name_text else ""
                        if any(pref in name.lower() for pref in preferred):
                            target = link
                            break
                    except Exception:
                        continue

                if not target:
                    target = random.choice(links)

                target_name = ""
                try:
                    target_name = target.name_text.text_content() if target.name_text else "unknown"
                except Exception:
                    target_name = "unknown"

                print(f"  -> Navigating to: {target_name}")
                target.link_element.click()
                self._wait(2)
                self._log_action("browse_subreddit", success=True, details={"subreddit": target_name})
            else:
                print("  Warning: No subreddit links found")
                self._log_action("browse_subreddit", success=False, error="No subreddit links found")
        except Exception as e:
            print(f"  Error browsing subreddit: {e}")
            self._log_action("browse_subreddit", success=False, error=str(e))

    def open_related_post(self):
        """Click on a trending or related post."""
        print("[AGENT] Opening related post...")
        self._begin_action("open_related_post")

        try:
            response = self.page.query_elements("""
            {
                trending_section {
                    trending_posts[] {
                        title_text
                        link_element
                    }
                }
            }
            """)

            if response.trending_section and response.trending_section.trending_posts:
                target = random.choice(response.trending_section.trending_posts[:3])
                title = ""
                try:
                    title = target.title_text.text_content() if target.title_text else "unknown"
                except Exception:
                    title = "unknown"
                print(f"  -> Clicking trending: {title}")
                target.link_element.click()
                self._wait(2)
                self.posts_visited += 1
                self._log_action("open_related_post", success=True, details={"post_title": title})
            else:
                print("  Warning: No trending posts found")
                self._log_action("open_related_post", success=False, error="No trending posts found")
        except Exception as e:
            print(f"  Error opening related post: {e}")
            self._log_action("open_related_post", success=False, error=str(e))

    def create_post(self):
        """Create a new post in a subreddit.

        Uses Playwright locators for the post creation form. The subreddit
        dropdown is a native <select> element — use select_option() instead
        of clicking option elements (which are invisible).
        """
        print("[AGENT] Creating post...")
        self._begin_action("create_post")

        if not self.is_authenticated:
            print("  Warning: Not authenticated — cannot create post")
            self._log_action("create_post", success=False, error="Not authenticated")
            return

        try:
            # Find and click the create/submit post link/button
            # FunCity uses a link or button with text like "Submit" or "Create Post"
            response = self.page.query_elements("""
            {
                create_post_button
            }
            """)

            if response.create_post_button:
                response.create_post_button.click()
                self._wait(1.5)
            else:
                print("  Warning: Create post button not found")
                self._log_action("create_post", success=False, error="Create post button not found")
                return

            # Use Playwright locators for the form (more reliable than AgentQL for native elements)
            # Select subreddit via native <select>
            subreddit_select = self.page.locator("select").first
            try:
                if subreddit_select.is_visible(timeout=3000):
                    options = subreddit_select.locator("option").all()
                    if len(options) > 1:
                        # Skip placeholder (index 0), pick index 1
                        value = options[1].get_attribute("value")
                        if value:
                            subreddit_select.select_option(value=value)
                        else:
                            subreddit_select.select_option(index=1)
                        self._wait(0.5)
                        print(f"  Selected subreddit option")
            except Exception as e:
                print(f"  Warning: Could not select subreddit: {e}")

            # Fill title — look for an input with placeholder containing "title"
            title_input = self.page.locator("input[placeholder*='itle'], input[placeholder*='Title']").first
            if not title_input.is_visible(timeout=2000):
                # Fallback: first text input on the page (outside navs)
                title_input = self.page.locator("form input[type='text'], main input").first

            if title_input.is_visible(timeout=2000):
                title = self._generate_comment(post_context="writing a title for an NYC community post")
                title_input.click()
                title_input.fill("")
                self._type_like_human(title_input, title[:100])
                self._wait(0.5)
            else:
                print("  Warning: Title input not found")

            # Fill body — look for a textarea
            try:
                body_input = self.page.locator("textarea").first
                if body_input.is_visible(timeout=2000):
                    body = self._generate_comment(post_context="writing body text for an NYC community post")
                    body_input.click()
                    body_input.fill("")
                    self._type_like_human(body_input, body)
                    self._wait(0.5)
            except Exception:
                pass  # Body is optional

            # Submit — find submit button, check if enabled
            submit_btn = self.page.locator("button[type='submit']").first
            if not submit_btn.is_visible(timeout=2000):
                submit_btn = self.page.locator("button:has-text('Post')").first

            try:
                is_disabled = submit_btn.is_disabled()
            except Exception:
                is_disabled = False

            if is_disabled:
                print("  Warning: Submit button is disabled — required fields may be missing")
                self._log_action("create_post", success=False, error="Submit button disabled — form incomplete")
            else:
                submit_btn.click(timeout=5000)
                self._wait(2)
                print("  Post submitted successfully")
                self._log_action("create_post", success=True)

        except Exception as e:
            print(f"  Error creating post: {e}")
            self._log_action("create_post", success=False, error=str(e))

    # ─── MAIN EXECUTION LOOP ───

    def run(self):
        """Execute the behavioral policy action sequence."""
        from feedback.session_logger import StuckDetector

        self.start()
        stuck_detector = StuckDetector(timeout_s=15)

        action_map = {
            "scan_feed": self.scan_feed,
            "open_post": self.open_post,
            "signup": self.signup,
            "login": self.login,
            "write_comment": self.write_comment,
            "vote_on_post": self.vote_on_post,
            "return_to_feed": self.return_to_feed,
            "browse_subreddit": self.browse_subreddit,
            "open_related_post": self.open_related_post,
            "create_post": self.create_post,
        }

        sequence = self.policy.get("action_sequence", [])
        print(f"\n{'='*50}")
        print(f"AGENT STARTING - {len(sequence)} actions planned")
        print(f"Target: {self.sandbox_url}")
        print(f"Behavioral profile: {self.policy['navigation_style']}, "
              f"{self.policy['browsing_speed']} speed")
        print(f"{'='*50}\n")

        for i, action_name in enumerate(sequence):
            print(f"\n--- Step {i+1}/{len(sequence)}: {action_name} ---")

            # Bail out if the browser was closed externally
            if not self._is_browser_alive():
                print("  BROWSER CLOSED — aborting remaining actions")
                remaining = len(sequence) - i
                if self.logger:
                    for skip_action in sequence[i:]:
                        self.logger.begin_action(skip_action, page_url="")
                        self.logger.end_action(success=False, error="Browser closed")
                break

            # Check if stuck
            try:
                current_url = self.page.url
            except Exception:
                break
            if stuck_detector.check(current_url):
                print("  STUCK DETECTED: Recovering by navigating home")
                if self.logger:
                    self.logger.log_stuck_event(action_name, "Timeout", current_url)
                self.page.goto(self.sandbox_url)
                self._wait(2)
                stuck_detector.reset()
                continue

            action_fn = action_map.get(action_name)
            if action_fn:
                try:
                    action_fn()
                    stuck_detector.reset()
                except Exception as e:
                    print(f"  Action failed: {e}")
                    self._wait(1)
            else:
                print(f"  Unknown action: {action_name}")

        print(f"\n{'='*50}")
        print("AGENT FINISHED")
        print(f"Posts visited: {self.posts_visited}")
        print(f"{'='*50}")

        self.stop()


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import validate_sandbox_url, validate_api_keys, SANDBOX_URL, MISTRAL_API_KEY, AGENT_LOGS_DIR, ensure_data_dirs
    from feedback.session_logger import SessionLogger

    validate_api_keys()
    sandbox = validate_sandbox_url()
    ensure_data_dirs()

    if len(sys.argv) < 2:
        print("Usage: python -m pipeline.stage5_execute <policy.json>")
        sys.exit(1)

    with open(sys.argv[1]) as f:
        policy = json.load(f)

    logger = SessionLogger(session_id="manual_run")
    agent = BehavioralAgent(
        policy=policy,
        sandbox_url=sandbox,
        mistral_api_key=MISTRAL_API_KEY,
        session_logger=logger,
    )
    agent.run()

    log_path = str(AGENT_LOGS_DIR / "manual_run_log.json")
    logger.save(log_path)

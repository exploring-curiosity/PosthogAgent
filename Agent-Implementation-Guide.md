# Building an AI Agent from PostHog Session Recordings
## Implementation Guide — Sandboxed Behavioral Mimicry & UX Feedback

---

## 0. Sandbox Environment Setup

**Critical:** Agents must never interact with the production FunCity instance. All agent traffic targets a **separate staging deployment** with its own database.

### 0.1 Why Sandbox Isolation Matters

- Agent-generated comments, votes, and posts would pollute real user data
- Agent login attempts could trigger rate limits or security alerts
- PostHog analytics would be contaminated with non-human sessions
- Errors during agent execution could corrupt production state

### 0.2 Setting Up the Sandbox

1. **Deploy a second FunCity instance** (e.g. `funcity-sandbox.vercel.app`) pointing to a separate database
2. **Seed the sandbox DB** with production-like data (posts, subreddits, users) so the agent has content to interact with
3. **Create agent test accounts** in the sandbox (e.g. `agent_user_01` / `agent_pass`)

### 0.3 Configuration

All configuration lives in `.env` (see `.env.example`):

```
MISTRAL_API_KEY=your_mistral_api_key_here
AGENTQL_API_KEY=your_agentql_api_key_here
SANDBOX_URL=http://localhost:3000
PRODUCTION_URL=https://fun-city-xi.vercel.app
```

### 0.4 Safety Enforcement

The `config.py` module validates the target URL before every agent run. If the URL matches any production hostname, the pipeline exits immediately:

```python
BLOCKED_HOSTS = [
    "fun-city-xi.vercel.app",
    "fun-city.vercel.app",
]

def validate_sandbox_url(url):
    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    for blocked in BLOCKED_HOSTS:
        if blocked in hostname:
            sys.exit("SAFETY BLOCK: Refusing to run agent against production!")
    return url
```

---

## 1. What You Have

A single PostHog session recording exported as JSON from `fun-city-xi.vercel.app` (your NYC Reddit clone). Here's exactly what's inside:

**Session metadata:**
- Session ID: `019ca0c1-4d2a-745d-828e-48c6fc0cd042`
- User: age_group=25-34, country=India, nyc_familiarity=live_here
- Device: Mac OS X, Chrome 145, Desktop
- Viewport: 1512×823
- Duration: ~89.5 seconds

**Data format:** PostHog exports rrweb recording data. The file structure is:
```
{
  "version": "2023-04-28",
  "data": {
    "id": "session_id",
    "person": { ... user properties including demographics ... },
    "snapshots": [ ... 528 rrweb events ... ]
  }
}
```

**What the 528 snapshots contain:**

| rrweb Event Type | Count | What It Captures |
|---|---|---|
| Type 2 — Full Snapshot | 1 | Complete DOM tree at session start (the entire page structure serialized with node IDs) |
| Type 3 — Incremental Snapshot | 444 | All changes during the session, broken down by source: |
| → Source 0: DOM Mutations | 135 | Elements added/removed/changed (page navigation, content loading) |
| → Source 1: Mouse Moves | 99 | Mouse position coordinates with timestamps |
| → Source 2: Mouse Interactions | 95 | Clicks, focus, blur events with coordinates and target node IDs |
| → Source 3: Scroll | 33 | Scroll position changes with timestamps |
| → Source 5: Input | 82 | Text typed into input fields (masked as `***` because PostHog masks inputs by default) |
| Type 4 — Meta | 1 | Page URL and viewport dimensions |
| Type 5 — Custom | 3 | PostHog session configuration |
| Type 6 — Plugin | 79 | Network requests captured by rrweb/network@1 plugin |

**Key observation:** Input text is masked (shows as `*` characters). This is PostHog's default `maskAllInputs: true` setting. The agent handles this by using Mistral Large to generate contextually relevant text at runtime — it learns interaction patterns (where to type, how fast, how long) from the recording, and generates appropriate content when executing.

---

## 2. The Reconstructed User Journey

From the raw data, here's exactly what this user did in 89.5 seconds:

```
[0.0s]  Landed on homepage (fun-city-xi.vercel.app/)
        API loaded: /api/posts?sort=hot, /api/subreddits, /api/trending

[0-7s]  Scrolled down quickly (0→647px in 2.4s), then back up
        Scrolled down again to 886px, back to 0
        → BEHAVIOR: Rapid overview scanning, getting the lay of the land

[9.8s]  CLICKED post (node 239) — navigated to post detail
        API loaded: /api/posts/40d2825b.../comments, post details

[13.7s] CLICKED button (node 34) — likely "Login" or action button in navbar

[16.0s] CLICKED input field (node 811) — started typing (login username?)
        Typed 10 masked characters, deleted some, retyped
        → BEHAVIOR: Made a typo, corrected it — natural human hesitation

[19.8s] CLICKED second input (node 814) — typed 6 masked characters (password?)

[22.3s] CLICKED submit button (node 798)
        API called: /api/auth/login
        → BEHAVIOR: Completed login flow in ~8.5 seconds

[27.3s] CLICKED comment input area (node 835)
        Typed 38 masked characters over ~15 seconds
        → BEHAVIOR: Wrote a medium-length comment, steady typing speed

[44.7s] CLICKED submit comment button (node 823)
        API called: POST /api/comments

[49.0s] CLICKED vote button (node 854)
        API called: /api/votes
        → BEHAVIOR: Voted on the post after commenting

[50.7s] CLICKED logo/home link (node 31) — navigated back to homepage
        API reloaded: posts, subreddits, trending

[51.6s] Quick scroll down (0→886px in 0.2s)
        → BEHAVIOR: Already knows the layout, scrolling fast

[52.8s] CLICKED second post (node 1357)
        API loaded: /api/posts/ef376747.../comments

[55.3s] CLICKED comment input (node 1487)
        Typed 18 masked characters over ~10 seconds

[66.9s] CLICKED submit comment (node 1488)
        API called: POST /api/comments

[69.3s] CLICKED vote button (node 1462)
        API called: /api/votes

[70.4s] CLICKED home link (node 31) — back to feed

[71.4s] Scrolled down and back up rapidly (886→0 in 1.7s)
        → BEHAVIOR: Quick scan, already familiar

[75.5s] CLICKED subreddit link (node 1701) — navigated to /r/food
        API called: /api/subreddits/food, /api/posts?sort=hot&subreddit_id=...

[79.7s] CLICKED subreddit link (node 2309) — navigated to /r/nature
        API called: /api/subreddits/nature

[81.8s] CLICKED post in nature subreddit (node 2567)
        API loaded post and comments

[83.3s] CLICKED vote button (node 2598) — voted immediately
        → BEHAVIOR: Quick vote without commenting this time

[86.4s] Small scroll

[88.0s] CLICKED trending/related post (node 2725) — navigated to another post
        Session ends
```

**Behavioral Summary:**
- Active, fast-paced user (89.5s session, 19 clicks, constant interaction)
- Pattern: Read post → Login → Comment → Vote → Next post → Comment → Vote → Browse subreddits → Quick vote → Browse more
- Comments on 2 of 4 posts visited (50% comment rate)
- Votes on 3 of 4 posts visited (75% vote rate)
- Made a typing mistake and corrected it (natural human behavior)
- Scrolling style: rapid overview scans, gets faster on return visits
- Explored subreddits (food, nature) after initial feed browsing
- Session flow: Feed → Post → Feed → Post → Feed → Subreddit → Subreddit → Post → Post

---

## 3. Architecture: From Recording to Agent

### 3.1 The Complete Pipeline

```
┌─────────────────────────────────────────────┐
│  STAGE 0: SANDBOX SETUP (prerequisite)      │
│  Deploy isolated FunCity instance            │
│  Separate DB, agent test accounts           │
│  Safety: production URL blocked in config   │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│  STAGE 1: PARSE                             │
│  PostHog JSON → Structured Event Sequence   │
│  Tool: Python (pipeline/stage1_parse.py)    │
│  Input: Raw rrweb JSON                      │
│  Output: Ordered list of semantic actions   │
└─────────────┬───────────────────────────────┘
              ↓
┌─────────────────────────────────────────────┐
│  STAGE 2: DESCRIBE                          │
│  Event Sequence → Natural Language Summary  │
│  Tool: Mistral Large                        │
│  Input: Structured events + DOM context     │
│  Output: Text description of session        │
└─────────────┬───────────────────────────────┘
              ↓
┌─────────────────────────────────────────────┐
│  STAGE 3: ENCODE                            │
│  Session Description → Behavioral Embedding │
│  Tool: Mistral Embed                        │
│  Input: Text session description            │
│  Output: Vector embedding                   │
│  (With multiple sessions: cluster into      │
│   persona profiles per demographic)         │
└─────────────┬───────────────────────────────┘
              ↓
┌─────────────────────────────────────────────┐
│  STAGE 4: GENERATE POLICY                   │
│  Behavioral Profile → Agent Instructions    │
│  Tool: Mistral Large                        │
│  Input: Behavioral description              │
│  Output: Structured behavioral policy JSON  │
└─────────────┬───────────────────────────────┘
              ↓
┌─────────────────────────────────────────────┐
│  STAGE 5: EXECUTE (targets SANDBOX only)    │
│  Agent Policy → Browser Interaction         │
│  Tool: AgentQL + Playwright                 │
│  Input: Behavioral policy + sandbox URL     │
│  Output: Agent session log (JSON)           │
└─────────────┬───────────────────────────────┘
              ↓
┌─────────────────────────────────────────────┐
│  STAGE 6: FEEDBACK REPORT                   │
│  Agent Session → UX Improvement Report      │
│  Tool: Mistral Large + metrics engine       │
│  Input: Agent log + real user session data  │
│  Output: Quant metrics + qual UX feedback   │
└─────────────────────────────────────────────┘
```

---

## 4. Stage 1: Parse — Extract Structured Events from rrweb Data

### 4.1 Understanding rrweb Event Types

The rrweb format uses numeric type codes. Here's what each one means:

| Type | Name | What It Contains |
|---|---|---|
| 0 | DomContentLoaded | Page load timing |
| 1 | Load | Page fully loaded |
| 2 | FullSnapshot | Complete serialized DOM tree with every element assigned a numeric `id`. This is your Rosetta Stone — it maps node IDs to actual HTML elements |
| 3 | IncrementalSnapshot | Changes to the page. Sub-typed by `source` field (see below) |
| 4 | Meta | Page URL, viewport width/height |
| 5 | Custom | Application-specific events (PostHog config in this case) |
| 6 | Plugin | Plugin data (network requests via rrweb/network@1) |

**IncrementalSnapshot sources (type=3, differentiated by `data.source`):**

| Source | Name | Data Fields |
|---|---|---|
| 0 | Mutation | `adds`, `removes`, `attributes`, `texts` — DOM changes |
| 1 | MouseMove | `positions[]` — array of `{x, y, id, timeOffset}` |
| 2 | MouseInteraction | `type` (0=MouseUp, 1=MouseDown, 2=Click, 5=Focus, 6=Blur), `x`, `y`, `id` (target node) |
| 3 | Scroll | `id` (scrolled element node), `x`, `y` (scroll position) |
| 5 | Input | `id` (input node), `text` (value — masked if PostHog masking enabled), `isChecked` |

### 4.2 The Node ID System

This is crucial to understand. In the FullSnapshot (type=2), rrweb serializes the entire DOM tree and assigns every element a numeric `id`. For example:

```json
{
  "type": 2,
  "data": {
    "node": {
      "type": 0,
      "childNodes": [
        {
          "type": 1,
          "tagName": "html",
          "id": 2,
          "childNodes": [
            {
              "tagName": "nav",
              "id": 28,
              "attributes": { "class": "sticky top-0 z-40 border-b..." }
            },
            {
              "tagName": "a",
              "id": 30,
              "attributes": { "href": "https://fun-city-xi.vercel.app/" }
            },
            {
              "tagName": "button",
              "id": 34,
              "attributes": { "class": "rounded-md px-3 py-1.5 text-sm..." }
            }
          ]
        }
      ]
    }
  }
}
```

When a click event says `"id": 34`, it refers to that specific button element. By building a node ID → element mapping from the FullSnapshot, you can resolve every click, input, and scroll to its actual DOM element.

**Important:** Node IDs are only stable within a single FullSnapshot. When the page navigates (SPA navigation), DOM Mutations (source=0) add new nodes with new IDs. You must track mutations to keep your node map current.

### 4.3 Parser Implementation

**Full source:** `pipeline/stage1_parse.py`

The parser (`SessionParser` class) does three key things:

1. **Builds a node map** from the FullSnapshot DOM tree — maps numeric node IDs to element info (tag, class, href, text)
2. **Tracks DOM mutations** (source=0) to keep the node map current during SPA navigation
3. **Collapses raw events** into high-level semantic actions via `get_high_level_actions()` — groups consecutive scrolls, filters mouse moves, resolves click targets

Key design decisions:
- Only Click events (interaction type=2) are captured; mouseup/down/focus/blur are skipped
- Consecutive scroll events are merged into single scroll summaries with direction, depth, and duration
- Mouse moves are sampled (last position only) and excluded from high-level view
- API calls are filtered to only `/api/` paths

**Run it:**
```bash
python -m pipeline.stage1_parse export-019ca0c1-*.json
```

**Output:** A clean JSON file saved to `data/parsed/` with ~50-80 high-level actions (clicks, scroll summaries, input events, API calls, page loads) instead of 528 raw rrweb events.

---

## 5. Stage 2: Describe — LLM Converts Events to Behavioral Narrative

### 5.1 Why This Step Exists

Mistral Embed (Stage 3) is a text embedding model. It can't process raw JSON event arrays. You need to convert the structured events into a natural language description of the user's behavior. This is where Mistral Large comes in.

### 5.2 Implementation

**Full source:** `pipeline/stage2_describe.py`

The describe stage:
1. Converts the parsed high-level actions into a human-readable action log (timestamped lines like `[9.8s] Clicked a (link to: /post/...)`)
2. Sends the action log + user profile to Mistral Large with a prompt requesting 8 behavioral dimensions
3. Saves the output to `data/descriptions/`

The prompt asks Mistral Large to analyze: navigation pattern, reading behavior, engagement style, interaction speed, content preferences, typing behavior, feature discovery, and session flow.

**Run it:**
```bash
python -m pipeline.stage2_describe data/parsed/parsed_<session_id>.json
```

### 5.3 Expected Output

Mistral Large produces a structured behavioral profile like:

```
NAVIGATION PATTERN: Exploratory then targeted. User starts with rapid feed scanning
(two quick scroll-downs in first 7 seconds), then shifts to a focused pattern of
visiting individual posts. Returns to feed between posts. Later session shows
subreddit browsing (food -> nature), indicating categorical exploration.

ENGAGEMENT STYLE: Highly active. Commented on 2 of 4 posts (50% comment rate).
Voted on 3 of 4 posts (75% vote rate). Pattern: Comment first, then vote, then leave.

INTERACTION SPEED: Fast. Average time between clicks: 4.5 seconds. Login completed
in 8.5 seconds. Comments written in 15s and 10s respectively.
...
```

---

## 6. Stage 3: Encode — Mistral Embed Creates Behavioral Vectors

### 6.1 What This Does

Converts the text behavioral description into a numeric vector. With multiple sessions, these vectors can be clustered to find behavioral patterns across demographics.

### 6.2 Implementation

**Full source:** `pipeline/stage3_encode.py`

```python
def encode_behavior(behavioral_description: str, api_key: str) -> list[float]:
    client = Mistral(api_key=api_key)
    response = client.embeddings.create(
        model="mistral-embed",
        inputs=[behavioral_description],
    )
    return response.data[0].embedding  # 1024-dimensional vector
```

Also supports `encode_multiple_sessions()` for batch encoding when you have multiple sessions.

### 6.3 With Multiple Sessions: Clustering (Future)

When you have 50+ sessions, cluster the embeddings using KMeans to find behavioral archetypes. This is not implemented in V1 (single session) but the pipeline is designed to support it.

### 6.4 For Your Feasibility Test (Single Session)

With only one session, skip clustering. Go directly from the behavioral description to policy generation. The embedding step proves the pipeline works; clustering proves it scales.

---

## 7. Stage 4: Generate Policy — Mistral Large Creates Agent Instructions

### 7.1 What This Does

Takes the behavioral description (from Stage 2) and converts it into a structured JSON policy that an agent can follow. This is the "behavioral DNA" that makes the agent act like the real user.

### 7.2 Implementation

**Full source:** `pipeline/stage4_policy.py`

The prompt instructs Mistral Large to output a JSON policy with `response_format={"type": "json_object"}`. The policy captures patterns, not content.

### 7.3 Expected Output for This Session

```json
{
  "session_target_duration_s": 90,
  "navigation_style": "exploratory",
  "browsing_speed": "fast",
  "feed_behavior": {
    "initial_scroll_pattern": "quick_scan",
    "sort_preference": "hot",
    "max_posts_to_visit": 4,
    "scroll_between_posts": true
  },
  "post_interaction": {
    "comment_probability": 0.5,
    "vote_probability": 0.75,
    "vote_then_comment_order": "comment_first",
    "avg_comment_length_chars": 28,
    "typing_speed": "fast",
    "makes_typos": true,
    "read_before_engaging_s": 5
  },
  "subreddit_exploration": {
    "explores_subreddits": true,
    "preferred_subreddits": ["food", "nature"],
    "subreddit_browse_depth": "surface"
  },
  "login_behavior": {
    "logs_in": true,
    "login_timing": "before_first_action"
  },
  "engagement_arc": {
    "early_session": "exploring",
    "mid_session": "engaging",
    "late_session": "browsing"
  },
  "action_sequence": [
    "scan_feed", "open_post", "login", "write_comment", "vote_on_post",
    "return_to_feed", "open_post", "write_comment", "vote_on_post",
    "return_to_feed", "browse_subreddit", "browse_subreddit",
    "open_post", "vote_on_post", "open_related_post"
  ]
}
```

Valid action names: `scan_feed`, `open_post`, `login`, `write_comment`, `vote_on_post`, `return_to_feed`, `browse_subreddit`, `open_related_post`

---

## 8. Stage 5: Execute — AgentQL + Playwright Runs the Agent

### 8.1 How AgentQL Works

AgentQL is a semantic query language for the DOM. You describe what you want in a structured query, and it finds the matching elements on the live page regardless of CSS classes, IDs, or DOM position.

AgentQL uses an LLM under the hood to match your semantic description to actual DOM elements. It's built on top of Playwright, so once it finds an element, Playwright handles the click/type/scroll.

### 8.2 Agent Implementation

**Full source:** `pipeline/stage5_execute.py`

Key changes from the original draft:

1. **AgentQL API corrected:** Uses `agentql.wrap(page)` and `page.query_elements()` instead of the deprecated `agentql.Session(page)` / `session.query()` pattern
2. **Sandbox-only execution:** Agent validates URL via `config.validate_sandbox_url()` and refuses to target production
3. **Session logging:** Every action is logged via `SessionLogger` (timing, success/failure, errors) for Stage 6 analysis
4. **Contextual comment generation:** Uses Mistral Large to generate relevant comments instead of hardcoded strings
5. **Stuck detection:** Integrated `StuckDetector` in the main execution loop with auto-recovery (navigate home)

Example AgentQL query (semantic, not CSS-based):
```
{
    post_links[] {
        title_text
        link_element
    }
}
```
This finds elements that semantically match "post links" regardless of whether the page is 375px mobile or 1512px desktop.

### 8.3 Session Logger

**Full source:** `feedback/session_logger.py`

Records every agent action with:
- Action name, timestamp, duration
- Success/failure status and error messages
- Page URL at time of action
- Stuck events (timeout-based detection)

Saved to `data/agent_logs/` as JSON for Stage 6 analysis.

### 8.4 Running Stage 5

```bash
python -m pipeline.stage5_execute data/policies/policy_<session_id>.json
```

---

## 9. Stage 6: Feedback Report — UX Improvement Analysis

### 9.1 What This Does

After the agent completes its session, Stage 6 compares the agent's experience against the real user's session and generates a combined quantitative + qualitative UX feedback report.

### 9.2 Quantitative Metrics

**Full source:** `feedback/metrics.py`

Computes:
- **Completion rate:** How many planned actions succeeded vs failed
- **Timing deltas:** Per-action timing comparison (agent vs real user)
- **Error analysis:** Which actions failed, error messages, stuck events grouped by action
- **Policy deviation:** How much the agent's actual sequence diverged from the planned sequence (edit-distance-like comparison)
- **Engagement metrics:** Comments written, votes cast (agent vs real user)

### 9.3 Qualitative Feedback

**Full source:** `feedback/stage6_report.py`

Sends the quantitative metrics + agent session log + real user behavioral description to Mistral Large with a prompt requesting:

1. **Friction points** — Where did the agent struggle? What does this suggest about the UI?
2. **Accessibility issues** — Were any elements hard to find or interact with?
3. **Flow problems** — Where did the agent's journey diverge from the real user's?
4. **Performance concerns** — Timing issues suggesting slow loading or unresponsive UI?
5. **Specific recommendations** — 3-5 concrete, actionable UX improvements ranked by impact

### 9.4 Output

Two files saved to `data/reports/`:
- **`feedback_report_<session_id>.json`** — Structured data for programmatic analysis
- **`feedback_report_<session_id>.md`** — Human-readable markdown with metrics tables + UX recommendations

---

## 10. End-to-End Run: Putting It All Together

### 10.1 The Orchestrator

**Full source:** `run_pipeline.py`

Wires all 6 stages together. Loads config from `.env`, validates sandbox URL, runs each stage in sequence, saves all intermediate artifacts to `data/` subdirectories.

### 10.2 Running It

```bash
# Install dependencies
pip install -r requirements.txt
playwright install chromium

# Configure environment
cp .env.example .env
# Edit .env with your API keys and SANDBOX_URL

# Run the full pipeline
python run_pipeline.py

# Or with a specific recording:
python run_pipeline.py path/to/recording.json
```

### 10.3 What You Get

After a successful run, `data/` contains:
```
data/
├── parsed/parsed_<session_id>.json           # Structured events
├── descriptions/description_<session_id>.txt # Behavioral narrative
├── embeddings/embedding_<session_id>.json    # 1024-dim vector
├── policies/policy_<session_id>.json         # Agent behavioral policy
├── agent_logs/agent_log_<session_id>.json    # Agent session log
└── reports/
    ├── feedback_report_<session_id>.json     # Structured metrics
    └── feedback_report_<session_id>.md       # Human-readable report
```

---

## 11. What This Proves (and What It Doesn't)

### Proven by this feasibility test:

| Component | Status | Evidence |
|---|---|---|
| PostHog rrweb data is parseable | ✅ | Stage 1 extracts structured events from 528 raw snapshots |
| Node ID → DOM element mapping works | ✅ | FullSnapshot provides complete element tree; clicks resolve to actual tags, classes, hrefs |
| Mouse interactions have coordinates AND node targets | ✅ | 19 clicks captured with (x,y) and node_id |
| Scroll behavior is captured quantitatively | ✅ | 33 scroll events with exact pixel positions and timestamps |
| Input events capture typing patterns | ✅ | 82 input events show keystroke timing (though text is masked) |
| Network requests reveal user intent | ✅ | API calls to /api/posts, /api/comments, /api/votes show what the user did semantically |
| Session can be reconstructed as a behavioral narrative | ✅ | Timeline reconstruction shows complete user journey |
| Mistral Large can interpret behavioral data | ✅ | Standard LLM task — structured data → natural language description |
| Mistral Embed can encode behavioral descriptions | ✅ | Standard embedding task — text → 1024-dim vector |
| AgentQL can find DOM elements semantically | ✅ | Queries like `{ post_links[] { title_text } }` work on arbitrary pages |
| Sandbox isolation enforced in code | ✅ | `config.py` blocks production URLs before agent execution |
| Agent session logging for post-analysis | ✅ | `SessionLogger` records timing, errors, stuck events per action |

### Not yet proven (needs multiple sessions):

| Component | Status | Needed |
|---|---|---|
| Behavioral clustering produces meaningful personas | ⏳ | 50+ sessions across demographics |
| Cloned agents behave more realistically than prompted agents | ⏳ | Comparative evaluation with both approaches |
| Agents discover new features in demographically accurate ways | ⏳ | Deploy V2 features, run agents, compare |

### Known limitations:

| Limitation | Impact | Mitigation |
|---|---|---|
| Input text is masked (`***`) | Agent can't learn WHAT users type, only WHERE and HOW FAST | Agent uses Mistral Large to generate contextually appropriate content at runtime |
| Single session = no clustering | Can't prove demographic differentiation | Collect 50+ sessions from library recruitment |
| AgentQL may fail on unfamiliar page structures | Agent gets stuck | Stuck detection + recovery logic (go home, retry) |
| Timing is approximate | Agent pacing won't perfectly match human | Good enough for behavioral pattern matching |

---

## 12. File Structure

```
PosthogAgent/
├── config.py                    # Env vars, sandbox URL validation, paths
├── run_pipeline.py              # End-to-end orchestrator (Stages 1-6)
├── pipeline/
│   ├── __init__.py
│   ├── stage1_parse.py          # PostHog JSON → structured events
│   ├── stage2_describe.py       # Events → behavioral narrative (Mistral Large)
│   ├── stage3_encode.py         # Narrative → embedding (Mistral Embed)
│   ├── stage4_policy.py         # Behavioral description → agent policy JSON
│   └── stage5_execute.py        # AgentQL + Playwright agent runner
├── feedback/
│   ├── __init__.py
│   ├── session_logger.py        # Records agent actions during execution
│   ├── metrics.py               # Quantitative metrics (agent vs real user)
│   └── stage6_report.py         # Combined quant+qual feedback report
├── data/
│   ├── recordings/              # Raw PostHog JSON exports
│   ├── parsed/                  # Stage 1 output
│   ├── descriptions/            # Stage 2 output
│   ├── embeddings/              # Stage 3 output
│   ├── policies/                # Stage 4 output
│   ├── agent_logs/              # Stage 5 output
│   └── reports/                 # Stage 6 output
├── .env.example                 # API key and URL template
├── requirements.txt
└── README.md
```

---

## 13. Future: NVIDIA Brev (Parallel Execution at Scale)

When you have multiple agent policies and need to run them in parallel, NVIDIA Brev provides on-demand GPU instances. Each agent runs a headless Chrome instance via Playwright; running 20+ agents simultaneously requires GPU-class compute. This is not part of V1 but the pipeline is designed to support it.

---

## 14. Future: Pixtral (Visual Feedback)

Pixtral is NOT in the V1 pipeline. Future uses:

1. **V3 Feedback Reports:** Screenshot the page after agent interaction, send to Pixtral for visual UX assessment
2. **Canvas fallback:** If map/canvas elements are added, Pixtral processes screenshots to understand spatial content
3. **Visual regression detection:** Compare screenshots before/after feature changes

---

## 15. Summary: Tool Justification

| Tool | Role in Pipeline | Why This Tool |
|---|---|---|
| **Python** | Stage 1: Parse rrweb data | Standard data processing |
| **Mistral Large** | Stage 2, 4, 5, 6: Behavioral description, policy, comments, feedback | Strong reasoning model, JSON output mode |
| **Mistral Embed** | Stage 3: Behavioral encoding | Text → vector embeddings for clustering |
| **AgentQL** | Stage 5: Semantic DOM querying | Finds elements by meaning, not selectors |
| **Playwright** | Stage 5: Browser automation | Industry standard, headless Chrome control |
| **python-dotenv** | Config: Env var management | Secure API key handling |
| **scikit-learn** | Future: Clustering (multi-session) | Standard ML clustering (KMeans) |
| **NVIDIA Brev** | Future: Parallel agent execution | GPU compute for 20+ browser agents |
| **Pixtral** | Future: Visual feedback reports | Vision-language model for screenshot analysis |

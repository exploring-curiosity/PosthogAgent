"""
Session Logger — Records agent actions, timing, errors, and stuck events
during execution for later analysis in Stage 6.
"""

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class ActionLog:
    action_name: str
    timestamp: float          # Unix timestamp when action started
    relative_s: float         # Seconds since session start
    duration_s: float = 0.0   # How long the action took
    success: bool = True
    error: str = ""
    page_url: str = ""
    details: dict = field(default_factory=dict)


class SessionLogger:
    """Records everything the agent does during a session for post-analysis."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.start_time: float = time.time()
        self.actions: list[ActionLog] = []
        self.stuck_events: list[dict] = []
        self.errors: list[dict] = []
        self._current_action_start: float = 0.0

    @property
    def elapsed_s(self) -> float:
        return round(time.time() - self.start_time, 2)

    def begin_action(self, action_name: str, page_url: str = ""):
        """Call before starting an action."""
        self._current_action_start = time.time()
        self._current_action_name = action_name
        self._current_page_url = page_url

    def end_action(self, success: bool = True, error: str = "", details: dict | None = None):
        """Call after completing an action."""
        now = time.time()
        duration = round(now - self._current_action_start, 2) if self._current_action_start else 0.0

        log = ActionLog(
            action_name=self._current_action_name,
            timestamp=self._current_action_start,
            relative_s=round(self._current_action_start - self.start_time, 2),
            duration_s=duration,
            success=success,
            error=error,
            page_url=self._current_page_url,
            details=details or {},
        )
        self.actions.append(log)

        if not success:
            self.errors.append({
                "action": self._current_action_name,
                "error": error,
                "relative_s": log.relative_s,
            })

    def log_stuck_event(self, action_name: str, reason: str, page_url: str = ""):
        """Record when the agent gets stuck."""
        self.stuck_events.append({
            "action": action_name,
            "reason": reason,
            "relative_s": self.elapsed_s,
            "page_url": page_url,
        })

    def get_summary(self) -> dict:
        """Get a summary of the agent session."""
        total_actions = len(self.actions)
        successful = sum(1 for a in self.actions if a.success)
        failed = total_actions - successful

        action_timings = {}
        for a in self.actions:
            if a.action_name not in action_timings:
                action_timings[a.action_name] = []
            action_timings[a.action_name].append(a.duration_s)

        avg_timings = {
            name: round(sum(times) / len(times), 2)
            for name, times in action_timings.items()
        }

        return {
            "session_id": self.session_id,
            "total_duration_s": self.elapsed_s,
            "total_actions": total_actions,
            "successful_actions": successful,
            "failed_actions": failed,
            "completion_rate": round(successful / total_actions, 2) if total_actions > 0 else 0,
            "stuck_events": len(self.stuck_events),
            "error_count": len(self.errors),
            "avg_action_timings": avg_timings,
            "action_sequence_actual": [a.action_name for a in self.actions],
        }

    def to_dict(self) -> dict:
        """Serialize the full session log."""
        return {
            "session_id": self.session_id,
            "start_time": self.start_time,
            "total_duration_s": self.elapsed_s,
            "actions": [asdict(a) for a in self.actions],
            "stuck_events": self.stuck_events,
            "errors": self.errors,
            "summary": self.get_summary(),
        }

    def save(self, output_path: str):
        """Save the session log to a JSON file."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"  Saved agent session log to {output_path}")


class StuckDetector:
    """Detects when the agent appears stuck and triggers recovery."""

    def __init__(self, timeout_s: int = 15):
        self.timeout = timeout_s
        self.last_url: str = ""
        self.last_action_time: float = time.time()

    def check(self, page_url: str) -> bool:
        """Returns True if agent appears stuck."""
        now = time.time()

        if now - self.last_action_time > self.timeout:
            return True

        self.last_url = page_url
        self.last_action_time = now
        return False

    def reset(self):
        """Reset the stuck timer after a successful action."""
        self.last_action_time = time.time()

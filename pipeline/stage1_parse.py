"""
Stage 1: Parse — Extract structured events from PostHog rrweb JSON.

Reads a PostHog session recording export and converts the raw rrweb events
into a clean sequence of high-level semantic actions (clicks, scrolls, inputs,
API calls, page loads).
"""

import gzip
import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class ParsedEvent:
    timestamp_ms: int
    relative_s: float
    event_type: str          # CLICK, SCROLL, INPUT, NAVIGATE, API_CALL, MOUSE_MOVE
    details: dict = field(default_factory=dict)


class SessionParser:
    def __init__(self, json_path: str):
        with open(json_path) as f:
            raw = json.load(f)

        self.session_id = raw["data"]["id"]
        self.person = raw["data"]["person"]
        self.snapshots = raw["data"]["snapshots"]
        self.node_map: dict[int, dict] = {}      # node_id -> element info
        self.events: list[ParsedEvent] = []
        self.first_ts: int = self.snapshots[0]["timestamp"]

    def relative_time(self, ts: int) -> float:
        return round((ts - self.first_ts) / 1000, 2)

    def build_node_map(self, node: dict, depth: int = 0):
        """Recursively walk the FullSnapshot DOM tree and build node_id -> element mapping."""
        if not isinstance(node, dict):
            return

        node_id = node.get("id")
        if node_id is not None:
            tag = node.get("tagName", "")
            attrs = node.get("attributes", {})
            text_content = ""

            # Extract text from childNodes
            for child in node.get("childNodes", []):
                if isinstance(child, dict) and child.get("type") == 3:  # text node
                    text_content += child.get("textContent", "")

            self.node_map[node_id] = {
                "tag": tag,
                "attrs": attrs,
                "text": text_content.strip()[:100],
                "depth": depth,
            }

        for child in node.get("childNodes", []):
            self.build_node_map(child, depth + 1)

    def resolve_node(self, node_id: int) -> dict:
        """Get element info for a node ID, with fallback."""
        info = self.node_map.get(node_id, {})
        return {
            "node_id": node_id,
            "tag": info.get("tag", "unknown"),
            "class": str(info.get("attrs", {}).get("class", ""))[:80],
            "href": info.get("attrs", {}).get("href", ""),
            "text": info.get("text", ""),
            "type_attr": info.get("attrs", {}).get("type", ""),
            "placeholder": info.get("attrs", {}).get("placeholder", ""),
        }

    def update_node_map_from_mutations(self, mutation_data: dict):
        """Track DOM mutations to keep node map current (for SPA navigations)."""
        for add in mutation_data.get("adds", []):
            if not isinstance(add, dict):
                continue
            node = add.get("node", {})
            if isinstance(node, dict) and node:
                self.build_node_map(node)

    @staticmethod
    def _decode_data(raw_data):
        """Decode snapshot data — handles gzip-compressed strings from 2024-04-30 format."""
        if isinstance(raw_data, str):
            try:
                raw_bytes = raw_data.encode("latin-1")
                if raw_bytes[:2] == b'\x1f\x8b':
                    return json.loads(gzip.decompress(raw_bytes))
            except Exception:
                pass
            try:
                return json.loads(raw_data)
            except Exception:
                return {}
        if isinstance(raw_data, dict):
            return raw_data
        return {}

    def parse(self) -> list[ParsedEvent]:
        """Parse all snapshots into a structured event sequence."""
        for snap in self.snapshots:
            ts = snap["timestamp"]
            rel = self.relative_time(ts)
            stype = snap["type"]
            data = self._decode_data(snap.get("data", {}))

            # Type 2: FullSnapshot — build initial node map
            if stype == 2:
                self.build_node_map(data.get("node", {}))
                self.events.append(ParsedEvent(
                    timestamp_ms=ts,
                    relative_s=rel,
                    event_type="FULL_SNAPSHOT",
                    details={"url": "initial_page_load"},
                ))

            # Type 4: Meta — page URL and viewport
            elif stype == 4:
                self.events.append(ParsedEvent(
                    timestamp_ms=ts,
                    relative_s=rel,
                    event_type="PAGE_META",
                    details={
                        "url": data.get("href", ""),
                        "width": data.get("width"),
                        "height": data.get("height"),
                    },
                ))

            # Type 3: IncrementalSnapshot
            elif stype == 3:
                source = data.get("source")

                # Source 0: Mutation — track DOM changes
                if source == 0:
                    self.update_node_map_from_mutations(data)

                # Source 2: MouseInteraction
                elif source == 2:
                    interaction_type = data.get("type")
                    # Only capture Click events (type=2), skip mouseup/down/focus/blur
                    if interaction_type == 2:
                        node_info = self.resolve_node(data.get("id", 0))
                        self.events.append(ParsedEvent(
                            timestamp_ms=ts,
                            relative_s=rel,
                            event_type="CLICK",
                            details={
                                "x": data.get("x"),
                                "y": data.get("y"),
                                **node_info,
                            },
                        ))

                # Source 3: Scroll
                elif source == 3:
                    self.events.append(ParsedEvent(
                        timestamp_ms=ts,
                        relative_s=rel,
                        event_type="SCROLL",
                        details={
                            "scroll_y": data.get("y", 0),
                            "scroll_x": data.get("x", 0),
                            "node_id": data.get("id"),
                        },
                    ))

                # Source 5: Input
                elif source == 5:
                    text = data.get("text", "")
                    if text:
                        node_info = self.resolve_node(data.get("id", 0))
                        self.events.append(ParsedEvent(
                            timestamp_ms=ts,
                            relative_s=rel,
                            event_type="INPUT",
                            details={
                                "text_length": len(text),
                                "is_masked": all(c == "*" for c in text),
                                **node_info,
                            },
                        ))

                # Source 1: MouseMove — sample to reduce noise
                elif source == 1:
                    positions = data.get("positions", [])
                    if positions:
                        last = positions[-1]
                        self.events.append(ParsedEvent(
                            timestamp_ms=ts,
                            relative_s=rel,
                            event_type="MOUSE_MOVE",
                            details={
                                "x": last.get("x"),
                                "y": last.get("y"),
                                "positions_count": len(positions),
                            },
                        ))

            # Type 6: Plugin — network requests
            elif stype == 6:
                payload = data.get("payload", {})
                requests = payload.get("requests", [])
                for req in requests:
                    if isinstance(req, dict):
                        url = req.get("name", "")
                        if "/api/" in url:
                            path = url.split(".app")[-1] if ".app" in url else url
                            self.events.append(ParsedEvent(
                                timestamp_ms=ts,
                                relative_s=rel,
                                event_type="API_CALL",
                                details={
                                    "path": path.split("?")[0],
                                    "full_path": path[:150],
                                    "method": req.get("method", "unknown"),
                                },
                            ))

        return self.events

    def get_user_profile(self) -> dict:
        """Extract demographic and device info."""
        props = self.person.get("properties", {})
        return {
            "session_id": self.session_id,
            "age_group": props.get("age_group"),
            "country": props.get("country"),
            "nyc_familiarity": props.get("nyc_familiarity"),
            "os": props.get("$os"),
            "browser": props.get("$browser"),
            "device_type": props.get("$device_type"),
            "viewport_width": props.get("$viewport_width"),
            "viewport_height": props.get("$viewport_height"),
            "screen_width": props.get("$screen_width"),
            "screen_height": props.get("$screen_height"),
        }

    def get_high_level_actions(self) -> list[dict]:
        """
        Collapse raw events into high-level semantic actions.
        This is what gets fed to the LLM for behavioral understanding.
        """
        actions = []
        scroll_buffer: list[ParsedEvent] = []

        def flush_scroll_buffer():
            if not scroll_buffer:
                return
            first_scroll = scroll_buffer[0]
            last_scroll = scroll_buffer[-1]
            max_y = max(s.details["scroll_y"] for s in scroll_buffer)
            min_y = min(s.details["scroll_y"] for s in scroll_buffer)
            duration = last_scroll.relative_s - first_scroll.relative_s
            actions.append({
                "time": first_scroll.relative_s,
                "action": "SCROLL",
                "scroll_from": scroll_buffer[0].details["scroll_y"],
                "scroll_to": last_scroll.details["scroll_y"],
                "max_depth": max_y,
                "duration_s": round(duration, 2),
                "direction": "down" if last_scroll.details["scroll_y"] > first_scroll.details["scroll_y"] else "up",
            })
            scroll_buffer.clear()

        for event in self.events:
            if event.event_type == "MOUSE_MOVE":
                continue  # Skip mouse moves for high-level view

            if event.event_type == "SCROLL":
                scroll_buffer.append(event)
                continue

            # Flush scroll buffer before processing non-scroll events
            flush_scroll_buffer()

            if event.event_type == "CLICK":
                d = event.details
                actions.append({
                    "time": event.relative_s,
                    "action": "CLICK",
                    "target_tag": d.get("tag"),
                    "target_text": d.get("text", "")[:50],
                    "target_href": d.get("href", ""),
                    "target_class": d.get("class", "")[:60],
                    "coordinates": (d.get("x"), d.get("y")),
                })

            elif event.event_type == "INPUT":
                d = event.details
                actions.append({
                    "time": event.relative_s,
                    "action": "TYPE",
                    "target_tag": d.get("tag"),
                    "text_length": d.get("text_length"),
                    "is_masked": d.get("is_masked"),
                    "target_placeholder": d.get("placeholder", ""),
                })

            elif event.event_type == "API_CALL":
                actions.append({
                    "time": event.relative_s,
                    "action": "API_CALL",
                    "path": event.details.get("path"),
                    "full_path": event.details.get("full_path"),
                })

            elif event.event_type == "PAGE_META":
                actions.append({
                    "time": event.relative_s,
                    "action": "PAGE_LOAD",
                    "url": event.details.get("url"),
                    "viewport": f"{event.details.get('width')}x{event.details.get('height')}",
                })

        # Flush remaining scroll buffer
        flush_scroll_buffer()

        return actions


def parse_recording(json_path: str, output_path: str | None = None) -> dict:
    """
    Parse a PostHog recording JSON and return structured output.
    Optionally saves to output_path.
    """
    parser = SessionParser(json_path)
    events = parser.parse()
    actions = parser.get_high_level_actions()
    profile = parser.get_user_profile()

    output = {
        "user_profile": profile,
        "session_duration_s": parser.relative_time(parser.snapshots[-1]["timestamp"]),
        "total_raw_events": len(events),
        "high_level_actions": actions,
    }

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"  Saved parsed session to {output_path}")

    return output


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import PARSED_DIR, ensure_data_dirs

    ensure_data_dirs()

    if len(sys.argv) < 2:
        print("Usage: python -m pipeline.stage1_parse <recording.json>")
        sys.exit(1)

    json_path = sys.argv[1]
    session_id = Path(json_path).stem.replace("export-", "").replace("-ph-recording", "")
    output_path = str(PARSED_DIR / f"parsed_{session_id}.json")

    result = parse_recording(json_path, output_path)
    print(f"Parsed {result['total_raw_events']} raw events -> {len(result['high_level_actions'])} high-level actions")
    print(f"User: {result['user_profile']['age_group']}, {result['user_profile']['country']}")
    print(f"Duration: {result['session_duration_s']}s")

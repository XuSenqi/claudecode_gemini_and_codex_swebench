"""Save per-instance run artifacts under a run output directory."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


SESSION_ID_RE = re.compile(r"session id:\s*([0-9a-f-]+)", re.IGNORECASE)
WORKDIR_RE = re.compile(r"workdir:\s*(.+)", re.IGNORECASE)
TOKENS_USED_RE = re.compile(r"tokens used\s*\n([\d,]+)", re.IGNORECASE)


def extract_codex_session_info(stderr: str) -> Tuple[Optional[str], Optional[str]]:
    """Parse session id and workdir from Codex exec stderr banner."""
    session_id = None
    cwd = None
    if stderr:
        m = SESSION_ID_RE.search(stderr)
        if m:
            session_id = m.group(1)
        m = WORKDIR_RE.search(stderr)
        if m:
            cwd = m.group(1).strip()
    return session_id, cwd


def find_codex_rollout(session_id: str) -> Optional[Path]:
    """Locate Codex rollout jsonl for a session id under CODEX_HOME."""
    codex_home = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
    sessions_root = codex_home / "sessions"
    if not sessions_root.exists():
        return None
    matches = sorted(sessions_root.rglob(f"*{session_id}*.jsonl"))
    return matches[-1] if matches else None


def load_rollout_events(rollout_path: Path) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for line in rollout_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        events.append(json.loads(line))
    return events


def rollout_to_session(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flatten Codex rollout jsonl into a session event array."""
    session: List[Dict[str, Any]] = []
    for event in events:
        row: Dict[str, Any] = {
            "type": event.get("type"),
            "timestamp": event.get("timestamp"),
            "ordinal": event.get("ordinal"),
        }
        payload = event.get("payload")
        if isinstance(payload, dict):
            row.update(payload)
        elif payload is not None:
            row["payload"] = payload
        session.append(row)
    return session


def rollout_to_stream(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build a stream-oriented view from Codex rollout events."""
    stream: List[Dict[str, Any]] = []
    for event in events:
        etype = event.get("type")
        payload = event.get("payload")
        if not isinstance(payload, dict):
            continue
        if etype == "session_meta":
            stream.append(
                {
                    "type": "system",
                    "subtype": "init",
                    "timestamp": event.get("timestamp"),
                    "cwd": payload.get("cwd"),
                    "session_id": payload.get("session_id") or payload.get("id"),
                    "model_provider": payload.get("model_provider"),
                    "originator": payload.get("originator"),
                    "source": payload.get("source"),
                    "cli_version": payload.get("cli_version"),
                }
            )
        elif etype == "turn_context":
            stream.append({"type": "turn_context", "timestamp": event.get("timestamp"), **payload})
        elif etype == "response_item":
            stream.append({"type": "response_item", "timestamp": event.get("timestamp"), **payload})
        elif etype == "event_msg":
            stream.append({"type": "event_msg", "timestamp": event.get("timestamp"), **payload})
        elif etype == "world_state":
            stream.append({"type": "world_state", "timestamp": event.get("timestamp"), **payload})
    return stream


def save_instance_artifacts(
    output_dir: Path,
    instance_id: str,
    *,
    cli_output: Dict[str, Any],
    backend: str,
) -> Path:
    """Write {instance_id}/{instance_id}.{session,stream}.json."""
    instance_dir = output_dir / instance_id
    instance_dir.mkdir(parents=True, exist_ok=True)

    session_events: List[Dict[str, Any]] = []
    stream_events: List[Dict[str, Any]] = []

    session_id = None
    if backend == "codex":
        session_id, _ = extract_codex_session_info(cli_output.get("stderr") or "")
        if session_id:
            rollout = find_codex_rollout(session_id)
            if rollout:
                rollout_events = load_rollout_events(rollout)
                session_events = rollout_to_session(rollout_events)
                stream_events = rollout_to_stream(rollout_events)

    prefix = instance_dir / instance_id
    (prefix.with_suffix(".session.json")).write_text(
        json.dumps(session_events, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (prefix.with_suffix(".stream.json")).write_text(
        json.dumps(stream_events, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return instance_dir


def run_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

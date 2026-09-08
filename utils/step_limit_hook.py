#!/usr/bin/env python3
"""Codex PreToolUse hooks: per-run execution guards.

Wired in by utils/codex_interface.py. Codex invokes this script before every
tool call, passing a JSON payload on stdin that includes a per-run unique
`session_id`. Two guards share one invocation:

1. Step limit (CODE_SWE_MAX_STEPS, default 0 = unlimited):
   deny once total tool-call count exceeds the limit.

2. Repetition loop guard (CODE_SWE_MAX_REPEATS, default 0 = disabled):
   fingerprint each call (tool_name + tool_input). When the same fingerprint
   appears N consecutive times, the model is stuck in a loop -> deny all
   further calls and instruct it to produce a final answer.

   Repeating the *same* call is useless: identical input produces identical
   output, so re-running can never add information. Deliberately NOT matched:
   identical output with *different* commands (that can be legitimate
   progress, e.g. a flaky test becoming green).

Fails open: any error (bad JSON, unwritable state) allows the tool call.
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path


def _int_env(name: str, default: int = 0) -> int:
    try:
        return int(os.environ.get(name, str(default)) or default)
    except ValueError:
        return default


def _state_dir() -> Path:
    return Path(os.environ.get("CODE_SWE_STEP_STATE_DIR", "/tmp/codex_step_counts"))


def _prune_old(state_dir: Path, max_age_s: int = 86400) -> None:
    """Best-effort cleanup of state files older than a day."""
    try:
        now = time.time()
        for f in state_dir.glob("*"):
            try:
                if now - f.stat().st_mtime > max_age_s:
                    f.unlink()
            except OSError:
                pass
    except OSError:
        pass


def _fingerprint(payload: dict) -> str:
    """Stable digest of what is being called, ignoring volatile ids."""
    key = json.dumps(
        {
            "tool_name": payload.get("tool_name"),
            "tool_input": payload.get("tool_input"),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _deny(reason: str) -> None:
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }))
    sys.exit(0)


def main() -> None:
    step_limit = _int_env("CODE_SWE_MAX_STEPS")
    repeat_limit = _int_env("CODE_SWE_MAX_REPEATS")

    try:
        payload = json.load(sys.stdin)
    except Exception:
        sys.exit(0)  # fail open

    session_id = payload.get("session_id") or "unknown"
    state_dir = _state_dir()
    fingerprint = _fingerprint(payload)

    # ---- 1. step limit ----
    count = 0
    if step_limit > 0:
        counter_file = state_dir / f"{session_id}.count"
        try:
            if counter_file.exists():
                count = int(counter_file.read_text().strip() or "0")
        except (OSError, ValueError):
            count = 0
        count += 1
        if count > step_limit:
            _deny(
                f"STEP LIMIT REACHED ({step_limit} tool calls). "
                "This and all further tool calls are denied. "
                "Do not attempt any more tool calls. Produce your final answer now, "
                "summarizing the work completed so far."
            )

    # ---- 2. repetition loop guard ----
    if repeat_limit > 0:
        hist_file = state_dir / f"{session_id}.history"
        history = []
        looped = False
        try:
            if hist_file.exists():
                history = json.loads(hist_file.read_text() or "[]")
                if not isinstance(history, list):
                    history = []
        except (OSError, ValueError):
            history = []

        # A "1" sentinel as the last entry marks "loop already detected":
        # every further tool call is denied unconditionally.
        if history and history[-1] == "1":
            _deny(
                "REPEATED IDENTICAL TOOL CALL LOOP PREVIOUSLY DETECTED. "
                "This and all further tool calls are denied. "
                "Stop retrying. Produce your final answer now, summarizing the "
                "work completed so far and any obstacle you could not get past."
            )

        # Count how many identical fingerprints trail the history
        consecutive = 0
        for fp in reversed(history):
            if fp == fingerprint:
                consecutive += 1
            else:
                break

        if consecutive >= repeat_limit:
            # Mark the session as looped; all later calls (identical or not)
            # are denied so the model must wrap up.
            history.append("1")
            try:
                state_dir.mkdir(parents=True, exist_ok=True)
                hist_file.write_text(json.dumps(history))
            except OSError:
                pass
            _deny(
                f"REPEATED IDENTICAL TOOL CALL x{consecutive + 1} "
                f"(limit {repeat_limit}). The same call with the same arguments "
                "has been made repeatedly and re-running it cannot produce new "
                "information. This and all further tool calls are denied. "
                "Stop retrying. Produce your final answer now, summarizing the "
                "work completed so far and any obstacle you could not get past."
            )

        history.append(fingerprint)
        # keep only the tail needed by the check (+1 for the sentinel slot)
        history = history[-(repeat_limit + 1):]

    # ---- persist state ----
    try:
        state_dir.mkdir(parents=True, exist_ok=True)
        if step_limit > 0:
            (state_dir / f"{session_id}.count").write_text(str(count))
        if repeat_limit > 0:
            (state_dir / f"{session_id}.history").write_text(json.dumps(history))
        _prune_old(state_dir)
    except OSError:
        sys.exit(0)  # fail open


if __name__ == "__main__":
    main()

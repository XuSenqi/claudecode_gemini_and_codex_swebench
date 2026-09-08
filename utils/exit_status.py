"""Map a prediction/CLI outcome onto mini-swe-agent-style exit statuses."""

from __future__ import annotations

import re
from typing import Dict, Optional

# CLI interfaces write this exact phrase on subprocess.TimeoutExpired.
_INSTANCE_TIMEOUT_RE = re.compile(
    r"command timed out after \d+ minutes",
    re.IGNORECASE,
)


def classify_exit_status(prediction: Dict, cli_result: Optional[Dict] = None) -> str:
    stderr = ((cli_result or {}).get("stderr") or "")
    error = prediction.get("error") or ""
    combined = f"{error}\n{stderr}"
    # Only the harness/CLI wall-clock timeout, not SQLite comments or pytest
    # `timeout=0.1` / "thread timed out waiting for the lock".
    if _INSTANCE_TIMEOUT_RE.search(combined):
        return "TimeLimitExceeded"
    lowered = combined.lower()
    if "repeated identical" in lowered or "repeated action" in lowered:
        return "RepeatedAction"
    if error:
        if "failed to set up" in error.lower():
            return "SetupFailed"
        return "ExecutionFailed"
    if (prediction.get("prediction") or "").strip():
        return "Submitted"
    return "NoPatch"

#!/usr/bin/env python3
"""Live step counters for the CLIs used as agent backends.

The agent CLIs write a live session log for every invocation:

* ``codex``: a rollout ``*.jsonl`` under ``$CODEX_HOME/sessions/YYYY/MM/DD/``
  (one line per event; ``response_item/function_call`` = one tool step).
* ``claude``: a session ``*.jsonl`` under
  ``~/.claude/projects/<dashified-cwd>/<session-id>.jsonl`` (assistant
  messages with ``tool_use`` content blocks = one tool step).

``BackgroundStepWatcher`` polls a backend-specific log while a CLI runs and
pushes ``Step N`` status messages through a ``ProgressReporter``, so the
batch progress display can show per-instance step counts while instances
are still running (mirroring mini-swe-agent's ``Step 26`` spinner lines).
Both watchers are best-effort: missing logs, races with the CLI creating
files, or unparsable lines just leave the status untouched.
"""

import json
import os
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from utils.progress_display import ProgressReporter

StepCallback = Callable[[int], None]


def dashify(path: str) -> str:
    """Claude slug: a directory path turned into a project dir name.

    Verified against the actual layout on this machine: ``/tmp/swe_bench_x``
    maps to ``-tmp-swe-bench-x`` (slashes and underscores both become dashes;
    the leading ``/`` supplies the single leading dash).
    """
    return str(path).replace("/", "-").replace("_", "-")


def claude_projects_dir() -> Path:
    return Path(os.environ.get("CLAUDE_CONFIG_DIR", str(Path.home() / ".claude"))) / "projects"


def codex_sessions_root() -> Path:
    return Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))) / "sessions"


class _PollingWatcher:
    def __init__(self, poll_interval: float = 2.0):
        self.poll_interval = poll_interval
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.steps = 0

    def _on_line(self, line: str) -> None:
        raise NotImplementedError

    def _tail(self) -> None:
        path, offset = self._find_log()
        while not self._stop_event.is_set():
            if path is None:
                path, offset = self._find_log()
                if path is None:
                    self._stop_event.wait(self.poll_interval)
                    continue
            try:
                size = path.stat().st_size
            except OSError:
                path, offset = None, 0
                continue
            if size < offset:  # rotated/truncated
                offset = 0
            try:
                with open(path, "rb") as f:
                    f.seek(offset)
                    for line in f:
                        self._on_line(line.decode("utf-8", "replace"))
                    offset = f.tell()
            except OSError:
                pass
            self._stop_event.wait(self.poll_interval)

    def _find_log(self) -> tuple[Optional[Path], int]:
        return None, 0

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._tail, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    def __enter__(self) -> "_PollingWatcher":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.stop()
        return False


class CodexRolloutWatcher(_PollingWatcher):
    """Count ``response_item/function_call`` events in a codex rollout file."""

    def __init__(self, sessions_root: Optional[Path] = None, session_id: Optional[str] = None,
                 poll_interval: float = 2.0, started_after: Optional[float] = None):
        super().__init__(poll_interval=poll_interval)
        self.sessions_root = sessions_root or codex_sessions_root()
        self.session_id = session_id
        self.started_after = started_after if started_after is not None else time.time()

    def _find_log(self) -> tuple[Optional[Path], int]:
        """Find the newest rollout jsonl matching the session id (or any
        rollout newer than ``started_after`` when the id is unknown)."""
        if not self.sessions_root.exists():
            return None, 0
        candidates = []
        try:
            paths = self.sessions_root.rglob("*.jsonl")
            for path in paths:
                try:
                    mtime = path.stat().st_mtime
                except OSError:
                    continue
                if mtime < self.started_after - 1:
                    continue
                if self.session_id and self.session_id not in path.name:
                    continue
                candidates.append((mtime, path))
        except OSError:
            return None, 0
        if not candidates:
            return None, 0
        return max(candidates)[1], 0

    def _on_line(self, line: str) -> None:
        text = line.strip()
        if not text:
            return
        try:
            event = json.loads(text)
        except json.JSONDecodeError:
            return
        if (
            event.get("type") == "response_item"
            and isinstance(event.get("payload"), dict)
            and event["payload"].get("type") == "function_call"
        ):
            self.steps += 1


class ClaudeSessionWatcher(_PollingWatcher):
    """Count ``tool_use`` blocks in a claude session jsonl log."""

    def __init__(self, cwd: str, projects_dir: Optional[Path] = None,
                 poll_interval: float = 2.0, started_after: Optional[float] = None):
        super().__init__(poll_interval=poll_interval)
        self.project_dir = (projects_dir or claude_projects_dir()) / dashify(cwd)
        self.started_after = started_after if started_after is not None else time.time()

    def _find_log(self) -> tuple[Optional[Path], int]:
        try:
            paths = list(self.project_dir.glob("*.jsonl"))
        except OSError:
            return None, 0
        candidates = []
        for path in paths:
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            if mtime >= self.started_after - 1:
                candidates.append((mtime, path))
        if not candidates:
            return None, 0
        return max(candidates)[1], 0

    def _on_line(self, line: str) -> None:
        text = line.strip()
        if not text:
            return
        try:
            event = json.loads(text)
        except json.JSONDecodeError:
            return
        if event.get("type") != "assistant":
            return
        message = event.get("message") or {}
        content = message.get("content")
        if isinstance(content, list):
            self.steps += sum(1 for block in content if block.get("type") == "tool_use")


class BackgroundStepWatcher:
    """Watch a CLI's live session log and report ``Step N`` per instance.

    ``reporter.instance_status`` is called only when the step count changes,
    roughly every poll interval, from a daemon thread.
    """

    def __init__(self, backend: str, instance_id: str, reporter: ProgressReporter,
                 cwd: Optional[str] = None, session_id: Optional[str] = None,
                 poll_interval: float = 2.0):
        self.backend = backend
        self.instance_id = instance_id
        self.reporter = reporter
        started_after = time.time()
        if backend == "codex":
            self._watcher = CodexRolloutWatcher(session_id=session_id, poll_interval=poll_interval,
                                                started_after=started_after)
        elif backend == "claude":
            self._watcher = ClaudeSessionWatcher(cwd=cwd or os.getcwd(), poll_interval=poll_interval,
                                                 started_after=started_after)
        else:
            self._watcher = None
        self._last_reported = 0

    def _report(self, steps: int) -> None:
        if steps > self._last_reported:
            self._last_reported = steps
            try:
                self.reporter.instance_status(self.instance_id, f"Step {steps:3d}")
            except Exception:
                pass  # progress reporting must never break a run

    def start(self) -> None:
        if self._watcher is None:
            return
        reporter_thread = threading.Thread(target=self._watch_and_report, daemon=True)
        reporter_thread.start()

    def _watch_and_report(self) -> None:
        self._watcher.start()
        try:
            while self._watcher._thread is not None and self._watcher._thread.is_alive():
                self._report(self._watcher.steps)
                time.sleep(self._watcher.poll_interval)
            self._report(self._watcher.steps)
        finally:
            self._watcher.stop()

    def stop(self) -> None:
        if self._watcher is not None:
            self._report(self._watcher.steps)
            self._watcher.stop()

    def __enter__(self) -> "BackgroundStepWatcher":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.stop()
        return False

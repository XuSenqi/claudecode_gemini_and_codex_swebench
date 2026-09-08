#!/usr/bin/env python3
"""Live batch progress display for SWE-bench runs.

The layout is ported from mini-swe-agent's ``RunBatchProgressManager``
(exit-status table + one spinner line per running instance + overall
progress bar with ETA), adapted to this project's process architecture:

* Worker processes in ``code_swe_agent.py`` append small JSON events to
  ``<run-dir>/progress.jsonl`` (``ProgressReporter``). Appends go through
  ``O_APPEND`` ``os.write`` calls so several processes can share one file.
* Whichever process owns the terminal (``run_benchmark_with_eval.py`` when
  driven from ``swe_bench.py``, or ``code_swe_agent.py`` when launched
  standalone) tails that file with ``ProgressViewer`` and renders it:

  - on a terminal: a rich ``Live`` display with animated spinners,
  - when stdout is a file or pipe (``nohup ... >> test.log``): a snapshot
    of the same render group every ``CODE_SWE_PROGRESS_INTERVAL`` seconds
    (default 30, 0 disables) plus a final frame when the run ends.

Exit statuses are mirrored to ``<run-dir>/exit_statuses.yaml`` as instances
finish, so a killed/interrupted run still leaves its state behind.
"""

import collections
import json
import os
import threading
import time
from datetime import timedelta
from pathlib import Path
from typing import Optional

import yaml
from rich.console import Console, Group
from rich.live import Live
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

DEFAULT_SNAPSHOT_INTERVAL = 30.0


def _shorten_str(s: str, max_len: int, shorten_left: bool = False) -> str:
    """Truncate ``s`` to ``max_len`` and pad it to exactly that width."""
    if not shorten_left:
        s = s[: max_len - 3] + "..." if len(s) > max_len else s
    else:
        s = "..." + s[-max_len + 3 :] if len(s) > max_len else s
    return f"{s:<{max_len}}"


def _snapshot_interval_from_env() -> float:
    raw = os.environ.get("CODE_SWE_PROGRESS_INTERVAL")
    if raw is None:
        return DEFAULT_SNAPSHOT_INTERVAL
    try:
        return max(0.0, float(raw))
    except ValueError:
        return DEFAULT_SNAPSHOT_INTERVAL


class ProgressReporter:
    """Append progress events to a JSONL file, safe across processes.

    One line per event, written with a single ``os.write`` on an ``O_APPEND``
    fd, so writers in different worker processes never corrupt each other.
    """

    def __init__(self, events_path):
        self.events_path = Path(events_path)

    def _emit(self, event: dict) -> None:
        event = {"ts": round(time.time(), 3), **event}
        line = (json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8")
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.events_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(fd, line)
        finally:
            os.close(fd)

    def clear(self) -> None:
        """Truncate leftover events from a previous run in the same directory."""
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.events_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        os.close(fd)

    def init_run(self, num_instances: int, backend: str, model: Optional[str] = None, workers: int = 1) -> None:
        """First event of a run: how many instances will be processed."""
        self._emit(
            {
                "event": "init",
                "num_instances": num_instances,
                "backend": backend,
                "model": model,
                "workers": workers,
            }
        )

    def instance_start(self, instance_id: str) -> None:
        self._emit({"event": "start", "instance_id": instance_id})

    def instance_status(self, instance_id: str, message: str) -> None:
        self._emit({"event": "status", "instance_id": instance_id, "message": message})

    def instance_end(self, instance_id: str, exit_status: str) -> None:
        self._emit({"event": "end", "instance_id": instance_id, "exit_status": exit_status})


class BatchProgressDisplay:
    """Exit-status table + per-instance spinners + overall progress bar.

    Ported from mini-swe-agent's ``RunBatchProgressManager``; the per-instance
    cost column was dropped because the CLIs do not report dollar costs.
    All mutations take an internal lock so a Live refresh thread and the
    event-tailer thread can share the renderables safely.
    """

    _MONOTONIC_AT_EPOCH_OFFSET = time.time() - time.monotonic()

    def __init__(self, num_instances: Optional[int] = None, yaml_report_path=None):
        self._spinner_tasks: dict = {}
        """Map instance ID to the rich task ID of its spinner line."""

        self._lock = threading.RLock()
        self._start_time = time.time()
        self._total_instances = num_instances

        self._instances_by_exit_status = collections.defaultdict(list)
        self._main_progress_bar = Progress(
            SpinnerColumn(spinner_name="dots2"),
            TextColumn("[progress.description]{task.description} (${task.fields[total_cost]})"),
            BarColumn(),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TextColumn("[cyan]{task.fields[eta]}[/cyan]"),
            # Wait 5 min before estimating speed
            speed_estimate_period=60 * 5,
        )
        self._task_progress_bar = Progress(
            SpinnerColumn(spinner_name="dots2"),
            TextColumn("{task.fields[instance_id]}"),
            TextColumn("{task.fields[status]}"),
            TimeElapsedColumn(),
        )
        """Task progress bar for individual instances: one task per instance."""

        self._main_task_id = self._main_progress_bar.add_task(
            "[cyan]Overall Progress", total=num_instances, total_cost="0.00", eta=""
        )

        self.render_group = Group(Table(), self._task_progress_bar, self._main_progress_bar)
        self._yaml_report_path = Path(yaml_report_path) if yaml_report_path else None

    @property
    def n_completed(self) -> int:
        return sum(len(instances) for instances in self._instances_by_exit_status.values())

    def set_total(self, num_instances: int) -> None:
        """Correct the total (e.g. when limit exceeds the dataset size)."""
        with self._lock:
            self._total_instances = num_instances
            self._main_progress_bar.update(self._main_task_id, total=num_instances)

    def _epoch_to_monotonic(self, epoch: float) -> float:
        return epoch - self._MONOTONIC_AT_EPOCH_OFFSET

    def set_start_time(self, start_time_epoch: float) -> None:
        """Anchor elapsed/ETA to when the run actually started (epoch secs)."""
        self._start_time = start_time_epoch
        # rich Progress tracks time on the monotonic clock; convert.
        monotonic_start = self._epoch_to_monotonic(start_time_epoch)
        with self._lock:
            task = next(
                (t for t in self._main_progress_bar.tasks if t.id == self._main_task_id), None
            )
            if task is not None:
                task.start_time = monotonic_start

    def reset_run(self, num_instances: int, start_time_epoch: Optional[float] = None) -> None:
        """Drop prior-run state so a reused events file cannot inflate counts."""
        with self._lock:
            for task_id in list(self._spinner_tasks.values()):
                try:
                    self._task_progress_bar.remove_task(task_id)
                except KeyError:
                    pass
            self._spinner_tasks.clear()
            self._instances_by_exit_status.clear()
            self._total_instances = num_instances
            self._main_progress_bar.update(
                self._main_task_id, completed=0, total=num_instances, eta=""
            )
        if start_time_epoch:
            self.set_start_time(start_time_epoch)
        else:
            self._start_time = time.time()
        self.update_exit_status_table()

    def _get_eta_text(self) -> str:
        """Estimate time remaining based on completed instances so far."""
        try:
            remaining = (self._total_instances or 0) - self.n_completed
            if remaining <= 0:
                return "eta: 0:00:00"
            estimated_remaining = (
                (time.time() - self._start_time) / self.n_completed * remaining
            )
            if estimated_remaining < 0:
                return "eta: 0:00:00"
            return f"eta: {timedelta(seconds=int(estimated_remaining))}"
        except (ZeroDivisionError, TypeError):
            return ""

    def _uncomplete(self, instance_id: str) -> bool:
        """Remove ``instance_id`` from the exit-status table if it was done."""
        removed = False
        for status, instances in list(self._instances_by_exit_status.items()):
            if instance_id not in instances:
                continue
            self._instances_by_exit_status[status] = [i for i in instances if i != instance_id]
            if not self._instances_by_exit_status[status]:
                del self._instances_by_exit_status[status]
            removed = True
        return removed

    def update_exit_status_table(self):
        # We cannot update the existing table in place, so create a new one and
        # assign it back into the render group.
        t = Table()
        t.add_column("Exit Status")
        t.add_column("Count", justify="right", style="bold cyan")
        t.add_column("Most recent instances", no_wrap=True)
        with self._lock:
            # Sort by number of instances in descending order
            sorted_items = sorted(self._instances_by_exit_status.items(), key=lambda x: len(x[1]), reverse=True)
            for status, instances in sorted_items:
                instances_str = _shorten_str(", ".join(reversed(instances)), 55)
                t.add_row(status, str(len(instances)), instances_str)
        self.render_group.renderables[0] = t

    def update_instance_status(self, instance_id: str, message: str):
        with self._lock:
            task_id = self._spinner_tasks.get(instance_id)
            if task_id is None:
                return
            self._task_progress_bar.update(
                task_id,
                status=_shorten_str(message, 30),
                instance_id=_shorten_str(instance_id, 25, shorten_left=True),
            )

    def on_instance_start(self, instance_id: str, started_at: Optional[float] = None):
        with self._lock:
            if self._uncomplete(instance_id):
                self._main_progress_bar.update(
                    self._main_task_id, completed=self.n_completed, eta=self._get_eta_text()
                )
            if instance_id in self._spinner_tasks:
                return
            self._spinner_tasks[instance_id] = self._task_progress_bar.add_task(
                description=f"Task {instance_id}",
                status="Task initialized",
                total=None,
                instance_id=instance_id,
            )
            if started_at:
                task = next(
                    (t for t in self._task_progress_bar.tasks if t.id == self._spinner_tasks[instance_id]),
                    None,
                )
                if task is not None:
                    task.start_time = self._epoch_to_monotonic(float(started_at))
        self.update_exit_status_table()

    def on_instance_end(self, instance_id: str, exit_status: Optional[str]) -> None:
        with self._lock:
            already_done = self._uncomplete(instance_id)
            self._uncomplete(instance_id)
            self._instances_by_exit_status[exit_status].append(instance_id)
            task_id = self._spinner_tasks.pop(instance_id, None)
            if task_id is not None:
                try:
                    self._task_progress_bar.remove_task(task_id)
                except KeyError:
                    pass
            self._main_progress_bar.update(
                self._main_task_id, completed=self.n_completed, eta=self._get_eta_text()
            )
        self.update_exit_status_table()
        self._save_overview_data_yaml()

    def on_uncaught_exception(self, instance_id: str, exception: Exception) -> None:
        self.on_instance_end(instance_id, f"Uncaught {type(exception).__name__}")

    def print_report(self, console: Optional[Console] = None) -> None:
        """Print a one-line summary of the instances processed so far."""
        console = console or Console()
        with self._lock:
            parts = [
                f"{status or 'Unknown'}: {len(instances)}"
                for status, instances in sorted(
                    self._instances_by_exit_status.items(), key=lambda x: str(x[0])
                )
            ]
        summary = ", ".join(parts) if parts else "no instances completed yet"
        console.print(f"Completed {self.n_completed}/{self._total_instances} instances ({summary})")

    def _get_overview_data(self) -> dict:
        """Get the instances grouped by exit status."""
        return {
            # convert defaultdict to dict because of serialization
            "instances_by_exit_status": dict(self._instances_by_exit_status),
        }

    def _save_overview_data_yaml(self) -> None:
        """Save a yaml report of the instances and their exit statuses."""
        if self._yaml_report_path is None:
            return
        with self._lock:
            self._yaml_report_path.write_text(yaml.dump(self._get_overview_data(), indent=4))


class ProgressViewer:
    """Tail ``progress.jsonl`` and render it through a BatchProgressDisplay.

    On a terminal the render group is attached to a rich ``Live`` display
    (animated spinners, 4 refreshes/s). When stdout is not a terminal (pipe or
    ``nohup`` log), a snapshot of the whole render group is printed every
    ``snapshot_interval`` seconds so ``tail -f`` still shows progress.
    """

    def __init__(
        self,
        events_path,
        num_instances: Optional[int] = None,
        yaml_report_path=None,
        console: Optional[Console] = None,
        snapshot_interval: Optional[float] = None,
        live: Optional[bool] = None,
    ):
        self.events_path = Path(events_path)
        self.console = console or Console()
        self.display = BatchProgressDisplay(num_instances, yaml_report_path=yaml_report_path)
        if snapshot_interval is None:
            snapshot_interval = _snapshot_interval_from_env()
        self.snapshot_interval = max(0.0, float(snapshot_interval))
        self.use_live = self.console.is_terminal if live is None else bool(live)
        self._live: Optional[Live] = None
        self._stop_event = threading.Event()
        self._tail_thread: Optional[threading.Thread] = None
        self._snapshot_thread: Optional[threading.Thread] = None
        self._file = None
        self._file_key = None
        self._offset = 0

    # ---- event file tailing ----

    def _open_events_file(self) -> bool:
        """(Re)open the events file; handles create/unlink/recreate races."""
        try:
            st = self.events_path.stat()
        except OSError:
            return False
        key = (st.st_dev, st.st_ino)
        if self._file is None or key != self._file_key:
            try:
                if self._file is not None:
                    self._file.close()
                self._file = open(self.events_path, "rb")
                self._offset = 0
                self._file_key = key
            except OSError:
                return False
        elif st.st_size < self._offset:
            # Same file truncated in place: restart from the beginning.
            self._file.seek(0)
            self._offset = 0
        return True

    def _dispatch(self, event: dict) -> None:
        etype = event.get("event")
        instance_id = event.get("instance_id")
        if etype == "init":
            started = event.get("ts")
            start_epoch = float(started) if isinstance(started, (int, float)) and started > 0 else None
            self.display.reset_run(int(event.get("num_instances") or 0), start_epoch)
        elif etype == "start":
            started = event.get("ts")
            start_epoch = float(started) if isinstance(started, (int, float)) and started > 0 else None
            self.display.on_instance_start(instance_id, started_at=start_epoch)
        elif etype == "status":
            self.display.update_instance_status(instance_id, event.get("message", ""))
        elif etype == "end":
            self.display.on_instance_end(instance_id, event.get("exit_status"))

    def pump(self) -> int:
        """Read and dispatch newly appended events; returns how many."""
        with self.display._lock:
            if not self._open_events_file():
                return 0
            dispatched = 0
            while True:
                line = self._file.readline()
                if not line:
                    break
                self._offset += len(line)
                text = line.decode("utf-8", "replace").strip()
                if not text:
                    continue
                try:
                    event = json.loads(text)
                except json.JSONDecodeError:
                    continue  # torn write; skip
                self._dispatch(event)
                dispatched += 1
            return dispatched

    def _tail_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.pump()
            except Exception:
                pass  # the display must never take the benchmark down
            self._stop_event.wait(0.2)

    def _snapshot_loop(self) -> None:
        while not self._stop_event.wait(self.snapshot_interval):
            try:
                self.print_snapshot()
            except Exception:
                pass

    # ---- rendering ----

    def print_snapshot(self) -> None:
        with self.display._lock:
            self.console.print(f"[dim]── progress at {time.strftime('%H:%M:%S')} ──[/dim]")
            self.console.print(self.display.render_group)

    # ---- context manager ----

    def __enter__(self) -> "ProgressViewer":
        try:
            self.pump()
        except Exception:
            pass
        if self.use_live:
            self._live = Live(self.display.render_group, console=self.console, refresh_per_second=4)
            self._live.start()
        else:
            self.print_snapshot()
        self._tail_thread = threading.Thread(target=self._tail_loop, daemon=True)
        self._tail_thread.start()
        if not self.use_live and self.snapshot_interval > 0:
            self._snapshot_thread = threading.Thread(target=self._snapshot_loop, daemon=True)
            self._snapshot_thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._stop_event.set()
        for thread in (self._tail_thread, self._snapshot_thread):
            if thread is not None:
                thread.join(timeout=3.0)
        try:
            self.pump()  # drain any events written in the meantime
        except Exception:
            pass
        if self._live is not None:
            self._live.stop()  # leaves the final frame on screen
            self._live = None
        else:
            self.print_snapshot()
        self.display.print_report(console=self.console)
        if self._file is not None:
            try:
                self._file.close()
            finally:
                self._file = None
        return False

import json
import os
import time
from pathlib import Path

import pytest
from rich.console import Console

from utils.progress_display import (
    BatchProgressDisplay,
    ProgressReporter,
    ProgressViewer,
)
from utils.session_watchers import (
    BackgroundStepWatcher,
    ClaudeSessionWatcher,
    CodexRolloutWatcher,
    dashify,
)


@pytest.fixture
def events_path(tmp_path):
    return tmp_path / "progress.jsonl"


@pytest.fixture
def yaml_path(tmp_path):
    return tmp_path / "exit_statuses.yaml"


class TestProgressReporter:
    def test_events_are_valid_jsonl(self, events_path):
        reporter = ProgressReporter(events_path)
        reporter.init_run(2, "codex", "gpt-5", workers=2)
        reporter.instance_start("astropy__astropy-12907")
        reporter.instance_status("astropy__astropy-12907", "Step 3")
        reporter.instance_end("astropy__astropy-12907", "Submitted")

        lines = events_path.read_text().strip().splitlines()
        assert len(lines) == 4
        events = [json.loads(line) for line in lines]
        assert events[0]["event"] == "init"
        assert events[0]["num_instances"] == 2
        assert events[1] == {
            "event": "start",
            "instance_id": "astropy__astropy-12907",
            "ts": events[1]["ts"],
        }
        assert events[2]["message"] == "Step 3"
        assert events[3]["exit_status"] == "Submitted"
        for event in events:
            assert "ts" in event

    def test_clear_truncates_previous_events(self, events_path):
        reporter = ProgressReporter(events_path)
        reporter.init_run(2, "codex", workers=2)
        reporter.instance_end("old", "Submitted")
        reporter.clear()
        reporter.init_run(3, "codex", workers=3)
        events = [json.loads(line) for line in events_path.read_text().splitlines() if line.strip()]
        assert [e["event"] for e in events] == ["init"]
        assert events[0]["num_instances"] == 3
        assert events[0]["workers"] == 3

    def test_multiple_writers_do_not_corrupt_lines(self, events_path):
        # Simulate two worker processes writing concurrently via O_APPEND.
        reporter_a = ProgressReporter(events_path)
        reporter_b = ProgressReporter(events_path)
        reporters = [reporter_a, reporter_b] * 5
        for i, reporter in enumerate(reporters):
            reporter.instance_status(f"inst-{i}", "status")
        lines = events_path.read_text().strip().splitlines()
        assert len(lines) == 10
        assert all(json.loads(line)["event"] == "status" for line in lines)


class TestBatchProgressDisplay:
    def test_lifecycle_updates_counts_and_completed(self):
        display = BatchProgressDisplay(num_instances=3)
        assert display.n_completed == 0

        display.on_instance_start("a")
        display.on_instance_start("b")
        assert len(display._spinner_tasks) == 2

        display.update_instance_status("a", "Step 4")
        task_id = display._spinner_tasks["a"]
        task = display._task_progress_bar.tasks[0]
        assert task.fields["status"].startswith("Step 4")

        display.on_instance_end("a", "Submitted")
        display.on_instance_end("b", "NoPatch")
        assert display.n_completed == 2
        assert display._instances_by_exit_status["Submitted"] == ["a"]
        assert "a" not in display._spinner_tasks
        # main bar advanced by 2
        main_task = display._main_progress_bar.tasks[0]
        assert main_task.completed == 2

    def test_eta_empty_before_any_completion(self):
        display = BatchProgressDisplay(num_instances=5)
        assert display._get_eta_text() == ""

    def test_yaml_report_written_on_end(self, tmp_path):
        yaml_path = tmp_path / "statuses.yaml"
        display = BatchProgressDisplay(num_instances=1, yaml_report_path=yaml_path)
        display.on_instance_start("x")
        display.on_instance_end("x", "Submitted")
        import yaml

        data = yaml.safe_load(yaml_path.read_text())
        assert data == {"instances_by_exit_status": {"Submitted": ["x"]}}

    def test_exit_status_table_shows_counts(self):
        display = BatchProgressDisplay(num_instances=3)
        display.on_instance_start("a")
        display.on_instance_end("a", "Submitted")
        display.update_exit_status_table()
        # render the group and inspect the table text
        from rich.console import Console

        console = Console(record=True, width=120, no_color=True, force_terminal=False)
        console.print(display.render_group)
        text = console.export_text()
        assert "Submitted" in text
        assert "Exit Status" in text and "Count" in text
        assert "Most recent instances" in text


class TestProgressViewer:
    def _write_events(self, events_path, events):
        with open(events_path, "a", encoding="utf-8") as f:
            for event in events:
                f.write(json.dumps(event) + "\n")

    def test_pump_dispatches_events(self, events_path, yaml_path):
        viewer = ProgressViewer(
            events_path, num_instances=2, yaml_report_path=yaml_path, live=False, snapshot_interval=0
        )
        self._write_events(
            events_path,
            [
                {"event": "start", "instance_id": "a", "ts": 1},
                {"event": "status", "instance_id": "a", "message": "Step 2", "ts": 2},
            ],
        )
        assert viewer.pump() == 2
        assert "a" in viewer.display._spinner_tasks

        # incremental append is picked up by the next pump
        self._write_events(events_path, [{"event": "end", "instance_id": "a", "exit_status": "Submitted", "ts": 3}])
        assert viewer.pump() == 1
        assert viewer.display.n_completed == 1

    def test_context_manager_snapshots_and_drains(self, events_path, yaml_path):
        console = Console(record=True, width=120, force_terminal=False, no_color=True)
        viewer = ProgressViewer(
            events_path, num_instances=1, yaml_report_path=yaml_path,
            console=console, live=False, snapshot_interval=0,
        )
        self._write_events(events_path, [{"event": "end", "instance_id": "a", "exit_status": "NoPatch", "ts": 1}])
        with viewer:
            pass
        output = console.export_text()
        assert "NoPatch" in output
        assert "1/1" in output  # completion report line

    def test_init_event_updates_total(self, events_path):
        viewer = ProgressViewer(events_path, num_instances=5, live=False, snapshot_interval=0)
        self._write_events(events_path, [{"event": "init", "num_instances": 7, "ts": 1}])
        viewer.pump()
        assert viewer.display._total_instances == 7

    def test_init_event_resets_prior_run_counts(self, events_path):
        viewer = ProgressViewer(events_path, num_instances=3, live=False, snapshot_interval=0)
        self._write_events(
            events_path,
            [
                {"event": "init", "num_instances": 3, "ts": 1},
                {"event": "end", "instance_id": "old-a", "exit_status": "Submitted", "ts": 2},
                {"event": "end", "instance_id": "old-b", "exit_status": "Submitted", "ts": 3},
                {"event": "end", "instance_id": "old-c", "exit_status": "Submitted", "ts": 4},
                {"event": "end", "instance_id": "old-d", "exit_status": "Submitted", "ts": 5},
                {"event": "init", "num_instances": 3, "ts": 10},
                {"event": "start", "instance_id": "new-a", "ts": 11},
            ],
        )
        viewer.pump()
        assert viewer.display._total_instances == 3
        assert viewer.display.n_completed == 0
        assert viewer.display._main_progress_bar.tasks[0].completed == 0
        assert "new-a" in viewer.display._spinner_tasks
        assert "old-a" not in viewer.display._spinner_tasks

    def test_duplicate_end_does_not_inflate_count(self):
        display = BatchProgressDisplay(num_instances=1)
        display.on_instance_start("a")
        display.on_instance_end("a", "Submitted")
        display.on_instance_end("a", "Submitted")
        assert display.n_completed == 1
        assert display._instances_by_exit_status["Submitted"] == ["a"]
        assert display._main_progress_bar.tasks[0].completed == 1

    def test_eta_is_zero_when_completed_exceeds_total(self):
        display = BatchProgressDisplay(num_instances=1)
        display._start_time = time.time() - 60
        display._instances_by_exit_status["Submitted"] = ["a", "b"]
        assert display._get_eta_text() == "eta: 0:00:00"

    def test_start_event_ts_anchors_spinner_elapsed(self, events_path):
        viewer = ProgressViewer(events_path, num_instances=1, live=False, snapshot_interval=0)
        started = time.time() - 120
        self._write_events(
            events_path,
            [{"event": "start", "instance_id": "a", "ts": started}],
        )
        viewer.pump()
        task = viewer.display._task_progress_bar.tasks[0]
        assert task.elapsed >= 110

    def test_init_event_anchors_start_time(self, events_path):
        import time as time_mod

        viewer = ProgressViewer(events_path, num_instances=2, live=False, snapshot_interval=0)
        start_epoch = time_mod.time() - 600
        self._write_events(
            events_path,
            [{"event": "init", "num_instances": 2, "ts": start_epoch}]
            + [
                {"event": "end", "instance_id": f"i{k}", "exit_status": "Submitted"}
                for k in range(2)
            ],
        )
        viewer.pump()
        assert viewer.display._start_time == pytest.approx(start_epoch)
        task = viewer.display._main_progress_bar.tasks[0]
        # overall elapsed should be ~600s, not ~0
        assert task.elapsed >= 590
        # with everything completed, no time remains (0 remaining instances)
        assert viewer.display._get_eta_text() == "eta: 0:00:00"

    def test_truncated_line_is_skipped(self, events_path):
        viewer = ProgressViewer(events_path, live=False, snapshot_interval=0)
        with open(events_path, "a", encoding="utf-8") as f:
            f.write('{"event": "start", "instance_i')  # torn write, no newline
        assert viewer.pump() == 0

    def test_missing_file_is_tolerated(self, tmp_path):
        viewer = ProgressViewer(tmp_path / "nonexistent.jsonl", live=False, snapshot_interval=0)
        assert viewer.pump() == 0


class TestCodexRolloutWatcher:
    def _write_rollout_line(self, path, event):
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")

    def test_counts_function_calls(self, tmp_path, monkeypatch):
        sessions = tmp_path / "sessions"
        rollout = sessions / "2026" / "09" / "04" / "rollout-2026-09-04T00-00-00-abc.jsonl"
        rollout.parent.mkdir(parents=True)
        watcher = CodexRolloutWatcher(sessions_root=sessions, started_after=time.time() - 60)
        self._write_rollout_line(rollout, {"type": "session_meta", "payload": {"session_id": "abc"}})
        self._write_rollout_line(rollout, {"type": "response_item", "payload": {"type": "function_call"}})
        self._write_rollout_line(rollout, {"type": "response_item", "payload": {"type": "message"}})
        self._write_rollout_line(rollout, {"type": "response_item", "payload": {"type": "function_call"}})
        self._write_rollout_line(rollout, {"type": "event_msg", "payload": {"type": "token_count"}})
        assert watcher._find_log()[0] == rollout
        # feed lines directly through the parsing hook
        for line in rollout.read_text().splitlines():
            watcher._on_line(line)
        assert watcher.steps == 2

    def test_old_rollouts_ignored(self, tmp_path):
        sessions = tmp_path / "sessions"
        rollout = sessions / "old.jsonl"
        rollout.parent.mkdir(parents=True)
        rollout.write_text("")
        os.utime(rollout, (time.time() - 3600, time.time() - 3600))
        watcher = CodexRolloutWatcher(sessions_root=sessions, started_after=time.time())
        assert watcher._find_log()[0] is None

    def test_session_id_filters_candidates(self, tmp_path):
        sessions = tmp_path / "sessions"
        sessions.mkdir()
        other = sessions / "rollout-other-id.jsonl"
        target = sessions / "rollout-target-id.jsonl"
        for p in (other, target):
            p.write_text("")
            os.utime(p, (time.time(), time.time()))
        watcher = CodexRolloutWatcher(sessions_root=sessions, session_id="target-id", started_after=time.time() - 60)
        assert watcher._find_log()[0] == target


class TestClaudeSessionWatcher:
    def test_counts_tool_use_blocks(self, tmp_path):
        projects = tmp_path / "projects"
        project_dir = projects / dashify("/tmp/swe_bench_x")
        project_dir.mkdir(parents=True)
        log = project_dir / "session1.jsonl"
        assistant = {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "text", "text": "hi"},
                    {"type": "tool_use", "name": "Bash"},
                ]
            },
        }
        plain = {"type": "assistant", "message": {"content": [{"type": "text", "text": "done"}]}}
        user = {"type": "user", "message": {"content": "obs"}}
        with open(log, "w", encoding="utf-8") as f:
            f.write(json.dumps(user) + "\n")
            f.write(json.dumps(assistant) + "\n")
            f.write(json.dumps(plain) + "\n")
        os.utime(log, (time.time(), time.time()))
        watcher = ClaudeSessionWatcher(cwd="/tmp/swe_bench_x", projects_dir=projects, started_after=time.time() - 60)
        assert watcher._find_log()[0] == log
        for line in log.read_text().splitlines():
            watcher._on_line(line)
        assert watcher.steps == 1

    def test_dashify(self):
        assert dashify("/tmp/swe_bench_x") == "-tmp-swe-bench-x"


class TestBackgroundStepWatcher:
    def test_reports_steps_to_progress_reporter(self, events_path, monkeypatch):
        reporter = ProgressReporter(events_path)

        class FakeWatcher:
            def __init__(self):
                self.steps = 0
                self.poll_interval = 0.01
                self._thread = None

            def start(self):
                self._running = True

            def stop(self):
                self._running = False

        fake = FakeWatcher()
        watcher = BackgroundStepWatcher("codex", "inst-1", reporter)
        watcher._watcher = fake

        watcher._report(1)
        watcher._report(1)  # unchanged -> no duplicate event
        watcher._report(3)  # jumps ahead
        lines = events_path.read_text().strip().splitlines()
        messages = [json.loads(line).get("message") for line in lines]
        assert messages == ["Step   1", "Step   3"]

    def test_unsupported_backend_is_noop(self, events_path):
        watcher = BackgroundStepWatcher("gemini", "inst-1", ProgressReporter(events_path))
        assert watcher._watcher is None
        watcher.start()
        watcher.stop()
        assert not events_path.exists()

    def test_no_reporter_is_tolerated(self):
        # gemini (no watcher) + None reporter: constructing/starting/stopping
        # must not raise
        watcher = BackgroundStepWatcher("gemini", "x", None)
        watcher.start()
        watcher.stop()

"""End-to-end wiring test for the batch progress display.

Simulates a 2-worker run of ``code_swe_agent.run_on_dataset`` with fake CLI
backends and local git repos, then asserts the progress events, the live
display state, and per-instance run.log files all line up.
"""

import json
import os
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.progress_display import BatchProgressDisplay, ProgressViewer


REPO_ROOT = Path(__file__).resolve().parent.parent

FAKE_CLI_SCRIPT = """#!/usr/bin/env bash
# fake CLI: sleeps a bit and writes a tool-call line into a session log
sleep {delay}
sleep 0.2
"""


def _make_repo(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "file.txt").write_text(f"content of {path.name}\n")
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "add", "-A"], cwd=path, check=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "init"], cwd=path, check=True)


class _FakeDataset:
    """Minimal stand-in for a HF dataset: selectable and iterable of dicts."""

    def __init__(self, rows):
        self._rows = rows

    def select(self, indices):
        return _FakeDataset([self._rows[i] for i in indices])

    def __len__(self):
        return len(self._rows)

    def __iter__(self):
        return iter([dict(row) for row in self._rows])


@pytest.fixture
def fake_env(tmp_path, monkeypatch):
    """Local git repos + fake CLIs + fake dataset, all under tmp_path."""
    repos = tmp_path / "repos"
    for i in (1, 2, 3):
        _make_repo(repos / f"repo-{i}")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for backend in ("claude", "codex", "gemini"):
        cli = bin_dir / backend
        cli.write_text(FAKE_CLI_SCRIPT.format(delay=0.1))
        cli.chmod(cli.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")

    dataset = [
        {
            "instance_id": f"fake__repo-{i}",
            "repo": f"fake/repo-{i}",
            "base_commit": "HEAD",
            "problem_statement": f"Fix bug {i}",
        }
        for i in (1, 2, 3)
    ]

    monkeypatch.setattr(
        "code_swe_agent.load_dataset",
        lambda name, split="test": _FakeDataset(dataset),
    )

    # redirect repo clones to the local fixtures
    def fake_setup(self, instance):
        repo = instance["repo"].split("/")[-1]
        return str(repos / repo)

    monkeypatch.setattr("code_swe_agent.CodeSWEAgent.setup_repository", fake_setup)
    return repos, dataset


def test_run_on_dataset_emits_progress_events(fake_env, tmp_path, monkeypatch):
    from code_swe_agent import CodeSWEAgent

    output_dir = tmp_path / "run"
    agent = CodeSWEAgent(backend="claude", output_dir=output_dir)
    predictions = agent.run_on_dataset("fake/dataset", limit=3, workers=2)

    assert len(predictions) == 3
    # the fake CLI made no edits -> all instances end as NoPatch
    for prediction in predictions:
        assert prediction["instance_id"].startswith("fake__repo-")
        assert "error" not in prediction

    events_path = output_dir / "progress.jsonl"
    assert events_path.exists()
    events = [json.loads(line) for line in events_path.read_text().splitlines() if line.strip()]
    kinds = [e["event"] for e in events]
    assert kinds[0] == "init"
    assert events[0]["num_instances"] == 3
    assert events[0]["workers"] == 2
    assert kinds.count("end") == 3
    statuses = [e["exit_status"] for e in events if e["event"] == "end"]
    assert statuses == ["NoPatch"] * 3
    # every instance that ran has a start event
    started = [e["instance_id"] for e in events if e["event"] == "start"]
    assert sorted(started) == ["fake__repo-1", "fake__repo-2", "fake__repo-3"]

    # exit-status yaml report exists
    yaml_files = list(output_dir.glob("exit_statuses_*.yaml"))
    assert len(yaml_files) == 1
    import yaml as yaml_mod

    data = yaml_mod.safe_load(yaml_files[0].read_text())
    # completion order is racy with parallel workers; membership is what matters
    assert sorted(data["instances_by_exit_status"]["NoPatch"]) == sorted(started)

    # each worker wrote its console output to run.log
    for i in (1, 2, 3):
        log = output_dir / f"fake__repo-{i}" / "run.log"
        assert log.exists()
        assert "Processing fake__repo" in log.read_text()

    # predictions were still saved
    pred_file = output_dir / "predictions.jsonl"
    assert pred_file.exists()
    assert len(pred_file.read_text().strip().splitlines()) == 3


def test_run_on_dataset_truncates_stale_progress_file(fake_env, tmp_path, monkeypatch):
    """Reusing -o must not replay leftover progress.jsonl from a previous run."""
    from code_swe_agent import CodeSWEAgent

    output_dir = tmp_path / "run"
    output_dir.mkdir()
    stale = output_dir / "progress.jsonl"
    stale.write_text(
        json.dumps({"event": "init", "num_instances": 3, "workers": 2, "ts": 1}) + "\n"
        + json.dumps({"event": "end", "instance_id": "old-1", "exit_status": "Submitted", "ts": 2}) + "\n"
        + json.dumps({"event": "end", "instance_id": "old-2", "exit_status": "Submitted", "ts": 3}) + "\n"
        + json.dumps({"event": "end", "instance_id": "old-3", "exit_status": "Submitted", "ts": 4}) + "\n"
        + json.dumps({"event": "end", "instance_id": "old-4", "exit_status": "Submitted", "ts": 5}) + "\n"
    )
    agent = CodeSWEAgent(backend="claude", output_dir=output_dir)
    agent.run_on_dataset("fake/dataset", limit=3, workers=1)

    events = [json.loads(line) for line in (output_dir / "progress.jsonl").read_text().splitlines() if line.strip()]
    assert events[0]["event"] == "init"
    assert events[0]["num_instances"] == 3
    assert all(e.get("instance_id") != "old-1" for e in events)
    assert [e["event"] for e in events].count("init") == 1
    assert [e["event"] for e in events].count("end") == 3


def test_viewer_replays_run_events(fake_env, tmp_path):
    """A viewer constructed after the fact can replay a full events file."""
    from code_swe_agent import CodeSWEAgent

    output_dir = tmp_path / "run"
    agent = CodeSWEAgent(backend="claude", output_dir=output_dir)
    agent.run_on_dataset("fake/dataset", limit=3, workers=1)

    viewer = ProgressViewer(
        output_dir / "progress.jsonl", num_instances=3, live=False, snapshot_interval=0
    )
    assert viewer.pump() >= 1 + 3 * 2  # init + start/end per instance (plus statuses)
    assert viewer.display.n_completed == 3
    assert not viewer.display._spinner_tasks  # all spinners removed
    assert viewer.display._total_instances == 3


def test_progress_state_matches_mini_swe_agent_semantics(tmp_path):
    """The display behaves like mini-swe-agent's RunBatchProgressManager."""
    display = BatchProgressDisplay(num_instances=4)
    display.on_instance_start("i1")
    display.on_instance_start("i2")
    display.on_instance_start("i3")
    display.update_instance_status("i1", "Step  10")
    display.on_instance_end("i1", "Submitted")
    display.on_instance_end("i2", "TimeLimitExceeded")
    # uncaught exception in a worker
    display.on_uncaught_exception("i3", RuntimeError("boom"))
    display.on_instance_start("i4")
    display.on_instance_end("i4", None)

    assert display.n_completed == 4
    by_status = {k: v for k, v in display._instances_by_exit_status.items()}
    assert by_status["Submitted"] == ["i1"]
    assert by_status["TimeLimitExceeded"] == ["i2"]
    assert by_status["Uncaught RuntimeError"] == ["i3"]
    assert by_status[None] == ["i4"]
    assert display._total_instances == 4
    # completion report renders every status
    from rich.console import Console

    console = Console(record=True, width=100, no_color=True, force_terminal=False)
    display.print_report(console=console)
    text = console.export_text()
    assert "Completed 4/4" in text
    assert "Submitted: 1" in text
    assert "TimeLimitExceeded: 1" in text

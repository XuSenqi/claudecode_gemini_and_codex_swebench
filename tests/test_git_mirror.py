"""Local git-mirror helpers used by CodeSWEAgent.setup_repository."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from utils.git_mirror import (
    checkout_commit,
    clone_from_mirror,
    ensure_mirror,
    github_url,
    mirror_path,
)


def _init_repo(path: Path) -> str:
    path.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=path, check=True)
    (path / "a.txt").write_text("one\n")
    subprocess.run(["git", "add", "a.txt"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "one"], cwd=path, check=True)
    first = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()
    (path / "a.txt").write_text("two\n")
    subprocess.run(["git", "commit", "-q", "-am", "two"], cwd=path, check=True)
    return first


def test_github_url() -> None:
    assert github_url("astropy/astropy") == "https://github.com/astropy/astropy.git"


def test_mirror_then_two_worktrees(tmp_path: Path) -> None:
    source = tmp_path / "upstream"
    first = _init_repo(source)
    second = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    assert first != second

    root = tmp_path / "mirrors"
    mirror = ensure_mirror("fake/src", str(source), root=root)
    assert mirror == mirror_path("fake/src", root)
    assert (mirror / "HEAD").is_file()

    # Second ensure is a fetch, not a new clone.
    again = ensure_mirror("fake/src", str(source), root=root)
    assert again == mirror

    work_a = tmp_path / "inst-a"
    work_b = tmp_path / "inst-b"
    clone_from_mirror(mirror, work_a)
    clone_from_mirror(mirror, work_b)
    checkout_commit(work_a, first)
    checkout_commit(work_b, second)
    assert (work_a / "a.txt").read_text() == "one\n"
    assert (work_b / "a.txt").read_text() == "two\n"

    # Fresh stamp skips a second fetch even if upstream is gone.
    shutil.rmtree(source)
    ensure_mirror("fake/src", str(source), root=root)


def test_checkout_missing_commit_raises(tmp_path: Path) -> None:
    source = tmp_path / "upstream"
    _init_repo(source)
    mirror = ensure_mirror("fake/src", str(source), root=tmp_path / "mirrors")
    work = tmp_path / "work"
    clone_from_mirror(mirror, work)
    with pytest.raises(RuntimeError, match="Failed to checkout"):
        checkout_commit(work, "deadbeef" * 5)

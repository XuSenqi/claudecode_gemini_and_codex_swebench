"""Local git mirrors so SWE-bench instances don't clone GitHub every time."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

try:
    import fcntl
except ImportError:  # pragma: no cover - non-Unix
    fcntl = None  # type: ignore[assignment]


DEFAULT_MIRROR_ROOT = Path.home() / ".cache" / "swe_git_mirrors"
DEFAULT_FETCH_TTL_S = 24 * 3600


def _fetch_ttl_s() -> float:
    raw = os.environ.get("SWE_GIT_MIRROR_FETCH_TTL_S", "").strip()
    if not raw:
        return float(DEFAULT_FETCH_TTL_S)
    try:
        return max(0.0, float(raw))
    except ValueError:
        return float(DEFAULT_FETCH_TTL_S)


def _stamp_path(mirror: Path) -> Path:
    return mirror / "swe_mirror_fetched"


def _mirror_is_fresh(mirror: Path, ttl_s: float) -> bool:
    if ttl_s <= 0:
        return False
    stamp = _stamp_path(mirror)
    try:
        age = time.time() - stamp.stat().st_mtime
    except OSError:
        return False
    return age < ttl_s


def _mark_fetched(mirror: Path) -> None:
    stamp = _stamp_path(mirror)
    stamp.write_text(str(time.time()), encoding="utf-8")


class _FileLock:
    def __init__(self, path: Path):
        self.path = path
        self._fd: Optional[int] = None

    def __enter__(self) -> "_FileLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o644)
        if fcntl is not None:
            fcntl.flock(self._fd, fcntl.LOCK_EX)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._fd is None:
            return
        if fcntl is not None:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        os.close(self._fd)
        self._fd = None


def mirror_enabled() -> bool:
    raw = os.environ.get("SWE_GIT_MIRROR", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def mirror_root() -> Path:
    raw = os.environ.get("SWE_GIT_MIRROR_ROOT", "").strip()
    return Path(raw).expanduser() if raw else DEFAULT_MIRROR_ROOT


def mirror_path(repo_name: str, root: Optional[Path] = None) -> Path:
    return (root or mirror_root()) / f"{repo_name}.git"


def github_url(repo_name: str) -> str:
    return f"https://github.com/{repo_name}.git"


def _run(args: list[str], *, cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        args,
        capture_output=True,
        text=True,
        cwd=str(cwd) if cwd else None,
    )


def _is_git_dir(path: Path) -> bool:
    return (path / "HEAD").is_file() and (path / "config").is_file()


def ensure_mirror(
    repo_name: str,
    clone_url: str,
    root: Optional[Path] = None,
    *,
    force_fetch: bool = False,
) -> Path:
    """Create or update a bare mirror of ``clone_url``; safe for parallel workers."""
    dest = mirror_path(repo_name, root)
    lock = dest.with_name(dest.name + ".lock")
    ttl_s = _fetch_ttl_s()
    with _FileLock(lock):
        if _is_git_dir(dest):
            if not force_fetch and _mirror_is_fresh(dest, ttl_s):
                return dest
            print(f"Updating git mirror {repo_name}")
            result = _run(["git", "--git-dir", str(dest), "fetch", "--prune", "--tags"])
            if result.returncode != 0:
                print(f"Warning: git fetch mirror {repo_name} failed: {result.stderr}")
            else:
                _mark_fetched(dest)
            return dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            shutil.rmtree(dest)
        print(f"Creating git mirror {repo_name} -> {dest}")
        result = _run(["git", "clone", "--mirror", clone_url, str(dest)])
        if result.returncode != 0:
            raise RuntimeError(f"Failed to create mirror {repo_name}: {result.stderr}")
        _mark_fetched(dest)
        return dest


def clone_from_mirror(mirror: Path, dest: Path) -> None:
    """Clone a working tree that reuses the mirror object store (no network)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    result = _run(
        [
            "git",
            "clone",
            "--reference",
            str(mirror),
            str(mirror),
            str(dest),
        ]
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to clone from mirror {mirror}: {result.stderr}")


def fetch_worktree(repo: Path) -> None:
    result = _run(["git", "fetch", "--prune", "--tags"], cwd=repo)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to fetch {repo}: {result.stderr}")


def checkout_commit(repo: Path, commit: str) -> None:
    result = _run(["git", "checkout", "--force", commit], cwd=repo)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to checkout {commit}: {result.stderr}")

import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

DEFAULT_INSTANCE_TIMEOUT = int(os.environ.get("CODE_SWE_INSTANCE_TIMEOUT", "7200"))

# Optional per-instance guards enforced via a Codex PreToolUse hook
# (utils/step_limit_hook.py):
#   CODE_SWE_MAX_STEPS  - total tool-call limit          (0 = unlimited)
#   CODE_SWE_MAX_REPEATS - identical-call loop guard      (0 = disabled)
# Either being > 0 enables the hook.
MAX_STEPS = int(os.environ.get("CODE_SWE_MAX_STEPS", "0") or "0")
MAX_REPEATS = int(os.environ.get("CODE_SWE_MAX_REPEATS", "0") or "0")
_STEP_LIMIT_HOOK = Path(__file__).resolve().parent / "step_limit_hook.py"


def _step_limit_config_args() -> List[str]:
    """Build `-c` CLI args enabling the guard hook, or [] when disabled.

    The hook lives in this repo (not ~/.codex), so we inline it as a single
    PreToolUse matcher group using raw TOML syntax (verified against codex
    v0.152: `hooks.PreToolUse=[ { matcher = ... } ]`). JSON-style values are
    rejected as "invalid type: string, expected a sequence".
    `--dangerously-bypass-hook-trust` is required in non-interactive exec mode
    because nobody can review/trust the hook there.
    """
    if (MAX_STEPS <= 0 and MAX_REPEATS <= 0) or not _STEP_LIMIT_HOOK.exists():
        return []
    hook_toml = (
        f'[ {{ matcher = "*", hooks = [ {{ type = "command", '
        f'command = "{_sys_python()} {_STEP_LIMIT_HOOK}", timeout = 10 }} ] }} ]'
    )
    return [
        "--dangerously-bypass-hook-trust",
        f"-c hooks.PreToolUse={hook_toml}",
    ]


def _sys_python() -> str:
    return sys.executable or "python3"


class CodexCodeInterface:
    """Interface for interacting with the Codex CLI."""

    def __init__(self):
        """Ensure the Codex CLI is available on the system."""
        try:
            result = subprocess.run(["codex", "--version"], capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(
                    "Codex CLI not found. Please ensure 'codex' is installed and in PATH"
                )
        except FileNotFoundError:
            raise RuntimeError(
                "Codex CLI not found. Please ensure 'codex' is installed and in PATH"
            )

    def execute_code_cli(self, prompt: str, cwd: str, model: str = None) -> Dict[str, any]:
        """Execute Codex via CLI and capture the response."""
        try:
            original_cwd = os.getcwd()
            os.chdir(cwd)
            cmd = [
                "codex",
                "exec",
                "--skip-git-repo-check",
                "--dangerously-bypass-approvals-and-sandbox",
            ]
            cmd.extend(_step_limit_config_args())
            if model:
                cmd.extend(["--model", model])
            env = os.environ.copy()
            env.setdefault("TERM", "xterm-256color")
            result = subprocess.run(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=DEFAULT_INSTANCE_TIMEOUT,
                env=env,
            )
            os.chdir(original_cwd)
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            }
        except subprocess.TimeoutExpired:
            os.chdir(original_cwd)
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Command timed out after {DEFAULT_INSTANCE_TIMEOUT // 60} minutes",
                "returncode": -1,
            }
        except Exception as e:
            os.chdir(original_cwd)
            return {
                "success": False,
                "stdout": "",
                "stderr": str(e),
                "returncode": -1,
            }

    def extract_file_changes(self, response: str) -> List[Dict[str, str]]:
        """Extract file changes from Codex's response (placeholder)."""
        return []

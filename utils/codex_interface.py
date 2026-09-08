import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

DEFAULT_INSTANCE_TIMEOUT = int(os.environ.get("CODE_SWE_INSTANCE_TIMEOUT", "7200"))

# How long a model stream may sit with no tokens before Codex aborts that
# HTTP request and retries it. 0 leaves Codex's own default (typically 5 min).
IDLE_TIMEOUT_MS = int(os.environ.get("CODE_SWE_CODEX_IDLE_TIMEOUT_MS", "120000") or "0")
REQUEST_RETRIES = int(os.environ.get("CODE_SWE_CODEX_REQUEST_RETRIES", "10") or "0")
STREAM_RETRIES = int(os.environ.get("CODE_SWE_CODEX_STREAM_RETRIES", "10") or "0")

# Optional per-instance guards enforced via a Codex PreToolUse hook
# (utils/step_limit_hook.py):
#   CODE_SWE_MAX_STEPS  - total tool-call limit          (0 = unlimited)
#   CODE_SWE_MAX_REPEATS - identical-call loop guard      (0 = disabled)
# Either being > 0 enables the hook.
MAX_STEPS = int(os.environ.get("CODE_SWE_MAX_STEPS", "0") or "0")
MAX_REPEATS = int(os.environ.get("CODE_SWE_MAX_REPEATS", "0") or "0")
_STEP_LIMIT_HOOK = Path(__file__).resolve().parent / "step_limit_hook.py"

_PROVIDER_RE = re.compile(r'(?m)^model_provider\s*=\s*"([^"]+)"')


def _codex_home() -> Path:
    return Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex")))


def _active_model_provider(config_text: Optional[str] = None) -> str:
    """Name of the configured model_provider, or Codex's built-in ``openai``."""
    if config_text is None:
        config_path = _codex_home() / "config.toml"
        try:
            config_text = config_path.read_text(encoding="utf-8")
        except OSError:
            return "openai"
    match = _PROVIDER_RE.search(config_text)
    return match.group(1) if match else "openai"


def _retry_config_args(provider: Optional[str] = None) -> List[str]:
    """``-c`` overrides so a hung model stream is aborted and retried.

    Codex already retries sampling requests on stream disconnect
    (``stream_idle_timeout_ms`` / ``stream_max_retries`` / ``request_max_retries``
    on ``ModelProviderInfo``). Without these, a silent gateway holds the
    instance until ``CODE_SWE_INSTANCE_TIMEOUT``.
    """
    if IDLE_TIMEOUT_MS <= 0 and REQUEST_RETRIES <= 0 and STREAM_RETRIES <= 0:
        return []
    name = provider if provider is not None else _active_model_provider()
    args: List[str] = []
    prefix = f"model_providers.{name}"
    if IDLE_TIMEOUT_MS > 0:
        args.extend(["-c", f"{prefix}.stream_idle_timeout_ms={IDLE_TIMEOUT_MS}"])
    if REQUEST_RETRIES > 0:
        args.extend(["-c", f"{prefix}.request_max_retries={REQUEST_RETRIES}"])
    if STREAM_RETRIES > 0:
        args.extend(["-c", f"{prefix}.stream_max_retries={STREAM_RETRIES}"])
    return args


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
            cmd.extend(_retry_config_args())
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

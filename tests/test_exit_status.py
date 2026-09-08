"""Exit-status classification for SWE-bench instance outcomes."""

from utils.exit_status import classify_exit_status as _classify_exit_status


def test_sqlite_comment_is_not_time_limit() -> None:
    prediction = {
        "prediction": "diff --git a/x.py b/x.py\n",
        "error": "",
    }
    cli = {
        "stderr": (
            "another thread timed out waiting for the lock the be released.\n"
            "test_results.next(timeout=0.1)\n"
        )
    }
    assert _classify_exit_status(prediction, cli) == "Submitted"


def test_harness_timeout_is_time_limit() -> None:
    prediction = {"prediction": "", "error": "Execution failed: Command timed out after 120 minutes"}
    cli = {"stderr": "Command timed out after 120 minutes"}
    assert _classify_exit_status(prediction, cli) == "TimeLimitExceeded"


def test_empty_patch_is_no_patch() -> None:
    assert _classify_exit_status({"prediction": "", "error": ""}, {"stderr": ""}) == "NoPatch"

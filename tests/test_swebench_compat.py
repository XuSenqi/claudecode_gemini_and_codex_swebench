import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from swe_bench import swebench_version_supported


def test_supported_5x_versions():
    assert swebench_version_supported("5.0.2")
    assert swebench_version_supported("5.1.0")
    assert swebench_version_supported("5.9.9")


def test_rejected_old_and_major():
    assert not swebench_version_supported("4.1.0")
    assert not swebench_version_supported("5.0.1")
    assert not swebench_version_supported("2.0.0")
    assert not swebench_version_supported("6.0.0")
    assert not swebench_version_supported("6.1.0")

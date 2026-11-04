"""Regression tests for scripts/check_known_failures.py.

The checker exists to make "is this failure mine?" mechanical, so its own
contract is what matters: a listed failure is known, anything else fails the
run, and an entry that no longer fails is reported instead of quietly staying.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_known_failures.py"

BASELINE = """\
# comment lines are ignored
tests/a/test_known.py::test_listed   # catalog drift
tests/a/test_file.py::*              # whole-file flake
"""

PYTEST_OUTPUT = """\
=========================== short test summary info ============================
FAILED tests/a/test_known.py::test_listed - AssertionError: assert 1 == 2
FAILED tests/a/test_file.py::test_any - RuntimeError: boom
"""

JUNIT = """\
<?xml version="1.0" encoding="utf-8"?>
<testsuites>
  <testsuite name="pytest">
    <testcase classname="tests.a.test_known" name="test_listed">
      <failure message="assert 1 == 2">traceback</failure>
    </testcase>
    <testcase classname="tests.a.test_known" name="test_green"/>
  </testsuite>
</testsuites>
"""


def _run(tmp_path: Path, report: str, *extra: str) -> subprocess.CompletedProcess[str]:
    baseline = tmp_path / "baseline.txt"
    baseline.write_text(BASELINE, encoding="utf-8")
    path = tmp_path / "report.txt"
    path.write_text(report, encoding="utf-8")
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--baseline", str(baseline), str(path), *extra],
        capture_output=True,
        text=True,
    )


def test_listed_failures_pass_and_unlisted_fail(tmp_path: Path) -> None:
    known_only = _run(tmp_path, PYTEST_OUTPUT)
    assert known_only.returncode == 0, known_only.stdout + known_only.stderr
    # Both entries matched real failures, so neither may be reported stale.
    assert "stale" not in known_only.stdout, known_only.stdout

    unlisted = PYTEST_OUTPUT + "FAILED tests/b/test_new.py::test_mine - AssertionError\n"
    result = _run(tmp_path, unlisted)
    assert result.returncode == 1
    assert "UNEXPECTED tests/b/test_new.py::test_mine" in result.stdout
    assert "known     tests/a/test_known.py::test_listed" in result.stdout
    assert "stale" not in result.stdout, result.stdout


def test_junit_report_is_understood(tmp_path: Path) -> None:
    result = _run(tmp_path, JUNIT)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "known     tests/a/test_known.py::test_listed" in result.stdout


def test_stale_entry_reported_and_optionally_fatal(tmp_path: Path) -> None:
    green = "1 passed in 0.01s\n"
    lenient = _run(tmp_path, green)
    assert lenient.returncode == 0
    assert "stale" in lenient.stdout

    strict = _run(tmp_path, green, "--fail-on-stale")
    assert strict.returncode == 1
    assert "stale baseline entry" in strict.stdout

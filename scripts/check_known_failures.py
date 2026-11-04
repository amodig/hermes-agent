#!/usr/bin/env python3
"""Compare test failures against the checked-in pre-existing baseline.

CI red is useless as a signal when the same unrelated tests fail on the base
branch: every run needs a manual "is this mine?" investigation, and reviewers
start merging past red because red is normal. The baseline makes that comparison
mechanical.

Usage:
    scripts/check_known_failures.py REPORT             # pytest output or junit xml
    scripts/check_known_failures.py --list             # print the baseline
    scripts/check_known_failures.py --fail-on-stale REPORT

Exit codes: 0 when every failure is listed (or there are none), 1 when at least
one failure is not in the baseline, 2 on usage error.

Baseline format (`.github/known-pre-existing-failures.txt`): one entry per line,
``<path>::<test>`` or ``<path>::*`` for a whole file, followed by an optional
``# reason``. When adding an entry, record the base commit where it was seen —
an entry without provenance is indistinguishable from a test someone disabled.
"""

from __future__ import annotations

import argparse
import fnmatch
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

DEFAULT_BASELINE = Path(__file__).resolve().parent.parent / ".github" / "known-pre-existing-failures.txt"

# pytest short summary lines: ``FAILED tests/x.py::test_y - AssertionError``
# and ``ERROR tests/x.py::test_y``. ``-`` separates the traceback's first line.
_FAILED_LINE = re.compile(r"^(?:FAILED|ERROR)\s+([\w./-]+\.py::[\w\[\]().:'\- ]+?)(?:\s+-\s|\s*$)")


def load_baseline(path: Path) -> list[tuple[str, str]]:
    """Return (pattern, reason) pairs; blank lines and comments are ignored."""
    entries: list[tuple[str, str]] = []
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        pattern, _, reason = line.partition("#")
        entries.append((pattern.strip(), reason.strip()))
    return entries


def failures_from_text(text: str) -> set[str]:
    found: set[str] = set()
    for line in text.splitlines():
        match = _FAILED_LINE.match(line.strip())
        if match:
            found.add(match.group(1).strip())
    return found


def _junit_id(case: ET.Element) -> str:
    classname = case.get("classname") or ""
    name = case.get("name") or ""
    if classname:
        path = classname.replace(".", "/") + ".py"
        return f"{path}::{name}"
    return name


def failures_from_junit(text: str) -> set[str]:
    root = ET.fromstring(text)
    found: set[str] = set()
    for case in root.iter("testcase"):
        if case.find("failure") is not None or case.find("error") is not None:
            found.add(_junit_id(case))
    return found


def collect_failures(report: Path) -> set[str]:
    text = report.read_text(encoding="utf-8-sig", errors="replace")
    if text.lstrip().startswith("<"):
        return failures_from_junit(text)
    return failures_from_text(text)


def matches(failure: str, patterns: list[tuple[str, str]]) -> str | None:
    for pattern, reason in patterns:
        if fnmatch.fnmatchcase(failure, pattern):
            return reason
    return None


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("report", nargs="?", type=Path, help="pytest output or junit xml")
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--list", action="store_true", help="print the baseline and exit")
    parser.add_argument("--fail-on-stale", action="store_true", help="also fail when a baseline entry no longer fails")
    args = parser.parse_args(argv)

    if not args.baseline.is_file():
        print(f"check_known_failures: no baseline at {args.baseline}", file=sys.stderr)
        return 2
    patterns = load_baseline(args.baseline)

    if args.list:
        for pattern, reason in patterns:
            print(f"{pattern}\t{reason}")
        return 0

    if args.report is None:
        parser.print_usage(sys.stderr)
        return 2
    if not args.report.is_file():
        print(f"check_known_failures: no report at {args.report}", file=sys.stderr)
        return 2

    try:
        failures = collect_failures(args.report)
    except ET.ParseError as exc:
        print(f"check_known_failures: unreadable junit report: {exc}", file=sys.stderr)
        return 2

    unexpected = []
    known = []
    for failure in sorted(failures):
        reason = matches(failure, patterns)
        if reason is None:
            unexpected.append(failure)
        else:
            known.append((failure, reason))

    for failure, reason in known:
        print(f"known     {failure}" + (f"  ({reason})" if reason else ""))
    for failure in unexpected:
        print(f"UNEXPECTED {failure}")

    # `matches` returns the reason, which may legitimately be "" — compare with
    # None, never truthiness, or every matched entry reports as stale.
    stale = [
        pattern for pattern, _ in patterns
        if not any(matches(f, [(pattern, "")]) is not None for f in failures)
    ]
    if stale:
        label = "STALE" if args.fail_on_stale else "stale    "
        for pattern in stale:
            print(f"{label} {pattern} (baseline entry did not fail in this report)")

    if unexpected:
        print(f"\n{len(unexpected)} unexpected failure(s) — this branch caused them, or the baseline needs an entry with provenance.")
        return 1
    if stale and args.fail_on_stale:
        print(f"\n{len(stale)} stale baseline entry/entries — remove them so the baseline keeps meaning something.")
        return 1
    print(f"\nno unexpected failures ({len(known)} known)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

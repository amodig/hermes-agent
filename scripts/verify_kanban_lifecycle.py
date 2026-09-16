#!/usr/bin/env python3
"""Run the typed Kanban lifecycle proof and emit an update receipt.

The source layer uses the repository's canonical per-file runner when pytest is
available, with a stdlib unittest fallback for lean runtime installations.
The ordered canonical and upgrade suites run in every source or installed
context.  Installed runs execute the same fixtures against an explicitly
selected runtime root.  Active verification is intentionally read-only and
requires a later control-plane probe from the layered verifier.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET


PROTOCOL = 1
REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_MODULES = (
    "tests.hermes_cli.test_kanban_lifecycle_conformance",
    "tests.hermes_cli.test_kanban_lifecycle_upgrade_conformance",
)
TEST_FILES = (
    REPO_ROOT / "tests" / "hermes_cli" / "test_kanban_lifecycle_conformance.py",
    REPO_ROOT / "tests" / "hermes_cli" / "test_kanban_lifecycle_upgrade_conformance.py",
)
SCENARIO_HEADER_RE = re.compile(
    r"^(?P<name>test_[^\s(]+)(?: \((?P<class>[^)]+)\))?"
)

def _policy_files(policy_root: Path) -> list[Path]:
    policy_files = (
        "scripts/apply.sh",
        "scripts/verify.sh",
        "scripts/verify_lifecycle.py",
        "scripts/remnic_plugin.py",
        "scripts/remnic-project-context",
        "scripts/hermes-github",
        "omp/remnic-project-context.ts",
        "admin/loota-dnf",
        "admin/install-package-helper",
        "plugins/consequence_guard/plugin.yaml",
        "plugins/consequence_guard/__init__.py",
    )
    files = [policy_root / relative for relative in policy_files]
    profiles_root = policy_root / "profiles"
    for profile in sorted(path for path in profiles_root.iterdir() if path.is_dir()):
        files.extend(path for path in (profile / "SOUL.md", profile / "config.yaml") if path.is_file())
    architecture = policy_root / "docs" / "ARCHITECTURE.md"
    if architecture.is_file():
        files.append(architecture)
    decisions = policy_root / "docs" / "decisions"
    if decisions.is_dir():
        files.extend(path for path in sorted(decisions.rglob("*")) if path.is_file())
    missing = [path.relative_to(policy_root).as_posix() for path in files if not path.is_file()]
    if missing:
        raise RuntimeError(f"policy artifact is missing: {', '.join(missing)}")
    return sorted(files, key=lambda path: path.relative_to(policy_root).as_posix())


def _policy_digest(policy_root: Path | None) -> str | None:
    if policy_root is None:
        return None
    digest = hashlib.sha256()
    for path in _policy_files(policy_root):
        digest.update(path.relative_to(policy_root).as_posix().encode() + b"\0")
        digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()


def _pythonpath(layer: str, runtime_root: Path) -> str:
    roots = (runtime_root,) if layer == "installed" else (runtime_root, REPO_ROOT)
    return os.pathsep.join(dict.fromkeys(str(value) for value in roots))




def _installed_unittest_runner() -> str:
    return """\
import importlib.util
import sys
import types
import unittest
from pathlib import Path

paths = sys.argv[1:]
if len(paths) != 2:
    raise SystemExit(f"expected two conformance test files, got {len(paths)}")

# The test modules live in the source checkout while production imports must
# resolve exclusively from the selected installed runtime.  Expose only the
# source test namespace through synthetic packages; never add the repository
# root to sys.path.
first = Path(paths[0]).resolve()
tests_package = types.ModuleType("tests")
tests_package.__path__ = [str(first.parent.parent)]
tests_package.__package__ = "tests"
sys.modules["tests"] = tests_package
hermes_cli_package = types.ModuleType("tests.hermes_cli")
hermes_cli_package.__path__ = [str(first.parent)]
hermes_cli_package.__package__ = "tests.hermes_cli"
sys.modules["tests.hermes_cli"] = hermes_cli_package

suite = unittest.TestSuite()
for path in paths:
    module_name = f"tests.hermes_cli.{Path(path).stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load conformance test module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))

result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(0 if result.wasSuccessful() else 1)
"""


def _probe_identity(runtime_root: Path, python: str, *, layer: str) -> dict:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": _pythonpath(layer, runtime_root),
        }
    )
    with tempfile.TemporaryDirectory(prefix="kanban-identity-home-") as home:
        env["HERMES_HOME"] = home
        env["HERMES_KANBAN_HOME"] = home
        result = subprocess.run(
            [python, "-m", "hermes_cli.main", "--runtime-identity"],
            cwd=runtime_root,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    if result.returncode != 0:
        raise RuntimeError(f"runtime identity probe failed: {result.stderr.strip()}")
    try:
        identity = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"runtime identity probe was not JSON: {result.stdout!r}") from exc
    if identity.get("protocol") != PROTOCOL:
        raise RuntimeError(f"unsupported runtime identity protocol: {identity.get('protocol')!r}")
    return identity


def _pytest_available(python: str) -> bool:
    return subprocess.run(
        [python, "-c", "import pytest"], capture_output=True, check=False
    ).returncode == 0


def _parse_scenarios(stdout: str, junit: Path | None) -> list[dict]:
    scenarios: list[dict] = []
    pending: dict | None = None

    def _status(text: str) -> str | None:
        if re.search(r"\bFAIL(?:ED)?\b", text):
            return "failed"
        if re.search(r"\bERROR\b", text):
            return "failed"
        if re.search(r"\bskipped\b", text):
            return "skipped"
        if re.search(r"\bok\b", text):
            return "passed"
        return None

    def _append(entry: dict, status: str | None) -> None:
        scenarios.append({**entry, "status": status or "failed"})

    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if pending is not None and line in {"ok", "FAIL", "ERROR"}:
            _append(pending, "passed" if line == "ok" else "failed")
            pending = None
            continue
        if "..." not in line:
            continue
        prefix, suffix = line.split("...", 1)
        match = SCENARIO_HEADER_RE.match(prefix.strip())
        if not match:
            continue
        if pending is not None:
            _append(pending, None)
        entry = {
            "id": match.group("name"),
            "name": match.group("class") or match.group("name"),
        }
        result = _status(suffix)
        if result is None:
            pending = entry
        else:
            _append(entry, result)
    if pending is not None:
        _append(pending, None)
    if junit is not None and junit.exists():
        scenarios = []
        for case in ET.parse(junit).getroot().iter("testcase"):
            status = "passed"
            if case.find("skipped") is not None:
                status = "skipped"
            elif case.find("failure") is not None or case.find("error") is not None:
                status = "failed"
            scenarios.append(
                {
                    "id": case.get("name") or "unknown",
                    "name": case.get("classname") or case.get("name") or "unknown",
                    "status": status,
                }
            )
    return scenarios


def _run_suite(layer: str, runtime_root: Path, python: str, junit: Path | None) -> tuple[int, str, str, str]:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "HERMES_PYTHON": python,
            "HERMES_CONFORMANCE_RUNTIME_ROOT": str(runtime_root),
            "HERMES_HOME": tempfile.mkdtemp(prefix="kanban-suite-home-"),
            "HERMES_KANBAN_HOME": tempfile.mkdtemp(prefix="kanban-suite-kanban-"),
            "PYTHONPATH": _pythonpath(layer, runtime_root),
        }
    )
    if layer == "source" and _pytest_available(python):
        command = [str(REPO_ROOT / "scripts" / "run_tests.sh"), *map(str, TEST_FILES)]
        runner = "scripts/run_tests.sh"
        cwd = REPO_ROOT
    elif layer == "installed":
        command = [python, "-c", _installed_unittest_runner(), *map(str, TEST_FILES)]
        runner = "python -c <installed conformance runner>"
        cwd = runtime_root
    else:
        command = [python, "-m", "unittest", "-v", *TEST_MODULES]
        runner = "python -m unittest"
        cwd = REPO_ROOT
    result = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
    )
    return result.returncode, result.stdout + result.stderr, runner, env["PYTHONPATH"]


def _run_scenario_probe(layer: str, runtime_root: Path, python: str) -> tuple[int, str]:
    """Recover test ids when the canonical runner only prints file summaries."""
    env = os.environ.copy()
    env.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "HERMES_CONFORMANCE_RUNTIME_ROOT": str(runtime_root),
            "HERMES_HOME": tempfile.mkdtemp(prefix="kanban-scenario-home-"),
            "HERMES_KANBAN_HOME": tempfile.mkdtemp(prefix="kanban-scenario-kanban-"),
            "PYTHONPATH": _pythonpath(layer, runtime_root),
        }
    )
    if layer == "installed":
        command = [python, "-c", _installed_unittest_runner(), *map(str, TEST_FILES)]
    else:
        command = [python, "-m", "unittest", "-v", *TEST_MODULES]
    result = subprocess.run(
        command,
        cwd=runtime_root if layer == "installed" else REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
    )
    return result.returncode, result.stdout + result.stderr



def _write_receipt(path: Path, receipt: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", choices=("source", "installed", "active"), required=True)
    parser.add_argument("--runtime-root", type=Path, default=None)
    parser.add_argument("--policy-root", type=Path, default=None)
    parser.add_argument("--receipt-dir", type=Path, default=REPO_ROOT / ".lifecycle-receipts")
    parser.add_argument("--receipt", type=Path, default=None)
    parser.add_argument("--junit", type=Path, default=None)
    args = parser.parse_args(argv)

    runtime_root = (args.runtime_root or REPO_ROOT).resolve()
    created = datetime.now(timezone.utc).isoformat()
    receipt = {
        "schema_version": 1,
        "created_at": created,
        "reference": {
            "layer": args.layer,
            "protocol": PROTOCOL,
            "runtime_root": str(runtime_root),
            "policy_digest": None,
        },
        "result": {"status": "incomplete", "scenarios": [], "diagnostics": []},
    }
    receipt_path = args.receipt or args.receipt_dir / f"kanban-lifecycle-{args.layer}-{int(time.time())}.json"
    try:
        receipt["reference"]["policy_digest"] = _policy_digest(
            args.policy_root.resolve() if args.policy_root else None
        )
        if args.layer == "active":
            receipt["result"]["diagnostics"].append(
                "active verification requires the layered verifier's live control-plane probe"
            )
            _write_receipt(receipt_path, receipt)
            print(json.dumps(receipt, indent=2, sort_keys=True))
            return 2
        if not runtime_root.is_dir():
            raise RuntimeError(f"runtime root does not exist: {runtime_root}")
        identity = _probe_identity(runtime_root, sys.executable, layer=args.layer)
        if args.layer == "installed":
            reported_root = identity.get("module_root")
            if not reported_root:
                raise RuntimeError("installed runtime identity did not report module_root")
            try:
                observed_root = Path(str(reported_root)).resolve()
            except (OSError, RuntimeError, ValueError) as exc:
                raise RuntimeError(
                    f"installed runtime identity has invalid module_root: {reported_root!r}"
                ) from exc
            if observed_root != runtime_root:
                raise RuntimeError(
                    "installed runtime identity root mismatch: "
                    f"expected {runtime_root}, got {observed_root}"
                )
        receipt["reference"].update(
            {
                "code_sha": identity.get("code_sha"),
                "fingerprint": identity.get("fingerprint"),
                "version": identity.get("version"),
                "module_root": identity.get("module_root"),
            }
        )
        return_code, output, runner, pythonpath = _run_suite(
            args.layer, runtime_root, sys.executable, args.junit
        )
        scenarios = _parse_scenarios(output, args.junit)
        if not scenarios:
            probe_code, probe_output = _run_scenario_probe(
                args.layer, runtime_root, sys.executable,
            )
            scenarios = _parse_scenarios(probe_output, None)
            receipt["reference"]["scenario_runner"] = "python -m unittest -v"
            if return_code == 0 and probe_code != 0:
                return_code = probe_code
                output += "\n" + probe_output
        receipt["reference"]["runner"] = runner
        receipt["reference"]["pythonpath"] = pythonpath
        receipt["result"]["scenarios"] = scenarios
        receipt["result"]["status"] = "pass" if return_code == 0 else "failed"
        if return_code != 0:
            receipt["result"]["diagnostics"].append(output[-4000:])
        receipt["result"]["exit_code"] = return_code
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        receipt["result"]["diagnostics"].append(str(exc))
    _write_receipt(receipt_path, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    status = receipt["result"]["status"]
    return 0 if status == "pass" else 1 if status == "failed" else 2


if __name__ == "__main__":
    raise SystemExit(main())

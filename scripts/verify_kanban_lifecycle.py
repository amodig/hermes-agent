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


def _dispatcher_probe_script() -> str:
    """Return the isolated dispatcher/materialization proof program."""
    return r"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import contextlib

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_dispatch as dispatcher
from hermes_cli import kanban_runtime as runtime
from hermes_cli import kanban_runtime_generation as generations
for finder in tuple(sys.meta_path):
    if getattr(finder, "__module__", "").startswith("__editable__"):
        sys.meta_path.remove(finder)
for name, module in tuple(sys.modules.items()):
    if name.startswith("__editable__") and isinstance(getattr(module, "MAPPING", None), dict):
        sys.modules.pop(name, None)


def _emit(payload):
    print("__HERMES_DISPATCHER_PROBE__" + json.dumps(payload, sort_keys=True))


profile = Path(os.environ["HERMES_HOME"]).resolve()
board_home = Path(os.environ["HERMES_KANBAN_HOME"]).resolve()
runtime_root = Path(sys.argv[1]).resolve()
expected = runtime.RuntimeIdentity.from_value(
    json.loads(os.environ["HERMES_VERIFY_EXPECTED_IDENTITY"])
)
profile.mkdir(parents=True, exist_ok=True)
board_home.mkdir(parents=True, exist_ok=True)
sentinel_token = f"synthetic-token-{uuid.uuid4().hex}"
state_marker = f"synthetic-state-db-{uuid.uuid4().hex}"
env_marker = f"SYNTHETIC_TOKEN={uuid.uuid4().hex}"
for directory in ("sessions", "logs", "state", "cache"):
    target = profile / directory
    target.mkdir(parents=True, exist_ok=True)
    for index in range(64):
        (target / f"seed-{index:03d}.dat").write_text("synthetic profile state\n", encoding="utf-8")
(profile / "state" / "fingerprint-race.bin").write_bytes(b"state" * 2_000_000)
(profile / "sessions" / "secret-sentinel.txt").write_text(
    sentinel_token + "\n", encoding="utf-8"
)
(profile / "state.db").write_bytes(state_marker.encode("ascii"))
(profile / "kanban.db").write_bytes(b"synthetic-kanban-db")
(profile / ".env.sentinel").write_text(env_marker + "\n", encoding="utf-8")

churn_stop = threading.Event()
churn_count = 0


def _churn_profile():
    global churn_count
    tick = 0
    while not churn_stop.is_set():
        tick += 1
        try:
            (profile / "sessions" / f"live-{tick % 32:02d}.log").write_text(
                f"synthetic session update {tick}\n", encoding="utf-8"
            )
            with (profile / "logs" / "gateway.log").open("ab") as stream:
                stream.write(f"synthetic log update {tick}\n".encode("ascii"))
            (profile / "state.db").write_bytes(f"synthetic-state-{tick}".encode("ascii"))
            (profile / "kanban.db").write_bytes(f"synthetic-kanban-{tick}".encode("ascii"))
            with (profile / "state" / "fingerprint-race.bin").open("ab") as stream:
                stream.write(b"x")
            (profile / "cache" / "board.cache").write_bytes(f"cache-{tick}".encode("ascii"))
            churn_count += 1
        except OSError:
            break
        time.sleep(0.001)


storage = profile.parent / "runtime-storage"
generations._runtime_storage_root = lambda: storage
dispatcher._profile_exists_fn = lambda: (lambda _name: True)
proof = {}
generation = None
churn_thread = threading.Thread(target=_churn_profile, daemon=True)
churn_thread.start()
try:
    source_identity = runtime.runtime_identity(runtime_root)
    if not runtime.same_code_identity(expected, source_identity):
        raise RuntimeError("layer runtime identity changed before materialization")

    db_path = board_home / "kanban.db"
    os.environ["HERMES_KANBAN_DB"] = str(db_path)
    os.environ["HERMES_KANBAN_WORKSPACES_ROOT"] = str(board_home / "workspaces")
    conn = kb.connect(db_path)
    try:
        task_id = kb.create_task(
            conn,
            title="layer dispatcher materialization proof",
            assignee="verify",
            initial_status="blocked",
            workspace_kind="scratch",
        )
        kb.unblock_task(conn, task_id)

        def spawn(task, workspace, *, board=None):
            nonlocal_generation = None
            try:
                nonlocal_generation = generations.prepare_runtime_generation(
                    source_identity,
                    workspace=workspace,
                    profile_home=profile,
                    project_plugins_enabled=False,
                )
                worker_env = {**os.environ, **nonlocal_generation.env}
                worker_env.update(
                    {
                        "HERMES_HOME": str(profile),
                        "HERMES_KANBAN_HOME": str(board_home),
                        "HERMES_PROFILE": "verify",
                    }
                )
                bootstrap_path = profile.parent / f"bootstrap-{uuid.uuid4().hex}.json"
                preparation_id = f"verify-{uuid.uuid4().hex}"
                worker_env.update(
                    {
                        "HERMES_KANBAN_BOOTSTRAP_PATH": str(bootstrap_path),
                        "HERMES_KANBAN_PREPARATION_ID": preparation_id,
                        "HERMES_KANBAN_EXPECTED_RUNTIME": json.dumps(
                            nonlocal_generation.identity.as_dict(), sort_keys=True,
                        ),
                        "HERMES_KANBAN_BOOTSTRAP_WAIT": "0",
                    }
                )
                worker = subprocess.run(
                    [*nonlocal_generation.command_prefix, "--runtime-identity"],
                    cwd=workspace,
                    env=worker_env,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=60,
                )
                lines = [line.strip() for line in worker.stdout.splitlines() if line.strip()]
                observed = json.loads(lines[-1]) if lines else {}
                bootstrap = (
                    json.loads(bootstrap_path.read_text(encoding="utf-8"))
                    if bootstrap_path.is_file()
                    else {}
                )
                same_generation = (
                    worker.returncode == 0
                    and isinstance(observed, dict)
                    and runtime.same_code_identity(nonlocal_generation.identity, observed)
                    and observed.get("module_root") == str(runtime_root)
                )
                manifest = generations.generation_manifest(nonlocal_generation.root)
                payload_names = {
                    path.relative_to(nonlocal_generation.root).as_posix()
                    for path in nonlocal_generation.root.rglob("*")
                }
                sentinels = (
                    "sessions/secret-sentinel.txt",
                    "state.db",
                    "kanban.db",
                    ".env.sentinel",
                )
                no_profile_state = not any(
                    name == sentinel or name.endswith("/" + sentinel)
                    for sentinel in sentinels
                    for name in payload_names
                )
                payload_bytes = (
                    path.read_bytes()
                    for path in nonlocal_generation.root.rglob("*")
                    if path.is_file()
                )
                no_profile_state = no_profile_state and not any(
                    marker in content
                    for content in payload_bytes
                    for marker in (
                        sentinel_token.encode("utf-8"),
                        state_marker.encode("ascii"),
                        env_marker.encode("ascii"),
                    )
                )
                worker_ready = bootstrap.get("ready") is True
                proof.update(
                    {
                        "worker_bootstrap": bool(same_generation and worker_ready and no_profile_state),
                        "runtime_identity": observed
                        if isinstance(observed, dict)
                        else nonlocal_generation.identity.as_dict(),
                        "diagnostics": []
                        if same_generation and worker_ready and no_profile_state
                        else ["worker identity or sealed payload proof failed"],
                        "manifest_identity": manifest.get("identity", {}),
                    }
                )
                with contextlib.suppress(OSError):
                    bootstrap_path.unlink()
                if not same_generation:
                    proof["diagnostics"].append("worker did not report the frozen runtime identity")
                if not worker_ready:
                    proof["diagnostics"].append("worker bootstrap did not report ready")
                if not no_profile_state:
                    proof["diagnostics"].append("profile-state sentinel appeared in the sealed payload")
                return dispatcher.WorkerLaunch(
                    pid=int(observed["pid"]),
                    runtime_identity=observed,
                    preparation_id=preparation_id,
                )
            except Exception as exc:
                proof["worker_bootstrap"] = False
                proof["runtime_identity"] = (
                    nonlocal_generation.identity.as_dict()
                    if nonlocal_generation is not None
                    else source_identity.as_dict()
                )
                proof["diagnostics"] = [
                    f"materialization raised {type(exc).__name__}: {str(exc)[:500]}",
                ]
                raise
            finally:
                if nonlocal_generation is not None:
                    generations.cleanup_runtime_generation(nonlocal_generation.root, force=True)

        result = dispatcher.dispatch_once(
            conn,
            spawn_fn=spawn,
            max_spawn=1,
            reconcile_orphans=False,
        )
        proof["dispatcher_spawned"] = any(
            entry[0] == task_id for entry in result.spawned
        )
    finally:
        conn.close()
finally:
    churn_stop.set()
    churn_thread.join(timeout=5)

evidence = {
    "profile_churn": bool(churn_count),
    "worker_bootstrap": bool(proof.get("worker_bootstrap")),
    "runtime_identity": proof.get("runtime_identity") or expected.as_dict(),
}
diagnostics = list(proof.get("diagnostics") or [])
if not proof.get("dispatcher_spawned"):
    diagnostics.append("dispatcher did not materialize a worker launch")
ok = (
    evidence["profile_churn"]
    and evidence["worker_bootstrap"]
    and bool(proof.get("dispatcher_spawned"))
)
_emit({"ok": ok, "diagnostics": diagnostics, "evidence": evidence})
raise SystemExit(0 if ok else 1)
"""


def _dispatcher_materialization_check(
    layer: str,
    runtime_root: Path,
    python: str,
    identity: dict,
    *,
    temp_root: Path | None = None,
) -> dict:
    """Exercise dispatch_once, sealed generation publication, and a worker."""
    evidence = {
        "profile_churn": False,
        "worker_bootstrap": False,
        "runtime_identity": dict(identity),
    }
    diagnostics: list[str] = []
    with tempfile.TemporaryDirectory(
        prefix="kanban-dispatch-proof-",
        dir=str(temp_root) if temp_root is not None else None,
    ) as raw:
        proof_root = Path(raw)
        profile = proof_root / "profile"
        profile.mkdir()
        board = proof_root / "board"
        env = os.environ.copy()
        env.update(
            {
                "PYTHONDONTWRITEBYTECODE": "1",
                "HERMES_HOME": str(profile),
                "HERMES_KANBAN_HOME": str(board),
                "HERMES_PROFILE": "verify",
                "HERMES_ENABLE_PROJECT_PLUGINS": "0",
                "HERMES_VERIFY_EXPECTED_IDENTITY": json.dumps(identity, sort_keys=True),
                "PYTHONPATH": _pythonpath(layer, runtime_root),
            }
        )
        for name in (
            "HERMES_KANBAN_RUNTIME_GENERATION",
            "HERMES_KANBAN_BOOTSTRAP_PATH",
            "HERMES_KANBAN_PREPARATION_ID",
            "HERMES_KANBAN_EXPECTED_RUNTIME",
            "HERMES_KANBAN_BOOTSTRAP_WAIT",
        ):
            env.pop(name, None)
        try:
            result = subprocess.run(
                [python, "-c", _dispatcher_probe_script(), str(runtime_root)],
                cwd=profile,
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=180,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            diagnostics.append(f"dispatcher materialization probe raised {type(exc).__name__}")
            return {
                "status": "failed",
                "diagnostics": diagnostics,
                "evidence": evidence,
            }
        marker = "__HERMES_DISPATCHER_PROBE__"
        payload = None
        for line in reversed(result.stdout.splitlines()):
            if line.startswith(marker):
                try:
                    payload = json.loads(line[len(marker):])
                except json.JSONDecodeError:
                    payload = None
                break
        if not isinstance(payload, dict):
            diagnostics.append(
                f"dispatcher materialization probe exited with status {result.returncode}"
            )
            return {
                "status": "failed",
                "diagnostics": diagnostics,
                "evidence": evidence,
            }
        raw_evidence = payload.get("evidence")
        if isinstance(raw_evidence, dict):
            evidence["profile_churn"] = bool(raw_evidence.get("profile_churn"))
            evidence["worker_bootstrap"] = bool(raw_evidence.get("worker_bootstrap"))
            if isinstance(raw_evidence.get("runtime_identity"), dict):
                evidence["runtime_identity"] = dict(raw_evidence["runtime_identity"])
        diagnostics.extend(
            value for value in payload.get("diagnostics", ())
            if isinstance(value, str)
        )
        required_identity = (
            "protocol",
            "code_sha",
            "version",
            "module_root",
            "fingerprint",
            "generation",
            "dependency_fingerprint",
        )
        observed = evidence["runtime_identity"]
        if any(key not in observed for key in required_identity):
            diagnostics.append("worker evidence omitted runtime identity fields")
        elif any(observed[key] != identity.get(key) for key in required_identity[:5]):
            diagnostics.append("worker evidence does not match the selected layer identity")
        if result.returncode != 0:
            diagnostics.append(
                f"dispatcher materialization probe exited with status {result.returncode}"
            )
        status = (
            "pass"
            if result.returncode == 0
            and bool(payload.get("ok"))
            and evidence["profile_churn"]
            and evidence["worker_bootstrap"]
            and not diagnostics
            else "failed"
        )
    return {
        "status": status,
        "diagnostics": diagnostics,
        "evidence": evidence,
    }


def _empty_materialization_check(identity: dict | None = None) -> dict:
    return {
        "status": "failed",
        "diagnostics": [],
        "evidence": {
            "profile_churn": False,
            "worker_bootstrap": False,
            "runtime_identity": dict(identity or {}),
        },
    }



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
        "schema_version": 2,
        "created_at": created,
        "reference": {
            "layer": args.layer,
            "protocol": PROTOCOL,
            "runtime_root": str(runtime_root),
            "policy_digest": None,
        },
        "result": {"status": "incomplete", "scenarios": [], "diagnostics": []},
    }
    if args.layer != "active":
        receipt["result"]["checks"] = {
            "dispatcher_materialization": _empty_materialization_check(),
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
        receipt["result"]["checks"]["dispatcher_materialization"] = _dispatcher_materialization_check(
            args.layer,
            runtime_root,
            sys.executable,
            identity,
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
        materialization = receipt["result"]["checks"]["dispatcher_materialization"]
        receipt["result"]["status"] = (
            "pass"
            if return_code == 0 and materialization["status"] == "pass"
            else "failed"
        )
        if return_code != 0:
            receipt["result"]["diagnostics"].append(output[-4000:])
        elif materialization["status"] != "pass":
            receipt["result"]["diagnostics"].append(
                "dispatcher materialization check failed"
            )
        receipt["result"]["exit_code"] = return_code
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        receipt["result"]["diagnostics"].append(str(exc))
    _write_receipt(receipt_path, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    status = receipt["result"]["status"]
    return 0 if status == "pass" else 1 if status == "failed" else 2


if __name__ == "__main__":
    raise SystemExit(main())

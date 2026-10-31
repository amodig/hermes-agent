#!/usr/bin/env python3
"""Real-dispatcher smoke for the granted worker verification boundary.

Default execution (``--case all``) retains main + death only, each in its own
disposable fixture home, board and Git repository:

1. ``main``     — dispatcher launch, retained grant, model-facing tools, source
                  inspection (summary/read/diff), an offline verification recipe and a
                  declared PostgreSQL service recipe whose verifier writes and re-reads
                  one fixture row through the generated private endpoint, then a native
                  ``APPROVE`` completion bound to the fixture head.
2. ``death``    — the dispatcher parent SIGKILLs the worker mid-verification, then
                  observes the journal's conmon deadline, payload teardown and cleanup.
3. ``recovery`` — opt-in real implementer/reviewer/tester dispatch, bounded evidence
                  traversal and exact-candidate validation; separate inspect-only
                  and withheld-workspace graphs prove native capability blocking.
                  This proves fixture criteria, never application acceptance.

The harness re-executes itself inside ``autonomous.slice``, so host admission, worker
scope placement and the concurrency slot are the real production checks. Every other
gate — grant, claim, approval hook, source digest, image identity, namespace
attestation, release marker and teardown — is likewise the production path. Images are prepared once into ``--image-store`` from the pinned
digests; nothing is pulled during a normal run.

Usage::

    verify_worker_execution.py --policy-root CTO_ROOT --image-store DIR --prepare-images
    verify_worker_execution.py --policy-root CTO_ROOT --image-store DIR --receipt FILE
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
from io import BytesIO
from pathlib import Path
from typing import Any

# Run from a checkout with any interpreter: the runtime under test is this one.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PYTHON_IMAGE = "docker.io/library/python@sha256:7dff173021016c8829ae58d5ce112ffc388073ebe67f351aeb6ba2909e35f847"
POSTGRES_IMAGE = "docker.io/library/postgres@sha256:ec4dfc6571f04fe0731ce948c1500d676af49a7224392dcc6995cc1ee33a94bb"
KEEPER_IMAGE = PYTHON_IMAGE
IMAGE_TAGS = {
    PYTHON_IMAGE: "localhost/worker-execution:python-fixture",
    POSTGRES_IMAGE: "localhost/worker-execution:postgres-fixture",
}

OFFLINE_RECIPE = {
    "image": PYTHON_IMAGE,
    "argv": ["/usr/local/bin/python3", "-I", "/workspace/check.py"],
    "timeout_seconds": 300,
    "services": [],
}
DB_SERVICE = {
    "id": "db", "image": POSTGRES_IMAGE,
    "argv": [
        "/bin/sh", "-c",
        "/usr/lib/postgresql/17/bin/initdb -D /data --auth-local=trust --auth-host=trust "
        "--username=fixture && exec /usr/lib/postgresql/17/bin/postgres -D /data "
        "-h 127.0.0.1 -p 55432 -k /scratch",
    ],
    "ready_argv": [
        "/usr/lib/postgresql/17/bin/pg_isready", "-h", "127.0.0.1", "-p", "55432",
        "-U", "fixture", "-d", "postgres",
    ],
    "port": 55432,
}
DB_RECIPE = {
    "image": POSTGRES_IMAGE,
    "argv": [
        "/usr/lib/postgresql/17/bin/psql", "-h", "127.0.0.1", "-p", "55432",
        "-U", "fixture", "-d", "postgres", "-v", "ON_ERROR_STOP=1", "-f", "/workspace/probe.sql",
    ],
    "timeout_seconds": 300,
    "services": [DB_SERVICE],
}
STALL_RECIPE = {
    "image": PYTHON_IMAGE,
    # A detached child proves the conmon deadline and the private PID namespace tear
    # down descendants, not just the container init.
    "argv": ["/bin/sh", "-c", "sleep 600 & sleep 600"],
    "timeout_seconds": 90,
    "services": [],
}

CHECK_PY = '''\
import os
import py_compile

ALLOWED_ENV = {
    "HOME", "TMPDIR", "PATH", "LANG", "CI", "HOSTNAME", "PWD", "SHLVL", "_",
    "VERIFY_SERVICE_DB_HOST", "VERIFY_SERVICE_DB_PORT",
}
assert set(os.environ) <= ALLOWED_ENV, sorted(set(os.environ) - ALLOWED_ENV)
assert not any(key.startswith("HERMES_") for key in os.environ)
for forbidden in ("DATABASE_URL", "PGHOST", "PGPORT", "DOCKER_HOST", "SSH_AUTH_SOCK"):
    assert forbidden not in os.environ, forbidden
for path in ("/var/run/docker.sock", "/run/podman/podman.sock", "/run/containers"):
    assert not os.path.exists(path), path
assert os.path.exists("/source/app.py")
assert os.path.exists("/workspace/app.py")
assert not os.path.exists("/workspace/.git")
assert not os.path.exists("/source/.env")
assert sum([1, 2, 3]) == 6
py_compile.compile("/workspace/app.py", cfile="/workspace/app.pyc", doraise=True)
with open("/workspace/app.py", "rb") as handle:
    assert b"return 2" in handle.read()
try:
    with open("/source/tamper.txt", "wb") as handle:
        handle.write(b"x")
except OSError:
    pass
else:
    raise AssertionError("the copied source bind is writable")
assert not os.path.exists("/source/.git")
print("offline-verification-ok")
'''

PROBE_SQL = """\
\\set ON_ERROR_STOP on
DO $$
BEGIN
  IF to_regclass('public.issue76_fixture') IS NOT NULL THEN
    RAISE EXCEPTION 'fixture table already exists';
  END IF;
END $$;
CREATE TABLE issue76_fixture (value integer NOT NULL);
INSERT INTO issue76_fixture (value) VALUES (76);
DO $$
BEGIN
  IF (SELECT count(*) FROM issue76_fixture WHERE value = 76) <> 1 THEN
    RAISE EXCEPTION 'fixture sentinel missing';
  END IF;
END $$;
SELECT value FROM issue76_fixture;
"""


_SCOPE_MARKER = "HERMES_ISSUE76_EXECUTION_PROBE_SCOPED"


_LAST_WORKER: dict[str, Any] = {"pid": None}


def _cancel_worker() -> None:
    pid = _LAST_WORKER.get("pid")
    if not pid:
        return
    try:
        os.kill(int(pid), getattr(signal, "SIGKILL", signal.SIGTERM))
    except (ProcessLookupError, PermissionError, ValueError):
        return
    try:
        from hermes_cli import kanban_worker_runtime as runtime

        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and runtime._pid_alive(int(pid)):
            runtime.reap_worker_zombies()
            time.sleep(0.2)
    except Exception:
        return


def _log_tail(fixture: Path, limit: int = 6000) -> str:
    markers = ("kanban worker", "execution", "verify", "ERROR", "WARNING", "Traceback")
    parts: list[str] = []
    logs = sorted(
        (path for root in (fixture / "kanban", fixture / "hermes") if root.exists()
         for path in root.rglob("*") if path.is_file() and path.suffix == ".log"),
        key=lambda path: path.stat().st_mtime,
    )
    for path in logs[-2:]:
        try:
            text = path.read_bytes().decode("utf-8", "replace")
        except OSError:
            continue
        matched = [line for line in text.splitlines() if any(m in line for m in markers)]
        parts.append(f"[{path.name} markers] " + " || ".join(matched[-20:])[:4000])
        parts.append(f"[{path.name} tail] " + text[-limit // 2:])
    return "\n".join(parts)[-3 * limit:] if parts else "<no logs>"


class SmokeFailure(RuntimeError):
    pass


def _in_autonomous_aggregate() -> bool:
    try:
        text = Path("/proc/self/cgroup").read_text(encoding="utf-8")
    except OSError:
        return False
    return "/autonomous.slice/" in text


def _enter_autonomous_scope() -> None:
    """Re-execute this harness inside the reviewed autonomous aggregate.

    The final host smoke must use the real native worker-scope admission, so a host
    that cannot place the probe refuses with the concrete prerequisite instead of
    falling back to a stub.
    """
    if sys.platform.startswith("linux") and os.environ.get(_SCOPE_MARKER) != "1" and not _in_autonomous_aggregate():
        unit = f"hermes-issue76-exec-smoke-{os.getpid()}-{os.urandom(4).hex()}"
        argv = [
            "systemd-run", "--user", "--scope", "--quiet", "--collect",
            "--expand-environment=no", "--slice=autonomous.slice", "--unit", unit,
            "--", sys.executable, *sys.argv,
        ]
        environment = dict(os.environ)
        environment[_SCOPE_MARKER] = "1"
        uid = os.getuid() if hasattr(os, "getuid") else -1  # windows-footgun: ok — hasattr-guarded; scope setup is Linux-only
        runtime_dir = environment.get("XDG_RUNTIME_DIR") or f"/run/user/{uid}"
        environment["XDG_RUNTIME_DIR"] = runtime_dir
        environment.setdefault("DBUS_SESSION_BUS_ADDRESS", f"unix:path={runtime_dir}/bus")
        try:
            scoped = subprocess.run(argv, env=environment, cwd=os.getcwd(), check=False)
        except (OSError, subprocess.SubprocessError) as exc:
            fail(f"autonomous boundary unavailable for the host smoke: {exc}")
        raise SystemExit(scoped.returncode)
    if sys.platform.startswith("linux") and not _in_autonomous_aggregate():
        fail("the host smoke requires the autonomous boundary; it could not be entered")


def fail(message: str) -> None:
    raise SmokeFailure(message)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Image preparation (supervised fixture setup only)
# ---------------------------------------------------------------------------


def _private_engine(run_root: Path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from hermes_cli.worker_execution_podman import Engine, _process_cgroup

    engine = Engine(run_root)
    engine.worker_cgroup = _process_cgroup(os.getpid()) or ""
    engine.prepare()
    return engine


def _rewrite_oci_archive(source: Path, destination: Path, tag: str) -> None:
    with tarfile.open(source) as reader, tarfile.open(destination, "w") as writer:
        for member in reader.getmembers():
            data = reader.extractfile(member).read() if member.isfile() else None
            if member.name == "index.json":
                index = json.loads(data or b"{}")
                for manifest in index.get("manifests") or []:
                    manifest.setdefault("annotations", {})[
                        "org.opencontainers.image.ref.name"
                    ] = tag
                data = json.dumps(index).encode()
            info = tarfile.TarInfo(member.name)
            info.size = len(data or b"")
            info.mode = member.mode
            info.mtime = member.mtime
            info.type = member.type
            writer.addfile(info, BytesIO(data or b""))


def _remove_tree(path: Path) -> None:
    from hermes_cli.worker_execution_podman import remove_tree

    if path.exists():
        remove_tree(path)


def _validate_images(store: Path) -> None:
    """Normal runs only validate the operator-provisioned store; never pull or rewrite."""
    inventory_path = store / "inventory.json"
    if not inventory_path.is_file():
        fail("image store is not prepared; run with --prepare-images first")
    inventory = json.loads(inventory_path.read_bytes().decode("utf-8"))
    images = inventory.get("images") if isinstance(inventory, dict) else None
    if not isinstance(images, dict):
        fail("image store inventory is malformed")
    for reference in IMAGE_TAGS:
        record = images.get(reference)
        if not isinstance(record, dict):
            fail(f"image store has no provisioned entry for {reference}")
        archive = store / str(record.get("archive") or "")
        if not archive.is_file() or "sha256:" + sha256_file(archive) != record.get("sha256"):
            fail(f"provisioned archive for {reference} is missing or changed")


def prepare_images(store: Path) -> dict[str, Any]:
    if not store.exists():
        store.mkdir(mode=0o700, parents=True)
    os.chmod(store, 0o700)
    inventory_path = store / "inventory.json"
    if inventory_path.is_file():
        existing = json.loads(inventory_path.read_bytes().decode("utf-8"))
        images = existing.get("images") or {}
        if all(reference in images for reference in IMAGE_TAGS):
            verified = True
            for reference, record in images.items():
                archive = store / record["archive"]
                if not archive.is_file() or "sha256:" + sha256_file(archive) != record["sha256"]:
                    verified = False
            if verified:
                print("images: already provisioned", flush=True)
                return existing
    run_root = store / ".prepare"
    _remove_tree(run_root)
    engine = _private_engine(run_root)
    images: dict[str, Any] = {}
    try:
        for reference, tag in IMAGE_TAGS.items():
            print(f"images: pulling {reference}", flush=True)
            pulled = engine.run(["pull", "--quiet", reference], timeout=1800.0)
            if pulled.returncode != 0:
                message = pulled.stderr.decode("utf-8", "replace").strip()
                fail(f"image pull failed for {reference}: {message}")
            pinned_digest = reference.rsplit("sha256:", 1)[-1]
            staging = store / f".{pinned_digest[:16]}.oci.tar"
            saved = engine.run(["save", "--format", "oci-archive", "-o", str(staging), reference], timeout=1800.0)
            if saved.returncode != 0:
                fail(f"image save failed for {reference}")
            tagged = store / f".tagged-{staging.stem}.oci.tar"
            _rewrite_oci_archive(staging, tagged, tag)
            staging.unlink()
            archive = store / f"{pinned_digest}.oci.tar"
            os.replace(tagged, archive)
            loaded = engine.run(["load", "-i", str(archive)], timeout=1800.0)
            if loaded.returncode != 0:
                fail(f"image load failed for {reference}: {loaded.stderr.decode('utf-8', 'replace')}")
            image_id = engine.text(["image", "inspect", "--format", "{{.Id}}", tag], timeout=120.0)
            images[reference] = {
                "archive": archive.name,
                "sha256": "sha256:" + sha256_file(archive),
                "image_id": image_id,
                "tag": tag,
            }
            print(f"images: pinned {reference} -> {archive.name}", flush=True)
    finally:
        _remove_tree(run_root)
    inventory = {"protocol": 1, "images": images}
    temporary = inventory_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(inventory, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, inventory_path)
    os.chmod(inventory_path, 0o600)
    return inventory


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _fixture_repo(path: Path) -> tuple[str, str]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(path), *args], check=True, capture_output=True, text=True,
        ).stdout.strip()

    path.mkdir(exist_ok=True)
    git("init", "-q")
    git("config", "user.email", "smoke@example.invalid")
    git("config", "user.name", "Executor Smoke")
    (path / "app.py").write_text("def value():\n    return 1\n", encoding="utf-8")
    (path / "check.py").write_text(CHECK_PY, encoding="utf-8")
    (path / "probe.sql").write_text(PROBE_SQL, encoding="utf-8")
    git("add", "-A")
    git("commit", "-qm", "base")
    base_sha = git("rev-parse", "HEAD")
    (path / "app.py").write_text("def value():\n    return 2\n", encoding="utf-8")
    git("add", "app.py")
    git("commit", "-qm", "head")
    head_sha = git("rev-parse", "HEAD")
    return base_sha, head_sha


def _policy_text(workspace: Path, image_store: Path, recipes: dict[str, Any]) -> str:
    lines = [
        "kanban:", "  worker_execution:", "    protocol: 1",
        f"    image_store: {image_store}", "    workspaces:",
        f"      - path: {workspace}", "        inspect: true",
        "        extra_source_paths: []", "        recipes:",
        "          offline:", "            image: " + json.dumps(OFFLINE_RECIPE["image"]),
        "            argv: " + json.dumps(OFFLINE_RECIPE["argv"]),
        f"            timeout_seconds: {OFFLINE_RECIPE['timeout_seconds']}",
        "            services: []",
        "          database:", "            image: " + json.dumps(DB_RECIPE["image"]),
        "            argv: " + json.dumps(DB_RECIPE["argv"]),
        f"            timeout_seconds: {DB_RECIPE['timeout_seconds']}",
        "            services:",
        "              - id: db",
        "                image: " + json.dumps(DB_SERVICE["image"]),
        "                argv: " + json.dumps(DB_SERVICE["argv"]),
        "                ready_argv: " + json.dumps(DB_SERVICE["ready_argv"]),
        f"                port: {DB_SERVICE['port']}",
    ]
    if "stall" in recipes:
        lines += [
            "          stall:", "            image: " + json.dumps(STALL_RECIPE["image"]),
            "            argv: " + json.dumps(STALL_RECIPE["argv"]),
            f"            timeout_seconds: {STALL_RECIPE['timeout_seconds']}",
            "            services: []",
        ]
    return "\n".join(lines) + "\n"


def _seed_board(connection: sqlite3.Connection, *, workspace: Path, base_sha: str, head_sha: str) -> tuple[str, str]:
    from hermes_cli import kanban_db as kb

    implementation = kb.create_task(
        connection, title="smoke implementation", assignee="implementer",
        workspace_kind="dir", workspace_path=str(workspace),
        lifecycle_contract={
            "kind": "code", "review_mode": "separate_card", "reviewer": "reviewer",
            "validation_required": False,
        },
    )
    review = kb.create_task(
        connection, title="smoke review", assignee="reviewer",
        workspace_kind="dir", workspace_path=str(workspace),
        lifecycle_contract={"kind": "review", "candidate_task_id": implementation},
    )
    kb.link_tasks(connection, implementation, review, requirement="phase_finished")
    run = kb.claim_task(connection, implementation, claimer="smoke:implementer")
    if run is None:
        fail("smoke implementation card could not be claimed")
    if not kb.complete_task(
        connection, implementation, expected_run_id=run.current_run_id,
        summary="smoke implementation handoff",
        metadata={"base_sha": base_sha, "head_sha": head_sha, "changed_files": ["app.py"]},
    ):
        fail("smoke implementation handoff was refused")
    return implementation, review


class _State:
    def __init__(self) -> None:
        self.catalogs: list[list[str]] = []
        self.startup: dict[str, Any] | None = None
        self.summary: dict[str, Any] | None = None
        self.read: dict[str, Any] | None = None
        self.diff: dict[str, Any] | None = None
        self.offline: dict[str, Any] | None = None
        self.database: dict[str, Any] | None = None
        self.database_again: dict[str, Any] | None = None
        self.failure: dict[str, Any] | None = None
        self.completed: dict[str, Any] | None = None
        self.last_tool_content: str | None = None
        self.steps: list[str] = []
        self.last_activity = time.monotonic()
        self.fixture: Path | None = None

    def mark(self, step: str) -> None:
        self.steps.append(step)
        self.last_activity = time.monotonic()


def _responder(state: _State, *, base_sha: str, head_sha: str, death: bool):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tests.fakes.fake_llm_provider import Text, ToolCall

    def responder(record: dict[str, Any]):
        body = record["body"]
        names = {tool["function"]["name"] for tool in body.get("tools") or []}
        state.catalogs.append(sorted(names))
        messages = body.get("messages") or []
        last = None
        call_name = None
        for message in messages:
            if message.get("role") == "assistant" and message.get("tool_calls"):
                calls = message["tool_calls"]
                if calls:
                    call_name = calls[-1].get("function", {}).get("name")
            if message.get("role") == "tool":
                last = message
        if last is None:
            return ToolCall("kanban_worker_startup", {})
        content = last.get("content")
        state.last_tool_content = str(content)[:800]
        try:
            payload = json.loads(content or "{}")
        except (TypeError, ValueError):
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        tool_name = last.get("name") or last.get("tool_name") or call_name
        if tool_name in {"kanban_worker_startup", "kanban_worker_inspect", "kanban_worker_verify"}:
            if payload.get("ok") is not True:
                state.failure = {"tool": tool_name, **payload}
                return Text("smoke: refused: " + json.dumps(payload, sort_keys=True)[:400])
        if tool_name == "kanban_worker_startup":
            state.startup = payload["startup"]
            state.mark("startup")
            return ToolCall("kanban_worker_inspect", {"operation": "summary"})
        if tool_name == "kanban_worker_inspect":
            operation = payload.get("operation")
            if operation == "summary":
                state.summary = payload
                state.mark("summary")
                return ToolCall("kanban_worker_inspect", {
                    "operation": "read", "path": "app.py", "line_count": 10,
                    "expected_source_digest": payload["source_digest"],
                })
            if operation == "read":
                state.read = payload
                state.mark("read")
                return ToolCall("kanban_worker_inspect", {
                    "operation": "diff", "base": base_sha, "head": head_sha,
                })
            if operation == "diff":
                state.diff = payload
                state.mark("diff")
                return ToolCall("kanban_worker_verify", {
                    "recipe": "offline", "source_digest": payload["source_digest"],
                })
            state.failure = {"tool": tool_name, "unexpected": payload}
            return Text("smoke: unexpected inspection result")
        if tool_name == "kanban_worker_verify":
            recipe = payload.get("recipe")
            if recipe == "offline":
                state.offline = payload
                state.mark("offline")
                if death:
                    return ToolCall("kanban_worker_verify", {
                        "recipe": "stall", "source_digest": state.summary["source_digest"],
                    })
                return ToolCall("kanban_worker_verify", {
                    "recipe": "database", "source_digest": state.summary["source_digest"],
                })
            if recipe == "database":
                if state.database is None:
                    state.database = payload
                    state.mark("database")
                    return ToolCall("kanban_worker_verify", {
                        "recipe": "database", "source_digest": state.summary["source_digest"],
                    })
                state.database_again = payload
                state.mark("database-again")
                return ToolCall("kanban_show", {})
            state.failure = {"tool": tool_name, "unexpected": payload}
            return Text("smoke: unexpected recipe result")
        if tool_name == "kanban_show":
            state.mark("show")
            return ToolCall("kanban_complete", {
                "summary": "smoke reviewed the exact implementation head",
                "verdict": "APPROVE",
                "metadata": {"reviewed_head_sha": head_sha},
            })
        if tool_name == "kanban_complete":
            state.completed = payload
            state.mark("complete")
            return Text("smoke: review complete")
        state.failure = {"tool": tool_name or "<unknown>", "payload": payload}
        return Text("smoke: unexpected tool result")

    return responder


def _run_case(
    root: Path,
    *,
    policy_root: Path,
    image_store: Path,
    death: bool,
) -> dict[str, Any]:
    root.mkdir(parents=True)
    for name in ("home", "config", "cache", "data", "tmp", "hermes", "kanban", "workspace"):
        (root / name).mkdir(exist_ok=True)
    workspace = root / "workspace"
    base_sha, head_sha = _fixture_repo(workspace)
    outside = root / "outside-sentinel.txt"
    outside.write_text("untouched\n", encoding="utf-8")
    board_path = root / "kanban" / "kanban.db"
    profile_home = root / "hermes" / "profiles" / "reviewer"
    plugin_source = policy_root / "plugins" / "consequence_guard"

    for key in [key for key in os.environ if key.startswith("HERMES_")]:
        del os.environ[key]
    os.environ.update({
        "HOME": str(root / "home"),
        "XDG_CONFIG_HOME": str(root / "config"),
        "XDG_CACHE_HOME": str(root / "cache"),
        "XDG_DATA_HOME": str(root / "data"),
        "HERMES_HOME": str(root / "hermes"),
        "HERMES_KANBAN_HOME": str(root / "kanban"),
        "HERMES_KANBAN_DB": str(board_path),
        "TMPDIR": str(root / "tmp"),
    })
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    # This harness's own directory is not a runtime input; leaving it on sys.path
    # would seal these scripts into the worker generation.
    scripts_dir = Path(__file__).resolve().parent
    sys.path[:] = [entry for entry in sys.path if Path(entry or os.getcwd()).resolve() != scripts_dir]
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    from tests.fakes.fake_llm_provider import FakeLLMServer, write_hermes_home

    state = _State()
    state.fixture = root
    with FakeLLMServer(_responder(state, base_sha=base_sha, head_sha=head_sha, death=death)) as server:
        write_hermes_home(
            profile_home, server.base_url,
            extra_config=(
                "toolsets:\n  - kanban\n"
                "tools:\n  tool_search:\n    enabled: \"on\"\n"
                "plugins:\n  enabled: [cto-consequence-guard]\n"
            ),
        )
        plugin = profile_home / "plugins" / "cto-consequence-guard"
        plugin.mkdir(parents=True)
        for name in ("__init__.py", "plugin.yaml"):
            shutil.copyfile(plugin_source / name, plugin / name)
        # Execution authority comes from the DISPATCHER's process home only.
        (root / "hermes" / "config.yaml").write_text(
            "toolsets:\n  - kanban\n" + _policy_text(workspace, image_store, {"stall": death}),
            encoding="utf-8",
        )
        return _launch(
            root=root, workspace=workspace, board_path=board_path,
            base_sha=base_sha, head_sha=head_sha, state=state, death=death,
            outside=outside, image_store=image_store,
        )


def _launch(
    *,
    root: Path,
    workspace: Path,
    board_path: Path,
    base_sha: str,
    head_sha: str,
    state: _State,
    death: bool,
    outside: Path,
    image_store: Path,
) -> dict[str, Any]:
    from unittest.mock import patch

    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    from hermes_cli import kanban_worker_runtime as runtime

    connection = kbc.connect(board_path)
    kb._ensure_lifecycle_schema(connection)
    kb._ensure_goal_revision_schema(connection)
    _implementation, review = _seed_board(
        connection, workspace=workspace, base_sha=base_sha, head_sha=head_sha,
    )
    # Real native admission: the harness re-executed itself inside autonomous.slice,
    # so the boundary, the worker-scope placement check and the concurrency slot are
    # the production checks. No patch touches them.
    import contextlib

    boundary = worker_check = capacity = contextlib.nullcontext()
    journal_roots = _journal_roots(root)
    receipt: dict[str, Any] = {
        "case": "death" if death else "main",
        "workspace": str(workspace),
        "board": str(board_path),
        "base_sha": base_sha,
        "head_sha": head_sha,
        "image_store": str(image_store),
        "fixture_digests": {
            "check.py": sha256_file(workspace / "check.py"),
            "probe.sql": sha256_file(workspace / "probe.sql"),
        },
    }
    tree_before = _tree_digest(workspace)
    worker_pid: int | None = None
    setattr(state, "worker_pid", None)
    with boundary, worker_check, capacity:
        result = kbd.dispatch_once(
            connection, max_spawn=1, max_in_progress=4, max_in_progress_per_profile=4,
            reconcile_orphans=False,
        )
        if not result.spawned:
            events = connection.execute(
                "SELECT kind, payload FROM task_events WHERE task_id = ? ORDER BY id", (review,),
            ).fetchall()
            fail(
                f"dispatcher spawned no worker: {result}; "
                f"events={[dict(row) for row in events]}; logs={_log_tail(root)}"
            )
        task = kb.get_task(connection, review)
        worker_pid = int(task.worker_pid or 0) or None
        state.worker_pid = worker_pid
        _LAST_WORKER["pid"] = worker_pid
        receipt["worker_pid"] = worker_pid
        receipt["spawned"] = [
            {"task_id": item[0], "assignee": item[1], "workspace": item[2] if len(item) > 2 else ""}
            for item in result.spawned
        ]
        if death:
            deadline = time.monotonic() + 600
            journals: list[Path] = []
            stall_journal: dict[str, Any] | None = None
            while time.monotonic() < deadline:
                journals = _journal_files(journal_roots)
                for path in journals:
                    payload = _journal_payload(path)
                    # Only the verifier release (not the earlier service release) and a
                    # recorded verifier process qualify.
                    if (
                        payload
                        and payload.get("recipe") == "stall"
                        and payload.get("phase") == "released"
                        and any(
                            record.get("kind") == "verifier" and record.get("pid")
                            for record in payload.get("containers") or []
                        )
                    ):
                        stall_journal = payload
                        break
                if stall_journal is not None:
                    break
                if state.failure or state.completed:
                    break
                if worker_pid and not runtime._pid_alive(worker_pid):
                    break
                time.sleep(0.2)
            if stall_journal is None:
                fail(
                    "the stall invocation never reached its released payload: "
                    f"steps={state.steps}; failure={state.failure}; "
                    f"last_tool={state.last_tool_content}; journals={[str(p) for p in journals]}; "
                    f"logs={_log_tail(root)}"
                )
            # Prove the payload and its detached child are live before the worker dies.
            verifier = next(
                (record for record in stall_journal.get("containers") or []
                 if record.get("kind") == "verifier" and record.get("pid")),
                None,
            )
            if verifier is None:
                fail(f"the stall journal has no verifier process: {stall_journal}")
            init_pid = int(verifier["pid"])
            init_start = int(verifier.get("pid_start") or 0)
            # The backend writes `released` just before the marker is created and the
            # gate polls it every 0.2s: observe until the payload and its long-lived
            # child are actually running, rather than racing the release.
            observation_deadline = time.monotonic() + 20.0
            others: list[tuple[int, int]] = []
            while time.monotonic() < observation_deadline:
                init_cmd = _proc_cmdline(init_pid)
                if "until [ -e /scratch/release ]" not in init_cmd and "sleep 600" in init_cmd:
                    others = [
                        (pid, start)
                        for pid, start in _descendant_pids(verifier.get("pid_inode"))
                        if pid != init_pid and _proc_cmdline(pid).strip() == "sleep 600"
                    ]
                    if others:
                        break
                time.sleep(0.2)
            if not others:
                fail(
                    "the stall payload did not reach its long-lived child before the worker was killed: "
                    f"init={_proc_cmdline(init_pid)[:200]!r} journals={[str(p) for p in journals]}"
                )
            descendants = others
            receipt["stall_journal"] = stall_journal
            receipt["payload_pids"] = [init_pid, *[pid for pid, _ in others]]
            if worker_pid:
                with __import__("contextlib").suppress(ProcessLookupError):
                    os.kill(worker_pid, getattr(signal, "SIGKILL", signal.SIGTERM))
            receipt["worker_killed"] = True

            # Observe payload teardown WITHOUT invoking any reaper first: this is
            # conmon's deadline killing the container and its detached children.
            from hermes_cli.worker_execution_podman import _process_identity

            deadline_epoch = float(stall_journal.get("deadline_epoch") or 0)
            watch_deadline = time.monotonic() + max(60.0, deadline_epoch - time.time() + 60.0)
            expired_at = 0.0
            while time.monotonic() < watch_deadline:
                alive = _process_identity(init_pid, init_start) != "gone"
                alive = alive or any(
                    _process_identity(pid, start) != "gone" for pid, start in descendants
                )
                if not alive:
                    expired_at = time.time()
                    break
                time.sleep(0.2)
            if not expired_at:
                fail("the stall payload or its detached child survived the conmon deadline")
            receipt["payload_expired_epoch"] = expired_at
            if expired_at < deadline_epoch - 2.0:
                fail(
                    "the payload expired before the recorded conmon deadline: "
                    f"expired={expired_at} deadline={deadline_epoch}"
                )
            cleaned, cleanup_results, cleaned_at = _await_cleanup(
                journal_roots, deadline=time.monotonic() + 300,
            )
            receipt["journals_cleaned"] = cleaned
            receipt["cleanup_observed_epoch"] = cleaned_at
            if cleaned and cleaned_at < float(stall_journal.get("deadline_epoch") or 0):
                fail(
                    "journals disappeared before the recorded conmon deadline: "
                    f"cleaned_at={cleaned_at} deadline={stall_journal.get('deadline_epoch')}"
                )
            receipt["cleanup_results"] = cleanup_results
            if not cleaned:
                fail(
                    f"journal cleanup did not complete after the worker died: "
                    f"results={cleanup_results}; logs={_log_tail(root)}"
                )
        else:
            deadline = time.monotonic() + 1500
            while time.monotonic() < deadline:
                runtime.reap_worker_zombies()
                task = kb.get_task(connection, review)
                if task.status in {"done", "blocked"}:
                    break
                if state.failure or state.completed:
                    break
                if worker_pid and not runtime._pid_alive(worker_pid):
                    break
                time.sleep(0.2)
    finally_kill = False
    final = kb.get_task(connection, review)
    events = connection.execute(
        "SELECT kind FROM task_events WHERE task_id = ? ORDER BY id", (review,),
    ).fetchall()
    runs = connection.execute(
        "SELECT id, status, outcome FROM task_runs WHERE task_id = ? ORDER BY id", (review,),
    ).fetchall()
    connection.close()
    receipt["final_status"] = final.status
    receipt["events"] = [row["kind"] for row in events]
    receipt["runs"] = [dict(row) for row in runs]
    receipt["steps"] = list(state.steps)
    receipt["catalogs"] = state.catalogs[:3]
    receipt["outside_sentinel"] = sha256_file(outside)
    receipt["tree_before"] = tree_before
    receipt["tree_after"] = _tree_digest(workspace)
    if state.failure:
        fail(
            f"a granted call was refused: {state.failure}; steps={state.steps}; "
            f"last_tool={state.last_tool_content}; logs={_log_tail(root)}"
        )
    if outside.read_bytes() != b"untouched\n":
        fail("an outside sentinel was modified")
    if receipt["tree_before"] != receipt["tree_after"]:
        fail("the granted workspace changed during verification")
    if death:
        receipt["ok"] = True
        return receipt
    if final.status != "done":
        fail(f"review card ended {final.status!r}: {receipt['runs']}")
    startup = state.startup or {}
    receipt["startup"] = startup
    if startup.get("task_id") != review or startup.get("profile") != "reviewer":
        fail(f"startup metadata is wrong: {startup!r}")
    summary = state.summary or {}
    if summary.get("git", {}).get("head") != head_sha:
        fail("summary HEAD is not the fixture head")
    offline = state.offline or {}
    if offline.get("ok") is not True or offline.get("accepted") is not True:
        fail(f"offline verification was not accepted: {offline!r}")
    if "offline-verification-ok" not in (offline.get("stdout") or ""):
        fail(f"offline verification printed the wrong output: {offline!r}")
    database = state.database or {}
    if database.get("ok") is not True or database.get("accepted") is not True:
        fail(f"database verification was not accepted: {database!r}")
    if "76" not in (database.get("stdout") or ""):
        fail(f"database verification did not read the fixture row: {database!r}")
    again = state.database_again or {}
    if again.get("ok") is not True or again.get("accepted") is not True:
        fail(f"the repeated database verification was not accepted: {again!r}")
    if "76" not in (again.get("stdout") or ""):
        fail(f"the repeated database verification did not read a fresh fixture row: {again!r}")
    receipt["database_again"] = again
    for payload, label in ((offline, "offline"), (database, "database"), (again, "database")):
        if not payload.get("cleanup_ok"):
            fail(f"{label} verification left resources behind")
        if payload.get("source_digest") != summary.get("source_digest"):
            fail(f"{label} verification reported a different source digest")
        if payload.get("recipe") != label:
            fail(f"{label} receipt names the wrong recipe")
    # The grant is retained during the constructor, before the first model call, so
    # every model-facing catalog here is post-grant; the synthetic startup probe
    # covers pre-grant invisibility.
    if not state.catalogs or not all("kanban_worker_inspect" in catalog for catalog in state.catalogs):
        fail(f"the granted inspection tool was not in every catalog: {state.catalogs}")
    if not any("kanban_worker_verify" in catalog for catalog in state.catalogs):
        fail("verification was never exposed after the grant")
    receipt["ok"] = True
    receipt["offline"] = offline
    receipt["database"] = database
    return receipt


def _recovery_git(path: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(path), *args], check=True, capture_output=True, text=True,
    ).stdout.strip()


def _recovery_require(condition: Any, message: str) -> None:
    if not condition:
        fail("recovery: " + message)


def _recovery_evidence(state: dict, task_id: str, section: str):
    """Consume native values and every explicit continuation, without a runner."""
    tokens = state["snapshots"]
    args = {"task_id": task_id, "board": state["startup"]["board"], "section": section}
    if task_id in tokens:
        args["snapshot"] = tokens[task_id]

    def page(arguments):
        arguments = {**arguments, "limit": 20, "chars": 8192}
        payload = yield ("kanban_show", arguments)
        _recovery_require(payload.get("ok") is True, f"bounded read refused: {payload}")
        token = payload["snapshot"]
        _recovery_require(tokens.setdefault(task_id, token) == token, "snapshot changed")
        size = len(json.dumps(payload, ensure_ascii=False).encode())
        _recovery_require(size <= 32768, f"bounded envelope exceeded limit: {size}")
        state["evidence_pages"].append({
            "arguments": arguments, "snapshot": token, "bytes": size,
            "continuations": [
                value["continuation"] for value in
                [payload, *payload.get("items", []), *payload.get("entries", [])]
                if "continuation" in value
            ],
        })
        return payload

    def node(arguments):
        result = yield from page(arguments)
        if "value" in result:
            return result["value"]
        if "chunk" in result:
            chunks = []
            expected = arguments.get("offset", 0)
            total = result["chunk"]["total_chars"]
            while True:
                chunk = result["chunk"]
                _recovery_require(chunk["offset"] == expected, "string chunk gap")
                chunks.append(chunk["text"])
                expected += len(chunk["text"])
                if chunk["next_offset"] is None:
                    _recovery_require(expected == total, "string ended prematurely")
                    return "".join(chunks)
                _recovery_require(chunk["next_offset"] == expected, "string cursor mismatch")
                result = yield from page(result["continuation"])
        if "items" in result:
            items = []
            while True:
                for entry in result["items"]:
                    value = (yield from node(entry["continuation"])) if entry.get("deferred") else entry
                    _recovery_require(not items or value["id"] > items[-1]["id"], "history repeated or unordered")
                    items.append(value)
                if not result["has_more"]:
                    return items
                result = yield from page(result["continuation"])
        entries = {}
        array = bool(result["entries"] and "index" in result["entries"][0])
        total = result["total_entries"]
        while True:
            for entry in result["entries"]:
                key = entry["index"] if array else entry["key"]
                _recovery_require(key not in entries, "container entry repeated")
                entries[key] = (yield from node(entry["continuation"])) if entry.get("deferred") else entry["value"]
            if result["next_offset"] is None:
                _recovery_require(len(entries) == total, "container ended prematurely")
                return [entries[i] for i in range(total)] if array else entries
            result = yield from page(result["continuation"])

    value = yield from node(args)
    state["evidence"].setdefault(task_id, {})[section] = (
        {"sha256": sha256_bytes(json.dumps(value, sort_keys=True).encode())}
        if section in {"task", "events", "comments"} else value
    )
    return value


def _recovery_program(state: dict, graph: dict, role: str):
    startup = yield ("kanban_worker_startup", {})
    state["startup"] = startup["startup"]
    own = graph[role]
    summary = state["startup"]
    _recovery_require(summary["task_id"] == own and summary["profile"] == role, "wrong startup identity")
    _recovery_require(summary["spawn_receipt"] == "verified", "worker spawn receipt not verified")
    execution = summary["execution"]
    if role == "reviewer" and graph["variant"] == "no_grant":
        _recovery_require(execution == {
            "granted": False, "inspect": False, "recipes": [], "refusal": "workspace_not_granted",
        }, f"wrong withheld capability: {execution}")
        yield ("kanban_block", {"kind": "capability", "reason": "Fixture workspace_not_granted; no inspection authority."})
        return
    _recovery_require(execution["granted"] and execution["inspect"] and execution["refusal"] is None,
                      f"missing inspection grant: {execution}")
    _recovery_require(execution["recipes"] == ([] if graph["variant"] == "inspect_only" else ["offline"]),
                      f"wrong recipes: {execution}")
    if role != "implementer":
        own_lifecycle = yield from _recovery_evidence(state, own, "lifecycle")
        own_dependencies = yield from _recovery_evidence(state, own, "dependencies")
        for task_id in dict.fromkeys([own, graph["implementer"], graph["reviewer"]]):
            sections = ("task", "effective_goal", "lifecycle", "dependencies", "provenance")
            if role == "tester" and task_id == graph["reviewer"]:
                sections = ("lifecycle", "provenance", "runs")
            for section in sections:
                value = yield from _recovery_evidence(state, task_id, section)
                if section == "provenance":
                    _recovery_require(value["verdict"] == "verified", "candidate receipt not verified")
                if section == "task" and task_id == graph["reviewer"]:
                    _recovery_require(value["body"] == graph["body"], "mandatory restriction body not reconstructed")
                    if graph["variant"] == "main":
                        _recovery_require(len(value["body"]) >= 321724 and "\n" not in value["body"],
                                          "mandatory single-line body is too small")
                    state["mandatory_evidence"] = {"body_chars": len(value["body"]),
                                                   "body_sha256": sha256_bytes(value["body"].encode())}
        if role == "reviewer":
            events = yield from _recovery_evidence(state, own, "events")
            comments = yield from _recovery_evidence(state, own, "comments")
            yield from _recovery_evidence(state, own, "runs")
            if graph["variant"] == "main":
                markers = [event["payload"]["marker"] for event in events if event["kind"] == "recovery_evidence"]
                _recovery_require(markers == list(range(61)), "first/middle/late history was lost")
                nested = next(event["payload"]["nested"]["text"] for event in events
                              if event["kind"] == "recovery_evidence" and event["payload"]["marker"] == 30)
                _recovery_require(nested == graph["nested"], "oversized nested event was lost")
                _recovery_require(any(comment["body"] == graph["comment"] for comment in comments),
                                  "oversized comment was lost")
                _recovery_require(any(p["arguments"].get("offset", 0) > 0 for p in state["evidence_pages"]),
                                  "no string continuation followed")
                _recovery_require(any(p["arguments"].get("after", 0) > 0 for p in state["evidence_pages"]),
                                  "no history continuation followed")
                _recovery_require(any("item" in p["arguments"] for p in state["evidence_pages"]),
                                  "no deferred history row followed")
                state["mandatory_evidence"].update({
                    "history_markers": markers, "nested_chars": len(nested),
                    "comment_chars": len(graph["comment"]),
                })
        bookend = yield from _recovery_evidence(state, own, "lifecycle")
        _recovery_require(bookend == own_lifecycle, "own lifecycle changed across graph reads")
        candidate = state["evidence"][graph["implementer"]]["lifecycle"]
        _recovery_require(candidate["acceptance"] == "pending", "eligibility was mistaken for acceptance")
        _recovery_require(own_dependencies["satisfied"], "role dependencies not satisfied")
        for field in ("candidate_task_id", "candidate_run_id", "head_sha", "goal_revision_ids"):
            _recovery_require(own_lifecycle[field] == candidate[field], f"candidate mismatch: {field}")
        if role == "tester":
            review = state["evidence"][graph["reviewer"]]["lifecycle"]
            _recovery_require(own_lifecycle["review_verdict"] is None, "tester local review is not null")
            _recovery_require(review["review_verdict"] == candidate["review_verdict"] == "APPROVE",
                              "fresh parent APPROVE missing")
            for field in ("candidate_task_id", "candidate_run_id", "head_sha", "goal_revision_ids"):
                _recovery_require(review[field] == candidate[field], f"review mismatch: {field}")
    inspected = yield ("kanban_worker_inspect", {"operation": "summary"})
    state["summary"] = inspected
    _recovery_require(inspected["git"]["head"] == graph["head_sha"], "inspection used moving main")
    if role == "implementer":
        yield ("kanban_complete", {
            "summary": "Precommitted harmless fixture B; execution: not_run; not an application claim.",
            "metadata": {"base_sha": graph["base_sha"], "head_sha": graph["head_sha"],
                         "workspace": summary["workspace_path"], "changed_files": ["app.py"], "execution": "not_run"},
        })
    elif role == "reviewer":
        state["read"] = yield ("kanban_worker_inspect", {
            "operation": "read", "path": "app.py", "line_count": 10,
            "expected_source_digest": inspected["source_digest"],
        })
        state["diff"] = yield ("kanban_worker_inspect", {
            "operation": "diff", "base": graph["base_sha"], "head": graph["head_sha"],
        })
        _recovery_require("return 2" in json.dumps(state["read"]), "typed read missed candidate source")
        _recovery_require("return 1" in json.dumps(state["diff"]) and "return 2" in json.dumps(state["diff"]),
                          "typed diff missed A..B")
        _recovery_require(state["diff"]["source_digest"] == inspected["source_digest"], "source changed")
        yield ("kanban_complete", {
            "verdict": "APPROVE",
            "summary": "Fixture source-only review of A..B; execution: not_run. Validation remains required.",
            "metadata": {"reviewed_head_sha": graph["head_sha"], "execution": "not_run",
                         "coverage_gap": "Offline recipe is the tester's required gate; no application gates covered."},
        })
    elif graph["variant"] == "inspect_only":
        yield ("kanban_block", {"kind": "capability", "reason": "Fixture inspection available, recipes empty; required offline gate unavailable, never PASS."})
    else:
        verified = yield ("kanban_worker_verify", {"recipe": "offline", "source_digest": inspected["source_digest"]})
        state["verification"] = verified
        _recovery_require(verified.get("accepted") is True and verified.get("exit_code") == 0
                          and verified.get("cleanup_ok") is True
                          and "offline-verification-ok" in verified.get("stdout", "")
                          and verified.get("source_digest") == inspected["source_digest"],
                          f"offline verification not accepted: {verified}")
        yield ("kanban_complete", {
            "verdict": "PASS", "summary": "Only the offline fixture criteria passed; no application acceptance claim.",
            "metadata": {"head_sha": graph["head_sha"], "execution": "offline",
                         "source_digest": verified["source_digest"], "verification": verified},
        })


def _recovery_responder(state: dict, graph: dict, role: str):
    from tests.fakes.fake_llm_provider import Text, ToolCall

    program = _recovery_program(state, graph, role)
    pending = None

    def respond(record):
        nonlocal pending
        names = sorted(tool["function"]["name"] for tool in record["body"].get("tools", []))
        state["catalogs"].append(names)
        try:
            if state.get("finished") or state.get("failure"):
                return Text("Fixture worker finished.")
            if pending is None:
                action = next(program)
            else:
                last = next(message for message in reversed(record["body"]["messages"]) if message["role"] == "tool")
                payload = json.loads(last["content"])
                state["calls"].append({"tool": pending[0], "arguments": pending[1], "ok": payload.get("ok"),
                                       "error": payload.get("error")})
                _recovery_require(payload.get("ok") is True, f"{pending[0]} refused: {payload}")
                if pending[0] in {"kanban_complete", "kanban_block"}:
                    state["terminal"] = payload
                action = program.send(payload)
            name, arguments = action
            _recovery_require(name in names, f"{name} missing from final catalog")
            pending = action
            return ToolCall(name, arguments)
        except StopIteration:
            state["finished"] = True
            return Text("Fixture worker finished.")
        except Exception as exc:
            state["failure"] = str(exc)
            return Text("Fixture proof refused: " + str(exc)[:600])

    return respond


def _run_recovery_case(root: Path, *, policy_root: Path, image_store: Path) -> dict:
    """Three isolated graphs; all grants, receipts and verdicts use native dispatch."""
    import yaml

    root.mkdir(parents=True)
    for name in ("home", "config", "cache", "data", "hermes", "kanban"):
        (root / name).mkdir()
    # Keep the inherited disk-backed TMPDIR. Only scratch homes are visible to workers.
    for key in list(os.environ):
        if key.startswith("HERMES_") or any(part in key for part in ("API_KEY", "TOKEN", "SECRET", "PASSWORD")):
            del os.environ[key]
    os.environ.update({
        "HOME": str(root / "home"), "XDG_CONFIG_HOME": str(root / "config"),
        "XDG_CACHE_HOME": str(root / "cache"), "XDG_DATA_HOME": str(root / "data"),
        "HERMES_HOME": str(root / "hermes"), "HERMES_KANBAN_HOME": str(root / "kanban"),
        "HERMES_KANBAN_DB": str(root / "kanban" / "kanban.db"), "PYTHONDONTWRITEBYTECODE": "1",
    })
    scripts_dir = Path(__file__).resolve().parent
    sys.path[:] = [entry for entry in sys.path if Path(entry or os.getcwd()).resolve() != scripts_dir]
    from tests.fakes.fake_llm_provider import FakeLLMServer, write_hermes_home
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    from hermes_cli import kanban_worker_runtime as runtime

    connection = kbc.connect(root / "kanban" / "kanban.db")
    kb._ensure_lifecycle_schema(connection)
    kb._ensure_goal_revision_schema(connection)
    receipt = {
        "case": "recovery", "fixture_root": str(root),
        "runtime_sha": _recovery_git(scripts_dir.parent, "rev-parse", "HEAD"),
        "policy_sha": _recovery_git(policy_root, "rev-parse", "HEAD"),
        "policy_source_digest": _tree_digest(policy_root),
        "driver_sha256": sha256_file(Path(__file__)),
        "image": {"reference": PYTHON_IMAGE, **json.loads((image_store / "inventory.json").read_text())["images"][PYTHON_IMAGE]},
        "recipe": {"id": "offline", **OFFLINE_RECIPE}, "profiles": {}, "graphs": {},
    }
    graphs = []
    workspaces = []
    for variant in ("main", "inspect_only", "no_grant"):
        repo = root / variant
        base, head = _fixture_repo(repo)
        implementation = kb.create_task(
            connection, title=f"{variant} fixture implementation", assignee="implementer", initial_status="blocked",
            workspace_kind="worktree", workspace_path=str(repo),
            lifecycle_contract={"kind": "code", "review_mode": "separate_card", "reviewer": "reviewer", "validation_required": True},
        )
        checkout = repo / ".worktrees" / implementation
        _recovery_git(repo, "worktree", "add", "-b", f"wt/{implementation}", str(checkout), head)
        (repo / "app.py").write_text("def value():\n    return 3\n", encoding="utf-8")
        _recovery_git(repo, "add", "app.py")
        _recovery_git(repo, "commit", "-qm", "main advances independently")
        main_head = _recovery_git(repo, "rev-parse", "HEAD")
        review = kb.create_task(
            connection, title=f"{variant} fixture review", assignee="reviewer",
            workspace_kind="worktree", workspace_path=str(repo),
            lifecycle_contract={"kind": "review", "candidate_task_id": implementation},
        )
        tester = kb.create_task(
            connection, title=f"{variant} fixture validation", assignee="tester",
            workspace_kind="worktree", workspace_path=str(repo),
            lifecycle_contract={"kind": "validation", "candidate_task_id": implementation},
        )
        kb.link_tasks(connection, implementation, review, requirement="phase_finished")
        kb.link_tasks(connection, review, tester, requirement="review_approved")
        graph = {"variant": variant, "repo": str(repo), "implementer": implementation, "reviewer": review,
                 "tester": tester, "base_sha": base, "head_sha": head, "main_sha": main_head,
                 "implementation_checkout": str(checkout)}
        graphs.append(graph)
        workspaces.append({
            "path": str(checkout if variant == "no_grant" else repo), "include_worktrees": True,
            "inspect": True, "extra_source_paths": [],
            "recipes": {} if variant == "inspect_only" else {"offline": OFFLINE_RECIPE},
        })
    policy = {"protocol": 1, "image_store": str(image_store), "workspaces": workspaces}
    (root / "hermes" / "config.yaml").write_text(yaml.safe_dump({"toolsets": ["kanban"], "kanban": {"worker_execution": policy}}))
    receipt["fixture_policy_digest"] = sha256_bytes(json.dumps(policy, sort_keys=True, separators=(",", ":")).encode())
    receipt["fixture_policy"] = policy
    active = {}

    def respond(record):
        return active["responder"](record)

    try:
        with FakeLLMServer(respond) as server:
            for role in ("implementer", "reviewer", "tester"):
                home = root / "hermes" / "profiles" / role
                write_hermes_home(home, server.base_url)
                fake = yaml.safe_load((home / "config.yaml").read_text())
                source = policy_root / "profiles" / role
                config = yaml.safe_load((source / "config.yaml").read_text())
                config["model"] = fake["model"]
                config["memory"] = {"memory_enabled": False, "user_profile_enabled": False}
                config.pop("remnic", None)
                config["hooks"] = {}
                config["plugins"]["enabled"] = ["cto-consequence-guard"]
                config["telemetry"] = fake["telemetry"]
                (home / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
                shutil.copyfile(source / "SOUL.md", home / "SOUL.md")
                plugin = home / "plugins" / "cto-consequence-guard"
                plugin.mkdir(parents=True)
                for name in ("__init__.py", "plugin.yaml"):
                    shutil.copyfile(policy_root / "plugins" / "consequence_guard" / name, plugin / name)
                receipt["profiles"][role] = {
                    "source_config_sha256": sha256_file(source / "config.yaml"),
                    "staged_config_sha256": sha256_file(home / "config.yaml"),
                    "soul_sha256": sha256_file(home / "SOUL.md"),
                    "plugin_sha256": sha256_file(plugin / "__init__.py"), "toolsets": config["toolsets"],
                    "coding_instructions": config["agent"]["coding_instructions"],
                }
            for graph in graphs:
                result = {"identity": dict(graph), "workers": {}, "lifecycle": {}}
                receipt["graphs"][graph["variant"]] = result
                _recovery_require(kb.unblock_task(connection, graph["implementer"]), "implementation release refused")
                roles = ("implementer", "reviewer") if graph["variant"] == "no_grant" else ("implementer", "reviewer", "tester")
                for role in roles:
                    state = {"catalogs": [], "calls": [], "snapshots": {}, "evidence_pages": [], "evidence": {}}
                    result["workers"][role] = state
                    active["responder"] = _recovery_responder(state, graph, role)
                    dispatched = kbd.dispatch_once(
                        connection, max_spawn=1, max_in_progress=4, max_in_progress_per_profile=4,
                        reconcile_orphans=False,
                    )
                    _recovery_require(len(dispatched.spawned) == 1 and dispatched.spawned[0][0] == graph[role],
                                      f"unexpected dispatch: {dispatched}; logs={_log_tail(root)}")
                    task = kb.get_task(connection, graph[role])
                    pid = int(task.worker_pid or 0)
                    _LAST_WORKER["pid"] = pid
                    state["pid"] = pid
                    state["workspace"] = task.workspace_path
                    state["checkout_sha"] = _recovery_git(Path(task.workspace_path), "rev-parse", "HEAD")
                    _recovery_require(state["checkout_sha"] == graph["head_sha"]
                                      and _recovery_git(Path(graph["repo"]), "rev-parse", "HEAD") == graph["main_sha"],
                                      "candidate/main checkout isolation failed")
                    if role != "implementer":
                        _recovery_require(task.workspace_path == str(Path(graph["repo"]) / ".worktrees" / graph[role])
                                          and task.workspace_path != graph["implementation_checkout"], "role shared implementation checkout")
                    deadline = time.monotonic() + 14400
                    while time.monotonic() < deadline:
                        runtime.reap_worker_zombies()
                        if state.get("failure") or not runtime._pid_alive(pid):
                            break
                        time.sleep(0.2)
                    _recovery_require(state.get("finished") and not state.get("failure"),
                                      f"{graph['variant']}/{role}: {state.get('failure')}; calls={state['calls'][-2:]}; logs={_log_tail(root)}")
                    _recovery_require(not runtime._pid_alive(pid), "worker did not terminate")
                    _LAST_WORKER["pid"] = None
                    task = kb.get_task(connection, graph[role])
                    state["status"] = task.status
                    state["block_kind"] = task.block_kind
                    state["runs"] = [dict(row) for row in connection.execute(
                        "SELECT * FROM task_runs WHERE task_id = ? ORDER BY id", (task.id,))]
                    state["events"] = [dict(row) for row in connection.execute(
                        "SELECT id, kind, payload FROM task_events WHERE task_id = ? AND kind IN ('claimed','spawned','blocked') ORDER BY id", (task.id,))]
                    allowed = {"kanban_worker_startup", "kanban_worker_inspect", "kanban_show", "kanban_complete", "kanban_block"}
                    if role == "tester" and graph["variant"] == "main":
                        allowed.add("kanban_worker_verify")
                    observed_calls = {}
                    for request in server.main_requests():
                        for message in request.get("messages", []):
                            for call in message.get("tool_calls", []):
                                observed_calls[call["id"]] = call["function"]
                    _recovery_require(len(observed_calls) == len(state["calls"]), "provider/tool-result audit incomplete")
                    _recovery_require(all(call["name"] in allowed and
                                          (call["name"] != "kanban_show" or "section" in json.loads(call["arguments"]))
                                          for call in observed_calls.values()), "forbidden fallback in provider records")
                    execution = state["startup"]["execution"]
                    _recovery_require(all(("kanban_worker_inspect" in catalog) == execution["inspect"]
                                          and ("kanban_worker_verify" in catalog) == bool(execution["recipes"])
                                          for catalog in state["catalogs"]), "catalog did not match retained capability")
                    state["provider_audit"] = {"ok": True, "calls": len(state["calls"]), "forbidden_fallbacks": []}
                    result["lifecycle"][role] = kb.get_lifecycle_state(connection, graph["implementer"])
                    blocked = (role == "tester" and graph["variant"] == "inspect_only") or (role == "reviewer" and graph["variant"] == "no_grant")
                    _recovery_require(task.status == ("blocked" if blocked else "done"), f"wrong terminal status: {task.status}")
                    if blocked:
                        _recovery_require(task.block_kind == "capability", "missing native capability block")
                    if role == "implementer":
                        graph["body"] = (
                            ("inert fixture evidence. " * 14000 if graph["variant"] == "main" else "")
                            + f" Operator authorizes source-only evidence review of {graph['base_sha']}..{graph['head_sha']}; execution: not_run. "
                            + "RECOVERY_RESTRICTION: fixture only; never application acceptance."
                        )
                        review_task = kb.get_task(connection, graph["reviewer"])
                        kb.update_task(connection, graph["reviewer"], expected_version=review_task.version,
                                       reason="Seed inert mandatory recovery evidence", body=graph["body"], author="fixture-operator")
                        if graph["variant"] == "main":
                            graph["nested"] = "nested fixture evidence. " * 1600 + "NESTED_LATE_RESTRICTION"
                            graph["comment"] = json.dumps({"nested": {"text": "comment fixture evidence. " * 1600 + "COMMENT_LATE_RESTRICTION"}})
                            kb.add_comment(connection, graph["reviewer"], "fixture-operator", graph["comment"])
                            with kb.write_txn(connection):
                                for index in range(61):
                                    payload = {"marker": index}
                                    if index == 30:
                                        payload["nested"] = {"text": graph["nested"]}
                                    kb._append_event(connection, graph["reviewer"], "recovery_evidence", payload)
                    server.requests.clear()
                final = kb.get_lifecycle_state(connection, graph["implementer"])
                _recovery_require(final["acceptance"] == ("accepted" if graph["variant"] == "main" else "pending"),
                                  f"wrong candidate acceptance: {final}")
                _recovery_require(final["validation_verdict"] == ("PASS" if graph["variant"] == "main" else None),
                                  "capability absence became a verdict")
                result["final_lifecycle"] = final
            cleaned, records, epoch = _await_cleanup(_journal_roots(root), deadline=time.monotonic() + 120)
            receipt["cleanup"] = {"ok": cleaned, "records": records, "observed_epoch": epoch}
            _recovery_require(cleaned, "execution journals unreconciled")
            receipt["ok"] = True
            return receipt
    finally:
        connection.close()
        # Retain observations even on a failed assertion, outside teardown's reach.
        (root / "recovery-observations.json").write_text(json.dumps(receipt, indent=2, sort_keys=True))


def _tree_digest(root: Path) -> str:
    entries: list[str] = []
    for path in sorted(root.rglob("*")):
        if ".git" in path.parts:
            continue
        if path.is_file() and not path.is_symlink():
            entries.append(f"{path.relative_to(root)}\x00{sha256_file(path)}")
    return sha256_bytes("\n".join(entries).encode())


def _journal_payload(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_bytes().decode("utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _descendant_pids(pid_inode: Any) -> list[tuple[int, int]]:
    """Processes of this UID inside one recorded PID namespace (excluding its init)."""
    if not isinstance(pid_inode, int):
        return []
    found: list[tuple[int, int]] = []
    for entry in os.scandir("/proc"):
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            info = os.stat(f"/proc/{pid}")
        except OSError:
            continue
        if int(info.st_uid) != (os.getuid() if hasattr(os, "getuid") else -1):  # windows-footgun: ok — hasattr-guarded; POSIX-only process scan
            continue
        try:
            observed = int(os.stat(f"/proc/{pid}/ns/pid").st_ino)
        except OSError:
            continue
        if observed != pid_inode:
            continue
        start = _proc_start_time(pid)
        if start is not None:
            found.append((pid, start))
    return found


def _proc_start_time(pid: int) -> int | None:
    try:
        text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8", errors="strict")
    except OSError:
        return None
    with __import__("contextlib").suppress(IndexError, ValueError):
        return int(text.rsplit(")", 1)[1].split()[19])
    return None


def _proc_cmdline(pid: int) -> str:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return ""
    return raw.replace(b"\x00", b" ").decode("utf-8", "replace")


def _journal_roots(root: Path) -> list[Path]:
    home = root / "hermes"
    roots = [home / "worker-executions"]
    profiles = home / "profiles"
    if profiles.is_dir():
        roots.extend(profile / "worker-executions" for profile in profiles.iterdir() if profile.is_dir())
    return roots


def _journal_files(roots: list[Path]) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        if root.is_dir():
            files.extend(sorted(root.rglob("*.json")))
    return files


def _await_cleanup(
    journal_roots: list[Path], *, deadline: float,
) -> tuple[bool, list[dict[str, Any]], float]:
    from hermes_cli import kanban_worker_runtime as runtime
    from hermes_cli.worker_execution_podman import reap_journals

    last: list[dict[str, Any]] = []
    while time.monotonic() < deadline:
        runtime.reap_worker_zombies()
        runtime._reap_worker_execution_journals()
        last = []
        for root in journal_roots:
            if root.is_dir():
                for directory in sorted(root.iterdir()):
                    if directory.is_dir() and not directory.is_symlink():
                        entries, _examined = reap_journals(directory)
                        last.extend(entries)
        if not _journal_files(journal_roots):
            return True, last, time.time()
        time.sleep(1.0)
    return False, last, 0.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-root", required=True, help="hermes-cto checkout with the guard plugin")
    parser.add_argument("--image-store", required=True, help="owner-only OCI archive store")
    parser.add_argument("--receipt", help="write the JSON receipt here")
    parser.add_argument("--prepare-images", action="store_true", help="supervised image preparation only")
    parser.add_argument("--case", choices=["all", "main", "death", "recovery"], default="all")
    args = parser.parse_args(argv)
    policy_root = Path(args.policy_root).resolve()
    image_store = Path(args.image_store).expanduser()
    if not (policy_root / "plugins" / "consequence_guard" / "plugin.yaml").is_file():
        fail(f"policy root has no consequence guard plugin: {policy_root}")
    if args.prepare_images:
        prepare_images(image_store)
        print("worker execution images: ok")
        return 0
    _validate_images(image_store)
    _enter_autonomous_scope()
    free = shutil.disk_usage(image_store).free
    if free < 8 * 1024 * 1024 * 1024:
        fail(f"verification requires at least 8 GiB free, found {free // (1024 ** 3)} GiB")
    receipts: dict[str, Any] = {}
    directory = tempfile.mkdtemp(prefix="worker-execution-smoke-")
    root = Path(directory)
    cases = ["main", "death"] if args.case == "all" else [args.case]
    # "all" deliberately preserves the historical main+death contract.
    failed = False
    reconciled = True
    preserved: str | None = None
    try:
        for case in cases:
            try:
                if case == "recovery":
                    receipts[case] = _run_recovery_case(
                        root / case, policy_root=policy_root, image_store=image_store,
                    )
                else:
                    receipts[case] = _run_case(
                        root / case, policy_root=policy_root, image_store=image_store,
                        death=case == "death",
                    )
            except BaseException:
                failed = True
                if case == "recovery":
                    observations = root / case / "recovery-observations.json"
                    if observations.is_file():
                        receipts[case] = json.loads(observations.read_text())
                _cancel_worker()
                raise
            if args.receipt:
                receipt_path = Path(args.receipt)
                receipt_path.parent.mkdir(parents=True, exist_ok=True)
                receipt_path.write_text(
                    json.dumps({"cases": receipts}, indent=2, sort_keys=True), encoding="utf-8",
                )
    finally:
        _cancel_worker()
        # Reconcile every fixture-owned execution journal before deleting the fixture
        # home: a failed (for example never-started) payload must not lose the only
        # metadata that reclaims its private engine store.
        reconciled = True
        for case_root in (root / case for case in cases):
            ok, _records, _epoch = _await_cleanup(
                _journal_roots(case_root), deadline=time.monotonic() + 120.0,
            )
            if not ok:
                reconciled = False
                remaining = [str(path) for path in _journal_files(_journal_roots(case_root))]
                print(f"worker execution journals unreconciled: {remaining}", flush=True)
        if not reconciled:
            preserved = str(root)
            print(f"worker execution fixture preserved with its journals: {root}", flush=True)
        elif failed and args.case == "recovery":
            preserved = str(root)
            print(f"worker execution fixture preserved with its evidence: {root}", flush=True)
        elif os.environ.get("HERMES_ISSUE76_KEEP") == "1":
            print(f"worker execution fixture kept: {root}", flush=True)
        else:
            _remove_tree(root)
        if args.receipt:
            receipt_path = Path(args.receipt)
            receipt_path.parent.mkdir(parents=True, exist_ok=True)
            receipt_path.write_text(json.dumps(
                {
                    "cases": receipts,
                    "reconciliation": {"ok": reconciled, "preserved": preserved},
                },
                indent=2, sort_keys=True,
            ), encoding="utf-8")
    if not reconciled:
        print(
            "worker execution surface: FAILED (execution journals were not reconciled; the fixture is preserved)",
            flush=True,
        )
        return 1
    if args.receipt:
        print(f"worker execution receipt: {args.receipt}", flush=True)
    print("worker execution surface: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

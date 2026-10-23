"""Authenticated, read-only startup metadata for a dispatcher-owned worker.

The ONLY authority this module accepts is the dispatcher's final runtime grant,
retained in-process by :func:`_retain_worker_grant` at the one point a worker has
proved its identity (``hermes_cli.kanban_runtime``: the post-import grant and the
validated direct ``grant=True`` release). There is deliberately no tool, CLI,
config or environment surface that can populate that grant: environment values
are only cross-checked for consistency against it, never used as its source.

``worker_startup_snapshot`` reads this worker's own task/run/event rows through a
read-only SQLite connection at the exact board path the dispatcher pinned and
returns bounded location/identity metadata. It performs no lifecycle action — no
adoption, heartbeat, claim, repair, migration, generation sweep, Git, shell or
subprocess execution. The guarantee is no logical board write plus no lifecycle
side effect; a SQLite ``mode=ro`` reader can still interact with the board's
shared-memory/WAL sidecars, which is a filesystem effect, not a database write.
"""

from __future__ import annotations

import contextlib
import json
import os
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:  # runtime imports are lazy: this module must import cheaply
    from hermes_cli.kanban_runtime import RuntimeIdentity

#: Profiles a dispatcher ever assigns to an unattended worker.
_WORKER_PROFILES = frozenset({"cto", "implementer", "reviewer", "tester"})

#: The dispatcher-supplied grant fields this capability is allowed to consume.
_CONTEXT_FIELDS = frozenset({"task_id", "profile", "board", "workspace", "db_path"})

#: Environment that must still agree with the retained grant on every call.
_ENV_FIELDS = (
    ("HERMES_KANBAN_TASK", "task_id"),
    ("HERMES_PROFILE", "profile"),
    ("HERMES_KANBAN_BOARD", "board"),
    ("HERMES_KANBAN_WORKSPACE", "workspace"),
    ("HERMES_KANBAN_CLAIM_LOCK", "claim_lock"),
)

_MAX_SNAPSHOT_BYTES = 16 * 1024
_TASK_SQL = (
    "SELECT status, claim_lock, claim_expires, worker_pid, worker_started_at, "
    "current_run_id, assignee, workspace_path, workspace_kind, branch_name "
    "FROM tasks WHERE id = ?"
)
_RUN_SQL = (
    "SELECT id, task_id, profile, status, claim_lock, claim_expires, worker_pid, "
    "worker_started_at, started_at, ended_at, metadata "
    "FROM task_runs WHERE id = ? AND task_id = ?"
)
_SPAWN_SQL = (
    "SELECT payload FROM task_events "
    "WHERE task_id = ? AND run_id = ? AND kind = 'spawned' ORDER BY id"
)


class WorkerStartupUnavailable(RuntimeError):
    """This worker has no authenticated startup metadata to report.

    Raised for a missing/foreign grant, a process or board state that no longer
    matches it, and any board read that is absent, locked, corrupt or
    incompatible. Callers must treat it as a refusal, never as an empty result.
    """


@dataclass(frozen=True)
class WorkerGrantContext:
    """The retained dispatcher grant this process may report startup metadata for."""

    task_id: str
    profile: str
    board: str
    workspace: str
    db_path: str
    run_id: int
    preparation_id: str
    claim_lock: str
    runtime_identity: "RuntimeIdentity"


_GRANT: Optional[WorkerGrantContext] = None


def _require_text(value: Any, label: str, *, limit: int = 4096) -> str:
    if not isinstance(value, str):
        raise WorkerStartupUnavailable(f"grant {label} is missing")
    text = value.strip()
    if not text or "\x00" in text or len(text) > limit:
        raise WorkerStartupUnavailable(f"grant {label} is malformed")
    return text


def _require_path(value: Any, label: str, *, directory: bool) -> str:
    text = _require_text(value, label, limit=8192)
    if not os.path.isabs(text):
        raise WorkerStartupUnavailable(f"grant {label} is not absolute")
    exists = os.path.isdir(text) if directory else os.path.isfile(text)
    if not exists:
        raise WorkerStartupUnavailable(f"grant {label} does not exist")
    return text


def _retain_worker_grant(grant: dict, actual: "RuntimeIdentity") -> None:
    """Keep the dispatcher's final grant as this process's only startup authority.

    Called by the worker bootstrap AFTER its final identity checks and BEFORE it
    publishes ``HERMES_KANBAN_RUNTIME_GRANTED``. An identical repeat is a no-op; a
    different grant for an already-retained process is refused.
    """
    global _GRANT
    from hermes_cli.kanban_runtime import RuntimeIdentity, process_start_time

    if not isinstance(grant, dict) or grant.get("grant") is not True:
        raise WorkerStartupUnavailable("final runtime grant is required")
    if not isinstance(actual, RuntimeIdentity):
        raise WorkerStartupUnavailable("granted runtime identity is required")
    if not actual.generation or not actual.dependency_fingerprint:
        raise WorkerStartupUnavailable("granted runtime identity is not sealed")
    if actual.pid != os.getpid() or actual.start_time != process_start_time():
        raise WorkerStartupUnavailable("granted runtime identity is not this process")
    context = grant.get("worker_context")
    if not isinstance(context, dict) or set(context) != _CONTEXT_FIELDS:
        raise WorkerStartupUnavailable("grant worker context is missing")
    run_id = grant.get("run_id")
    if isinstance(run_id, bool) or not isinstance(run_id, int) or run_id <= 0:
        raise WorkerStartupUnavailable("grant run is missing")
    candidate = WorkerGrantContext(
        task_id=_require_text(context["task_id"], "task"),
        profile=_require_text(context["profile"], "profile"),
        board=_require_text(context["board"], "board"),
        workspace=_require_path(context["workspace"], "workspace", directory=True),
        db_path=_require_path(context["db_path"], "board path", directory=False),
        run_id=run_id,
        preparation_id=_require_text(grant.get("preparation_id"), "preparation"),
        claim_lock=_require_text(grant.get("claim_lock"), "claim lock"),
        runtime_identity=actual,
    )
    if _GRANT is not None:
        if _GRANT == candidate:
            return
        raise WorkerStartupUnavailable("worker startup grant is already retained")
    _GRANT = candidate


def _retained_grant() -> WorkerGrantContext:
    grant = _GRANT
    if grant is None:
        raise WorkerStartupUnavailable("no runtime grant for this process")
    if grant.runtime_identity.pid != os.getpid():
        raise WorkerStartupUnavailable("retained grant belongs to another process")
    return grant


def worker_startup_available() -> bool:
    """Cheap catalog/check_fn availability; never opens a board or imports a runtime."""
    try:
        if not sys.platform.startswith("linux"):
            return False
        grant = _retained_grant()
        from agent.delegation_context import is_dispatcher_owned_worker_context

        if not is_dispatcher_owned_worker_context():
            return False
        return grant.profile in _WORKER_PROFILES
    except Exception:
        return False


def _authorize() -> WorkerGrantContext:
    """Re-establish every grant condition; visibility is not authorization."""
    if not sys.platform.startswith("linux"):
        # The native macOS liveness path spawns ``ps``; refuse before any I/O.
        raise WorkerStartupUnavailable("worker startup inspection is Linux-only")
    grant = _retained_grant()
    if sys.modules.get("hermes_cli.kanban_runtime") is None:
        # A retained grant cannot exist without this module, so this is a defensive
        # refusal: never import a runtime on behalf of an unbound call.
        raise WorkerStartupUnavailable("worker runtime is not loaded")
    from hermes_cli.kanban_runtime import (
        assert_runtime_import_root,
        process_start_time,
        runtime_identity,
        same_runtime_identity,
    )

    if process_start_time() != grant.runtime_identity.start_time:
        raise WorkerStartupUnavailable("process start marker no longer matches the grant")
    frozen = runtime_identity()
    if not same_runtime_identity(grant.runtime_identity, frozen):
        raise WorkerStartupUnavailable("loaded runtime no longer matches the grant")
    try:
        assert_runtime_import_root(expected=grant.runtime_identity, identity=frozen)
    except Exception as exc:
        raise WorkerStartupUnavailable(f"loaded runtime is not the granted runtime: {exc}")
    from agent.delegation_context import is_delegated_child_process_context, is_dispatcher_owned_worker_context

    if is_delegated_child_process_context() or not is_dispatcher_owned_worker_context():
        raise WorkerStartupUnavailable("not a dispatcher-owned worker context")
    if grant.profile not in _WORKER_PROFILES:
        raise WorkerStartupUnavailable("granted profile is not an unattended worker role")
    if os.environ.get("HERMES_KANBAN_RUNTIME_GRANTED") != "1":
        raise WorkerStartupUnavailable("runtime grant flag is not set")
    if os.environ.get("HERMES_KANBAN_RUN_ID", "") != str(grant.run_id):
        raise WorkerStartupUnavailable("current run does not match the grant")
    for variable, field in _ENV_FIELDS:
        if os.environ.get(variable, "") != getattr(grant, field):
            raise WorkerStartupUnavailable(f"current {field} does not match the grant")
    return grant


def _open_board(path: str) -> sqlite3.Connection:
    """Open the pinned board read-only; never create, migrate, repair or retry."""
    try:
        connection = sqlite3.connect(
            Path(path).as_uri() + "?mode=ro", uri=True, timeout=2.0,
            isolation_level=None,
        )
    except sqlite3.Error as exc:
        raise WorkerStartupUnavailable(f"board is unavailable: {exc}")
    try:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only = ON")
        connection.execute("BEGIN")
    except sqlite3.Error as exc:
        connection.close()
        raise WorkerStartupUnavailable(f"board is unavailable: {exc}")
    return connection


def _same_existing_path(left: Any, right: str) -> bool:
    """True only for the same absolute, still-present workspace as the grant."""
    if not isinstance(left, str) or not left.strip() or not os.path.isabs(left.strip()):
        return False
    if os.path.abspath(left.strip()) != os.path.abspath(right):
        return False
    return os.path.isdir(right)


def _run_metadata(row: sqlite3.Row) -> dict:
    raw = row["metadata"]
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise WorkerStartupUnavailable(f"run metadata is malformed: {exc}")
    if not isinstance(parsed, dict):
        raise WorkerStartupUnavailable("run metadata is malformed")
    return parsed


def _spawn_receipt(connection: sqlite3.Connection, grant: WorkerGrantContext) -> str:
    """``verified`` only for a present identity-bearing receipt; else ``pending``.

    The retained grant plus the claim-time run metadata already authenticate this
    call, and the post-grant ``spawned`` record can lag or fail, so its absence is
    reported as pending rather than as a failure.
    """
    from hermes_cli.kanban_runtime import RuntimeIdentity, RuntimeIdentityError, same_runtime_identity

    receipts = []
    for row in connection.execute(_SPAWN_SQL, (grant.task_id, grant.run_id)):
        try:
            payload = json.loads(row["payload"] or "{}")
        except (TypeError, ValueError) as exc:
            raise WorkerStartupUnavailable(f"spawned receipt is malformed: {exc}")
        if not isinstance(payload, dict):
            raise WorkerStartupUnavailable("spawned receipt is malformed")
        receipts.append(payload)
    if not receipts:
        return "pending"
    verified = False
    for payload in receipts:
        pid = payload.get("pid")
        if isinstance(pid, bool) or not isinstance(pid, int) or pid != grant.runtime_identity.pid:
            raise WorkerStartupUnavailable("spawned receipt names another process")
        if "runtime_identity" not in payload:
            continue  # legacy PID-only receipt: matches, but proves nothing further
        try:
            recorded = RuntimeIdentity.from_value(payload["runtime_identity"])
        except (RuntimeIdentityError, TypeError, ValueError) as exc:
            raise WorkerStartupUnavailable(f"spawned receipt identity is malformed: {exc}")
        if not same_runtime_identity(recorded, grant.runtime_identity):
            raise WorkerStartupUnavailable("spawned receipt identity conflicts with the grant")
        if str(payload.get("preparation_id") or "") != grant.preparation_id:
            raise WorkerStartupUnavailable("spawned receipt preparation conflicts with the grant")
        board = payload.get("board")
        if board is not None and str(board) != grant.board:
            raise WorkerStartupUnavailable("spawned receipt board conflicts with the grant")
        verified = True
    return "verified" if verified else "pending"


def _read_startup(connection: sqlite3.Connection, grant: WorkerGrantContext) -> dict:
    from hermes_cli import kanban_db
    from hermes_cli.kanban_runtime import same_runtime_identity

    identity = grant.runtime_identity
    task = connection.execute(_TASK_SQL, (grant.task_id,)).fetchone()
    if task is None:
        raise WorkerStartupUnavailable("worker task is not on the retained board")
    run = connection.execute(_RUN_SQL, (grant.run_id, grant.task_id)).fetchone()
    if run is None:
        raise WorkerStartupUnavailable("worker run is not on the retained board")

    if task["status"] != "running" or run["status"] != "running" or run["ended_at"] is not None:
        raise WorkerStartupUnavailable("worker run is no longer running")
    for row, label in ((task, "task"), (run, "run")):
        if not row["claim_lock"] or str(row["claim_lock"]) != grant.claim_lock:
            raise WorkerStartupUnavailable(f"{label} claim is no longer owned by this grant")
        if row["worker_pid"] != identity.pid:
            raise WorkerStartupUnavailable(f"{label} pid no longer matches the grant")
    if task["current_run_id"] != grant.run_id or run["id"] != grant.run_id:
        raise WorkerStartupUnavailable("worker run is no longer the claimed run")
    if str(task["assignee"] or "") != grant.profile or str(run["profile"] or "") != grant.profile:
        raise WorkerStartupUnavailable("worker profile no longer matches the grant")
    if not _same_existing_path(task["workspace_path"], grant.workspace):
        raise WorkerStartupUnavailable("worker workspace no longer matches the grant")
    metadata = _run_metadata(run)
    if not same_runtime_identity(identity, metadata.get("runtime_identity") or {}):
        raise WorkerStartupUnavailable("claim-time runtime identity does not match the grant")
    if (
        metadata.get("worker_pid") != identity.pid
        or metadata.get("worker_start_time") != identity.start_time
        or str(metadata.get("preparation_id") or "") != grant.preparation_id
    ):
        raise WorkerStartupUnavailable("claim-time worker provenance does not match the grant")
    # Native liveness semantics in addition to (never instead of) the exact
    # fingerprint above: an expired-but-live native claim stays valid, and a
    # recycled PID refuses instead of being repaired here.
    try:
        live = bool(kanban_db._claim_is_live(task))
    except Exception as exc:
        raise WorkerStartupUnavailable(f"worker liveness is unavailable: {exc}")
    if not live:
        raise WorkerStartupUnavailable("worker claim is not live")

    snapshot = {
        "task_id": grant.task_id,
        "run_id": grant.run_id,
        "profile": grant.profile,
        "board": grant.board,
        "workspace_path": os.path.abspath(grant.workspace),
        "workspace_kind": str(task["workspace_kind"] or ""),
        "branch_name": str(task["branch_name"] or ""),
        "preparation_id": grant.preparation_id,
        "runtime_identity": identity.as_dict(),
        "spawn_receipt": _spawn_receipt(connection, grant),
    }
    encoded = json.dumps(snapshot, sort_keys=True)
    if len(encoded.encode("utf-8")) > _MAX_SNAPSHOT_BYTES:
        raise WorkerStartupUnavailable("startup metadata exceeds the reporting bound")
    return snapshot


def worker_startup_snapshot() -> dict:
    """Return this worker's authenticated startup metadata as a fresh dictionary."""
    grant = _authorize()
    connection = _open_board(grant.db_path)
    try:
        return _read_startup(connection, grant)
    except sqlite3.Error as exc:
        raise WorkerStartupUnavailable(f"board is unavailable: {exc}")
    finally:
        with contextlib.suppress(sqlite3.Error):
            connection.execute("ROLLBACK")
        connection.close()

"""Kanban board export / import — move a whole board between machines.

Backs ``hermes kanban export|import``, the ``/boards/{slug}/export`` and
``/boards/import`` REST endpoints, and the desktop board switcher. Archive
layout (``<slug>.tar.gz``, one top-level dir named for the source slug):
``manifest.json`` (format/version/provenance/counts), ``board.json`` (display
metadata, machine-local fields stripped), ``kanban.db`` (consistent snapshot),
``attachments/<task>/…`` (unless --no-attachments), ``logs/<task>.log`` (only
with --include-logs).

Two things make this more than ``tar czf`` of the board directory: the DB is
live (WAL mode, dispatcher may be mid-write) so export uses SQLite's online
backup instead of a file copy that would miss the ``-wal`` sidecar; and rows
carry machine-local state (claims, PIDs, heartbeats, absolute paths, gateway
chat subscriptions, session ids) that would import a stranger's claims or push
events into a stranger's Telegram thread — scrubbed on export and re-scrubbed
on import (an archive is untrusted); see :func:`_scrub_local_state` and
:func:`_relocate_imported_rows`. Imports always land as a **new** board (slug
auto-suffixes on collision), never ``default``, so the importer can ignore the
default board's split on-disk layout.
"""

from __future__ import annotations

import contextlib
import json
import shutil
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli.archive_safe import (
    archive_root_dirs,
    copy_regular_files,
    make_targz,
    safe_extract_targz,
)

ARCHIVE_FORMAT = "hermes-kanban-board"
ARCHIVE_FORMAT_VERSION = 1

# Statuses from which the dispatcher can still act on a task. A task whose
# workspace cannot be rebuilt on this machine is parked in ``triage`` only
# if it is in one of these — terminal and already-parked tasks are left
# alone rather than having their history rewritten.
_DISPATCHABLE_STATUSES = ("ready", "running", "todo", "scheduled", "review")
_COUNTED_TABLES = ("tasks", "task_links", "task_comments", "task_events", "task_runs", "task_attachments")


def _placeholders(items) -> str:
    return ", ".join("?" * len(items))


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def _snapshot_db(source: Path, target: Path) -> None:
    """Consistent copy of ``source`` via the online-backup API (a file copy
    would miss pages still in the ``-wal`` sidecar and could tear)."""
    with contextlib.closing(sqlite3.connect(str(source))) as src, \
            contextlib.closing(sqlite3.connect(str(target))) as dst:
        src.backup(dst)


def _scrub_local_state(conn: sqlite3.Connection) -> None:
    """Strip machine-local runtime state from an exported or imported board."""
    conn.execute("DELETE FROM kanban_notify_subs")
    running = conn.execute(
        "SELECT id, current_run_id FROM tasks WHERE status = 'running' ORDER BY id"
    ).fetchall()
    for row in running:
        retry_status = kb._retry_status_for_run(conn, row["id"], row["current_run_id"])
        conn.execute(
            "UPDATE tasks SET status = ?, claim_lock = NULL, claim_expires = NULL, "
            "worker_pid = NULL, current_run_id = NULL, last_heartbeat_at = NULL, "
            "session_id = NULL, project_id = NULL, consecutive_failures = 0, "
            "last_failure_error = NULL WHERE id = ? AND status = 'running'",
            (retry_status, row["id"]),
        )
    conn.execute(
        """
        UPDATE tasks
           SET claim_lock           = NULL,
               claim_expires        = NULL,
               worker_pid           = NULL,
               current_run_id       = NULL,
               last_heartbeat_at    = NULL,
               session_id           = NULL,
               project_id           = NULL,
               consecutive_failures = 0,
               last_failure_error   = NULL
         WHERE status != 'running'
        """
    )
    conn.execute(
        """
        UPDATE task_runs
           SET status            = 'released',
               outcome           = COALESCE(outcome, 'reclaimed'),
               ended_at          = COALESCE(ended_at, ?),
               last_heartbeat_at = NULL
         WHERE status = 'running'
        """,
        (int(time.time()),),
    )
    conn.execute("UPDATE task_runs SET claim_lock = NULL, worker_pid = NULL")
    _scrub_handoff_paths(conn)


def _scrub_handoff_paths(conn: sqlite3.Connection) -> None:
    """Remove exporter-local paths from durable handoff records."""
    for row in conn.execute(
        "SELECT id, metadata FROM task_runs WHERE metadata IS NOT NULL"
    ).fetchall():
        try:
            payload = json.loads(row["metadata"])
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        changed = False
        for key in ("branch_name", "workspace_path", "patch_artifact"):
            if payload.pop(key, None) is not None:
                changed = True
        routing = payload.get("lifecycle_routing")
        if isinstance(routing, dict):
            for key in ("branch_name", "workspace_path", "patch_artifact"):
                if routing.pop(key, None) is not None:
                    changed = True
        if changed:
            conn.execute(
                "UPDATE task_runs SET metadata = ? WHERE id = ?",
                (json.dumps(payload, ensure_ascii=False, sort_keys=True), row["id"]),
            )
    for row in conn.execute(
        "SELECT id, kind, payload FROM task_events "
        "WHERE kind IN ('completed', 'review_requested', 'handoff_requeued') "
        "AND payload IS NOT NULL"
    ).fetchall():
        try:
            payload = json.loads(row["payload"])
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        changed = False
        for key in ("branch_name", "workspace_path", "patch_artifact"):
            if payload.pop(key, None) is not None:
                changed = True
        legacy = payload.get("legacy_handoff")
        if isinstance(legacy, dict):
            for key in ("branch_name", "workspace_path", "patch_artifact"):
                if legacy.pop(key, None) is not None:
                    changed = True
        workspace = payload.get("workspace")
        if isinstance(workspace, dict):
            for key in ("branch", "path"):
                if workspace.pop(key, None) is not None:
                    changed = True
        if changed:
            conn.execute(
                "UPDATE task_events SET payload = ? WHERE id = ?",
                (json.dumps(payload, ensure_ascii=False, sort_keys=True), row["id"]),
            )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _count_rows(conn: sqlite3.Connection) -> dict[str, int]:
    return {t: int(conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]) for t in _COUNTED_TABLES}


def export_board(
    board: Optional[str],
    output_path: str,
    *,
    include_attachments: bool = True,
    include_logs: bool = False,
) -> dict[str, Any]:
    """Export ``board`` to a ``tar.gz`` (suffix optional on ``output_path``);
    returns a summary dict. Workspaces are never included — large,
    machine-local, rebuilt on demand."""
    slug = kb._normalize_board_slug(board) or kb.get_current_board()
    if not kb.board_exists(slug):
        raise ValueError(f"board {slug!r} does not exist")

    db_path = kb.kanban_db_path(slug)
    if not db_path.exists():
        raise FileNotFoundError(f"board {slug!r} has no database at {db_path}")

    base = str(Path(output_path).expanduser()).removesuffix(".tar.gz").removesuffix(".tgz")
    Path(base).parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        staged = Path(tmpdir) / slug
        staged.mkdir(parents=True)

        _snapshot_db(db_path, staged / "kanban.db")
        # The snapshot is a private file with no other writers, so plain
        # commit/close is enough — no need for the board DB's WAL dance.
        with contextlib.closing(sqlite3.connect(str(staged / "kanban.db"))) as snapshot:
            _scrub_local_state(snapshot)
            snapshot.commit()
            counts = _count_rows(snapshot)

        meta = kb.read_board_metadata(slug)
        # Both name a location on the exporting machine; the importer
        # resolves its own.
        meta.pop("db_path", None)
        meta["default_workdir"] = None
        meta["project_id"] = None
        _write_json(staged / "board.json", meta)

        attachments = copy_regular_files(kb.attachments_root(slug), staged / "attachments") if include_attachments else 0
        logs = copy_regular_files(kb.worker_logs_dir(slug), staged / "logs") if include_logs else 0

        try:
            from hermes_cli import __version__ as hermes_version
        except Exception:
            hermes_version = ""

        manifest = {
            "format": ARCHIVE_FORMAT,
            "format_version": ARCHIVE_FORMAT_VERSION,
            "board": slug,
            "board_name": meta.get("name") or slug,
            "exported_at": int(time.time()),
            "hermes_version": str(hermes_version),
            "includes": {"attachments": bool(include_attachments), "logs": bool(include_logs)},
            "counts": {**counts, "attachment_files": attachments, "log_files": logs},
        }
        _write_json(staged / "manifest.json", manifest)

        archive = make_targz(base, tmpdir, slug)

    return {
        "board": slug,
        "archive": archive,
        "size": Path(archive).stat().st_size,
        "counts": manifest["counts"],
    }


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------

def _available_slug(preferred: str) -> str:
    """``preferred`` or the first free ``<preferred>-N``. ``default`` always
    exists, so a default-board export lands as ``default-2``."""
    if not kb.board_exists(preferred):
        return preferred
    # Leave headroom for the suffix inside the 64-char slug limit.
    stem = preferred[:58].rstrip("-_") or "board"
    n = 2
    while kb.board_exists(f"{stem}-{n}"):
        n += 1
    return f"{stem}-{n}"


def _read_manifest(root: Path) -> dict[str, Any]:
    path = root / "manifest.json"
    if not path.exists():
        raise ValueError("archive is not a Hermes kanban board export (no manifest.json)")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"archive manifest is not valid JSON: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("format") != ARCHIVE_FORMAT:
        raise ValueError(
            "archive is not a Hermes kanban board export "
            f"(format={manifest.get('format') if isinstance(manifest, dict) else None!r})"
        )
    version = manifest.get("format_version")
    if not isinstance(version, int) or version > ARCHIVE_FORMAT_VERSION:
        raise ValueError(
            f"archive format version {version!r} is newer than this Hermes "
            f"understands (max {ARCHIVE_FORMAT_VERSION}) — update Hermes and retry"
        )
    return manifest


def _read_board_metadata(path: Path) -> dict[str, Any]:
    """Read an archive's ``board.json``, tolerating a missing/broken file."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return raw if isinstance(raw, dict) else {}


def _is_same_card_review(
    conn: sqlite3.Connection,
    task_id: str,
    status: str,
    contract: Optional[dict],
) -> bool:
    return bool(
        contract
        and contract.get("kind") == "code"
        and contract.get("review_mode") == "same_card"
        and (
            status == "review"
            or (
                status == "blocked"
                and kb._resume_status_from_events(conn, task_id) == "review"
            )
        )
    )


def _candidate_handoff_ids(conn: sqlite3.Connection, row: sqlite3.Row) -> list[str]:
    contract = kb.safe_decode_contract(row["lifecycle_contract"])
    kind = contract.get("kind") if contract else None
    same_card = _is_same_card_review(conn, row["id"], row["status"], contract)
    is_role = kind in {"review", "validation"} or (
        not contract
        and str(row["assignee"] or "").strip().casefold() in kb.HANDOFF_CHILD_ASSIGNEES
    )
    if not same_card and not is_role:
        return []
    candidate_ids = [row["id"], *kb.parent_ids(conn, row["id"])]
    candidate_id = contract.get("candidate_task_id") if contract else None
    if candidate_id:
        candidate_ids.append(str(candidate_id))
    seen: set[str] = set()
    candidate_handoffs: list[str] = []
    for candidate_id in candidate_ids:
        if candidate_id in seen:
            continue
        seen.add(candidate_id)
        if kb.latest_handoff(conn, candidate_id).get("head_sha"):
            candidate_handoffs.append(candidate_id)
    return candidate_handoffs


def _candidate_handoff_needs_rebind(conn: sqlite3.Connection, row: sqlite3.Row) -> bool:
    """True when an imported review lane needs a local immutable handoff."""
    return bool(_candidate_handoff_ids(conn, row))
def _requeue_imported_candidate(
    conn: sqlite3.Connection, task_id: str,
) -> tuple[list[str], list[str]]:
    """Invalidate a typed candidate whose workspace was not transferred."""
    candidate = kb.get_task(conn, task_id)
    contract = candidate.lifecycle_contract if candidate else None
    if (
        candidate is None
        or not contract
        or contract.get("kind") != "code"
        or (
            candidate.status not in {"done", "review"}
            and not _is_same_card_review(
                conn, candidate.id, candidate.status, contract,
            )
        )
    ):
        return [], []

    parent_ids = kb.parent_ids(conn, task_id)
    parent_rows = [kb.get_task(conn, parent_id) for parent_id in parent_ids]
    candidate_status = (
        "triage"
        if candidate.workspace_kind in {"dir", "worktree"}
        else "ready"
        if all(parent is not None and parent.status in {"done", "archived"} for parent in parent_rows)
        else "todo"
    )
    rows = conn.execute(
        """
        WITH RECURSIVE graph(id) AS (
            SELECT ?
            UNION
            SELECT child_id FROM task_links JOIN graph ON parent_id = graph.id
        )
        SELECT t.id, t.status, t.version, t.completed_at, t.result
        FROM graph JOIN tasks t ON t.id = graph.id ORDER BY t.id
        """,
        (task_id,),
    ).fetchall()
    invalidated: list[dict[str, Any]] = []
    for row in rows:
        if row["status"] == "archived":
            continue
        new_status = candidate_status if row["id"] == task_id else (
            "blocked"
            if row["status"] == "blocked" and kb._has_sticky_block(conn, row["id"])
            else "triage"
            if candidate_status == "triage"
            else "todo"
        )
        conn.execute(
            """
            UPDATE tasks SET status = ?, version = version + 1,
                completed_at = NULL, result = NULL, current_run_id = NULL,
                claim_lock = NULL, claim_expires = NULL, worker_pid = NULL,
                last_heartbeat_at = NULL, session_id = NULL, project_id = NULL,
                candidate_run_id = NULL, consecutive_failures = 0,
                last_failure_error = NULL,
                block_kind = CASE WHEN ? = 'blocked' THEN block_kind ELSE NULL END,
                block_recurrences = CASE WHEN ? = 'blocked' THEN block_recurrences ELSE 0 END
            WHERE id = ?
            """,
            (new_status, new_status, new_status, row["id"]),
        )
        invalidated.append({
            "id": row["id"],
            "prior_status": row["status"],
            "new_status": new_status,
            "prior_version": int(row["version"] or 1),
            "prior_completed_at": row["completed_at"],
            "prior_result": row["result"],
        })
    if invalidated:
        kb._append_event(
            conn,
            task_id,
            "import_requeued",
            {
                "reason": "typed immutable handoff requires a local workspace",
                "candidate_status": candidate_status,
                "invalidated": invalidated,
            },
        )
    invalidated_ids = [entry["id"] for entry in invalidated]
    triaged_ids = [entry["id"] for entry in invalidated if entry["new_status"] == "triage"]
    return invalidated_ids, triaged_ids

def _relocate_imported_rows(conn: sqlite3.Connection, slug: str) -> tuple[dict[str, int], list[str]]:
    """Re-anchor an imported board's rows to this machine; returns ``(stats, warnings)``.

    * Attachment rows are repointed at this board's tree; rows whose blob
      did not travel (``--no-attachments``) are dropped, since a dangling row
      breaks download in every UI.
    * Workspace paths are cleared. ``scratch`` regenerates on next claim;
      dispatchable ``dir``/``worktree`` tasks are parked in ``triage``.
      Typed candidates with immutable handoffs are requeued for a fresh
      scratch workspace or parked when their workspace needs rebinding.
    * Runtime state is scrubbed again (untrusted input, one UPDATE).
    """
    warnings: list[str] = []
    now = int(time.time())
    attachments_dir = kb.attachments_root(slug)

    with kb.write_txn(conn):
        _scrub_local_state(conn)

        dropped = rehomed = 0
        for row in conn.execute("SELECT id, task_id, stored_path FROM task_attachments").fetchall():
            landed = attachments_dir / row["task_id"] / Path(row["stored_path"]).name
            if landed.is_file():
                conn.execute("UPDATE task_attachments SET stored_path = ? WHERE id = ?", (str(landed), row["id"]))
                rehomed += 1
            else:
                conn.execute("DELETE FROM task_attachments WHERE id = ?", (row["id"],))
                dropped += 1
        if dropped:
            warnings.append(f"{dropped} attachment record(s) dropped — the files were not in the archive")

        tasks = conn.execute(
            "SELECT id, status, workspace_kind, assignee, lifecycle_contract FROM tasks"
        ).fetchall()
        candidate_roots: set[str] = set()
        for row in tasks:
            if row["status"] not in _DISPATCHABLE_STATUSES and row["status"] != "blocked":
                continue
            for candidate_id in _candidate_handoff_ids(conn, row):
                candidate = kb.get_task(conn, candidate_id)
                contract = candidate.lifecycle_contract if candidate else None
                if (
                    candidate is not None
                    and contract
                    and contract.get("kind") == "code"
                    and (
                        candidate.status in {"done", "review"}
                        or _is_same_card_review(
                            conn, candidate.id, candidate.status, contract,
                        )
                    )
                ):
                    candidate_roots.add(candidate_id)

        requeued_ids: set[str] = set()
        triaged_requeued: set[str] = set()
        scratch_requeued = triage_requeued = 0
        for candidate_id in sorted(candidate_roots):
            invalidated_ids, triaged_ids = _requeue_imported_candidate(conn, candidate_id)
            if not invalidated_ids:
                continue
            requeued_ids.update(invalidated_ids)
            triaged_requeued.update(triaged_ids)
            if triaged_ids:
                triage_requeued += 1
            else:
                scratch_requeued += 1

        tasks = conn.execute(
            "SELECT id, status, workspace_kind, assignee, lifecycle_contract FROM tasks"
        ).fetchall()
        workspace_parked = [
            row["id"]
            for row in tasks
            if row["workspace_kind"] in {"dir", "worktree"}
            and row["status"] in _DISPATCHABLE_STATUSES
        ]
        workspace_parked_set = set(workspace_parked)
        candidate_parked = [
            row["id"]
            for row in tasks
            if row["id"] not in workspace_parked_set
            and row["id"] not in requeued_ids
            and row["status"] in _DISPATCHABLE_STATUSES
            and _candidate_handoff_needs_rebind(conn, row)
        ]
        parked = list(dict.fromkeys(
            workspace_parked + sorted(triaged_requeued) + candidate_parked
        ))
        conn.execute("UPDATE tasks SET workspace_path = NULL, branch_name = NULL")
        if parked:
            conn.execute(f"UPDATE tasks SET status = 'triage' WHERE id IN ({_placeholders(parked)})", parked)
        if scratch_requeued:
            warnings.append(
                f"{scratch_requeued} typed candidate(s) requeued with a fresh local scratch workspace"
            )
        if triage_requeued:
            warnings.append(
                f"{triage_requeued} typed candidate graph(s) moved to triage — "
                "their workspace needs a local rebind before work can resume"
            )
        if workspace_parked:
            warnings.append(
                f"{len(workspace_parked)} task(s) moved to triage — their workspace was a directory or git "
                f"worktree on the exporting machine and needs to be pointed somewhere on this one"
            )
        if candidate_parked:
            warnings.append(
                f"{len(candidate_parked)} candidate task(s) moved to triage — their immutable handoff "
                "needs a local workspace and branch rebind before review can resume"
            )

        for row in conn.execute("SELECT id FROM tasks").fetchall():
            conn.execute(
                "INSERT INTO task_events (task_id, run_id, kind, payload, created_at) "
                "VALUES (?, NULL, 'imported', ?, ?)",
                (row["id"], json.dumps({"board": slug, "parked": row["id"] in parked}, ensure_ascii=False), now),
            )

    return {"attachments": rehomed, "parked": len(parked)}, warnings


def import_board(
    archive_path: str,
    slug: Optional[str] = None,
    *,
    activate: bool = False,
) -> dict[str, Any]:
    """Import an archive as a NEW board (``slug`` overrides the archive's;
    either way it auto-suffixes if taken). Returns a summary dict."""
    archive = Path(archive_path).expanduser()
    if not archive.exists():
        raise FileNotFoundError(f"archive not found: {archive}")

    roots = archive_root_dirs(archive)
    if len(roots) != 1:
        raise ValueError("a kanban board archive must contain exactly one top-level directory")
    archive_root = roots.pop()

    with tempfile.TemporaryDirectory() as tmpdir:
        staging = Path(tmpdir)
        safe_extract_targz(archive, staging)
        extracted = staging / archive_root

        manifest = _read_manifest(extracted)
        staged_db = extracted / "kanban.db"
        if not staged_db.is_file():
            raise ValueError("archive is missing kanban.db")

        requested = kb._normalize_board_slug(slug or manifest.get("board") or archive_root)
        if not requested:
            raise ValueError(
                "cannot determine a board name from the archive — pass one "
                "explicitly with --as <slug>"
            )
        target = _available_slug(requested)

        staged_meta = _read_board_metadata(extracted / "board.json")

        board_root = kb.board_dir(target)
        board_root.mkdir(parents=True, exist_ok=True)
        shutil.move(str(staged_db), str(board_root / "kanban.db"))
        for tree in ("attachments", "logs"):
            src = extracted / tree
            if src.is_dir():
                shutil.move(str(src), str(board_root / tree))

    # Rewritten rather than moved across: the archive's copy names a slug
    # and a workdir that belong to the exporting machine.
    name = str(staged_meta.get("name") or manifest.get("board_name") or target)
    kb.write_board_metadata(
        target,
        name=name,
        description=str(staged_meta.get("description") or ""),
        icon=str(staged_meta.get("icon") or ""),
        color=str(staged_meta.get("color") or ""),
        archived=False,
    )
    # Bring the imported schema up to this install's version before the
    # relocation pass writes to it.
    kb.init_db(board=target)

    with kbc.connect_closing(board=target) as conn:
        stats, warnings = _relocate_imported_rows(conn, target)
        counts = _count_rows(conn)

    if activate:
        kb.set_current_board(target)

    return {
        "board": target,
        "requested_board": requested,
        "renamed": target != requested,
        "name": name,
        "path": str(kb.board_dir(target)),
        "db_path": str(kb.kanban_db_path(target)),
        "source": {
            "board": manifest.get("board"),
            "exported_at": manifest.get("exported_at"),
            "hermes_version": manifest.get("hermes_version"),
        },
        "counts": counts,
        "attachments_restored": stats["attachments"],
        "tasks_parked": stats["parked"],
        "warnings": warnings,
        "activated": bool(activate),
    }

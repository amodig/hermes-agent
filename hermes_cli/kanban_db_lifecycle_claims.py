"""Kanban claim, dependency, and stale-worker lifecycle paths."""
from __future__ import annotations

import sqlite3
import time
from typing import Any, Optional

from hermes_cli.kanban_db_lazy import _kb
from hermes_cli.kanban_db_lifecycle_evidence import (
    _parent_handoff_start_error,
    _record_parent_handoff_start_error,
)
from hermes_cli.kanban_lifecycle import (
    LifecycleContractError,
    evaluate_dependencies,
    safe_decode_contract,
)

def _parents_satisfied(conn: sqlite3.Connection, task_id: str) -> bool:
    """Compatibility adapter for every dependency gate."""
    return bool(evaluate_dependencies(conn, task_id).get("satisfied"))

def _runtime_claim_metadata(
    runtime_identity: Any = None,
    worker_pid: Optional[int] = None,
    worker_start_time: Optional[int] = None,
    preparation_id: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    values = (runtime_identity, worker_pid, worker_start_time, preparation_id)
    if not any(value is not None for value in values):
        return None
    if not all(value is not None for value in values):
        raise LifecycleContractError(
            "runtime identity, worker pid, worker start time, and preparation id are all required"
        )
    from hermes_cli.kanban_runtime import decode_identity

    value = runtime_identity.as_dict() if hasattr(runtime_identity, "as_dict") else runtime_identity
    identity = decode_identity(value)
    if identity.pid != int(worker_pid) or identity.start_time != int(worker_start_time):
        raise LifecycleContractError("runtime identity does not match the worker pid/start time")
    return {
        "runtime_identity": identity.as_dict(),
        "worker_pid": int(worker_pid),
        "worker_start_time": int(worker_start_time),
        "preparation_id": str(preparation_id),
    }


_CLAIM_EXECUTION_FIELDS = (
    "version",
    "assignee",
    "tenant",
    "workspace_kind",
    "workspace_path",
    "branch_name",
    "max_runtime_seconds",
    "workflow_template_id",
    "current_step_key",
    "skills",
    "model_override",
    "provider_override",
    "reasoning_effort",
    "goal_mode",
    "goal_max_turns",
)


def _claim_execution_snapshot(task: Any) -> tuple[Any, ...]:
    return tuple(
        tuple(getattr(task, field, None) or ())
        if field == "skills"
        else getattr(task, field, None)
        for field in _CLAIM_EXECUTION_FIELDS
    )


def _claim_snapshot_matches(
    conn: sqlite3.Connection, task_id: str, expected_task: Any,
) -> bool:
    current = _kb.get_task(conn, task_id)
    return (
        current is not None
        and _claim_execution_snapshot(current) == _claim_execution_snapshot(expected_task)
    )


def _claim_rejected_for_snapshot(
    conn: sqlite3.Connection, task_id: str,
) -> None:
    _kb._append_event(
        conn,
        task_id,
        "claim_rejected",
        {"reason": "dispatch_snapshot_changed"},
    )


def _forced_promotion_active(conn: sqlite3.Connection, task_id: str) -> bool:
    """Keep a forced promotion live across assignment bookkeeping only."""
    row = conn.execute(
        "SELECT kind, payload FROM task_events "
        "WHERE task_id = ? AND kind != 'assigned' ORDER BY id DESC LIMIT 1",
        (task_id,),
    ).fetchone()
    return bool(
        row is not None
        and _kb._row_get(row, "kind") == "promoted_manual"
        and _kb._json_dict(_kb._row_get(row, "payload")).get("forced") is True
    )

def _claim_and_open_run(
    conn: sqlite3.Connection, task_id: str, source_status: str, lock: str, expires: int, now: int,
    *, event_extra: Optional[dict] = None, runtime_claim: Optional[dict[str, Any]] = None,
    expected_task: Any = None,
) -> Optional[int]:
    """CAS ``source_status -> running``, open a run row, emit ``claimed``; None
    when the CAS lost. Caller holds the txn."""
    if expected_task is not None and not _claim_snapshot_matches(conn, task_id, expected_task):
        _claim_rejected_for_snapshot(conn, task_id)
        return None
    cur = conn.execute(
        f"""
        UPDATE tasks
           SET status        = 'running',
               claim_lock    = ?,
               claim_expires = ?,
               worker_pid    = ?,
               started_at    = COALESCE(started_at, ?)
         WHERE id = ?
           AND status = '{source_status}'
           AND claim_lock IS NULL
        """,
        (
            lock,
            expires,
            runtime_claim.get("worker_pid") if runtime_claim else None,
            now,
            task_id,
        ),
    )
    if cur.rowcount != 1:
        return None
    trow = conn.execute(
        "SELECT assignee, max_runtime_seconds, current_step_key, workspace_path, "
        "branch_name, lifecycle_contract FROM tasks WHERE id = ?", (task_id,),
    ).fetchone()
    run_metadata = dict(runtime_claim or {})
    contract = safe_decode_contract(_kb._row_get(trow, "lifecycle_contract"))
    if source_status == "ready" and contract and contract.get("kind") == "code":
        run_metadata["lifecycle_routing"] = {
            "implementer": trow["assignee"],
            "reviewer": contract.get("reviewer"),
            "workspace_path": trow["workspace_path"],
            "branch_name": trow["branch_name"],
        }
        conn.execute("UPDATE tasks SET candidate_run_id = NULL WHERE id = ?", (task_id,))
    run_cur = conn.execute(
        """
        INSERT INTO task_runs (
            task_id, profile, step_key, status, metadata,
            claim_lock, claim_expires, worker_pid, max_runtime_seconds,
            started_at
        ) VALUES (?, ?, ?, 'running', ?, ?, ?, ?, ?, ?)
        """,
        (
            task_id,
            trow["assignee"] if trow else None,
            trow["current_step_key"] if trow else None,
            _kb._json_or_null(run_metadata or None),
            lock,
            expires,
            runtime_claim.get("worker_pid") if runtime_claim else None,
            trow["max_runtime_seconds"] if trow else None,
            now,
        ),
    )
    run_id = run_cur.lastrowid
    conn.execute("UPDATE tasks SET current_run_id = ? WHERE id = ?", (run_id, task_id))
    _kb._append_event(
        conn, task_id, "claimed",
        {"lock": lock, "expires": expires, "run_id": run_id, **(event_extra or {})}, run_id=run_id,
    )
    return run_id

def claim_task(
    conn: sqlite3.Connection, task_id: str, *, ttl_seconds: Optional[int] = None,
    claimer: Optional[str] = None, runtime_identity: Any = None,
    worker_pid: Optional[int] = None, worker_start_time: Optional[int] = None,
    preparation_id: Optional[str] = None,
    expected_task: Any = None,
) -> Optional[_kb.Task]:
    """Atomically transition ``ready -> running``.

    Returns the claimed ``Task`` on success, ``None`` if the task was
    already claimed (or is not in ``ready`` status).
    """
    runtime_claim = _runtime_claim_metadata(
        runtime_identity, worker_pid, worker_start_time, preparation_id,
    )
    now = int(time.time())
    lock = claimer or _kb._claimer_id()
    expires = now + _kb._resolve_claim_ttl_seconds(ttl_seconds)
    with _kb.write_txn(conn):
        handoff_error = _parent_handoff_start_error(conn, task_id)
        if handoff_error is not None:
            _record_parent_handoff_start_error(conn, task_id, handoff_error)
            return None
        # Single enforcement point: never ready -> running with an undone
        # parent, whichever writer set 'ready'. Demote to 'todo';
        # recompute_ready re-promotes when the parents finish.
        dependencies = evaluate_dependencies(conn, task_id)
        if not dependencies["satisfied"] and not _forced_promotion_active(conn, task_id):
            conn.execute(
                "UPDATE tasks SET status = 'todo' "
                "WHERE id = ? AND status = 'ready'", (task_id,),
            )
            _kb._append_event(
                conn,
                task_id,
                "claim_rejected",
                {"reason": "dependencies_unsatisfied", "blockers": dependencies["blockers"]},
            )
            return None
        # Close a leaked prior run so the CAS below doesn't strand it.
        _kb._reclaim_dangling_run(
            conn, task_id, statuses=("ready",), now=now, note="invariant recovery on re-claim",
        )
        run_id = _claim_and_open_run(
            conn, task_id, "ready", lock, expires, now,
            runtime_claim=runtime_claim, expected_task=expected_task,
        )
        if run_id is None:
            return None
        claimed = _kb.get_task(conn, task_id)
    _kb._fire_task_hook("kanban_task_claimed", claimed, task_id, run_id)
    return claimed

def claim_review_task(
    conn: sqlite3.Connection, task_id: str, *, ttl_seconds: Optional[int] = None,
    claimer: Optional[str] = None, runtime_identity: Any = None,
    worker_pid: Optional[int] = None, worker_start_time: Optional[int] = None,
    preparation_id: Optional[str] = None,
    expected_task: Any = None,
) -> Optional[_kb.Task]:
    """Atomic ``review -> running`` (None when lost). Parents are re-checked
    (one may have reopened meanwhile) and a NEW run tracks the reviewer
    separately from the implementer."""
    runtime_claim = _runtime_claim_metadata(
        runtime_identity, worker_pid, worker_start_time, preparation_id,
    )
    now = int(time.time())
    lock = claimer or _kb._claimer_id()
    expires = now + _kb._resolve_claim_ttl_seconds(ttl_seconds)
    with _kb.write_txn(conn):
        handoff_error = _parent_handoff_start_error(conn, task_id)
        if handoff_error is not None:
            _record_parent_handoff_start_error(conn, task_id, handoff_error)
            return None
        dependencies = evaluate_dependencies(conn, task_id)
        if not dependencies["satisfied"] and not _forced_promotion_active(conn, task_id):
            demoted = conn.execute(
                "UPDATE tasks SET status = 'todo' "
                "WHERE id = ? AND status = 'review' AND claim_lock IS NULL", (task_id,),
            )
            if demoted.rowcount == 1:
                _kb._append_event(
                    conn,
                    task_id,
                    "dependency_wait",
                    {
                        "reason": "dependencies_unsatisfied",
                        "blockers": dependencies["blockers"],
                        "source_status": "review",
                    },
                )
            return None
        run_id = _claim_and_open_run(
            conn, task_id, "review", lock, expires, now,
            event_extra={"source_status": "review"},
            runtime_claim=runtime_claim, expected_task=expected_task,
        )
        if run_id is None:
            return None
        return _kb.get_task(conn, task_id)

def release_stale_claims(conn: sqlite3.Connection, *, signal_fn=None) -> int:
    """Reclaim ``running`` tasks whose claim expired; returns the count reclaimed.

    A host-local worker that is still alive gets its claim *extended* instead
    (a slow model can sit longer than the TTL inside one tool-free call, so no
    heartbeat) — unless ``last_heartbeat_at`` is older than
    ``DEFAULT_CLAIM_HEARTBEAT_MAX_STALE_SECONDS`` (wedged; ``_touch_activity``
    keeps any genuinely active worker fresh). Safe to call often.

    Reclaiming a live worker mid-flight produces the spawn- then-immediately-reclaim loop seen on slow
    models that spend longer than ``DEFAULT_CLAIM_TTL_SECONDS`` inside a single tool-free LLM call (#23025):
    no tool calls means no ``kanban_heartbeat``, even though the subprocess is healthy.
    Backstop (#29747 gap 3): if the worker's PID is still alive but its ``last_heartbeat_at`` is stale by
    more than ``DEFAULT_CLAIM_HEARTBEAT_MAX_STALE_SECONDS`` (1h), the worker has been making no observable
    progress and we reclaim anyway — even if ``_pid_alive`` is still true. This catches the
    wedged-in-a-logic-loop case where the process is technically running but accomplishing nothing.
    ``_touch_activity`` (run_agent.py) bridges chunk-level liveness into ``last_heartbeat_at`` via #31752,
    so any genuinely active worker keeps its heartbeat fresh as a side effect of normal API traffic.
    ``enforce_max_runtime`` and ``detect_crashed_workers`` remain the upper bounds for genuinely wedged or
    dead workers.
    """
    now = int(time.time())
    reclaimed = 0
    host_prefix = _kb._host_prefix()
    stale = conn.execute(
        "SELECT id, claim_lock, worker_pid, claim_expires, last_heartbeat_at, "
        "       assignee "
        "FROM tasks "
        "WHERE status = 'running' AND claim_expires IS NOT NULL "
        "  AND claim_expires < ?", (now,),
    ).fetchall()
    for row in stale:
        host_local = (row["claim_lock"] or "").startswith(host_prefix)
        hb = row["last_heartbeat_at"]
        # Backstop: a heartbeat older than the max-stale threshold means no
        # observable progress — reclaim even if the PID is alive (logic loop).
        heartbeat_stale = hb is not None and (now - int(hb)) > _kb.DEFAULT_CLAIM_HEARTBEAT_MAX_STALE_SECONDS
        if host_local and row["worker_pid"] and _kb._pid_alive(row["worker_pid"]) and not heartbeat_stale:
            _kb._extend_live_stale_claim(conn, row, now)
            continue

        termination = _kb._terminate_reclaimed_worker(
            row["worker_pid"], row["claim_lock"], signal_fn=signal_fn,
        )
        # A live worker of ours must keep its claim (else a duplicate spawns beside it).
        if _kb._worker_survived_termination(termination):
            _kb._defer_reclaim_for_live_worker(
                conn, row["id"], row["claim_lock"], now, termination,
                reason="ttl_expired_worker_alive",
            )
            continue
        with _kb.write_txn(conn):
            retry_status = _kb._retry_status_for_run(conn, row["id"])
            cur = conn.execute(
                "UPDATE tasks SET status = ?, claim_lock = NULL, "
                "claim_expires = NULL, worker_pid = NULL "
                "WHERE id = ? AND status = 'running' AND claim_lock IS ? "
                "AND claim_expires IS NOT NULL AND claim_expires < ?",
                (retry_status, row["id"], row["claim_lock"], now),
            )
            if cur.rowcount != 1:
                continue
            run_id = _kb._record_reclaim(
                conn, row["id"], termination,
                error=f"stale_lock={row['claim_lock']}",
                payload={
                    "stale_lock": row["claim_lock"],
                    "worker_pid": _kb._opt_int(row["worker_pid"]),
                    "claim_expires": int(row["claim_expires"]),
                    "last_heartbeat_at": _kb._opt_int(row["last_heartbeat_at"]),
                    "now": now,
                    "host_local": host_local,
                    "heartbeat_stale": bool(heartbeat_stale),
                    "retry_status": retry_status,
                },
            )
            reclaimed += 1
        # Post-commit observer; every non-reclaim branch ``continue``d above.
        if _kb._kanban_observer_consumed("on_kanban_worker_stale_claim"):
            _kb._fire_kanban_lifecycle_hook(
                "on_kanban_worker_stale_claim", row["id"], board=_kb.get_current_board(),
                assignee=row["assignee"], run_id=run_id, worker_pid=_kb._opt_int(row["worker_pid"]),
                heartbeat_stale=bool(heartbeat_stale), retry_status=retry_status,
            )
    return reclaimed
def _landing_status_after_parents(conn: sqlite3.Connection, task_id: str) -> str:
    """``ready`` if every parent is terminal else ``todo`` — the re-gate shared by
    unblock/reopen so neither can spawn a child whose upstream is unfinished."""
    return _kb._lifecycle_ready_status(conn, task_id) if _parents_satisfied(conn, task_id) else "todo"

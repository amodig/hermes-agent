"""Lifecycle task-update planning and persistence helpers."""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Optional

from hermes_cli import kanban_db as _kb
from hermes_cli.kanban_db_lifecycle_claims import (
    _landing_status_after_parents,
    _parents_satisfied,
)
from hermes_cli.kanban_db_lifecycle_evidence import (
    _capture_acceptance,
    _emit_acceptance_changes,
    _implementation_routing,
)
from hermes_cli.kanban_lifecycle import (
    LifecycleContractError,
    decode_contract,
    encode_contract,
    safe_decode_contract,
    validate_edge,
)


@dataclass
class _UpdateRequest:
    expected_version: int
    reason_text: str
    transition_text: Optional[str]
    title: Any
    body: Any
    assignee: Any
    model: Any
    provider: Any
    goal_mode: Any
    lifecycle_contract: Any
    lifecycle_json: Any
    normalized_lifecycle: Any


@dataclass
class _UpdatePlan:
    current_version: int
    old_values: dict[str, Any]
    new_values: dict[str, Any]
    new_title: str
    new_body: Optional[str]
    new_assignee: Optional[str]
    new_model: Optional[str]
    new_provider: Optional[str]
    new_goal_mode: bool
    new_lifecycle: Optional[dict]
    new_status: str
    stored_lifecycle: Any
    candidate_run_id: Any
    goal_changed: bool
    lifecycle_changed: bool
    goal_reopened: bool
    changed_fields: list[str]
    goal_revision_id: int


def _validate_lifecycle_role_identity(
    conn: sqlite3.Connection,
    contract: Optional[dict],
    assignee: Optional[str],
    *,
    task_id: Optional[str] = None,
) -> None:
    # Late import preserves the facade/sibling import boundary and the facade
    # monkeypatch seam used by callers.
    from hermes_cli import kanban_db_lifecycle as lifecycle

    lifecycle._validate_lifecycle_role_identity(
        conn, contract, assignee, task_id=task_id,
    )


def _invalidate_descendants_for_parent_reopen(
    conn: sqlite3.Connection, task_id: str, *, author: str,
) -> dict[str, Any]:
    from hermes_cli import kanban_db_lifecycle as lifecycle

    return lifecycle.invalidate_descendants_for_parent_reopen(conn, task_id, author=author)


def _validate_update_request(
    conn: sqlite3.Connection,
    *,
    expected_version: int,
    reason: str,
    title: Any,
    body: Any,
    assignee: Any,
    model: Any,
    provider: Any,
    goal_mode: Any,
    lifecycle_contract: Any,
    transition: Optional[str],
) -> _UpdateRequest:
    if isinstance(expected_version, bool) or not isinstance(expected_version, int):
        raise ValueError("expected_version must be an integer")
    if expected_version < 1:
        raise ValueError("expected_version must be >= 1")
    reason_text = str(reason or "").strip()
    if not reason_text:
        raise ValueError("reason is required")
    transition_text = str(transition or "").strip() or None
    if transition_text not in (None, "triage_to_ready"):
        raise ValueError(
            f"unsupported transition {transition_text!r}; "
            "only 'triage_to_ready' is supported"
        )
    if goal_mode is not _kb._UPDATE_UNSET and not isinstance(goal_mode, bool):
        raise ValueError("goal_mode must be a boolean")
    _kb._ensure_goal_revision_schema(conn)

    lifecycle_json = _kb._UPDATE_UNSET
    normalized_lifecycle = _kb._UPDATE_UNSET
    if lifecycle_contract is not _kb._UPDATE_UNSET:
        if lifecycle_contract is None:
            raise LifecycleContractError(
                "lifecycle_contract cannot be cleared; bind an explicit contract"
            )
        lifecycle_json = encode_contract(lifecycle_contract, default_on_none=False)
        normalized_lifecycle = decode_contract(lifecycle_json)
        candidate_id = (
            normalized_lifecycle.get("candidate_task_id")
            if normalized_lifecycle
            and normalized_lifecycle.get("kind") in {"review", "validation"}
            else None
        )
        if candidate_id and not conn.execute(
            "SELECT 1 FROM tasks WHERE id = ?", (candidate_id,)
        ).fetchone():
            raise LifecycleContractError(f"candidate task {candidate_id} does not exist")
    return _UpdateRequest(
        expected_version=expected_version,
        reason_text=reason_text,
        transition_text=transition_text,
        title=title,
        body=body,
        assignee=assignee,
        model=model,
        provider=provider,
        goal_mode=goal_mode,
        lifecycle_contract=lifecycle_contract,
        lifecycle_json=lifecycle_json,
        normalized_lifecycle=normalized_lifecycle,
    )


def _ensure_goal_revision(
    conn: sqlite3.Connection, row: sqlite3.Row,
) -> tuple[int, sqlite3.Row]:
    goal_revision_id = _kb._row_get(row, "goal_revision_id")
    goal_row = (
        conn.execute(
            "SELECT * FROM task_goal_revisions WHERE id = ?", (goal_revision_id,)
        ).fetchone()
        if goal_revision_id is not None
        else None
    )
    if goal_row is None:
        goal_row = conn.execute(
            "SELECT * FROM task_goal_revisions "
            "WHERE task_id = ? ORDER BY version DESC, id DESC LIMIT 1",
            (row["id"],),
        ).fetchone()
    if goal_row is None:
        now = int(time.time())
        goal_author = str(_kb._row_get(row, "created_by") or "system").strip() or "system"
        goal_cur = conn.execute(
            """
            INSERT INTO task_goal_revisions
                (task_id, version, title, body, goal_mode, author,
                 created_at, reason, prior_version)
            VALUES (?, 1, ?, ?, ?, ?, ?, 'initial goal', NULL)
            """,
            (
                row["id"],
                row["title"],
                row["body"],
                int(bool(_kb._row_get(row, "goal_mode"))),
                goal_author,
                int(_kb._row_get(row, "created_at") or now),
            ),
        )
        goal_revision_id = int(goal_cur.lastrowid)
        goal_row = conn.execute(
            "SELECT * FROM task_goal_revisions WHERE id = ?", (goal_revision_id,)
        ).fetchone()
        conn.execute(
            "UPDATE tasks SET goal_revision_id = ? WHERE id = ?",
            (goal_revision_id, row["id"]),
        )
    return int(goal_revision_id), goal_row


def _build_update_plan(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    request: _UpdateRequest,
    goal_revision_id: int,
) -> _UpdatePlan:
    current_version = int(_kb._row_get(row, "version", 1) or 1)
    old_title = row["title"]
    old_body = row["body"]
    old_assignee = row["assignee"]
    old_model = _kb._row_get(row, "model_override")
    old_provider = _kb._row_get(row, "provider_override")
    old_goal_mode = bool(_kb._row_get(row, "goal_mode"))
    try:
        old_lifecycle = decode_contract(_kb._row_get(row, "lifecycle_contract"))
    except LifecycleContractError:
        old_lifecycle = None

    new_title = old_title
    new_body = old_body
    new_assignee = old_assignee
    new_model = old_model
    new_provider = old_provider
    new_goal_mode = old_goal_mode
    new_lifecycle = old_lifecycle
    if request.title is not _kb._UPDATE_UNSET:
        if request.title is None or not str(request.title).strip():
            raise ValueError("title cannot be blank")
        new_title = str(request.title).strip()
    if request.body is not _kb._UPDATE_UNSET:
        new_body = None if request.body is None else str(request.body)
    if request.assignee is not _kb._UPDATE_UNSET:
        new_assignee = _kb._canonical_assignee(
            None if request.assignee is None or not str(request.assignee).strip()
            else str(request.assignee)
        )
    model_supplied = request.model is not _kb._UPDATE_UNSET
    provider_supplied = request.provider is not _kb._UPDATE_UNSET
    if model_supplied:
        new_model = None if request.model is None else str(request.model).strip() or None
        if provider_supplied:
            new_provider = None if request.provider is None else str(request.provider).strip() or None
        if new_model is None:
            new_provider = None
    elif provider_supplied:
        new_provider = None if request.provider is None else str(request.provider).strip() or None
    if new_provider and not new_model:
        raise ValueError("provider requires a model")
    if request.goal_mode is not _kb._UPDATE_UNSET:
        new_goal_mode = bool(request.goal_mode)
    if request.lifecycle_contract is not _kb._UPDATE_UNSET:
        new_lifecycle = request.normalized_lifecycle

    if request.assignee is not _kb._UPDATE_UNSET or request.lifecycle_contract is not _kb._UPDATE_UNSET:
        _validate_lifecycle_role_identity(
            conn, new_lifecycle, new_assignee, task_id=row["id"],
        )
    lifecycle_changed = (
        request.lifecycle_contract is not _kb._UPDATE_UNSET and new_lifecycle != old_lifecycle
    )
    if lifecycle_changed and old_lifecycle is not None:
        raise LifecycleContractError(
            "lifecycle_contract is immutable after classification; repair only historical NULL contracts"
        )
    if lifecycle_changed:
        conn.execute(
            "UPDATE tasks SET lifecycle_contract = ? WHERE id = ?",
            (encode_contract(new_lifecycle, default_on_none=False), row["id"]),
        )
        for edge in conn.execute(
            "SELECT parent_id, child_id, requirement FROM task_links "
            "WHERE parent_id = ? OR child_id = ? ORDER BY parent_id, child_id",
            (row["id"], row["id"]),
        ).fetchall():
            requirement = edge["requirement"]
            if not requirement:
                continue
            endpoints = conn.execute(
                "SELECT lifecycle_contract FROM tasks WHERE id IN (?, ?)",
                (edge["parent_id"], edge["child_id"]),
            ).fetchall()
            if any(safe_decode_contract(endpoint["lifecycle_contract"]) is None for endpoint in endpoints):
                continue
            validate_edge(conn, edge["parent_id"], edge["child_id"], requirement)

    new_status = row["status"]
    if request.transition_text:
        if row["status"] != "triage":
            raise ValueError(
                "transition 'triage_to_ready' requires a task in "
                f"status 'triage' (current status {row['status']!r})"
            )
        new_status = _kb._lifecycle_ready_status(conn, row["id"]) if _parents_satisfied(conn, row["id"]) else "todo"
    elif lifecycle_changed:
        if row["status"] in {"blocked", "scheduled", "triage"}:
            new_status = row["status"]
        elif new_lifecycle and new_lifecycle.get("kind") != "general":
            new_status = _kb._lifecycle_ready_status(conn, row["id"]) if _parents_satisfied(conn, row["id"]) else "todo"
        elif new_lifecycle and new_lifecycle.get("kind") == "general":
            new_status = "done" if row["status"] == "done" else "ready"

    goal_changed = (
        old_title != new_title
        or old_body != new_body
        or old_goal_mode != new_goal_mode
    )
    if (
        goal_changed
        and row["status"] in {"done", "archived"}
        and new_lifecycle
        and new_lifecycle.get("kind") in {"review", "validation"}
    ):
        raise LifecycleContractError(
            "completed typed role cards cannot be edited; repair the lifecycle graph first"
        )
    goal_reopened = bool(
        goal_changed
        and new_lifecycle
        and new_lifecycle.get("kind") == "code"
        and (
            _kb._row_get(row, "candidate_run_id") is not None
            or row["status"] in {"done", "review"}
        )
    )
    if goal_reopened:
        new_status = _landing_status_after_parents(conn, row["id"])
        if new_lifecycle.get("review_mode") == "same_card" and row["status"] in {"review", "done"}:
            routing = _implementation_routing(conn, row["id"])
            implementation_assignee = routing.get("implementer")
            if not implementation_assignee:
                review_event = _kb._latest_event(conn, row["id"], "review_requested")
                implementation_assignee = _kb._json_dict(
                    _kb._row_get(review_event, "payload")
                ).get("implementer")
            if implementation_assignee:
                new_assignee = _kb._canonical_assignee(str(implementation_assignee))

    old_values = {
        "title": old_title,
        "body": old_body,
        "assignee": old_assignee,
        "model": old_model,
        "provider": old_provider,
        "goal_mode": old_goal_mode,
        "lifecycle_contract": old_lifecycle,
        "status": row["status"],
        "version": current_version,
    }
    new_values = {
        "title": new_title,
        "body": new_body,
        "assignee": new_assignee,
        "model": new_model,
        "provider": new_provider,
        "goal_mode": new_goal_mode,
        "lifecycle_contract": new_lifecycle,
        "status": new_status,
        "version": current_version + 1,
    }
    changed_fields = [
        key for key in (
            "title", "body", "assignee", "model", "provider", "goal_mode",
            "lifecycle_contract", "status",
        )
        if old_values[key] != new_values[key]
    ]
    return _UpdatePlan(
        current_version=current_version,
        old_values=old_values,
        new_values=new_values,
        new_title=new_title,
        new_body=new_body,
        new_assignee=new_assignee,
        new_model=new_model,
        new_provider=new_provider,
        new_goal_mode=new_goal_mode,
        new_lifecycle=new_lifecycle,
        new_status=new_status,
        stored_lifecycle=(
            request.lifecycle_json
            if request.lifecycle_contract is not _kb._UPDATE_UNSET
            else _kb._row_get(row, "lifecycle_contract")
        ),
        candidate_run_id=(
            None
            if lifecycle_changed or goal_reopened
            else _kb._row_get(row, "candidate_run_id")
        ),
        goal_changed=goal_changed,
        lifecycle_changed=lifecycle_changed,
        goal_reopened=goal_reopened,
        changed_fields=changed_fields,
        goal_revision_id=goal_revision_id,
    )


def _persist_update(
    conn: sqlite3.Connection,
    task_id: str,
    request: _UpdateRequest,
    plan: _UpdatePlan,
    *,
    actor: str,
    goal_row: sqlite3.Row,
) -> tuple[list[str], list[dict[str, Any]], list[tuple[Optional[int], Optional[str]]]]:
    changed_fields = list(plan.changed_fields)
    goal_invalidated: list[dict[str, Any]] = []
    goal_terminations: list[tuple[Optional[int], Optional[str]]] = []
    goal_revision = None
    goal_revision_id = plan.goal_revision_id
    if plan.goal_changed:
        prior_goal_version = int(goal_row["version"])
        goal_cur = conn.execute(
            """
            INSERT INTO task_goal_revisions
                (task_id, version, title, body, goal_mode, author,
                 created_at, reason, prior_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                task_id,
                prior_goal_version + 1,
                plan.new_title,
                plan.new_body,
                int(plan.new_goal_mode),
                actor,
                int(time.time()),
                request.reason_text,
                prior_goal_version,
            ),
        )
        goal_revision_id = int(goal_cur.lastrowid)
        changed_fields.append("goal_revision")
    if plan.goal_reopened:
        invalidation = _invalidate_descendants_for_parent_reopen(
            conn, task_id, author=actor,
        )
        goal_invalidated.extend(invalidation["invalidated"])
        goal_terminations.extend(invalidation["terminations"])

    cur = conn.execute(
        """
        UPDATE tasks SET
            version = ?, title = ?, body = ?, assignee = ?,
            model_override = ?, provider_override = ?, goal_mode = ?,
            status = ?, goal_revision_id = ?, lifecycle_contract = ?,
            candidate_run_id = ?
        WHERE id = ? AND version = ?
        """,
        (
            plan.current_version + 1,
            plan.new_title,
            plan.new_body,
            plan.new_assignee,
            plan.new_model,
            plan.new_provider,
            int(plan.new_goal_mode),
            plan.new_status,
            goal_revision_id,
            plan.stored_lifecycle,
            plan.candidate_run_id,
            task_id,
            request.expected_version,
        ),
    )
    if cur.rowcount != 1:
        raise _kb.TaskUpdateConflict(
            f"task {task_id} update conflict: version changed while updating"
        )
    if plan.goal_reopened:
        conn.execute("UPDATE tasks SET completed_at = NULL WHERE id = ?", (task_id,))
        changed_fields.append("completed_at")
    if plan.goal_changed:
        goal_revision = _kb.get_effective_goal(conn, task_id)
    payload = {
        "actor": actor,
        "author": actor,
        "reason": request.reason_text,
        "expected_version": request.expected_version,
        "old": plan.old_values,
        "new": plan.new_values,
        "old_values": plan.old_values,
        "new_values": plan.new_values,
        "changed_fields": changed_fields,
        "transition": request.transition_text,
        "goal_revision": goal_revision,
    }
    _kb._append_event(
        conn,
        task_id,
        "lifecycle_bound" if plan.lifecycle_changed else "goal_revised" if plan.goal_changed else "updated",
        payload,
    )
    return changed_fields, goal_invalidated, goal_terminations


def update_task(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    expected_version: int,
    reason: str,
    title: Any = _kb._UPDATE_UNSET,
    body: Any = _kb._UPDATE_UNSET,
    assignee: Any = _kb._UPDATE_UNSET,
    model: Any = _kb._UPDATE_UNSET,
    provider: Any = _kb._UPDATE_UNSET,
    goal_mode: Any = _kb._UPDATE_UNSET,
    lifecycle_contract: Any = _kb._UPDATE_UNSET,
    transition: Optional[str] = None,
    author: Optional[str] = None,
) -> bool:
    """Atomically revise and optionally requeue one existing task."""
    request = _validate_update_request(
        conn,
        expected_version=expected_version,
        reason=reason,
        title=title,
        body=body,
        assignee=assignee,
        model=model,
        provider=provider,
        goal_mode=goal_mode,
        lifecycle_contract=lifecycle_contract,
        transition=transition,
    )
    actor = _kb._update_actor(author)
    acceptance_before = _capture_acceptance(conn, task_id)
    with _kb.write_txn(conn):
        row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        if row is None:
            return False
        current_version = int(_kb._row_get(row, "version", 1) or 1)
        if current_version != request.expected_version:
            raise _kb.TaskUpdateConflict(
                f"task {task_id} update conflict: expected version "
                f"{request.expected_version}, current version {current_version}"
            )
        if row["status"] == "archived":
            raise RuntimeError(f"cannot update archived task {task_id}")
        if (
            row["status"] == "running"
            or _kb._row_get(row, "claim_lock") is not None
            or _kb._row_get(row, "current_run_id") is not None
        ):
            raise _kb.TaskUpdateConflict(
                f"cannot update task {task_id}: currently claimed by a worker; "
                "reclaim it before revising or requeuing"
            )
        goal_revision_id, goal_row = _ensure_goal_revision(conn, row)
        plan = _build_update_plan(conn, row, request, goal_revision_id)
        changed_fields, goal_invalidated, goal_terminations = _persist_update(
            conn, task_id, request, plan, actor=actor, goal_row=goal_row,
        )
    for pid, claim_lock in goal_terminations:
        _kb._terminate_reclaimed_worker(pid, claim_lock)
    _kb.notify_task_updated(conn, task_id, changed_fields or ["version"])
    for entry in goal_invalidated:
        _kb.notify_task_updated(
            conn, entry["id"], ("status", "version", "completed_at", "candidate_run_id"),
        )
    _emit_acceptance_changes(conn, acceptance_before, source_task_id=task_id)
    return True

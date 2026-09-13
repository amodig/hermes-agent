"""Lifecycle-aware Kanban mutations and evidence writes.

This sibling owns the typed lifecycle write paths; the facade imports its public
entry points at the end of module initialization.
"""
from __future__ import annotations

import contextlib
import json
import os
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterable, Optional

from hermes_cli import kanban_db as _kb
from hermes_cli.kanban_lifecycle import (
    LifecycleContractError,
    LifecycleEvidenceError,
    _latest_head,
    decode_contract,
    encode_contract,
    evaluate_dependencies,
    get_lifecycle_state,
    infer_edge_requirement,
    lifecycle_metadata,
    safe_decode_contract,
    validate_edge,
)

def _ensure_lifecycle_schema(conn: sqlite3.Connection) -> None:
    """Add lifecycle columns without classifying historical rows."""
    tables = {
        row["name"]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    if "tasks" in tables:
        cols = {row["name"] for row in conn.execute("PRAGMA table_info(tasks)")}
        if "lifecycle_contract" not in cols:
            _kb._add_column_if_missing(conn, "tasks", "lifecycle_contract", "lifecycle_contract TEXT")
        if "candidate_run_id" not in cols:
            _kb._add_column_if_missing(conn, "tasks", "candidate_run_id", "candidate_run_id INTEGER")
    if "task_links" in tables:
        cols = {row["name"] for row in conn.execute("PRAGMA table_info(task_links)")}
        if "requirement" not in cols:
            _kb._add_column_if_missing(conn, "task_links", "requirement", "requirement TEXT")
    if "tasks" in tables:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_candidate_run ON tasks(candidate_run_id)")
    if "task_links" in tables:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_links_requirement ON task_links(requirement)")

def _validate_lifecycle_role_identity(
    conn: sqlite3.Connection,
    contract: Optional[dict],
    assignee: Optional[str],
    *,
    task_id: Optional[str] = None,
    phase: Optional[str] = None,
) -> None:
    """Keep implementation, review, and validation identities independent."""
    if not contract:
        return
    actor = _kb._canonical_assignee(assignee) if assignee else None
    kind = contract.get("kind")
    task_status = None
    if task_id is not None:
        task_row = conn.execute("SELECT status FROM tasks WHERE id = ?", (task_id,)).fetchone()
        task_status = _kb._row_get(task_row, "status")
    if kind == "code":
        if (
            actor
            and contract.get("review_mode") == "same_card"
            and phase != "implementation"
            and task_status in {"review", "done"}
        ):
            if actor != _kb._canonical_assignee(contract.get("reviewer")):
                raise LifecycleContractError(
                    "same-card review assignee must match the declared reviewer"
                )
            return
        if not actor:
            return
        if actor == _kb._canonical_assignee(contract.get("reviewer")):
            raise LifecycleContractError("reviewer must differ from the implementation assignee")
        if task_id is None:
            return
        validation_rows = conn.execute(
            """
            WITH RECURSIVE descendants(id) AS (
                SELECT child_id FROM task_links WHERE parent_id = ?
                UNION
                SELECT l.child_id
                FROM task_links l
                JOIN descendants d ON d.id = l.parent_id
            )
            SELECT t.assignee, t.lifecycle_contract
            FROM tasks t
            JOIN descendants d ON d.id = t.id
            """,
            (task_id,),
        ).fetchall()
        for row in validation_rows:
            child_contract = safe_decode_contract(row["lifecycle_contract"])
            if (
                child_contract
                and child_contract.get("kind") == "validation"
                and child_contract.get("candidate_task_id") == task_id
                and actor == _kb._canonical_assignee(row["assignee"])
            ):
                raise LifecycleContractError(
                    "validator must differ from the implementation and reviewer"
                )
        return
    if kind not in {"review", "validation"}:
        return

    candidate_id = contract.get("candidate_task_id")
    candidate = conn.execute(
        "SELECT assignee, lifecycle_contract FROM tasks WHERE id = ?",
        (candidate_id,),
    ).fetchone()
    candidate_contract = safe_decode_contract(
        _kb._row_get(candidate, "lifecycle_contract")
    ) if candidate is not None else None
    if candidate is None or not candidate_contract or candidate_contract.get("kind") != "code":
        raise LifecycleContractError("typed role card requires a classified code candidate")
    candidate_assignee = _kb._canonical_assignee(candidate["assignee"])
    declared_reviewer = _kb._canonical_assignee(candidate_contract.get("reviewer"))

    if kind == "review":
        if not actor:
            raise LifecycleContractError(
                "review cards require an assignee matching the declared reviewer"
            )
        if actor != declared_reviewer:
            raise LifecycleContractError(
                "reviewer must match the implementation's declared reviewer"
            )
        return

    if not actor:
        return
    forbidden = {candidate_assignee, declared_reviewer}
    if candidate_contract.get("review_mode") == "same_card":
        original_implementer = _kb._canonical_assignee(
            _implementation_routing(conn, candidate_id).get("implementer")
        )
        if original_implementer:
            forbidden.add(original_implementer)
    if actor in forbidden:
        raise LifecycleContractError(
            "validator must differ from the implementation and reviewer"
        )
    review_rows = conn.execute(
        "SELECT t.assignee, t.lifecycle_contract "
        "FROM task_links l JOIN tasks t ON t.id = l.child_id "
        "WHERE l.parent_id = ? ORDER BY t.id",
        (candidate_id,),
    ).fetchall()
    for row in review_rows:
        review_contract = safe_decode_contract(row["lifecycle_contract"])
        if (
            review_contract
            and review_contract.get("kind") == "review"
            and review_contract.get("candidate_task_id") == candidate_id
            and actor == _kb._canonical_assignee(row["assignee"])
        ):
            raise LifecycleContractError("validator must differ from the reviewer")

def create_task(
    conn: sqlite3.Connection, *, title: str, body: Optional[str] = None,
    assignee: Optional[str] = None, created_by: Optional[str] = None,
    workspace_kind: str = "scratch", workspace_path: Optional[str] = None,
    branch_name: Optional[str] = None, tenant: Optional[str] = None, priority: int = 0,
    parents: Iterable[str] = (), triage: bool = False, idempotency_key: Optional[str] = None,
    max_runtime_seconds: Optional[int] = None, skills: Optional[Iterable[str]] = None,
    max_retries: Optional[int] = None, model_override: Optional[str] = None,
    provider_override: Optional[str] = None, reasoning_effort: Optional[str] = None,
    goal_mode: bool = False, goal_max_turns: Optional[int] = None, initial_status: str = "running",
    session_id: Optional[str] = None, board: Optional[str] = None, project_id: Optional[str] = None,
    project_source_task_id: Optional[str] = None,
    lifecycle_contract: Optional[dict] = None,
) -> str:
    """Create a task (optionally under ``parents``); returns its id.

    New tasks always carry an explicit general contract when no contract is
    supplied.  Historical rows with NULL contracts are never backfilled.
    """
    _kb._ensure_goal_revision_schema(conn)
    lifecycle_json = encode_contract(lifecycle_contract, default_on_none=True)
    normalized_lifecycle = decode_contract(lifecycle_json)
    candidate_id = (
        normalized_lifecycle.get("candidate_task_id")
        if normalized_lifecycle and normalized_lifecycle.get("kind") in {"review", "validation"}
        else None
    )
    candidate_contract = None
    if candidate_id:
        candidate_row = conn.execute(
            "SELECT lifecycle_contract FROM tasks WHERE id = ?", (candidate_id,),
        ).fetchone()
        if not candidate_row:
            raise LifecycleContractError(f"candidate task {candidate_id} does not exist")
        candidate_contract = safe_decode_contract(candidate_row["lifecycle_contract"])
    assignee = _kb._canonical_assignee(assignee)
    _validate_lifecycle_role_identity(conn, normalized_lifecycle, assignee)
    model_override, provider_override = _kb._validate_model_override(model_override, provider_override)
    reasoning_effort = _kb.normalize_reasoning_effort(reasoning_effort)
    if not title or not title.strip():
        raise ValueError("title is required")
    if initial_status not in _kb.VALID_INITIAL_STATUSES:
        raise ValueError(f"initial_status must be one of {sorted(_kb.VALID_INITIAL_STATUSES)}")
    if workspace_kind not in _kb.VALID_WORKSPACE_KINDS:
        raise ValueError(
            f"workspace_kind must be one of {sorted(_kb.VALID_WORKSPACE_KINDS)}, "
            f"got {workspace_kind!r}"
        )
    if branch_name is not None:
        branch_name = str(branch_name).strip() or None
    if branch_name and workspace_kind != "worktree":
        raise ValueError("branch_name is only valid for worktree workspaces")

    # A project-scoped board anchors every new task to its project's repo
    # (deterministic worktree + branch) without each surface repeating it.
    if project_id is None:
        try:
            project_id = (_kb._board_meta_for(board).get("project_id") or "").strip() or None
        except Exception:
            pass

    project_id, project_obj, project_repo, workspace_kind = _kb._resolve_project_link(
        conn, project_id, project_source_task_id, workspace_kind, workspace_path
    )
    parents = tuple(p for p in parents if p)
    role_missing_candidate_edge = False
    if candidate_id:
        role_kind = normalized_lifecycle["kind"]
        candidate_is_same_card = (
            candidate_contract is not None
            and candidate_contract.get("kind") == "code"
            and candidate_contract.get("review_mode") == "same_card"
        )
        if role_kind == "review" or (role_kind == "validation" and candidate_is_same_card):
            role_missing_candidate_edge = candidate_id not in parents
        elif role_kind == "validation":
            role_missing_candidate_edge = True
            for parent_id in parents:
                parent_row = conn.execute(
                    "SELECT lifecycle_contract FROM tasks WHERE id = ?", (parent_id,),
                ).fetchone()
                parent_contract = (
                    safe_decode_contract(parent_row["lifecycle_contract"])
                    if parent_row else None
                )
                if (
                    parent_contract
                    and parent_contract.get("kind") == "review"
                    and parent_contract.get("candidate_task_id") == candidate_id
                ):
                    role_missing_candidate_edge = False
                    break
    skills_list = _kb._normalize_task_skills(skills)

    # Idempotency check BEFORE the write txn (no lock held); a concurrent-create
    # race may insert twice, the next lookup stabilises on the newest.
    if idempotency_key:
        row = conn.execute(
            "SELECT id FROM tasks WHERE idempotency_key = ? "
            "AND status != 'archived' "
            "ORDER BY created_at DESC LIMIT 1", (idempotency_key,),
        ).fetchone()
        if row:
            return row["id"]

    now = int(time.time())

    # Only persistent kinds inherit the board ``default_workdir``: a scratch
    # task inheriting it would point cleanup at the user's source tree.
    if workspace_path is None and project_repo is None and workspace_kind in {"dir", "worktree"}:
        board_default = _kb._board_meta_for(board).get("default_workdir")
        if board_default:
            workspace_path = str(board_default)

    # Retry once on the extremely unlikely id collision.
    for attempt in range(2):
        task_id = _kb._new_task_id()
        try:
            # allow_nested: graph builders compose create_task under one outer
            # commit so the dispatcher never sees a half-built graph.
            with _kb.write_txn(conn, allow_nested=True):
                task_status = _kb._initial_task_status(conn, parents, initial_status, triage)
                if task_status == "ready" and normalized_lifecycle and normalized_lifecycle.get("kind") == "review":
                    task_status = "review"
                if (
                    role_missing_candidate_edge
                    and task_status not in {"blocked", "triage"}
                ):
                    task_status = "todo"
                # Project worktree: fresh dir under the repo + deterministic
                # branch, instead of the random ``wt/<id>`` worker fallback.
                if project_obj is not None and workspace_kind == "worktree":
                    if project_repo and not workspace_path:
                        workspace_path = os.path.join(project_repo, ".worktrees", task_id)
                    if not branch_name:
                        branch_name = _kb._project_branch_name(project_obj, task_id, title)

                conn.execute(
                    """
                    INSERT INTO tasks (
                        id, title, body, assignee, status, priority,
                        created_by, created_at, workspace_kind, workspace_path,
                        branch_name, project_id, tenant, idempotency_key,
                        max_runtime_seconds,
                        skills, max_retries, model_override, provider_override,
                        reasoning_effort,
                        goal_mode, goal_max_turns, session_id,
                        lifecycle_contract, candidate_run_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        task_id, title.strip(), body, assignee, task_status, priority,
                        created_by, now, workspace_kind, workspace_path,
                        branch_name, project_id, tenant, idempotency_key,
                        _kb._opt_int(max_runtime_seconds),
                        json.dumps(skills_list) if skills_list is not None else None,
                        _kb._opt_int(max_retries), model_override, provider_override, reasoning_effort,
                        1 if goal_mode else 0, _kb._opt_int(goal_max_turns), session_id,
                        lifecycle_json, None,
                    ),
                )
                goal_author = str(created_by or os.environ.get("HERMES_PROFILE") or "system").strip() or "system"
                goal_revision_cur = conn.execute(
                    """
                    INSERT INTO task_goal_revisions
                        (task_id, version, title, body, goal_mode, author,
                         created_at, reason, prior_version)
                    VALUES (?, 1, ?, ?, ?, ?, ?, 'initial goal', NULL)
                    """,
                    (
                        task_id,
                        title.strip(),
                        body,
                        int(bool(goal_mode)),
                        goal_author,
                        now,
                    ),
                )
                goal_revision_id = int(goal_revision_cur.lastrowid)
                conn.execute(
                    "UPDATE tasks SET goal_revision_id = ? WHERE id = ?",
                    (goal_revision_id, task_id),
                )
                goal_revision = {
                    "id": goal_revision_id,
                    "task_id": task_id,
                    "version": 1,
                    "title": title.strip(),
                    "body": body,
                    "goal_mode": bool(goal_mode),
                    "author": goal_author,
                    "timestamp": now,
                    "reason": "initial goal",
                    "prior_version": None,
                }
                for pid in parents:
                    requirement = infer_edge_requirement(conn, pid, task_id)
                    _kb._link(conn, pid, task_id, requirement=requirement)
                if task_status == "ready" and not _parents_satisfied(conn, task_id):
                    task_status = "todo"
                    conn.execute(
                        "UPDATE tasks SET status = 'todo' WHERE id = ?", (task_id,)
                    )
                _kb._append_event(
                    conn,
                    task_id,
                    "created",
                    {
                        "assignee": assignee,
                        "status": task_status,
                        "parents": list(parents),
                        "tenant": tenant,
                        "workspace_kind": workspace_kind,
                        "workspace_path": workspace_path,
                        "lifecycle_contract": normalized_lifecycle,
                        "skills": list(skills_list) if skills_list else None,
                        "goal_mode": bool(goal_mode) or None,
                        "model_override": model_override,
                        "provider_override": provider_override,
                        "version": 1,
                        "goal_revision": goal_revision,
                    },
                )
                # ACK-edge: the originating channel hears a child BLOCK, not just the fan-in.
                _kb._inherit_notify_subs(conn, task_id, parents, created_at=now)
            return task_id
        except sqlite3.IntegrityError:
            if attempt == 1:
                raise
    raise RuntimeError("unreachable")

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
            raise LifecycleContractError("lifecycle_contract cannot be cleared; bind an explicit contract")
        lifecycle_json = encode_contract(lifecycle_contract, default_on_none=False)
        normalized_lifecycle = decode_contract(lifecycle_json)
        candidate_id = (
            normalized_lifecycle.get("candidate_task_id")
            if normalized_lifecycle and normalized_lifecycle.get("kind") in {"review", "validation"}
            else None
        )
        if candidate_id and not conn.execute(
            "SELECT 1 FROM tasks WHERE id = ?", (candidate_id,)
        ).fetchone():
            raise LifecycleContractError(f"candidate task {candidate_id} does not exist")

    changed_fields: list[str] = []
    actor = _kb._update_actor(author)
    acceptance_before = _capture_acceptance(conn, task_id)
    goal_invalidated: list[dict[str, Any]] = []
    goal_terminations: list[tuple[Optional[int], Optional[str]]] = []
    with _kb.write_txn(conn):
        row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        if row is None:
            return False
        current_version = int(_kb._row_get(row, "version", 1) or 1)
        if current_version != expected_version:
            raise _kb.TaskUpdateConflict(
                f"task {task_id} update conflict: expected version "
                f"{expected_version}, current version {current_version}"
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

        goal_revision_id = _kb._row_get(row, "goal_revision_id")
        goal_row = (
            conn.execute(
                "SELECT * FROM task_goal_revisions WHERE id = ?",
                (goal_revision_id,),
            ).fetchone()
            if goal_revision_id is not None
            else None
        )
        if goal_row is None:
            goal_row = conn.execute(
                "SELECT * FROM task_goal_revisions "
                "WHERE task_id = ? ORDER BY version DESC, id DESC LIMIT 1",
                (task_id,),
            ).fetchone()
        if goal_row is None:
            now = int(time.time())
            goal_author = (
                str(_kb._row_get(row, "created_by") or "system").strip() or "system"
            )
            goal_cur = conn.execute(
                """
                INSERT INTO task_goal_revisions
                    (task_id, version, title, body, goal_mode, author,
                     created_at, reason, prior_version)
                VALUES (?, 1, ?, ?, ?, ?, ?, 'initial goal', NULL)
                """,
                (
                    task_id,
                    row["title"],
                    row["body"],
                    int(bool(_kb._row_get(row, "goal_mode"))),
                    goal_author,
                    int(_kb._row_get(row, "created_at") or now),
                ),
            )
            goal_revision_id = int(goal_cur.lastrowid)
            goal_row = conn.execute(
                "SELECT * FROM task_goal_revisions WHERE id = ?",
                (goal_revision_id,),
            ).fetchone()
            conn.execute(
                "UPDATE tasks SET goal_revision_id = ? WHERE id = ?",
                (goal_revision_id, task_id),
            )
        else:
            goal_revision_id = int(goal_row["id"])

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

        if title is not _kb._UPDATE_UNSET:
            if title is None or not str(title).strip():
                raise ValueError("title cannot be blank")
            new_title = str(title).strip()
        if body is not _kb._UPDATE_UNSET:
            new_body = None if body is None else str(body)
        if assignee is not _kb._UPDATE_UNSET:
            new_assignee = _kb._canonical_assignee(
                None if assignee is None or not str(assignee).strip() else str(assignee)
            )
        model_supplied = model is not _kb._UPDATE_UNSET
        provider_supplied = provider is not _kb._UPDATE_UNSET
        if model_supplied:
            new_model = None if model is None else str(model).strip() or None
            if provider_supplied:
                new_provider = (
                    None if provider is None else str(provider).strip() or None
                )
            if new_model is None:
                new_provider = None
        elif provider_supplied:
            new_provider = None if provider is None else str(provider).strip() or None
        if new_provider and not new_model:
            raise ValueError("provider requires a model")
        if goal_mode is not _kb._UPDATE_UNSET:
            new_goal_mode = bool(goal_mode)
        if lifecycle_contract is not _kb._UPDATE_UNSET:
            new_lifecycle = normalized_lifecycle

        if assignee is not _kb._UPDATE_UNSET or lifecycle_contract is not _kb._UPDATE_UNSET:
            _validate_lifecycle_role_identity(
                conn, new_lifecycle, new_assignee, task_id=task_id,
            )
        lifecycle_changed = (
            lifecycle_contract is not _kb._UPDATE_UNSET and new_lifecycle != old_lifecycle
        )
        if lifecycle_changed and old_lifecycle is not None:
            raise LifecycleContractError(
                "lifecycle_contract is immutable after classification; repair only historical NULL contracts"
            )
        if lifecycle_changed:
            # Let edge and ready-lane validation observe the prospective
            # contract; rollback restores the historical NULL on failure.
            conn.execute(
                "UPDATE tasks SET lifecycle_contract = ? WHERE id = ?",
                (encode_contract(new_lifecycle, default_on_none=False), task_id),
            )
            for edge in conn.execute(
                "SELECT parent_id, child_id, requirement FROM task_links "
                "WHERE parent_id = ? OR child_id = ? ORDER BY parent_id, child_id",
                (task_id, task_id),
            ).fetchall():
                requirement = edge["requirement"]
                if not requirement:
                    # Historical linked graphs may need both endpoint contracts
                    # and edge requirements repaired in separate transactions.
                    continue
                endpoints = conn.execute(
                    "SELECT lifecycle_contract FROM tasks WHERE id IN (?, ?)",
                    (edge["parent_id"], edge["child_id"]),
                ).fetchall()
                if any(safe_decode_contract(row["lifecycle_contract"]) is None for row in endpoints):
                    continue
                validate_edge(conn, edge["parent_id"], edge["child_id"], requirement)
        new_status = row["status"]
        if transition_text:
            if row["status"] != "triage":
                raise ValueError(
                    "transition 'triage_to_ready' requires a task in "
                    f"status 'triage' (current status {row['status']!r})"
                )
            new_status = _kb._lifecycle_ready_status(conn, task_id) if _parents_satisfied(conn, task_id) else "todo"
        elif lifecycle_changed:
            if new_lifecycle and new_lifecycle.get("kind") != "general":
                new_status = _kb._lifecycle_ready_status(conn, task_id) if _parents_satisfied(conn, task_id) else "todo"
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
            new_status = _landing_status_after_parents(conn, task_id)
            if (
                new_lifecycle.get("review_mode") == "same_card"
                and row["status"] in {"review", "done"}
            ):
                routing = _implementation_routing(conn, task_id)
                implementation_assignee = routing.get("implementer")
                if not implementation_assignee:
                    review_event = _kb._latest_event(conn, task_id, "review_requested")
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
        for key in (
            "title",
            "body",
            "assignee",
            "model",
            "provider",
            "goal_mode",
            "lifecycle_contract",
            "status",
        ):
            if old_values[key] != new_values[key]:
                changed_fields.append(key)

        goal_revision = None
        if goal_changed:
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
                    new_title,
                    new_body,
                    int(new_goal_mode),
                    actor,
                    int(time.time()),
                    reason_text,
                    prior_goal_version,
                ),
            )
            goal_revision_id = int(goal_cur.lastrowid)
            changed_fields.append("goal_revision")
        if goal_reopened:
            invalidation = invalidate_descendants_for_parent_reopen(
                conn, task_id, author=actor,
            )
            goal_invalidated.extend(invalidation["invalidated"])
            goal_terminations.extend(invalidation["terminations"])

        candidate_run_id = (
            None
            if lifecycle_changed or goal_reopened
            else _kb._row_get(row, "candidate_run_id")
        )

        stored_lifecycle = (
            lifecycle_json
            if lifecycle_contract is not _kb._UPDATE_UNSET
            else _kb._row_get(row, "lifecycle_contract")
        )
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
                current_version + 1,
                new_title,
                new_body,
                new_assignee,
                new_model,
                new_provider,
                int(new_goal_mode),
                new_status,
                goal_revision_id,
                stored_lifecycle,
                candidate_run_id,
                task_id,
                expected_version,
            ),
        )
        if cur.rowcount != 1:
            raise _kb.TaskUpdateConflict(
                f"task {task_id} update conflict: version changed while updating"
            )
        if goal_reopened:
            conn.execute("UPDATE tasks SET completed_at = NULL WHERE id = ?", (task_id,))
            changed_fields.append("completed_at")

        if goal_changed:
            goal_revision = _kb.get_effective_goal(conn, task_id)
        payload = {
            "actor": actor,
            "author": actor,
            "reason": reason_text,
            "expected_version": expected_version,
            "old": old_values,
            "new": new_values,
            "old_values": old_values,
            "new_values": new_values,
            "changed_fields": changed_fields,
            "transition": transition_text,
            "goal_revision": goal_revision,
        }
        _kb._append_event(
            conn,
            task_id,
            "lifecycle_bound" if lifecycle_changed else "goal_revised" if goal_changed else "updated",
            payload,
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

def bind_lifecycle_contract(
    conn: sqlite3.Connection,
    task_id: str,
    lifecycle_contract: dict,
    *,
    expected_version: int,
    reason: str,
    author: Optional[str] = None,
) -> bool:
    """Classify one historical NULL-contract task with an auditable CAS."""
    task = _kb.get_task(conn, task_id)
    if task is None:
        return False
    if task.lifecycle_contract is not None:
        raise LifecycleContractError(
            "lifecycle binding only applies to historical NULL-contract tasks"
        )
    return update_task(
        conn,
        task_id,
        expected_version=expected_version,
        reason=reason,
        lifecycle_contract=lifecycle_contract,
        author=author,
    )

def assign_task(conn: sqlite3.Connection, task_id: str, profile: Optional[str]) -> bool:
    """Assign/reassign; raises RuntimeError while the task is running under a claim."""
    profile = _kb._canonical_assignee(profile)
    with _kb.write_txn(conn):
        row = conn.execute(
            "SELECT status, claim_lock, assignee, lifecycle_contract FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()
        if not row:
            return False
        _validate_lifecycle_role_identity(
            conn, safe_decode_contract(_kb._row_get(row, "lifecycle_contract")), profile,
            task_id=task_id,
        )
        if row["claim_lock"] is not None and row["status"] == "running":
            raise RuntimeError(
                f"cannot reassign {task_id}: currently running (claimed). "
                "Wait for completion or reclaim the stale lock first."
            )
        if row["assignee"] != profile:
            # The failure streak is per task/profile; a new profile starts fresh.
            conn.execute(
                "UPDATE tasks SET assignee = ?, consecutive_failures = 0, "
                "last_failure_error = NULL WHERE id = ?", (profile, task_id),
            )
        else:
            conn.execute("UPDATE tasks SET assignee = ? WHERE id = ?", (profile, task_id))
        _kb._append_event(conn, task_id, "assigned", {"assignee": profile})
    # Observer fires AFTER commit so subscribers see durable state.
    _kb.notify_task_updated(conn, task_id, ("assignee",))
    return True

def link_tasks(
    conn: sqlite3.Connection,
    parent_id: str,
    child_id: str,
    *,
    requirement: Optional[str] = None,
    expected_parent_version: Optional[int] = None,
    expected_child_version: Optional[int] = None,
    reason: Optional[str] = None,
    author: Optional[str] = None,
) -> None:
    if parent_id == child_id:
        raise ValueError("a task cannot depend on itself")
    acceptance_before = _capture_acceptance(conn, child_id)
    with _kb.write_txn(conn):
        missing = _kb._missing_task_ids(conn, [parent_id, child_id])
        if missing:
            raise ValueError(f"unknown task(s): {', '.join(missing)}")
        if _kb._would_cycle(conn, parent_id, child_id):
            raise ValueError(f"linking {parent_id} -> {child_id} would create a cycle")
        requested = infer_edge_requirement(conn, parent_id, child_id) if requirement is None else validate_edge(
            conn, parent_id, child_id, requirement)
        existing = conn.execute(
            "SELECT requirement FROM task_links WHERE parent_id = ? AND child_id = ?",
            (parent_id, child_id),
        ).fetchone()
        if existing is not None:
            current = existing["requirement"]
            if current == requested:
                return
            if expected_parent_version is None or expected_child_version is None or not str(reason or "").strip():
                raise LifecycleContractError(
                    "rebinding an existing dependency requires both expected versions and a reason"
                )
            versions = conn.execute(
                "SELECT id, version, status, claim_lock, current_run_id FROM tasks "
                "WHERE id IN (?, ?)", (parent_id, child_id),
            ).fetchall()
            by_id = {row["id"]: row for row in versions}
            for task_id, expected_version in (
                (parent_id, expected_parent_version), (child_id, expected_child_version),
            ):
                row = by_id[task_id]
                if int(row["version"] or 1) != int(expected_version):
                    raise _kb.TaskUpdateConflict(
                        f"task {task_id} update conflict: expected version {expected_version}, "
                        f"current version {row['version']}"
                    )
                if row["status"] == "running" or row["claim_lock"] or row["current_run_id"]:
                    raise _kb.TaskUpdateConflict(f"cannot rebind edge while task {task_id} is claimed")
            conn.execute(
                "UPDATE task_links SET requirement = ? WHERE parent_id = ? AND child_id = ?",
                (requested, parent_id, child_id),
            )
            conn.execute(
                "UPDATE tasks SET version = version + 1 WHERE id IN (?, ?)",
                (parent_id, child_id),
            )
            _kb._append_event(
                conn,
                child_id,
                "link_rebound",
                {
                    "parent": parent_id,
                    "child": child_id,
                    "old_requirement": current,
                    "requirement": requested,
                    "reason": str(reason).strip(),
                    "author": str(author or os.environ.get("HERMES_PROFILE") or "orchestrator"),
                    "expected_parent_version": int(expected_parent_version),
                    "expected_child_version": int(expected_child_version),
                },
            )
        else:
            _kb._link(conn, parent_id, child_id, requirement=requested)
            _kb._append_event(
                conn, child_id, "linked",
                {"parent": parent_id, "child": child_id, "requirement": requested},
            )
            _kb._inherit_notify_subs(conn, child_id, (parent_id,))
        # A child is ready only when every typed parent requirement is satisfied.
        if not _parents_satisfied(conn, child_id):
            conn.execute(
                "UPDATE tasks SET status = 'todo' WHERE id = ? AND status = 'ready'", (child_id,),
            )
    _emit_acceptance_changes(conn, acceptance_before, source_task_id=child_id)

def recompute_ready(conn: sqlite3.Connection, failure_limit: int = None) -> int:
    """Promote ``todo``/``blocked`` tasks whose parents are all done/archived;
    returns the count. Opens its own IMMEDIATE txn — call OUTSIDE any write txn.

    ``blocked`` is skipped when sticky (explicit ``kanban_block``) or when
    ``consecutive_failures`` reached the limit (else the breaker could never
    trip). Limit order matches ``_record_task_failure``: ``max_retries`` >
    ``failure_limit`` > ``DEFAULT_FAILURE_LIMIT``.

    1. The most recent block event was a worker-initiated ``kanban_block`` — those stay blocked until an
    explicit ``kanban_unblock`` (#28712).
    """
    if failure_limit is None:
        failure_limit = _kb.DEFAULT_FAILURE_LIMIT
    promoted = 0
    with _kb.write_txn(conn):
        todo_rows = conn.execute(
            "SELECT id, status, consecutive_failures, max_retries "
            "FROM tasks WHERE status IN ('todo', 'blocked')"
        ).fetchall()
        for row in todo_rows:
            task_id = row["id"]
            cur_status = row["status"]
            if cur_status == "blocked" and _kb._has_sticky_block(conn, task_id):
                # Explicit human-intervention block; only ``unblock_task`` may exit it.
                continue
            if _parents_satisfied(conn, task_id):
                resume_status = _kb._resume_status_from_events(conn, task_id)
                if resume_status == "ready":
                    resume_status = _kb._lifecycle_ready_status(conn, task_id)
                if cur_status == "blocked":
                    # At the breaker limit, no auto-recovery (else block ->
                    # recover -> respawn -> exhaust -> block forever). The
                    # counter is preserved so it accumulates across cycles.
                    failures = int(row["consecutive_failures"] or 0)
                    task_limit = row["max_retries"]
                    effective_limit = (
                        int(task_limit) if task_limit is not None
                        else int(failure_limit)
                    )
                    if failures >= effective_limit:
                        continue
                    conn.execute(
                        "UPDATE tasks SET status = ? "
                        "WHERE id = ? AND status = 'blocked'", (resume_status, task_id),
                    )
                else:
                    conn.execute(
                        "UPDATE tasks SET status = ? WHERE id = ? AND status = 'todo'",
                        (resume_status, task_id),
                    )
                _kb._append_event(
                    conn, task_id, "promoted",
                    {"status": resume_status} if resume_status != "ready" else None,
                )
                promoted += 1
    return promoted

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

def _claim_and_open_run(
    conn: sqlite3.Connection, task_id: str, source_status: str, lock: str, expires: int, now: int,
    *, event_extra: Optional[dict] = None, runtime_claim: Optional[dict[str, Any]] = None,
) -> Optional[int]:
    """CAS ``source_status -> running``, open a run row, emit ``claimed``; None
    when the CAS lost. Caller holds the txn."""
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
        if not dependencies["satisfied"]:
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
            conn, task_id, "ready", lock, expires, now, runtime_claim=runtime_claim,
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
        dependencies = evaluate_dependencies(conn, task_id)
        if not dependencies["satisfied"]:
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
            event_extra={"source_status": "review"}, runtime_claim=runtime_claim,
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

def _latest_lifecycle_run_id(
    conn: sqlite3.Connection, task_id: str, phase: str,
) -> Optional[int]:
    pointer = conn.execute(
        "SELECT candidate_run_id FROM tasks WHERE id = ?", (task_id,)
    ).fetchone()
    if pointer is not None and pointer["candidate_run_id"] is not None:
        run_id = int(pointer["candidate_run_id"])
        row = conn.execute(
            "SELECT metadata FROM task_runs WHERE id = ? AND task_id = ?",
            (run_id, task_id),
        ).fetchone()
        lifecycle = _kb._json_dict(_kb._row_get(row, "metadata")).get("lifecycle")
        if isinstance(lifecycle, dict) and lifecycle.get("phase") == phase:
            return run_id
        return None
    return None

def _stamp_lifecycle_metadata(
    conn: sqlite3.Connection,
    task_id: str,
    metadata: Optional[dict],
    *,
    phase: str,
    run_id: Optional[int],
    verdict: Optional[str],
) -> Optional[dict]:
    """Build typed evidence from the task/run snapshot being committed."""
    task = _kb.get_task(conn, task_id)
    contract = task.lifecycle_contract if task else None
    if not contract:
        if verdict is not None:
            raise LifecycleEvidenceError("unclassified tasks cannot record typed verdicts")
        return metadata
    updated = dict(metadata) if isinstance(metadata, dict) else {}
    candidate_task_id = (
        task_id
        if phase == "implementation"
        or (
            phase == "review"
            and contract.get("kind") == "code"
            and contract.get("review_mode") == "same_card"
        )
        else contract.get("candidate_task_id")
    )
    if not candidate_task_id:
        raise LifecycleEvidenceError("typed lifecycle evidence requires a candidate task")
    candidate_task_id = str(candidate_task_id)
    candidate_run_id = (
        run_id
        if phase == "implementation"
        else _latest_lifecycle_run_id(conn, candidate_task_id, "implementation")
    )

    if candidate_run_id is None:
        candidate_run_id = _kb._row_get(
            conn.execute(
                "SELECT candidate_run_id FROM tasks WHERE id = ?", (task_id,)
            ).fetchone(),
            "candidate_run_id",
        )
    supplied_head = (
        updated.get("head_sha")
        or updated.get("reviewed_head_sha")
        or _kb._handoff_fields(updated).get("head_sha")
    )
    current_head = _latest_head(conn, candidate_task_id)
    if phase == "implementation":
        prior_rework = conn.execute(
            "SELECT kind, payload FROM task_events WHERE task_id = ? "
            "AND kind IN ('review_rework_requested', 'changes_requested') "
            "ORDER BY id DESC LIMIT 1",
            (task_id,),
        ).fetchone()
        prior_payload = _kb._json_dict(_kb._row_get(prior_rework, "payload"))
        prior_lifecycle = prior_payload.get("lifecycle")
        if not isinstance(prior_lifecycle, dict):
            prior_lifecycle = {}
        rejected_head = (
            prior_payload.get("rejected_head_sha")
            or prior_lifecycle.get("head_sha")
        )
        if rejected_head and (supplied_head or current_head) == rejected_head:
            raise LifecycleEvidenceError("rework requires a new implementation head")
    if phase != "implementation" and current_head and supplied_head and supplied_head != current_head:
        raise LifecycleEvidenceError("lifecycle evidence head_sha does not match the candidate head")
    head_sha = supplied_head or current_head or _kb.latest_handoff(conn, candidate_task_id).get("head_sha")
    envelope = lifecycle_metadata(
        conn,
        task_id,
        phase=phase,
        run_id=candidate_run_id,
        verdict=verdict,
        candidate_task_id=candidate_task_id,
        head_sha=head_sha,
    )
    existing = updated.get("lifecycle")
    if existing is not None and existing != envelope:
        raise LifecycleEvidenceError("lifecycle evidence conflicts with the current candidate snapshot")
    updated["lifecycle"] = envelope
    return updated

def _implementation_routing(conn: sqlite3.Connection, task_id: str) -> dict[str, Any]:
    rows = conn.execute(
        "SELECT metadata FROM task_runs WHERE task_id = ? ORDER BY id DESC",
        (task_id,),
    ).fetchall()
    for row in rows:
        routing = _kb._json_dict(row["metadata"]).get("lifecycle_routing")
        if isinstance(routing, dict):
            return routing
    return {}

def _lifecycle_observed_tasks(conn: sqlite3.Connection, task_id: str) -> tuple[str, ...]:
    task = _kb.get_task(conn, task_id)
    if task is None or not task.lifecycle_contract:
        return ()
    contract = task.lifecycle_contract
    if contract.get("kind") == "code":
        return (task_id,)
    candidate = contract.get("candidate_task_id")
    return (str(candidate),) if candidate else ()

def _capture_acceptance(conn: sqlite3.Connection, task_id: str) -> dict[str, str]:
    return {
        observed_id: get_lifecycle_state(conn, observed_id).get("acceptance", "unclassified")
        for observed_id in _lifecycle_observed_tasks(conn, task_id)
    }

def _emit_acceptance_changes(
    conn: sqlite3.Connection,
    before: dict[str, str],
    *,
    source_task_id: Optional[str] = None,
) -> None:
    if not before:
        return
    changes: list[tuple[str, str, str, str, Any]] = []
    for task_id, old in before.items():
        projection = get_lifecycle_state(conn, task_id)
        new = projection.get("acceptance", "unclassified")
        if new != old:
            phase = (
                "validation"
                if projection.get("validation_verdict") is not None
                else "review"
                if projection.get("review_verdict") is not None
                else "implementation"
            )
            result = (
                projection.get("validation_verdict")
                or projection.get("review_verdict")
                or projection.get("execution_outcome")
            )
            changes.append((task_id, old, new, phase, result))
    if not changes:
        return
    with _kb.write_txn(conn):
        for task_id, old, new, phase, result in changes:
            _kb._append_event(
                conn,
                task_id,
                "acceptance_changed",
                {
                    "old": old,
                    "new": new,
                    "phase": phase,
                    "result": result,
                    "source_task_id": source_task_id,
                },
            )

def _handoff_children(conn: sqlite3.Connection, task_id: str) -> list[tuple[str, str]]:
    rows = conn.execute(
        "SELECT t.id, t.assignee, t.lifecycle_contract "
        "FROM task_links l JOIN tasks t ON t.id = l.child_id "
        "WHERE l.parent_id = ? ORDER BY t.id",
        (task_id,),
    ).fetchall()
    children: list[tuple[str, str]] = []
    for row in rows:
        contract = safe_decode_contract(row["lifecycle_contract"])
        if contract and contract.get("kind") in {"review", "validation"}:
            children.append((row["id"], contract["kind"]))
            continue
        assignee = str(row["assignee"] or "").strip().casefold()
        if assignee in _kb.HANDOFF_CHILD_ASSIGNEES:
            children.append((row["id"], assignee))
    return children

def _completion_contract_snapshot(
    conn: sqlite3.Connection,
    task_id: str,
) -> tuple[Optional[int], Optional[int], tuple[tuple[str, str], ...]]:
    goal = _kb.get_effective_goal(conn, task_id)
    return (
        int(goal["id"]) if goal and goal.get("id") is not None else None,
        int(goal.get("version", 1)) if goal else None,
        tuple(_handoff_children(conn, task_id)),
    )

def _parent_handoff_context(
    conn: sqlite3.Connection,
    task_id: str,
) -> Optional[dict[str, Any]]:
    for parent_id in _kb.parent_ids(conn, task_id):
        parent = _kb.get_task(conn, parent_id)
        if parent is None or parent.status not in {"done", "archived"}:
            continue
        handoff = _kb.latest_handoff(conn, parent_id)
        if handoff:
            handoff.setdefault("branch_name", parent.branch_name)
            handoff.setdefault("workspace_path", parent.workspace_path)
            handoff["parent_task_id"] = parent_id
            return handoff
    return None

def _prepare_completion_handoff(
    conn: sqlite3.Connection,
    task_id: str,
    metadata: Optional[dict],
    *,
    phase: Optional[str] = None,
) -> tuple[Optional[dict], dict[str, Any]]:
    task = _kb.get_task(conn, task_id)
    if task is None:
        return metadata, {}
    updated = dict(metadata) if isinstance(metadata, dict) else {}
    supplied = _kb._handoff_fields(updated)
    dependent_children = _handoff_children(conn, task_id)
    effective_goal = _kb.get_effective_goal(conn, task_id)
    contract_reason = _kb._completion_contract_reason(
        effective_goal,
        supplied.get("changed_files"),
        dependent_children,
    )
    if contract_reason:
        raise _kb.CompletionContractError(
            task_id,
            contract_reason,
            changed_files=supplied.get("changed_files"),
        )
    task_contract = task.lifecycle_contract or {}
    if (
        task.workflow_template_id == "kanban_swarm_v1"
        and task_contract.get("kind") != "code"
    ):
        return metadata, supplied
    is_review_card = task_contract.get("kind") == "review" or (
        not task_contract
        and str(task.assignee or "").strip().casefold() == "reviewer"
    )
    # A reviewer handing off to a tester reviews the implementation's exact
    # commit; do not require a second commit in the reviewer's scratch workspace.
    if (
        phase == "review"
        and task_contract.get("kind") == "code"
        and task_contract.get("review_mode") == "same_card"
    ):
        implementation_handoff = _kb.latest_handoff(conn, task_id)
        if not implementation_handoff.get("head_sha"):
            raise _kb.HandoffValidationError(
                task_id, "implementation handoff head_sha is required"
            )
        for key in _kb.HANDOFF_KEYS:
            if (
                supplied.get(key)
                and implementation_handoff.get(key)
                and supplied[key] != implementation_handoff[key]
            ):
                raise _kb.HandoffValidationError(
                    task_id, f"{key} does not match the implementation handoff"
                )
        updated.update(implementation_handoff)
        return updated, _kb._handoff_fields(updated)
    if dependent_children and is_review_card:
        parent_handoff = _parent_handoff_context(conn, task_id)
        if parent_handoff and parent_handoff.get("head_sha"):
            for key in ("base_sha", "head_sha"):
                if supplied.get(key) and supplied[key] != parent_handoff.get(key):
                    raise _kb.HandoffValidationError(
                        task_id, f"{key} does not match the reviewed parent"
                    )
            for key in (
                "base_sha",
                "head_sha",
                "changed_files",
                "branch_name",
                "workspace_path",
                "dirty_state",
                "patch_artifact",
                "patch_sha256",
            ):
                if key in parent_handoff:
                    updated[key] = parent_handoff[key]
            updated["handoff_provenance"] = {
                "kind": "reviewed_parent",
                "parent_task_id": parent_handoff["parent_task_id"],
                "parent_provenance": parent_handoff.get("provenance"),
            }
            merged_handoff = _kb._handoff_fields(updated)
            contract_reason = _kb._completion_contract_reason(
                effective_goal,
                merged_handoff.get("changed_files"),
                dependent_children,
            )
            if contract_reason:
                raise _kb.CompletionContractError(
                    task_id,
                    contract_reason,
                    changed_files=merged_handoff.get("changed_files"),
                )
            return updated, merged_handoff

    requires_immutable = (
        task_contract.get("kind") == "code"
        or task.workspace_kind == "worktree"
        or any(
            supplied.get(key)
            for key in ("base_sha", "head_sha", "changed_files", "patch_artifact")
        )
    )
    if (
        (not dependent_children and task_contract.get("kind") != "code")
        or not requires_immutable
    ):
        return metadata, supplied
    base = str(supplied.get("base_sha") or "").strip()
    head = str(supplied.get("head_sha") or "").strip()
    if not base:
        raise _kb.HandoffValidationError(task_id, "base_sha is required")
    if not head:
        raise _kb.HandoffValidationError(task_id, "head_sha is required")
    branch = str(supplied.get("branch_name") or task.branch_name or "").strip() or None
    workspace = str(supplied.get("workspace_path") or task.workspace_path or "").strip()
    snapshot = _kb._git_snapshot(workspace, branch)
    if snapshot is None:
        raise _kb.HandoffValidationError(
            task_id, f"worktree {workspace or '(unresolved)'!r} is unavailable"
        )
    snapshot_files = snapshot.get("dirty_files") if snapshot is not None else None
    contract_reason = _kb._completion_contract_reason(
        effective_goal,
        snapshot_files,
        dependent_children,
    )
    if contract_reason:
        raise _kb.CompletionContractError(
            task_id,
            contract_reason,
            changed_files=snapshot_files,
        )
    if branch and snapshot["current_branch"] and snapshot["current_branch"] != branch:
        raise _kb.HandoffValidationError(
            task_id,
            f"worktree is on branch {snapshot['current_branch']!r}, not task branch {branch!r}",
        )
    if snapshot["dirty_files"]:
        raise _kb.HandoffValidationError(
            task_id,
            f"worktree has dirty or untracked files ({', '.join(snapshot['dirty_files'])})",
        )
    resolved_base = _kb._resolve_commit(snapshot["path"], base)
    resolved_head = _kb._resolve_commit(snapshot["path"], head)
    if resolved_base is None or resolved_head is None:
        raise _kb.HandoffValidationError(
            task_id, "base_sha and head_sha must resolve to commits"
        )
    if resolved_head != snapshot["branch_head"]:
        raise _kb.HandoffValidationError(
            task_id, "recorded head_sha does not match the task branch head"
        )
    if not _kb._is_ancestor(snapshot["path"], resolved_base, resolved_head):
        raise _kb.HandoffValidationError(task_id, "base_sha is not an ancestor of head_sha")
    changed_raw = (
        _kb._git_out(
            snapshot["path"], "diff", "--name-only", "-z", resolved_base, resolved_head
        )
        or ""
    )
    updated.update({
        "base_sha": resolved_base,
        "head_sha": resolved_head,
        "changed_files": [name for name in changed_raw.split("\0") if name],
        "branch_name": branch,
        "workspace_path": workspace,
        "dirty_state": "clean",
    })
    recovery = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'handoff_requeued' "
        "ORDER BY id DESC LIMIT 1",
        (task_id,),
    ).fetchone()
    if recovery:
        payload = _kb._json_dict(recovery["payload"])
        updated["provenance"] = {
            "kind": "legacy_handoff_recompletion",
            "actor": payload.get("actor"),
            "reason": payload.get("reason"),
            "supersedes_completion_event_id": payload.get(
                "supersedes_completion_event_id"
            ),
        }
    normalized_handoff = _kb._handoff_fields(updated)
    contract_reason = _kb._completion_contract_reason(
        effective_goal,
        normalized_handoff.get("changed_files"),
        dependent_children,
    )
    if contract_reason:
        raise _kb.CompletionContractError(
            task_id,
            contract_reason,
            changed_files=normalized_handoff.get("changed_files"),
        )
    return updated, normalized_handoff

def _parent_handoff_start_error(
    conn: sqlite3.Connection,
    task_id: str,
) -> Optional[dict[str, Any]]:
    task = _kb.get_task(conn, task_id)
    if task is None or task.workflow_template_id == "kanban_swarm_v1":
        return None
    contract = task.lifecycle_contract or {}
    is_role_card = contract.get("kind") in {"review", "validation"} or (
        not contract and str(task.assignee or "").strip().casefold() in _kb.HANDOFF_CHILD_ASSIGNEES
    )
    if not is_role_card:
        return None
    for parent_id in _kb.parent_ids(conn, task_id):
        parent = _kb.get_task(conn, parent_id)
        if parent is None or parent.status not in {"done", "archived"}:
            continue
        handoff = _kb.latest_handoff(conn, parent_id)
        expected = handoff.get("head_sha")
        if not expected and (
            parent.lifecycle_contract
            and parent.lifecycle_contract.get("kind") == "code"
            and parent.lifecycle_contract.get("review_mode") == "same_card"
        ):
            expected = _latest_head(conn, parent_id)
        if not expected:
            return {
                "kind": "handoff_unverifiable",
                "parent_id": parent_id,
                "reason": "parent completion has no head_sha",
            }
        branch = handoff.get("branch_name") or parent.branch_name
        workspace = handoff.get("workspace_path") or parent.workspace_path
        snapshot = _kb._git_snapshot(workspace, branch)
        actual = snapshot["branch_head"] if snapshot else None
        resolved = _kb._resolve_commit(snapshot["path"], expected) if snapshot else None
        if actual is None:
            return {
                "kind": "handoff_unverifiable",
                "parent_id": parent_id,
                "expected_head_sha": expected,
                "reason": "parent branch/worktree head is unavailable",
            }
        if resolved != actual:
            return {
                "kind": "handoff_head_moved",
                "parent_id": parent_id,
                "expected_head_sha": resolved or expected,
                "actual_head_sha": actual,
                "reason": "parent task branch moved after approval",
            }
    return None

def _record_parent_handoff_start_error(
    conn: sqlite3.Connection,
    task_id: str,
    error: dict[str, Any],
) -> None:
    kind = str(error.get("kind") or "handoff_unverifiable")
    encoded = json.dumps(error, ensure_ascii=False, sort_keys=True)
    previous = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = ? ORDER BY id DESC LIMIT 1",
        (task_id, kind),
    ).fetchone()
    if previous and previous["payload"] == encoded:
        return
    _kb._append_event(conn, task_id, kind, error)

def _rework_fingerprint(
    implementation_id: str,
    reviewer_id: Optional[str],
    tester_id: Optional[str],
    expected: dict[str, Optional[int]],
    reason: str,
    author: str,
) -> str:
    import hashlib

    payload = {
        "implementation_id": implementation_id,
        "reviewer_id": reviewer_id,
        "tester_id": tester_id,
        "expected_versions": expected,
        "reason": reason,
        "author": author,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()

def _typed_rework_graph(
    conn: sqlite3.Connection,
    implementation_id: str,
    reviewer_id: Optional[str],
    tester_id: Optional[str],
    *,
    expected_implementation_version: int,
    expected_reviewer_version: Optional[int],
    expected_tester_version: Optional[int],
    reason: str,
    author: str,
) -> dict[str, Any]:
    implementation = _kb.get_task(conn, implementation_id)
    contract = implementation.lifecycle_contract if implementation else None
    if not implementation or not contract or contract.get("kind") != "code":
        raise ValueError("typed rework requires a classified code implementation task")
    expected_values = (
        ("expected_implementation_version", expected_implementation_version),
        ("expected_reviewer_version", expected_reviewer_version),
        ("expected_tester_version", expected_tester_version),
    )
    for name, value in expected_values:
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 1
        ):
            raise ValueError(f"{name} must be an integer >= 1")
    same_card = contract.get("review_mode") == "same_card"
    if same_card and bool(reviewer_id) != bool(tester_id):
        raise ValueError("same-card rework reviewer_id and tester_id must be supplied together")
    if not same_card and not reviewer_id:
        raise ValueError("separate-card rework requires reviewer_id")
    if same_card and reviewer_id and str(reviewer_id) != implementation_id:
        raise ValueError("same-card reviewer_id must identify the implementation card")

    def _child(parent_id: str, kind: str) -> Optional[str]:
        rows = conn.execute(
            "SELECT t.id, t.lifecycle_contract FROM task_links l JOIN tasks t ON t.id = l.child_id "
            "WHERE l.parent_id = ? ORDER BY t.id",
            (parent_id,),
        ).fetchall()
        for row in rows:
            child = safe_decode_contract(row["lifecycle_contract"])
            if (
                child
                and child.get("kind") == kind
                and child.get("candidate_task_id") == implementation_id
            ):
                return row["id"]
        return None

    validation_id = _child(implementation_id if same_card else str(reviewer_id), "validation")
    if contract.get("validation_required"):
        if validation_id is None:
            raise ValueError("typed rework requires the declared validation card")
        if not same_card and tester_id is None:
            raise ValueError("separate-card rework requires tester_id")
    elif tester_id is not None:
        raise ValueError("tester_id is only valid when validation_required is true")
    if tester_id is not None and str(tester_id) != validation_id:
        raise ValueError("tester_id does not identify the declared validation card")
    if expected_reviewer_version is not None and same_card:
        raise ValueError("same-card rework uses expected_implementation_version only for review")
    if not same_card and expected_reviewer_version is None:
        raise ValueError("expected_reviewer_version is required for separate-card rework")
    if validation_id is not None and expected_tester_version is None:
        raise ValueError("expected_tester_version is required for typed validation rework")

    actual_reviewer_id = implementation_id if same_card else str(reviewer_id)
    actual_tester_id = validation_id
    expected: dict[str, Optional[int]] = {implementation_id: expected_implementation_version}
    if not same_card:
        expected[actual_reviewer_id] = expected_reviewer_version
    if actual_tester_id:
        expected[actual_tester_id] = expected_tester_version
    fingerprint = _rework_fingerprint(
        implementation_id, reviewer_id, tester_id, expected, reason, author
    )
    previous_rows = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'review_rework_requested' "
        "ORDER BY id DESC",
        (implementation_id,),
    ).fetchall()
    for previous in previous_rows:
        prior_payload = _kb._json_dict(previous["payload"])
        if prior_payload.get("fingerprint") != fingerprint:
            continue
        prior_result = prior_payload.get("result")
        if isinstance(prior_result, dict):
            return prior_result
        raise ValueError("rework_fingerprint_conflict: stored result is malformed")

    acceptance_before = _capture_acceptance(conn, implementation_id)
    review_state = get_lifecycle_state(conn, actual_reviewer_id)
    validation_state = (
        get_lifecycle_state(conn, actual_tester_id) if actual_tester_id else None
    )
    if review_state.get("review_verdict") != "REQUEST_CHANGES" and not (
        validation_state and validation_state.get("validation_verdict") == "FAIL"
    ):
        raise ValueError("typed rework requires REQUEST_CHANGES or FAIL evidence")
    if (
        review_state.get("diagnostics")
        and not (same_card and review_state.get("review_verdict") == "REQUEST_CHANGES")
    ) or (validation_state and validation_state.get("diagnostics")):
        raise ValueError("typed rework evidence is stale or malformed")
    rejected_head = None
    for evidence in conn.execute(
        "SELECT metadata FROM task_runs WHERE task_id = ? ORDER BY id DESC",
        (implementation_id,),
    ).fetchall():
        lifecycle = _kb._json_dict(evidence["metadata"]).get("lifecycle")
        if isinstance(lifecycle, dict) and lifecycle.get("head_sha"):
            rejected_head = str(lifecycle["head_sha"]).strip()
            break
    if not rejected_head:
        raise ValueError("typed rework requires an immutable rejected implementation head")

    implementation_assignee = None
    if same_card:
        implementation_assignee = _kb._canonical_assignee(
            _implementation_routing(conn, implementation_id).get("implementer")
        )
        if not implementation_assignee:
            raise ValueError("same-card rework requires original implementation routing")
        _validate_lifecycle_role_identity(
            conn,
            contract,
            implementation_assignee,
            task_id=implementation_id,
            phase="implementation",
        )

    reset_ids = [implementation_id]
    if not same_card:
        reset_ids.append(actual_reviewer_id)
    if actual_tester_id:
        reset_ids.append(actual_tester_id)
    with _kb.write_txn(conn):
        rows = {
            row["id"]: row
            for row in conn.execute(
                "SELECT id, status, version, assignee, claim_lock, current_run_id, worker_pid, "
                "completed_at, result, block_kind, block_recurrences "
                "FROM tasks WHERE id IN (" + ",".join("?" for _ in reset_ids) + ")",
                tuple(reset_ids),
            ).fetchall()
        }
        for task_id, expected_version in expected.items():
            row = rows.get(task_id)
            if row is None:
                raise ValueError(f"typed rework task {task_id} was not found")
            if int(row["version"] or 1) != int(expected_version):
                raise ValueError(
                    f"task {task_id} update conflict: expected version {expected_version}, "
                    f"current version {row['version']}"
                )
        descendants = conn.execute(
            """
            WITH RECURSIVE graph(id) AS (
                SELECT child_id FROM task_links WHERE parent_id = ?
                UNION
                SELECT l.child_id FROM task_links l JOIN graph g ON g.id = l.parent_id
            )
            SELECT t.id, t.status, t.version, t.claim_lock, t.current_run_id,
                   t.worker_pid, t.completed_at, t.result, t.block_kind
            FROM graph JOIN tasks t ON t.id = graph.id ORDER BY t.id
            """,
            (implementation_id,),
        ).fetchall()
        for row in descendants:
            if row["status"] == "running" or row["claim_lock"] or row["current_run_id"] or row["worker_pid"]:
                raise ValueError(f"cannot rework graph while task {row['id']} is claimed")
        for row in [rows[task_id] for task_id in reset_ids]:
            if row["status"] == "archived":
                raise ValueError(f"cannot rework archived task {row['id']}")
        implementation_status = "ready" if _parents_satisfied(conn, implementation_id) else "todo"
        invalidated: list[dict[str, Any]] = []
        reset_set = set(reset_ids)
        all_rows = [rows[task_id] for task_id in reset_ids]
        all_rows.extend(row for row in descendants if row["id"] not in reset_set)
        for row in all_rows:
            if row["status"] == "archived":
                continue
            new_status = implementation_status if row["id"] == implementation_id else "todo"
            if row["status"] == "blocked" and _kb._has_sticky_block(conn, row["id"]):
                new_status = "blocked"
            assign_implementation = same_card and row["id"] == implementation_id
            new_assignee = (
                implementation_assignee if assign_implementation else row["assignee"]
            )
            assignment_sql = ", assignee = ?" if assign_implementation else ""
            params: list[Any] = [new_status]
            if assign_implementation:
                params.append(implementation_assignee)
            params.extend((new_status, new_status, row["id"]))
            conn.execute(
                "UPDATE tasks SET status = ?" + assignment_sql + ", version = version + 1, "
                "completed_at = NULL, result = NULL, current_run_id = NULL, "
                "claim_lock = NULL, claim_expires = NULL, worker_pid = NULL, "
                "candidate_run_id = NULL, "
                "block_kind = CASE WHEN ? = 'blocked' THEN block_kind ELSE NULL END, "
                "block_recurrences = CASE WHEN ? = 'blocked' THEN block_recurrences ELSE 0 END "
                "WHERE id = ?",
                tuple(params),
            )
            entry = {
                "id": row["id"],
                "prior_assignee": row["assignee"],
                "new_assignee": new_assignee,
                "prior_status": row["status"],
                "new_status": new_status,
                "prior_version": int(row["version"] or 1),
                "prior_completed_at": row["completed_at"],
                "prior_result": row["result"],
            }
            invalidated.append(entry)
            _kb._append_event(
                conn,
                row["id"],
                "acceptance_invalidated",
                {
                    "implementation": implementation_id,
                    "reviewer": None if same_card else actual_reviewer_id,
                    "tester": actual_tester_id,
                    "reason": reason,
                    "rejected_head_sha": rejected_head,
                    "actor": author,
                    **entry,
                },
            )
        result = {
            "implementation_id": implementation_id,
            "reviewer_id": None if same_card else actual_reviewer_id,
            "tester_id": actual_tester_id,
            "status": implementation_status,
            "rejected_head_sha": rejected_head,
            "invalidated": invalidated,
            "implementation_version": int(rows[implementation_id]["version"]) + 1,
            "reviewer_version": (
                int(rows[actual_reviewer_id]["version"]) + 1 if not same_card else None
            ),
            "tester_version": (
                int(rows[actual_tester_id]["version"]) + 1 if actual_tester_id else None
            ),
        }
        _kb._append_event(
            conn,
            implementation_id,
            "review_rework_requested",
            {
                "actor": author,
                "reason": reason,
                "reviewer": None if same_card else actual_reviewer_id,
                "tester": actual_tester_id,
                "rejected_head_sha": rejected_head,
                "fingerprint": fingerprint,
                "result": result,
            },
        )
    _emit_acceptance_changes(conn, acceptance_before, source_task_id=implementation_id)
    for entry in invalidated:
        fields = ("status", "version", "completed_at", "result")
        if entry["id"] == implementation_id and same_card:
            fields += ("assignee",)
        _kb.notify_task_updated(conn, entry["id"], fields)
    return result

def rework_review_graph(
    conn: sqlite3.Connection,
    implementation_id: str,
    reviewer_id: Optional[str] = None,
    tester_id: Optional[str] = None,
    *,
    expected_implementation_version: int,
    expected_reviewer_version: Optional[int] = None,
    expected_tester_version: Optional[int] = None,
    reason: str,
    author: Optional[str] = None,
) -> dict[str, Any]:
    """Atomically requeue one rejected implementation/review/test chain.

    The three cards and their direct edges are the durable identity of the
    workflow.  Rework changes only their current scheduling state; runs,
    comments, goal revisions, and completion events remain historical.
    """
    typed_impl = _kb.get_task(conn, implementation_id)
    if typed_impl and typed_impl.lifecycle_contract and typed_impl.lifecycle_contract.get("kind") == "code":
        typed_reason = str(reason or "").strip()
        if not typed_reason:
            raise ValueError("reason is required")
        typed_actor = str(author or os.environ.get("HERMES_PROFILE") or "orchestrator").strip() or "orchestrator"
        return _typed_rework_graph(
            conn,
            implementation_id,
            reviewer_id,
            tester_id,
            expected_implementation_version=expected_implementation_version,
            expected_reviewer_version=expected_reviewer_version,
            expected_tester_version=expected_tester_version,
            reason=typed_reason,
            author=typed_actor,
        )
    task_ids = (implementation_id, reviewer_id, tester_id)
    if any(not isinstance(task_id, str) or not task_id.strip() for task_id in task_ids):
        raise ValueError("implementation, reviewer, and tester ids are required")
    if len(set(task_ids)) != 3:
        raise ValueError("implementation, reviewer, and tester ids must be distinct")
    expected = {
        implementation_id: expected_implementation_version,
        reviewer_id: expected_reviewer_version,
        tester_id: expected_tester_version,
    }
    if any(
        isinstance(version, bool) or not isinstance(version, int) or version < 1
        for version in expected.values()
    ):
        raise ValueError("expected versions must be integers >= 1")
    reason = str(reason or "").strip()
    if not reason:
        raise ValueError("reason is required")
    actor = str(author or os.environ.get("HERMES_PROFILE") or "orchestrator")

    with _kb.write_txn(conn):
        rows = {
            row["id"]: row
            for row in conn.execute(
                "SELECT id, status, version, claim_lock, current_run_id, worker_pid "
                "FROM tasks WHERE id IN (?, ?, ?)",
                task_ids,
            ).fetchall()
        }
        if set(rows) != set(expected):
            raise ValueError("implementation, reviewer, or tester task was not found")
        required_statuses = {
            implementation_id: "done",
            reviewer_id: "done",
        }
        for task_id, version in expected.items():
            row = rows[task_id]
            if int(row["version"] or 1) != version:
                raise ValueError(
                    f"task {task_id} update conflict: expected version {version}, "
                    f"current version {row['version']}"
                )
            if task_id == tester_id:
                status_ok = row["status"] in {"blocked", "todo"}
                expected_status = "'blocked' or 'todo'"
            else:
                status_ok = row["status"] == required_statuses[task_id]
                expected_status = repr(required_statuses[task_id])
            if not status_ok:
                raise ValueError(
                    f"task {task_id} must be {expected_status}, "
                    f"current {row['status']!r}"
                )
            if row["status"] == "running" or row["claim_lock"] or row["current_run_id"] or row["worker_pid"]:
                raise ValueError(f"task {task_id} is currently claimed")

        links = {
            tuple(row)
            for row in conn.execute(
                "SELECT parent_id, child_id FROM task_links "
                "WHERE (parent_id = ? AND child_id = ?) "
                "OR (parent_id = ? AND child_id = ?)",
                (implementation_id, reviewer_id, reviewer_id, tester_id),
            ).fetchall()
        }
        if links != {
            (implementation_id, reviewer_id),
            (reviewer_id, tester_id),
        }:
            raise ValueError("tasks are not an implementation -> reviewer -> tester chain")

        handoff = _kb.latest_handoff(conn, implementation_id)
        head_sha = _nonblank_str(handoff.get("head_sha"))
        if not head_sha:
            raise _kb.HandoffValidationError(implementation_id, "immutable head_sha is required")
        review_run = conn.execute(
            "SELECT id, metadata FROM task_runs WHERE task_id = ? AND outcome = 'completed' "
            "ORDER BY id DESC LIMIT 1",
            (reviewer_id,),
        ).fetchone()
        review_metadata = _kb._json_dict(_kb._row_get(review_run, "metadata"))
        if str(review_metadata.get("verdict") or "").strip().upper() != "REQUEST_CHANGES":
            raise ValueError("reviewer's latest completed verdict is not REQUEST_CHANGES")
        reviewed_head = _nonblank_str(
            review_metadata.get("reviewed_head_sha")
            or review_metadata.get("head_ref_reverified")
            or review_metadata.get("head_sha")
        )
        if reviewed_head != head_sha:
            raise ValueError("review verdict does not match the implementation head")

        descendants = conn.execute(
            """
            WITH RECURSIVE graph(id) AS (
                SELECT ?
                UNION
                SELECT l.child_id FROM task_links l JOIN graph g ON g.id = l.parent_id
            )
            SELECT t.id, t.status, t.version, t.completed_at, t.result,
                   t.claim_lock, t.current_run_id, t.worker_pid,
                   t.block_kind, t.block_recurrences
            FROM graph g JOIN tasks t ON t.id = g.id
            ORDER BY t.id
            """,
            (reviewer_id,),
        ).fetchall()
        for row in descendants:
            if row["status"] == "running" or row["claim_lock"] or row["current_run_id"] or row["worker_pid"]:
                raise ValueError(f"cannot rework graph while task {row['id']} is claimed")

        completion = conn.execute(
            "SELECT id FROM task_events WHERE task_id = ? AND kind = 'completed' "
            "ORDER BY id DESC LIMIT 1",
            (implementation_id,),
        ).fetchone()
        if completion is None or review_run is None:
            raise ValueError("completion or review verdict provenance is unavailable")
        implementation_status = "ready" if _parents_satisfied(conn, implementation_id) else "todo"
        implementation_update = conn.execute(
            """
            UPDATE tasks SET status = ?, version = version + 1, completed_at = NULL,
                result = NULL, current_run_id = NULL, claim_lock = NULL,
                claim_expires = NULL, worker_pid = NULL
            WHERE id = ? AND version = ?
            """,
            (implementation_status, implementation_id, expected_implementation_version),
        )
        if implementation_update.rowcount != 1:
            raise ValueError(
                f"task {implementation_id} update conflict: version changed while reworking"
            )
        invalidated: list[dict[str, Any]] = []
        for row in descendants:
            if row["status"] == "archived":
                continue
            preserve_sticky_block = (
                row["status"] == "blocked"
                and _kb._has_sticky_block(conn, row["id"])
            )
            if preserve_sticky_block:
                conn.execute(
                    "UPDATE tasks SET status = 'blocked', version = version + 1, "
                    "completed_at = NULL, result = NULL, current_run_id = NULL, "
                    "claim_lock = NULL, claim_expires = NULL, worker_pid = NULL "
                    "WHERE id = ?",
                    (row["id"],),
                )
            else:
                conn.execute(
                    "UPDATE tasks SET status = 'todo', version = version + 1, "
                    "completed_at = NULL, result = NULL, current_run_id = NULL, "
                    "claim_lock = NULL, claim_expires = NULL, worker_pid = NULL, "
                    "block_kind = NULL, block_recurrences = 0 WHERE id = ?",
                    (row["id"],),
                )
            invalidated.append({
                "id": row["id"],
                "prior_status": row["status"],
                "prior_version": int(row["version"] or 1),
                "prior_completed_at": row["completed_at"],
                "prior_result": row["result"],
            })
            _kb._append_event(
                conn,
                row["id"],
                "acceptance_invalidated",
                {
                    "implementation": implementation_id,
                    "reviewer": reviewer_id,
                    "tester": tester_id,
                    "reason": reason,
                    "rejected_head_sha": head_sha,
                    "review_run_id": int(review_run["id"]),
                    "actor": actor,
                    "prior_status": row["status"],
                    "prior_version": int(row["version"] or 1),
                    "prior_completed_at": row["completed_at"],
                    "prior_result": row["result"],
                },
            )
        _kb._append_event(
            conn,
            implementation_id,
            "review_rework_requested",
            {
                "actor": actor,
                "reason": reason,
                "reviewer": reviewer_id,
                "tester": tester_id,
                "rejected_head_sha": head_sha,
                "review_run_id": int(review_run["id"]),
                "supersedes_completion_event_id": int(completion["id"]),
                "invalidated": invalidated,
                "status": implementation_status,
                "expected_versions": expected,
            },
        )

    _kb.notify_task_updated(conn, implementation_id, ("status", "version", "completed_at", "result"))
    for entry in invalidated:
        _kb.notify_task_updated(conn, entry["id"], ("status", "version", "completed_at", "result"))
    implementation = _kb.get_task(conn, implementation_id)
    reviewer = _kb.get_task(conn, reviewer_id)
    tester = _kb.get_task(conn, tester_id)
    return {
        "implementation_id": implementation_id,
        "reviewer_id": reviewer_id,
        "tester_id": tester_id,
        "status": implementation_status,
        "rejected_head_sha": head_sha,
        "review_run_id": int(review_run["id"]),
        "invalidated": invalidated,
        "implementation_version": implementation.version if implementation else None,
        "reviewer_version": reviewer.version if reviewer else None,
        "tester_version": tester.version if tester else None,
    }

class ArtifactPreservationError(RuntimeError):
    """Raised when a declared scratch deliverable cannot be preserved."""

def complete_task(
    conn: sqlite3.Connection, task_id: str, *, result: Optional[str] = None,
    summary: Optional[str] = None, metadata: Optional[dict] = None,
    created_cards: Optional[Iterable[str]] = None, expected_run_id: Optional[int] = None,
    verdict: Optional[str] = None,
    fire_lifecycle_hook: bool = True,
) -> bool:
    """``running|ready|blocked|review -> done``; records ``result``.

    ``ready`` is accepted for manual CLI completion, ``review`` for human
    approval; with no active run the handoff fields survive via
    :func:`_synthesize_ended_run`. ``summary`` (defaults to ``result``) and
    ``metadata`` land on the closing run for :func:`build_worker_context`.
    ``created_cards`` are verified first — a phantom id raises
    :class:`HallucinatedCardsError` after an auditable event; afterwards the
    prose is scanned for unresolvable ``t_<hex>`` refs (advisory event only).
    """
    task_before = _kb.get_task(conn, task_id)
    typed_phase: Optional[str] = None
    if task_before and task_before.lifecycle_contract:
        kind = task_before.lifecycle_contract.get("kind")
        if kind == "general":
            if verdict is not None:
                raise LifecycleEvidenceError("general tasks cannot carry a lifecycle verdict")
        elif kind == "review":
            typed_phase = "review"
            if verdict is None:
                raise LifecycleEvidenceError("review completion requires verdict=APPROVE or REQUEST_CHANGES")
            if str(verdict).strip().upper() not in {"APPROVE", "REQUEST_CHANGES"}:
                raise LifecycleEvidenceError("review completion verdict must be APPROVE or REQUEST_CHANGES")
        elif kind == "validation":
            typed_phase = "validation"
            if verdict is None:
                raise LifecycleEvidenceError("validation completion requires verdict=PASS or FAIL")
            if str(verdict).strip().upper() not in {"PASS", "FAIL"}:
                raise LifecycleEvidenceError("validation completion verdict must be PASS or FAIL")
        elif kind == "code":
            claimed_event = (
                _kb._latest_event(conn, task_id, "claimed", task_before.current_run_id)
                if task_before and task_before.current_run_id
                else None
            )
            claimed_source = _kb._json_dict(_kb._row_get(claimed_event, "payload")).get("source_status")
            completed_review = (
                task_before.status == "done"
                and task_before.lifecycle_contract.get("review_mode") == "same_card"
                and get_lifecycle_state(conn, task_id).get("review_verdict") is not None
            )
            typed_phase = (
                "review"
                if task_before.status == "review" or claimed_source == "review" or completed_review
                else "implementation"
            )
            if typed_phase == "review":
                if verdict is None:
                    raise LifecycleEvidenceError("same-card review completion requires a verdict")
                if str(verdict).strip().upper() not in {"APPROVE", "REQUEST_CHANGES"}:
                    raise LifecycleEvidenceError("same-card review verdict must be APPROVE or REQUEST_CHANGES")
            elif verdict is not None:
                raise LifecycleEvidenceError("implementation completion cannot carry a review verdict")
        else:
            if verdict is not None:
                raise LifecycleEvidenceError("unknown lifecycle contract cannot carry a verdict")
    elif verdict is not None:
        raise LifecycleEvidenceError("unclassified/general tasks cannot carry a lifecycle verdict")
    same_card_handoff = bool(
        task_before
        and task_before.lifecycle_contract
        and task_before.lifecycle_contract.get("kind") == "code"
        and task_before.lifecycle_contract.get("review_mode") == "same_card"
        and typed_phase == "implementation"
    )
    same_card_changes = bool(
        task_before
        and task_before.lifecycle_contract
        and task_before.lifecycle_contract.get("kind") == "code"
        and task_before.lifecycle_contract.get("review_mode") == "same_card"
        and typed_phase == "review"
        and str(verdict or "").strip().upper() == "REQUEST_CHANGES"
    )
    if task_before and task_before.status == "done" and typed_phase is not None:
        projection = get_lifecycle_state(conn, task_id)
        existing_verdict = (
            projection.get("review_verdict")
            if typed_phase == "review"
            else projection.get("validation_verdict")
            if typed_phase == "validation"
            else None
        )
        if typed_phase in {"review", "validation"}:
            normalized = str(verdict or "").strip().upper()
            if normalized == existing_verdict:
                return True
            raise LifecycleEvidenceError("verdict_conflict: terminal lifecycle verdict differs from stored evidence")
        supplied_head = (
            metadata.get("head_sha")
            if isinstance(metadata, dict)
            else None
        )
        if supplied_head and projection.get("head_sha") and supplied_head != projection["head_sha"]:
            raise LifecycleEvidenceError("verdict_conflict: terminal implementation head differs from stored evidence")
        return True
    acceptance_before = _capture_acceptance(conn, task_id)
    now = int(time.time())
    # Cheap pre-check; re-checked inside the txn to close the parent-reopen race.
    if not _parents_satisfied(conn, task_id):
        return False
    preflight_contract = _completion_contract_snapshot(conn, task_id)
    verified_cards = _gate_created_cards(conn, task_id, created_cards, summary or result)
    metadata = _merge_completion_prose_artifacts(
        conn, task_id, metadata, summary=summary, result=result,
    )
    try:
        metadata, _handoff = _prepare_completion_handoff(
            conn, task_id, metadata, phase=typed_phase,
        )
    except _kb.CompletionContractError as error:
        with _kb.write_txn(conn):
            _kb._append_event(
                conn,
                task_id,
                "completion_blocked_contract",
                {"reason": error.reason, "changed_files": error.changed_files},
                run_id=expected_run_id,
            )
        raise
    except _kb.HandoffValidationError as error:
        with _kb.write_txn(conn):
            _kb._append_event(
                conn, task_id, "completion_blocked_handoff",
                {"reason": error.reason}, run_id=expected_run_id,
            )
        raise
    if typed_phase is not None:
        try:
            metadata = _stamp_lifecycle_metadata(
                conn,
                task_id,
                metadata,
                phase=typed_phase,
                run_id=expected_run_id or (task_before.current_run_id if task_before else None),
                verdict=verdict,
            )
        except LifecycleEvidenceError as error:
            with _kb.write_txn(conn):
                _kb._append_event(
                    conn,
                    task_id,
                    "completion_blocked_lifecycle",
                    {"reason": str(error)},
                    run_id=expected_run_id,
                )
            raise
    handoff_summary = summary if summary is not None else result
    contract_err: Optional[_kb.CompletionContractError] = None
    run_id: Optional[int] = None
    with _kb.write_txn(conn):
        # Hard invariant even for human review approval: a parent may have
        # reopened while this task waited.
        if not _parents_satisfied(conn, task_id):
            return False
        if _completion_contract_snapshot(conn, task_id) != preflight_contract:
            contract_err = _kb.CompletionContractError(
                task_id,
                "effective goal revision or dependent review/test graph changed while preparing completion",
            )
            _kb._append_event(
                conn,
                task_id,
                "completion_blocked_contract",
                {"reason": contract_err.reason, "changed_files": contract_err.changed_files},
                run_id=expected_run_id,
            )
        else:
            prior_status = _kb._task_status(conn, task_id)
            implementation_routing = _implementation_routing(conn, task_id)
            target_status = (
                _landing_status_after_parents(conn, task_id)
                if same_card_changes
                else "review"
                if same_card_handoff
                else "done"
            )
            completed_at = None if target_status != "done" else now
            reviewer_profile = (
                task_before.lifecycle_contract.get("reviewer")
                if same_card_handoff and task_before and task_before.lifecycle_contract
                else implementation_routing.get("implementer")
                if same_card_changes
                else None
            )
            assignee_sql = ", assignee = ?" if reviewer_profile else ""
            sql = """
                    UPDATE tasks
                       SET status       = ?,
                           result       = ?,
                           completed_at = ?,
                           claim_lock   = NULL,
                           claim_expires= NULL,
                           worker_pid   = NULL,
                           block_kind   = NULL,
                           block_recurrences = 0
                """ + assignee_sql + """
                     WHERE id = ?
                       AND status IN ('running', 'ready', 'blocked', 'review')
                    """
            params: tuple = (
                target_status,
                result,
                completed_at,
                *((reviewer_profile,) if reviewer_profile else ()),
                task_id,
            )
            if expected_run_id is not None:
                sql += " AND current_run_id = ?"
                params = (*params, int(expected_run_id))
            if conn.execute(sql, params).rowcount != 1:
                return False
            if same_card_changes:
                conn.execute("UPDATE tasks SET candidate_run_id = NULL WHERE id = ?", (task_id,))
            if (
                typed_phase == "implementation"
                and isinstance(metadata, dict)
                and isinstance(metadata.get("lifecycle"), dict)
            ):
                lifecycle = metadata["lifecycle"]
                conn.execute(
                    "UPDATE tasks SET candidate_run_id = ? WHERE id = ?",
                    (lifecycle.get("candidate_run_id"), task_id),
                )
            if isinstance(metadata, dict):
                _stage_completion_artifacts(conn, task_id, metadata, now)
            run_outcome = (
                "review_requested"
                if same_card_handoff
                else "changes_requested"
                if same_card_changes
                else "completed"
            )
            run_id = _kb._end_run(
                conn,
                task_id,
                outcome=run_outcome,
                status=target_status,
                summary=handoff_summary,
                metadata=metadata,
            )
            # Never-claimed task: synthesize a run so the handoff fields survive.
            if run_id is None and (summary or metadata or result or prior_status == "review"):
                synth_summary, synth_metadata = handoff_summary, metadata
                if prior_status == "review" and not synth_summary and not synth_metadata:
                    synth_summary = _kb._REVIEW_APPROVED_NOTE
                    synth_metadata = {"source_status": "review", "approval": "manual"}
                run_id = _kb._synthesize_ended_run(
                    conn,
                    task_id,
                    outcome=run_outcome,
                    summary=synth_summary,
                    metadata=synth_metadata,
                )
            event_summary = handoff_summary
            if prior_status == "review" and not event_summary:
                event_summary = _kb._REVIEW_APPROVED_NOTE
            if same_card_changes:
                _kb._append_event(
                    conn,
                    task_id,
                    "changes_requested",
                    {
                        "reason": handoff_summary,
                        "implementer": reviewer_profile,
                        "reviewer": task_before.assignee if task_before else None,
                        "status": target_status,
                        "lifecycle": (
                            metadata.get("lifecycle")
                            if isinstance(metadata, dict)
                            else None
                        ),
                    },
                    run_id=run_id,
                )
            elif same_card_handoff:
                _kb._append_event(
                    conn,
                    task_id,
                    "review_requested",
                    {
                        "summary": _kb._first_line(event_summary, 400) or None,
                        "implementer": task_before.assignee if task_before else None,
                        "reviewer": reviewer_profile,
                        "lifecycle": (
                            metadata.get("lifecycle")
                            if isinstance(metadata, dict)
                            else None
                        ),
                    },
                    run_id=run_id,
                )
            else:
                _kb._append_event(
                    conn,
                    task_id,
                    "completed",
                    _completed_event_payload(result, event_summary, verified_cards, metadata),
                    run_id=run_id,
                )
    if contract_err is not None:
        raise contract_err
    _flag_phantom_prose_refs(conn, task_id, run_id, summary, result, verified_cards)
    # Success wipes the breaker counter (history stays on the event log).
    _kb._clear_failure_counter(conn, task_id)
    recompute_ready(conn)  # separate txn so children see ``done``
    _emit_acceptance_changes(conn, acceptance_before, source_task_id=task_id)
    _done_task = _kb.get_task(conn, task_id)
    if _done_task and _done_task.status == "done":
        _kb._cleanup_workspace(conn, task_id)
    if fire_lifecycle_hook and _done_task and _done_task.status == "done":
        _kb._fire_task_hook("kanban_task_completed", _done_task, task_id, run_id, summary=handoff_summary)
    return True

def _gate_created_cards(
    conn: sqlite3.Connection, task_id: str, created_cards: Optional[Iterable[str]], preview_text: Optional[str],
) -> list[str]:
    """Verify ``created_cards`` BEFORE the main write txn; returns the verified
    ids. A phantom id is recorded in its own tiny txn (auditable) then raised
    as :class:`HallucinatedCardsError` without touching task state."""
    if not created_cards:
        return []
    verified_cards, phantom_cards = _kb._verify_created_cards(conn, task_id, created_cards)
    if phantom_cards:
        with _kb.write_txn(conn):
            _kb._append_event(
                conn, task_id, "completion_blocked_hallucination",
                {
                    "phantom_cards": phantom_cards,
                    "verified_cards": verified_cards,
                    "summary_preview": _kb._first_line(preview_text, 200) or None,
                },
            )
        raise _kb.HallucinatedCardsError(phantom_cards, task_id)
    return verified_cards

def _stage_completion_artifacts(conn: sqlite3.Connection, task_id: str, metadata: dict, now: int) -> None:
    """Copy scratch artifacts to the attachments dir and record each as an attachment row."""
    _persist_scratch_completion_artifacts(conn, task_id, metadata)
    for stored_path in metadata.pop("_staged_artifacts", []):
        path = Path(stored_path)
        _insert_completion_attachment(
            conn, task_id, filename=path.name, stored_path=str(path),
            size=path.stat().st_size, created_at=now,
        )

def _completed_event_payload(
    result: Optional[str], event_summary: Optional[str], verified_cards: list[str], metadata: Any,
) -> dict:
    """``completed`` event payload: first summary line (400 chars) so gateway
    notifiers / dashboard WS render without a second round-trip; verified
    cards; and ``metadata["artifacts"]`` promoted so the notifier can upload
    them as native attachments without fetching the run row."""
    # Mirror CLI's _show_voice_status: include STT/TTS provider availability so the user can tell at a
    # glance *why* voice mode isn't working ("STT provider: MISSING ..." is the common case). ``record_key``
    # mirrors the configured ``voice.record_key`` so the TUI can both bind it (frontend
    # ``isVoiceToggleKey``) and display it in /voice status — previously the TUI hardcoded Ctrl+B and
    # ignored the config (#18994).
    payload: dict = {
        "result_len": len(result) if result else 0,
        "summary": _kb._first_line(event_summary, 400) or None,
    }
    if verified_cards:
        payload["verified_cards"] = verified_cards
    if isinstance(metadata, dict):
        md_artifacts = metadata.get("artifacts")
        if isinstance(md_artifacts, (list, tuple)):
            cleaned = [str(p).strip() for p in md_artifacts if isinstance(p, str) and str(p).strip()]
            if cleaned:
                payload["artifacts"] = cleaned
        payload.update(_kb._handoff_fields(metadata))
        if isinstance(metadata.get("lifecycle"), dict):
            payload["lifecycle"] = dict(metadata["lifecycle"])
    return payload

def _flag_phantom_prose_refs(
    conn: sqlite3.Connection, task_id: str, run_id: Optional[int],
    summary: Optional[str], result: Optional[str], verified_cards: list[str],
) -> None:
    """Advisory post-commit scan of summary+result for unresolvable ``t_<hex>``
    references; emits ``suspected_hallucinated_references`` in its own txn so
    the completion is already durable. Never blocks."""
    scan_text = " ".join(filter(None, [summary, result]))
    if not scan_text:
        return
    phantom_refs = [p for p in _kb._scan_prose_for_phantom_ids(conn, scan_text) if p not in set(verified_cards)]
    if phantom_refs:
        with _kb.write_txn(conn):
            _kb._append_event(
                conn, task_id, "suspected_hallucinated_references",
                {"phantom_refs": phantom_refs, "source": "completion_summary"}, run_id=run_id,
            )

def _merge_completion_prose_artifacts(
    conn: sqlite3.Connection, task_id: str, metadata: Optional[dict], *, summary: Optional[str],
    result: Optional[str],
) -> Optional[dict]:
    """Legacy workers named deliverables only by absolute path in prose; add
    those that exist under the scratch workspace to ``metadata["artifacts"]``
    before cleanup can erase them."""
    workspace = _kb._scratch_workspace(conn, task_id)
    if workspace is None:
        return metadata
    if not _kb._is_managed_scratch_path(workspace):
        return metadata
    text = "\n".join(part for part in (summary, result) if part)
    if not text:
        return metadata
    prefix = re.escape(str(workspace))
    discovered: list[str] = []
    for match in re.finditer(prefix + r"(?:[/\\][^\s`\"'<>]+)", text):
        raw = match.group(0).rstrip(".,;:!?)]}")
        candidate = Path(raw)
        if candidate.is_file():
            discovered.append(str(candidate))
    if not discovered:
        return metadata
    updated = dict(metadata) if isinstance(metadata, dict) else {}
    existing = updated.get("artifacts")
    merged = list(existing) if isinstance(existing, (list, tuple)) else []
    seen = {str(path) for path in merged}
    for path in discovered:
        if path not in seen:
            merged.append(path)
            seen.add(path)
    updated["artifacts"] = merged
    return updated

def _persist_scratch_completion_artifacts(
    conn: sqlite3.Connection, task_id: str, metadata: dict,
) -> None:
    """Copy scratch-workspace completion artifacts before cleanup removes them."""
    raw_artifacts = metadata.get("artifacts")
    if not isinstance(raw_artifacts, (list, tuple)):
        return

    workspace = _kb._scratch_workspace(conn, task_id)
    if workspace is None:
        return
    is_managed, board = _kb._managed_scratch_path_info(workspace)
    if not is_managed:
        return

    try:
        workspace_root = workspace.resolve()
    except OSError:
        return

    attachment_dir = _kb.task_attachments_dir(task_id, board=board)
    persisted: list[str] = []
    used_destinations: set[Path] = set()
    changed = False

    def _discard_copies() -> None:
        for copied in used_destinations:
            with contextlib.suppress(OSError):
                copied.unlink(missing_ok=True)
        with contextlib.suppress(OSError):
            attachment_dir.rmdir()

    for item in raw_artifacts:
        artifact = str(item).strip() if isinstance(item, str) else ""
        if not artifact:
            continue
        src = Path(artifact).expanduser()
        try:
            resolved_src = src.resolve()
        except OSError:
            persisted.append(artifact)
            continue

        if not resolved_src.is_relative_to(workspace_root):
            persisted.append(artifact)
            continue

        problem = None
        if not src.is_file():
            problem = f"declared scratch artifact is unavailable or not a regular file: {artifact}"
        elif resolved_src.stat().st_size > _kb.KANBAN_ATTACHMENT_MAX_BYTES:
            problem = (
                f"declared scratch artifact exceeds the "
                f"{_kb.KANBAN_ATTACHMENT_MAX_BYTES}-byte limit: {artifact}"
            )
        if problem:
            _discard_copies()
            raise ArtifactPreservationError(problem)

        dest: Optional[Path] = None
        try:
            attachment_dir.mkdir(parents=True, exist_ok=True)
            dest = _unique_attachment_path(attachment_dir, resolved_src.name, used_destinations)
            _copy_capped(resolved_src, dest, artifact)
        except Exception as exc:
            if dest is not None:
                with contextlib.suppress(OSError):
                    dest.unlink(missing_ok=True)
            _discard_copies()
            if isinstance(exc, ArtifactPreservationError):
                raise
            raise ArtifactPreservationError(
                f"could not preserve declared scratch artifact {artifact}: {exc}"
            ) from exc
        used_destinations.add(dest)
        persisted.append(str(dest.resolve()))
        changed = True

    if changed:
        metadata["artifacts"] = persisted
        metadata["_staged_artifacts"] = [
            path for path in persisted if path.startswith(str(attachment_dir.resolve()))
        ]

def _copy_capped(src: Path, dest: Path, artifact: str) -> None:
    """Chunked copy that aborts if the file grows past the attachment cap mid-copy."""
    with src.open("rb") as source_file, dest.open("xb") as destination_file:
        copied = 0
        while chunk := source_file.read(1024 * 1024):
            copied += len(chunk)
            if copied > _kb.KANBAN_ATTACHMENT_MAX_BYTES:
                raise ArtifactPreservationError(
                    f"declared scratch artifact grew beyond the size limit: {artifact}"
                )
            destination_file.write(chunk)

def _insert_completion_attachment(
    conn: sqlite3.Connection, task_id: str, *, filename: str, stored_path: str, size: int,
    created_at: int,
) -> None:
    """Record a worker-produced artifact in the existing attachment table."""
    conn.execute(
        "INSERT INTO task_attachments "
        "(task_id, filename, stored_path, content_type, size, uploaded_by, created_at) "
        "VALUES (?, ?, ?, NULL, ?, 'kanban_complete', ?)",
        (task_id, filename, stored_path, size, created_at),
    )
    _kb._append_event(conn, task_id, "attached", {"filename": filename, "size": size, "by": "kanban_complete"})

def _unique_attachment_path(directory: Path, filename: str, used: set[Path]) -> Path:
    """Return a non-conflicting path under ``directory`` for ``filename``."""
    safe_name = Path(filename).name or "artifact"
    stem, suffix = Path(safe_name).stem or "artifact", Path(safe_name).suffix
    candidate = directory / safe_name
    idx = 1
    while candidate in used or candidate.exists():
        candidate = directory / f"{stem}_{idx}{suffix}"
        idx += 1
    return candidate

def edit_completed_task_result(
    conn: sqlite3.Connection, task_id: str, *, result: str, summary: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> bool:
    """Backfill the user-visible result for an already completed task."""
    if isinstance(metadata, dict) and (
        "lifecycle" in metadata or "lifecycle_routing" in metadata
    ):
        raise LifecycleEvidenceError("completed result edits cannot change lifecycle evidence or routing")
    handoff_summary = summary if summary is not None else result
    with _kb.write_txn(conn):
        if _kb._task_status(conn, task_id) != "done":
            return False
        conn.execute("UPDATE tasks SET result = ? WHERE id = ?", (result, task_id))
        run = conn.execute(
            """
            SELECT id, metadata FROM task_runs
             WHERE task_id = ?
               AND outcome = 'completed'
             ORDER BY COALESCE(ended_at, started_at, 0) DESC, id DESC
             LIMIT 1
            """,
            (task_id,),
        ).fetchone()
        if run is None:
            run_id = _kb._synthesize_ended_run(
                conn, task_id, outcome="completed", summary=handoff_summary, metadata=metadata,
            )
        else:
            run_id = int(run["id"])
            conn.execute("UPDATE task_runs SET summary = ? WHERE id = ?", (handoff_summary, run_id))
            if metadata is not None:
                merged_metadata = _kb._json_dict(_kb._row_get(run, "metadata"))
                merged_metadata.update(metadata)
                conn.execute(
                    "UPDATE task_runs SET metadata = ? WHERE id = ?",
                    (json.dumps(merged_metadata, ensure_ascii=False), run_id),
                )
        _kb._append_event(
            conn, task_id, "edited",
            {
                "fields": ["result", "summary"] + (["metadata"] if metadata is not None else []),
                "result_len": len(result) if result else 0,
                "summary": _kb._first_line(handoff_summary, 400) or None,
            },
            run_id=run_id,
        )
    return True

def request_review(
    conn: sqlite3.Connection, task_id: str, *, summary: Optional[str] = None,
    metadata: Optional[dict] = None, reviewer: Optional[str] = None,
    expected_run_id: Optional[int] = None, force: bool = False, with_reason: bool = False,
):
    """``running``/``ready`` -> ``review``; never touches block recurrence accounting.

    Implementer and reviewer are recorded on the event so requested changes
    route back to the right profile; ``reviewer`` reassigns the task, and on
    re-review defaults to the latest ``changes_requested`` provenance. A live
    claim is only cleared with proof of ownership (``expected_run_id``) or
    ``force=True``. Returns ``bool``, or ``(ok, reason)`` with ``with_reason``.
    """

    def _ret(ok: bool, reason: Optional[str] = None):
        return (ok, reason) if with_reason else ok
    task_before = _kb.get_task(conn, task_id)
    typed_code = (
        task_before.lifecycle_contract
        if task_before and task_before.lifecycle_contract and task_before.lifecycle_contract.get("kind") == "code"
        else None
    )
    if (
        task_before
        and task_before.lifecycle_contract
        and task_before.lifecycle_contract.get("kind") in {"review", "validation"}
    ):
        return _ret(False, "typed review/validation cards complete with an explicit verdict")
    if typed_code and typed_code.get("review_mode") == "separate_card":
        return _ret(False, "separate-card code tasks complete implementation before dispatching the review card")
    if typed_code and reviewer is not None:
        requested_reviewer = _kb._canonical_assignee(reviewer)
        if requested_reviewer != typed_code.get("reviewer"):
            return _ret(False, "reviewer does not match the declared lifecycle reviewer")
    if typed_code:
        reviewer = typed_code.get("reviewer")
    acceptance_before = _capture_acceptance(conn, task_id)

    summary = _kb.redact_review_value(summary)
    metadata = _kb.redact_review_value(metadata)
    preflight_contract = _completion_contract_snapshot(conn, task_id)
    try:
        metadata, _handoff = _prepare_completion_handoff(conn, task_id, metadata)
    except _kb.CompletionContractError as error:
        with _kb.write_txn(conn):
            _kb._append_event(
                conn,
                task_id,
                "completion_blocked_contract",
                {"reason": error.reason, "changed_files": error.changed_files},
            )
        raise
    except _kb.HandoffValidationError as error:
        with _kb.write_txn(conn):
            _kb._append_event(
                conn,
                task_id,
                "completion_blocked_handoff",
                {"reason": error.reason},
            )
        return _ret(False, str(error))
    if typed_code:
        try:
            metadata = _stamp_lifecycle_metadata(
                conn,
                task_id,
                metadata,
                phase="implementation",
                run_id=expected_run_id or (task_before.current_run_id if task_before else None),
                verdict=None,
            )
        except LifecycleEvidenceError as error:
            with _kb.write_txn(conn):
                _kb._append_event(
                    conn, task_id, "completion_blocked_lifecycle", {"reason": str(error)},
                    run_id=expected_run_id,
                )
            return _ret(False, str(error))
    contract_err: Optional[_kb.CompletionContractError] = None
    with _kb.write_txn(conn):
        if _completion_contract_snapshot(conn, task_id) != preflight_contract:
            contract_err = _kb.CompletionContractError(
                task_id,
                "effective goal revision or dependent review/test graph changed while preparing review",
            )
            _kb._append_event(
                conn,
                task_id,
                "completion_blocked_contract",
                {"reason": contract_err.reason, "changed_files": contract_err.changed_files},
                run_id=expected_run_id,
            )
        else:
            if not _parents_satisfied(conn, task_id):
                return _ret(False, "parent dependencies are not satisfied")
            trow = conn.execute(
                "SELECT assignee, status, claim_lock, current_run_id "
                "FROM tasks WHERE id = ?", (task_id,),
            ).fetchone()
            if trow is None:
                return _ret(False, "task not found")
            # Refuse to clear a live worker's claim without proof of ownership
            # (expected_run_id) or an explicit human override (force=True).
            if (
                expected_run_id is None
                and not force
                and trow["status"] == "running"
                and trow["claim_lock"] is not None
            ):
                return _ret(
                    False, "task is running under a live claim; pass expected_run_id "
                    "(worker ownership) or force=True (explicit operator "
                    "override) instead of clearing the live run's claim",
                )
            implementer = trow["assignee"]
            if reviewer is None:
                reviewer = _prior_reviewer(conn, task_id)
                if reviewer is False:
                    return _ret(
                        False, "re-review has no durable reviewer provenance (the "
                        "latest changes_requested event is missing or "
                        "malformed); pass reviewer= explicitly",
                    )
            reviewer = _kb._canonical_assignee(reviewer)
            assignee_sql = ", assignee = ?" if reviewer is not None else ""
            run_guard = "" if expected_run_id is None else " AND current_run_id = ?"
            params: tuple[Any, ...] = (
                *(() if reviewer is None else (reviewer,)),
                task_id,
                *((int(expected_run_id),) if expected_run_id is not None else ()),
            )
            cur = conn.execute(
                """
                UPDATE tasks
                   SET status        = 'review',
                       claim_lock    = NULL,
                       claim_expires = NULL,
                       worker_pid    = NULL
                """ + assignee_sql + """
                 WHERE id = ?
                   AND status IN ('running', 'ready')
                """ + run_guard,
                params,
            )
            if cur.rowcount != 1:
                return _ret(
                    False, "task is not in running/ready (or expected_run_id did not match the current run)",
                )
            run_id = _kb._end_or_synthesize_run(
                conn, task_id, outcome="review_requested", status="review",
                summary=summary, metadata=metadata, synthesize=bool(summary or metadata),
            )
            lifecycle = metadata.get("lifecycle") if isinstance(metadata, dict) else None
            if isinstance(lifecycle, dict):
                conn.execute(
                    "UPDATE tasks SET candidate_run_id = ? WHERE id = ?",
                    (lifecycle.get("candidate_run_id"), task_id),
                )
            _kb._append_event(
                conn,
                task_id,
                "review_requested",
                {
                    "summary": _kb._first_line(summary, 400) or None,
                    "implementer": implementer,
                    "reviewer": reviewer,
                    "lifecycle": lifecycle,
                },
                run_id=run_id,
            )
    if contract_err is not None:
        raise contract_err
    _emit_acceptance_changes(conn, acceptance_before, source_task_id=task_id)
    return _ret(True)

def _prior_reviewer(conn: sqlite3.Connection, task_id: str):
    """Reviewer recorded by the latest ``changes_requested`` run's event.
    ``None`` = first review (no such run); ``False`` = a run exists but its
    provenance is missing/malformed."""
    changes_run = conn.execute(
        "SELECT id FROM task_runs "
        "WHERE task_id = ? AND outcome = 'changes_requested' "
        "ORDER BY id DESC LIMIT 1", (task_id,),
    ).fetchone()
    if changes_run is None:
        return None
    changes_event = _kb._latest_event(conn, task_id, "changes_requested", changes_run["id"])
    reviewer = _kb._json_dict(_kb._row_get(changes_event, "payload")).get("reviewer")
    return reviewer if isinstance(reviewer, str) and reviewer.strip() else False

def _nonblank_str(value: Any) -> Optional[str]:
    return value if isinstance(value, str) and value.strip() else None

def request_changes(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    reason: str,
    metadata: Optional[dict] = None,
    expected_run_id: Optional[int] = None,
) -> tuple[bool, Optional[str]]:
    """Close an active reviewer run (claimed from ``review``) and hand the task
    back to the implementer from the latest ``review_requested`` event, parent
    gating reapplied. Returns ``(ok, implementer | reason)``."""
    reason = str(_kb.redact_review_value(reason or "")).strip()
    if not reason:
        return False, "reason is required"
    metadata = _kb.redact_review_value(metadata)
    task_before = _kb.get_task(conn, task_id)
    contract_kind = (
        task_before.lifecycle_contract.get("kind")
        if task_before and task_before.lifecycle_contract
        else None
    )
    if contract_kind == "review":
        return False, "complete the review card with REQUEST_CHANGES, then rework the lifecycle graph"
    if contract_kind == "validation":
        return False, "complete the validation card with FAIL, then rework the lifecycle graph"
    acceptance_before = _capture_acceptance(conn, task_id)

    with _kb.write_txn(conn):
        task_row = conn.execute(
            "SELECT status, assignee, current_run_id FROM tasks WHERE id = ?", (task_id,),
        ).fetchone()
        if task_row is None:
            return False, "task not found"
        current_run_id = task_row["current_run_id"]
        if task_row["status"] != "running" or current_run_id is None:
            return False, "task is not in an active review run"
        if expected_run_id is not None and int(current_run_id) != int(expected_run_id):
            return False, "run_id mismatch"

        claimed_event = _kb._latest_event(conn, task_id, "claimed", current_run_id)
        claimed_payload = _kb._json_dict(_kb._row_get(claimed_event, "payload"))
        if claimed_payload.get("source_status") != "review":
            return False, "active run was not claimed from review"

        requested_event = _kb._latest_event(conn, task_id, "review_requested")
        if requested_event is None:
            return False, "no prior review_requested event"
        implementer = _nonblank_str(_kb._json_dict(requested_event["payload"]).get("implementer"))
        if implementer is None:
            return False, "review handoff has no valid implementer provenance"
        reviewer = _kb._canonical_assignee(_nonblank_str(task_row["assignee"]))
        lifecycle_for_run = None
        task_obj = _kb.get_task(conn, task_id)
        if (
            task_obj
            and task_obj.lifecycle_contract
            and task_obj.lifecycle_contract.get("kind") == "code"
        ):
            try:
                lifecycle_for_run = _stamp_lifecycle_metadata(
                    conn,
                    task_id,
                    metadata,
                    phase="review",
                    run_id=int(current_run_id),
                    verdict="REQUEST_CHANGES",
                )
            except LifecycleEvidenceError as error:
                return False, str(error)

        new_status = _landing_status_after_parents(conn, task_id)
        # consecutive_failures deliberately PRESERVED: a review transition is
        # not evidence the pathology cleared; only complete_task resets it.
        cur = conn.execute(
            """
            UPDATE tasks
               SET status = ?,
                   assignee = COALESCE(?, assignee),
                   claim_lock = NULL,
                   claim_expires = NULL,
                   worker_pid = NULL
             WHERE id = ? AND status = 'running' AND current_run_id = ?
            """,
            (new_status, implementer, task_id, int(current_run_id)),
        )
        if cur.rowcount != 1:
            return False, "task changed during review handoff"
        run_id = _kb._end_run(
            conn, task_id, outcome="changes_requested", status=new_status, summary=reason,
            metadata=lifecycle_for_run,
        )
        if contract_kind == "code":
            conn.execute("UPDATE tasks SET candidate_run_id = NULL WHERE id = ?", (task_id,))
        _kb._append_event(
            conn,
            task_id,
            "changes_requested",
            {
                "reason": reason,
                "implementer": implementer,
                "reviewer": reviewer,
                "status": new_status,
                "lifecycle": (
                    lifecycle_for_run.get("lifecycle")
                    if isinstance(lifecycle_for_run, dict)
                    else None
                ),
            },
            run_id=run_id,
        )
    _emit_acceptance_changes(conn, acceptance_before, source_task_id=task_id)
    return True, implementer

def promote_task(
    conn: sqlite3.Connection, task_id: str, *, actor: str, reason: Optional[str] = None,
    force: bool = False, dry_run: bool = False,
) -> tuple[bool, Optional[str]]:
    """Operator promotion ``todo``/``blocked`` -> ``ready`` with an audit event.
    Refused while a parent is unfinished unless ``force``; ``dry_run`` only
    validates. Returns ``(ok, reason)``."""
    cur_status = _kb._task_status(conn, task_id)
    if cur_status is None:
        return False, f"task {task_id} not found"

    if cur_status not in ("todo", "blocked"):
        return False, (
            f"task {task_id} is {cur_status!r}; promote only applies to "
            f"'todo' or 'blocked'"
        )

    dependency = evaluate_dependencies(conn, task_id)
    if not dependency["satisfied"]:
        blockers = dependency.get("blockers") or []
        detail = "; ".join(
            f"{item.get('parent_id')}: {item.get('code')}" for item in blockers
        )
        return False, f"unsatisfied lifecycle dependencies: {detail or 'unknown blocker'}"

    if dry_run:
        return True, None
    promoted_status = _kb._lifecycle_ready_status(conn, task_id)

    with _kb.write_txn(conn):
        upd = conn.execute(
            "UPDATE tasks SET status = ? "
            "WHERE id = ? AND status IN ('todo', 'blocked')", (promoted_status, task_id),
        )
        if upd.rowcount != 1:
            return False, f"task {task_id} status changed during promotion"
        _kb._append_event(
            conn, task_id, "promoted_manual", {"actor": actor, "reason": reason, "forced": force},
        )

    return True, None

def _landing_status_after_parents(conn: sqlite3.Connection, task_id: str) -> str:
    """``ready`` if every parent is terminal else ``todo`` — the re-gate shared by
    unblock/reopen so neither can spawn a child whose upstream is unfinished."""
    return _kb._lifecycle_ready_status(conn, task_id) if _parents_satisfied(conn, task_id) else "todo"

def unblock_task(conn: sqlite3.Connection, task_id: str) -> bool:
    """``blocked``/``scheduled`` -> its resumable phase (parent re-gated; ``review``
    when that is where it left off), closing any leaked run first."""
    now = int(time.time())
    with _kb.write_txn(conn):
        resume_status = (
            _kb._resume_status_from_events(conn, task_id)
            if _kb._task_status(conn, task_id) == "blocked"
            else "ready"
        )
        _kb._reclaim_dangling_run(
            conn, task_id, statuses=("blocked", "scheduled"), now=now,
            note="invariant recovery on unblock",
        )
        # Re-gate on parent completion before restoring the source phase.
        landing_status = _landing_status_after_parents(conn, task_id)
        new_status = (
            "review"
            if landing_status == "ready" and resume_status == "review"
            else landing_status
        )
        # ``block_kind``/``block_recurrences`` deliberately survive the unblock:
        # resetting them is the amnesia that let cron-unblock <-> re-block loop
        # unbounded; only complete_task clears them. ``consecutive_failures``
        # (the dispatcher's spawn/crash counter) IS reset — a deliberate unblock
        # is a fresh start for the retry budget.
        cur = conn.execute(
            "UPDATE tasks SET status = ?, current_run_id = NULL, "
            "consecutive_failures = 0, last_failure_error = NULL "
            "WHERE id = ? AND status IN ('blocked', 'scheduled')", (new_status, task_id),
        )
        if cur.rowcount != 1:
            return False
        _kb._append_event(
            conn, task_id, "unblocked",
            (
                {"status": new_status, "resume_status": resume_status}
                if new_status != "ready" or resume_status != "ready"
                else None
            ),
        )
        return True

def reopen_review_task(conn: sqlite3.Connection, task_id: str) -> bool:
    """``review`` -> ``ready``/``todo`` so the implementer re-runs on the new
    comments; restores the implementer from the ``review_requested`` event.
    Preserves ``consecutive_failures`` and the block loop counter (review is
    not a block; only :func:`complete_task` clears them)."""
    task = _kb.get_task(conn, task_id)
    if task and task.lifecycle_contract and task.lifecycle_contract.get("kind") != "general":
        raise LifecycleContractError(
            "typed lifecycle review must be completed or reworked through its declared evidence path"
        )
    now = int(time.time())
    with _kb.write_txn(conn):
        _kb._reclaim_dangling_run(
            conn, task_id, statuses=("review",), now=now,
            note="invariant recovery on review reopen",
        )
        new_status = _landing_status_after_parents(conn, task_id)
        review_event = _kb._latest_event(conn, task_id, "review_requested")
        handoff = _kb._json_dict(_kb._row_get(review_event, "payload"))
        implementer = _nonblank_str(handoff.get("implementer"))
        params: tuple[Any, ...] = (new_status, *((implementer,) if implementer else ()), task_id)
        cur = conn.execute(
            # consecutive_failures deliberately PRESERVED: review reopen is not
            # a success signal; only complete_task resets the breaker (#35072).
            "UPDATE tasks SET status = ?, current_run_id = NULL, "
            "claim_lock = NULL, claim_expires = NULL, worker_pid = NULL "
            + (", assignee = ?" if implementer else "")
            + " WHERE id = ? AND status = 'review'",
            params,
        )
        if cur.rowcount != 1:
            return False
        payload: dict[str, Any] = {"status": new_status}
        if implementer:
            payload["implementer"] = implementer
        _kb._append_event(
            conn, task_id, "review_reopened", payload if payload != {"status": "ready"} else None,
        )
        return True

def invalidate_descendants_for_parent_reopen(
    conn: sqlite3.Connection, task_id: str, *, author: str,
) -> dict[str, Any]:
    """THE done-reopen invalidation: every ``ready``/``review``/``running``/``done``
    descendant of a reopened ancestor is demoted to ``todo`` and re-gated.
    Every surface that reopens a done task (dashboard PATCH/drag) routes here.

    Composes under the caller's txn (``allow_nested=True``) so the flip and the
    retractions commit atomically. Each descendant gets a
    ``descendant_invalidated`` event, the legacy ``status`` event the live feed
    renders, and a comment naming the ancestor. Running descendants are closed
    ``reclaimed`` and their workers killed strictly post-commit (audit trail
    before death) — when composed, the CALLER must drain ``terminations``
    after its own commit. ``consecutive_failures`` resets (deliberate operator
    action), the opposite of :func:`reopen_review_task`.

    Returns ``{"invalidated": [{id, prior_status, new_status, resume_status}],
    "terminations": [(worker_pid, claim_lock)]}``.
    """
    caller_owns_txn = bool(conn.in_transaction)
    now = int(time.time())
    invalidated: list[dict[str, Any]] = []
    terminations: list[tuple[Optional[int], Optional[str]]] = []
    with _kb.write_txn(conn, allow_nested=True):
        rows = conn.execute(
            """
            WITH RECURSIVE descendants(id) AS (
                SELECT child_id FROM task_links WHERE parent_id = ?
                UNION
                SELECT l.child_id
                FROM task_links l
                JOIN descendants d ON d.id = l.parent_id
            )
            SELECT t.id, t.status, t.current_run_id, t.worker_pid, t.claim_lock
            FROM descendants d
            JOIN tasks t ON t.id = d.id
            ORDER BY t.id
            """,
            (task_id,),
        ).fetchall()
        for row in rows:
            previous_status = row["status"]
            if previous_status not in {"ready", "review", "running", "done"}:
                continue
            resume_status = "ready"
            run_id = None
            if previous_status == "review":
                resume_status = "review"
            elif previous_status == "running":
                resume_status = _kb._retry_status_for_run(conn, row["id"], row["current_run_id"])
                terminations.append((row["worker_pid"], row["claim_lock"]))
                run_id = _kb._end_run(
                    conn, row["id"], outcome="reclaimed", status="todo",
                    summary=f"ancestor {task_id} reopened",
                )
            # consecutive_failures = 0: deliberate operator reset — see
            # docstring for why this diverges from reopen_review_task.
            conn.execute(
                "UPDATE tasks SET status = 'todo', completed_at = NULL, "
                "claim_lock = NULL, claim_expires = NULL, worker_pid = NULL, "
                "current_run_id = NULL, candidate_run_id = NULL, consecutive_failures = 0 WHERE id = ?", (row["id"],),
            )
            entry = {
                "id": row["id"], "prior_status": previous_status,
                "new_status": "todo", "resume_status": resume_status,
            }
            _kb._append_event(
                conn, row["id"], "descendant_invalidated",
                {"ancestor": task_id, **{k: v for k, v in entry.items() if k != "id"}},
                run_id=run_id,
            )
            # Legacy 'status' event so existing live-feed consumers still see
            # the move without learning the new event kind.
            _kb._append_event(
                conn, row["id"], "status",
                {
                    "status": "todo", "reason": "ancestor_reopened", "parent": task_id,
                    "previous_status": previous_status, "resume_status": resume_status,
                },
                run_id=run_id,
            )
            _kb._insert_comment(
                conn, row["id"], author, f"Invalidated: ancestor {task_id} was reopened; "
                f"retracted from '{previous_status}' to 'todo' "
                f"(will resume via '{resume_status}').", now,
            )
            invalidated.append(entry)
    if not caller_owns_txn:
        # Standalone: committed above, audit trail durable, safe to kill now.
        # Composed calls leave this to the caller post-commit.
        for pid, claim_lock in terminations:
            _kb._terminate_reclaimed_worker(pid, claim_lock)
    return {"invalidated": invalidated, "terminations": terminations}

def _assert_no_lifecycle_role_references(
    conn: sqlite3.Connection, task_id: str,
) -> None:
    role_ids: list[str] = []
    for row in conn.execute(
        "SELECT id, lifecycle_contract FROM tasks "
        "WHERE id != ? AND lifecycle_contract IS NOT NULL",
        (task_id,),
    ):
        contract = safe_decode_contract(row["lifecycle_contract"])
        if (
            contract
            and contract.get("kind") in {"review", "validation"}
            and contract.get("candidate_task_id") == task_id
        ):
            role_ids.append(str(row["id"]))
    if role_ids:
        raise LifecycleContractError(
            f"cannot delete candidate task {task_id}: lifecycle role card(s) still reference it "
            f"({', '.join(sorted(role_ids))})"
        )

def delete_archived_task(conn: sqlite3.Connection, task_id: str) -> bool:
    """Hard-delete an ARCHIVED task (+ related rows); anything else must be
    archived first so data loss takes two deliberate actions."""
    with _kb.write_txn(conn):
        if _kb._task_status(conn, task_id) != "archived":
            return False
        _assert_no_lifecycle_role_references(conn, task_id)

        _kb._delete_task_relations(conn, task_id)
        cur = conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
        return cur.rowcount == 1

def delete_task(conn: sqlite3.Connection, task_id: str) -> bool:
    """Hard-delete a task and its related rows in one txn; False when not found."""
    with _kb.write_txn(conn):
        if not conn.execute("SELECT 1 FROM tasks WHERE id = ?", (task_id,)).fetchone():
            return False
        _assert_no_lifecycle_role_references(conn, task_id)
        cur = conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
        if cur.rowcount != 1:
            return False
        _kb._delete_task_relations(conn, task_id)
    recompute_ready(conn)
    return True

def build_worker_context(conn: sqlite3.Connection, task_id: str) -> str:
    """Everything a worker should read about its task: header, body,
    attachments, prior attempts, done-parent handoffs, the assignee's recent
    work, comments. Lists are tail-capped and fields char-capped
    (``_CTX_MAX_*``) so the prompt stays bounded on pathological boards."""
    task = _kb.get_task(conn, task_id)
    if not task:
        raise ValueError(f"unknown task {task_id}")
    # One clock reading so every relative age in this rendering agrees.
    now = int(time.time())
    lines: list[str] = []
    effective_goal = _kb.get_effective_goal(conn, task_id)
    lifecycle = get_lifecycle_state(conn, task_id)
    dependency = evaluate_dependencies(conn, task_id)
    routing_task_id = (
        lifecycle.get("candidate_task_id")
        if task.lifecycle_contract
        and task.lifecycle_contract.get("kind") in {"review", "validation"}
        else task_id
    )
    snapshot = {
        "contract": task.lifecycle_contract,
        "state": lifecycle,
        "dependencies": dependency,
        "routing": _implementation_routing(conn, str(routing_task_id)),
    }
    lines.append("## Lifecycle snapshot (read before prose)")
    lines.append("```json")
    lines.append(json.dumps(snapshot, ensure_ascii=False, sort_keys=True))
    lines.append("```")
    lines.append("")
    _kb._ctx_attachments(lines, _kb.list_attachments(conn, task_id))
    _kb._ctx_prior_attempts(lines, conn, task_id, now)
    _kb._ctx_header(lines, task, effective_goal=effective_goal)
    _kb._ctx_parent_results(lines, conn, task_id, now)
    _kb._ctx_role_history(lines, conn, task, now)
    _kb._ctx_comments(lines, _kb.list_comments(conn, task_id), now)
    return "\n".join(lines).rstrip() + "\n"

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

from hermes_cli.kanban_db_lazy import _kb
from hermes_cli.kanban_lifecycle import (
    LifecycleContractError,
    LifecycleEvidenceError,
    _latest_head,
    _forceable_dependency_override,
    decode_contract,
    encode_contract,
    evaluate_dependencies,
    get_lifecycle_state,
    infer_edge_requirement,
    lifecycle_metadata,
    safe_decode_contract,
    is_required_lifecycle_edge,
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
    review_phase = task_status in {"review", "done"} or (
        task_status == "blocked"
        and task_id is not None
        and _kb._resume_status_from_events(conn, task_id) == "review"
    )
    if kind == "code":
        if (
            actor
            and contract.get("review_mode") == "same_card"
            and phase != "implementation"
            and review_phase
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
            "SELECT assignee, lifecycle_contract FROM tasks "
            "WHERE lifecycle_contract IS NOT NULL"
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
        if candidate_contract.get("review_mode") != "separate_card":
            raise LifecycleContractError(
                "review cards require a separate-card code candidate"
            )
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
                _validate_lifecycle_role_identity(
                    conn, normalized_lifecycle, assignee, task_id=task_id,
                )
                # ACK-edge: the originating channel hears a child BLOCK, not just the fan-in.
                _kb._inherit_notify_subs(conn, task_id, parents, created_at=now)
            return task_id
        except sqlite3.IntegrityError:
            if attempt == 1:
                raise
    raise RuntimeError("unreachable")


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

    if not dependency["satisfied"] and (
        not force or not _forceable_dependency_override(dependency)
    ):
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
    required_children = [
        str(row["child_id"])
        for row in conn.execute(
            "SELECT child_id FROM task_links WHERE parent_id = ?", (task_id,),
        )
        if is_required_lifecycle_edge(conn, task_id, str(row["child_id"]))
    ]
    if required_children:
        raise LifecycleContractError(
            f"cannot delete lifecycle task {task_id}: required child role card(s) still "
            f"reference it ({', '.join(sorted(set(required_children)))})"
        )
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
from hermes_cli.kanban_db_lifecycle_claims import (
    _CLAIM_EXECUTION_FIELDS,
    _claim_and_open_run,
    _claim_execution_snapshot,
    _claim_rejected_for_snapshot,
    _claim_snapshot_matches,
    _landing_status_after_parents,
    _parents_satisfied,
    _runtime_claim_metadata,
    claim_review_task,
    claim_task,
    release_stale_claims,
)
from hermes_cli.kanban_db_lifecycle_evidence import (
    _capture_acceptance,
    _completion_contract_snapshot,
    _emit_acceptance_changes,
    _handoff_children,
    _implementation_routing,
    _latest_lifecycle_run_id,
    _lifecycle_observed_tasks,
    _parent_handoff_context,
    _parent_handoff_start_error,
    _prepare_completion_handoff,
    _record_parent_handoff_start_error,
    _stamp_lifecycle_metadata,
)
from hermes_cli.kanban_db_lifecycle_rework import (
    _nonblank_str,
    _prior_reviewer,
    _rework_fingerprint,
    _typed_rework_graph,
    request_changes,
    rework_review_graph,
)
from hermes_cli.kanban_db_lifecycle_completion import (
    ArtifactPreservationError,
    _completed_event_payload,
    _copy_capped,
    _flag_phantom_prose_refs,
    _gate_created_cards,
    _insert_completion_attachment,
    _merge_completion_prose_artifacts,
    _persist_scratch_completion_artifacts,
    _stage_completion_artifacts,
    _unique_attachment_path,
    complete_task,
    edit_completed_task_result,
    request_review,
)

from hermes_cli.kanban_db_lifecycle_update import update_task

"""Kanban lifecycle evidence, handoff, and acceptance helpers."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Optional

from hermes_cli import kanban_db as _kb
from hermes_cli.kanban_lifecycle import (
    LifecycleEvidenceError,
    _latest_head,
    get_lifecycle_state,
    lifecycle_metadata,
    safe_decode_contract,
)

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
    if phase in {"review", "validation"}:
        handoff_error = _parent_handoff_start_error(conn, task_id, phase=phase)
        if handoff_error is not None:
            raise _kb.HandoffValidationError(task_id, handoff_error["reason"])
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
    assigned_branch = str(task.branch_name or "").strip() or None
    branch = str(supplied.get("branch_name") or assigned_branch or "").strip() or None
    assigned_workspace = str(task.workspace_path or "").strip()
    workspace = str(supplied.get("workspace_path") or assigned_workspace).strip()
    if assigned_branch and branch != assigned_branch:
        raise _kb.HandoffValidationError(
            task_id, f"handoff branch {branch!r} does not match task branch {assigned_branch!r}",
        )
    if assigned_workspace and (
        Path(workspace).expanduser().resolve()
        != Path(assigned_workspace).expanduser().resolve()
    ):
        raise _kb.HandoffValidationError(
            task_id,
            f"handoff workspace {workspace!r} does not match task workspace {assigned_workspace!r}",
        )
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
    *,
    phase: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    task = _kb.get_task(conn, task_id)
    if task is None or task.workflow_template_id == "kanban_swarm_v1":
        return None
    contract = task.lifecycle_contract or {}
    is_role_card = contract.get("kind") in {"review", "validation"} or (
        not contract and str(task.assignee or "").strip().casefold() in _kb.HANDOFF_CHILD_ASSIGNEES
    )
    same_card_review = (
        contract.get("kind") == "code"
        and contract.get("review_mode") == "same_card"
        and (task.status == "review" or phase == "review")
    )
    if not is_role_card and not same_card_review:
        return None
    parent_ids = [task_id] if same_card_review else _kb.parent_ids(conn, task_id)
    for parent_id in parent_ids:
        parent = task if same_card_review else _kb.get_task(conn, parent_id)
        if parent is None or (
            not same_card_review and parent.status not in {"done", "archived"}
        ):
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


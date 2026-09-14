"""Kanban typed review rework and change-request lifecycle paths."""
from __future__ import annotations

import json
import os
import sqlite3
from typing import Any, Optional

from hermes_cli.kanban_db_lazy import _kb
from hermes_cli.kanban_db_lifecycle_claims import (
    _landing_status_after_parents,
    _parents_satisfied,
)
from hermes_cli.kanban_db_lifecycle_evidence import (
    _capture_acceptance,
    _emit_acceptance_changes,
    _implementation_routing,
    _stamp_lifecycle_metadata,
)
from hermes_cli.kanban_lifecycle import get_lifecycle_state, safe_decode_contract

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

    def _validated_evidence() -> tuple[dict[str, Any], Optional[dict[str, Any]], str]:
        review_state = get_lifecycle_state(conn, actual_reviewer_id)
        validation_state = (
            get_lifecycle_state(conn, actual_tester_id) if actual_tester_id else None
        )
        if review_state.get("review_verdict") != "REQUEST_CHANGES" and not (
            validation_state and validation_state.get("validation_verdict") == "FAIL"
        ):
            raise ValueError("typed rework requires REQUEST_CHANGES or FAIL evidence")
        review_diagnostics = set(review_state.get("diagnostics") or ())
        allowed_same_card_rework = (
            same_card
            and review_state.get("review_verdict") == "REQUEST_CHANGES"
            and review_diagnostics <= {"candidate_missing"}
        )
        if (
            (review_diagnostics and not allowed_same_card_rework)
            or (validation_state and validation_state.get("diagnostics"))
        ):
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
        return review_state, validation_state, rejected_head

    def _evidence_key(
        evidence: tuple[dict[str, Any], Optional[dict[str, Any]], str],
    ) -> tuple[Any, ...]:
        review_state, validation_state, rejected_head = evidence

        def _state_key(state: Optional[dict[str, Any]]) -> Optional[tuple[Any, ...]]:
            if state is None:
                return None
            return (
                state.get("head_sha"),
                state.get("candidate_run_id"),
                state.get("review_verdict"),
                state.get("validation_verdict"),
                state.get("acceptance"),
                tuple(state.get("diagnostics") or ()),
            )

        return _state_key(review_state), _state_key(validation_state), rejected_head

    evidence_before = _validated_evidence()
    rejected_head = evidence_before[2]

    implementation_assignee = None
    if same_card:
        implementation_assignee = _kb._canonical_assignee(
            _implementation_routing(conn, implementation_id).get("implementer")
        )
        if not implementation_assignee:
            raise ValueError("same-card rework requires original implementation routing")
        _kb._validate_lifecycle_role_identity(
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
        current_evidence = _validated_evidence()
        if _evidence_key(current_evidence) != _evidence_key(evidence_before):
            raise ValueError("typed rework evidence changed during update")
        acceptance_before = _capture_acceptance(conn, implementation_id)
        rejected_head = current_evidence[2]
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
        for row in rows.values():
            if row["status"] == "running" or row["claim_lock"] or row["current_run_id"] or row["worker_pid"]:
                raise ValueError(f"cannot rework graph while task {row['id']} is claimed")
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
        acceptance_before = _capture_acceptance(conn, task_id)
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

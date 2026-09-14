"""Kanban completion, review handoff, and artifact lifecycle paths."""
from __future__ import annotations

import contextlib
import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterable, Optional

from hermes_cli.kanban_db_lazy import _kb
from hermes_cli.kanban_db_lifecycle_claims import (
    _landing_status_after_parents,
    _parents_satisfied,
)
from hermes_cli.kanban_db_lifecycle_evidence import (
    _capture_acceptance,
    _completion_contract_snapshot,
    _emit_acceptance_changes,
    _implementation_routing,
    _latest_lifecycle_run_id,
)
from hermes_cli.kanban_db_lifecycle_rework import _prior_reviewer
from hermes_cli.kanban_lifecycle import LifecycleEvidenceError, get_lifecycle_state

class ArtifactPreservationError(RuntimeError):
    """Raised when a declared scratch deliverable cannot be preserved."""


def _finish_synthetic_run(
    conn: sqlite3.Connection,
    task_id: str,
    run_id: int,
    *,
    outcome: str,
    summary: Optional[str],
    metadata: Optional[dict],
) -> None:
    conn.execute(
        "UPDATE task_runs SET status = ?, outcome = ?, summary = ?, metadata = ? "
        "WHERE id = ? AND task_id = ?",
        (
            outcome,
            outcome,
            summary,
            _kb._json_or_null(metadata),
            run_id,
            task_id,
        ),
    )

def _completion_mode_general(
    conn: sqlite3.Connection,
    task_before: Any,
    task_id: str,
    verdict: Optional[str],
) -> Optional[str]:
    if verdict is not None:
        raise LifecycleEvidenceError("general tasks cannot carry a lifecycle verdict")
    return None


def _completion_mode_review(
    conn: sqlite3.Connection,
    task_before: Any,
    task_id: str,
    verdict: Optional[str],
) -> str:
    if verdict is None:
        raise LifecycleEvidenceError("review completion requires verdict=APPROVE or REQUEST_CHANGES")
    if str(verdict).strip().upper() not in {"APPROVE", "REQUEST_CHANGES"}:
        raise LifecycleEvidenceError("review completion verdict must be APPROVE or REQUEST_CHANGES")
    return "review"


def _completion_mode_validation(
    conn: sqlite3.Connection,
    task_before: Any,
    task_id: str,
    verdict: Optional[str],
) -> str:
    if verdict is None:
        raise LifecycleEvidenceError("validation completion requires verdict=PASS or FAIL")
    if str(verdict).strip().upper() not in {"PASS", "FAIL"}:
        raise LifecycleEvidenceError("validation completion verdict must be PASS or FAIL")
    return "validation"


def _completion_mode_code(
    conn: sqlite3.Connection,
    task_before: Any,
    task_id: str,
    verdict: Optional[str],
) -> str:
    claimed_event = (
        _kb._latest_event(conn, task_id, "claimed", task_before.current_run_id)
        if task_before.current_run_id
        else None
    )
    claimed_source = _kb._json_dict(_kb._row_get(claimed_event, "payload")).get("source_status")
    resumed_review = _kb._resume_status_from_events(conn, task_id) == "review"
    completed_review = (
        task_before.status == "done"
        and task_before.lifecycle_contract.get("review_mode") == "same_card"
        and get_lifecycle_state(conn, task_id).get("review_verdict") is not None
    )
    typed_phase = (
        "review"
        if (
            task_before.status == "review"
            or claimed_source == "review"
            or completed_review
            or resumed_review
        )
        else "implementation"
    )
    if typed_phase == "review":
        if verdict is None:
            raise LifecycleEvidenceError("same-card review completion requires a verdict")
        if str(verdict).strip().upper() not in {"APPROVE", "REQUEST_CHANGES"}:
            raise LifecycleEvidenceError("same-card review verdict must be APPROVE or REQUEST_CHANGES")
    elif verdict is not None:
        raise LifecycleEvidenceError("implementation completion cannot carry a review verdict")
    return typed_phase


_COMPLETION_MODE_HANDLERS = {
    "general": _completion_mode_general,
    "review": _completion_mode_review,
    "validation": _completion_mode_validation,
    "code": _completion_mode_code,
}

def _completion_modes(
    conn: sqlite3.Connection,
    task_before: Any,
    task_id: str,
    verdict: Optional[str],
) -> tuple[Optional[str], bool, bool]:
    typed_phase: Optional[str] = None
    contract = task_before.lifecycle_contract if task_before else None
    if contract:
        handler = _COMPLETION_MODE_HANDLERS.get(contract.get("kind"))
        if handler is not None:
            typed_phase = handler(conn, task_before, task_id, verdict)
        elif verdict is not None:
            raise LifecycleEvidenceError("unknown lifecycle contract cannot carry a verdict")
    elif verdict is not None:
        raise LifecycleEvidenceError("unclassified/general tasks cannot carry a lifecycle verdict")
    same_card_handoff = bool(
        task_before
        and contract
        and contract.get("kind") == "code"
        and contract.get("review_mode") == "same_card"
        and typed_phase == "implementation"
    )
    same_card_changes = bool(
        task_before
        and contract
        and contract.get("kind") == "code"
        and contract.get("review_mode") == "same_card"
        and typed_phase == "review"
        and str(verdict or "").strip().upper() == "REQUEST_CHANGES"
    )
    return typed_phase, same_card_handoff, same_card_changes
def _completion_handoff_kind(
    task_before: Any,
    *,
    typed_phase: Optional[str],
    same_card_handoff: bool,
    same_card_changes: bool,
    verdict: Optional[str],
) -> Optional[str]:
    if same_card_changes:
        return "changes_requested"
    if same_card_handoff:
        return "review_requested"
    contract = task_before.lifecycle_contract if task_before else None
    if not isinstance(contract, dict) or contract.get("kind") != "code":
        return None
    if typed_phase == "implementation" and contract.get("review_mode") == "separate_card":
        return "review_requested"
    if (
        typed_phase == "review"
        and contract.get("review_mode") == "same_card"
        and contract.get("validation_required")
        and str(verdict or "").strip().upper() == "APPROVE"
    ):
        return "validation_requested"
    return None



def _commit_completion(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    task_before: Any,
    metadata: Optional[dict],
    preflight_contract: tuple[Any, ...],
    verified_cards: list[str],
    typed_phase: Optional[str],
    same_card_handoff: bool,
    same_card_changes: bool,
    expected_run_id: Optional[int],
    now: int,
    handoff_summary: Optional[str],
    summary: Optional[str],
    result: Optional[str],
    verdict: Optional[str],
) -> tuple[bool, Optional[int], Optional[Exception]]:
    contract_err: Optional[_kb.CompletionContractError] = None
    run_id: Optional[int] = None
    synthetic_run_id: Optional[int] = None
    with _kb.write_txn(conn):
        # Hard invariant even for human review approval: a parent may have
        # reopened while this task waited.
        if not _parents_satisfied(conn, task_id):
            return False, None, None
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
            try:
                metadata, _handoff = _kb._prepare_completion_handoff(
                    conn, task_id, metadata, phase=typed_phase,
                )
            except _kb.CompletionContractError as error:
                _kb._append_event(
                    conn, task_id, "completion_blocked_contract",
                    {"reason": error.reason, "changed_files": error.changed_files},
                    run_id=expected_run_id,
                )
                return False, None, error
            except _kb.HandoffValidationError as error:
                _kb._append_event(
                    conn, task_id, "completion_blocked_handoff",
                    {"reason": error.reason}, run_id=expected_run_id,
                )
                return False, None, error
            handoff_kind = _completion_handoff_kind(
                task_before,
                typed_phase=typed_phase,
                same_card_handoff=same_card_handoff,
                same_card_changes=same_card_changes,
                verdict=verdict,
            )
            if typed_phase is not None:
                stamp_run_id = expected_run_id or _kb._current_run_id(conn, task_id)
                if stamp_run_id is None:
                    synthetic_run_id = _kb._synthesize_ended_run(
                        conn,
                        task_id,
                        outcome=handoff_kind or "completed",
                        summary=handoff_summary,
                        metadata=metadata,
                    )
                    stamp_run_id = synthetic_run_id
                should_stamp = synthetic_run_id is not None or not (
                    isinstance(metadata, dict)
                    and isinstance(metadata.get("lifecycle"), dict)
                )
                if should_stamp:
                    try:
                        metadata = _kb._stamp_lifecycle_metadata(
                            conn,
                            task_id,
                            metadata,
                            phase=typed_phase,
                            run_id=stamp_run_id,
                            verdict=verdict,
                        )
                    except LifecycleEvidenceError as error:
                        if synthetic_run_id is not None:
                            conn.execute(
                                "DELETE FROM task_runs WHERE id = ? AND task_id = ?",
                                (synthetic_run_id, task_id),
                            )
                        _kb._append_event(
                            conn,
                            task_id,
                            "completion_blocked_lifecycle",
                            {"reason": str(error)},
                            run_id=expected_run_id,
                        )
                        return False, None, error
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
            handoff_reviewer = (
                task_before.lifecycle_contract.get("reviewer")
                if (
                    handoff_kind in {"review_requested", "validation_requested"}
                    and task_before
                    and task_before.lifecycle_contract
                    and task_before.lifecycle_contract.get("kind") == "code"
                )
                else None
            )
            reviewer_profile = (
                handoff_reviewer
                if same_card_handoff
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
                if synthetic_run_id is not None:
                    conn.execute(
                        "DELETE FROM task_runs WHERE id = ? AND task_id = ?",
                        (synthetic_run_id, task_id),
                    )
                return False, None, None
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
            run_outcome = handoff_kind or "completed"
            run_id = _kb._end_run(
                conn,
                task_id,
                outcome=run_outcome,
                status=target_status,
                summary=handoff_summary,
                metadata=metadata,
            )
            if synthetic_run_id is not None:
                run_id = synthetic_run_id
                _finish_synthetic_run(
                    conn,
                    task_id,
                    run_id,
                    outcome=run_outcome,
                    summary=handoff_summary,
                    metadata=metadata,
                )
            # Never-claimed task: synthesize a run so the handoff fields survive.
            elif run_id is None and (summary or metadata or result or prior_status == "review"):
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
            if handoff_kind == "changes_requested":
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
            elif handoff_kind == "review_requested":
                _kb._append_event(
                    conn,
                    task_id,
                    "review_requested",
                    {
                        "summary": _kb._first_line(event_summary, 400) or None,
                        "implementer": task_before.assignee if task_before else None,
                        "reviewer": handoff_reviewer or reviewer_profile,
                        "lifecycle": (
                            metadata.get("lifecycle")
                            if isinstance(metadata, dict)
                            else None
                        ),
                    },
                    run_id=run_id,
                )
            elif handoff_kind == "validation_requested":
                _kb._append_event(
                    conn,
                    task_id,
                    "validation_requested",
                    {
                        "summary": _kb._first_line(event_summary, 400) or None,
                        "reviewer": handoff_reviewer,
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
    return True, run_id, None

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
    typed_phase, same_card_handoff, same_card_changes = _completion_modes(
        conn, task_before, task_id, verdict,
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
        metadata, _handoff = _kb._prepare_completion_handoff(
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
        stamp_run_id = expected_run_id or (
            task_before.current_run_id if task_before else None
        )
        if stamp_run_id is not None:
            try:
                metadata = _kb._stamp_lifecycle_metadata(
                    conn,
                    task_id,
                    metadata,
                    phase=typed_phase,
                    run_id=stamp_run_id,
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
    committed, run_id, boundary_error = _commit_completion(
        conn,
        task_id,
        task_before=task_before,
        metadata=metadata,
        preflight_contract=preflight_contract,
        verified_cards=verified_cards,
        typed_phase=typed_phase,
        same_card_handoff=same_card_handoff,
        same_card_changes=same_card_changes,
        expected_run_id=expected_run_id,
        now=now,
        handoff_summary=handoff_summary,
        summary=summary,
        result=result,
        verdict=verdict,
    )
    if boundary_error is not None:
        raise boundary_error
    if not committed:
        return False
    _flag_phantom_prose_refs(conn, task_id, run_id, summary, result, verified_cards)
    # Success wipes the breaker counter (history stays on the event log).
    _kb._clear_failure_counter(conn, task_id)
    _kb.recompute_ready(conn)  # separate txn so children see ``done``
    accepted_task_ids = _emit_acceptance_changes(
        conn, acceptance_before, source_task_id=task_id,
    )
    _done_task = _kb.get_task(conn, task_id)
    acceptance = (
        get_lifecycle_state(conn, task_id).get("acceptance")
        if _done_task and _done_task.lifecycle_contract
        else "accepted"
    )
    direct_hook_run_id = run_id
    if (
        _done_task
        and _done_task.lifecycle_contract
        and _done_task.lifecycle_contract.get("kind") == "code"
        and typed_phase == "review"
        and acceptance == "accepted"
    ):
        direct_hook_run_id = _latest_lifecycle_run_id(conn, task_id, "implementation")
    if _done_task and _done_task.status == "done":
        _kb._cleanup_workspace(conn, task_id)
        if (
            fire_lifecycle_hook
            and acceptance in {"accepted", "not_applicable"}
            and (typed_phase != "review" or direct_hook_run_id is not None)
        ):
            _kb._fire_task_hook(
                "kanban_task_completed",
                _done_task,
                task_id,
                direct_hook_run_id,
                summary=handoff_summary,
            )
    if fire_lifecycle_hook:
        for accepted_task_id in accepted_task_ids:
            if accepted_task_id == task_id:
                continue
            candidate_run_id = _latest_lifecycle_run_id(
                conn, accepted_task_id, "implementation",
            )
            if candidate_run_id is None:
                continue
            accepted_task = _kb.get_task(conn, accepted_task_id)
            _kb._fire_task_hook(
                "kanban_task_completed",
                accepted_task,
                accepted_task_id,
                candidate_run_id,
                summary=handoff_summary,
            )
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
        metadata, _handoff = _kb._prepare_completion_handoff(conn, task_id, metadata)
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
        stamp_run_id = expected_run_id or (
            task_before.current_run_id if task_before else None
        )
        if stamp_run_id is not None:
            try:
                metadata = _kb._stamp_lifecycle_metadata(
                    conn,
                    task_id,
                    metadata,
                    phase="implementation",
                    run_id=stamp_run_id,
                    verdict=None,
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
                return _ret(False, str(error))
    contract_err: Optional[_kb.CompletionContractError] = None
    synthetic_run_id: Optional[int] = None
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
            if typed_code:
                stamp_run_id = expected_run_id or trow["current_run_id"]
                if stamp_run_id is None:
                    synthetic_run_id = _kb._synthesize_ended_run(
                        conn,
                        task_id,
                        outcome="review_requested",
                        summary=summary,
                        metadata=metadata,
                    )
                    stamp_run_id = synthetic_run_id
                should_stamp = synthetic_run_id is not None or not (
                    isinstance(metadata, dict)
                    and isinstance(metadata.get("lifecycle"), dict)
                )
                if should_stamp:
                    try:
                        metadata = _kb._stamp_lifecycle_metadata(
                            conn,
                            task_id,
                            metadata,
                            phase="implementation",
                            run_id=stamp_run_id,
                            verdict=None,
                        )
                    except LifecycleEvidenceError as error:
                        if synthetic_run_id is not None:
                            conn.execute(
                                "DELETE FROM task_runs WHERE id = ? AND task_id = ?",
                                (synthetic_run_id, task_id),
                            )
                        _kb._append_event(
                            conn, task_id, "completion_blocked_lifecycle", {"reason": str(error)},
                            run_id=expected_run_id,
                        )
                        return _ret(False, str(error))

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
                if synthetic_run_id is not None:
                    conn.execute(
                        "DELETE FROM task_runs WHERE id = ? AND task_id = ?",
                        (synthetic_run_id, task_id),
                    )
                return _ret(
                    False, "task is not in running/ready (or expected_run_id did not match the current run)",
                )
            if synthetic_run_id is not None:
                run_id = synthetic_run_id
                _finish_synthetic_run(
                    conn,
                    task_id,
                    run_id,
                    outcome="review_requested",
                    summary=summary,
                    metadata=metadata,
                )
            else:
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


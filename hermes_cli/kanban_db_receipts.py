"""Runtime-identity receipts: the spawn record owns identity, completions agree.

``runtime_identity`` is written when the dispatcher spawns a worker and mirrored in
that run's ``spawned`` event. Completion metadata is author-supplied, so it may only
corroborate what the receipt already says. This module holds the rule once, for both
merge paths that can write a run's metadata: closing the run and editing a completed
task.
"""

from __future__ import annotations

import json
import re
from typing import Optional

from hermes_cli.kanban_lifecycle import LifecycleEvidenceError

_GENERATION_DIGEST = re.compile(r"[0-9a-f]{64}")


def _identity_disagreement(
    receipt: dict, author,
) -> Optional[tuple[str, str]]:
    """``(verdict, field)`` when an author copy may not stand, else ``None``.

    ``verdict`` is ``author_malformed`` when the copy is not an object, or a field is
    empty or not a 64-character digest, and ``conflict`` when both sides are
    well-formed and differ. One rule, because the writer refuses with it and the
    reader reports it.
    """
    if author is None:
        return None
    if not isinstance(author, dict):
        return ("author_malformed", "runtime_identity")
    for field in ("generation", "code_sha"):
        if field not in author:
            continue
        expected, given = receipt.get(field), author.get(field)
        if not expected:
            continue
        if given in (None, ""):
            return ("author_malformed", field)
        if field == "generation" and not _GENERATION_DIGEST.fullmatch(str(given)):
            return ("author_malformed", field)
        if str(expected) != str(given):
            return ("conflict", field)
    return None


def receipt_runtime_identity(
    original: dict, supplied: Optional[dict], *, run_id: int,
) -> Optional[dict]:
    """The launch receipt's ``runtime_identity``, or ``None`` when the run has none.

    An author copy may only agree with the receipt. A value that is not an object, a
    generation that is not a 64-character digest, an explicitly empty identity field,
    or a differing ``generation``/``code_sha`` raises a typed refusal naming the field
    and the authoritative source — so a disagreement here can never read as a
    provenance or head verdict, and nothing is normalized on the way through.
    """
    receipt = original.get("runtime_identity")
    if not isinstance(receipt, dict) or not receipt:
        return None
    if not isinstance(supplied, dict) or "runtime_identity" not in supplied:
        return receipt
    author = supplied.get("runtime_identity")
    disagreement = _identity_disagreement(receipt, author)
    if disagreement is not None:
        verdict, field = disagreement
        if field == "runtime_identity":
            raise LifecycleEvidenceError(
                f"run {run_id} runtime_identity must be an object; the launch receipt "
                "recorded in task_runs.metadata is authoritative"
            )
        given = author.get(field) if isinstance(author, dict) else None
        if given in (None, ""):
            detail = f"runtime_identity.{field} is empty"
        elif verdict == "conflict":
            detail = (
                f"runtime_identity.{field} does not match the launch receipt; "
                "completion metadata may corroborate it, not replace it"
            )
        else:
            detail = f"runtime_identity.{field} is not a 64-character digest"
        raise LifecycleEvidenceError(
            f"run {run_id} {detail}; the launch receipt recorded in task_runs.metadata "
            "is authoritative"
        )
    return receipt


def candidate_identity(conn, task_id: str, run_id: Optional[int] = None) -> Optional[dict]:
    """A run's immutable launch receipt beside its author-side copy, with a verdict.

    Reconciliation reads the run's ``spawned`` event, which the dispatcher writes
    once and nothing rewrites. The sealed payload's ``generation.json`` is a cache
    the sweeper may reclaim, so it is not evidence and its absence is not a failure:
    whoever reconciles a candidate compares against this receipt instead.

    Verdicts: ``verified`` (no author copy, or it agrees), ``author_malformed`` (the
    author copy is not a 64-character digest), ``conflict`` (both are digests and
    they differ), ``receipt_missing`` (the run has no spawn receipt).
    """
    def _receipt_for(rid: int) -> Optional[dict]:
        # A run can carry more than one ``spawned`` event (the pid is recorded
        # separately from the identity), so take the newest that actually holds one.
        rows = conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'spawned' "
            "AND run_id = ? ORDER BY id DESC",
            (task_id, rid),
        ).fetchall()
        for row in rows:
            try:
                payload = json.loads(row[0]) if row and row[0] else {}
            except (TypeError, ValueError):
                continue  # a damaged event is no receipt; never fail the reader
            identity = payload.get("runtime_identity") if isinstance(payload, dict) else None
            if isinstance(identity, dict) and identity:
                return identity
        return None

    if run_id is None:
        # The attempt to describe is the run the card would report: the active run if
        # there is one, else the newest run row. Selecting by "has a receipt" would
        # skip an attempt that died before spawning and describe an older one instead.
        row = conn.execute(
            "SELECT current_run_id FROM tasks WHERE id = ?", (task_id,),
        ).fetchone()
        run_id = int(row[0]) if row and row[0] is not None else None
        if run_id is None:
            row = conn.execute(
                "SELECT MAX(id) FROM task_runs WHERE task_id = ?", (task_id,),
            ).fetchone()
            run_id = int(row[0]) if row and row[0] is not None else None
        if run_id is None:
            return None
    receipt = _receipt_for(int(run_id))
    if receipt is None:
        return {"run_id": int(run_id), "verdict": "receipt_missing"}
    row = conn.execute(
        "SELECT metadata FROM task_runs WHERE id = ? AND task_id = ?", (run_id, task_id),
    ).fetchone()
    try:
        metadata = json.loads(row[0]) if row and row[0] else {}
    except (TypeError, ValueError):
        metadata = None
    if not isinstance(metadata, dict):
        # A record that will not decode is present and untrustworthy, never absent:
        # reporting it as "no copy" would verify exactly the corruption this exists
        # to surface.
        author = "corrupt"
    else:
        author = metadata.get("runtime_identity")
    disagreement = _identity_disagreement(receipt, author)
    verdict = disagreement[0] if disagreement else "verified"
    given = author.get("generation") if isinstance(author, dict) else None
    expected = receipt.get("generation")
    return {
        "run_id": run_id,
        "verdict": verdict,
        "receipt_generation": expected,
        "author_generation": given,
        "code_sha": receipt.get("code_sha"),
    }


def rebind_candidate(
    conn, task_id: str, *, run_id: int, head_sha: str, reason: str,
    author: Optional[str] = None,
) -> dict:
    """Bind a review candidate back to an existing run, on proof and with an audit row.

    Recovery for a card whose candidate binding was cleared by an out-of-phase claim.
    It re-binds the *existing* run rather than re-running the work, and it refuses
    anything it cannot prove: the run must belong to this card, the requested head
    must be the head that run recorded, and the run must carry a launch receipt
    (a malformed author copy is tolerated; a missing or conflicting one is not).
    Nothing here rewrites history or infers acceptance.
    """
    from hermes_cli import kanban_db as _db

    run = conn.execute(
        "SELECT id, status, outcome, metadata FROM task_runs WHERE id = ? AND task_id = ?",
        (run_id, task_id),
    ).fetchone()
    if run is None:
        raise LifecycleEvidenceError(f"run {run_id} does not belong to {task_id}")
    try:
        metadata = json.loads(run["metadata"]) if run["metadata"] else {}
    except (TypeError, ValueError):
        metadata = {}
    # Only a finished implementation handoff is a candidate: an active, blocked or
    # failed attempt has produced nothing for a review to cover.
    if str(run["status"]) == "running" or str(run["outcome"] or "") not in {
        "review_requested", "completed",
    }:
        raise LifecycleEvidenceError(
            f"run {run_id} is {run['status']}/{run['outcome'] or 'no outcome'}; "
            "only a finished implementation handoff can be bound as a candidate"
        )
    recorded = metadata.get("head_sha") if isinstance(metadata, dict) else None
    if not recorded:
        raise LifecycleEvidenceError(
            f"run {run_id} recorded no candidate head; nothing to bind"
        )
    if str(recorded) != str(head_sha):
        raise LifecycleEvidenceError(
            f"run {run_id} recorded head {recorded}, not the requested {head_sha}; "
            "a rebind may restore an existing candidate, never redefine it"
        )
    identity = candidate_identity(conn, task_id, run_id)
    if not identity or identity.get("verdict") == "receipt_missing":
        raise LifecycleEvidenceError(
            f"run {run_id} has no launch receipt; it cannot be attested as a candidate"
        )
    if identity.get("verdict") == "conflict":
        raise LifecycleEvidenceError(
            f"run {run_id} runtime identity conflicts with its launch receipt; "
            "resolve the provenance before rebinding"
        )
    # Validation and write in one transaction: the fence that refuses delegated-child
    # board mutations is enforced here, and the candidate read cannot go stale
    # between the check and the update.
    with _db.write_txn(conn):
        bound = conn.execute(
            "SELECT candidate_run_id FROM tasks WHERE id = ?", (task_id,),
        ).fetchone()
        current = bound[0] if bound else None
        if current is not None and int(current) != int(run_id):
            raise LifecycleEvidenceError(
                f"{task_id} is already bound to candidate run {current}; recovering one "
                "candidate must never replace another"
            )
        updated = conn.execute(
            "UPDATE tasks SET candidate_run_id = ? WHERE id = ? AND "
            "(candidate_run_id IS NULL OR candidate_run_id = ?)",
            (run_id, task_id, run_id),
        )
        if updated.rowcount != 1:
            raise LifecycleEvidenceError(
                f"{task_id} candidate binding changed during the rebind; nothing written"
            )
        payload = {
            "candidate_run_id": run_id,
            "head_sha": str(head_sha),
            "receipt_generation": identity.get("receipt_generation"),
            "author_verdict": identity.get("verdict"),
            "reason": reason,
            "author": author,
        }
        _db._append_event(conn, task_id, "candidate_rebound", payload)
    return payload


def restored_candidate_run(conn, task_id: str, metadata: Optional[dict]) -> Optional[int]:
    """The card's already-bound candidate run, when it can stand as this handoff's candidate.

    A card whose candidate was restored (``rebind-candidate``) must keep that run:
    synthesizing a handoff row here would replace the launch that actually holds the
    receipt, leaving review with a candidate nothing can attest. Returns ``None``
    whenever the binding cannot stand - wrong phase, unfinished, or a different head -
    so the caller falls back to its normal path.
    """
    row = conn.execute(
        "SELECT candidate_run_id FROM tasks WHERE id = ?", (task_id,),
    ).fetchone()
    run_id = row[0] if row and row[0] is not None else None
    if run_id is None:
        return None
    run = conn.execute(
        "SELECT status, outcome, metadata FROM task_runs WHERE id = ? AND task_id = ?",
        (int(run_id), task_id),
    ).fetchone()
    if run is None:
        return None
    if str(run["status"]) == "running" or str(run["outcome"] or "") not in {
        "review_requested", "completed",
    }:
        return None
    try:
        run_metadata = json.loads(run["metadata"]) if run["metadata"] else {}
    except (TypeError, ValueError):
        return None
    run_head = run_metadata.get("head_sha") if isinstance(run_metadata, dict) else None
    handed_head = metadata.get("head_sha") if isinstance(metadata, dict) else None
    if run_head and handed_head and str(run_head) != str(handed_head):
        return None
    if not run_head and not handed_head:
        return None
    identity = candidate_identity(conn, task_id, int(run_id))
    if not identity or identity.get("verdict") == "receipt_missing":
        return None
    return int(run_id)

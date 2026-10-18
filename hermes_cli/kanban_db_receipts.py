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
        candidates = [
            int(row[0]) for row in conn.execute(
                "SELECT DISTINCT run_id FROM task_events WHERE task_id = ? "
                "AND kind = 'spawned' AND run_id IS NOT NULL ORDER BY id DESC",
                (task_id,),
            )
        ]
        if not candidates:
            return None
        # Prefer the newest run that actually recorded an identity; a run whose only
        # spawn record is the pid still reports ``receipt_missing`` rather than
        # silently answering for an older run.
        run_id = next(
            (rid for rid in candidates if _receipt_for(rid) is not None), candidates[0],
        )
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

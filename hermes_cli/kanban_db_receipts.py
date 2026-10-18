"""Runtime-identity receipts: the spawn record owns identity, completions agree.

``runtime_identity`` is written when the dispatcher spawns a worker and mirrored in
that run's ``spawned`` event. Completion metadata is author-supplied, so it may only
corroborate what the receipt already says. This module holds the rule once, for both
merge paths that can write a run's metadata: closing the run and editing a completed
task.
"""

from __future__ import annotations

import re
from typing import Optional

from hermes_cli.kanban_lifecycle import LifecycleEvidenceError

_GENERATION_DIGEST = re.compile(r"[0-9a-f]{64}")


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
    if author is None:
        return receipt  # explicitly absent: the receipt stands unchallenged
    if not isinstance(author, dict):
        raise LifecycleEvidenceError(
            f"run {run_id} runtime_identity must be an object; the launch receipt "
            "recorded in task_runs.metadata is authoritative"
        )
    generation = author.get("generation")
    if generation is not None and not _GENERATION_DIGEST.fullmatch(str(generation)):
        raise LifecycleEvidenceError(
            f"run {run_id} runtime_identity.generation is not a 64-character digest; "
            "the launch receipt recorded in task_runs.metadata is authoritative"
        )
    for field in ("generation", "code_sha"):
        expected = receipt.get(field)
        if not expected or field not in author:
            continue
        given = author.get(field)
        if given in (None, ""):
            raise LifecycleEvidenceError(
                f"run {run_id} runtime_identity.{field} is empty; the launch receipt "
                "recorded in task_runs.metadata is authoritative"
            )
        if str(expected) != str(given):
            raise LifecycleEvidenceError(
                f"run {run_id} runtime_identity.{field} does not match the launch "
                "receipt; completion metadata may corroborate it, not replace it"
            )
    return receipt

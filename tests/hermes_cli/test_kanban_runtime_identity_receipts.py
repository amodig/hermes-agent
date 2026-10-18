"""The launch receipt owns ``runtime_identity``; completion metadata may only agree.

The dispatcher writes the identity when it spawns the worker and mirrors it in the
run's ``spawned`` event. A completion written from the worker's own metadata must
not be able to redefine that identity, and a malformed value has to surface as a
typed refusal naming the field — never as something that reads like a provenance or
head verdict.
"""

from __future__ import annotations

import pytest

from hermes_cli import kanban_db as _kb  # noqa: F401  facade first: breaks the import cycle
from hermes_cli.kanban_db_lifecycle_completion import _receipt_runtime_identity
from hermes_cli.kanban_lifecycle import LifecycleEvidenceError

RECEIPT = {"code_sha": "a" * 40, "generation": "b" * 64}


def test_completion_metadata_cannot_redefine_the_launch_receipt():
    original = {"runtime_identity": RECEIPT}

    # No author copy, or one that agrees, keeps the receipt in place.
    assert _receipt_runtime_identity(original, {}, run_id=7) == RECEIPT
    assert (
        _receipt_runtime_identity(original, {"runtime_identity": dict(RECEIPT)}, run_id=7)
        == RECEIPT
    )

    # A truncated digest is a typed refusal naming the field.
    with pytest.raises(LifecycleEvidenceError, match="64-character digest"):
        _receipt_runtime_identity(
            original,
            {"runtime_identity": {**RECEIPT, "generation": "b" * 62}},
            run_id=7,
        )

    # A different digest conflicts with the receipt instead of replacing it.
    with pytest.raises(LifecycleEvidenceError, match="does not match the launch receipt"):
        _receipt_runtime_identity(
            original,
            {"runtime_identity": {**RECEIPT, "generation": "c" * 64}},
            run_id=7,
        )

    # A run with no receipt has nothing to defend.
    assert _receipt_runtime_identity({}, {"runtime_identity": RECEIPT}, run_id=7) is None

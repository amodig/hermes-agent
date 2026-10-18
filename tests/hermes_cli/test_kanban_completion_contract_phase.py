"""Plan-only goal prose constrains implementation, never a review or validation handoff.

A review card inherits the patch under review as its evidence, so a goal whose body
says "No implementation" - a prohibition on *editing* implementation - used to be
read as a plan-only objective and refused the reviewer's own completion (#68).
"""

from __future__ import annotations

from hermes_cli import kanban_db as kb

PLAN_ONLY_GOAL = {
    "version": 105,
    "title": "Independently review withdrawal forward-port",
    "body": "Review the candidate. No implementation in this card; report findings.",
}
CHANGED = ["collection-app/lib/withdrawal.ts"]
CHILDREN = [("t_245a5526", "validation")]


def _reason(**kwargs):
    return kb._completion_contract_reason(
        PLAN_ONLY_GOAL, CHANGED, CHILDREN, lifecycle_contract=kwargs.pop("contract", {}),
        **kwargs,
    )


def test_plan_only_prose_blocks_implementation_but_not_review_or_validation():
    # A genuine plan-only objective still blocks an implementation handoff that
    # carries a patch.
    assert _reason() is not None

    # The same prose does not reclassify a review handoff: the patch is inherited
    # evidence, not authored work.
    assert _reason(phase="review") is None
    assert _reason(phase="validation") is None
    assert _reason(contract={"kind": "review"}) is None
    assert _reason(contract={"kind": "validation"}) is None

    # A typed implementation contract keeps its existing authority over prose.
    assert _reason(contract={"kind": "code"}) is None

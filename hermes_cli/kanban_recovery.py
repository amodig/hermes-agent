"""Kanban CLI recovery verbs: ``hermes kanban supersede-candidate …``.

Recovers a *quarantined, never-reviewed* moved-head candidate: the recorded
implementation handoff of an unreviewed candidate whose branch head then moved.
``kanban_db.supersede_candidate`` owns every precondition and mutation; this
module owns only the CLI seam (flags -> call -> human/``--json`` output), so
``hermes_cli.kanban`` stays a facade with dispatch and public entry points.

Sibling of ``kanban_boards``/``kanban_ops``; the facade late-imports it, and it
late-imports ``hermes_cli.kanban`` back inside the handler (no module-level
cycle) so the facade's ``_profile_author`` stays the single identity seam.
"""

from __future__ import annotations

import argparse
import json
import os

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli.kanban_output import _err, _json_out


def _cmd_supersede_candidate(args: argparse.Namespace) -> int:
    """``supersede-candidate`` — orchestrator-only; requires an explicit ``--board``."""
    from hermes_cli.kanban import _profile_author

    if os.environ.get("HERMES_KANBAN_TASK"):
        return _err(
            "kanban supersede-candidate is orchestrator-only; workers must hand off "
            "their assigned task"
        )
    board = str(getattr(args, "board", "") or "").strip()
    if not board:
        return _err("kanban supersede-candidate requires an explicit --board <slug>", 2)
    reason = " ".join(getattr(args, "reason", ()) or ())
    reason = str(kb.redact_review_value(reason)).strip()
    if not reason:
        return _err("kanban supersede-candidate requires a non-empty --reason", 2)
    try:
        expected_versions = json.loads(getattr(args, "expected_versions", None) or "")
    except (TypeError, ValueError):
        return _err(
            "kanban supersede-candidate requires --expected-versions as a JSON object "
            'of task ids to versions, e.g. \'{"T-1":7,"T-2":4}\'',
            2,
        )
    with kbc.connect_closing() as conn:
        try:
            outcome = kb.supersede_candidate(
                conn,
                args.implementation_id,
                expected_versions=expected_versions,
                expected_candidate_run_id=args.expected_candidate_run_id,
                expected_head_sha=args.expected_head_sha,
                replacement_head_sha=args.replacement_head_sha,
                reason=reason,
                author=_profile_author(),
            )
        except (ValueError, kb.LifecycleEvidenceError) as exc:
            return _err(f"kanban: supersede-candidate: {exc}")
    if _json_out(args, outcome):
        return 0
    print(
        f"Superseded {args.implementation_id} candidate "
        f"{outcome['superseded_candidate_run_id']} "
        f"{outcome['superseded_head_sha'][:12]} -> {outcome['replacement_head_sha'][:12]}; "
        f"status {outcome['status']} (invalidated {len(outcome['invalidated'])} card(s))"
    )
    return 0

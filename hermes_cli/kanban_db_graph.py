"""Task graph initialization and atomic decomposition persistence."""
from __future__ import annotations

import sqlite3
import time
from typing import Any, Optional

from hermes_cli.kanban_lifecycle import (
    LifecycleContractError, decode_contract, encode_contract, infer_edge_requirement,
)

def inherit_creator_origin(
    conn: sqlite3.Connection, task_id: str, creator_task_id: Optional[str], *,
    created_at: int,
) -> None:
    """Copy durable origin inside creation's transaction, never adding dependencies."""
    if not creator_task_id:
        return
    from hermes_cli.kanban_db import _inherit_notify_subs

    conn.execute(
        "UPDATE tasks SET session_id = COALESCE(session_id, "
        "(SELECT session_id FROM tasks WHERE id = ?)) WHERE id = ?",
        (creator_task_id, task_id),
    )
    _inherit_notify_subs(conn, task_id, (creator_task_id,), created_at=created_at)


def initial_task_state(
    conn: sqlite3.Connection, parents: tuple[str, ...], initial_status: str,
    triage: bool, tenant: Optional[str],
) -> tuple[str, Optional[str]]:
    """Resolve state and tenant under the creator's write transaction.

    Parent order breaks ties in this soft namespace; explicit tenant wins.
    Validate parents even for parked tasks so links never dangle.
    """
    rows = {}
    if parents:
        rows = {row["id"]: row for row in conn.execute(
            "SELECT id, status, tenant FROM tasks WHERE id IN "
            "(" + ",".join("?" * len(parents)) + ")", parents,
        )}
        missing = [pid for pid in parents if pid not in rows]
        if missing:
            raise ValueError(f"unknown parent task(s): {', '.join(missing)}")
        if tenant is None:
            tenant = next((rows[pid]["tenant"] for pid in parents if rows[pid]["tenant"]), None)
    if initial_status == "blocked":
        return "blocked", tenant
    if triage:
        return "triage", tenant
    if any(row["status"] not in ("done", "archived") for row in rows.values()):
        return "todo", tenant
    return "ready", tenant


def _validate_children_graph(children: list) -> None:
    """DB-free shape check + Kahn's cycle check on the sibling graph (a cycle
    would deadlock every involved child in ``todo`` forever)."""
    for idx, child in enumerate(children):
        if not isinstance(child, dict):
            raise ValueError(f"child[{idx}] is not a dict")
        title = child.get("title")
        if not isinstance(title, str) or not title.strip():
            raise ValueError(f"child[{idx}].title is required")
        parents_idx = child.get("parents") or []
        if not isinstance(parents_idx, list):
            raise ValueError(f"child[{idx}].parents must be a list")
        for p in parents_idx:
            if not isinstance(p, int) or p < 0 or p >= len(children):
                raise ValueError(f"child[{idx}].parents[{p}] is not a valid index into children")
            if p == idx:
                raise ValueError(f"child[{idx}] cannot list itself as a parent")
        contract = child.get("lifecycle_contract")
        kind = (
            str(contract.get("kind") or "").strip().casefold()
            if isinstance(contract, dict)
            else ""
        )
        if kind == "code":
            review_children = [
                role_idx
                for role_idx, role in enumerate(children)
                if (
                    isinstance(role.get("lifecycle_contract"), dict)
                    and str(
                        role["lifecycle_contract"].get("kind") or ""
                    ).strip().casefold() == "review"
                    and role["lifecycle_contract"].get("candidate_task_index") == idx
                )
            ]
            review_mode = str(contract.get("review_mode") or "").strip().casefold()
            if review_mode == "separate_card":
                if len(review_children) != 1:
                    raise ValueError(
                        f"child[{idx}] separate-card code contract requires exactly one review child"
                    )
            elif review_children:
                raise ValueError(
                    f"child[{idx}] same-card code contract must not have a review child"
                )
            validation_children = [
                role_idx
                for role_idx, role in enumerate(children)
                if (
                    isinstance(role.get("lifecycle_contract"), dict)
                    and str(
                        role["lifecycle_contract"].get("kind") or ""
                    ).strip().casefold() == "validation"
                    and role["lifecycle_contract"].get("candidate_task_index") == idx
                )
            ]
            if bool(contract.get("validation_required")):
                if len(validation_children) != 1:
                    raise ValueError(
                        f"child[{idx}] code contract requires exactly one validation child"
                    )
            elif validation_children:
                raise ValueError(
                    f"child[{idx}] code contract must not have a validation child"
                )
        if kind in {"review", "validation"}:
            candidate_idx = contract.get("candidate_task_index")
            if (
                isinstance(candidate_idx, bool)
                or not isinstance(candidate_idx, int)
                or candidate_idx < 0
                or candidate_idx >= len(children)
                or candidate_idx == idx
            ):
                raise ValueError(f"child[{idx}] has an invalid candidate_task_index")
            candidate_contract = children[candidate_idx].get("lifecycle_contract")
            separate_validation = (
                kind == "validation"
                and isinstance(candidate_contract, dict)
                and str(candidate_contract.get("kind") or "").strip().casefold() == "code"
                and str(candidate_contract.get("review_mode") or "").strip().casefold()
                == "separate_card"
            )
            if kind == "review" or not separate_validation:
                if candidate_idx not in parents_idx:
                    raise ValueError(
                        f"child[{idx}] role lifecycle contract must list "
                        "candidate_task_index as a parent"
                    )
            elif (
                candidate_idx in parents_idx
                or not any(
                    isinstance(children[parent_idx].get("lifecycle_contract"), dict)
                    and str(
                        children[parent_idx]["lifecycle_contract"].get("kind") or ""
                    ).strip().casefold() == "review"
                    and children[parent_idx]["lifecycle_contract"].get("candidate_task_index")
                    == candidate_idx
                    for parent_idx in parents_idx
                )
            ):
                raise ValueError(
                    f"child[{idx}] separate-card validation must depend on its review parent"
                )

    in_deg = [0] * len(children)
    adj: list[list[int]] = [[] for _ in children]
    for i, c in enumerate(children):
        for p in (c.get("parents") or []):
            adj[p].append(i)
            in_deg[i] += 1
    queue = [i for i in range(len(children)) if in_deg[i] == 0]
    seen = 0
    while queue:
        seen += 1
        for nb in adj[queue.pop()]:
            in_deg[nb] -= 1
            if in_deg[nb] == 0:
                queue.append(nb)
    if seen != len(children):
        raise ValueError("cyclic dependency detected in decomposed children list")


def decompose_triage_task(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    root_assignee: Optional[str],
    children: list[dict],
    author: Optional[str] = None,
    auto_promote: bool = True,
    expected_goal_revision_id: Optional[int] = None,
    expected_goal_revision_version: Optional[int] = None,
) -> Optional[list[str]]:
    """Fan a triage task out into children and move the root to ``todo``; the root
    waits on every child and wakes (``ready``) when all are done.

    ``children``: dicts of ``title`` (required), ``body``, ``assignee``,
    ``parents`` (indices into this list), optional workspace overrides.
    Returns child ids in input order, or None when the root is missing / not
    in triage. Atomic: a malformed entry aborts the whole fan-out.
    """
    from hermes_cli.kanban_db import (
        _canonical_assignee, _new_task_id, _link, _append_event, _insert_comment,
        _ensure_goal_revision_schema, _validate_lifecycle_role_identity,
        task_decomposition_context, write_txn, recompute_ready,
    )
    if not children:
        return None
    _ensure_goal_revision_schema(conn)
    if root_assignee is not None:
        root_assignee = _canonical_assignee(root_assignee)
    _validate_children_graph(children)
    now = int(time.time())
    with write_txn(conn):
        root_row = conn.execute(
            "SELECT id, status, tenant, workspace_kind, workspace_path, "
            "goal_revision_id, lifecycle_contract FROM tasks WHERE id = ?", (task_id,),
        ).fetchone()
        if root_row is None or root_row["status"] != "triage":
            return None
        if conn.execute(
            "SELECT 1 FROM task_events WHERE task_id = ? AND kind = 'decomposed' LIMIT 1",
            (task_id,),
        ).fetchone():
            return None
        # Re-read the graph and comments only after BEGIN IMMEDIATE. This is
        # the last race barrier before any child can be inserted.
        current_context = task_decomposition_context(conn, task_id)
        actual_goal_id = current_context.get("goal_revision_id")
        actual_goal_version = current_context.get("goal_revision_version")
        if (
            expected_goal_revision_id is not None
            and actual_goal_id != int(expected_goal_revision_id)
        ) or (
            expected_goal_revision_version is not None
            and actual_goal_version != int(expected_goal_revision_version)
        ):
            _append_event(
                conn, task_id, "decompose_proposed",
                {
                    "dry_run": True,
                    "mutation": False,
                    "reason": "effective goal revision changed during decomposition",
                    "expected_goal_revision_id": expected_goal_revision_id,
                    "expected_goal_revision_version": expected_goal_revision_version,
                    "actual_goal_revision_id": actual_goal_id,
                    "actual_goal_revision_version": actual_goal_version,
                    "author": str(author or "decomposer").strip() or "decomposer",
                },
            )
            return None
        if current_context.get("nontrivial_graph"):
            ids = [comment["id"] for comment in current_context["comments"]]
            _append_event(
                conn, task_id, "decompose_proposed",
                {
                    "dry_run": True,
                    "mutation": False,
                    "reason": "existing task graph",
                    "parents": current_context["parents"],
                    "children": current_context["children"],
                    "descendants": current_context["descendants"],
                    "comment_ids": ids,
                    "comment_ids_seen": ids,
                    "prior_purpose": current_context["prior_purpose"],
                    "author": str(author or "decomposer").strip() or "decomposer",
                },
            )
            return None
        child_ids = [_new_task_id() for _ in children]
        for idx, child in enumerate(children):
            _insert_decomposed_child(
                conn, task_id, root_row, child, author, now,
                new_id=child_ids[idx], child_ids=child_ids,
            )
        for child_id, child in zip(child_ids, children):
            child_contract = decode_contract(
                conn.execute(
                    "SELECT lifecycle_contract FROM tasks WHERE id = ?",
                    (child_id,),
                ).fetchone()["lifecycle_contract"]
            )
            _validate_lifecycle_role_identity(conn, child_contract, child.get("assignee"))
        # Sibling edges within the decomposed graph.
        for idx, child in enumerate(children):
            for p_idx in child.get("parents") or []:
                parent_id, child_id = child_ids[p_idx], child_ids[idx]
                requirement = infer_edge_requirement(conn, parent_id, child_id)
                _link(conn, parent_id, child_id, requirement=requirement)
                _append_event(
                    conn, child_id, "linked",
                    {"parent": parent_id, "child": child_id, "requirement": requirement},
                )
        # Root waits for the whole graph: link it under EVERY child (simpler
        # than computing leaves; cycle-free since the root is only ever a child).
        for cid in child_ids:
            requirement = (
                "phase_finished"
                if root_row["lifecycle_contract"] is None
                else infer_edge_requirement(conn, cid, task_id)
            )
            _link(conn, cid, task_id, requirement=requirement)
        # Flip the root triage -> todo, assignee -> orchestrator.
        sets = ["status = 'todo'"]
        params: list[Any] = []
        if root_assignee is not None:
            sets.append("assignee = ?")
            params.append(root_assignee)
        params.append(task_id)
        conn.execute(f"UPDATE tasks SET {', '.join(sets)} WHERE id = ?", tuple(params))
        if author and author.strip():
            _insert_comment(
                conn, task_id, author.strip(),
                "Decomposed into " + ", ".join(child_ids)
                + ". Root will wake when all children complete.",
                now,
            )
        _append_event(
            conn, task_id, "decomposed",
            {
                "child_ids": child_ids,
                "root_assignee": root_assignee,
                "comment_ids_seen": [comment["id"] for comment in current_context["comments"]],
            },
        )
    # Outside the txn (own IMMEDIATE txn). ``auto_promote=False`` leaves the
    # children in ``todo`` for manual-review-first workflows.
    if auto_promote:
        recompute_ready(conn)
    return child_ids


def _insert_decomposed_child(
    conn: sqlite3.Connection, root_id: str, root_row: sqlite3.Row, child: dict,
    author: Optional[str], now: int, *,
    new_id: Optional[str] = None, child_ids: Optional[list[str]] = None,
) -> str:
    """Insert one decomposed child as ``todo``; links are added after all
    sibling ids exist so typed role contracts can reference any sibling."""
    from hermes_cli.kanban_db import _new_task_id, _canonical_assignee, _append_event
    root_ws_kind = root_row["workspace_kind"] or "scratch"
    child_ws_kind = child.get("workspace_kind") or root_ws_kind
    if child.get("workspace_path"):
        child_ws_path = child.get("workspace_path")
    elif child_ws_kind == "worktree":
        child_ws_path = None
    elif child_ws_kind == root_ws_kind:
        child_ws_path = root_row["workspace_path"]
    else:
        child_ws_path = None
    new_id = new_id or _new_task_id()
    body = child.get("body")
    raw_contract = child.get("lifecycle_contract")
    if raw_contract is None:
        lifecycle_json = encode_contract(None, default_on_none=True)
    else:
        contract = dict(raw_contract)
        kind = str(contract.get("kind") or "").strip().casefold()
        if kind in {"review", "validation"}:
            candidate_index = contract.pop("candidate_task_index", None)
            if (
                child_ids is None
                or isinstance(candidate_index, bool)
                or not isinstance(candidate_index, int)
                or not 0 <= candidate_index < len(child_ids)
            ):
                raise LifecycleContractError(
                    "decomposed role child has an invalid candidate_task_index"
                )
            contract["candidate_task_id"] = child_ids[candidate_index]
        lifecycle_json = encode_contract(contract, default_on_none=False)
    normalized_lifecycle = decode_contract(lifecycle_json)
    conn.execute(
        "INSERT INTO tasks "
        "(id, title, body, assignee, status, workspace_kind, "
        " workspace_path, tenant, created_at, created_by, lifecycle_contract) "
        "VALUES (?, ?, ?, ?, 'todo', ?, ?, ?, ?, ?, ?)",
        (
            new_id, child["title"].strip(), body if isinstance(body, str) else None,
            _canonical_assignee(child.get("assignee")), child_ws_kind, child_ws_path,
            root_row["tenant"], now, (author or "decomposer"), lifecycle_json,
        ),
    )
    child_goal_author = str(author or "decomposer").strip() or "decomposer"
    child_goal_cur = conn.execute(
        """
        INSERT INTO task_goal_revisions
            (task_id, version, title, body, goal_mode, author,
             created_at, reason, prior_version)
        VALUES (?, 1, ?, ?, 0, ?, ?, 'initial goal', NULL)
        """,
        (
            new_id, child["title"].strip(), body if isinstance(body, str) else None,
            child_goal_author, now,
        ),
    )
    conn.execute(
        "UPDATE tasks SET goal_revision_id = ? WHERE id = ?",
        (int(child_goal_cur.lastrowid), new_id),
    )
    _append_event(
        conn,
        new_id,
        "created",
        {
            "by": author or "decomposer",
            "from_decompose_of": root_id,
            "lifecycle_contract": normalized_lifecycle,
        },
    )
    inherit_creator_origin(conn, new_id, root_id, created_at=now)
    return new_id

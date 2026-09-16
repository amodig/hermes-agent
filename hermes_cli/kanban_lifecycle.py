"""Typed Kanban lifecycle contracts and the read-only dependency projection.

The database remains the event/run ledger.  This module only normalizes the
small contract vocabulary, validates typed edges, and derives current state
from durable rows; it never stores a mutable acceptance flag.
"""
from __future__ import annotations

import json
import sqlite3
from typing import Any, Mapping, Optional


LIFECYCLE_SCHEMA_VERSION = 1
VALID_REQUIREMENTS = frozenset({"phase_finished", "review_approved", "validation_passed"})
VALID_CONTRACT_KINDS = frozenset({"general", "code", "review", "validation"})
VALID_REVIEW_MODES = frozenset({"same_card", "separate_card"})
VALID_VERDICTS = frozenset({"APPROVE", "REQUEST_CHANGES", "PASS", "FAIL"})
_VALID_VERDICTS_BY_PHASE = {
    "review": frozenset({"APPROVE", "REQUEST_CHANGES"}),
    "validation": frozenset({"PASS", "FAIL"}),
}


class LifecycleContractError(ValueError):
    """A contract or dependency edge cannot be admitted."""


class LifecycleEvidenceError(ValueError):
    """A handoff does not contain a complete, current lifecycle envelope."""


def _profile(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        raise LifecycleContractError("reviewer must be a non-empty profile")
    try:
        from hermes_cli.profiles import normalize_profile_name

        return normalize_profile_name(text)
    except Exception:
        return text.casefold()


def _task_id(value: Any, field: str = "candidate_task_id") -> str:
    text = str(value or "").strip()
    if not text:
        raise LifecycleContractError(f"{field} must be a non-empty task id")
    return text


def _normalize_general_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    if set(value) != {"kind"}:
        raise LifecycleContractError("general lifecycle_contract must be exactly {'kind': 'general'}")
    return {"kind": "general"}


def _normalize_code_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    expected = {"kind", "review_mode", "reviewer", "validation_required"}
    if set(value) != expected:
        raise LifecycleContractError(
            "code lifecycle_contract must contain exactly kind, review_mode, reviewer, validation_required"
        )
    mode = str(value.get("review_mode") or "").strip().casefold()
    if mode not in VALID_REVIEW_MODES:
        raise LifecycleContractError("review_mode must be 'same_card' or 'separate_card'")
    if not isinstance(value.get("validation_required"), bool):
        raise LifecycleContractError("validation_required must be boolean")
    return {
        "kind": "code",
        "review_mode": mode,
        "reviewer": _profile(value.get("reviewer")),
        "validation_required": bool(value["validation_required"]),
    }


def _normalize_role_contract(value: Mapping[str, Any], kind: str) -> dict[str, Any]:
    if set(value) != {"kind", "candidate_task_id"}:
        raise LifecycleContractError(
            f"{kind} lifecycle_contract must contain exactly kind and candidate_task_id"
        )
    return {"kind": kind, "candidate_task_id": _task_id(value.get("candidate_task_id"))}


def _normalize_review_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    return _normalize_role_contract(value, "review")


def _normalize_validation_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    return _normalize_role_contract(value, "validation")


_CONTRACT_NORMALIZERS = {
    "general": _normalize_general_contract,
    "code": _normalize_code_contract,
    "review": _normalize_review_contract,
    "validation": _normalize_validation_contract,
}

def normalize_contract(value: Any, *, default_on_none: bool = False) -> Optional[dict[str, Any]]:
    """Return a canonical contract or ``None`` for an historical NULL.

    New rows pass ``default_on_none=True`` and therefore get the explicit
    general contract. Existing NULL rows stay NULL until an operator binds
    them deliberately.
    """
    if value is None:
        return {"kind": "general"} if default_on_none else None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError) as exc:
            raise LifecycleContractError("lifecycle_contract must be valid JSON") from exc
    if not isinstance(value, Mapping):
        raise LifecycleContractError("lifecycle_contract must be an object")
    kind = str(value.get("kind") or "").strip().casefold()
    normalizer = _CONTRACT_NORMALIZERS.get(kind)
    if normalizer is None:
        raise LifecycleContractError(f"lifecycle_contract.kind must be one of {sorted(VALID_CONTRACT_KINDS)}")
    return normalizer(value)


def encode_contract(value: Any, *, default_on_none: bool = False) -> Optional[str]:
    contract = normalize_contract(value, default_on_none=default_on_none)
    return json.dumps(contract, ensure_ascii=False, sort_keys=True) if contract is not None else None


def decode_contract(value: Any) -> Optional[dict[str, Any]]:
    return normalize_contract(value, default_on_none=False)

def safe_decode_contract(value: Any) -> Optional[dict[str, Any]]:
    """Decode stored lifecycle JSON without letting malformed history break reads."""
    try:
        return decode_contract(value)
    except LifecycleContractError:
        return None


def _row_contract(row: sqlite3.Row) -> Optional[dict[str, Any]]:
    return safe_decode_contract(row["lifecycle_contract"] if "lifecycle_contract" in row.keys() else None)



def _contract_for(conn: sqlite3.Connection, task_id: str) -> Optional[dict[str, Any]]:
    row = conn.execute("SELECT lifecycle_contract FROM tasks WHERE id = ?", (task_id,)).fetchone()
    return _row_contract(row) if row else None




class _ProjectionCache:
    """Read-only snapshot of the requested lifecycle dependency closure."""

    _QUERY_BATCH_SIZE = 400

    def __init__(self, conn: sqlite3.Connection, task_ids: list[str]) -> None:
        self.conn = conn
        self.tasks: dict[str, sqlite3.Row] = {}
        self.links_by_parent: dict[str, list[dict[str, Any]]] = {}
        self.links_by_child: dict[str, list[dict[str, Any]]] = {}
        self.links_by_pair: dict[tuple[str, str], dict[str, Any]] = {}

        pending = set(task_ids)
        dependency_targets = set(task_ids)
        state_targets = set(task_ids)
        queried_task_ids: set[str] = set()
        queried_parent_ids: set[str] = set()
        queried_role_parent_ids: set[str] = set()
        role_expansion_targets: set[str] = set()
        while True:
            task_batch = sorted(pending - queried_task_ids)[:self._QUERY_BATCH_SIZE]
            if task_batch:
                queried_task_ids.update(task_batch)
                placeholders = ",".join("?" for _ in task_batch)
                for row in conn.execute(
                    f"SELECT * FROM tasks WHERE id IN ({placeholders})", task_batch,
                ).fetchall():
                    self.tasks[row["id"]] = row

            parent_batch = sorted(dependency_targets - queried_parent_ids)[:self._QUERY_BATCH_SIZE]
            if parent_batch:
                queried_parent_ids.update(parent_batch)
                placeholders = ",".join("?" for _ in parent_batch)
                for row in conn.execute(
                    "SELECT l.parent_id, l.child_id, l.requirement, "
                    "p.status AS parent_status, p.lifecycle_contract AS parent_contract, "
                    "t.lifecycle_contract AS child_contract "
                    "FROM task_links l "
                    "JOIN tasks p ON p.id = l.parent_id "
                    "JOIN tasks t ON t.id = l.child_id "
                    f"WHERE l.child_id IN ({placeholders}) ORDER BY p.id",
                    parent_batch,
                ).fetchall():
                    pending.add(row["parent_id"])
                    parent_contract = safe_decode_contract(row["parent_contract"])
                    requirement = row["requirement"] or "phase_finished"
                    if (
                        requirement != "phase_finished"
                        and row["child_id"] in dependency_targets
                        and parent_contract
                        and parent_contract["kind"] in {"code", "review", "validation"}
                    ):
                        state_targets.add(row["parent_id"])
                    self._store_link(
                        row["parent_id"],
                        row["child_id"],
                        row["requirement"],
                        parent_status=row["parent_status"],
                        parent_contract=row["parent_contract"],
                        child_contract=row["child_contract"],
                    )

            for task_id in state_targets:
                row = self.tasks.get(task_id)
                contract = _row_contract(row) if row is not None else None
                if row is None or not contract:
                    continue
                if contract["kind"] in {"review", "validation"}:
                    pending.add(str(contract["candidate_task_id"]))
                if contract["kind"] == "code":
                    role_expansion_targets.add(task_id)

            role_parent_ids: list[str] = []
            for task_id in sorted(role_expansion_targets - queried_role_parent_ids):
                row = self.tasks.get(task_id)
                if row is None:
                    if task_id in queried_task_ids:
                        queried_role_parent_ids.add(task_id)
                    continue
                contract = _row_contract(row)
                if contract is None:
                    queried_role_parent_ids.add(task_id)
                    continue
                if contract["kind"] == "code":
                    role_parent_ids.append(task_id)
                    continue
                if contract["kind"] != "review":
                    queried_role_parent_ids.add(task_id)
                    continue
                candidate_id = str(contract.get("candidate_task_id") or "")
                if candidate_id not in queried_task_ids:
                    continue
                candidate_row = self.tasks.get(candidate_id)
                candidate_contract = (
                    _row_contract(candidate_row) if candidate_row is not None else None
                )
                if (
                    candidate_contract
                    and candidate_contract.get("kind") == "code"
                    and candidate_contract.get("validation_required")
                ):
                    role_parent_ids.append(task_id)
                else:
                    queried_role_parent_ids.add(task_id)

            if role_parent_ids:
                queried_role_parent_ids.update(role_parent_ids)
                for start in range(0, len(role_parent_ids), self._QUERY_BATCH_SIZE):
                    role_batch = role_parent_ids[start:start + self._QUERY_BATCH_SIZE]
                    placeholders = ",".join("?" for _ in role_batch)
                    for row in conn.execute(
                        "SELECT l.parent_id, l.child_id, l.requirement, "
                        "c.lifecycle_contract AS child_contract "
                        "FROM task_links l JOIN tasks c ON c.id = l.child_id "
                        f"WHERE l.parent_id IN ({placeholders}) ORDER BY c.id",
                        role_batch,
                    ).fetchall():
                        parent_contract = _row_contract(self.tasks[row["parent_id"]])
                        child_contract = safe_decode_contract(row["child_contract"])
                        if self._is_role_child(
                            row["parent_id"], parent_contract, child_contract,
                        ):
                            pending.add(row["child_id"])
                            state_targets.add(row["child_id"])
                            if (
                                parent_contract
                                and parent_contract.get("kind") == "code"
                                and parent_contract.get("review_mode") == "separate_card"
                                and parent_contract.get("validation_required")
                                and child_contract
                                and child_contract.get("kind") == "review"
                            ):
                                role_expansion_targets.add(row["child_id"])
                            self._store_link(
                                row["parent_id"],
                                row["child_id"],
                                row["requirement"],
                                parent_status=self.tasks[row["parent_id"]]["status"],
                                parent_contract=self.tasks[row["parent_id"]]["lifecycle_contract"],
                                child_contract=row["child_contract"],
                            )

            if not task_batch and not parent_batch and not role_parent_ids:
                break

        for links in (*self.links_by_parent.values(), *self.links_by_child.values()):
            links.sort(key=lambda link: (link["parent_id"], link["child_id"]))

        self.runs_by_id: dict[int, sqlite3.Row] = {}
        self.runs_by_task: dict[str, list[sqlite3.Row]] = {}
        self.events_by_task: dict[str, list[sqlite3.Row]] = {}
        history_ids: set[str] = set()
        for task_id in state_targets:
            row = self.tasks.get(task_id)
            contract = _row_contract(row) if row is not None else None
            if not contract:
                continue
            if contract["kind"] in {"code", "review", "validation"}:
                history_ids.add(task_id)
            candidate_id = contract.get("candidate_task_id")
            if candidate_id and candidate_id in self.tasks:
                history_ids.add(str(candidate_id))
        history_ids_list = sorted(history_ids)
        for start in range(0, len(history_ids_list), self._QUERY_BATCH_SIZE):
            batch = history_ids_list[start:start + self._QUERY_BATCH_SIZE]
            placeholders = ",".join("?" for _ in batch)
            for row in conn.execute(
                f"SELECT id, task_id, metadata, outcome FROM task_runs "
                f"WHERE task_id IN ({placeholders}) ORDER BY id DESC",
                batch,
            ).fetchall():
                self.runs_by_id[int(row["id"])] = row
                self.runs_by_task.setdefault(row["task_id"], []).append(row)
            for row in conn.execute(
                f"SELECT id, task_id, kind, payload FROM task_events "
                f"WHERE task_id IN ({placeholders}) ORDER BY id DESC",
                batch,
            ).fetchall():
                self.events_by_task.setdefault(row["task_id"], []).append(row)
        self.lifecycle: dict[str, dict[str, Any]] = {}
        self.dependencies: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _is_role_child(
        parent_id: str,
        parent_contract: Optional[dict[str, Any]],
        child_contract: Optional[dict[str, Any]],
    ) -> bool:
        if not parent_contract or not child_contract:
            return False
        child_kind = child_contract.get("kind")
        candidate_id = child_contract.get("candidate_task_id")
        if parent_contract.get("kind") == "code":
            return (
                parent_contract.get("review_mode") == "separate_card"
                and child_kind == "review"
                and candidate_id == parent_id
            ) or (
                parent_contract.get("review_mode") == "same_card"
                and parent_contract.get("validation_required")
                and child_kind == "validation"
                and candidate_id == parent_id
            )
        return (
            parent_contract.get("kind") == "review"
            and child_kind == "validation"
            and candidate_id == parent_contract.get("candidate_task_id")
        )

    def _store_link(
        self,
        parent_id: str,
        child_id: str,
        requirement: Optional[str],
        *,
        parent_status: Optional[str] = None,
        parent_contract: Any = None,
        child_contract: Any = None,
    ) -> None:
        pair = (parent_id, child_id)
        link = self.links_by_pair.get(pair)
        if link is None:
            link = {
                "parent_id": parent_id,
                "child_id": child_id,
                "requirement": requirement,
                "parent_status": parent_status,
                "parent_contract": parent_contract,
                "child_contract": child_contract,
            }
            self.links_by_pair[pair] = link
            self.links_by_parent.setdefault(parent_id, []).append(link)
            self.links_by_child.setdefault(child_id, []).append(link)
            return
        if parent_status is not None:
            link["parent_status"] = parent_status
        if parent_contract is not None:
            link["parent_contract"] = parent_contract
        if child_contract is not None:
            link["child_contract"] = child_contract

    def role_children(self, task_id: str) -> list[dict[str, Any]]:
        return [
            {
                "id": link["child_id"],
                "lifecycle_contract": link["child_contract"],
            }
            for link in self.links_by_parent.get(task_id, ())
        ]

    def review_rows(self, task_id: str) -> list[dict[str, Any]]:
        return [
            {
                "requirement": link["requirement"],
                "lifecycle_contract": link["parent_contract"],
            }
            for link in self.links_by_child.get(task_id, ())
        ]

    def parent_rows(self, task_id: str) -> list[dict[str, Any]]:
        return [
            {
                "parent_id": link["parent_id"],
                "parent_status": link["parent_status"],
                "lifecycle_contract": link["parent_contract"],
                "requirement": link["requirement"],
            }
            for link in self.links_by_child.get(task_id, ())
        ]


def _projection_cache_for(
    conn: sqlite3.Connection, cache: Optional[_ProjectionCache],
) -> Optional[_ProjectionCache]:
    return cache if cache is not None and cache.conn is conn else None

def _task_row(
    conn: sqlite3.Connection, task_id: str, cache: Optional[_ProjectionCache] = None,
) -> Optional[sqlite3.Row]:
    cache = _projection_cache_for(conn, cache)
    if cache is not None:
        return cache.tasks.get(task_id)
    return conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()


def _task_goal_revision_id(row: Optional[sqlite3.Row]) -> Optional[int]:
    if row is None or "goal_revision_id" not in row.keys() or row["goal_revision_id"] is None:
        return None
    return int(row["goal_revision_id"])


def _validate_general_edge(
    parent_id: str,
    child_id: str,
    parent: dict[str, Any],
    child: dict[str, Any],
    child_kind: str,
    requirement: str,
) -> bool:
    return requirement == "phase_finished" and child_kind not in {"review", "validation"}


def _validate_code_edge(
    parent_id: str,
    child_id: str,
    parent: dict[str, Any],
    child: dict[str, Any],
    child_kind: str,
    requirement: str,
) -> bool:
    if child_kind == "review" and child.get("candidate_task_id") == parent_id:
        return requirement == "phase_finished" and parent.get("review_mode") == "separate_card"
    if child_kind == "validation" and child.get("candidate_task_id") == parent_id:
        return requirement == "review_approved" and parent.get("review_mode") == "same_card"
    return child_kind == "general" and requirement == "phase_finished"


def _validate_review_edge(
    parent_id: str,
    child_id: str,
    parent: dict[str, Any],
    child: dict[str, Any],
    child_kind: str,
    requirement: str,
) -> bool:
    return (
        (
            child_kind == "validation"
            and child.get("candidate_task_id") == parent.get("candidate_task_id")
            and requirement == "review_approved"
        )
        or (child_kind == "general" and requirement == "review_approved")
    )


def _validate_validation_edge(
    parent_id: str,
    child_id: str,
    parent: dict[str, Any],
    child: dict[str, Any],
    child_kind: str,
    requirement: str,
) -> bool:
    return requirement == "validation_passed"


_LIFECYCLE_EDGE_VALIDATORS = {
    "general": _validate_general_edge,
    "code": _validate_code_edge,
    "review": _validate_review_edge,
    "validation": _validate_validation_edge,
}

def validate_edge(
    conn: sqlite3.Connection,
    parent_id: str,
    child_id: str,
    requirement: str,
) -> str:
    """Validate the typed topology and return its canonical requirement."""
    requirement = str(requirement or "").strip().casefold()
    if requirement not in VALID_REQUIREMENTS:
        raise LifecycleContractError(f"requirement must be one of {sorted(VALID_REQUIREMENTS)}")
    parent = _task_row(conn, parent_id)
    child = _task_row(conn, child_id)
    if parent is None or child is None:
        raise LifecycleContractError("cannot link a task with an unknown endpoint")
    p = _row_contract(parent)
    c = _row_contract(child)
    if p is None or c is None:
        raise LifecycleContractError(
            "lifecycle_unclassified: bind both task contracts before creating a typed edge"
        )
    pk = p["kind"]
    ck = c["kind"]
    if ck == "validation":
        candidate_contract = _contract_for(conn, c["candidate_task_id"])
        if (
            candidate_contract is None
            or candidate_contract.get("kind") != "code"
            or not candidate_contract.get("validation_required")
        ):
            raise LifecycleContractError(
                "validation edges require a code candidate with validation_required=true"
            )
    validator = _LIFECYCLE_EDGE_VALIDATORS.get(pk)
    legal = validator(parent_id, child_id, p, c, ck, requirement) if validator else False
    if not legal:
        raise LifecycleContractError(
            f"illegal lifecycle edge {parent_id}({pk}) -> {child_id}({ck}) with {requirement}"
        )
    if ck in {"review", "validation"}:
        existing_rows = conn.execute(
            "SELECT t.id, t.lifecycle_contract "
            "FROM task_links l JOIN tasks t ON t.id = l.child_id "
            "WHERE l.parent_id = ? AND t.id != ?",
            (parent_id, child_id),
        ).fetchall()
        for existing in existing_rows:
            existing_contract = _row_contract(existing)
            if (
                existing_contract
                and existing_contract.get("kind") == ck
                and existing_contract.get("candidate_task_id") == c.get("candidate_task_id")
            ):
                raise LifecycleContractError(
                    f"duplicate lifecycle {ck} edge for candidate {c.get('candidate_task_id')}"
                )
    return requirement


def is_required_lifecycle_edge(
    conn: sqlite3.Connection, parent_id: str, child_id: str,
) -> bool:
    """Return whether removing this edge would break a typed role workflow."""
    parent = _task_row(conn, parent_id)
    child = _task_row(conn, child_id)
    p = _row_contract(parent)
    c = _row_contract(child)
    if p is None or c is None:
        return False
    edge = conn.execute(
        "SELECT requirement FROM task_links WHERE parent_id = ? AND child_id = ?",
        (parent_id, child_id),
    ).fetchone()
    requirement = edge["requirement"] if edge is not None else None
    if p["kind"] == "code" and c["kind"] == "review":
        return (
            c.get("candidate_task_id") == parent_id
            and p.get("review_mode") == "separate_card"
        )
    if p["kind"] == "code" and c["kind"] == "validation":
        return (
            c.get("candidate_task_id") == parent_id
            and p.get("review_mode") == "same_card"
        )
    if p["kind"] == "review" and c["kind"] == "general":
        return requirement == "review_approved"
    if p["kind"] == "validation":
        return requirement == "validation_passed"
    return (
        p["kind"] == "review"
        and c["kind"] == "validation"
        and c.get("candidate_task_id") == p.get("candidate_task_id")
    )



def infer_edge_requirement(conn: sqlite3.Connection, parent_id: str, child_id: str) -> str:
    """Choose the only legal requirement for a newly declared edge."""
    valid: list[str] = []
    for requirement in sorted(VALID_REQUIREMENTS):
        try:
            validate_edge(conn, parent_id, child_id, requirement)
        except LifecycleContractError:
            continue
        valid.append(requirement)
    if len(valid) != 1:
        raise LifecycleContractError(
            f"cannot infer a unique lifecycle requirement for {parent_id} -> {child_id}"
        )
    return valid[0]


def _json_column(value: Any) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _lifecycle_from_metadata(value: Any) -> Optional[dict[str, Any]]:
    metadata = _json_column(value)
    lifecycle = metadata.get("lifecycle")
    return lifecycle if isinstance(lifecycle, dict) else None


def _evidence_rows(
    conn: sqlite3.Connection,
    task_id: str,
    phase: Optional[str] = None,
    *,
    cache: Optional[_ProjectionCache] = None,
) -> list[tuple[int, dict[str, Any], str]]:
    snapshot = _projection_cache_for(conn, cache)
    rows = (
        snapshot.runs_by_task.get(task_id, ())
        if snapshot is not None
        else conn.execute(
            "SELECT id, metadata, outcome FROM task_runs WHERE task_id = ? ORDER BY id DESC",
            (task_id,),
        ).fetchall()
    )
    found: list[tuple[int, dict[str, Any], str]] = []
    for row in rows:
        lifecycle = _lifecycle_from_metadata(row["metadata"])
        if not lifecycle or lifecycle.get("schema") != LIFECYCLE_SCHEMA_VERSION:
            continue
        if phase is not None and lifecycle.get("phase") != phase:
            continue
        found.append((int(row["id"]), lifecycle, str(row["outcome"] or "")))
    return found


def _latest_head(
    conn: sqlite3.Connection, task_id: str, *, cache: Optional[_ProjectionCache] = None,
) -> Optional[str]:
    snapshot = _projection_cache_for(conn, cache)
    run_rows = (
        snapshot.runs_by_task.get(task_id, ())
        if snapshot is not None
        else conn.execute(
            "SELECT metadata FROM task_runs WHERE task_id = ? ORDER BY id DESC",
            (task_id,),
        ).fetchall()
    )
    for row in run_rows:
        metadata = _json_column(row["metadata"])
        lifecycle = metadata.get("lifecycle") if isinstance(metadata.get("lifecycle"), dict) else {}
        head = lifecycle.get("head_sha") or metadata.get("head_sha")
        if isinstance(head, str) and head.strip():
            return head.strip()
    event_rows = (
        snapshot.events_by_task.get(task_id, ())
        if snapshot is not None
        else conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? AND kind IN ('completed', 'review_requested') "
            "ORDER BY id DESC",
            (task_id,),
        ).fetchall()
    )
    for row in event_rows:
        if snapshot is not None and row["kind"] not in {"completed", "review_requested"}:
            continue
        payload = _json_column(row["payload"])
        lifecycle = payload.get("lifecycle") if isinstance(payload.get("lifecycle"), dict) else {}
        head = lifecycle.get("head_sha") or payload.get("head_sha")
        if isinstance(head, str) and head.strip():
            return head.strip()
    return None


def _latest_run_id(conn: sqlite3.Connection, task_id: str, phase: str) -> Optional[int]:
    rows = _evidence_rows(conn, task_id, phase)
    return rows[0][0] if rows else None


def _direct_role_children(
    conn: sqlite3.Connection,
    task_id: str,
    kind: str,
    *,
    candidate_id: Optional[str] = None,
    cache: Optional[_ProjectionCache] = None,
) -> list[str]:
    snapshot = _projection_cache_for(conn, cache)
    rows = (
        snapshot.role_children(task_id)
        if snapshot is not None
        else conn.execute(
            "SELECT t.id, t.lifecycle_contract FROM task_links l JOIN tasks t ON t.id = l.child_id "
            "WHERE l.parent_id = ? ORDER BY t.id",
            (task_id,),
        ).fetchall()
    )
    expected_candidate = candidate_id or task_id
    matches: list[str] = []
    for row in rows:
        contract = safe_decode_contract(row["lifecycle_contract"])
        if (
            contract
            and contract.get("kind") == kind
            and contract.get("candidate_task_id") == expected_candidate
        ):
            matches.append(str(row["id"]))
    return matches


def _direct_role_child(
    conn: sqlite3.Connection,
    task_id: str,
    kind: str,
    *,
    candidate_id: Optional[str] = None,
    cache: Optional[_ProjectionCache] = None,
) -> Optional[str]:
    matches = _direct_role_children(
        conn, task_id, kind, candidate_id=candidate_id, cache=cache,
    )
    return matches[0] if matches else None


def _candidate_snapshot(
    conn: sqlite3.Connection,
    candidate_id: str,
    *,
    cache: Optional[_ProjectionCache] = None,
) -> dict[str, Any]:
    snapshot = _projection_cache_for(conn, cache)
    row = _task_row(conn, candidate_id, cache)
    candidate_run_id = (
        int(row["candidate_run_id"])
        if row is not None
        and "candidate_run_id" in row.keys()
        and row["candidate_run_id"] is not None
        else None
    )
    if candidate_run_id is None:
        return {
            "task_id": candidate_id,
            "run_id": None,
            "head_sha": None,
            "goal_revision_id": _task_goal_revision_id(row),
        }
    run = (
        snapshot.runs_by_id.get(candidate_run_id)
        if snapshot is not None
        else conn.execute(
            "SELECT metadata FROM task_runs WHERE id = ? AND task_id = ?",
            (candidate_run_id, candidate_id),
        ).fetchone()
    )
    if snapshot is not None and run is not None and run["task_id"] != candidate_id:
        run = None
    lifecycle = _lifecycle_from_metadata(run["metadata"]) if run else None
    return {
        "task_id": candidate_id,
        "run_id": candidate_run_id,
        "head_sha": (
            lifecycle.get("head_sha") if lifecycle
            else _latest_head(conn, candidate_id, cache=cache)
        ),
        "goal_revision_id": _task_goal_revision_id(row),
    }


def _execution_outcome(
    conn: sqlite3.Connection,
    task_id: str,
    run_id: Optional[int],
    *,
    cache: Optional[_ProjectionCache] = None,
) -> Optional[str]:
    if run_id is None:
        return None
    snapshot = _projection_cache_for(conn, cache)
    row = (
        snapshot.runs_by_id.get(run_id)
        if snapshot is not None
        else conn.execute(
            "SELECT outcome FROM task_runs WHERE id = ? AND task_id = ?",
            (run_id, task_id),
        ).fetchone()
    )
    if snapshot is not None and row is not None and row["task_id"] != task_id:
        row = None
    if row is None:
        return None
    outcome = str(row["outcome"] or "").strip()
    return outcome or None





def _freshness(
    conn: sqlite3.Connection,
    lifecycle: Optional[dict[str, Any]],
    candidate_id: str,
    *,
    evidence_task_id: Optional[str] = None,
    cache: Optional[_ProjectionCache] = None,
) -> tuple[bool, Optional[str]]:
    if not lifecycle:
        return False, "verdict_missing"
    expected = _candidate_snapshot(conn, candidate_id, cache=cache)
    if expected["run_id"] is None:
        return False, "candidate_missing"
    candidate_run = lifecycle.get("candidate_run_id")
    if candidate_run != expected["run_id"]:
        return False, "candidate_head_mismatch"
    candidate_head = expected["head_sha"]
    if candidate_head is None:
        return False, "candidate_missing"
    if lifecycle.get("head_sha") != candidate_head:
        return False, "candidate_head_mismatch"
    goal_ids = lifecycle.get("goal_revision_ids")
    candidate_goal = expected["goal_revision_id"]
    task_row = _task_row(conn, evidence_task_id or candidate_id, cache)
    task_goal = _task_goal_revision_id(task_row)
    if (
        not isinstance(goal_ids, dict)
        or goal_ids.get(candidate_id) != candidate_goal
        or lifecycle.get("task_goal_revision_id") != task_goal
    ):
        return False, "goal_revision_stale"
    return True, None


def _verdict_for(
    conn: sqlite3.Connection,
    task_id: str,
    phase: str,
    *,
    cache: Optional[_ProjectionCache] = None,
) -> tuple[Optional[str], Optional[dict[str, Any]], Optional[str]]:
    rows = _evidence_rows(conn, task_id, phase, cache=cache)
    if not rows:
        return None, None, "verdict_missing"
    _, latest, _ = rows[0]
    def evidence_key(lifecycle: dict[str, Any]) -> tuple[Any, ...]:
        return (
            lifecycle.get("candidate_task_id"),
            lifecycle.get("candidate_run_id"),
            lifecycle.get("head_sha"),
            json.dumps(lifecycle.get("goal_revision_ids"), sort_keys=True),
            lifecycle.get("task_goal_revision_id"),
        )

    latest_key = evidence_key(latest)
    verdicts = {
        lifecycle.get("verdict")
        for _, lifecycle, _ in rows
        if evidence_key(lifecycle) == latest_key and lifecycle.get("verdict") is not None
    }
    if len(verdicts) > 1:
        return None, latest, "verdict_conflict"
    verdict = latest.get("verdict")
    if verdict is None:
        return None, latest, "verdict_missing"
    if not isinstance(verdict, str) or verdict not in _VALID_VERDICTS_BY_PHASE.get(phase, ()):
        return None, latest, "verdict_malformed"
    return verdict, latest, None



def _get_lifecycle_state(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    cache: Optional[_ProjectionCache] = None,
) -> dict[str, Any]:
    """Derive execution/review/validation/acceptance from runs and contracts."""
    row = _task_row(conn, task_id, cache)
    contract = _row_contract(row) if row else None
    state: dict[str, Any] = {
        "execution_outcome": None,
        "review_verdict": None,
        "validation_verdict": None,
        "acceptance": "unclassified" if contract is None else "not_applicable",
        "candidate_task_id": None,
        "candidate_run_id": None,
        "head_sha": None,
        "goal_revision_ids": {},
        "diagnostics": [],
    }
    if row is None or contract is None:
        return state
    kind = contract["kind"]
    if kind == "general":
        return state
    candidate_id = contract.get("candidate_task_id")
    if kind in {"review", "validation"}:
        candidate_id = str(candidate_id)
        state["candidate_task_id"] = candidate_id
        snapshot = _candidate_snapshot(conn, candidate_id, cache=cache)
        state["candidate_run_id"] = snapshot["run_id"]
        state["execution_outcome"] = _execution_outcome(
            conn, candidate_id, snapshot["run_id"], cache=cache,
        )
        state["head_sha"] = snapshot["head_sha"]
        state["goal_revision_ids"] = {candidate_id: snapshot["goal_revision_id"]}

        phase = "review" if kind == "review" else "validation"
        verdict, lifecycle, error = _verdict_for(conn, task_id, phase, cache=cache)
        if kind == "review":
            state["review_verdict"] = verdict
        else:
            state["validation_verdict"] = verdict
        if error and error != "verdict_missing":
            state["diagnostics"].append(error)
        freshness_reason: Optional[str] = None
        if verdict in VALID_VERDICTS and lifecycle:
            fresh, freshness_reason = _freshness(
                conn, lifecycle, candidate_id,
                evidence_task_id=task_id, cache=cache,
            )
            if not fresh and freshness_reason:
                state["diagnostics"].append(freshness_reason)
        if freshness_reason:
            state["acceptance"] = "stale"
        elif verdict in {"REQUEST_CHANGES", "FAIL"} and row["status"] == "done":
            state["acceptance"] = "rejected"
        elif verdict in {"APPROVE", "PASS"} and row["status"] == "done":
            state["acceptance"] = "accepted"
        else:
            state["acceptance"] = "pending"
        return state

    # Code card: implementation evidence belongs to this task; review and
    # validation evidence may be same-card or typed child cards.
    implementation_snapshot = _candidate_snapshot(conn, task_id, cache=cache)
    state["candidate_task_id"] = task_id
    state["candidate_run_id"] = implementation_snapshot["run_id"]
    state["head_sha"] = implementation_snapshot["head_sha"]
    state["goal_revision_ids"] = {task_id: implementation_snapshot["goal_revision_id"]}
    state["execution_outcome"] = _execution_outcome(
        conn, task_id, implementation_snapshot["run_id"], cache=cache,
    )
    role_conflict = False
    if contract.get("review_mode") == "same_card":
        review_id = task_id
    else:
        review_children = _direct_role_children(conn, task_id, "review", cache=cache)
        role_conflict = len(review_children) > 1
        review_id = review_children[0] if len(review_children) == 1 else None
    validation_parent_id = task_id if contract.get("review_mode") == "same_card" else review_id
    if contract.get("validation_required") and validation_parent_id:
        validation_children = _direct_role_children(
            conn,
            validation_parent_id,
            "validation",
            candidate_id=task_id,
            cache=cache,
        )
        validation_id = validation_children[0] if len(validation_children) == 1 else None
    else:
        validation_id = None
    if role_conflict:
        state["diagnostics"].append("role_conflict")
    review_accepted = False
    review_rejected = False
    if review_id:
        if review_id == task_id:
            verdict, lifecycle, error = _verdict_for(conn, task_id, "review", cache=cache)
            review_accepted = verdict == "APPROVE" and row["status"] == "done"
            review_rejected = verdict == "REQUEST_CHANGES" and row["status"] == "done"
        else:
            review_state = get_lifecycle_state(conn, review_id, _cache=cache)
            verdict, lifecycle, error = review_state["review_verdict"], None, None
            review_accepted = review_state["acceptance"] == "accepted"
            review_rejected = review_state["acceptance"] == "rejected"
            if verdict:
                rows = _evidence_rows(conn, review_id, "review", cache=cache)
                lifecycle = rows[0][1] if rows else None
        state["review_verdict"] = verdict
        if error and error != "verdict_missing":
            state["diagnostics"].append(error)
        if verdict in {"APPROVE", "REQUEST_CHANGES"} and lifecycle:
            fresh, reason = _freshness(
                conn, lifecycle, task_id,
                evidence_task_id=review_id, cache=cache,
            )
            if not fresh and reason:
                state["diagnostics"].append(reason)
                if verdict == "APPROVE":
                    state["acceptance"] = "stale"
                    review_accepted = False
    validation_accepted = not contract.get("validation_required")
    validation_rejected = False
    if contract.get("validation_required"):
        if validation_id:
            validation_state = get_lifecycle_state(conn, validation_id, _cache=cache)
            state["validation_verdict"] = validation_state["validation_verdict"]
            state["diagnostics"].extend(validation_state.get("diagnostics") or [])
            validation_accepted = validation_state["acceptance"] == "accepted"
            validation_rejected = validation_state["acceptance"] == "rejected"
        else:
            state["diagnostics"].append("candidate_missing")
            validation_accepted = False
    required_review_ok = review_accepted
    required_validation_ok = validation_accepted
    if row["status"] == "archived":
        state["acceptance"] = "stale"
    elif state["diagnostics"] and any(
        d in {"candidate_missing", "candidate_head_mismatch", "goal_revision_stale", "role_conflict"}
        for d in state["diagnostics"]
    ):
        state["acceptance"] = "stale"
    elif review_rejected or validation_rejected:
        state["acceptance"] = "rejected"
    elif required_review_ok and required_validation_ok and row["status"] == "done":
        state["acceptance"] = "accepted"
    else:
        state["acceptance"] = "pending"
    return state



def get_lifecycle_state(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    _cache: Optional[_ProjectionCache] = None,
) -> dict[str, Any]:
    cache = _projection_cache_for(conn, _cache)
    if cache is None:
        return _get_lifecycle_state(conn, task_id)
    if task_id not in cache.lifecycle:
        cache.lifecycle[task_id] = _get_lifecycle_state(conn, task_id, cache=cache)
    return cache.lifecycle[task_id]
def get_goal_acceptance(conn: sqlite3.Connection, root_id: str) -> dict[str, Any]:
    """Project acceptance across one deterministic root-to-leaf goal graph."""
    pending = [root_id]
    seen: set[str] = set()
    tasks: dict[str, dict[str, Any]] = {}
    while pending:
        task_id = pending.pop(0)
        if task_id in seen:
            continue
        seen.add(task_id)
        state = get_lifecycle_state(conn, task_id)
        tasks[task_id] = state
        children = conn.execute(
            "SELECT child_id FROM task_links WHERE parent_id = ? ORDER BY child_id",
            (task_id,),
        ).fetchall()
        pending.extend(str(row["child_id"]) for row in children)
    typed = [state for state in tasks.values() if state["acceptance"] != "not_applicable"]
    if not typed:
        acceptance = "unclassified" if any(
            state["acceptance"] == "unclassified" for state in tasks.values()
        ) else "not_applicable"
    elif any(state["acceptance"] == "unclassified" for state in typed):
        acceptance = "unclassified"
    elif any(state["acceptance"] == "rejected" for state in typed):
        acceptance = "rejected"
    elif any(state["acceptance"] == "stale" for state in typed):
        acceptance = "stale"
    elif any(state["acceptance"] == "pending" for state in typed):
        acceptance = "pending"
    else:
        acceptance = "accepted"
    diagnostics = sorted({
        diagnostic
        for state in tasks.values()
        for diagnostic in state.get("diagnostics") or []
    })
    return {
        "root_task_id": root_id,
        "acceptance": acceptance,
        "tasks": tasks,
        "diagnostics": diagnostics,
    }


def _evaluate_dependencies(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    cache: Optional[_ProjectionCache] = None,
) -> dict[str, Any]:
    """Evaluate direct edges while preserving historical NULL contracts."""
    task = _task_row(conn, task_id, cache)
    if task is None:
        return {
            "satisfied": False,
            "blockers": [{
                "parent_id": None,
                "requirement": None,
                "code": "candidate_missing",
                "message": "task does not exist",
            }],
        }
    task_contract = _row_contract(task)
    raw_task_contract = task["lifecycle_contract"]
    if task_contract is None and raw_task_contract is not None:
        return {
            "satisfied": False,
            "blockers": [{
                "parent_id": None,
                "requirement": None,
                "code": "lifecycle_unclassified",
                "message": "task lifecycle contract is unclassified",
            }],
        }
    blockers: list[dict[str, Any]] = []
    if task_contract and task_contract.get("kind") in {"review", "validation"}:
        candidate_id = str(task_contract.get("candidate_task_id") or "")
        candidate = _task_row(conn, candidate_id, cache)
        candidate_contract = _row_contract(candidate)
        same_card_validation = (
            task_contract.get("kind") == "validation"
            and candidate_contract is not None
            and candidate_contract.get("kind") == "code"
            and candidate_contract.get("review_mode") == "same_card"
        )
        requires_candidate_edge = (
            task_contract.get("kind") == "review" or same_card_validation
        )
        candidate_edge = (
            _projection_cache_for(conn, cache).links_by_pair.get((candidate_id, task_id))
            if _projection_cache_for(conn, cache) is not None
            else conn.execute(
                "SELECT requirement FROM task_links WHERE parent_id = ? AND child_id = ?",
                (candidate_id, task_id),
            ).fetchone()
        )
        review_edge = None
        if task_contract.get("kind") == "validation" and not requires_candidate_edge:
            review_rows = (
                _projection_cache_for(conn, cache).review_rows(task_id)
                if _projection_cache_for(conn, cache) is not None
                else conn.execute(
                    """
                    SELECT l.requirement, p.lifecycle_contract
                    FROM task_links l
                    JOIN tasks p ON p.id = l.parent_id
                    WHERE l.child_id = ?
                    ORDER BY p.id
                    """,
                    (task_id,),
                ).fetchall()
            )
            for row in review_rows:
                parent_contract = safe_decode_contract(row["lifecycle_contract"])
                if (
                    parent_contract
                    and parent_contract.get("kind") == "review"
                    and parent_contract.get("candidate_task_id") == candidate_id
                ):
                    review_edge = row
                    break
        if candidate_id and (
            (requires_candidate_edge and candidate_edge is None)
            or (not requires_candidate_edge and review_edge is None)
        ):
            dependency = "candidate task" if requires_candidate_edge else "its review card"
            blockers.append({
                "parent_id": candidate_id,
                "requirement": None,
                "code": "candidate_edge_missing",
                "message": f"role card must depend directly on {dependency}",
            })

    rows = (
        _projection_cache_for(conn, cache).parent_rows(task_id)
        if _projection_cache_for(conn, cache) is not None
        else conn.execute(
            "SELECT p.id AS parent_id, p.status AS parent_status, p.lifecycle_contract, "
            "       l.requirement FROM task_links l JOIN tasks p ON p.id = l.parent_id "
            "WHERE l.child_id = ? ORDER BY p.id", (task_id,),
        ).fetchall()
    )
    for row in rows:
        parent_id = row["parent_id"]
        requirement = row["requirement"]
        if requirement is None:
            requirement = "phase_finished"
        base = {"parent_id": parent_id, "requirement": requirement}
        raw_parent_contract = row["lifecycle_contract"]
        parent_contract = safe_decode_contract(raw_parent_contract)
        if parent_contract is None and raw_parent_contract is not None:
            blockers.append({
                **base,
                "code": "lifecycle_unclassified",
                "message": "parent lifecycle contract is unclassified",
            })
            continue
        if requirement not in VALID_REQUIREMENTS:
            blockers.append({
                **base,
                "code": "lifecycle_unclassified",
                "message": "dependency edge has no lifecycle requirement",
            })
            continue
        if row["parent_status"] == "archived":
            if parent_contract is not None and parent_contract["kind"] != "general":
                blockers.append({
                    **base,
                    "code": "goal_revision_stale",
                    "message": "typed code/review/validation parent was archived",
                })
            continue
        if requirement == "phase_finished":
            if row["parent_status"] != "done":
                blockers.append({
                    **base,
                    "code": "verdict_missing",
                    "message": "parent phase has not finished",
                })
            continue
        if parent_contract is None:
            blockers.append({
                **base,
                "code": "lifecycle_unclassified",
                "message": "parent lifecycle contract is unclassified",
            })
            continue
        if requirement == "review_approved":
            # A stale prior validation must not block the fresh review that
            # releases its replacement. Gate this phase, not aggregate acceptance.
            verdict, lifecycle, _ = _verdict_for(conn, parent_id, "review", cache=cache)
            if verdict is None:
                blockers.append({
                    **base,
                    "code": "verdict_missing",
                    "message": "review approval is missing",
                })
            elif verdict == "REQUEST_CHANGES":
                blockers.append({
                    **base,
                    "code": "verdict_conflict",
                    "message": "review requested changes",
                })
            elif verdict != "APPROVE":
                blockers.append({
                    **base,
                    "code": "verdict_malformed",
                    "message": "review verdict is invalid",
                })
            elif not _freshness(
                conn,
                lifecycle,
                parent_contract.get("candidate_task_id") or parent_id,
                evidence_task_id=parent_id,
                cache=cache,
            )[0]:
                blockers.append({
                    **base,
                    "code": "candidate_head_mismatch",
                    "message": "review evidence is stale",
                })
            elif row["parent_status"] != "done":
                blockers.append({
                    **base,
                    "code": "verdict_missing",
                    "message": "approved review card is not complete",
                })
        elif requirement == "validation_passed":
            projection = get_lifecycle_state(conn, parent_id, _cache=cache)
            verdict = projection.get("validation_verdict")
            if verdict is None:
                blockers.append({
                    **base,
                    "code": "verdict_missing",
                    "message": "validation pass is missing",
                })
            elif verdict == "FAIL":
                blockers.append({
                    **base,
                    "code": "verdict_conflict",
                    "message": "validation failed",
                })
            elif verdict != "PASS":
                blockers.append({
                    **base,
                    "code": "verdict_malformed",
                    "message": "validation verdict is invalid",
                })
            elif projection.get("acceptance") == "stale":
                blockers.append({
                    **base,
                    "code": "candidate_head_mismatch",
                    "message": "validation evidence is stale",
                })
            elif row["parent_status"] != "done":
                blockers.append({
                    **base,
                    "code": "verdict_missing",
                    "message": "validation card is not complete",
                })
    return {"satisfied": not blockers, "blockers": blockers}


def evaluate_dependencies(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    _cache: Optional[_ProjectionCache] = None,
) -> dict[str, Any]:
    cache = _projection_cache_for(conn, _cache)
    if cache is None:
        return _evaluate_dependencies(conn, task_id)
    if task_id not in cache.dependencies:
        cache.dependencies[task_id] = _evaluate_dependencies(conn, task_id, cache=cache)
    return cache.dependencies[task_id]


def get_lifecycle_projections(
    conn: sqlite3.Connection, task_ids: list[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    ids = list(dict.fromkeys(task_ids))
    if not ids:
        return {}, {}
    cache = _ProjectionCache(conn, ids)
    lifecycle = {
        task_id: get_lifecycle_state(conn, task_id, _cache=cache)
        for task_id in ids
    }
    dependencies = {
        task_id: evaluate_dependencies(conn, task_id, _cache=cache)
        for task_id in ids
    }
    return lifecycle, dependencies


def _forceable_dependency_override(dependencies: Mapping[str, Any]) -> bool:
    """Allow force only for unfinished parent-phase dependencies."""
    blockers = dependencies.get("blockers") or ()
    return bool(
        not dependencies.get("satisfied")
        and blockers
        and all(blocker.get("code") == "verdict_missing" for blocker in blockers)
    )


def lifecycle_metadata(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    phase: str,
    run_id: Optional[int],
    verdict: Optional[str],
    candidate_task_id: Optional[str] = None,
    head_sha: Optional[str] = None,
) -> dict[str, Any]:
    """Build an envelope from rows, never from caller identity."""
    if phase not in {"implementation", "review", "validation"}:
        raise LifecycleEvidenceError(f"unsupported lifecycle phase {phase!r}")
    if verdict is not None:
        verdict = str(verdict).strip().upper()
        if verdict not in VALID_VERDICTS:
            raise LifecycleEvidenceError(f"invalid lifecycle verdict {verdict!r}")
    task = _task_row(conn, task_id)
    contract = _row_contract(task) if task else None
    if task is None or contract is None:
        raise LifecycleEvidenceError("typed lifecycle evidence requires a classified task")
    if candidate_task_id is None:
        if phase == "implementation":
            candidate_task_id = task_id
        else:
            candidate_task_id = contract.get("candidate_task_id") or task_id
    candidate_task_id = _task_id(candidate_task_id)
    candidate = _task_row(conn, candidate_task_id)
    if candidate is None:
        raise LifecycleEvidenceError(f"candidate task {candidate_task_id} does not exist")
    if run_id is None:
        run_id = int(task["current_run_id"]) if task["current_run_id"] else None
    if run_id is None:
        raise LifecycleEvidenceError("lifecycle evidence requires a candidate run id")
    if head_sha is None:
        head_sha = _latest_head(conn, candidate_task_id)
    if not isinstance(head_sha, str) or not head_sha.strip():
        raise LifecycleEvidenceError("lifecycle evidence requires an immutable head_sha")
    candidate_goal_id = _task_goal_revision_id(candidate)
    goal_revision_ids = {candidate_task_id: candidate_goal_id}
    if task_id != candidate_task_id:
        goal_revision_ids[task_id] = _task_goal_revision_id(task)
    return {
        "schema": LIFECYCLE_SCHEMA_VERSION,
        "phase": phase,
        "candidate_task_id": candidate_task_id,
        "candidate_run_id": int(run_id),
        "head_sha": head_sha.strip(),
        "goal_revision_ids": goal_revision_ids,
        "task_goal_revision_id": _task_goal_revision_id(task),
        "verdict": verdict,
    }


def contract_requires_review(contract: Optional[Mapping[str, Any]]) -> bool:
    return bool(contract and contract.get("kind") == "code")

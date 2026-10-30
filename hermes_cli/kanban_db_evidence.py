"""Read-only, revision-fenced projections for opt-in ``kanban_show`` pages.

Continuations are ordinary tool arguments, not authority. A fresh page re-hashes
connected evidence in its caller's read transaction before returning any data.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import time

from hermes_cli import kanban_db as kb

_SECTIONS = frozenset((
    "task", "lifecycle", "effective_goal", "provenance", "dependencies",
    "dispatch_guard", "parents", "children", "comments", "events", "runs",
    "active_handoff",
))
_HISTORY = {"comments": "task_comments", "events": "task_events", "runs": "task_runs"}
_MAX_BYTES = 32768
_TOKEN = re.compile(r"kb1:([0-9a-f]{64}):(0|[1-9][0-9]{0,18})\Z")
_ENCODER = json.JSONEncoder(ensure_ascii=False, sort_keys=True, allow_nan=False)


class _Unavailable(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def _error(code):
    from tools.registry import tool_error

    return json.loads(tool_error(code, ok=False, code=code))


def _require(condition, code="invalid_evidence_request"):
    if not condition:
        raise _Unavailable(code)


def _segments(pointer):
    _require(isinstance(pointer, str), "invalid_pointer")
    if not pointer:
        return []
    _require(pointer.startswith("/") and not re.search(r"~(?![01])", pointer), "invalid_pointer")
    parts = pointer[1:].split("/")
    _require(len(parts) <= 64, "invalid_pointer")
    return [part.replace("~1", "/").replace("~0", "~") for part in parts]


def _validate(section, after, item, pointer, offset, limit, chars, snapshot):
    _require(isinstance(section, str) and section in _SECTIONS)
    for value, low, high in ((after, 0, 2**63 - 1), (offset, 0, 2**63 - 1),
                             (limit, 1, 200), (chars, 1, 8192)):
        _require(type(value) is int and low <= value <= high)
    _require(item is None or (type(item) is int and 1 <= item <= 2**63 - 1))
    _require(section in _HISTORY or (after == 0 and item is None))
    _require(not (item is not None and after))
    parts = _segments(pointer)
    _require(not (section in _HISTORY and item is None and (pointer or offset)))
    if snapshot is not None:
        _require(isinstance(snapshot, str) and _TOKEN.fullmatch(snapshot) is not None)
        _require(int(snapshot.rsplit(":", 1)[1]) <= 2**63 - 1)
    _require(snapshot is not None or not (after or item is not None or pointer or offset),
             "snapshot_required")
    return parts


def _preflight(conn):
    # No projections before this: get_effective_goal's legacy fallback attempts DDL.
    required = {
        "tasks": set(kb.Task.__dataclass_fields__) | {"worker_started_at"},
        "task_links": {"parent_id", "child_id", "requirement"},
        "task_goal_revisions": set("id task_id version title body goal_mode author created_at reason prior_version".split()),
        "task_runs": set(kb.Run.__dataclass_fields__) | {"worker_started_at"},
        "task_comments": set(kb.Comment.__dataclass_fields__),
        "task_events": set(kb.Event.__dataclass_fields__),
    }
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    _require(required.keys() <= tables, "board_unavailable")
    for table, columns in required.items():
        info = {row["name"]: row for row in conn.execute(f"PRAGMA table_info({table})")}
        _require(columns <= info.keys(), "board_unavailable")
        primary = ({"parent_id": 1, "child_id": 2} if table == "task_links" else {"id": 1})
        _require({name: row["pk"] for name, row in info.items() if row["pk"]} == primary,
                 "board_unavailable")
        for name in primary:
            expected = "TEXT" if table in ("tasks", "task_links") else "INTEGER"
            _require(info[name]["type"].upper() == expected, "board_unavailable")


def _depth(value, depth=0):
    _require(depth <= 64, "evidence_unavailable")
    if isinstance(value, dict):
        for key, child in value.items():
            _require(isinstance(key, str), "evidence_unavailable")
            _depth(child, depth + 1)
    elif isinstance(value, list):
        for child in value:
            _depth(child, depth + 1)


def _json(value):
    def pairs(items):
        result = {}
        for key, child in items:
            _require(key not in result, "evidence_unavailable")
            result[key] = child
        return result

    def constant(_):
        raise _Unavailable("evidence_unavailable")

    parsed = json.loads(value, object_pairs_hook=pairs, parse_constant=constant)
    _depth(parsed)
    return parsed


def _scope(conn, task_id):
    neighbors = {}
    for row in conn.execute("SELECT parent_id, child_id FROM task_links"):
        parent, child = row
        neighbors.setdefault(parent, set()).add(child)
        neighbors.setdefault(child, set()).add(parent)
    for row in conn.execute("SELECT id, lifecycle_contract FROM tasks"):
        try:
            contract = _json(row["lifecycle_contract"]) if row["lifecycle_contract"] is not None else None
        except (ValueError, TypeError, RecursionError):
            # Scoped rows are validated below; unrelated damaged cards are not evidence.
            continue
        candidate = contract.get("candidate_task_id") if isinstance(contract, dict) else None
        if isinstance(candidate, str) and candidate:
            neighbors.setdefault(row["id"], set()).add(candidate)
            neighbors.setdefault(candidate, set()).add(row["id"])
    scope, pending = set(), [task_id]
    while pending:
        current = pending.pop()
        if current not in scope:
            scope.add(current)
            pending.extend(neighbors.get(current, ()) - scope if current in neighbors else ())
    return sorted(scope)


def _fingerprint(conn, board, task_id, watermark):
    scope = _scope(conn, task_id)
    path = next(row[2] for row in conn.execute("PRAGMA database_list") if row[1] == "main")
    identity = os.stat(path)
    digest = hashlib.sha256()

    def feed(value):
        for chunk in _ENCODER.iterencode(value):
            digest.update(chunk.encode("utf-8"))
        digest.update(b"\n")

    feed([1, identity.st_dev, identity.st_ino, board, task_id, scope, watermark])
    ids = json.dumps(scope)
    # ponytail: O(connected evidence bytes) hashing per page; add a durable revision
    # mechanism only when measured latency warrants its schema/write-path cost.
    for table, order, where, params, json_columns in (
        ("tasks", "id", "id IN (SELECT value FROM json_each(?))", (ids,), ("lifecycle_contract", "skills")),
        ("task_links", "parent_id, child_id", "parent_id IN (SELECT value FROM json_each(?)) OR child_id IN (SELECT value FROM json_each(?))", (ids, ids), ()),
        ("task_goal_revisions", "id", "task_id IN (SELECT value FROM json_each(?))", (ids,), ()),
        ("task_runs", "id", "task_id IN (SELECT value FROM json_each(?))", (ids,), ("metadata",)),
        ("task_comments", "id", "task_id IN (SELECT value FROM json_each(?))", (ids,), ()),
        ("task_events", "id", "task_id IN (SELECT value FROM json_each(?)) AND (id <= ? OR kind NOT IN ('heartbeat', 'claim_extended'))", (ids, watermark), ("payload",)),
    ):
        feed(table)
        for row in conn.execute(f"SELECT * FROM {table} WHERE {where} ORDER BY {order}", params):
            for column in json_columns:
                if row[column] is not None:
                    _json(row[column])
            value = dict(row)
            if table in ("tasks", "task_runs"):
                value.pop("last_heartbeat_at", None)
                value.pop("claim_expires", None)
            # Preserve SQLite storage types in the digest, even for non-display columns.
            feed({key: ["blob", val.hex()] if isinstance(val, bytes) else [type(val).__name__, val]
                  for key, val in value.items()})
    return f"kb1:{digest.hexdigest()}:{watermark}"


def _fits(value):
    return len(json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")) <= _MAX_BYTES


def _continuation(base, *, item=None, pointer="", offset=0, after=None):
    result = {key: base[key] for key in ("task_id", "board", "section", "snapshot")}
    if after is not None:
        result["after"] = after
    else:
        result.update(pointer=pointer, offset=offset)
        if item is not None:
            result["item"] = item
    return result


def _pointer(parent, key):
    return parent + "/" + str(key).replace("~", "~0").replace("/", "~1")


def _node(base, value, *, item, pointer, parts, offset, limit, chars):
    _depth(value)
    value = kb.redact_review_value(value)
    for part in parts:
        if isinstance(value, dict):
            _require(part in value, "invalid_pointer")
            value = value[part]
        elif isinstance(value, list):
            _require(re.fullmatch(r"0|[1-9][0-9]*", part) is not None and len(part) <= 19,
                     "invalid_pointer")
            index = int(part)
            _require(index < len(value), "invalid_pointer")
            value = value[index]
        else:
            raise _Unavailable("invalid_pointer")
    if isinstance(value, str):
        _require(offset <= len(value))

        def chunk(length):
            end = offset + length
            result = {**base, "chunk": {"pointer": pointer, "offset": offset,
                      "text": value[offset:end], "total_chars": len(value),
                      "next_offset": end if end < len(value) else None}}
            if end < len(value):
                result["continuation"] = _continuation(base, item=item, pointer=pointer, offset=end)
            return result

        low, high = 0, min(chars, len(value) - offset)
        while low < high:
            middle = (low + high + 1) // 2
            if _fits(chunk(middle)):
                low = middle
            else:
                high = middle - 1
        result = chunk(low)
        _require(_fits(result) and (low > 0 or offset == len(value)), "evidence_unavailable")
        return result
    if not isinstance(value, (dict, list)):
        _require(offset == 0)
        result = {**base, "value": value}
        _require(_fits(result), "evidence_unavailable")
        return result
    _require(offset <= len(value))
    if offset == 0 and len(value) <= limit and _fits({**base, "value": value}):
        return {**base, "value": value}
    keys = sorted(value) if isinstance(value, dict) else range(len(value))
    entries = []

    def page(end):
        result = {**base, "entries": entries, "total_entries": len(value),
                  "next_offset": end if end < len(value) else None}
        if end < len(value):
            result["continuation"] = _continuation(base, item=item, pointer=pointer, offset=end)
        return result

    end = offset
    for key in keys[offset:offset + limit]:
        child = value[key]
        child_pointer = _pointer(pointer, key)
        kind = ("object" if isinstance(child, dict) else "array" if isinstance(child, list)
                else "string" if isinstance(child, str) else "null" if child is None
                else "boolean" if isinstance(child, bool) else "number")
        entry = {"key" if isinstance(value, dict) else "index": key,
                 "pointer": child_pointer, "type": kind, "value": child}
        entries.append(entry)
        if not _fits(page(end + 1)):
            entry.pop("value")
            entry.update(deferred=True, continuation=_continuation(base, item=item, pointer=child_pointer))
            if not _fits(page(end + 1)):
                entries.pop()
                _require(bool(entries), "evidence_unavailable")
                break
        end += 1
    result = page(end)
    _require(_fits(result), "evidence_unavailable")
    return result


def _history(conn, base, *, after, item, pointer, parts, offset, limit, chars, watermark):
    from tools.kanban_tools import _COMMENT_FIELDS, _EVENT_FIELDS, _RUN_FIELDS, _fields

    section = base["section"]
    table = _HISTORY[section]
    decoder, fields = {"comments": (kb.Comment, ("id",) + _COMMENT_FIELDS),
                       "events": (kb.Event, ("id",) + _EVENT_FIELDS),
                       "runs": (kb.Run, _RUN_FIELDS)}[section]
    query = f"SELECT * FROM {table} WHERE task_id = ? AND id {'=' if item is not None else '>'} ?"
    params = [base["task_id"], item if item is not None else after]
    if section == "events":
        query += " AND id <= ?"
        params.append(watermark)
    cursor = conn.execute(query + " ORDER BY id LIMIT ?", (*params, 1 if item is not None else limit + 1))
    if item is not None:
        row = cursor.fetchone()
        _require(row is not None)
        return _node(base, _fields(decoder.from_row(row), fields), item=item, pointer=pointer,
                     parts=parts, offset=offset, limit=limit, chars=chars)
    items = []
    last = after

    def page(more):
        result = {**base, "items": items, "returned": len(items), "has_more": more,
                  "next_after": last if more else None}
        if more:
            result["continuation"] = _continuation(base, after=last)
        return result

    row = cursor.fetchone()
    while row is not None and len(items) < limit:
        value = _fields(decoder.from_row(row), fields)
        _depth(value)
        value = kb.redact_review_value(value)
        previous = last
        last = row["id"]
        items.append(value)
        if not _fits(page(True)):
            items[-1] = {"id": last, "deferred": True, "item": last, "pointer": "",
                         "continuation": _continuation(base, item=last)}
            if not _fits(page(True)):
                items.pop()
                last = previous
                _require(bool(items), "evidence_unavailable")
                break
        row = cursor.fetchone()
    result = page(row is not None)
    _require(_fits(result), "evidence_unavailable")
    return result


def read_evidence(conn, *, board: str, task_id: str, section: str, after: int = 0,
                  item: int | None = None, pointer: str = "", offset: int = 0,
                  limit: int = 20, chars: int = 4096, snapshot: str | None = None) -> dict:
    """Return one native-JSON page, or a bounded structured refusal without row data."""
    from hermes_cli.kanban_db_dispatch import get_dispatch_guard
    from hermes_cli.kanban_db_receipts import candidate_identity
    from hermes_cli.kanban_lifecycle import evaluate_dependencies, get_lifecycle_state
    from tools.kanban_tools import _TASK_FIELDS, _fields

    try:
        parts = _validate(section, after, item, pointer, offset, limit, chars, snapshot)
        _require(isinstance(board, str) and isinstance(task_id, str))
        _require(conn.in_transaction, "board_unavailable")
        _preflight(conn)
        watermark = (int(snapshot.rsplit(":", 1)[1]) if snapshot is not None else
                     conn.execute("SELECT COALESCE(MAX(id), 0) FROM task_events").fetchone()[0])
        token = _fingerprint(conn, board, task_id, watermark)
        _require(snapshot is None or snapshot == token, "stale_snapshot")
        task = kb.get_task(conn, task_id)
        _require(task is not None, "task_not_found")
        base = {"ok": True, "task_id": task_id, "board": board, "version": task.version,
                "section": section, "snapshot": token, "evaluated_at": int(time.time())}
        _require(_fits(base), "evidence_unavailable")
        if section in _HISTORY:
            return _history(conn, base, after=after, item=item, pointer=pointer, parts=parts,
                            offset=offset, limit=limit, chars=chars, watermark=watermark)
        if section == "provenance":
            lifecycle = get_lifecycle_state(conn, task_id)
            value = candidate_identity(conn, lifecycle.get("candidate_task_id") or task_id,
                                       lifecycle.get("candidate_run_id"))
        else:
            value = {
                "task": lambda: _fields(task, _TASK_FIELDS),
                "lifecycle": lambda: get_lifecycle_state(conn, task_id),
                "effective_goal": lambda: kb.get_effective_goal(conn, task_id),
                "dependencies": lambda: evaluate_dependencies(conn, task_id),
                "dispatch_guard": lambda: get_dispatch_guard(conn, task_id, board=board),
                "parents": lambda: kb.parent_ids(conn, task_id),
                "children": lambda: kb.child_ids(conn, task_id),
                "active_handoff": lambda: kb.latest_handoff(conn, task_id),
            }[section]()
        return _node(base, value, item=None, pointer=pointer, parts=parts, offset=offset,
                     limit=limit, chars=chars)
    except _Unavailable as exc:
        return _error(exc.code)
    except sqlite3.Error:
        return _error("board_unavailable")
    except OSError:
        return _error("board_unavailable")
    except (ValueError, TypeError, KeyError, OverflowError, RecursionError):
        return _error("evidence_unavailable")

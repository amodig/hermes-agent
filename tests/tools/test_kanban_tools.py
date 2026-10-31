"""Tests for the Kanban tool surface (tools/kanban_tools.py).

Verifies:
  - Tools are gated on HERMES_KANBAN_TASK: a normal chat session sees
    zero kanban tools in its schema; a worker session sees the kanban set.
  - Each handler's happy path.
  - Error paths (missing required args, bad metadata type, etc).
"""
from __future__ import annotations

import json
import os

import pytest


# ---------------------------------------------------------------------------
# Gating
# ---------------------------------------------------------------------------

def test_kanban_tools_hidden_without_env_var(monkeypatch, tmp_path):
    """Normal `hermes chat` sessions (no HERMES_KANBAN_TASK) must have
    zero kanban_* tools in their schema."""
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))

    from tools.registry import invalidate_check_fn_cache, registry
    from toolsets import resolve_toolset

    invalidate_check_fn_cache()
    schema = registry.get_definitions(set(resolve_toolset("hermes-cli")), quiet=True)
    names = {s["function"].get("name") for s in schema if "function" in s}
    kanban = {n for n in names if n and n.startswith("kanban_")}
    assert kanban == set(), (
        f"kanban tools leaked into normal chat schema: {kanban}"
    )


# ---------------------------------------------------------------------------
# Handler happy paths
# ---------------------------------------------------------------------------

@pytest.fixture
def worker_env(monkeypatch, tmp_path):
    """Simulate being a worker: HERMES_HOME isolated, HERMES_KANBAN_TASK set
    after we've created the task."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "test-worker")
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    from pathlib import Path as _Path
    monkeypatch.setattr(_Path, "home", lambda: tmp_path)

    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="worker-test", assignee="test-worker")
        kb.claim_task(conn, tid)
        run_id = kb._current_run_id(conn, tid)
    finally:
        conn.close()
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    # A real dispatcher always pins the worker's run id; simulate that so the
    # run-lifecycle tools can prove ownership (see test_unbound_worker_cannot_mutate_card).
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    return tid


def test_show_defaults_to_env_task_id(worker_env):
    from tools import kanban_tools as kt
    out = kt._handle_show({})
    d = json.loads(out)
    assert "task" in d
    assert d["task"]["id"] == worker_env
    assert d["task"]["status"] == "running"
    assert "worker_context" in d
    assert "runs" in d


def _page_whole_file(path: str) -> str:
    """Page *path* with the real read_file tool, exactly as a worker is told to."""
    from tools.file_tools import read_file_tool

    pages: list[str] = []
    offset = 1
    for _ in range(50):
        out = json.loads(read_file_tool(path, offset=offset, limit=2000))
        assert not out.get("error"), out
        pages.append(out.get("content") or "")
        if out.get("next_offset"):
            offset = int(out["next_offset"])
        elif out.get("truncated"):
            offset += 2000
        else:
            return "\n".join(pages)
    raise AssertionError("did not reach the end of the spill within 50 pages")


def test_show_spills_but_its_decision_inputs_and_receipt_stay_pageable(tmp_path, monkeypatch):
    """A worker's kanban_show on a card carrying hundreds of KB of event history.

    Compact single-line JSON used to make the whole answer one unreadable line once
    the runtime spilled it, so a worker could not reach its own lifecycle, its
    authorization, or the candidate's launch receipt. The spill is unchanged; what
    changed is that the result is now indented, so read_file offset/limit paging
    works — and the decision inputs lead the payload instead of trailing the history.
    """
    from pathlib import Path as _Path

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "test-validator")
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setattr(_Path, "home", lambda: tmp_path)

    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd

    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    receipt = {"code_sha": "a" * 40, "generation": "b" * 64}
    head = "e" * 40
    authorization = "Do not widen the grant; report the exact host gate you ran.\n" * 100
    late_marker = "LATE-EVENT-MARKER-9c1f"

    conn = kbc.connect()
    try:
        implementation = kb.create_task(
            conn, title="candidate", assignee="coder",
            lifecycle_contract={
                "kind": "code", "review_mode": "same_card", "reviewer": "reviewer",
                "validation_required": True,
            })
        assert kb.claim_task(conn, implementation, claimer=kb._claimer_id()) is not None
        kbd._set_worker_pid(conn, implementation, os.getpid())
        impl_run = kb._current_run_id(conn, implementation)
        kb._append_event(
            conn, implementation, "spawned", {"runtime_identity": receipt}, run_id=impl_run)
        conn.execute(
            "UPDATE task_runs SET metadata = ?, status = 'review', "
            "outcome = 'review_requested' WHERE id = ?",
            (json.dumps({"head_sha": head}), impl_run))
        conn.execute(
            "UPDATE tasks SET candidate_run_id = ? WHERE id = ?", (impl_run, implementation))
        validation = kb.create_task(
            conn, title="validate the candidate", assignee="test-validator",
            body=authorization,
            lifecycle_contract={"kind": "validation", "candidate_task_id": implementation})
        # ~360 KiB of nested history, the shape the real card carried.
        for seq in range(4):
            kb._append_event(conn, validation, "note",
                             {"seq": seq, "detail": {"nested": ["x" * 300] * 300}})
        kb._append_event(conn, validation, "note", {"marker": late_marker})
        conn.commit()
    finally:
        conn.close()

    from tools import kanban_tools as kt
    from tools.tool_result_storage import PERSISTED_OUTPUT_TAG, maybe_persist_tool_result

    raw = kt._handle_show({"task_id": validation})
    assert len(raw) > 321 * 1024, len(raw)
    original = json.loads(raw)
    assert original["task"]["body"] == authorization
    assert original["lifecycle"]["candidate_run_id"] == impl_run
    assert original["provenance"]["verdict"] == "verified"
    # Key order is the fix's point: decision inputs before the history arrays.
    assert raw.index('"lifecycle"') < raw.index('"events"')
    assert raw.index('"worker_context"') < raw.index('"events"')
    assert raw.index('"provenance"') < raw.index('"events"')

    spilled = maybe_persist_tool_result(raw, "kanban_show", "tc_show_spill", env=None)
    assert PERSISTED_OUTPUT_TAG in spilled, spilled[:400]
    path = spilled.split("Full output saved to: ", 1)[1].splitlines()[0].strip()

    paged = _page_whole_file(path)
    # The last event the card wrote is reachable, so history is not the casualty.
    assert late_marker in paged
    assert f'"candidate_task_id": "{implementation}"' in paged
    assert f'"candidate_run_id": {impl_run}' in paged
    assert receipt["generation"] in paged          # the launch receipt, not a synthetic OK
    assert "candidate_missing" not in paged
    assert authorization.splitlines()[0] in paged
    assert '"worker_context"' in paged


@pytest.fixture
def evidence_board(worker_env, monkeypatch):
    from hermes_cli import kanban_db as kb, kanban_db_connect as kbc

    for key in ("HERMES_KANBAN_TASK", "HERMES_KANBAN_RUN_ID",
                "HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD"):
        monkeypatch.delenv(key, raising=False)
    conn = kbc.connect()
    try:
        yield kb, conn, worker_env
    finally:
        conn.close()


def _evidence(args, *, ok=True):
    from tools import kanban_tools as kt
    from tools.tool_result_storage import maybe_persist_tool_result

    raw = kt._handle_show(args)
    assert len(raw.encode("utf-8")) <= 32768
    assert maybe_persist_tool_result(raw, "kanban_show", "bounded-evidence", env=None) == raw
    page = json.loads(raw)
    assert page.get("ok") is ok, page
    assert "worker_context" not in page
    if not ok:
        assert set(page) == {"ok", "error", "code"}, page
    return page


def _continue_evidence(args):
    assert {"task_id", "board", "section", "snapshot"} <= set(args)
    assert "after" in args or {"pointer", "offset"} <= set(args)
    return _evidence({**args, "chars": 8192, "limit": 200})


def _reconstruct_evidence(page):
    if "value" in page:
        return page["value"]
    if "chunk" in page:
        pieces = []
        start = page["chunk"]["offset"]
        while True:
            chunk = page["chunk"]
            assert chunk["offset"] == start + sum(map(len, pieces))
            pieces.append(chunk["text"])
            if chunk["next_offset"] is None:
                assert start + sum(map(len, pieces)) == chunk["total_chars"]
                return "".join(pieces)
            assert chunk["next_offset"] > chunk["offset"]
            page = _continue_evidence(page["continuation"])
    entries = []
    total = page["total_entries"]
    while True:
        for entry in page["entries"]:
            value = (_reconstruct_evidence(_continue_evidence(entry["continuation"]))
                     if entry.get("deferred") else entry["value"])
            entries.append((entry, value))
        if page["next_offset"] is None:
            break
        page = _continue_evidence(page["continuation"])
    assert len(entries) == total
    if not entries or "key" in entries[0][0]:
        keys = [entry["key"] for entry, _ in entries]
        assert keys == sorted(set(keys))
        return {entry["key"]: value for entry, value in entries}
    assert [entry["index"] for entry, _ in entries] == list(range(total))
    return [value for _, value in entries]


def _reconstruct_history(args):
    page = _evidence(args)
    items = []
    while True:
        assert page["returned"] == len(page["items"])
        for item in page["items"]:
            value = (_reconstruct_evidence(_continue_evidence(item["continuation"]))
                     if item.get("deferred") else item)
            assert value["id"] == item["id"]
            items.append(value)
        if not page["has_more"]:
            assert page["next_after"] is None
            break
        assert page["next_after"] == items[-1]["id"]
        page = _continue_evidence(page["continuation"])
    ids = [item["id"] for item in items]
    assert ids == sorted(set(ids))
    return items


def test_bounded_show_reconstructs_large_evidence(evidence_board, monkeypatch, capsys):
    """Single-line restrictions and every history row need structured continuations."""
    kb, conn, tid = evidence_board
    import sys

    sizes = []
    read_page = _evidence

    def measured(args, **kwargs):
        page = read_page(args, **kwargs)
        sizes.append(len(json.dumps(page, ensure_ascii=False).encode("utf-8")))
        return page

    monkeypatch.setattr(sys.modules[__name__], "_evidence", measured)

    body = 'é🦄"\\' * 81000 + " LATE-RESTRICTION: never authorize production writes."
    assert len(body) > 321724
    conn.execute("UPDATE tasks SET body = ? WHERE id = ?", (body, tid))
    comment = json.dumps({"nested": [body]})
    kb.add_comment(conn, tid, author="operator", body=comment)
    for seq in range(65):
        kb._append_event(conn, tid, "note", {
            "seq": seq, "detail": {"nested": [body if seq == 32 else f"marker-{seq}"]}})
    conn.commit()
    from tools.kanban_tools import _COMMENT_FIELDS, _EVENT_FIELDS, _RUN_FIELDS, _TASK_FIELDS, _fields

    first = _evidence({"task_id": tid, "section": "lifecycle"})
    token = first["snapshot"]
    assert token.startswith("kb1:") and len(token.split(":")[1]) == 64
    args = {"task_id": tid, "snapshot": token, "limit": 200, "chars": 8192}
    task = _reconstruct_evidence(_evidence({**args, "section": "task"}))
    assert task == kb.redact_review_value(_fields(kb.get_task(conn, tid), _TASK_FIELDS))
    assert task["body"] == body
    comments = _reconstruct_history({**args, "section": "comments"})
    assert comments == kb.redact_review_value([
        _fields(c, ("id",) + _COMMENT_FIELDS) for c in kb.list_comments(conn, tid)])
    assert comments[-1]["body"] == comment
    events = _reconstruct_history({**args, "section": "events", "limit": 3})
    assert events == kb.redact_review_value([
        _fields(e, ("id",) + _EVENT_FIELDS) for e in kb.list_events(conn, tid)])
    assert [e["payload"]["seq"] for e in events if e["kind"] == "note"] == list(range(65))
    assert events[-33]["payload"]["detail"]["nested"] == [body]
    assert _reconstruct_history({**args, "section": "runs"}) == [
        _fields(r, _RUN_FIELDS) for r in kb.list_runs(conn, tid)]
    # Explicit cross-section bookend: this exact token still covers the own lifecycle.
    assert _evidence({**args, "section": "lifecycle"})["snapshot"] == token
    assert max(sizes) <= 32768
    with capsys.disabled():
        print(f"\nbounded evidence: {len(body)} body codepoints; {len(events)} history events; "
              f"{len(sizes)} pages; maximum envelope {max(sizes)} UTF-8 bytes")


@pytest.mark.parametrize("extra,code", [
    ({"limit": 2}, "invalid_evidence_request"),
    ({"section": "nope"}, "invalid_evidence_request"),
    ({"section": None}, "invalid_evidence_request"),
    ({"section": "task", "after": 0}, "invalid_evidence_request"),
    ({"section": "events", "after": True}, "invalid_evidence_request"),
    ({"section": "events", "after": -1}, "invalid_evidence_request"),
    ({"section": "events", "after": 2**63}, "invalid_evidence_request"),
    ({"section": "events", "item": True}, "invalid_evidence_request"),
    ({"section": "events", "item": 0}, "invalid_evidence_request"),
    ({"section": "events", "item": 1, "after": 0}, "invalid_evidence_request"),
    ({"section": "events", "item": None}, "invalid_evidence_request"),
    ({"section": "task", "pointer": 0}, "invalid_pointer"),
    ({"section": "task", "pointer": "body"}, "invalid_pointer"),
    ({"section": "task", "pointer": "/~2"}, "invalid_pointer"),
    ({"section": "task", "pointer": "/x" * 65}, "invalid_pointer"),
    ({"section": "task", "offset": True}, "invalid_evidence_request"),
    ({"section": "task", "offset": -1}, "invalid_evidence_request"),
    ({"section": "task", "limit": False}, "invalid_evidence_request"),
    ({"section": "task", "limit": 201}, "invalid_evidence_request"),
    ({"section": "task", "limit": 0}, "invalid_evidence_request"),
    ({"section": "task", "chars": True}, "invalid_evidence_request"),
    ({"section": "task", "chars": 8193}, "invalid_evidence_request"),
    ({"section": "task", "chars": 0}, "invalid_evidence_request"),
    ({"section": "task", "chars": 1.5}, "invalid_evidence_request"),
    ({"section": "task", "snapshot": None}, "invalid_evidence_request"),
    ({"section": "task", "snapshot": "forged"}, "invalid_evidence_request"),
    ({"section": "task", "pointer": "/body"}, "snapshot_required"),
    ({"section": "task", "offset": 1}, "snapshot_required"),
    ({"section": "events", "after": 1}, "snapshot_required"),
    ({"section": "events", "item": 1}, "snapshot_required"),
])
def test_bounded_show_rejects_invalid_arguments(evidence_board, extra, code):
    _, _, tid = evidence_board
    assert _evidence({"task_id": tid, **extra}, ok=False)["code"] == code


def test_bounded_show_pointer_boundaries_and_row_isolation(evidence_board):
    kb, conn, tid = evidence_board
    kb._append_event(conn, tid, "note", {"a/~": ["", "hello", None], "": "empty-key"})
    event = conn.execute("SELECT MAX(id) FROM task_events").fetchone()[0]
    other = kb.create_task(conn, title="other")
    foreign = kb.add_comment(conn, other, author="operator", body="foreign")
    conn.commit()
    token = _evidence({"task_id": tid, "section": "task"})["snapshot"]
    args = {"task_id": tid, "section": "events", "item": event, "snapshot": token}
    assert _reconstruct_evidence(_evidence({**args, "pointer": "/payload/a~1~0/1"})) == "hello"
    assert _reconstruct_evidence(_evidence({**args, "pointer": "/payload/"})) == "empty-key"
    empty = _evidence({**args, "pointer": "/payload/a~1~0/0"})
    assert empty["chunk"]["text"] == "" and empty["chunk"]["next_offset"] is None
    end = _evidence({**args, "pointer": "/payload/a~1~0/1", "offset": 5})
    assert end["chunk"]["text"] == "" and end["chunk"]["next_offset"] is None
    for pointer in ("/payload/a~1~0/01", "/payload/a~1~0/-", "/payload/a~1~0/3", "/absent"):
        assert _evidence({**args, "pointer": pointer}, ok=False)["code"] == "invalid_pointer"
    assert _evidence({**args, "pointer": "/payload/a~1~0/1", "offset": 6},
                     ok=False)["code"] == "invalid_evidence_request"
    assert _evidence({**args, "section": "comments", "item": foreign},
                     ok=False)["code"] == "invalid_evidence_request"
    empty_history = _evidence({"task_id": tid, "section": "comments"})
    assert empty_history["items"] == [] and empty_history["next_after"] is None
    assert _evidence({"task_id": "missing", "section": "task"},
                     ok=False)["code"] == "task_not_found"


@pytest.fixture
def evidence_graph(evidence_board):
    kb, conn, tid = evidence_board
    parent = kb.create_task(conn, title="parent goal", body="goal restrictions")
    candidate = kb.create_task(conn, title="candidate", lifecycle_contract={
        "kind": "code", "review_mode": "separate_card", "reviewer": "reviewer",
        "validation_required": True})
    review = kb.create_task(conn, title="review", assignee="reviewer", lifecycle_contract={
        "kind": "review", "candidate_task_id": candidate})
    # A reverse-only lifecycle reference must join the scope even without an edge.
    reverse = kb.create_task(conn, title="other review", assignee="reviewer", lifecycle_contract={
        "kind": "review", "candidate_task_id": candidate})
    conn.execute("UPDATE tasks SET lifecycle_contract = ? WHERE id = ?",
                 (json.dumps({"kind": "validation", "candidate_task_id": candidate}), tid))
    for left, right in ((parent, candidate), (candidate, review), (review, tid)):
        conn.execute("INSERT OR IGNORE INTO task_links(parent_id, child_id) VALUES (?, ?)", (left, right))
    run = conn.execute(
        "INSERT INTO task_runs(task_id, status, started_at, metadata) VALUES (?, 'done', 1, ?)",
        (candidate, json.dumps({"head_sha": "a" * 40}))).lastrowid
    conn.execute("UPDATE tasks SET candidate_run_id = ? WHERE id = ?", (run, candidate))
    kb._append_event(conn, review, "note", {"lifecycle": {"verdict": "APPROVE"}})
    event = conn.execute("SELECT MAX(id) FROM task_events").fetchone()[0]
    comment = kb.add_comment(conn, parent, author="operator", body="authorization")
    conn.commit()
    return kb, conn, {"tid": tid, "parent": parent, "candidate": candidate,
                      "review": review, "reverse": reverse, "run": run,
                      "event": event, "comment": comment}


@pytest.mark.parametrize("mutation", [
    "own_body", "own_version", "parent_goal", "candidate_head", "candidate_run",
    "review_verdict", "run_metadata", "new_link", "delete_link", "new_event",
    "new_comment", "edit_event", "delete_event", "edit_comment", "delete_comment",
    "reverse_reference", "claim_lock", "receipt",
])
def test_bounded_show_cross_section_snapshot_rejects_semantic_mutations(evidence_graph, mutation):
    import sqlite3

    kb, conn, ids = evidence_graph
    tid = ids["tid"]
    token = _evidence({"task_id": tid, "section": "lifecycle"})["snapshot"]
    version = kb.get_task(conn, tid).version
    writer = sqlite3.connect(kb.kanban_db_path())
    try:
        statements = {
            "own_body": ("UPDATE tasks SET body = 'changed' WHERE id = ?", (tid,)),
            "own_version": ("UPDATE tasks SET version = version + 1 WHERE id = ?", (tid,)),
            "parent_goal": ("UPDATE task_goal_revisions SET body = 'changed restriction' WHERE task_id = ?", (ids["parent"],)),
            "candidate_head": ("UPDATE task_runs SET metadata = ? WHERE id = ?", (json.dumps({"head_sha": "b" * 40}), ids["run"])),
            "candidate_run": ("UPDATE tasks SET candidate_run_id = NULL WHERE id = ?", (ids["candidate"],)),
            "review_verdict": ("UPDATE task_events SET payload = ? WHERE id = ?", (json.dumps({"lifecycle": {"verdict": "REQUEST_CHANGES"}}), ids["event"])),
            "run_metadata": ("UPDATE task_runs SET metadata = ? WHERE task_id = ?", (json.dumps({"authorization": "new"}), tid)),
            "new_link": ("INSERT INTO task_links(parent_id, child_id) VALUES (?, ?)", (ids["parent"], tid)),
            "delete_link": ("DELETE FROM task_links WHERE parent_id = ? AND child_id = ?", (ids["review"], tid)),
            "new_event": ("INSERT INTO task_events(task_id, kind, payload, created_at) VALUES (?, 'note', '{}', 1)", (ids["parent"],)),
            "new_comment": ("INSERT INTO task_comments(task_id, author, body, created_at) VALUES (?, 'operator', 'new authorization', 1)", (ids["parent"],)),
            "edit_event": ("UPDATE task_events SET created_at = created_at + 1 WHERE id = ?", (ids["event"],)),
            "delete_event": ("DELETE FROM task_events WHERE id = ?", (ids["event"],)),
            "edit_comment": ("UPDATE task_comments SET body = 'revoked' WHERE id = ?", (ids["comment"],)),
            "delete_comment": ("DELETE FROM task_comments WHERE id = ?", (ids["comment"],)),
            "reverse_reference": ("UPDATE tasks SET body = 'reverse-only edit' WHERE id = ?", (ids["reverse"],)),
            "claim_lock": ("UPDATE tasks SET claim_lock = 'new-owner' WHERE id = ?", (ids["candidate"],)),
            "receipt": ("INSERT INTO task_events(task_id, run_id, kind, payload, created_at) VALUES (?, ?, 'spawned', '{}', 1)", (ids["candidate"], ids["run"])),
        }
        writer.execute(*statements[mutation])
        writer.commit()
    finally:
        writer.close()
    if mutation != "own_version":
        assert kb.get_task(conn, tid).version == version
    for section in ("task", "comments", "events"):
        assert _evidence({"task_id": tid, "section": section, "snapshot": token},
                         ok=False)["code"] == "stale_snapshot"


@pytest.mark.parametrize("edit", ["update", "delete"])
def test_bounded_show_pins_heartbeat_history_without_starvation(evidence_board, edit):
    kb, conn, tid = evidence_board
    kb._append_event(conn, tid, "heartbeat", {"note": "old heartbeat note"})
    pinned = conn.execute("SELECT MAX(id) FROM task_events").fetchone()[0]
    conn.commit()
    token = _evidence({"task_id": tid, "section": "events"})["snapshot"]
    for table in ("tasks", "task_runs"):
        key = "id" if table == "tasks" else "task_id"
        conn.execute(f"UPDATE {table} SET last_heartbeat_at = 999, claim_expires = 999 WHERE {key} = ?", (tid,))
    for kind in ("heartbeat", "claim_extended"):
        kb._append_event(conn, tid, kind, {"note": "new lease note"})
    conn.commit()
    args = {"task_id": tid, "section": "events", "snapshot": token}
    events = _reconstruct_history(args)
    assert events[-1]["id"] == pinned and events[-1]["payload"]["note"] == "old heartbeat note"
    assert _evidence({**args, "section": "task"})["snapshot"] == token
    if edit == "update":
        conn.execute("UPDATE task_events SET payload = ? WHERE id = ?",
                     (json.dumps({"note": "silently edited"}), pinned))
    else:
        conn.execute("DELETE FROM task_events WHERE id = ?", (pinned,))
    conn.commit()
    assert _evidence(args, ok=False)["code"] == "stale_snapshot"


def test_bounded_show_snapshot_binds_task_board_and_file(evidence_board, monkeypatch, tmp_path):
    import sqlite3

    kb, conn, tid = evidence_board
    other = kb.create_task(conn, title="other task")
    token = _evidence({"task_id": tid, "section": "task"})["snapshot"]
    assert _evidence({"task_id": other, "section": "task", "snapshot": token},
                     ok=False)["code"] == "stale_snapshot"
    with monkeypatch.context() as patch, kb.pin_first_board_resolution():
        patch.setenv("HERMES_KANBAN_DB", str(kb.kanban_db_path()))
        assert _evidence({"task_id": tid, "board": "other-board", "section": "task", "snapshot": token},
                         ok=False)["code"] == "stale_snapshot"
    replacement = tmp_path / "replacement.db"
    target = sqlite3.connect(replacement)
    conn.backup(target)
    target.close()
    # Same exact evidence bytes at another inode cannot reuse a token.
    with monkeypatch.context() as patch:
        patch.setenv("HERMES_KANBAN_DB", str(replacement))
        assert _evidence({"task_id": tid, "section": "task", "snapshot": token},
                         ok=False)["code"] == "stale_snapshot"
    # A replacement at the SAME board path is equally fenced.
    with monkeypatch.context() as patch:
        patch.setenv("HERMES_KANBAN_DB", str(replacement))
        copied_token = _evidence({"task_id": tid, "section": "task"})["snapshot"]
        second = tmp_path / "second.db"
        target = sqlite3.connect(second)
        conn.backup(target)
        target.close()
        second.replace(replacement)
        assert _evidence({"task_id": tid, "section": "task", "snapshot": copied_token},
                         ok=False)["code"] == "stale_snapshot"


def test_bounded_show_projection_and_fingerprint_share_one_transaction(evidence_board, monkeypatch):
    kb, conn, tid = evidence_board
    from hermes_cli import kanban_db_evidence as evidence

    original = kb.get_task(conn, tid).body
    fingerprint = evidence._fingerprint

    def interleave(reader, *args):
        token = fingerprint(reader, *args)
        conn.execute("UPDATE tasks SET body = 'concurrent authorization' WHERE id = ?", (tid,))
        conn.commit()
        return token

    with monkeypatch.context() as patch:
        patch.setattr(evidence, "_fingerprint", interleave)
        first = _evidence({"task_id": tid, "section": "task", "limit": 200})
        assert first["value"]["body"] == original
    assert _evidence({"task_id": tid, "section": "task", "snapshot": first["snapshot"]},
                     ok=False)["code"] == "stale_snapshot"


@pytest.mark.parametrize("column,value", [
    ("payload", "{"), ("metadata", "{"), ("lifecycle_contract", "{"),
    ("payload", '{"duplicate":1,"duplicate":2}'),
    ("payload", "[" * 65 + "0" + "]" * 65),
    ("payload", '{"bad":NaN}'),
])
def test_bounded_show_refuses_damaged_evidence(evidence_board, column, value):
    kb, conn, tid = evidence_board
    table = {"payload": "task_events", "metadata": "task_runs", "lifecycle_contract": "tasks"}[column]
    key = "id" if table == "tasks" else "task_id"
    conn.execute(f"UPDATE {table} SET {column} = ? WHERE {key} = ?", (value, tid))
    conn.commit()
    assert _evidence({"task_id": tid, "section": "task"},
                     ok=False)["code"] == "evidence_unavailable"


def test_bounded_show_redacts_complete_values_before_chunking(evidence_board):
    kb, conn, tid = evidence_board
    secret = "sk-" + "aB3x" * 12
    prefix = ("public restriction " * 500)[:8179] + " api_key="
    body = prefix + secret + " no production writes"
    assert len(prefix) < 8192 < len(prefix + secret)
    conn.execute("UPDATE tasks SET body = ? WHERE id = ?", (body, tid))
    conn.commit()
    token = _evidence({"task_id": tid, "section": "task"})["snapshot"]
    page = _evidence({"task_id": tid, "section": "task", "pointer": "/body",
                      "snapshot": token, "chars": 8192})
    result = _reconstruct_evidence(page)
    assert result == kb.redact_review_value(body)
    assert secret not in result and result.endswith(" no production writes")


def test_bounded_show_escape_budget_and_nested_run_arrays(evidence_board, capsys):
    kb, conn, tid = evidence_board
    body = "\x00" * 20000 + "é🦄 restriction: no production writes"
    conn.execute("UPDATE tasks SET body = ? WHERE id = ?", (body, tid))
    metadata = {"a/~": [{"index": index, "restriction": "no writes " * 50} for index in range(250)]}
    conn.execute("UPDATE task_runs SET metadata = ? WHERE task_id = ?", (json.dumps(metadata), tid))
    conn.commit()
    token = _evidence({"task_id": tid, "section": "task"})["snapshot"]
    args = {"task_id": tid, "section": "task", "snapshot": token, "pointer": "/body", "chars": 8192}
    first = _evidence(args)
    assert len(first["chunk"]["text"]) < 8192
    assert _reconstruct_evidence(first) == body
    size = len(json.dumps(first, ensure_ascii=False).encode("utf-8"))
    assert 32760 <= size <= 32768
    runs = _reconstruct_history({**args, "section": "runs", "pointer": ""})
    assert runs[0]["metadata"] == metadata
    end = _evidence({"task_id": tid, "section": "runs", "item": runs[0]["id"],
                     "snapshot": token, "pointer": "/metadata/a~1~0", "offset": 250})
    assert end["entries"] == [] and end["next_offset"] is None
    with capsys.disabled():
        print(f"\nbounded escaping: first near-ceiling envelope {size} UTF-8 bytes")


def test_bounded_show_reuses_all_native_section_projections(evidence_graph):
    kb, conn, ids = evidence_graph
    from hermes_cli.kanban_db_receipts import candidate_identity

    tid = ids["tid"]
    lifecycle = kb.get_lifecycle_state(conn, tid)
    expected = {
        "lifecycle": lifecycle,
        "effective_goal": kb.get_effective_goal(conn, tid),
        "dependencies": kb.evaluate_dependencies(conn, tid),
        "dispatch_guard": kb.get_dispatch_guard(conn, tid, board="default"),
        "parents": kb.parent_ids(conn, tid),
        "children": kb.child_ids(conn, tid),
        "active_handoff": kb.latest_handoff(conn, tid),
        "provenance": candidate_identity(conn, lifecycle["candidate_task_id"],
                                         lifecycle["candidate_run_id"]),
    }
    token = _evidence({"task_id": tid, "section": "lifecycle"})["snapshot"]
    for section, value in expected.items():
        assert _reconstruct_evidence(_evidence({
            "task_id": tid, "section": section, "snapshot": token,
        })) == kb.redact_review_value(value)


@pytest.mark.parametrize("schema", ["absent", "empty", "legacy", "missing_goal_column", "missing_primary_key"])
def test_bounded_show_never_initializes_or_migrates(evidence_board, monkeypatch, tmp_path, schema):
    import sqlite3
    from hermes_cli import worker_startup

    kb, _, tid = evidence_board
    path = tmp_path / "incompatible.db"
    if schema != "absent":
        writer = sqlite3.connect(path)
        if schema == "legacy":
            writer.execute("CREATE TABLE tasks(id TEXT PRIMARY KEY)")
        elif schema == "missing_goal_column":
            writer.executescript(kb.SCHEMA_SQL)
            writer.execute("ALTER TABLE task_goal_revisions DROP COLUMN reason")
        elif schema == "missing_primary_key":
            writer.executescript(kb.SCHEMA_SQL)
            writer.executescript(
                "CREATE TABLE malformed AS SELECT * FROM task_comments; "
                "DROP TABLE task_comments; ALTER TABLE malformed RENAME TO task_comments;")
        writer.close()
    before = path.read_bytes() if path.exists() else None
    monkeypatch.setenv("HERMES_KANBAN_DB", str(path))
    monkeypatch.setattr(kb, "_ensure_goal_revision_schema",
                        lambda *_: pytest.fail("attempted migration"))
    real_open = worker_startup._open_board
    statements = []

    def read_only(path):
        reader = real_open(path)
        reader.set_trace_callback(statements.append)
        return reader

    monkeypatch.setattr(worker_startup, "_open_board", read_only)
    assert _evidence({"task_id": tid, "section": "effective_goal"},
                     ok=False)["code"] == "board_unavailable"
    assert (path.read_bytes() if path.exists() else None) == before
    assert all(statement.startswith(("SELECT ", "PRAGMA table_info(", "ROLLBACK"))
               for statement in statements)


def test_bounded_show_read_only_authorizer_and_unrepresentable_descriptor(evidence_board, monkeypatch):
    import sqlite3
    from hermes_cli import worker_startup

    kb, conn, tid = evidence_board
    connect = sqlite3.connect
    actions = []

    def readonly_connect(*args, **kwargs):
        reader = connect(*args, **kwargs)

        def authorize(action, first, second, *_):
            actions.append((action, first, second))
            allowed = action in (sqlite3.SQLITE_READ, sqlite3.SQLITE_SELECT,
                                 sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_TRANSACTION,
                                 sqlite3.SQLITE_RECURSIVE)
            allowed |= action == sqlite3.SQLITE_PRAGMA and (
                first in ("table_info", "database_list")
                or (first == "query_only" and second == "ON"))
            return sqlite3.SQLITE_OK if allowed else sqlite3.SQLITE_DENY

        reader.set_authorizer(authorize)
        return reader

    with monkeypatch.context() as patch:
        patch.setattr(worker_startup.sqlite3, "connect", readonly_connect)
        _evidence({"task_id": tid, "section": "effective_goal"})
        _evidence({"task_id": tid, "section": "dispatch_guard"})
    assert actions
    kb._append_event(conn, tid, "note", {"x" * 40000: "value"})
    conn.commit()
    page = _evidence({"task_id": tid, "section": "events"})
    row = page["items"][-1]
    assert row["deferred"]
    item = _continue_evidence(row["continuation"])
    payload = next(entry for entry in item["entries"] if entry["key"] == "payload")
    assert _evidence(payload["continuation"], ok=False)["code"] == "evidence_unavailable"


@pytest.fixture
def evidence_worker(evidence_board, monkeypatch, tmp_path):
    from dataclasses import replace
    from hermes_cli import kanban_runtime as runtime, worker_startup

    kb, conn, tid = evidence_board
    identity = replace(runtime.runtime_identity(), generation="fixture-generation",
                       dependency_fingerprint="fixture-dependencies")
    monkeypatch.setattr(runtime, "runtime_identity", lambda: identity)
    monkeypatch.setattr(runtime, "assert_runtime_import_root", lambda **_: identity)
    monkeypatch.setattr(worker_startup, "_GRANT", None)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    run = kb._current_run_id(conn, tid)
    claim = kb.get_task(conn, tid).claim_lock
    metadata = {"runtime_identity": identity.as_dict(), "worker_pid": identity.pid,
                "worker_start_time": identity.start_time, "preparation_id": "preparation"}
    conn.execute("UPDATE tasks SET worker_pid = ?, worker_started_at = NULL, workspace_path = ? WHERE id = ?",
                 (identity.pid, str(workspace), tid))
    conn.execute("UPDATE task_runs SET worker_pid = ?, worker_started_at = NULL, metadata = ? WHERE id = ?",
                 (identity.pid, json.dumps(metadata), run))
    parent = kb.create_task(conn, title="same-board parent")
    conn.execute("INSERT INTO task_links(parent_id, child_id) VALUES (?, ?)", (parent, tid))
    conn.commit()
    path = str(kb.kanban_db_path())
    for key, value in {
        "HERMES_KANBAN_TASK": tid, "HERMES_KANBAN_RUN_ID": str(run),
        "HERMES_KANBAN_BOARD": "default", "HERMES_KANBAN_DB": path,
        "HERMES_PROFILE": "test-worker", "HERMES_KANBAN_WORKSPACE": str(workspace),
        "HERMES_KANBAN_CLAIM_LOCK": claim, "HERMES_KANBAN_RUNTIME_GRANTED": "1",
    }.items():
        monkeypatch.setenv(key, value)
    worker_startup._retain_worker_grant({
        "grant": True, "runtime_identity": identity.as_dict(), "preparation_id": "preparation",
        "run_id": run, "claim_lock": claim,
        "worker_context": {"task_id": tid, "profile": "test-worker", "board": "default",
                           "workspace": str(workspace), "db_path": path},
    }, identity)
    return kb, conn, tid, parent


def test_bounded_show_authenticated_worker_can_read_same_board_parent(evidence_worker, monkeypatch):
    from hermes_cli import worker_startup

    kb, conn, tid, parent = evidence_worker
    assert _evidence({"section": "task"})["task_id"] == tid
    assert _evidence({"section": "task", "task_id": parent})["task_id"] == parent
    assert _evidence({"section": "task", "board": "other"},
                     ok=False)["code"] == "invalid_evidence_request"
    conn.execute("UPDATE tasks SET claim_lock = 'successor' WHERE id = ?", (tid,))
    conn.commit()
    # Parent access must also recheck the worker's OWN claim in the same transaction.
    assert _evidence({"section": "task", "task_id": parent},
                     ok=False)["code"] == "invalid_evidence_request"


@pytest.mark.parametrize("key", [
    "HERMES_KANBAN_TASK", "HERMES_KANBAN_RUN_ID", "HERMES_KANBAN_DB",
    "HERMES_KANBAN_BOARD", "HERMES_PROFILE", "HERMES_KANBAN_WORKSPACE",
    "HERMES_KANBAN_CLAIM_LOCK", "HERMES_KANBAN_RUNTIME_GRANTED",
    "HERMES_KANBAN_PREPARATION_ID",
])
def test_bounded_show_refuses_conflicting_inherited_worker_state(evidence_worker, monkeypatch, key):
    _, _, tid, _ = evidence_worker
    monkeypatch.setenv(key, "foreign")
    assert _evidence({"section": "task", "task_id": tid},
                     ok=False)["code"] == "invalid_evidence_request"


def test_bounded_show_refuses_missing_grants_and_delegation(evidence_worker, monkeypatch):
    from agent.delegation_context import delegated_child_context, non_dispatcher_owned_context
    from hermes_cli import worker_startup

    _, _, tid, parent = evidence_worker
    for context in (delegated_child_context, non_dispatcher_owned_context):
        with context():
            assert _evidence({"section": "task", "task_id": parent},
                             ok=False)["code"] == "invalid_evidence_request"
    with monkeypatch.context() as patch:
        patch.setenv("HERMES_DELEGATED_CHILD_CONTEXT", "1")
        assert _evidence({"section": "task", "task_id": tid},
                         ok=False)["code"] == "invalid_evidence_request"
    monkeypatch.setattr(worker_startup, "_GRANT", None)
    assert _evidence({"section": "task", "task_id": tid},
                     ok=False)["code"] == "invalid_evidence_request"


def test_show_bare_call_outside_worker_returns_orientation_not_error(monkeypatch, worker_env):
    """#91431: chat profiles with the kanban toolset call kanban_show bare to orient
    themselves; with no dispatcher task in scope there is nothing to show, so the
    answer must be a pointer to kanban_list — not an error naming an env var no
    chat caller can set."""
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    from tools import kanban_tools as kt
    out = json.loads(kt._handle_show({}))
    assert "error" not in out, out
    assert out.get("current_task") is None
    assert "kanban_list" in out["hint"]


def test_comment_bare_call_outside_worker_names_kanban_list(monkeypatch, worker_env):
    """#91431 sibling: a comment cannot post without a target task, so a bare call
    stays an error — but the error must point at kanban_list instead of the env."""
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    from tools import kanban_tools as kt
    out = json.loads(kt._handle_comment({"body": "hello"}))
    assert out.get("error"), out
    assert "kanban_list" in out["error"]
    assert "set HERMES_KANBAN_TASK" not in out["error"]


def test_worker_tool_bare_call_outside_worker_error_is_actionable(monkeypatch, worker_env):
    """#91431 sibling: the mutation/lifecycle tools still require a target, but the
    rejection a non-worker receives must name what it can do (kanban_list), not the
    env var it cannot set."""
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    from tools import kanban_tools as kt
    for handler, name in ((kt._handle_heartbeat, "kanban_heartbeat"),
                           (kt._handle_attachments, "kanban_attachments")):
        out = json.loads(handler({}))
        assert out.get("error"), (name, out)
        assert "kanban_list" in out["error"], (name, out)
        assert "set HERMES_KANBAN_TASK" not in out["error"], (name, out)


def test_kanban_task_id_descriptions_state_the_non_worker_case():
    """#91431 contract: any task_id description that promises the HERMES_KANBAN_TASK
    default must also say the default only exists for dispatcher-spawned workers —
    an unqualified invite to omit the argument is what made chat profiles call bare."""
    from tools import kanban_tools_schemas as kts

    seen = []
    for attr, schema in vars(kts).items():
        if not (attr.startswith("KANBAN_") and attr.endswith("_SCHEMA")):
            continue
        desc = (schema["parameters"]["properties"].get("task_id") or {}).get("description") or ""
        if "HERMES_KANBAN_TASK" not in desc:
            continue
        assert "worker" in desc, (
            schema["name"], "env default promised without the non-worker case (#91431)")
        assert "no default" in desc or "orientation" in desc, (
            schema["name"], "must state what a non-worker bare call does (#91431)")
        seen.append(schema["name"])
    assert "kanban_show" in seen


def test_list_filters_tasks(monkeypatch, worker_env):
    """kanban_list gives orchestrators filtered board discovery."""
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    conn = kbc.connect()
    try:
        a = kb.create_task(conn, title="alpha", assignee="factory", priority=5)
        b = kb.create_task(conn, title="beta", assignee="reviewer")
        c = kb.create_task(conn, title="gamma", assignee="factory", tenant="other")
    finally:
        conn.close()

    from tools import kanban_tools as kt
    out = kt._handle_list({"assignee": "factory", "status": "ready", "limit": 10})
    d = json.loads(out)
    ids = [t["id"] for t in d["tasks"]]
    assert ids == [a, c]
    assert d["count"] == 2
    assert d["tasks"][0]["title"] == "alpha"
    assert d["tasks"][0]["parent_count"] == 0
    assert b not in ids

    tenant_out = kt._handle_list({
        "assignee": "factory",
        "status": "ready",
        "tenant": "other",
    })
    tenant_ids = [t["id"] for t in json.loads(tenant_out)["tasks"]]
    assert tenant_ids == [c]


def test_complete_happy_path(worker_env):
    from tools import kanban_tools as kt
    out = kt._handle_complete({
        "summary": "got the thing done",
        "metadata": {"files": 2},
    })
    d = json.loads(out)
    assert d["ok"] is True
    assert d["task_id"] == worker_env
    # Verify via kernel
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    conn = kbc.connect()
    try:
        run = kb.latest_run(conn, worker_env)
        assert run.outcome == "completed"
        assert run.summary == "got the thing done"
        # Caller metadata must survive; the run also carries additive provenance.
        assert run.metadata["files"] == 2
    finally:
        conn.close()


def test_complete_retry_with_empty_created_cards_succeeds(worker_env):
    """After a phantom rejection, retrying kanban_complete with
    created_cards=[] (the documented escape hatch) must complete the
    task. Regression for #22923."""
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from tools import kanban_tools as kt

    # Hit the gate first.
    rejected = json.loads(kt._handle_complete({
        "summary": "oops",
        "created_cards": ["t_phantomdeadbeef"],
    }))
    assert rejected.get("error")

    # Retry with the escape hatch.
    ok = json.loads(kt._handle_complete({
        "summary": "retry without claims",
        "created_cards": [],
    }))
    assert ok.get("ok") is True

    conn = kbc.connect()
    try:
        assert kb.get_task(conn, worker_env).status == "done"
    finally:
        conn.close()


def test_complete_reports_registered_attachments(worker_env):
    """#117360: artifact staging is atomic with the completion write, so the
    worker's pre-completion `kanban_attachments` readback is always empty and
    workers narrated "registered at completion: none" even when the rows landed.
    The completion result must report the card's durable attachment set, in the
    same shape the readback tool returns."""
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_workspace as kbw
    from tools import kanban_tools as kt

    with kbc.connect() as conn:
        task = kb.get_task(conn, worker_env)
        ws = kbw.resolve_workspace(task)
        kbw.set_workspace_path(conn, worker_env, ws)
    artifact = ws / "corpus.json"
    artifact.write_bytes(b"{}")

    out = kt._handle_complete({
        "summary": "done",
        "artifacts": [str(artifact)],
    })
    d = json.loads(out)
    assert d["ok"] is True, d
    assert [(a["filename"], a["size"], a["uploaded_by"]) for a in d["attachments"]] == [
        ("corpus.json", 2, "kanban_complete")]

    readback = json.loads(kt._handle_attachments({"task_id": worker_env}))
    assert readback["attachments"] == d["attachments"]


def test_request_review_rejects_unknown_reviewer_without_mutation(monkeypatch, worker_env, tmp_path):
    """#106163: a non-profile ``reviewer`` (e.g. the literal "reviewer") must be
    refused with an error the model sees, leaving the task running under the
    implementer — never parked in ``review`` on an assignee nobody can spawn."""
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from tools import kanban_tools as kt

    (tmp_path / ".hermes" / "profiles" / "verifier").mkdir(parents=True)
    (tmp_path / ".hermes" / "profiles" / "verifier" / "config.yaml").write_text("{}\n", encoding="utf-8")  # identity marker
    with kbc.connect() as conn:
        before = kb.get_task(conn, worker_env)
        before_events = kb.list_events(conn, worker_env)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(before.current_run_id))

    out = json.loads(kt._handle_request_review({"summary": "Ready for review.", "reviewer": "reviewer"}))

    assert "'reviewer'" in out["error"] and "verifier" in out["error"]
    with kbc.connect() as conn:
        after = kb.get_task(conn, worker_env)
        assert (after.status, after.assignee, after.current_run_id) == ("running", "test-worker", before.current_run_id)
        assert kb.list_events(conn, worker_env) == before_events


def test_request_review_accepts_installed_profile(monkeypatch, worker_env, tmp_path):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from tools import kanban_tools as kt

    (tmp_path / ".hermes" / "profiles" / "verifier").mkdir(parents=True)
    (tmp_path / ".hermes" / "profiles" / "verifier" / "config.yaml").write_text("{}\n", encoding="utf-8")  # identity marker
    with kbc.connect() as conn:
        monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(kb.get_task(conn, worker_env).current_run_id))

    out = json.loads(kt._handle_request_review({"summary": "Ready for review.", "reviewer": "verifier"}))

    assert out["ok"] is True
    with kbc.connect() as conn:
        task = kb.get_task(conn, worker_env)
        assert (task.status, task.assignee) == ("review", "verifier")


def test_unbound_worker_cannot_mutate_card(monkeypatch, worker_env):
    """A dispatcher-spawned worker that cannot resolve its run id must be refused
    on every run-lifecycle mutation. ``expected_run_id=None`` would silently skip
    the run-ownership CAS in kanban_db, so an unbound stale worker could complete
    a card a live successor owns (regression for #116239)."""
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from tools import kanban_tools as kt

    # Worker is scoped to the task (HERMES_KANBAN_TASK set by the fixture) but
    # has NO run id — the unbound state the dispatcher never produces.
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)

    for handler, args in [
        (kt._handle_complete, {"summary": "stale worker says done"}),
        (kt._handle_block, {"reason": "stale worker blocks"}),
        (kt._handle_schedule, {"reason": "stale worker parks"}),
        (kt._handle_request_review, {"summary": "stale worker hands off"}),
        (kt._handle_request_changes, {"reason": "stale worker requests changes"}),
    ]:
        out = json.loads(handler(args))
        assert "refused" in out.get("error", ""), f"{handler.__name__} did not refuse: {out}"

    # Nothing moved: the card is still running under its original run.
    with kbc.connect() as conn:
        task = kb.get_task(conn, worker_env)
        assert task.status == "running"
        assert task.current_run_id is not None

    # A bound worker (run id present) still completes normally — the guard only
    # fires on the unbound state, never on the legitimate dispatcher path.
    with kbc.connect() as conn:
        run_id = kb.get_task(conn, worker_env).current_run_id
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    out = json.loads(kt._handle_complete({"summary": "bound worker done"}))
    assert out.get("ok") is True
    with kbc.connect() as conn:
        assert kb.get_task(conn, worker_env).status == "done"


def test_malformed_run_id_refused_but_nonlifecycle_allowed(monkeypatch, worker_env):
    """A malformed (non-integer) HERMES_KANBAN_RUN_ID is treated as unbound and
    refuses run-lifecycle mutations, while non-lifecycle tools (heartbeat /
    attach) that do not terminate a run stay available to the worker."""
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from tools import kanban_tools as kt

    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "not-an-int")

    # Run-lifecycle mutations are refused on a malformed run id.
    out = json.loads(kt._handle_complete({"summary": "stale worker says done"}))
    assert "refused" in out.get("error", "")
    out = json.loads(kt._handle_schedule({"reason": "stale worker parks"}))
    assert "refused" in out.get("error", "")

    # Non-lifecycle tools are NOT gated: heartbeat still extends the claim.
    out = json.loads(kt._handle_heartbeat({}))
    assert out.get("ok") is True
    with kbc.connect() as conn:
        assert kb.get_task(conn, worker_env).status == "running"


def test_complete_goal_mode_rejected_by_judge(monkeypatch, tmp_path):
    """Goal-mode tasks must pass the auxiliary judge before completion.
    Regression for #38367: workers bypassing the judge via early kanban_complete."""
    from pathlib import Path as _Path
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from tools import kanban_tools as kt

    # Set up isolated HERMES_HOME
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "test-worker")
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    monkeypatch.setattr(_Path, "home", lambda: tmp_path)

    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    conn = kbc.connect()
    try:
        goal_task_id = kb.create_task(
            conn, title="goal-mode-test", assignee="test-worker",
            body="Must achieve X with verified evidence.", goal_mode=True
        )
        kb.claim_task(conn, goal_task_id)
        run_id = kb._current_run_id(conn, goal_task_id)
    finally:
        conn.close()
    monkeypatch.setenv("HERMES_KANBAN_TASK", goal_task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))

    # Mock the judge to reject the completion. The gate only runs when a
    # judge is reachable, so force the availability probe True as well.
    def mock_judge_goal(goal, last_response, **_kwargs):
        # Match the real judge_goal contract:
        # (verdict, reason, parse_failed, wait_directive, transport_failed)
        return "continue", "missing verification evidence", False, None, False

    monkeypatch.setattr("tools.kanban_tools.judge_goal", mock_judge_goal)
    monkeypatch.setattr("tools.kanban_tools._goal_judge_available", lambda: True)

    # Attempt to complete should be rejected
    out = kt._handle_complete({"summary": "I did some stuff but not X"})
    d = json.loads(out)
    assert "error" in d
    assert "Goal completion rejected by judge" in d["error"]
    assert "missing verification evidence" in d["error"]

    # Verify the task is NOT completed in the DB
    conn2 = kbc.connect()
    try:
        task = kb.get_task(conn2, goal_task_id)
        assert task.status == "running"  # Should still be running, not done
    finally:
        conn2.close()


def test_block_happy_path(worker_env):
    from tools import kanban_tools as kt
    out = kt._handle_block({"reason": "need clarification"})
    d = json.loads(out)
    assert d["ok"] is True
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    conn = kbc.connect()
    try:
        assert kb.get_task(conn, worker_env).status == "blocked"
    finally:
        conn.close()


def test_goal_mode_review_quarantine_requires_operator_context(
    worker_env, monkeypatch, tmp_path,
):
    from hermes_cli import kanban_db as kb, kanban_db_connect as kbc
    from tests.hermes_cli.test_kanban_handoff import _repo, _commit
    from tools import kanban_tools  # noqa: F401 — ensure registration
    from tools.registry import registry

    handler = registry.get_entry("kanban_block").handler
    with kbc.connect_closing() as conn:
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET goal_mode = 1 WHERE id = ?", (worker_env,))
        before = list(conn.iterdump())
    ordinary = json.loads(handler({
        "task_id": worker_env, "kind": "capability", "reason": "operator recovery",
    }))
    assert "goal_mode tasks can only block" in ordinary["error"]
    worker = json.loads(handler({
        "task_id": worker_env, "quarantine_review": True, "expected_version": 1,
        "kind": "capability", "reason": "operator recovery",
    }))
    assert "cannot use expected_run_id" in worker["error"]
    with kbc.connect_closing() as conn:
        assert list(conn.iterdump()) == before

    monkeypatch.delenv("HERMES_KANBAN_TASK")
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID")
    repo, base, branch = _repo(tmp_path)
    with kbc.connect_closing() as conn:
        candidate = kb.create_task(
            conn, title="goal implementation", assignee="implementer", goal_mode=True,
            workspace_kind="worktree", workspace_path=str(repo), branch_name=branch,
            lifecycle_contract={
                "kind": "code", "review_mode": "same_card",
                "reviewer": "reviewer", "validation_required": False,
            },
        )
        run = kb.claim_task(conn, candidate)
        head = _commit(repo)
        assert kb.request_review(
            conn, candidate, expected_run_id=run.current_run_id,
            summary="implemented", metadata={"base_sha": base, "head_sha": head},
        )
        _commit(repo, "src/moved.py")
        version = kb.get_task(conn, candidate).version
        handoff = kb.latest_handoff(conn, candidate)
        runs = list(conn.execute("SELECT * FROM task_runs WHERE task_id = ?", (candidate,)))
    operator = json.loads(handler({
        "task_id": candidate, "quarantine_review": True, "expected_version": version,
        "kind": "capability", "reason": "immutable candidate moved",
    }))
    assert operator["ok"] is True
    with kbc.connect_closing() as conn:
        task = kb.get_task(conn, candidate)
        assert task.status == "blocked" and task.goal_mode
        assert task.version == version + 1 and task.current_run_id is None
        assert kb.latest_handoff(conn, candidate) == handoff
        assert list(conn.execute("SELECT * FROM task_runs WHERE task_id = ?", (candidate,))) == runs


def test_schedule_parks_current_worker_with_reason(worker_env):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from tools import kanban_tools  # noqa: F401 — ensure registration
    from tools.registry import registry

    reason = "SCHEDULED_UNTIL=2026-09-13T00:00:00Z waiting for reconnect"
    entry = registry.get_entry("kanban_schedule")
    assert entry is not None and entry.toolset == "kanban"
    out = json.loads(entry.handler({"reason": reason}))

    assert out == {
        "ok": True, "task_id": worker_env, "run_id": out["run_id"],
        "status": "scheduled", "reason": reason,
    }
    with kbc.connect() as conn:
        assert kb.get_task(conn, worker_env).status == "scheduled"
        run = kb.latest_run(conn, worker_env)
        assert (run.outcome, run.summary) == ("scheduled", reason)
        assert any(
            event.kind == "scheduled" and event.payload == {"reason": reason}
            for event in kb.list_events(conn, worker_env)
        )


def test_schedule_rejects_invalid_reason_and_unowned_contexts(monkeypatch, worker_env):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from tools import kanban_tools as kt

    with kbc.connect() as conn:
        other = kb.create_task(conn, title="sibling", assignee="peer")

    invalid = json.loads(kt._handle_schedule({"reason": {"until": "tomorrow"}}))
    foreign = json.loads(kt._handle_schedule({"task_id": other, "reason": "wait"}))
    monkeypatch.setattr(
        kt, "_delegation_ctx",
        lambda predicate, default: predicate == "is_delegated_child_process_context",
    )
    delegated = json.loads(kt._handle_schedule({"task_id": worker_env, "reason": "wait"}))

    assert "reason must be a string" in invalid["error"]
    assert "refusing to mutate" in foreign["error"]
    assert "delegate_task child agents are not Kanban run owners" in delegated["error"]
    with kbc.connect() as conn:
        assert kb.get_task(conn, worker_env).status == "running"
        assert kb.get_task(conn, other).status == "ready"


def _make_goal_mode_worker_env(monkeypatch, tmp_path):
    """Set up an isolated HERMES_HOME with one claimed goal_mode task,
    matching the pattern used by the kanban_complete judge gate tests."""
    from pathlib import Path as _Path
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "test-worker")
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    monkeypatch.setattr(_Path, "home", lambda: tmp_path)

    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    conn = kbc.connect()
    try:
        goal_task_id = kb.create_task(
            conn, title="goal-mode-block-test", assignee="test-worker",
            body="Must achieve X.", goal_mode=True,
        )
        kb.claim_task(conn, goal_task_id)
        run_id = kb._current_run_id(conn, goal_task_id)
    finally:
        conn.close()
    monkeypatch.setenv("HERMES_KANBAN_TASK", goal_task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    return goal_task_id


def test_block_goal_mode_rejects_missing_kind(monkeypatch, tmp_path):
    """A goal_mode worker calling kanban_block with no kind must not be able
    to use it as an unguarded escape from the goal loop (Issue #38696,
    sibling of the kanban_complete judge gate / Issue #38367)."""
    from tools import kanban_tools as kt
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    tid = _make_goal_mode_worker_env(monkeypatch, tmp_path)
    out = kt._handle_block({"reason": "giving up"})
    d = json.loads(out)
    assert "error" in d
    assert "goal_mode" in d["error"]

    conn = kbc.connect()
    try:
        assert kb.get_task(conn, tid).status == "running"
    finally:
        conn.close()


def test_block_goal_mode_rejects_disallowed_kind(monkeypatch, tmp_path):
    """`capability` / `transient` are valid kinds in general but must not
    let a goal_mode worker exit the loop without going through the judge."""
    from tools import kanban_tools as kt
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    tid = _make_goal_mode_worker_env(monkeypatch, tmp_path)
    for kind in ("capability", "transient"):
        out = kt._handle_block({"reason": "blocked", "kind": kind})
        d = json.loads(out)
        assert "error" in d, f"kind={kind} should be rejected for goal_mode"

    conn = kbc.connect()
    try:
        assert kb.get_task(conn, tid).status == "running"
    finally:
        conn.close()


def test_schedule_goal_mode_refused(monkeypatch, tmp_path):
    """``scheduled`` ends the goal loop like ``blocked``, so a goal_mode worker
    must not use kanban_schedule to exit without the completion judge."""
    from tools import kanban_tools as kt
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    tid = _make_goal_mode_worker_env(monkeypatch, tmp_path)
    d = json.loads(kt._handle_schedule({"reason": "waiting for CI"}))
    assert "goal_mode" in d.get("error", "")

    conn = kbc.connect()
    try:
        assert kb.get_task(conn, tid).status == "running"
    finally:
        conn.close()


def test_schedule_exposed_to_codex_runtime_workers():
    """Codex app-server workers only reach Hermes tools named in EXPOSED_TOOLS."""
    from agent.transports.hermes_tools_mcp_server import EXPOSED_TOOLS

    assert "kanban_schedule" in EXPOSED_TOOLS


def test_block_dependency_without_open_parent_is_rekinded(worker_env):
    """kind=dependency with no incomplete parent must not park in todo; the
    tool reports the landed kind and tells the worker why."""
    from tools import kanban_tools as kt

    d = json.loads(kt._handle_block({"reason": "upstream input is missing", "kind": "dependency"}))
    assert (d["ok"], d["status"], d["block_kind"]) == (True, "blocked", "needs_input")
    assert d["requested_kind"] == "dependency"
    assert "no parent is open" in d["note"]


def test_heartbeat_extends_claim_expires(worker_env):
    """The kanban_heartbeat tool MUST extend claim_expires, not just
    update last_heartbeat_at — otherwise long-running workers loop the
    heartbeat tool diligently and still get reclaimed by
    release_stale_claims at DEFAULT_CLAIM_TTL_SECONDS.

    Regression test for the bug where _handle_heartbeat called
    heartbeat_worker but never heartbeat_claim, so claim_expires sat
    static while last_heartbeat_at advanced.
    """
    import time as _time
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from tools import kanban_tools as kt

    # Rewind claim_expires into the past so any forward movement is
    # unambiguous (avoids time.sleep flakiness).
    conn = kbc.connect()
    try:
        conn.execute(
            "UPDATE tasks SET claim_expires = ? WHERE id = ?",
            (1, worker_env),
        )
        conn.commit()
        before = conn.execute(
            "SELECT claim_expires FROM tasks WHERE id = ?", (worker_env,)
        ).fetchone()["claim_expires"]
    finally:
        conn.close()
    assert before == 1

    out = kt._handle_heartbeat({"note": "still alive"})
    assert json.loads(out).get("ok") is True

    conn = kbc.connect()
    try:
        after = conn.execute(
            "SELECT claim_expires FROM tasks WHERE id = ?", (worker_env,)
        ).fetchone()["claim_expires"]
    finally:
        conn.close()

    now = int(_time.time())
    # claim_expires should be roughly now + DEFAULT_CLAIM_TTL_SECONDS.
    # We assert a generous floor (now + half the default TTL) to keep the
    # test stable against future TTL changes.
    assert after > before, (
        f"claim_expires did not advance ({before} -> {after}); workers "
        f"would be reclaimed at TTL despite heartbeating"
    )
    assert after >= now + (kb.DEFAULT_CLAIM_TTL_SECONDS // 2), (
        f"claim_expires={after} is suspiciously close to now={now}; "
        f"expected at least now + {kb.DEFAULT_CLAIM_TTL_SECONDS // 2}"
    )


def _expire_claim(conn, tid):
    conn.execute("UPDATE tasks SET claim_expires = 1 WHERE id = ?", (tid,))
    conn.commit()


def test_worker_the_dispatcher_never_recorded_keeps_its_claim_or_never_starts(monkeypatch, worker_env):
    """``worker_env`` is a claim whose dispatcher died between spawning the worker and recording its
    pid. The worker registers itself, so the expired claim is extended, not handed to a second worker;
    a worker that starts only after its run was reclaimed is told not to work the card."""
    import os as _os
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from tools import kanban_tools as kt

    assert kt.register_current_worker_from_env() is True
    with kbc.connect_closing() as conn:
        assert kb.get_task(conn, worker_env).worker_pid == _os.getpid()
        _expire_claim(conn, worker_env)
        assert kb.release_stale_claims(conn) == 0
        assert kb.get_task(conn, worker_env).status == "running"

        late = kb.create_task(conn, title="late orphan", assignee="test-worker")
        kb.claim_task(conn, late)
        stale_run = kb._current_run_id(conn, late)
        _expire_claim(conn, late)
        assert kb.release_stale_claims(conn) == 1
        kb.claim_task(conn, late)
    monkeypatch.setenv("HERMES_KANBAN_TASK", late)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(stale_run))
    assert kt.register_current_worker_from_env() is False
    with kbc.connect_closing() as conn:
        assert kb.get_task(conn, late).worker_pid is None


def test_reclaim_loses_to_a_worker_registering_mid_sweep(monkeypatch, worker_env):
    """The worker registers between the stale-claim SELECT and its UPDATE: the claim stays its own."""
    import os as _os
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from tools import kanban_tools as kt

    real_terminate = kb._terminate_reclaimed_worker

    def _register_then_terminate(*args, **kwargs):
        assert kt.register_current_worker_from_env() is True
        return real_terminate(*args, **kwargs)

    monkeypatch.setattr(kb, "_terminate_reclaimed_worker", _register_then_terminate)
    with kbc.connect_closing() as conn:
        _expire_claim(conn, worker_env)
        assert kb.release_stale_claims(conn) == 0
        task = kb.get_task(conn, worker_env)
    assert (task.status, task.worker_pid) == ("running", _os.getpid())


def test_update_attributes_goal_and_event_to_bound_profile(monkeypatch, worker_env, tmp_path):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from tools import kanban_tools as kt

    monkeypatch.delenv("HERMES_KANBAN_TASK")
    served = tmp_path / ".hermes/profiles/served"
    served.mkdir(parents=True)
    token = set_hermes_home_override(served)
    try:
        with kbc.connect_closing() as conn:
            task_id = kb.create_task(conn, title="original goal")
            version = kb.get_task(conn, task_id).version
        result = json.loads(kt._handle_update({
            "task_id": task_id, "expected_version": version,
            "title": "revised goal", "reason": "secondary profile turn",
        }))
        assert result["ok"], result
        with kbc.connect_closing() as conn:
            goal = kb.get_effective_goal(conn, task_id)
            event = json.loads(conn.execute(
                "SELECT payload FROM task_events WHERE task_id=? AND kind='goal_revised'",
                (task_id,),
            ).fetchone()[0])
        assert goal["title"] == "revised goal"
        assert goal["author"] == "served"
        assert event["author"] == event["actor"] == "served"
    finally:
        reset_hermes_home_override(token)


def test_comment_rejects_caller_supplied_author(worker_env):
    """Reject an undeclared author override before a worker can forge a comment."""
    from tools import kanban_tools as kt
    out = kt._handle_comment({
        "task_id": worker_env, "body": "hi", "author": "hermes-system",
    })
    assert "author" in json.loads(out)["error"]
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    conn = kbc.connect()
    try:
        assert kb.list_comments(conn, worker_env) == []
        out = kt._handle_comment({"task_id": worker_env, "body": "hi"})
        assert json.loads(out)["ok"]
        assert kb.list_comments(conn, worker_env)[0].author == "test-worker"
    finally:
        conn.close()


def test_create_happy_path(worker_env):
    from tools import kanban_tools as kt
    out = kt._handle_create({
        "title": "child task",
        "assignee": "peer",
        "parents": [worker_env],
    })
    d = json.loads(out)
    assert d["ok"] is True
    assert d["task_id"]
    assert d["status"] == "todo"  # parent isn't done yet
    assert d["gated"] is True and d["gated_by"] == worker_env
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    conn = kbc.connect()
    try:
        child = kb.get_task(conn, d["task_id"])
        assert child.title == "child task"
        assert child.assignee == "peer"
    finally:
        conn.close()


@pytest.mark.parametrize("explicit", [{"workspace_kind": "scratch"}, {"project": ""}])
@pytest.mark.parametrize("target_scoped", [False, True])
def test_create_explicit_scratch_ignores_ambient_board_project(
    worker_env, tmp_path, explicit, target_scoped,
):
    """#106342: an explicit scratch / empty project wins over the project the
    session's current board (and, when scoped, the target board itself) carries.
    Omitting both still inherits the target board's project."""
    from hermes_cli import kanban_db as kb
    from hermes_cli import projects_db as pdb
    from tools import kanban_tools as kt

    repo = tmp_path / "repo"
    repo.mkdir()
    with pdb.connect_closing() as pconn:
        project_id = pdb.create_project(pconn, name="Ambient", primary_path=str(repo))
    kb.write_board_metadata("default", project_id=project_id)
    kb.create_board("target", name="Target", project_id=project_id if target_scoped else "")

    def create(**extra):
        result = json.loads(kt._handle_create(
            {"board": "target", "title": "card", "assignee": "peer", **extra}))
        assert result["ok"] is True
        return result["workspace_kind"], result["project_id"]

    assert create(**explicit) == ("scratch", None)
    assert create() == (("worktree", project_id) if target_scoped else ("scratch", None))


def test_link_running_child_allows_owner_but_rejects_foreign(monkeypatch, worker_env):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from tools import kanban_tools as kt

    with kbc.connect() as conn:
        own_parent = kb.create_task(conn, title="own review")
        own_run_id = kb.get_task(conn, worker_env).current_run_id
        foreign_parent = kb.create_task(conn, title="foreign review")
        foreign_child = kb.create_task(conn, title="foreign worker")
        assert kb.claim_task(conn, foreign_child, claimer="other") is not None

    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(own_run_id))
    own = json.loads(kt._handle_link({"parent_id": own_parent, "child_id": worker_env}))
    foreign = json.loads(kt._handle_link(
        {"parent_id": foreign_parent, "child_id": foreign_child},
    ))

    assert own["ok"] is True
    assert "child is already running" in foreign["error"]
    with kbc.connect() as conn:
        assert kb.parent_ids(conn, worker_env) == [own_parent]
        assert kb.parent_ids(conn, foreign_child) == []


def test_unblock_happy_path(monkeypatch, worker_env):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="blocked", assignee="worker")
        kb.block_task(conn, tid, reason="waiting")
    finally:
        conn.close()

    from tools import kanban_tools as kt
    out = kt._handle_unblock({"task_id": tid})
    d = json.loads(out)
    assert d["ok"] is True
    assert d["status"] == "ready"

    conn = kbc.connect()
    try:
        assert kb.get_task(conn, tid).status == "ready"
    finally:
        conn.close()


def test_unblock_with_pending_parents_returns_todo(monkeypatch, tmp_path):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "orchestrator")
    from pathlib import Path as _Path
    monkeypatch.setattr(_Path, "home", lambda: tmp_path)

    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    conn = kbc.connect()
    try:
        parent = kb.create_task(conn, title="parent", assignee="worker")
        child = kb.create_task(conn, title="child", assignee="worker", parents=[parent])
        conn.execute("UPDATE tasks SET status='blocked' WHERE id=?", (child,))
        conn.commit()
    finally:
        conn.close()

    from tools import kanban_tools as kt
    out = kt._handle_unblock({"task_id": child})
    d = json.loads(out)
    assert d["ok"] is True
    assert d["status"] == "todo"

    conn = kbc.connect()
    try:
        assert kb.get_task(conn, child).status == "todo"
    finally:
        conn.close()


def test_worker_lifecycle_through_tools(worker_env):
    """Drive the full claim -> heartbeat -> comment -> complete lifecycle
    exclusively through the tools, then verify the DB state matches what
    the dispatcher/notifier expect."""
    from tools import kanban_tools as kt

    # 1. show — worker orientation
    show = json.loads(kt._handle_show({}))
    assert show["task"]["id"] == worker_env

    # 2. heartbeat during long op
    assert json.loads(kt._handle_heartbeat({"note": "warming up"}))["ok"]

    # 3. comment for a future peer
    assert json.loads(kt._handle_comment({
        "task_id": worker_env,
        "body": "note: using stdlib sqlite3 bindings",
    }))["ok"]

    # 4. spawn a child task for follow-up
    child_out = json.loads(kt._handle_create({
        "title": "write integration test",
        "assignee": "qa",
        "parents": [worker_env],
    }))
    assert child_out["ok"]

    # 5. complete with structured handoff
    comp = json.loads(kt._handle_complete({
        "summary": "implemented + spawned QA follow-up",
        "metadata": {"child_task": child_out["task_id"]},
    }))
    assert comp["ok"]

    # Verify final state
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    conn = kbc.connect()
    try:
        parent = kb.get_task(conn, worker_env)
        assert parent.status == "done"
        assert parent.current_run_id is None
        run = kb.latest_run(conn, worker_env)
        assert run.outcome == "completed"
        # Caller metadata must survive; the run also carries additive provenance.
        assert run.metadata["child_task"] == child_out["task_id"]
        # Child is todo (parent just finished, but recompute_ready may
        # have promoted it — complete_task runs recompute internally).
        child = kb.get_task(conn, child_out["task_id"])
        assert child.status == "ready", (
            f"child should be ready after parent done, got {child.status}"
        )
        # Comment is visible
        assert len(kb.list_comments(conn, worker_env)) == 1
        # Heartbeat event recorded
        hb = [e for e in kb.list_events(conn, worker_env) if e.kind == "heartbeat"]
        assert len(hb) == 1
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# System-prompt guidance injection
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Worker task-ownership enforcement (regression tests for #19534)
# ---------------------------------------------------------------------------
#
# A worker process has HERMES_KANBAN_TASK set to its own task id. The
# destructive tools (kanban_complete, kanban_block, kanban_heartbeat,
# kanban_unblock) must refuse to operate
# on any OTHER task id, even if the caller supplies an explicit `task_id`
# argument. Workers legitimately call kanban_show / kanban_list /
# kanban_comment / kanban_create / kanban_link on other tasks, so those
# are unrestricted.
#
# Orchestrator profiles (no HERMES_KANBAN_TASK in env) are intentionally
# exempt — their job is routing, and they sometimes close out child
# tasks on behalf of the child.


def test_worker_complete_rejects_foreign_task_id(worker_env):
    """A worker cannot complete a task that isn't its own (#19534)."""
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    conn = kbc.connect()
    try:
        other = kb.create_task(conn, title="sibling")
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (other,))
        conn.commit()
    finally:
        conn.close()

    from tools import kanban_tools as kt
    out = kt._handle_complete({"task_id": other, "summary": "HIJACK"})
    d = json.loads(out)
    assert d.get("ok") is not True
    assert "refusing to mutate" in d.get("error", "")

    # Sibling task must be untouched.
    conn = kbc.connect()
    try:
        assert kb.get_task(conn, other).status == "ready"
    finally:
        conn.close()


def test_worker_can_comment_on_foreign_task(worker_env):
    """Cross-task commenting must remain unrestricted (#19713 policy).

    The author-forgery hardening removed args['author'] but deliberately
    did NOT add an ownership gate to kanban_comment — comments are the
    documented handoff channel between tasks. This test pins that policy
    so a future change accidentally adding ``_enforce_worker_task_ownership``
    to ``_handle_comment`` would fail CI immediately.
    """
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    conn = kbc.connect()
    try:
        other = kb.create_task(conn, title="sibling")
    finally:
        conn.close()

    from tools import kanban_tools as kt
    out = kt._handle_comment({
        "task_id": other,
        "body": "handoff: see prior findings before starting",
    })
    d = json.loads(out)
    assert d.get("ok") is True, f"cross-task comment must succeed: {d}"

    # The comment lands on the foreign task, attributed to the worker's
    # HERMES_PROFILE — never to a caller-controlled string.
    conn = kbc.connect()
    try:
        comments = kb.list_comments(conn, other)
        assert len(comments) == 1
        assert comments[0].author == "test-worker"
        assert comments[0].body.startswith("handoff:")
    finally:
        conn.close()


def test_worker_unblock_rejects_foreign_task_id(worker_env):
    """A worker cannot unblock any task — kanban_unblock is orchestrator-only.

    The check fires before the per-task ownership check, so the error
    surface is the orchestrator-only refusal rather than the
    cross-task-ownership refusal. Either is fine — the property we're
    pinning is "worker cannot mutate foreign task via kanban_unblock".
    """
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    conn = kbc.connect()
    try:
        other = kb.create_task(conn, title="blocked sibling", assignee="peer")
        kb.block_task(conn, other, reason="waiting")
    finally:
        conn.close()

    from tools import kanban_tools as kt
    out = kt._handle_unblock({"task_id": other})
    d = json.loads(out)
    err = d.get("error", "")
    assert "orchestrator-only" in err or "refusing to mutate" in err, (
        f"expected worker-rejection error, got {err}"
    )

    conn = kbc.connect()
    try:
        assert kb.get_task(conn, other).status == "blocked"
    finally:
        conn.close()


def test_orchestrator_complete_any_task_allowed(monkeypatch, tmp_path):
    """Orchestrator profiles (no HERMES_KANBAN_TASK) can still complete
    any task via explicit task_id. The check only applies to workers."""
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    from pathlib import Path as _P
    monkeypatch.setattr(_P, "home", lambda: tmp_path)

    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="child to close out")
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
        conn.commit()
    finally:
        conn.close()

    from tools import kanban_tools as kt
    out = kt._handle_complete({"task_id": tid, "summary": "orchestrator close"})
    d = json.loads(out)
    assert d.get("ok") is True and d.get("task_id") == tid


# ---------------------------------------------------------------------------
# kanban_create auto-subscribe behaviour
#
# When a worker calls kanban_create from inside a session that has a
# persistent delivery channel, the originating session should be
# subscribed to the new task's completion/block events automatically.
# - Gateway sessions: HERMES_SESSION_PLATFORM + HERMES_SESSION_CHAT_ID set.
# - TUI sessions: HERMES_SESSION_KEY (or HERMES_SESSION_ID) set, with
#   the platform/chat_id ContextVars intentionally empty.
# - CLI / cron / test sessions: no delivery channel -> no subscription.
# - Config gate kanban.auto_subscribe_on_create: false -> no subscription
#   even when the session has a delivery channel.
# ---------------------------------------------------------------------------

def _list_subs_for_task(task_id):
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_notify as kbn
    conn = kbc.connect()
    try:
        return list(kbn.list_notify_subs(conn, task_id))
    finally:
        conn.close()


def _sub_index(subs):
    """Normalise a list of notify-subs (dicts or objects) into dicts
    keyed by platform+chat_id, so assertions work regardless of the
    return shape."""
    out = []
    for s in subs:
        if isinstance(s, dict):
            out.append(s)
        else:
            out.append({
                "platform": getattr(s, "platform", None),
                "chat_id": getattr(s, "chat_id", None),
                "thread_id": getattr(s, "thread_id", None),
                "user_id": getattr(s, "user_id", None),
                "delivery_metadata": getattr(s, "delivery_metadata", None),
                "notifier_profile": getattr(s, "notifier_profile", None),
            })
    return out


def test_create_subscribes_gateway_session(monkeypatch, worker_env):
    """A gateway session (platform + chat_id set) gets auto-subscribed
    to its own kanban_create result, and the response surfaces the
    ``subscribed`` flag so the orchestrator can react."""
    from tools import kanban_tools as kt
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "telegram")
    monkeypatch.setenv("HERMES_SESSION_CHAT_ID", "chat-42")
    monkeypatch.setenv("HERMES_SESSION_THREAD_ID", "thread-7")
    monkeypatch.setenv("HERMES_SESSION_USER_ID", "user-9")
    monkeypatch.setenv("HERMES_SESSION_USER_ID_ALT", "alt-user-9")
    monkeypatch.setenv("HERMES_SESSION_CHAT_TYPE", "forum")

    out = kt._handle_create({
        "title": "auto-sub gateway",
        "assignee": "peer",
    })
    d = json.loads(out)
    assert d["ok"] is True
    new_tid = d["task_id"]
    assert d["subscribed"] is True, d

    subs = _sub_index(_list_subs_for_task(new_tid))
    assert len(subs) == 1
    s = subs[0]
    assert s["platform"] == "telegram"
    assert s["chat_id"] == "chat-42"
    assert s["thread_id"] == "thread-7"
    assert s["user_id"] == "user-9"
    assert s["user_id_alt"] == "alt-user-9"
    assert s["chat_type"] == "forum"
    assert s["delivery_mode"] == "notify+wake"


def test_create_subscribes_tui_session_via_session_key(monkeypatch, worker_env):
    """TUI / desktop sessions don't have a platform/chat_id (single
    local channel), but the parent process exports HERMES_SESSION_KEY.
    We should still auto-subscribe, with platform='tui' and
    chat_id=<key>."""
    from tools import kanban_tools as kt
    monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)
    monkeypatch.delenv("HERMES_SESSION_CHAT_ID", raising=False)
    monkeypatch.delenv("HERMES_SESSION_THREAD_ID", raising=False)
    monkeypatch.delenv("HERMES_SESSION_USER_ID", raising=False)
    monkeypatch.setenv("HERMES_SESSION_KEY", "tui-session-abc")
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)

    out = kt._handle_create({
        "title": "auto-sub tui",
        "assignee": "peer",
    })
    d = json.loads(out)
    assert d["ok"] is True
    new_tid = d["task_id"]
    assert d["subscribed"] is True, d

    subs = _sub_index(_list_subs_for_task(new_tid))
    assert len(subs) == 1
    assert subs[0]["platform"] == "tui"
    assert subs[0]["chat_id"] == "tui-session-abc"
    assert subs[0]["chat_type"] == "dm"
    assert subs[0]["delivery_mode"] == "notify"


def test_create_does_not_subscribe_in_cli_session(monkeypatch, worker_env):
    """CLI / cron / test sessions have no persistent delivery channel.
    _maybe_auto_subscribe returns False and no row is written."""
    from tools import kanban_tools as kt
    monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)
    monkeypatch.delenv("HERMES_SESSION_CHAT_ID", raising=False)
    monkeypatch.delenv("HERMES_SESSION_KEY", raising=False)
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)

    out = kt._handle_create({
        "title": "no sub cli",
        "assignee": "peer",
    })
    d = json.loads(out)
    assert d["ok"] is True
    assert d["subscribed"] is False, d

    assert _list_subs_for_task(d["task_id"]) == []


def test_create_tui_subscription_binds_to_live_session_after_compaction_fork(monkeypatch, worker_env, tmp_path):
    """#110068: the inherited HERMES_SESSION_KEY can name a session already superseded
    by a compaction fork. Auto-subscribe must bind to the live continuation tip resolved
    from the session store, not the stale key the process was launched with."""
    import hermes_state
    from hermes_state import SessionDB
    from tools import kanban_tools as kt

    stale, live = "20260912_231110_stale", "20260913_000649_live"
    db = SessionDB(tmp_path / "state.db")
    db.create_session(stale, source="tui")
    db.append_message(stale, role="user", content="pre-fork turn")
    db.end_session(stale, "compression")
    db.create_session(live, source="tui", parent_session_id=stale)
    db.append_message(live, role="assistant", content="post-fork turn")

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")
    monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)
    monkeypatch.delenv("HERMES_SESSION_CHAT_ID", raising=False)
    monkeypatch.setenv("HERMES_SESSION_KEY", stale)
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    monkeypatch.setenv("HERMES_PROFILE", "default")

    out = kt._handle_create({
        "title": "auto-sub tui post-fork",
        "assignee": "peer",
    })
    d = json.loads(out)
    assert d["ok"] is True
    assert d["subscribed"] is True, d

    subs = _sub_index(_list_subs_for_task(d["task_id"]))
    assert len(subs) == 1
    assert subs[0]["platform"] == "tui"
    assert subs[0]["chat_id"] == live


def test_live_tui_session_key_uses_profile_store(monkeypatch, tmp_path):
    """A named-profile session's lineage lives in profiles/<x>/state.db, not the default
    store: resolution must open THAT store."""
    from hermes_cli import profiles as profiles_mod
    from hermes_state import SessionDB
    from tools import kanban_tools as kt

    stale, live = "stale-teamx", "live-teamx"
    store = tmp_path / "profiles" / "teamx"
    store.mkdir(parents=True)
    db = SessionDB(store / "state.db")
    db.create_session(stale, source="tui")
    db.append_message(stale, role="user", content="pre-fork turn")
    db.end_session(stale, "compression")
    db.create_session(live, source="tui", parent_session_id=stale)
    db.append_message(live, role="user", content="post-fork turn")

    monkeypatch.setattr(profiles_mod, "profile_exists", lambda name: name == "teamx")
    monkeypatch.setattr(profiles_mod, "get_profile_dir", lambda name: str(store))
    assert kt._live_tui_session_key(stale, "teamx") == live


def test_live_tui_session_key_fails_open_to_original_key(monkeypatch):
    """Store unavailability must degrade to current behaviour (subscribe to the inherited
    key), never fail the kanban_create call."""
    import hermes_state_registry as registry
    from tools import kanban_tools as kt

    def _boom(db_path=None):
        raise RuntimeError("store unavailable")

    monkeypatch.setattr(registry, "acquire", _boom)
    assert kt._live_tui_session_key("stale-key", "default") == "stale-key"
    assert kt._live_tui_session_key("stale-key", None) == "stale-key"
    # An unresolvable profile also falls through to the default store path.
    import hermes_cli.profiles as profiles_mod
    monkeypatch.setattr(profiles_mod, "profile_exists", lambda name: False)
    assert kt._live_tui_session_key("stale-key", "ghost-profile") == "stale-key"


def test_create_respects_auto_subscribe_on_create_false(monkeypatch, worker_env, tmp_path):
    """The config gate kanban.auto_subscribe_on_create=false must
    suppress auto-subscription even when the session has a delivery
    channel. This is the knob that addresses the upstream design
    concern from PR #19718 (reverted in #19721) — users who want
    explicit kanban_notify-subscribe calls per task get that."""
    # worker_env already created <tmp>/.hermes; use a fresh sibling
    # home to avoid mkdir() colliding with the worker's directory.
    home = tmp_path / "gate-home" / ".hermes"
    home.mkdir(parents=True)
    (home / "config.yaml").write_text(
        "kanban:\n  auto_subscribe_on_create: false\n"
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "discord")
    monkeypatch.setenv("HERMES_SESSION_CHAT_ID", "channel-1")

    from tools import kanban_tools as kt
    out = kt._handle_create({
        "title": "no sub gated",
        "assignee": "peer",
    })
    d = json.loads(out)
    assert d["ok"] is True
    assert d["subscribed"] is False, d

    assert _list_subs_for_task(d["task_id"]) == []


def test_maybe_auto_subscribe_swallows_add_notify_sub_failure(monkeypatch, worker_env):
    """If add_notify_sub itself raises (e.g. DB locked, schema drift),
    _maybe_auto_subscribe must NOT bubble that up and fail the parent
    kanban_create. The function returns False and the parent create
    still succeeds with subscribed=False."""
    from tools import kanban_tools as kt
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "telegram")
    monkeypatch.setenv("HERMES_SESSION_CHAT_ID", "chat-42")

    from hermes_cli import kanban_db_notify as kbn

    def _boom(*a, **kw):
        raise RuntimeError("simulated DB failure")

    monkeypatch.setattr(kbn, "add_notify_sub", _boom)

    out = kt._handle_create({
        "title": "auto-sub tolerates add_notify_sub failure",
        "assignee": "peer",
    })
    d = json.loads(out)
    assert d["ok"] is True, d
    assert d["subscribed"] is False, d


# ---------------------------------------------------------------------------
# Attachments — kanban_attach / kanban_attach_url / kanban_attachments
# ---------------------------------------------------------------------------


def test_attach_url_rejects_non_http_scheme(worker_env):
    from tools import kanban_tools as kt

    out = kt._handle_attach_url({"url": "file:///etc/passwd"})
    d = json.loads(out)
    assert "error" in d
    assert "scheme" in d["error"]


# ---------------------------------------------------------------------------
# kanban_attach_url — SSRF guard (tools/url_safety.is_safe_url per hop)
# ---------------------------------------------------------------------------


@pytest.fixture
def default_url_guard(monkeypatch):
    """Force the SSRF guard to its secure default for this test.

    Clears HERMES_ALLOW_PRIVATE_URLS and resets url_safety's process-lifetime
    cache on both sides so a prior test's opt-in can't leak in.
    """
    from tools import url_safety

    monkeypatch.delenv("HERMES_ALLOW_PRIVATE_URLS", raising=False)
    url_safety._reset_allow_private_cache()
    yield
    url_safety._reset_allow_private_cache()


def _assert_attach_url_blocked(worker_env, url):
    """Call kanban_attach_url with ``url`` and assert the SSRF guard fired
    (clean tool error, no attachment row, no network fetch needed)."""
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from tools import kanban_tools as kt

    out = kt._handle_attach_url({"url": url})
    d = json.loads(out)
    assert "error" in d, out
    assert "SSRF" in d["error"] or "blocked" in d["error"].lower(), out
    conn = kbc.connect()
    try:
        assert kb.list_attachments(conn, worker_env) == []
    finally:
        conn.close()


def test_attach_url_blocks_loopback(worker_env, default_url_guard):
    """http://127.0.0.1/ is rejected before any connection is made."""
    _assert_attach_url_blocked(worker_env, "http://127.0.0.1/")


def _fake_public_dns(monkeypatch, mapping):
    """Patch url_safety's getaddrinfo so hostnames in ``mapping`` resolve to
    the given (public) IPs and literal IPs resolve to themselves — no real
    DNS or network traffic."""
    import ipaddress
    import socket as _socket

    real_af, real_sock = _socket.AF_INET, _socket.SOCK_STREAM

    def fake_getaddrinfo(host, *args, **kwargs):
        ip = mapping.get(host)
        if ip is None:
            # Literal IPs pass through; unknown hostnames fail like NXDOMAIN.
            try:
                ipaddress.ip_address(host)
            except ValueError:
                raise _socket.gaierror(f"fake DNS: unknown host {host!r}")
            ip = host
        return [(real_af, real_sock, 6, "", (ip, 0))]

    from tools import url_safety
    monkeypatch.setattr(url_safety.socket, "getaddrinfo", fake_getaddrinfo)


class _FakeStreamResponse:
    def __init__(self, *, status_code=200, headers=None, body=b""):
        self.status_code = status_code
        self.headers = headers or {}
        self._body = body

    @property
    def is_redirect(self):
        return 300 <= self.status_code < 400 and "location" in {
            k.lower() for k in self.headers
        }

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_bytes(self, chunk_size):
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i:i + chunk_size]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_attach_url_happy_path_public_host(worker_env, default_url_guard, monkeypatch):
    """A public URL passes the guard and the bytes are stored (mocked fetch)."""
    from pathlib import Path

    import httpx

    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from tools import kanban_tools as kt

    _fake_public_dns(monkeypatch, {"files.example.com": "93.184.216.34"})

    payload = b"public fetch body"

    def fake_stream(method, url, **kwargs):
        assert url == "http://files.example.com/docs/spec.pdf"
        return _FakeStreamResponse(
            status_code=200,
            headers={"content-type": "application/pdf; charset=binary"},
            body=payload,
        )

    monkeypatch.setattr(httpx, "stream", fake_stream)

    out = kt._handle_attach_url({"url": "http://files.example.com/docs/spec.pdf"})
    d = json.loads(out)
    assert d.get("ok") is True, out
    assert d["size"] == len(payload)

    conn = kbc.connect()
    try:
        atts = kb.list_attachments(conn, worker_env)
        assert [a.filename for a in atts] == ["spec.pdf"]
        assert atts[0].content_type == "application/pdf"
        assert Path(atts[0].stored_path).read_bytes() == payload
    finally:
        conn.close()


@pytest.fixture
def multi_board_env(tmp_path, monkeypatch):
    from pathlib import Path
    from hermes_cli import kanban_db as kb

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "test-orchestrator")
    for name in ("HERMES_KANBAN_TASK", "HERMES_KANBAN_RUN_ID", "HERMES_KANBAN_BOARD",
                 "HERMES_KANBAN_DB", "HERMES_KANBAN_HOME", "HERMES_KANBAN_WORKSPACES_ROOT"):
        monkeypatch.delenv(name, raising=False)
    kb.init_db(board="alt")
    return home


def test_orchestrator_repairs_generated_subtree_through_tools(multi_board_env):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from tools import kanban_tools as kt

    with kbc.connect_closing(board="alt") as conn:
        valid = kb.create_task(conn, title="valid implementation")
        invalid = kb.create_task(conn, title="invalid generated prerequisite")
        root = kb.create_task(
            conn, title="original implementation", parents=[valid, invalid]
        )
        review = kb.create_task(
            conn, title="review", assignee="reviewer", parents=[root]
        )
        test = kb.create_task(conn, title="test", assignee="tester", parents=[review])
        generated_child = kb.create_task(
            conn, title="invalid generated child", parents=[invalid]
        )
        conn.execute("UPDATE tasks SET status = 'done' WHERE id = ?", (valid,))
        conn.commit()

    first = json.loads(
        kt._handle_unlink({
            "board": "alt",
            "parent_id": invalid,
            "child_id": root,
            "expected_parent_version": 1,
            "expected_child_version": 1,
            "reason": "remove legacy auto-decomposition edge",
        })
    )
    assert first["ok"] and first["removed"] and first["child_status"] == "ready"

    second = json.loads(
        kt._handle_unlink({
            "board": "alt",
            "parent_id": invalid,
            "child_id": generated_child,
            "expected_parent_version": 2,
            "expected_child_version": 1,
            "reason": "detach invalid generated subtree",
        })
    )
    assert second["ok"] and second["removed"]

    for task_id, version in ((invalid, 3), (generated_child, 2)):
        archived = json.loads(
            kt._handle_archive({
                "board": "alt",
                "task_id": task_id,
                "expected_version": version,
                "reason": "retire detached generated card without fake completion",
            })
        )
        assert (
            archived["ok"] and archived["archived"] and archived["status"] == "archived"
        )

    with kbc.connect_closing(board="alt") as conn:
        assert kb.parent_ids(conn, root) == [valid]
        assert kb.parent_ids(conn, review) == [root]
        assert kb.parent_ids(conn, test) == [review]
        event = [
            event for event in kb.list_events(conn, root) if event.kind == "unlinked"
        ][-1]
        assert event.payload["actor"] == "test-orchestrator"
        assert event.payload["old_graph"]["child_parents"] == sorted([invalid, valid])
        assert event.payload["new_graph"]["child_parents"] == [valid]

    retry = json.loads(
        kt._handle_unlink({
            "board": "alt",
            "parent_id": invalid,
            "child_id": root,
            "expected_parent_version": 1,
            "expected_child_version": 1,
            "reason": "idempotent retry",
        })
    )
    assert retry["ok"] and retry["removed"] is False

def test_graph_repair_rejects_stale_claimed_and_linked_tasks(multi_board_env):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from tools import kanban_tools as kt

    with kbc.connect_closing(board="alt") as conn:
        parent = kb.create_task(conn, title="parent")
        child = kb.create_task(conn, title="child", parents=[parent])

    stale = json.loads(
        kt._handle_unlink({
            "board": "alt",
            "parent_id": parent,
            "child_id": child,
            "expected_parent_version": 99,
            "expected_child_version": 1,
            "reason": "stale repair",
        })
    )
    assert "update conflict" in stale["error"]

    linked = json.loads(
        kt._handle_archive({
            "board": "alt",
            "task_id": parent,
            "expected_version": 1,
            "reason": "unsafe linked archive",
        })
    )
    assert "detach all dependency edges first" in linked["error"]

    with kbc.connect_closing(board="alt") as conn:
        kb.claim_task(conn, parent)
    claimed = json.loads(
        kt._handle_unlink({
            "board": "alt",
            "parent_id": parent,
            "child_id": child,
            "expected_parent_version": 1,
            "expected_child_version": 1,
            "reason": "unsafe concurrent repair",
        })
    )
    assert "currently claimed" in claimed["error"]


class TestDefaultTaskId:
    def test_numeric_task_id_is_coerced(self, monkeypatch):
        from tools import kanban_tools as kt
        monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
        assert kt._default_task_id(12345) == "12345"
        assert kt._default_task_id(0) == "0"

    def test_whitespace_task_id_is_stripped(self, monkeypatch):
        from tools import kanban_tools as kt
        monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
        assert kt._default_task_id("  task-99  ") == "task-99"

    def test_blank_task_id_falls_back(self, monkeypatch, worker_env):
        from tools import kanban_tools as kt
        monkeypatch.setattr(kt, "_is_dispatcher_owned_worker", lambda: True)
        assert kt._default_task_id("") == worker_env
        assert kt._default_task_id("   ") == worker_env
        assert kt._default_task_id(None) == worker_env


# ---------------------------------------------------------------------------
# supersede-candidate — orchestrator-only unreviewed moved-head recovery
# ---------------------------------------------------------------------------

def _supersede_candidate_tool_env(monkeypatch, tmp_path):
    """Isolated home + named board holding one quarantined moved-head candidate."""
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from tests.hermes_cli.test_kanban_cli import _supersede_candidate_case

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "cto")
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)
    kb._INITIALIZED_PATHS.clear()
    kb.create_board("recovery")
    with kbc.connect_closing(board="recovery") as conn:
        case = _supersede_candidate_case(conn, tmp_path)
        quarantined = list(conn.iterdump())
    from tools.registry import discover_builtin_tools
    discover_builtin_tools()
    return case, quarantined


def test_supersede_candidate_tool_recovers_named_board(monkeypatch, tmp_path):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from tools.registry import registry

    case, _quarantined = _supersede_candidate_tool_env(monkeypatch, tmp_path)
    out = json.loads(registry.dispatch("kanban_supersede_candidate", {
        "board": "recovery",
        "implementation_id": case["implementation"],
        "expected_versions": case["versions"],
        "expected_candidate_run_id": case["candidate_run_id"],
        "expected_head_sha": case["head"],
        "replacement_head_sha": case["moved_head"],
        "reason": "supersede the unreviewed moved candidate",
    }))
    assert out["ok"] is True, out
    assert out["implementation_id"] == case["implementation"]
    assert out["reviewer_id"] == case["reviewer"]
    assert out["tester_id"] == case["tester"]
    assert out["status"] == "ready"
    assert (out["superseded_head_sha"], out["replacement_head_sha"]) == (
        case["head"], case["moved_head"],
    )

    with kbc.connect_closing(board="recovery") as conn:
        assert kb.get_task(conn, case["implementation"]).status == "ready"
        assert kb.get_task(conn, case["reviewer"]).status == "todo"
        assert kb.claim_review_task(conn, case["reviewer"]) is None
        assert kb.claim_task(conn, case["tester"]) is None


def test_supersede_candidate_tool_refuses_missing_board_and_worker_context(
    monkeypatch, tmp_path,
):
    from hermes_cli import kanban_db_connect as kbc
    from tools.registry import registry

    case, quarantined = _supersede_candidate_tool_env(monkeypatch, tmp_path)
    payload = {
        "implementation_id": case["implementation"],
        "expected_versions": case["versions"],
        "expected_candidate_run_id": case["candidate_run_id"],
        "expected_head_sha": case["head"],
        "replacement_head_sha": case["moved_head"],
        "reason": "supersede the unreviewed moved candidate",
    }
    missing_board = json.loads(registry.dispatch("kanban_supersede_candidate", dict(payload)))
    assert "board is required" in missing_board["error"]

    monkeypatch.setenv("HERMES_KANBAN_TASK", case["implementation"])
    worker = json.loads(registry.dispatch("kanban_supersede_candidate", {**payload, "board": "recovery"}))
    assert "orchestrator-only" in worker["error"]

    with kbc.connect_closing(board="recovery") as conn:
        assert list(conn.iterdump()) == quarantined



# ---------------------------------------------------------------------------
# Native per-task reasoning effort (#58)
# ---------------------------------------------------------------------------


def test_effort_create_contract_through_registry(worker_env):
    """The model-facing create surface: pinned, refused, and never silently inherited."""
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from tools.registry import discover_builtin_tools, registry

    discover_builtin_tools()
    assert "reasoning_effort" in registry.get_schema(
        "kanban_create")["parameters"]["properties"]

    created = json.loads(registry.dispatch("kanban_create", {
        "title": "pinned child", "assignee": "peer", "reasoning_effort": "XHIGH"}))
    assert created["ok"], created
    with kbc.connect_closing() as conn:
        assert kb.get_task(conn, created["task_id"]).reasoning_effort == "xhigh"

    # A falsey non-string and an unknown level are both refused before any card exists,
    # so neither can quietly fall back to the profile's effort.
    for bad in (False, "turbo"):
        refused = json.loads(registry.dispatch("kanban_create", {
            "title": "bad effort", "assignee": "peer", "reasoning_effort": bad}))
        assert "reasoning_effort" in refused["error"], refused
    with kbc.connect_closing() as conn:
        assert [t.id for t in kb.list_tasks(conn) if t.title == "bad effort"] == []


def test_effort_update_contract_through_registry(worker_env, monkeypatch):
    """The model-facing update surface: levels, clear precedence and refusals."""
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from tools.registry import discover_builtin_tools, invalidate_check_fn_cache, registry

    discover_builtin_tools()
    monkeypatch.delenv("HERMES_KANBAN_TASK")
    invalidate_check_fn_cache()
    assert {"reasoning_effort", "clear_reasoning_effort"} <= set(
        registry.get_schema("kanban_update")["parameters"]["properties"])

    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="effort card", assignee="peer")

    def update(**args):
        with kbc.connect_closing() as conn:
            version = kb.get_task(conn, tid).version
        result = json.loads(registry.dispatch("kanban_update", {
            "task_id": tid, "expected_version": version, "reason": "route", **args}))
        assert result["ok"], result
        with kbc.connect_closing() as conn:
            return kb.get_task(conn, tid).reasoning_effort

    assert update(reasoning_effort="xhigh") == "xhigh"
    assert update(title="renamed") == "xhigh"          # omission preserves the pin
    assert update(reasoning_effort="none") == "none"   # a real level, not a clear
    assert update(clear_reasoning_effort=True) is None
    assert update(reasoning_effort="low", clear_reasoning_effort=True) is None  # clear wins

    # An unknown spelling never reaches a mutation, and a falsey non-string is refused.
    with kbc.connect_closing() as conn:
        version = kb.get_task(conn, tid).version
    for payload in ({"reasoning": "xhigh"}, {"reasoning_effort": 0}):
        out = registry.dispatch("kanban_update", {
            "task_id": tid, "expected_version": version, "reason": "bad", **payload})
        assert "error" in str(out), out
    with kbc.connect_closing() as conn:
        task = kb.get_task(conn, tid)
    assert (task.reasoning_effort, task.version) == (None, version)


def test_orchestrator_fenced_unblock_requires_the_current_version(monkeypatch, tmp_path):
    """The version fence is a recovery CAS: stale or active-claim calls change nothing."""
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)
    from pathlib import Path

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="blocked recovery", assignee="worker")
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status = 'running' WHERE id = ?", (tid,))
        assert kb.block_task(conn, tid, reason="needs an operator")
        # Two versions, so the stale fence below is still a valid version number.
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET version = version + 1 WHERE id = ?", (tid,))
        version = kb.get_task(conn, tid).version
    finally:
        conn.close()

    from tools import kanban_tools as kt

    stale = json.loads(kt._handle_unblock({"task_id": tid, "expected_version": version - 1}))
    assert "stale version" in stale.get("error", ""), stale
    again = kbc.connect()
    try:
        parked = kb.get_task(again, tid)
        assert (parked.status, parked.version) == ("blocked", version)
        assert not [c for c in kb.list_comments(again, tid) if c.body.startswith("UNBLOCK:")]
    finally:
        again.close()

    fenced = json.loads(kt._handle_unblock({"task_id": tid, "expected_version": version}))
    assert fenced.get("ok") is True and fenced.get("status") == "ready", fenced
    final = kbc.connect()
    try:
        assert kb.get_task(final, tid).version == version + 1
        # The ordinary unfenced call is untouched by the fence (fresh card, so the
        # block-loop breaker cannot route it out of the blocked bucket).
        other = kb.create_task(final, title="plain unblock", assignee="worker")
        with kb.write_txn(final):
            final.execute("UPDATE tasks SET status = 'running' WHERE id = ?", (other,))
        assert kb.block_task(final, other, reason="needs an operator")
        plain_version = kb.get_task(final, other).version
    finally:
        final.close()
    plain = json.loads(kt._handle_unblock({"task_id": other}))
    assert plain.get("ok") is True, plain
    check = kbc.connect()
    try:
        assert kb.get_task(check, other).version == plain_version
    finally:
        check.close()


def test_orchestrator_fenced_unblock_records_the_reason_it_was_given(monkeypatch, tmp_path):
    """The tool path must be able to say why recovery happened, with a real actor."""
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)
    from pathlib import Path

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="fenced recovery", assignee="worker")
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status = 'running' WHERE id = ?", (tid,))
        assert kb.block_task(conn, tid, reason="needs an operator")
        # Two versions, so the stale fence below is still a valid version number.
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET version = version + 1 WHERE id = ?", (tid,))
        version = kb.get_task(conn, tid).version
    finally:
        conn.close()

    from tools import kanban_tools as kt

    # A reason without the fence is refused: the ordinary unblock writes no comment.
    refused = json.loads(kt._handle_unblock({"task_id": tid, "reason": "no fence"}))
    assert "requires expected_version" in refused.get("error", ""), refused

    stale = json.loads(kt._handle_unblock(
        {"task_id": tid, "expected_version": version - 1, "reason": "too early"}))
    assert "stale version" in stale.get("error", ""), stale
    check = kbc.connect()
    try:
        assert not [c for c in kb.list_comments(check, tid) if c.body.startswith("UNBLOCK:")]
    finally:
        check.close()

    fenced = json.loads(kt._handle_unblock(
        {"task_id": tid, "expected_version": version, "reason": "policy installed"}))
    assert fenced.get("ok") is True, fenced
    final = kbc.connect()
    try:
        comment = [c for c in kb.list_comments(final, tid) if c.body.startswith("UNBLOCK:")][-1]
        assert comment.body == "UNBLOCK: policy installed"
        assert comment.author and comment.author.strip()
    finally:
        final.close()


def test_bounded_pages_always_say_whether_the_value_is_complete(evidence_board):
    """An unfinished read must be visible in the result, never merely implied.

    Pagination exists because the whole value cannot be delivered at all; every page
    therefore states `complete`, so a caller cannot mistake a first page for the whole
    section, the whole body, or the whole history.
    """
    kb, conn, tid = evidence_board
    body = "restriction-free filler " * 1000 + "LATE-RESTRICTION: never authorize writes"
    conn.execute("UPDATE tasks SET body = ? WHERE id = ?", (body, tid))
    for seq in range(3):
        kb._append_event(conn, tid, "note", {"seq": seq})
    conn.commit()

    token = _evidence({"task_id": tid, "section": "lifecycle"})["snapshot"]
    args = {"task_id": tid, "snapshot": token, "chars": 1024, "limit": 1}

    # A section that fits in one page is complete.
    whole = {"task_id": tid, "snapshot": token, "limit": 200, "chars": 8192}
    assert _evidence({**whole, "section": "lifecycle"})["complete"] is True
    # A chunked body is explicitly incomplete until its final chunk.
    page = _evidence({**args, "section": "task", "pointer": "/body"})
    assert page["complete"] is False
    assert page["chunk"]["next_offset"] is not None
    pages = 1
    while page["chunk"]["next_offset"] is not None:
        page = _evidence(page["continuation"])
        pages += 1
    assert pages > 1
    assert page["complete"] is True
    assert page["chunk"]["next_offset"] is None
    # History is explicitly incomplete while rows remain, and complete at the end.
    history = _evidence({**args, "section": "events", "limit": 1})
    assert (history["complete"], history["has_more"]) == (False, True)
    while history["has_more"]:
        history = _evidence(history["continuation"])
    assert (history["complete"], history["next_after"]) == (True, None)

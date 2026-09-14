"""Tests for kanban lifecycle plugin hooks.

Verifies that claim/complete/block transitions fire the
kanban_task_claimed / kanban_task_completed / kanban_task_blocked plugin
hooks AFTER the board DB change is committed, with the documented kwargs,
and that a misbehaving hook callback never breaks the transition.
"""

from __future__ import annotations

import json
from pathlib import Path
import threading
from unittest.mock import patch

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_lifecycle_evidence as evidence
from hermes_cli.plugins import VALID_HOOKS, get_plugin_manager


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def captured_hooks(monkeypatch):
    """Register capturing callbacks for the three kanban lifecycle hooks.

    Patches the plugin manager's _hooks dict directly (the same registry
    invoke_hook reads) and restores it afterward.
    """
    mgr = get_plugin_manager()
    events: list[tuple[str, dict]] = []
    saved = {k: list(v) for k, v in mgr._hooks.items()}
    for hook in ("kanban_task_claimed", "kanban_task_completed", "kanban_task_blocked"):
        mgr._hooks.setdefault(hook, []).append(
            lambda _h=hook, **kw: events.append((_h, kw))
        )
    try:
        yield events
    finally:
        mgr._hooks = saved




def test_claim_fires_hook(kanban_home, captured_hooks):
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="t", assignee="worker")
        claimed = kb.claim_task(conn, tid)
        assert claimed is not None
    finally:
        conn.close()
    fired = [e for e in captured_hooks if e[0] == "kanban_task_claimed"]
    assert len(fired) == 1
    kw = fired[0][1]
    assert kw["task_id"] == tid
    assert kw["assignee"] == "worker"
    assert "profile_name" in kw
    assert kw["run_id"] is not None




def test_review_claim_fires_hook(kanban_home, captured_hooks):
    conn = kbc.connect()
    try:
        task_id = kb.create_task(conn, title="review", assignee="builder")
        implementation = kb.claim_task(conn, task_id, claimer="builder")
        assert implementation is not None
        assert kb.request_review(
            conn, task_id, reviewer="reviewer",
            expected_run_id=implementation.current_run_id,
        )
        claimed = kb.claim_review_task(conn, task_id, claimer="reviewer")
        assert claimed is not None
    finally:
        conn.close()
    fired = [
        event for event in captured_hooks
        if event[0] == "kanban_task_claimed"
        and event[1]["run_id"] == claimed.current_run_id
    ]
    assert len(fired) == 1
    assert fired[0][1]["task_id"] == task_id
    assert fired[0][1]["assignee"] == "reviewer"

def test_general_completion_fires_hook(kanban_home, captured_hooks):
    conn = kbc.connect()
    try:
        task_id = kb.create_task(conn, title="general", assignee="worker")
        claimed = kb.claim_task(conn, task_id)
        assert claimed is not None
        assert kb.complete_task(conn, task_id, summary="done") is True
    finally:
        conn.close()

    completed = [
        event
        for event in captured_hooks
        if event[0] == "kanban_task_completed"
    ]
    assert len(completed) == 1
    assert completed[0][1]["task_id"] == task_id
    assert completed[0][1]["run_id"] == claimed.current_run_id

def test_acceptance_event_projection_holds_write_lock(kanban_home):
    conn = kbc.connect()
    writer_start = threading.Event()
    writer_acquired = threading.Event()
    writer_done = threading.Event()
    writer_errors = []
    try:
        task_id = kb.create_task(conn, title="race", assignee="worker")

        def mutate_task():
            competing = kbc.connect()
            try:
                writer_start.wait(timeout=5)
                with kbc.write_txn(competing):
                    writer_acquired.set()
                    competing.execute(
                        "UPDATE tasks SET version = version + 1 WHERE id = ?",
                        (task_id,),
                    )
            except BaseException as error:
                writer_errors.append(error)
            finally:
                competing.close()
                writer_done.set()

        writer = threading.Thread(target=mutate_task)
        writer.start()

        def projected_state(connection, observed_id):
            state = {
                "acceptance": "accepted",
                "review_verdict": "APPROVE",
                "validation_verdict": None,
                "execution_outcome": None,
            }
            if observed_id == task_id and not connection.in_transaction:
                writer_start.set()
                assert writer_done.wait(timeout=5)
            return state

        with patch.object(evidence, "get_lifecycle_state", side_effect=projected_state):
            accepted = evidence._emit_acceptance_changes(
                conn,
                {task_id: "pending"},
                source_task_id=task_id,
            )

        assert accepted == [task_id]
        assert not writer_acquired.is_set()
        event = conn.execute(
            "SELECT payload FROM task_events "
            "WHERE task_id = ? AND kind = 'acceptance_changed' "
            "ORDER BY id DESC LIMIT 1",
            (task_id,),
        ).fetchone()
        assert event is not None
        assert json.loads(event["payload"])["new"] == "accepted"

        writer_start.set()
        assert writer_done.wait(timeout=5)
        writer.join(timeout=5)
        assert not writer.is_alive()
        assert writer_errors == []
    finally:
        conn.close()

def test_acceptance_event_deduplicates_stale_capture(kanban_home):
    conn = kbc.connect()
    try:
        task_id = kb.create_task(conn, title="duplicate", assignee="worker")
        with kbc.write_txn(conn):
            kb._append_event(
                conn,
                task_id,
                "acceptance_changed",
                {"old": "pending", "new": "accepted", "phase": "review"},
            )

        with patch.object(
            evidence,
            "get_lifecycle_state",
            return_value={"acceptance": "accepted"},
        ):
            assert evidence._emit_acceptance_changes(
                conn,
                {task_id: "pending"},
                source_task_id=task_id,
            ) == []

        count = conn.execute(
            "SELECT COUNT(*) FROM task_events "
            "WHERE task_id = ? AND kind = 'acceptance_changed'",
            (task_id,),
        ).fetchone()[0]
        assert count == 1
    finally:
        conn.close()




def test_misbehaving_hook_does_not_break_transition(kanban_home, monkeypatch):
    """A hook callback that raises must not break the board transition."""
    mgr = get_plugin_manager()
    saved = {k: list(v) for k, v in mgr._hooks.items()}

    def _boom(**kw):
        raise RuntimeError("plugin exploded")

    mgr._hooks.setdefault("kanban_task_completed", []).append(_boom)
    try:
        conn = kbc.connect()
        try:
            tid = kb.create_task(conn, title="t", assignee="worker")
            kb.claim_task(conn, tid)
            # Despite the raising hook, completion succeeds and persists.
            assert kb.complete_task(conn, tid, summary="ok") is True
            assert kb.get_task(conn, tid).status == "done"
        finally:
            conn.close()
    finally:
        mgr._hooks = saved

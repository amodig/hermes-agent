"""Tests for hermes_cli.kanban_diagnostics — rule-engine that produces
structured distress signals (diagnostics) for kanban tasks.

These tests exercise each rule in isolation using minimal in-memory
task/event/run fixtures (no DB) plus a few integration-style cases
that round-trip through the real kanban_db to make sure the rule
engine works on sqlite3.Row objects as well as dataclasses.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_diagnostics as kd


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _task(**overrides):
    base = {
        "id": "t_demo00",
        "title": "demo task",
        "assignee": "demo",
        "status": "ready",
        "consecutive_failures": 0,
        "last_failure_error": None,
    }
    base.update(overrides)
    return base


def _event(kind, ts=None, **payload):
    return {
        "kind": kind,
        "created_at": int(ts if ts is not None else time.time()),
        "payload": payload or None,
    }


def _run(outcome="completed", run_id=1, error=None):
    return {
        "id": run_id,
        "outcome": outcome,
        "error": error,
    }


# ---------------------------------------------------------------------------
# Each rule — positive + negative + clearing
# ---------------------------------------------------------------------------
















def test_running_with_open_parents_fires_only_while_running():
    """A running card whose parent is not terminal is flagged; the same graph
    on a ready/todo card (the gate is holding it) and a done parent are not."""
    graph = {"parents": [{"id": "t_parent", "title": "p", "status": "todo"}], "children": []}
    diags = kd.compute_task_diagnostics(_task(status="running", started_at=100), [], [], graph=graph)
    assert [d.kind for d in diags] == ["running_with_open_parents"]
    assert diags[0].data["open_parents"] == [{"id": "t_parent", "status": "todo"}]
    assert "hermes kanban unlink t_parent t_demo00" in diags[0].actions[0].payload["command"]
    assert kd.compute_task_diagnostics(_task(status="todo"), [], [], graph=graph) == []
    done_graph = {"parents": [{"id": "t_parent", "title": "p", "status": "done"}], "children": []}
    assert kd.compute_task_diagnostics(_task(status="running"), [], [], graph=done_graph) == []


def test_stuck_in_blocked_fires_past_threshold():
    now = int(time.time())
    task = _task(status="blocked")
    events = [
        _event("blocked", ts=now - 3600 * 48, reason="needs approval"),
    ]
    diags = kd.compute_task_diagnostics(
        task, events, [], now=now,
    )
    assert len(diags) == 1
    d = diags[0]
    assert d.kind == "stuck_in_blocked"
    assert d.severity == "warning"
    assert d.data["age_hours"] >= 48








# ---------------------------------------------------------------------------
# Severity sorting
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Integration — runs through real kanban_db so sqlite.Row fields work
# ---------------------------------------------------------------------------


def test_engine_works_on_sqlite_row_objects(kanban_home):
    """Regression: the rule functions must handle sqlite3.Row (which
    supports mapping access but not attribute access and isn't a dict)
    as well as dataclass Task / plain dict. The API layer passes Row
    objects directly.
    """
    conn = kbc.connect()
    try:
        parent = kb.create_task(conn, title="p", assignee="w")
        real = kb.create_task(conn, title="r", assignee="x", created_by="w")
        with pytest.raises(kb.HallucinatedCardsError):
            kb.complete_task(
                conn, parent,
                summary="with phantom", created_cards=[real, "t_deadbeef1"],
            )
        # Pull Row objects the way the API helper does.
        row = conn.execute(
            "SELECT * FROM tasks WHERE id = ?", (parent,),
        ).fetchone()
        events = list(conn.execute(
            "SELECT * FROM task_events WHERE task_id = ? ORDER BY id",
            (parent,),
        ).fetchall())
        runs = list(conn.execute(
            "SELECT * FROM task_runs WHERE task_id = ? ORDER BY id",
            (parent,),
        ).fetchall())
        diags = kd.compute_task_diagnostics(row, events, runs)
        assert len(diags) == 1
        assert diags[0].kind == "hallucinated_cards"
        assert "t_deadbeef1" in diags[0].data["phantom_ids"]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Error-tolerance: a broken rule shouldn't 500 the whole compute call
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# stranded_in_ready
#
# Surfaces ready tasks that nobody has claimed within the threshold.
# Identity-agnostic by design: catches typo'd assignees, deleted profiles,
# down external worker pools, and misconfigured dispatchers in one rule.
# ---------------------------------------------------------------------------


def test_stranded_in_ready_fires_when_age_exceeds_threshold():
    """Default threshold = 30 min. A ready task promoted 45 min ago
    with no claim should fire as a warning."""
    now = 100_000
    task = _task(status="ready", assignee="demo", claim_lock=None)
    # 45 min = 2700s, threshold = 1800s.
    events = [_event("created", ts=now - 45 * 60)]
    diags = kd.compute_task_diagnostics(task, events, [], now=now)
    stranded = [d for d in diags if d.kind == "stranded_in_ready"]
    assert len(stranded) == 1
    assert stranded[0].severity == "warning"
    assert stranded[0].data["age_seconds"] == 45 * 60
    assert stranded[0].data["assignee"] == "demo"




# ---------------------------------------------------------------------------
# triage_aux_unavailable rule — auto-decompose aware
# ---------------------------------------------------------------------------


def _triage_task():
    return _task(id="t_triage1", status="triage")








def test_severity_at_or_above_uses_threshold_semantics():
    assert kd.severity_at_or_above("warning", "warning") is True
    assert kd.severity_at_or_above("error", "warning") is True
    assert kd.severity_at_or_above("critical", "warning") is True
    assert kd.severity_at_or_above("critical", "error") is True
    assert kd.severity_at_or_above("warning", "error") is False
    assert kd.severity_at_or_above("error", "critical") is False
    assert kd.severity_at_or_above("mystery", "warning") is False
    assert kd.severity_at_or_above("warning", None) is True


# ---------------------------------------------------------------------------
# respawn_guarded — the dispatcher's LIVE guard projection
#
# Ready is queue admission, not execution: a card can be queued forever while
# the respawn guard refuses to spawn it. The signal must come from the current
# guard state, never from old respawn_guarded events (an authorization makes
# those stale without deleting them).
# ---------------------------------------------------------------------------


def test_respawn_guarded_rule_reports_the_live_hold():
    now = 100_000
    task = _task(status="ready", assignee="demo", claim_lock=None)
    guard = {
        "reason": "active_pr",
        "recovery": "A recent comment records an existing GitHub PR ...",
        "command": "hermes kanban update t_demo00 --expected-version 3 --transition continue_existing_pr",
    }
    diags = kd.compute_task_diagnostics(
        task, [_event("respawn_guarded", ts=now - 60, reason="active_pr")], [],
        now=now, dispatch_guard=guard,
    )
    held = [d for d in diags if d.kind == "respawn_guarded"]
    assert len(held) == 1
    assert held[0].severity == "warning"
    assert held[0].title == "Dispatch held: active_pr"
    assert held[0].data["reason"] == "active_pr"
    # The drawer copies the command field verbatim, so it carries no prose.
    assert held[0].detail == guard["recovery"]
    hints = [a for a in held[0].actions if a.kind == "cli_hint"]
    assert any(a.payload["command"] == guard["command"] for a in hints)


def test_respawn_guarded_rule_offers_no_copy_action_without_a_command():
    """Guidance-only holds must not offer a clipboard action at all.

    A hint built here rather than taken from the dispatcher's projection cannot
    know which board the caller selected, so it would suggest a command for the
    wrong board.
    """
    task = _task(status="ready", assignee="demo", claim_lock=None)
    diags = kd.compute_task_diagnostics(
        task, [], [], now=100_000,
        dispatch_guard={"reason": "recent_success", "recovery": "Re-queue it", "command": ""},
    )
    held = [d for d in diags if d.kind == "respawn_guarded"][0]
    assert held.detail == "Re-queue it"
    assert [a for a in held.actions if a.kind == "cli_hint"] == []


def test_respawn_guarded_absent_without_a_live_projection():
    """A historical guard event is not a current hold."""
    now = 100_000
    task = _task(status="ready", assignee="demo", claim_lock=None)
    diags = kd.compute_task_diagnostics(
        task, [_event("respawn_guarded", ts=now - 60, reason="active_pr")], [], now=now,
    )
    assert [d for d in diags if d.kind == "respawn_guarded"] == []


def test_live_hold_suppresses_generic_stranded_ready_advice():
    now = 100_000
    task = _task(status="ready", assignee="demo", claim_lock=None)
    events = [_event("created", ts=now - 45 * 60)]
    without = kd.compute_task_diagnostics(task, events, [], now=now)
    assert [d.kind for d in without] == ["stranded_in_ready"]
    with_hold = kd.compute_task_diagnostics(
        task, events, [], now=now,
        dispatch_guard={"reason": "active_pr", "recovery": "fix it"},
    )
    assert [d.kind for d in with_hold] == ["respawn_guarded"]


def test_live_guard_clears_after_authorization_despite_recorded_events(kanban_home):
    """The projection is re-evaluated, so authorization clears the warning
    immediately even though the old respawn_guarded event stays in the log."""
    from hermes_cli import kanban_db_dispatch as kbd

    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="PR card", assignee="default")
        kb.add_comment(conn, tid, author="worker", body="https://github.com/o/r/pull/7")
        with kb.write_txn(conn):
            kb._append_event(conn, tid, "respawn_guarded", {"reason": "active_pr"})
        row = conn.execute("SELECT * FROM tasks WHERE id = ?", (tid,)).fetchone()
        events = list(conn.execute(
            "SELECT * FROM task_events WHERE task_id = ? ORDER BY id", (tid,),
        ).fetchall())

        def _kinds():
            return sorted(d.kind for d in kd.compute_task_diagnostics(
                row, events, [], config={"stranded_threshold_seconds": 1},
                dispatch_guard=kb.get_dispatch_guard(conn, tid),
            ))

        assert "respawn_guarded" in _kinds()
        assert kb.update_task(
            conn, tid,
            expected_version=kb.get_task(conn, tid).version,
            reason="explicitly authorized", transition="continue_existing_pr",
            authorized_pr_urls=["https://github.com/o/r/pull/7"],
        )
        assert "respawn_guarded" not in _kinds()

        # A held card whose assignee is not a real profile is nonspawnable
        # BEFORE the guard is consulted, so the guard is not its blocker: the
        # precise hold would otherwise hide the assignee failure.
        unheard = kb.create_task(conn, title="typo assignee", assignee="no-such-profile")
        kb.add_comment(conn, unheard, author="worker", body="https://github.com/o/r/pull/8")
        assert kb.get_dispatch_guard(conn, unheard, board="default") is None
        row = conn.execute("SELECT * FROM tasks WHERE id = ?", (unheard,)).fetchone()
        events = list(conn.execute(
            "SELECT * FROM task_events WHERE task_id = ? ORDER BY id", (unheard,),
        ).fetchall())
        kinds = sorted(d.kind for d in kd.compute_task_diagnostics(
            row, events, [], now=int(time.time()) + 3600,
            config={"stranded_threshold_seconds": 1},
            dispatch_guard=kb.get_dispatch_guard(conn, unheard),
        ))
        assert kinds == ["stranded_in_ready"], kinds
    finally:
        conn.close()

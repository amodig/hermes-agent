"""Tests for typed block reasons + the unblock-loop breaker.

Covers the built-in fix for the kanban "blocked loop" — a worker blocks a
task, a cron unblocks it, the worker re-blocks for the same reason, repeat
forever. The fix gives ``block_task`` a typed ``kind`` and a persistent
``block_recurrences`` counter:

* ``dependency`` blocks route to ``todo`` (parent-gated, auto-resumed) and
  never enter the human ``blocked`` bucket a cron would keep unblocking —
  unless no parent is open, in which case the wait can never be satisfied
  and the block is recorded as ``needs_input`` (sticky, loop-counted).
* ``needs_input`` / ``capability`` / un-typed blocks land in ``blocked``;
  each same-cause re-block after an unblock increments ``block_recurrences``,
  and at ``BLOCK_RECURRENCE_LIMIT`` the task routes to ``triage`` for a human.
* ``unblock_task`` deliberately does NOT reset ``block_recurrences`` (the
  amnesia that let the loop run unbounded).
* A successful ``complete_task`` resets the loop memory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from hermes_cli import kanban as kanban_cli
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _running_task(conn, title="t", parents=()):
    """Create a task (linked under ``parents`` first) and drive it to ``running`` so block_task can act."""
    tid = kb.create_task(conn, title=title, assignee="worker")
    for parent in parents:
        kb.link_tasks(conn, parent_id=parent, child_id=tid)
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
    claimed = kb.claim_task(conn, tid, claimer="worker")
    assert claimed is not None
    return tid


def _make_running_again(conn, tid):
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
    assert kb.claim_task(conn, tid, claimer="worker") is not None


# ---------------------------------------------------------------------------
# Loop breaker
# ---------------------------------------------------------------------------










def test_block_loop_detected_event_emitted(kanban_home: Path) -> None:
    with kbc.connect_closing() as conn:
        tid = _running_task(conn)
        kb.block_task(conn, tid, reason="x", kind="capability")
        kb.unblock_task(conn, tid)
        _make_running_again(conn, tid)
        kb.block_task(conn, tid, reason="x", kind="capability")
        events = [e for e in kb.list_events(conn, tid)
                  if e.kind == "block_loop_detected"]
        assert events, "expected a block_loop_detected event"
        payload = events[-1].payload or {}
        assert payload.get("recurrences") == 2
        assert payload.get("kind") == "capability"

def test_block_recurrence_normalizes_cause_and_resets_across_kinds(
    kanban_home: Path,
) -> None:
    with kbc.connect_closing() as conn:
        tid = _running_task(conn)

        assert kb.block_task(
            conn,
            tid,
            reason="Authorization is required from a human approver",
        )
        first = kb.get_task(conn, tid)
        assert first is not None
        assert first.status == "blocked"
        assert first.block_kind == "needs_input"
        assert first.block_recurrences == 1

        assert kb.unblock_task(conn, tid)
        _make_running_again(conn, tid)
        assert kb.block_task(
            conn,
            tid,
            reason="GitHub access is unavailable",
        )
        second = kb.get_task(conn, tid)
        assert second is not None
        assert second.status == "blocked"
        assert second.block_kind == "capability"
        assert second.block_recurrences == 1

        assert kb.unblock_task(conn, tid)
        _make_running_again(conn, tid)
        assert kb.block_task(
            conn,
            tid,
            reason="The access credential has expired",
            kind="auth",
        )
        third = kb.get_task(conn, tid)
        assert third is not None
        assert third.status == "triage"
        assert third.block_kind == "capability"
        assert third.block_recurrences == 2
        loop_events = [
            event
            for event in kb.list_events(conn, tid)
            if event.kind == "block_loop_detected"
        ]
        assert loop_events
        assert loop_events[-1].payload["kind"] == "capability"
        assert loop_events[-1].payload["recurrences"] == 2





# ---------------------------------------------------------------------------
# Dependency routing
# ---------------------------------------------------------------------------


def test_dependency_then_parent_done_promotes(kanban_home: Path) -> None:
    """A dependency-parked child becomes ready once its parent completes."""
    with kbc.connect_closing() as conn:
        parent = kb.create_task(conn, title="parent", assignee="worker")
        child = _running_task(conn, title="child")
        kb.link_tasks(
            conn,
            parent_id=parent,
            child_id=child,
            expected_child_run_id=kb.get_task(conn, child).current_run_id,
        )
        kb.block_task(conn, child, reason="wait", kind="dependency")
        assert kb.get_task(conn, child).status == "todo"
        # Finish the parent, then let recompute_ready run.
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (parent,))
        kb.claim_task(conn, parent, claimer="worker")
        kb.complete_task(conn, parent, result="done")
        kb.recompute_ready(conn)
        assert kb.get_task(conn, child).status == "ready"


def test_dependency_block_with_terminal_parents_parks_then_escalates(
    kanban_home: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """A ``dependency`` block whose parents are all terminal can never be
    satisfied by ``recompute_ready``: it must park in ``blocked`` as
    ``needs_input`` (no ``dependency_wait``, no re-promotion), say so on the
    CLI, and count toward the loop breaker so a re-block after an unblock
    reaches ``triage``."""
    with kbc.connect_closing() as conn:
        parent = kb.create_task(conn, title="already-done-parent", assignee="worker")
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='done' WHERE id=?", (parent,))
        # The edge exists before the run: link_tasks refuses to gate a running child retroactively.
        child = _running_task(conn, title="child-of-done", parents=(parent,))

        # `hermes kanban block <child> --kind dependency waiting on upstream`
        args = argparse.Namespace(task_id=child, ids=None, reason=["waiting", "on", "upstream"], kind="dependency")
        assert kanban_cli._cmd_block(args) == 0
        assert "needs_input" in capsys.readouterr().out
        parked = kb.get_task(conn, child)
        assert (parked.status, parked.block_kind, parked.block_recurrences) == ("blocked", "needs_input", 1)
        events = kb.list_events(conn, child)
        assert not [e for e in events if e.kind == "dependency_wait"]
        blocked = [e for e in events if e.kind == "blocked"][-1].payload
        assert (blocked["requested_kind"], blocked["rekind_reason"]) == ("dependency", "no_open_parent")
        assert kb.recompute_ready(conn) == 0
        assert kb.get_task(conn, child).status == "blocked"

        # A cron/human unblocks; the worker re-declares the same impossible wait.
        assert kb.unblock_task(conn, child)
        assert kb.claim_task(conn, child, claimer="worker") is not None
        assert kb.block_task(conn, child, reason="still waiting", kind="dependency")
        assert kb.get_task(conn, child).status == "triage"
        loop = [e for e in kb.list_events(conn, child) if e.kind == "block_loop_detected"][-1].payload
        assert loop["recurrences"] == kb.BLOCK_RECURRENCE_LIMIT


def test_dependency_block_with_open_parent_stays_parked_across_dispatch_tick(
    kanban_home: Path, all_assignees_spawnable,
) -> None:
    """Control: a genuine wait on an incomplete parent parks in ``todo``
    without counting a recurrence, survives a dispatch tick unspawned, and
    resumes once the parent finishes."""
    spawns: list[str] = []

    def fake_spawn(task, workspace, board=None):
        spawns.append(task.id)
        return 4242

    with kbc.connect() as conn:
        parent = kb.create_task(conn, title="open-parent", assignee="alice")
        # Linked while todo (a running child cannot be gated retroactively); the parent then stays
        # open while the child runs — the reopened-parent shape, forced the same way the loop below does.
        child = kb.create_task(conn, title="waiter", assignee="worker")
        kb.link_tasks(conn, parent_id=parent, child_id=child)
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='running' WHERE id=?", (child,))
        for _ in range(kb.BLOCK_RECURRENCE_LIMIT + 1):
            assert kb.block_task(conn, child, reason="wait", kind="dependency")
            parked = kb.get_task(conn, child)
            assert (parked.status, parked.block_kind, parked.block_recurrences) == ("todo", "dependency", 0)
            with kb.write_txn(conn):
                conn.execute("UPDATE tasks SET status='running' WHERE id=?", (child,))
        assert kb.block_task(conn, child, reason="wait", kind="dependency")
        res = kbd.dispatch_once(conn, spawn_fn=fake_spawn)
        assert kb.get_task(conn, child).status == "todo"
        assert child not in spawns
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (parent,))
        kb.claim_task(conn, parent, claimer="alice")
        kb.complete_task(conn, parent, result="done")
        res2 = kbd.dispatch_once(conn, spawn_fn=fake_spawn)
        assert child in [row[0] for row in res2.spawned]


# ---------------------------------------------------------------------------
# Completion resets loop memory
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Validation + back-compat
# ---------------------------------------------------------------------------



def test_legacy_null_block_kind_reuses_historical_recurrence_cause(
    kanban_home: Path,
) -> None:
    with kbc.connect_closing() as conn:
        tid = _running_task(conn)
        assert kb.block_task(conn, tid, reason="GitHub access is unavailable")

        # Simulate an upgraded legacy row: the persisted kind was not
        # backfilled, but its recurrence count and blocking history remain.
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status = 'ready', block_kind = NULL, "
                "block_recurrences = 1 WHERE id = ?",
                (tid,),
            )

        assert kb.block_task(conn, tid, reason="GitHub access is unavailable")
        task = kb.get_task(conn, tid)
        loop_events = [
            event
            for event in kb.list_events(conn, tid)
            if event.kind == "block_loop_detected"
        ]

    assert task is not None
    assert task.status == "triage"
    assert task.block_kind == "capability"
    assert task.block_recurrences == 2
    assert loop_events
    assert loop_events[-1].payload["cause"] == "capability"
    assert loop_events[-1].payload["recurrences"] == 2


# ---------------------------------------------------------------------------
# Version-fenced unblock (operator recovery)
# ---------------------------------------------------------------------------


def _blocked_task(conn, title="fenced"):
    tid = _running_task(conn, title=title)
    assert kb.block_task(conn, tid, reason="needs a human")
    return tid


def _snapshot(conn, tid):
    task = kb.get_task(conn, tid)
    return (
        task.status, task.version, task.consecutive_failures, task.last_failure_error,
    ), [event.kind for event in kb.list_events(conn, tid)], [
        (comment.author, comment.body) for comment in kb.list_comments(conn, tid)
    ]


def test_fenced_unblock_succeeds_once_and_increments_version_once(kanban_home: Path) -> None:
    with kbc.connect_closing() as conn:
        tid = _blocked_task(conn)
        version = kb.get_task(conn, tid).version

        assert kb.unblock_task(
            conn, tid, expected_version=version, reason="policy installed", author="operator",
        )
        task = kb.get_task(conn, tid)
        assert (task.status, task.version) == ("ready", version + 1)
        assert kb.list_comments(conn, tid)[-1].body == "UNBLOCK: policy installed"
        assert kb.list_comments(conn, tid)[-1].author == "operator"
        assert [e for e in kb.list_events(conn, tid) if e.kind == "unblocked"]

        # The same fence can never act twice.
        assert _snapshot(conn, tid) == _snapshot(conn, tid)
        before = _snapshot(conn, tid)
        assert kb.unblock_task(
            conn, tid, expected_version=version, reason="policy installed", author="operator",
        ) is False
        assert _snapshot(conn, tid) == before


def test_fenced_unblock_refuses_a_stale_version_without_any_write(kanban_home: Path) -> None:
    with kbc.connect_closing() as conn:
        tid = _blocked_task(conn)
        version = kb.get_task(conn, tid).version
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET version = version + 1, consecutive_failures = 3, "
                "last_failure_error = 'prior failure' WHERE id = ?", (tid,),
            )
        before = _snapshot(conn, tid)

        assert kb.unblock_task(
            conn, tid, expected_version=version, reason="stale recovery", author="operator",
        ) is False
        assert _snapshot(conn, tid) == before


def test_fenced_unblock_refuses_an_active_claim_before_reclaiming(kanban_home: Path) -> None:
    with kbc.connect_closing() as conn:
        tid = _blocked_task(conn)
        with kb.write_txn(conn):
            run_id = conn.execute(
                "INSERT INTO task_runs (task_id, profile, status, started_at) "
                "VALUES (?, 'worker', 'blocked', 1)", (tid,),
            ).lastrowid
            conn.execute(
                "UPDATE tasks SET claim_lock = 'lock-fixture', current_run_id = ?, "
                "worker_pid = 424242 WHERE id = ?", (run_id, tid),
            )
        version = kb.get_task(conn, tid).version
        before = _snapshot(conn, tid)

        assert kb.unblock_task(
            conn, tid, expected_version=version, reason="recover", author="operator",
        ) is False
        assert _snapshot(conn, tid) == before
        # The dangling run is left exactly as found: no reclaim on an uncertain owner.
        run = conn.execute("SELECT status, ended_at FROM task_runs WHERE id = ?", (run_id,)).fetchone()
        assert (run["status"], run["ended_at"]) == ("blocked", None)


def test_fenced_unblock_refuses_a_card_that_is_not_blocked(kanban_home: Path) -> None:
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="already ready", assignee="worker")
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (tid,))
        before = _snapshot(conn, tid)
        assert kb.unblock_task(conn, tid, expected_version=kb.get_task(conn, tid).version) is False
        assert _snapshot(conn, tid) == before


def test_fenced_unblock_rejects_invalid_versions(kanban_home: Path) -> None:
    with kbc.connect_closing() as conn:
        tid = _blocked_task(conn)
        before = _snapshot(conn, tid)
        for value in (True, False, 0, -1, 1.5, "2", [1]):
            with pytest.raises(ValueError):
                kb.unblock_task(conn, tid, expected_version=value)
        assert _snapshot(conn, tid) == before


def test_fenced_unblock_preserves_resume_gating_and_phase(kanban_home: Path) -> None:
    with kbc.connect_closing() as conn:
        parent = kb.create_task(conn, title="open parent", assignee="alice")
        # Linked while todo (a running child cannot be gated retroactively); the parent
        # then stays open while the child runs, exactly as the dispatch-tick test forces it.
        child = kb.create_task(conn, title="waiter", assignee="worker")
        kb.link_tasks(conn, parent_id=parent, child_id=child)
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status = 'running' WHERE id = ?", (child,))
        # An untyped block parks it on the human bucket, so it really is unblockable…
        assert kb.block_task(conn, child, reason="waiting on upstream")
        parked = kb.get_task(conn, child)
        assert parked.status == "blocked"

        assert kb.unblock_task(
            conn, child, expected_version=parked.version, reason="dependency restored",
            author="operator",
        )
        resumed = kb.get_task(conn, child)
        # …but the open parent still wins over the resume phase, and the version moved once.
        assert (resumed.status, resumed.version) == ("todo", parked.version + 1)


def test_unfenced_unblock_keeps_its_old_behaviour(kanban_home: Path) -> None:
    with kbc.connect_closing() as conn:
        tid = _blocked_task(conn)
        version = kb.get_task(conn, tid).version
        assert kb.unblock_task(conn, tid)
        task = kb.get_task(conn, tid)
        assert task.status == "ready"
        # No fence, no version bump and no comment: the existing contract is untouched.
        assert task.version == version
        assert not [c for c in kb.list_comments(conn, tid) if c.body.startswith("UNBLOCK:")]


def test_cli_fenced_unblock_records_reason_only_on_success(
    kanban_home: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    with kbc.connect_closing() as conn:
        tid = _blocked_task(conn)
        # Two versions, so the stale fence below is still a valid version number.
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET version = version + 1 WHERE id = ?", (tid,))
        version = kb.get_task(conn, tid).version

        stale = argparse.Namespace(
            task_ids=[tid], reason="operator recovery", expected_version=version - 1,
        )
        assert kanban_cli._cmd_unblock(stale) == 1
        assert "stale version" in capsys.readouterr().err
        assert not [c for c in kb.list_comments(conn, tid) if c.body.startswith("UNBLOCK:")]

        fenced = argparse.Namespace(
            task_ids=[tid], reason="operator recovery", expected_version=version,
        )
        assert kanban_cli._cmd_unblock(fenced) == 0
        assert "Unblocked" in capsys.readouterr().out
        assert kb.list_comments(conn, tid)[-1].body == "UNBLOCK: operator recovery"
        assert kb.get_task(conn, tid).status == "ready"


def test_cli_fenced_unblock_requires_a_single_task(
    kanban_home: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    with kbc.connect_closing() as conn:
        first = _blocked_task(conn, title="one")
        second = _blocked_task(conn, title="two")
        before = _snapshot(conn, first), _snapshot(conn, second)
        args = argparse.Namespace(
            task_ids=[first, second], reason="bulk", expected_version=kb.get_task(conn, first).version,
        )
        assert kanban_cli._cmd_unblock(args) == 2
        assert "--expected-version requires exactly one task id" in capsys.readouterr().err
        assert (_snapshot(conn, first), _snapshot(conn, second)) == before

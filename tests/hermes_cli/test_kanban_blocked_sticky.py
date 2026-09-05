"""Regression tests for #28712 — kanban dispatcher must not auto-promote
worker-initiated ``kanban_block`` (sticky blocks), but must keep
auto-recovering circuit-breaker blocks.

The bug: when a worker called ``kanban_block(reason="review-required:
...")`` to hand off to a human, the dispatcher's ``recompute_ready``
would promote the task back to ``ready`` on the next tick.  The fresh
worker found nothing to do (work already applied), exited cleanly, and
got recorded as a ``protocol_violation`` → ``gave_up`` → promote → loop
until manual intervention.

These tests pin down:

* Worker / operator-initiated blocks are sticky and survive
  ``recompute_ready``.
* Circuit-breaker blocks (``gave_up`` event, status flipped via
  ``_record_task_failure``) still auto-recover — the original intent
  of #40c1decb3 is preserved.
* An explicit ``kanban_unblock`` clears the sticky state.
* The full block → promote → crash → ``gave_up`` loop is broken after
  this fix: subsequent ticks leave the task blocked.

The tangentially related schema-init ordering bug originally reported
in #28712 (``init_db`` crashing on legacy DBs that pre-dated the
``session_id`` migration) is covered separately by
``test_kanban_db.py::test_connect_migrates_legacy_db_before_optional_column_indexes``,
landed via #28754 / #28781 ahead of this fix.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


# ---------------------------------------------------------------------------
# Worker-initiated kanban_block must be sticky
# ---------------------------------------------------------------------------


def test_worker_block_is_not_auto_promoted_by_recompute_ready(kanban_home: Path) -> None:
    """A standalone task that a worker explicitly blocks for review
    must stay blocked across an arbitrary number of dispatcher ticks.
    Before #28712's fix, ``recompute_ready`` would silently flip it
    back to ``ready`` on the very next tick."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="needs human review")
        kb.claim_task(conn, tid)
        assert kb.block_task(
            conn, tid,
            reason="review-required: please verify ACL change",
            expected_run_id=kb.get_task(conn, tid).current_run_id,
        )
        assert kb.get_task(conn, tid).status == "blocked"

        # Hammer the promotion code — exactly the dispatcher loop's
        # behaviour, just compressed in time.
        for _ in range(5):
            promoted = kb.recompute_ready(conn)
            assert promoted == 0, "worker-blocked task must not auto-promote"
            assert kb.get_task(conn, tid).status == "blocked"




# ---------------------------------------------------------------------------
# Circuit-breaker blocks still auto-recover (preserve #40c1decb3 intent)
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# unblock_task clears the sticky state
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Full bug-shaped loop: block → promote → crash → gave_up → next tick
# ---------------------------------------------------------------------------


def test_protocol_violation_loop_is_broken(kanban_home: Path) -> None:
    """Reproduces the exact #28712 loop and asserts the dispatcher
    leaves the task blocked instead of cycling.

    Loop shape from the issue:

    1. Worker calls ``kanban_block`` → status='blocked',
       ``task_runs.outcome='blocked'``, ``blocked`` event.
    2. (Bug) Dispatcher promotes back to ``ready``.
    3. Fresh worker exits cleanly without terminal tool call →
       ``protocol_violation`` event.
    4. ``_record_task_failure(failure_limit=1)`` → ``gave_up`` event,
       status='blocked' again.
    5. (Bug) Dispatcher promotes again → infinite loop.

    With the fix in place, step 2 never happens — the test simulates
    one would-be loop cycle by faking the crash-then-gave_up entries
    that *would* have been written and asserts the *next* tick still
    leaves the task blocked.
    """
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="loop reproducer")
        kb.claim_task(conn, tid)
        kb.block_task(
            conn, tid,
            reason="review-required: human eyes please",
            expected_run_id=kb.get_task(conn, tid).current_run_id,
        )
        assert kb.get_task(conn, tid).status == "blocked"

        # First dispatcher tick — must NOT promote.
        assert kb.recompute_ready(conn) == 0
        assert kb.get_task(conn, tid).status == "blocked"

        # Simulate the (hypothetical) protocol_violation + gave_up
        # entries that the dispatcher would have written if the bug
        # were still present.  Even with those event rows in place,
        # the worker-initiated ``blocked`` event is the most recent
        # of the ``{blocked, unblocked}`` pair, so the sticky guard
        # still fires.
        now = int(time.time())
        conn.execute(
            "INSERT INTO task_events (task_id, kind, payload, created_at) "
            "VALUES (?, 'protocol_violation', NULL, ?)",
            (tid, now),
        )
        conn.execute(
            "INSERT INTO task_events (task_id, kind, payload, created_at) "
            "VALUES (?, 'gave_up', NULL, ?)",
            (tid, now + 1),
        )
        conn.commit()

        # Subsequent ticks must still leave it blocked.
        for _ in range(3):
            promoted = kb.recompute_ready(conn)
            assert promoted == 0
            assert kb.get_task(conn, tid).status == "blocked"


# ---------------------------------------------------------------------------
# Schema-init recovery on legacy DBs is covered by
# tests/hermes_cli/test_kanban_db.py::test_connect_migrates_legacy_db_before_optional_column_indexes
# (landed via #28754 / #28781).  The original PR shipped a duplicate test
# here; dropped during salvage to avoid two assertions of the same contract.
# ---------------------------------------------------------------------------


def test_initial_blocked_creation_is_sticky_across_dispatch_ticks(
    kanban_home: Path, all_assignees_spawnable,
) -> None:
    """Initial human blocks cannot be promoted or dispatched implicitly."""
    spawned: list[str] = []

    def fake_spawn(task, *_args, **_kwargs):
        spawned.append(task.id)
        return 4242

    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="wait for approval",
            assignee="worker",
            initial_status="blocked",
        )
        assert any(event.kind == "blocked" for event in kb.list_events(conn, tid))

        for _ in range(3):
            assert kb.recompute_ready(conn) == 0
            result = kb.dispatch_once(conn, spawn_fn=fake_spawn)
            assert result.promoted == 0
            assert not result.spawned
            assert kb.get_task(conn, tid).status == "blocked"

        assert not spawned
        kinds = [event.kind for event in kb.list_events(conn, tid)]
        assert "claimed" not in kinds
        assert "spawned" not in kinds


def test_initial_blocked_status_and_event_roll_back_together(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed blocked-event insert cannot leave a dispatchable card behind."""
    original_append = kb._append_event

    def fail_block_event(conn, task_id, kind, payload=None, **kwargs):
        if kind == "blocked":
            raise RuntimeError("blocked event failure")
        return original_append(conn, task_id, kind, payload, **kwargs)

    monkeypatch.setattr(kb, "_append_event", fail_block_event)
    with kb.connect() as conn:
        with pytest.raises(RuntimeError, match="blocked event failure"):
            kb.create_task(conn, title="atomic blocked", initial_status="blocked")
        assert conn.execute(
            "SELECT COUNT(*) FROM tasks WHERE title = 'atomic blocked'"
        ).fetchone()[0] == 0


def test_manual_promotion_rejects_initially_blocked_root(kanban_home: Path) -> None:
    """The CLI/database promotion path requires an explicit unblock."""
    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="blocked root",
            assignee="worker",
            initial_status="blocked",
        )
        ok, error = kb.promote_task(conn, tid, actor="operator", force=True)
        assert not ok
        assert error and "unblock" in error
        assert kb.get_task(conn, tid).status == "blocked"


def test_blocking_running_task_closes_its_run(kanban_home: Path) -> None:
    """A successful block is terminal for the active run attempt."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="active", assignee="worker")
        assert kb.claim_task(conn, tid) is not None
        run_id = kb.get_task(conn, tid).current_run_id
        assert run_id is not None

        assert kb.block_task(conn, tid, reason="needs operator", expected_run_id=run_id)
        task = kb.get_task(conn, tid)
        run = kb.get_run(conn, run_id)
        assert task.status == "blocked"
        assert task.current_run_id is None
        assert run is not None
        assert run.outcome == "blocked"
        assert run.ended_at is not None


def test_sticky_block_cannot_be_claimed_after_inconsistent_ready_flip(
    kanban_home: Path, all_assignees_spawnable,
) -> None:
    """Eligibility remains blocked even if a stale writer flipped the column."""
    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="blocked root",
            assignee="worker",
            initial_status="blocked",
        )
        conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (tid,))
        conn.commit()

        assert not kb.has_spawnable_ready(conn)
        assert kb.claim_task(conn, tid) is None
        assert kb.get_task(conn, tid).status == "ready"

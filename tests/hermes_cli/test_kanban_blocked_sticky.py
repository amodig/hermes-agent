"""Regression tests for issue #8 — exhausted kanban runs are final.

Worker / operator blocks remain sticky, and a circuit-breaker ``gave_up`` event
is now sticky as well.  The dispatcher must not promote an exhausted task or
spawn another worker until an explicit operator retry/unblock transition.

The worker-side stop guard and protocol accounting are covered in the focused
agent/core tests; this module pins the dispatcher state-machine contract.
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
# Circuit-breaker exhaustion is final until an operator recovery transition
# ---------------------------------------------------------------------------


def test_gave_up_is_not_auto_promoted_until_unblocked(kanban_home: Path) -> None:
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="exhausted", max_retries=1)
        assert kb.claim_task(conn, tid) is not None
        assert kb._record_task_failure(
            conn,
            tid,
            "worker crashed",
            outcome="crashed",
            failure_limit=1,
            release_claim=True,
            end_run=True,
        )
        assert kb.get_task(conn, tid).status == "blocked"
        assert kb._has_sticky_block(conn, tid)

        # Exercise the event guard rather than only the legacy counter guard.
        conn.execute(
            "UPDATE tasks SET consecutive_failures = 0 WHERE id = ?", (tid,),
        )
        conn.commit()
        for _ in range(3):
            assert kb.recompute_ready(conn) == 0
            assert kb.get_task(conn, tid).status == "blocked"

        assert not kb.promote_task(conn, tid, actor="operator")[0]
        assert kb.unblock_task(conn, tid)
        assert kb.get_task(conn, tid).status == "ready"
        assert not kb._has_sticky_block(conn, tid)


def test_gave_up_notifications_are_deduplicated_per_incident(kanban_home: Path) -> None:
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="notify exhaustion")
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-1")
        kb._append_event(conn, tid, "gave_up", {"incident": "same"})
        kb._append_event(conn, tid, "gave_up", {"incident": "same"})
        conn.commit()

        _, _, events = kb.claim_unseen_events_for_sub(
            conn,
            task_id=tid,
            platform="telegram",
            chat_id="chat-1",
            kinds=["gave_up"],
        )
        assert len(events) == 1

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
        # Simulate the protocol_violation + gave_up entries that an exhausted
        # clean-exit run produces. The latest ``gave_up`` recovery event is
        # itself terminal, so the sticky guard must continue to fire.
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

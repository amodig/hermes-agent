"""Ordinary worker crashes must be booked against the CONFIGURED failure limit.

``dispatch_once`` has always threaded ``kanban.failure_limit`` into
``_run_reclaim_phase``, but the reclaim phase dropped it on the way to
``detect_crashed_workers``, so a crash was always counted against
``DEFAULT_FAILURE_LIMIT`` (2). A CTO dispatcher configured with 3 therefore
blocked a card on its second crash instead of its third — the accounting half
of issue #43.

Deliberate systemic-limit (>= 3 identical fingerprints), protocol-violation and
quota-wall policies keep their own precedence and are not re-specified here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd

DEAD_PID = 98765


@pytest.fixture
def conn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    # The fork -> /proc grace window exists for freshly spawned workers, not for
    # a worker that never had a live PID in this test.
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    with kbc.connect() as open_conn:
        yield open_conn


def _claim_dead_worker(conn, tid: str) -> None:
    """Claim ``tid`` and record a worker PID that is not alive."""
    assert kb.claim_task(conn, tid, claimer=f"{kb._claimer_id().split(':', 1)[0]}:{tid}") is not None
    kbd._set_worker_pid(conn, tid, DEAD_PID)


def _row(conn, tid: str):
    return conn.execute(
        "SELECT status, consecutive_failures FROM tasks WHERE id = ?", (tid,),
    ).fetchone()


def _gave_up(conn, tid: str) -> dict:
    row = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'gave_up' "
        "ORDER BY id DESC LIMIT 1", (tid,),
    ).fetchone()
    return json.loads(row["payload"]) if row is not None and row["payload"] else {}


def _reclaim(conn, *, failure_limit: int | None):
    """One crash/reclaim cycle through the real reclaim phase."""
    if failure_limit is None:
        return kbd.detect_crashed_workers(conn)
    return kbd.detect_crashed_workers(conn, failure_limit=failure_limit)


def test_reclaim_phase_forwards_the_configured_limit(conn):
    """The exact defect: three crashes with limit 3 stay retryable until the third."""
    tid = kb.create_task(conn, title="reclaim phase limit", assignee="w")

    for expected_failures in (1, 2):
        _claim_dead_worker(conn, tid)
        result = kbd.DispatchResult()
        kbd._run_reclaim_phase(
            conn, result, stale_timeout_seconds=0, failure_limit=3,
            reconcile_orphans=True,
        )
        assert result.crashed == [tid]
        assert result.auto_blocked == []
        row = _row(conn, tid)
        assert row["status"] == "ready", f"crash {expected_failures} must stay retryable"
        assert row["consecutive_failures"] == expected_failures

    _claim_dead_worker(conn, tid)
    result = kbd.DispatchResult()
    kbd._run_reclaim_phase(
        conn, result, stale_timeout_seconds=0, failure_limit=3, reconcile_orphans=True,
    )
    assert result.auto_blocked == [tid]
    row = _row(conn, tid)
    assert (row["status"], row["consecutive_failures"]) == ("blocked", 3)
    payload = _gave_up(conn, tid)
    assert payload["effective_limit"] == 3
    assert payload["limit_source"] == "dispatcher"
    assert payload["trigger_outcome"] == "crashed"


def test_configured_limit_keeps_ordinary_crashes_retryable_until_three(conn):
    """Same contract through ``detect_crashed_workers``'s own keyword."""
    tid = kb.create_task(conn, title="crash limit 3", assignee="w")

    for expected_failures in (1, 2):
        _claim_dead_worker(conn, tid)
        assert _reclaim(conn, failure_limit=3) == [tid]
        assert getattr(kbd.detect_crashed_workers, "_last_auto_blocked") == []
        row = _row(conn, tid)
        assert row["status"] == "ready"
        assert row["consecutive_failures"] == expected_failures

    _claim_dead_worker(conn, tid)
    assert _reclaim(conn, failure_limit=3) == [tid]
    assert getattr(kbd.detect_crashed_workers, "_last_auto_blocked") == [tid]
    assert _row(conn, tid)["status"] == "blocked"
    payload = _gave_up(conn, tid)
    assert payload["effective_limit"] == 3
    assert payload["limit_source"] == "dispatcher"


def test_default_limit_still_applies_to_direct_callers(conn):
    """``failure_limit=None`` keeps ``DEFAULT_FAILURE_LIMIT``: existing callers unchanged."""
    tid = kb.create_task(conn, title="default limit", assignee="w")

    _claim_dead_worker(conn, tid)
    assert _reclaim(conn, failure_limit=None) == [tid]
    assert _row(conn, tid)["status"] == "ready"

    _claim_dead_worker(conn, tid)
    assert _reclaim(conn, failure_limit=None) == [tid]
    row = _row(conn, tid)
    assert row["status"] == "blocked"
    assert row["consecutive_failures"] == kb.DEFAULT_FAILURE_LIMIT
    assert _gave_up(conn, tid)["effective_limit"] == kb.DEFAULT_FAILURE_LIMIT


def test_task_max_retries_outranks_the_configured_limit(conn):
    tid = kb.create_task(conn, title="task override", assignee="w", max_retries=1)

    _claim_dead_worker(conn, tid)
    assert _reclaim(conn, failure_limit=5) == [tid]
    row = _row(conn, tid)
    assert row["status"] == "blocked"
    payload = _gave_up(conn, tid)
    assert payload["limit_source"] == "task"
    assert payload["effective_limit"] == 1


def test_repeated_reclaim_does_not_charge_the_same_crash_twice(conn):
    tid = kb.create_task(conn, title="no double charge", assignee="w")

    _claim_dead_worker(conn, tid)
    assert _reclaim(conn, failure_limit=3) == [tid]
    assert _row(conn, tid)["consecutive_failures"] == 1

    assert kbd.detect_crashed_workers(conn, failure_limit=3) == []
    assert kbd.detect_crashed_workers(conn) == []
    assert _row(conn, tid)["consecutive_failures"] == 1

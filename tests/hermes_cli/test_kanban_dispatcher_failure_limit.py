"""Ordinary worker crashes must be booked against the CONFIGURED failure limit.

One invariant on one surface (the dispatcher's crash/breaker accounting):
``effective limit = task max_retries > kanban.failure_limit > built-in default``,
charged once per crash. ``dispatch_once`` always threaded the configured limit
into ``_run_reclaim_phase``, which dropped it before crash accounting, so a
dispatcher configured for three attempts blocked a card on its second crash —
the accounting half of issue #43.

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


def _crash(conn, tid: str, *, failure_limit: int | None, phase: bool) -> list[str]:
    """One ordinary worker crash/reclaim cycle.

    ``phase=True`` drives the real reclaim phase (the wiring this fixes);
    ``phase=False`` calls crash accounting directly, which must keep the
    built-in default for callers that pass no limit.
    """
    assert kb.claim_task(conn, tid, claimer=f"{kb._claimer_id().split(':', 1)[0]}:{tid}") is not None
    kbd._set_worker_pid(conn, tid, DEAD_PID)
    if not phase:
        return kbd.detect_crashed_workers(conn, failure_limit=failure_limit)
    result = kbd.DispatchResult()
    kbd._run_reclaim_phase(
        conn, result, stale_timeout_seconds=0, failure_limit=failure_limit,
        reconcile_orphans=True,
    )
    return result.crashed


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


def test_configured_failure_limit_bounds_ordinary_crashes_until_it_is_reached(conn):
    configured = kb.create_task(conn, title="configured limit", assignee="w")
    for crashes in (1, 2):
        assert _crash(conn, configured, failure_limit=3, phase=True) == [configured]
        row = _row(conn, configured)
        assert (row["status"], row["consecutive_failures"]) == ("ready", crashes), (
            f"crash {crashes} must stay retryable with kanban.failure_limit=3"
        )

    assert _crash(conn, configured, failure_limit=3, phase=True) == [configured]
    row = _row(conn, configured)
    assert (row["status"], row["consecutive_failures"]) == ("blocked", 3)
    payload = _gave_up(conn, configured)
    assert payload["effective_limit"] == 3
    assert payload["limit_source"] == "dispatcher"
    assert payload["trigger_outcome"] == "crashed"

    # A reclaim that finds nothing new must not charge the same crash twice.
    assert kbd.detect_crashed_workers(conn, failure_limit=3) == []
    assert kbd.detect_crashed_workers(conn) == []
    assert _row(conn, configured)["consecutive_failures"] == 3

    # Direct callers that pass no limit keep the built-in default.
    defaulted = kb.create_task(conn, title="default limit", assignee="w")
    for crashes in (1, 2):
        assert _crash(conn, defaulted, failure_limit=None, phase=False) == [defaulted]
        row = _row(conn, defaulted)
        if crashes == 1:
            assert (row["status"], row["consecutive_failures"]) == ("ready", 1)
        else:
            assert (row["status"], row["consecutive_failures"]) == (
                "blocked", kb.DEFAULT_FAILURE_LIMIT,
            )
    assert _gave_up(conn, defaulted)["effective_limit"] == kb.DEFAULT_FAILURE_LIMIT


def test_task_max_retries_outranks_the_configured_limit(conn):
    tid = kb.create_task(conn, title="task override", assignee="w", max_retries=1)

    assert _crash(conn, tid, failure_limit=5, phase=True) == [tid]
    assert _row(conn, tid)["status"] == "blocked"
    payload = _gave_up(conn, tid)
    assert payload["limit_source"] == "task"
    assert payload["effective_limit"] == 1

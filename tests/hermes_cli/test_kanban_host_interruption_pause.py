"""A replaced host instantiation must PAUSE the card, never kill or charge it.

Issue #43: a reboot (or container restart) leaves ``running`` cards whose worker
is gone but whose PID may already belong to an unrelated process. This file pins
the contracts that break independently:

* the pause itself (reclaim state machine: exactly once, sticky, no charge);
* what does and does not count as proof of a new instantiation;
* what the operator gets back on unblock (phase, pins, graph).

The epoch swap is INJECTED (``kb._current_host_epoch`` is monkeypatched); none of
this is a real reboot, which needs a disposable host.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd

OLD_EPOCH = "11111111-1111-1111-1111-111111111111:10"
NEW_EPOCH = "22222222-2222-2222-2222-222222222222:20"
LIVE_PID = 999_991
HOST = "thishost"


@pytest.fixture
def conn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    with kbc.connect() as open_conn:
        yield open_conn


def _liveness(monkeypatch: pytest.MonkeyPatch, alive: bool) -> None:
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: alive)


def _forbid_pid_use(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make any consult of the stale PID fatal.

    "Do not probe or signal a possibly reused PID" is only proven by making the
    consult fail, not by stubbing liveness to a plausible answer.
    """
    def _probe(pid):
        raise AssertionError(f"the stale PID {pid} must never be probed")

    def _terminate(pid, claim_lock, *, signal_fn=None):
        raise AssertionError(f"the stale PID {pid} must never be signalled")

    monkeypatch.setattr(kb, "_pid_alive", _probe)
    monkeypatch.setattr(kb, "_terminate_reclaimed_worker", _terminate)


def _claim(conn, tid: str, *, lane: str = "ready", epoch: str = OLD_EPOCH,
           monkeypatch) -> int:
    """Claim ``tid`` as this host with a recorded epoch; returns the run id."""
    monkeypatch.setattr(kb, "_current_host_epoch", lambda: epoch)
    claimer = f"{kb._claimer_id().split(':', 1)[0]}:t"
    if lane == "review":
        assert kb.claim_review_task(conn, tid, claimer=claimer) is not None
    else:
        assert kb.claim_task(conn, tid, claimer=claimer) is not None
    kbd._set_worker_pid(conn, tid, LIVE_PID)
    run_id = kb._current_run_id(conn, tid)
    assert run_id is not None
    return run_id


def _reclaim(conn) -> kbd.DispatchResult:
    result = kbd.DispatchResult()
    kbd._run_reclaim_phase(
        conn, result, stale_timeout_seconds=0, failure_limit=3, reconcile_orphans=True,
    )
    return result


def _row(conn, tid: str):
    return conn.execute(
        "SELECT status, consecutive_failures, claim_lock, worker_pid, block_kind, "
        "       block_recurrences, model_override, provider_override, "
        "       reasoning_effort, branch_name, current_step_key "
        "FROM tasks WHERE id = ?", (tid,),
    ).fetchone()


def _run_row(conn, run_id: int):
    return conn.execute(
        "SELECT status, outcome, ended_at FROM task_runs WHERE id = ?", (run_id,),
    ).fetchone()


def _events(conn, tid: str, kind: str) -> list[dict]:
    rows = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = ? ORDER BY id",
        (tid, kind),
    ).fetchall()
    return [json.loads(row["payload"]) for row in rows if row["payload"]]


def _set_recorded_epoch(conn, run_id: int, value: str | None) -> None:
    """Force (or strip) the run's recorded epoch, as an old receipt would look."""
    conn.execute(
        "UPDATE task_runs SET metadata = json_set(metadata, '$.host_epoch', ?) "
        "WHERE id = ?", (value, run_id),
    )
    conn.commit()


def test_positive_host_change_pauses_the_run_exactly_once(conn, monkeypatch):
    _forbid_pid_use(monkeypatch)
    tid = kb.create_task(conn, title="reused pid", assignee="w")
    run_id = _claim(conn, tid, monkeypatch=monkeypatch)

    monkeypatch.setattr(kb, "_current_host_epoch", lambda: NEW_EPOCH)
    result = _reclaim(conn)

    assert (result.interrupted, result.crashed, result.auto_blocked) == ([tid], [], [])
    row = _row(conn, tid)
    assert row["status"] == "blocked"
    assert row["consecutive_failures"] == 0, "a reboot must not charge a retry"
    assert row["claim_lock"] is None and row["worker_pid"] is None
    run = _run_row(conn, run_id)
    assert (run["status"], run["outcome"]) == ("interrupted", "interrupted")
    assert run["ended_at"] is not None
    blocked = _events(conn, tid, "blocked")
    assert len(blocked) == 1
    assert blocked[0]["reason"] == kbd.HOST_RESTART_BLOCK_REASON
    assert blocked[0]["recorded_host_epoch"] == OLD_EPOCH
    assert blocked[0]["host_epoch"] == NEW_EPOCH
    assert blocked[0]["retry_status"] == "ready"
    assert blocked[0]["run_id"] == run_id

    # Repeating the reclaim must not pause the same interruption twice, and the
    # sticky block must survive ordinary ticks and only yield to ``unblock``.
    assert _reclaim(conn).interrupted == []
    assert _reclaim(conn).interrupted == []
    for _ in range(3):
        tick = kbd.dispatch_once(conn, failure_limit=3)
        assert (tick.interrupted, tick.promoted, tick.spawned) == ([], 0, [])
        assert _row(conn, tid)["status"] == "blocked"

    assert kb.unblock_task(conn, tid) is True
    row = _row(conn, tid)
    assert row["status"] == "ready"
    assert row["consecutive_failures"] == 0, "unblock's counter semantics are unchanged"
    # Payload is NULL for a plain ready resume; the transition itself is the record.
    unblocked = conn.execute(
        "SELECT COUNT(*) AS n FROM task_events WHERE task_id = ? AND kind = 'unblocked'",
        (tid,),
    ).fetchone()
    assert unblocked["n"] == 1


def test_only_positive_provenance_exempts_a_card(conn, monkeypatch):
    """Every unproven case keeps its pre-existing behavior: no pause."""
    _liveness(monkeypatch, True)
    cases: list[tuple[str, str | None, str, dict]] = [
        ("legacy receipt", None, NEW_EPOCH, {}),
        ("malformed epoch", "not-an-epoch", NEW_EPOCH, {}),
        ("right characters, wrong grouping", "1111111-11111-1111-1111-111111111111:10",
         NEW_EPOCH, {}),
        ("noncanonical 32-hex then 4", "11111111111111111111111111111111-1111:10",
         NEW_EPOCH, {}),
        ("trailing hyphen", "11111111-1111-1111-1111-111111111111-:10", NEW_EPOCH, {}),
        ("unreadable live host", OLD_EPOCH, "", {}),
        ("foreign run claim", OLD_EPOCH, NEW_EPOCH, {"run_claim": "otherhost:5",
                                                    "task_claim": "otherhost:5"}),
        ("conflicting task claim", OLD_EPOCH, NEW_EPOCH, {"task_claim": "otherhost:9"}),
    ]
    for name, recorded, live, mutation in cases:
        with pytest.MonkeyPatch.context() as patch:
            tid = kb.create_task(conn, title=name, assignee="w")
            run_id = _claim(conn, tid, monkeypatch=patch)
            if mutation:
                conn.execute(
                    "UPDATE tasks SET claim_lock = ? WHERE id = ?", (mutation["task_claim"], tid),
                )
                if "run_claim" in mutation:
                    conn.execute(
                        "UPDATE task_runs SET claim_lock = ? WHERE id = ?",
                        (mutation["run_claim"], run_id),
                    )
                conn.commit()
            if recorded is None:
                conn.execute(
                    "UPDATE task_runs SET metadata = json_remove(metadata, '$.host_epoch') "
                    "WHERE id = ?", (run_id,),
                )
                conn.commit()
            elif recorded != OLD_EPOCH:
                _set_recorded_epoch(conn, run_id, recorded)

            patch.setattr(kb, "_current_host_epoch", lambda live=live: live)
            result = _reclaim(conn)

            # Earlier cases in this table stay ``running`` on the same board, so
            # every assertion is scoped to this case's own card.
            assert tid not in result.interrupted, f"{name} must not be exempted"
            row = _row(conn, tid)
            assert row["status"] == "running", f"{name}: card must be left alone"
            assert row["consecutive_failures"] == 0, f"{name}: nothing may be charged"
            assert _run_row(conn, run_id)["ended_at"] is None, f"{name}: run must stay open"
            assert _events(conn, tid, "blocked") == []


def test_same_boot_worker_death_is_still_an_ordinary_crash(conn, monkeypatch):
    """The complement of the rule above: a missing worker on the SAME boot counts."""
    _liveness(monkeypatch, False)
    tid = kb.create_task(conn, title="same boot crash", assignee="w")
    _claim(conn, tid, monkeypatch=monkeypatch)

    result = _reclaim(conn)

    assert (result.interrupted, result.crashed) == ([], [tid])
    row = _row(conn, tid)
    assert (row["status"], row["consecutive_failures"]) == ("ready", 1)
    assert _events(conn, tid, "blocked") == []


@pytest.mark.parametrize("lane", ["ready", "review"])
def test_unblock_restores_the_recorded_phase_and_keeps_pins_and_graph(
    conn, monkeypatch, lane,
):
    _forbid_pid_use(monkeypatch)
    parent = kb.create_task(conn, title="parent", assignee="w")
    kb.complete_task(conn, parent, result="upstream done")
    tid = kb.create_task(conn, title=f"preserve {lane}", assignee="w", parents=[parent])
    child = kb.create_task(conn, title="child", assignee="w", parents=[tid])
    if lane == "review":
        implementation = kb.claim_task(conn, tid, claimer="builder:t")
        assert implementation is not None
        assert kb.request_review(
            conn, tid, summary="ready", reviewer="reviewer",
            expected_run_id=implementation.current_run_id,
        )
    conn.execute(
        "UPDATE tasks SET model_override='m-1', provider_override='p-1', "
        "reasoning_effort='high', branch_name='wt/pinned', current_step_key='impl' "
        "WHERE id = ?", (tid,),
    )
    conn.commit()
    before = _row(conn, tid)
    run_id = _claim(conn, tid, lane=lane, monkeypatch=monkeypatch)

    monkeypatch.setattr(kb, "_current_host_epoch", lambda: NEW_EPOCH)
    assert _reclaim(conn).interrupted == [tid]
    assert _events(conn, tid, "blocked")[0]["retry_status"] == lane
    assert _run_row(conn, run_id)["outcome"] == "interrupted"

    after = _row(conn, tid)
    for field in ("model_override", "provider_override", "reasoning_effort",
                  "branch_name", "current_step_key"):
        assert after[field] == before[field], f"{field} must survive the pause"

    assert kb.unblock_task(conn, tid) is True
    assert _row(conn, tid)["status"] == lane

    parents = conn.execute(
        "SELECT parent_id FROM task_links WHERE child_id = ? ORDER BY parent_id", (tid,),
    ).fetchall()
    assert [row["parent_id"] for row in parents] == [parent]
    assert kb.get_task(conn, child) is not None

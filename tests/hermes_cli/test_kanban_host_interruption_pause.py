"""A replaced host instantiation must PAUSE the card, never kill or charge it.

Issue #43: a reboot (or container restart) leaves ``running`` cards whose worker
is gone but whose PID may already belong to an unrelated process. This file pins
the contracts that break independently:

* the pause itself (reclaim state machine: exactly once, sticky, no charge);
* what does and does not count as proof of a new instantiation;
* what the operator gets back on unblock (phase, pins, graph).

The epoch swap is INJECTED (``kanban_runtime.current_host_epoch`` is monkeypatched); none of
this is a real reboot, which needs a disposable host.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_runtime as _kr

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

    def _terminate(pid, claim_lock, *, signal_fn=None, started_at=None):
        raise AssertionError(f"the stale PID {pid} must never be signalled")

    monkeypatch.setattr(kb, "_pid_alive", _probe)
    monkeypatch.setattr(kb, "_terminate_reclaimed_worker", _terminate)


def _claim(conn, tid: str, *, lane: str = "ready", epoch: str = OLD_EPOCH,
           monkeypatch) -> int:
    """Claim ``tid`` as this host with a recorded epoch; returns the run id."""
    monkeypatch.setattr(_kr, "current_host_epoch", lambda: epoch)
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
        "SELECT status, consecutive_failures, claim_lock, worker_pid, worker_started_at, block_kind, "
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

    monkeypatch.setattr(_kr, "current_host_epoch", lambda: NEW_EPOCH)
    result = _reclaim(conn)

    assert (result.interrupted, result.crashed, result.auto_blocked) == ([tid], [], [])
    row = _row(conn, tid)
    assert row["status"] == "blocked"
    assert row["consecutive_failures"] == 0, "a reboot must not charge a retry"
    assert row["claim_lock"] is None and row["worker_pid"] is None
    assert row["worker_started_at"] is None
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

    # A SECOND interruption must not escalate the card. ``block_kind``/
    # ``block_recurrences`` survive an unblock by design, so routing these
    # through the operator block-loop breaker would reach
    # ``BLOCK_RECURRENCE_LIMIT`` and move the card to ``triage`` — a status
    # ``unblock_task`` cannot release and the auto-decomposer rewrites.
    _claim(conn, tid, monkeypatch=monkeypatch)
    # ``_claim`` re-records the claim-time epoch; make the live host the new one.
    monkeypatch.setattr(_kr, "current_host_epoch", lambda: NEW_EPOCH)
    assert _reclaim(conn).interrupted == [tid]
    row = _row(conn, tid)
    assert row["status"] == "blocked", "a repeat interruption stays operator-recoverable"
    assert row["block_recurrences"] == 0, "an interruption is not an operator block loop"
    assert row["block_kind"] is None
    assert _events(conn, tid, "block_loop_detected") == []
    assert kb.unblock_task(conn, tid) is True
    assert _row(conn, tid)["status"] == "ready"


@pytest.mark.parametrize("terminal_before_restart,current_epoch", [
    (True, NEW_EPOCH),
    (False, NEW_EPOCH),
    (False, ""),
])
def test_terminal_reaper_never_consults_a_previous_hosts_worker(
    conn, monkeypatch, terminal_before_restart, current_epoch,
):
    tid = kb.create_task(conn, title="old host terminal worker", assignee="w")
    run_id = _claim(conn, tid, monkeypatch=monkeypatch)
    if terminal_before_restart:
        assert kb.complete_task(conn, tid, result="done", expected_run_id=run_id)
    monkeypatch.setattr(_kr, "current_host_epoch", lambda: NEW_EPOCH)
    if not terminal_before_restart:
        assert _reclaim(conn).interrupted == [tid]
    conn.execute(
        "UPDATE task_runs SET ended_at = ended_at - ? WHERE id = ?",
        (kbd.TERMINAL_WORKER_REAP_GRACE_SECONDS, run_id),
    )
    conn.commit()
    # Once interruption was proven, a later unreadable epoch cannot make the
    # old PID safe again. Closed runs from before the reboot need the same fence.
    monkeypatch.setattr(_kr, "current_host_epoch", lambda: current_epoch)
    pid_uses = []

    def probe(pid, *_args):
        pid_uses.append(pid)
        return True

    def terminate(pid, *_args, **_kwargs):
        pid_uses.append(pid)
        return {"terminated": True}

    monkeypatch.setattr(kb, "_pid_alive", probe)
    monkeypatch.setattr(kbd, "_worker_alive", probe)
    monkeypatch.setattr(kbd, "_terminate_reclaimed_worker", terminate)
    assert _reclaim(conn).reaped_terminal_workers == []
    assert pid_uses == []
    retained = conn.execute(
        "SELECT worker_pid, worker_started_at FROM task_runs WHERE id = ?", (run_id,),
    ).fetchone()
    assert tuple(retained) == (None, None)
    assert _events(conn, tid, "terminal_worker_reaped") == []


def test_provenance_rule_decides_between_a_pause_and_a_crash(conn, monkeypatch):
    """Positive provenance pauses; every unproven case keeps ordinary behavior."""
    cases = [
        # name, recorded epoch, live epoch, task mutation, worker alive, expected
        ("legacy receipt", None, NEW_EPOCH, {}, True, "untouched"),
        ("malformed epoch", "not-an-epoch", NEW_EPOCH, {}, True, "untouched"),
        ("right characters, wrong grouping",
         "1111111-11111-1111-1111-111111111111:10", NEW_EPOCH, {}, True, "untouched"),
        ("noncanonical 32-hex then 4",
         "11111111111111111111111111111111-1111:10", NEW_EPOCH, {}, True, "untouched"),
        ("trailing hyphen", "11111111-1111-1111-1111-111111111111-:10", NEW_EPOCH, {},
         True, "untouched"),
        ("unreadable live host", OLD_EPOCH, "", {}, True, "untouched"),
        ("foreign run claim", OLD_EPOCH, NEW_EPOCH,
         {"run_claim": "otherhost:5", "task_claim": "otherhost:5"}, True, "untouched"),
        ("conflicting task claim", OLD_EPOCH, NEW_EPOCH, {"task_claim": "otherhost:9"},
         True, "untouched"),
        ("same boot worker death", OLD_EPOCH, OLD_EPOCH, {}, False, "crash"),
    ]
    for name, recorded, live, mutation, alive, expected in cases:
        with pytest.MonkeyPatch.context() as patch:
            _liveness(patch, alive)
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

            patch.setattr(_kr, "current_host_epoch", lambda live=live: live)
            result = _reclaim(conn)

            # Earlier cases in this table share the board, so every assertion is
            # scoped to this case's own card.
            if expected == "crash":
                # Every earlier card is still ``running`` and shares the same
                # liveness answer, so scope to this card.
                assert tid in result.crashed, name
                assert tid not in result.interrupted, name
                assert _row(conn, tid)["consecutive_failures"] == 1, name
                assert _events(conn, tid, "blocked") == []
                continue
            assert tid not in result.interrupted, f"{name} must not be exempted"
            row = _row(conn, tid)
            assert row["status"] == "running", f"{name}: card must be left alone"
            assert row["consecutive_failures"] == 0, f"{name}: nothing may be charged"
            assert _run_row(conn, run_id)["ended_at"] is None, f"{name}: run must stay open"
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

    monkeypatch.setattr(_kr, "current_host_epoch", lambda: NEW_EPOCH)
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

"""A replaced host instantiation must PAUSE the card, never kill or charge it.

Issue #43: a reboot (or container restart) leaves ``running`` cards whose worker
is gone but whose PID may already belong to an unrelated process. The dispatcher
must recognise that from recorded host provenance, end the run as
``interrupted``, and hand the card to the operator through the existing sticky
block + ``unblock_task`` path — without charging or resetting the failure
counter, and without probing or signalling the stale PID.

The epoch swap is INJECTED (``kb._current_host_epoch`` is monkeypatched). None
of this is a real reboot; the real-reboot acceptance lives on a disposable host.
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


def _claim_running(conn, tid: str, *, lane: str = "ready", epoch: str = OLD_EPOCH, monkeypatch) -> int:
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


def _row(conn, tid: str):
    return conn.execute(
        "SELECT status, consecutive_failures, block_kind, block_recurrences, "
        "       model_override, provider_override, reasoning_effort, branch_name, "
        "       current_step_key, claim_lock, worker_pid "
        "FROM tasks WHERE id = ?", (tid,),
    ).fetchone()


def _run_row(conn, run_id: int):
    return conn.execute(
        "SELECT status, outcome, ended_at, claim_lock FROM task_runs WHERE id = ?",
        (run_id,),
    ).fetchone()


def _events(conn, tid: str, kind: str) -> list[dict]:
    rows = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = ? ORDER BY id",
        (tid, kind),
    ).fetchall()
    return [json.loads(row["payload"]) for row in rows if row["payload"]]


def _drop_host_epoch(conn, run_id: int) -> None:
    """Strip the recorded epoch, as an installed-before-this-change run would look."""
    conn.execute(
        "UPDATE task_runs SET metadata = json_remove(metadata, '$.host_epoch') WHERE id = ?",
        (run_id,),
    )
    conn.commit()


def _forbid_pid_use(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail the test if any reclaim path probes or signals a stale worker PID.

    "Do not consult a possibly reused PID" is only proven by making the consult
    fatal, not by stubbing liveness to a plausible answer.
    """
    def _probe(pid):
        raise AssertionError(f"the stale PID {pid} must never be probed")

    def _terminate(pid, claim_lock, *, signal_fn=None):
        raise AssertionError(f"the stale PID {pid} must never be signalled")

    monkeypatch.setattr(kb, "_pid_alive", _probe)
    monkeypatch.setattr(kb, "_terminate_reclaimed_worker", _terminate)


def _reclaim(conn) -> kbd.DispatchResult:
    result = kbd.DispatchResult()
    kbd._run_reclaim_phase(
        conn, result, stale_timeout_seconds=0, failure_limit=3, reconcile_orphans=True,
    )
    return result


def test_changed_boot_pauses_run_with_a_reused_live_pid(conn, monkeypatch):
    """The reboot case: liveness and termination are never consulted at all."""
    _forbid_pid_use(monkeypatch)
    tid = kb.create_task(conn, title="reused pid", assignee="w")
    run_id = _claim_running(conn, tid, monkeypatch=monkeypatch)

    monkeypatch.setattr(kb, "_current_host_epoch", lambda: NEW_EPOCH)
    result = _reclaim(conn)

    assert result.interrupted == [tid]
    assert result.crashed == []
    assert result.auto_blocked == []

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


def test_pause_is_sticky_across_ticks_and_only_unblock_resumes(conn, monkeypatch):
    _forbid_pid_use(monkeypatch)
    tid = kb.create_task(conn, title="sticky pause", assignee="w")
    _claim_running(conn, tid, monkeypatch=monkeypatch)
    monkeypatch.setattr(kb, "_current_host_epoch", lambda: NEW_EPOCH)
    assert _reclaim(conn).interrupted == [tid]

    # Normal dispatcher ticks must leave the card exactly where the operator
    # found it: still blocked, still uncharged, not re-paused.
    for _ in range(3):
        result = kbd.dispatch_once(conn, failure_limit=3)
        assert result.interrupted == []
        assert result.promoted == 0
        assert result.spawned == []
        assert _row(conn, tid)["status"] == "blocked"

    assert kb.unblock_task(conn, tid) is True
    row = _row(conn, tid)
    assert row["status"] == "ready"
    # ``unblock_task``'s established counter semantics are unchanged.
    assert row["consecutive_failures"] == 0
    # Payload is NULL for a plain ready resume; the transition itself is the record.
    unblocked = conn.execute(
        "SELECT COUNT(*) AS n FROM task_events WHERE task_id = ? AND kind = 'unblocked'",
        (tid,),
    ).fetchone()
    assert unblocked["n"] == 1


def test_repeated_reclaim_pauses_an_interruption_only_once(conn, monkeypatch):
    _forbid_pid_use(monkeypatch)
    tid = kb.create_task(conn, title="idempotent pause", assignee="w")
    run_id = _claim_running(conn, tid, monkeypatch=monkeypatch)
    monkeypatch.setattr(kb, "_current_host_epoch", lambda: NEW_EPOCH)

    assert _reclaim(conn).interrupted == [tid]
    assert _reclaim(conn).interrupted == []
    assert _reclaim(conn).interrupted == []
    assert _run_row(conn, run_id)["outcome"] == "interrupted"


def test_unblock_resumes_the_implementation_phase(conn, monkeypatch):
    _forbid_pid_use(monkeypatch)
    tid = kb.create_task(conn, title="resume implementation", assignee="w")
    _claim_running(conn, tid, monkeypatch=monkeypatch)
    monkeypatch.setattr(kb, "_current_host_epoch", lambda: NEW_EPOCH)
    assert _reclaim(conn).interrupted == [tid]

    assert kb.unblock_task(conn, tid) is True
    assert _row(conn, tid)["status"] == "ready"


def test_unblock_resumes_the_review_phase(conn, monkeypatch):
    _forbid_pid_use(monkeypatch)
    tid = kb.create_task(conn, title="resume review", assignee="w")
    implementation = kb.claim_task(conn, tid, claimer="builder:t")
    assert implementation is not None
    assert kb.request_review(
        conn, tid, summary="ready", reviewer="reviewer",
        expected_run_id=implementation.current_run_id,
    )
    _claim_running(conn, tid, lane="review", monkeypatch=monkeypatch)

    monkeypatch.setattr(kb, "_current_host_epoch", lambda: NEW_EPOCH)
    assert _reclaim(conn).interrupted == [tid]
    assert _events(conn, tid, "blocked")[0]["retry_status"] == "review"

    assert kb.unblock_task(conn, tid) is True
    assert _row(conn, tid)["status"] == "review"


def test_pause_preserves_pins_graph_and_step(conn, monkeypatch):
    _forbid_pid_use(monkeypatch)
    parent = kb.create_task(conn, title="parent", assignee="w")
    kb.complete_task(conn, parent, result="upstream done")
    tid = kb.create_task(conn, title="preserve", assignee="w", parents=[parent])
    child = kb.create_task(conn, title="child", assignee="w", parents=[tid])

    conn.execute(
        "UPDATE tasks SET model_override='m-1', provider_override='p-1', "
        "reasoning_effort='high', branch_name='wt/pinned', current_step_key='impl' "
        "WHERE id = ?", (tid,),
    )
    conn.commit()
    before = _row(conn, tid)
    run_id = _claim_running(conn, tid, monkeypatch=monkeypatch)

    monkeypatch.setattr(kb, "_current_host_epoch", lambda: NEW_EPOCH)
    assert _reclaim(conn).interrupted == [tid]

    after = _row(conn, tid)
    for field in ("model_override", "provider_override", "reasoning_effort",
                  "branch_name", "current_step_key"):
        assert after[field] == before[field], f"{field} must survive the pause"
    assert _run_row(conn, run_id)["outcome"] == "interrupted"

    parents = conn.execute(
        "SELECT parent_id FROM task_links WHERE child_id = ? ORDER BY parent_id", (tid,),
    ).fetchall()
    assert [row["parent_id"] for row in parents] == [parent]
    assert kb.get_task(conn, child) is not None


def test_same_boot_worker_death_still_counts_as_a_crash(conn, monkeypatch):
    """A missing worker on the SAME boot is a crash, not an interruption."""
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    tid = kb.create_task(conn, title="same boot crash", assignee="w")
    _claim_running(conn, tid, monkeypatch=monkeypatch)

    result = _reclaim(conn)
    assert result.interrupted == []
    assert result.crashed == [tid]
    row = _row(conn, tid)
    assert row["status"] == "ready"
    assert row["consecutive_failures"] == 1
    assert _events(conn, tid, "blocked") == []


def test_run_without_recorded_host_epoch_is_not_exempted(conn, monkeypatch):
    """Legacy receipts keep their pre-#43 behavior: no provenance, no pause."""
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: True)
    tid = kb.create_task(conn, title="legacy receipt", assignee="w")
    run_id = _claim_running(conn, tid, monkeypatch=monkeypatch)
    _drop_host_epoch(conn, run_id)

    monkeypatch.setattr(kb, "_current_host_epoch", lambda: NEW_EPOCH)
    result = _reclaim(conn)
    assert result.interrupted == []
    assert _row(conn, tid)["status"] == "running"
    assert _run_row(conn, run_id)["ended_at"] is None


def test_malformed_host_epoch_is_not_evidence(conn, monkeypatch):
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: True)
    tid = kb.create_task(conn, title="malformed epoch", assignee="w")
    run_id = _claim_running(conn, tid, monkeypatch=monkeypatch)
    conn.execute(
        "UPDATE task_runs SET metadata = json_set(metadata, '$.host_epoch', 'not-an-epoch') "
        "WHERE id = ?", (run_id,),
    )
    conn.commit()

    monkeypatch.setattr(kb, "_current_host_epoch", lambda: NEW_EPOCH)
    assert _reclaim(conn).interrupted == []
    assert _row(conn, tid)["status"] == "running"


@pytest.mark.parametrize("noncanonical", [
    # Right characters, wrong grouping: still malformed provenance.
    "11111111111111111111111111111111-1111:10",
    "11111111-1111-1111-1111-111111111111-:10",
    "1111111-11111-1111-1111-111111111111:10",
])
def test_noncanonical_boot_id_is_not_evidence(conn, monkeypatch, noncanonical):
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: True)
    tid = kb.create_task(conn, title="noncanonical boot id", assignee="w")
    run_id = _claim_running(conn, tid, monkeypatch=monkeypatch)
    conn.execute(
        "UPDATE task_runs SET metadata = json_set(metadata, '$.host_epoch', ?) WHERE id = ?",
        (noncanonical, run_id),
    )
    conn.commit()

    monkeypatch.setattr(kb, "_current_host_epoch", lambda: NEW_EPOCH)
    assert _reclaim(conn).interrupted == []
    assert _row(conn, tid)["status"] == "running"


def test_unreadable_current_epoch_is_not_evidence(conn, monkeypatch):
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: True)
    tid = kb.create_task(conn, title="unreadable host", assignee="w")
    _claim_running(conn, tid, monkeypatch=monkeypatch)

    monkeypatch.setattr(kb, "_current_host_epoch", lambda: "")
    assert _reclaim(conn).interrupted == []
    assert _row(conn, tid)["status"] == "running"


def test_conflicting_task_claim_is_not_ours_to_pause(conn, monkeypatch):
    """A run claim of ours does not license clearing a DIFFERENT claim.

    The card was re-claimed after this run was opened: the current claim is not
    the one this run recorded, so ownership is not ours to infer.
    """
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: True)
    tid = kb.create_task(conn, title="conflicting claim", assignee="w")
    run_id = _claim_running(conn, tid, monkeypatch=monkeypatch)
    conn.execute("UPDATE tasks SET claim_lock = 'otherhost:9' WHERE id = ?", (tid,))
    conn.commit()

    monkeypatch.setattr(kb, "_current_host_epoch", lambda: NEW_EPOCH)
    assert _reclaim(conn).interrupted == []
    row = _row(conn, tid)
    assert row["status"] == "running"
    assert row["claim_lock"] == "otherhost:9"
    assert _run_row(conn, run_id)["ended_at"] is None


def test_foreign_claim_is_not_ours_to_pause(conn, monkeypatch):
    """Another host's run must never be reinterpreted as this host's work."""
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: True)
    tid = kb.create_task(conn, title="foreign claim", assignee="w")
    run_id = _claim_running(conn, tid, monkeypatch=monkeypatch)
    conn.execute(
        "UPDATE tasks SET claim_lock = 'otherhost:5' WHERE id = ?", (tid,),
    )
    conn.execute(
        "UPDATE task_runs SET claim_lock = 'otherhost:5' WHERE id = ?", (run_id,),
    )
    conn.commit()

    monkeypatch.setattr(kb, "_current_host_epoch", lambda: NEW_EPOCH)
    assert _reclaim(conn).interrupted == []
    assert _row(conn, tid)["status"] == "running"
    assert _run_row(conn, run_id)["ended_at"] is None

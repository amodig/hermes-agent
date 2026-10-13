"""Reaping the worker scope an ended attempt leaves standing (issue #57).

A transient ``systemd-run --scope`` unit stays ACTIVE while anything at all
remains in its cgroup: ``--collect`` only removes it once the cgroup is empty.
A worker-owned descendant that outlives the worker -- the detached
``browser_harness`` daemon is the observed case -- therefore keeps its
``autonomous-workers.slice`` slot after the run is over, and with
``ConcurrencyHardMax=1`` that slot is the sole admission authority, so every
queued card stalls behind it while the board shows no running task.

Stopping a scope kills every process in it, so these tests are mostly about
what must NOT be stopped: a live worker, a closed run whose agent is still
finalising its handoff, a newer run on the same card, a stranger's PID that
reused the recorded number, a scope outside the worker slice, an unreadable
identity, and a suffix this code cannot attribute to a launch. Each of those
must refuse and leave the scope exactly as it is.

The last test drives a REAL transient scope and a REAL stranded child inside a
disposable slice topology, because the gate that decides all of this reads real
cgroup population -- and because the regression is only fixed if the slot
actually comes free for the next worker.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_worker_runtime as kwr
from tools import process_registry as registry

WORKERS_GROUP = "/stub/autonomous.slice/autonomous-workers.slice"
PREPARATION_ID = "430ae778cd154a23a965d49632eda73d"


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    return home


@pytest.fixture
def conn(kanban_home):
    with kbc.connect() as c:
        yield c


def _unit_name(task_id: str, *, preparation_id=None, run_id=None) -> str:
    suffix = (
        f"preparation-{preparation_id}" if preparation_id is not None else f"run-{run_id}"
    )
    return f"hermes-worker-kanban-{task_id}-{suffix}.scope"


def _inventory(*cgroups: str) -> dict:
    """A boundary report holding exactly *cgroups* as live worker scopes."""
    return {
        "workers": {"cgroup": WORKERS_GROUP},
        "worker_scopes": list(cgroups),
        "misplaced_worker_scopes": [],
        "worker_slot_occupied": bool(cgroups),
    }


def _stub_inventory(monkeypatch, *cgroups: str, activation="now") -> list[str]:
    """Install an inventory and a recording stop; return the stopped units.

    ``activation`` is what the manager would report for the scope: ``"now"`` (inside
    the attempt's window), an int epoch, or ``None`` for unreadable evidence.
    """
    stopped: list[str] = []
    monkeypatch.setattr(
        registry, "autonomous_boundary_inventory", lambda: _inventory(*cgroups),
    )
    monkeypatch.setattr(
        registry, "_stop_systemd_unit", lambda unit: stopped.append(unit) or True,
    )
    reported = int(time.time()) if activation == "now" else activation
    monkeypatch.setattr(kwr, "_scope_activation_epoch", lambda unit: reported)
    return stopped


def _insert_run(
    conn,
    task_id: str,
    run_id: int,
    *,
    status: str = "done",
    ended: bool = True,
    worker_pid=None,
    worker_started_at=None,
    metadata=None,
) -> None:
    with kb.write_txn(conn):
        conn.execute(
            "INSERT INTO task_runs (id, task_id, profile, status, started_at, ended_at, "
            "outcome, worker_pid, worker_started_at, metadata) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                task_id,
                "implementer",
                status,
                int(time.time()) - 600,
                int(time.time()) - 300 if ended else None,
                "completed" if ended else None,
                worker_pid,
                worker_started_at,
                kb._json_or_null(metadata or {}),
            ),
        )


def _dead_pid() -> int:
    proc = subprocess.Popen(["true"])
    proc.wait()
    return proc.pid


def _handoff_metadata(*, preparation_id: str, pid: int, fingerprint, host_epoch=None) -> dict:
    """The shape ``_set_worker_pid`` plus a worker handoff leave on the run."""
    return {
        "preparation_id": preparation_id,
        "worker_pid": pid,
        "worker_start_time": 60017552,
        "host_epoch": host_epoch or kwr._current_host_epoch(),
        "runtime_identity": {
            "worker_pid": pid,
            "worker_started_at": fingerprint,
            "preparation_id": preparation_id,
        },
    }


def _reclaim_events(conn, task_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = ?",
        (task_id, kwr.WORKER_SCOPE_RECLAIM_EVENT),
    ).fetchall()
    return [kb._json_dict(row["payload"]) for row in rows]


def test_ended_run_with_a_dead_worker_frees_its_scope(conn, monkeypatch):
    """The regression: the run is over, its worker is gone, its scope is not."""
    task_id = kb.create_task(conn, title="stalled", assignee="implementer")
    unit = _unit_name(task_id, preparation_id=PREPARATION_ID)
    cgroup = f"{WORKERS_GROUP}/{unit}"
    pid = _dead_pid()
    _insert_run(
        conn, task_id, 367, metadata=_handoff_metadata(
            preparation_id=PREPARATION_ID, pid=pid, fingerprint=f"c7ddbf44:13|{pid}",
        ),
    )
    stopped = _stub_inventory(monkeypatch, cgroup)

    events = kwr.reconcile_orphan_worker_scopes(conn)

    assert stopped == [unit], "the exact stranded scope is the only thing stopped"
    assert events == [{
        "unit": unit,
        "cgroup": cgroup,
        "task_id": task_id,
        "run_id": 367,
        "preparation_id": PREPARATION_ID,
        "pid": pid,
        "reason": "run_ended_worker_gone",
        "outcome": "reaped",
        "slot_freed": True,
    }]
    recorded = _reclaim_events(conn, task_id)
    assert [event["unit"] for event in recorded] == [unit]
    assert recorded[0]["outcome"] == "reaped"
    # The bounded event carries no command line, argument or credential field.
    assert set(recorded[0]) == {
        "unit", "cgroup", "task_id", "run_id", "preparation_id", "pid", "reason",
        "outcome", "slot_freed",
    }


def test_ended_run_named_by_run_id_is_reconciled_the_same_way(conn, monkeypatch):
    """The post-claim suffix names the run directly and must reconcile too."""
    task_id = kb.create_task(conn, title="post-claim", assignee="implementer")
    unit = _unit_name(task_id, run_id=367)
    pid = _dead_pid()
    _insert_run(
        conn, task_id, 367,
        worker_pid=pid,
        worker_started_at=f"c7ddbf44:13|1",
    )
    stopped = _stub_inventory(monkeypatch, f"{WORKERS_GROUP}/{unit}")

    events = kwr.reconcile_orphan_worker_scopes(conn)

    assert stopped == [unit]
    assert events[0]["run_id"] == 367
    assert events[0]["outcome"] == "reaped"


def test_live_worker_in_an_ended_run_is_never_stopped(conn, monkeypatch):
    """A card can finish before its agent does: the worker still owns the scope."""
    task_id = kb.create_task(conn, title="finalising", assignee="implementer")
    unit = _unit_name(task_id, preparation_id=PREPARATION_ID)
    worker = subprocess.Popen(["sleep", "30"])
    try:
        _insert_run(
            conn, task_id, 366, metadata=_handoff_metadata(
                preparation_id=PREPARATION_ID, pid=worker.pid,
                fingerprint=kbd._process_fingerprint(worker.pid),
            ),
        )
        stopped = _stub_inventory(monkeypatch, f"{WORKERS_GROUP}/{unit}")

        events = kwr.reconcile_orphan_worker_scopes(conn)

        assert stopped == [], "a live worker's scope is not this module's to stop"
        assert events[0]["reason"] == "worker_alive"
        assert events[0]["outcome"] == "refused"
        assert _reclaim_events(conn, task_id) == []
    finally:
        worker.terminate()
        worker.wait()


def test_live_worker_keeps_its_scope_while_its_run_is_still_open(conn, monkeypatch):
    """A running attempt is refused before identity is even consulted."""
    task_id = kb.create_task(conn, title="running", assignee="implementer")
    unit = _unit_name(task_id, run_id=9)
    _insert_run(conn, task_id, 9, status="running", ended=False)
    stopped = _stub_inventory(monkeypatch, f"{WORKERS_GROUP}/{unit}")

    events = kwr.reconcile_orphan_worker_scopes(conn)

    assert stopped == []
    assert events[0]["reason"] == "run_not_terminal"


def test_newer_live_run_on_the_card_refuses(conn, monkeypatch):
    """An older scope must not be cleared while a newer attempt is in flight."""
    task_id = kb.create_task(conn, title="retried", assignee="implementer")
    unit = _unit_name(task_id, run_id=10)
    _insert_run(conn, task_id, 10, worker_pid=_dead_pid())
    _insert_run(conn, task_id, 11, status="running", ended=False)
    stopped = _stub_inventory(monkeypatch, f"{WORKERS_GROUP}/{unit}")

    events = kwr.reconcile_orphan_worker_scopes(conn)

    assert stopped == []
    assert events[0]["reason"] == "newer_run_live"


def test_reused_pid_does_not_keep_a_finished_workers_scope(conn, monkeypatch):
    """A stranger that inherited the recorded PID is not the worker.

    The stop targets the scope's cgroup, never the PID, so an unrelated process
    holding the recycled number is untouched -- and the finished worker's slot is
    still released instead of waiting for that stranger to exit.
    """
    task_id = kb.create_task(conn, title="recycled", assignee="implementer")
    unit = _unit_name(task_id, preparation_id=PREPARATION_ID)
    stranger = subprocess.Popen(["sleep", "30"])
    try:
        _insert_run(
            conn, task_id, 12, metadata=_handoff_metadata(
                preparation_id=PREPARATION_ID, pid=stranger.pid,
                fingerprint="c7ddbf44:12|1",  # another boot epoch and start time
            ),
        )
        stopped = _stub_inventory(monkeypatch, f"{WORKERS_GROUP}/{unit}")

        events = kwr.reconcile_orphan_worker_scopes(conn)

        assert stopped == [unit]
        assert events[0]["outcome"] == "reaped"
    finally:
        stranger.terminate()
        stranger.wait()


def test_unreadable_worker_identity_refuses(conn, monkeypatch):
    """No recorded identity means no proof the worker exited: fail closed."""
    task_id = kb.create_task(conn, title="anonymous", assignee="implementer")
    unit = _unit_name(task_id, preparation_id=PREPARATION_ID)
    _insert_run(conn, task_id, 13, metadata={"preparation_id": PREPARATION_ID})
    stopped = _stub_inventory(monkeypatch, f"{WORKERS_GROUP}/{unit}")

    events = kwr.reconcile_orphan_worker_scopes(conn)

    assert stopped == []
    assert events[0]["reason"] == "worker_identity_unknown"


def test_scope_outside_the_worker_slice_refuses(conn, monkeypatch):
    """A survivor elsewhere is a placement fault to report, not a slot to clear."""
    task_id = kb.create_task(conn, title="misplaced", assignee="implementer")
    unit = _unit_name(task_id, preparation_id=PREPARATION_ID)
    cgroup = f"/stub/app.slice/{unit}"
    _insert_run(
        conn, task_id, 14, metadata={"preparation_id": PREPARATION_ID, "worker_pid": _dead_pid()},
    )
    stopped = _stub_inventory(monkeypatch, cgroup)

    events = kwr.reconcile_orphan_worker_scopes(conn)

    assert stopped == []
    assert events[0]["reason"] == "scope_not_in_worker_slice"
    assert events[0]["unit"] == unit


@pytest.mark.parametrize("unit", [
    "hermes-worker-cron-job1-exec-e2.scope",
    "hermes-worker-kanban-t_abc-run-missing.scope",
    "hermes-worker-terminal-proc1234.scope",
])
def test_unknown_scope_suffix_is_left_alone(conn, monkeypatch, unit):
    """Only a suffix that names an attempt may be acted on at all."""
    stopped = _stub_inventory(monkeypatch, f"{WORKERS_GROUP}/{unit}")

    events = kwr.reconcile_orphan_worker_scopes(conn)

    assert kwr._kanban_scope_owner(unit) is None
    assert stopped == []
    assert events == []


def test_scope_without_a_recorded_run_refuses(conn, monkeypatch):
    """A preparation id no run record matches proves nothing about an attempt."""
    unit = _unit_name("t_missing", preparation_id=PREPARATION_ID)
    stopped = _stub_inventory(monkeypatch, f"{WORKERS_GROUP}/{unit}")

    events = kwr.reconcile_orphan_worker_scopes(conn)

    assert stopped == []
    assert events[0]["reason"] == "run_not_recorded"


def test_dry_run_reports_without_stopping(conn, monkeypatch):
    """A preview names the scope and its verdict, and touches nothing."""
    task_id = kb.create_task(conn, title="preview", assignee="implementer")
    unit = _unit_name(task_id, preparation_id=PREPARATION_ID)
    _insert_run(
        conn, task_id, 15,
        metadata={"preparation_id": PREPARATION_ID, "worker_pid": _dead_pid()},
    )
    stopped = _stub_inventory(monkeypatch, f"{WORKERS_GROUP}/{unit}")

    events = kwr.reconcile_orphan_worker_scopes(conn, apply=False)

    assert stopped == []
    assert events[0]["outcome"] == "would_reap"
    assert events[0]["slot_freed"] is False
    assert _reclaim_events(conn, task_id) == []


def test_stop_failure_is_reported_and_recorded(conn, monkeypatch):
    """Cleanup refusal must be visible, not a silent retry loop."""
    task_id = kb.create_task(conn, title="unstoppable", assignee="implementer")
    unit = _unit_name(task_id, preparation_id=PREPARATION_ID)
    cgroup = f"{WORKERS_GROUP}/{unit}"
    _insert_run(
        conn, task_id, 16,
        metadata={"preparation_id": PREPARATION_ID, "worker_pid": _dead_pid()},
    )
    _stub_inventory(monkeypatch, cgroup)
    monkeypatch.setattr(registry, "_stop_systemd_unit", lambda unit: False)

    events = kwr.reconcile_orphan_worker_scopes(conn)

    assert events[0]["outcome"] == "failed"
    assert events[0]["slot_freed"] is False
    assert [e["outcome"] for e in _reclaim_events(conn, task_id)] == ["failed"]


def test_a_scope_activated_outside_the_attempt_is_not_stopped(conn, monkeypatch):
    """Another board's live scope can carry the same task/run identity.

    The unit name holds a task and a run/preparation identity but no board, and a
    board import preserves those identities, so a matching name is not proof the
    live scope belongs to the attempt that ended here. An activation long after
    that attempt was over means the unit came from a different launch.
    """
    task_id = kb.create_task(conn, title="not ours", assignee="implementer")
    unit = _unit_name(task_id, run_id=31)
    _insert_run(conn, task_id, 31, worker_pid=_dead_pid())
    stopped = _stub_inventory(monkeypatch, f"{WORKERS_GROUP}/{unit}", activation=int(time.time()) + 86400)

    events = kwr.reconcile_orphan_worker_scopes(conn)

    assert stopped == []
    assert events[0]["reason"] == "scope_not_from_attempt"
    assert events[0]["outcome"] == "refused"


def test_an_unreadable_activation_refuses(conn, monkeypatch):
    """No activation evidence means no proof of ownership: fail closed."""
    task_id = kb.create_task(conn, title="unreadable", assignee="implementer")
    unit = _unit_name(task_id, run_id=32)
    _insert_run(conn, task_id, 32, worker_pid=_dead_pid())
    stopped = _stub_inventory(monkeypatch, f"{WORKERS_GROUP}/{unit}", activation=None)

    events = kwr.reconcile_orphan_worker_scopes(conn)

    assert stopped == []
    assert events[0]["reason"] == "scope_not_from_attempt"


def test_an_attempt_from_a_previous_boot_refuses(conn, monkeypatch):
    """A transient scope cannot survive re-instantiation, so the name match lies."""
    task_id = kb.create_task(conn, title="previous boot", assignee="implementer")
    unit = _unit_name(task_id, preparation_id=PREPARATION_ID)
    _insert_run(
        conn, task_id, 33,
        metadata=_handoff_metadata(
            preparation_id=PREPARATION_ID, pid=_dead_pid(), fingerprint="boot:1|1",
            host_epoch="00000000-0000-0000-0000-000000000000:1",
        ),
    )
    stopped = _stub_inventory(monkeypatch, f"{WORKERS_GROUP}/{unit}")

    events = kwr.reconcile_orphan_worker_scopes(conn)

    assert stopped == []
    assert events[0]["reason"] == "stale_boot"


def test_slot_free_is_only_ever_reported_on_evidence(monkeypatch, tmp_path):
    """A collected or emptied scope reads free; anything else does not."""
    monkeypatch.setattr(kwr, "CGROUP_ROOT", tmp_path)
    group = "/user.slice/worker.slice/hermes-worker-kanban-t_x-run-1.scope"
    assert kwr._scope_slot_free(group) is True, "a collected scope is a free slot"

    directory = tmp_path / group.lstrip("/")
    directory.mkdir(parents=True)
    assert kwr._scope_slot_free(group) is False, "unreadable population is not a free slot"

    (directory / "cgroup.events").write_text("populated 1\nfrozen 0\n", encoding="utf-8")
    assert kwr._scope_slot_free(group) is False, "a populated scope holds the slot"

    (directory / "cgroup.events").write_text("populated 0\nfrozen 0\n", encoding="utf-8")
    assert kwr._scope_slot_free(group) is True, "an emptied scope frees the slot"


def test_unreadable_boundary_is_not_fatal(conn, monkeypatch):
    """No inventory means no evidence: report nothing and stop nothing."""
    def _explode():
        raise RuntimeError("boundary unreadable")

    monkeypatch.setattr(registry, "autonomous_boundary_inventory", _explode)

    assert kwr.reconcile_orphan_worker_scopes(conn) == []


def test_repeated_reconciliation_after_a_reap_is_idempotent(conn, monkeypatch):
    """A scope already gone is not reported or stopped again."""
    task_id = kb.create_task(conn, title="once", assignee="implementer")
    unit = _unit_name(task_id, preparation_id=PREPARATION_ID)
    cgroup = f"{WORKERS_GROUP}/{unit}"
    _insert_run(
        conn, task_id, 17,
        metadata={"preparation_id": PREPARATION_ID, "worker_pid": _dead_pid()},
    )
    stopped = _stub_inventory(monkeypatch, cgroup)
    assert kwr.reconcile_orphan_worker_scopes(conn)[0]["outcome"] == "reaped"

    # The stop emptied the cgroup, so ``--collect`` removed the unit: the next
    # tick's inventory no longer reports it and nothing is stopped twice.
    stopped.clear()
    monkeypatch.setattr(registry, "autonomous_boundary_inventory", lambda: _inventory())
    assert kwr.reconcile_orphan_worker_scopes(conn) == []
    assert stopped == []
    assert len(_reclaim_events(conn, task_id)) == 1


def test_a_run_closed_earlier_in_the_same_tick_still_frees_its_scope(conn, monkeypatch):
    """Reconciliation must run after the sweeps that can close a run.

    A worker that dies while its run is still open leaves a populated scope; the
    run only becomes provably finished when a later sweep in the same reclaim phase
    ends it. Reconciling before those sweeps would refuse ``run_not_terminal`` and
    defer recovery to the next tick, so this asserts the same-call ordering.
    """
    task_id = kb.create_task(conn, title="closed this tick", assignee="implementer")
    unit = _unit_name(task_id, preparation_id=PREPARATION_ID)
    _insert_run(
        conn, task_id, 34, status="running", ended=False,
        metadata=_handoff_metadata(
            preparation_id=PREPARATION_ID, pid=_dead_pid(), fingerprint="boot:1|1",
        ),
    )
    stopped = _stub_inventory(monkeypatch, f"{WORKERS_GROUP}/{unit}")

    def _close_the_run(conn, **_kwargs):
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE task_runs SET status = 'timed_out', outcome = 'timed_out', ended_at = ? "
                "WHERE id = 34", (int(time.time()),),
            )
        return []

    monkeypatch.setattr(kbd, "detect_stale_running", _close_the_run)
    result = kbd.DispatchResult()

    kbd._run_reclaim_phase(
        conn, result, stale_timeout_seconds=1, failure_limit=3, reconcile_orphans=True,
    )

    assert stopped == [unit], "the tick that closed the run must free the slot itself"
    assert [event["outcome"] for event in result.reconciled_worker_scopes] == ["reaped"]


def test_the_reclaim_phase_reconciles_scopes_before_capacity_is_measured(conn, monkeypatch):
    """The dispatch tick must free the slot it is about to measure."""
    task_id = kb.create_task(conn, title="tick", assignee="implementer")
    unit = _unit_name(task_id, preparation_id=PREPARATION_ID)
    _insert_run(
        conn, task_id, 18,
        metadata={"preparation_id": PREPARATION_ID, "worker_pid": _dead_pid()},
    )
    stopped = _stub_inventory(monkeypatch, f"{WORKERS_GROUP}/{unit}")
    result = kbd.DispatchResult()

    kbd._run_reclaim_phase(
        conn, result, stale_timeout_seconds=0, failure_limit=3, reconcile_orphans=True,
    )

    assert stopped == [unit]
    assert [event["unit"] for event in result.reconciled_worker_scopes] == [unit]


def test_a_dry_run_tick_previews_instead_of_stopping(conn, monkeypatch):
    """``dry_run`` must stay non-destructive for this host-level action too."""
    task_id = kb.create_task(conn, title="dry", assignee="implementer")
    unit = _unit_name(task_id, preparation_id=PREPARATION_ID)
    _insert_run(
        conn, task_id, 19,
        metadata={"preparation_id": PREPARATION_ID, "worker_pid": _dead_pid()},
    )
    stopped = _stub_inventory(monkeypatch, f"{WORKERS_GROUP}/{unit}")
    result = kbd.DispatchResult()

    kbd._run_reclaim_phase(
        conn, result, stale_timeout_seconds=0, failure_limit=3, reconcile_orphans=True,
        dry_run=True,
    )

    assert stopped == []
    assert [event["outcome"] for event in result.reconciled_worker_scopes] == ["would_reap"]


@pytest.mark.platforms("linux")
@pytest.mark.live_system_guard_bypass  # creates and stops disposable transient scopes
def test_a_stranded_child_blocks_the_next_worker_until_it_is_reconciled(conn, monkeypatch):
    """The real regression: a real stranded child holds a real slot.

    Driven through real transient scopes in a disposable slice topology, with the
    production names and the production stop primitive: a worker-style scope whose
    payload exits and leaves a detached descendant holds the fixture slice's
    single ``ConcurrencyHardMax`` slot -- a second worker cannot start beside it --
    and the reconciliation releases the slot so the next worker really does start.
    """
    from tests._fixtures import autonomous_slice as fixture

    fixture.require_user_bus()
    if _host_worker_scopes():
        # The fixture validator walks the whole user manager, so a worker scope of
        # the REAL boundary reads as "outside this fixture's worker slice". State
        # that precondition instead of asserting against a busy host.
        pytest.skip("a worker scope is live in the real worker slice; the fixture cannot be validated beside it")
    task_id = kb.create_task(conn, title="stranded", assignee="implementer")
    unit = _unit_name(task_id, preparation_id=PREPARATION_ID)
    bus_env = fixture.user_bus_env()

    with fixture.disposable_boundary() as boundary:
        boundary.install(monkeypatch)
        # The test runner blanks the environment, and the boundary readers query
        # the user manager as the runtime's own processes do: hand them the bus.
        for name in ("XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS"):
            if bus_env.get(name):
                monkeypatch.setenv(name, bus_env[name])
        # The autouse stub replaces the host verdict for hermetic tests; this one
        # needs the REAL one against the fixture topology.
        monkeypatch.setattr(
            registry, "autonomous_boundary_inventory",
            lambda: registry.check_autonomous_boundary(),
        )
        _insert_run(
            conn, task_id, 20,
            metadata=_handoff_metadata(
                preparation_id=PREPARATION_ID, pid=_dead_pid(),
                fingerprint="c7ddbf44:13|1",
            ),
        )
        _start_scope(boundary, unit, bus_env)
        stranded = _wait_for_scope(boundary, unit, populated=True)

        assert registry.check_autonomous_boundary()["worker_slot_occupied"]
        assert not _slot_free(boundary, unit), "the stranded child holds the slot"
        # Fails before the fix: nothing else may start in the slice's single slot.
        assert not _second_worker_starts(boundary, bus_env)

        events = kwr.reconcile_orphan_worker_scopes(conn)

        assert [event["outcome"] for event in events] == ["reaped"], events
        assert events[0]["slot_freed"] is True
        assert stranded.poll() is not None, "the stranded descendant died with its scope"
        assert not registry.check_autonomous_boundary()["worker_slot_occupied"]
        assert _slot_free(boundary, unit)
        # Passes after the fix: the next worker actually claims and starts.
        assert _second_worker_starts(boundary, bus_env)


def _host_worker_scopes() -> list[str]:
    """Live worker scopes on the REAL host boundary, if this host has one."""
    from tools import autonomous_resources as resources

    completed = subprocess.run(
        ["systemctl", "--user", "show", "autonomous-workers.slice",
         "--property=ControlGroup", "--value"],
        capture_output=True, text=True, timeout=30, env=_bus_env(), stdin=subprocess.DEVNULL,
    )
    group = completed.stdout.strip()
    if completed.returncode != 0 or not group.startswith("/"):
        return []
    return resources.live_worker_scope_cgroups(
        resources.CGROUP_ROOT, resources.user_manager_root(group),
    )


def _bus_env():
    from tests._fixtures import autonomous_slice as fixture

    return fixture.user_bus_env()


def _start_scope(boundary, unit: str, bus_env) -> subprocess.Popen:
    """Start a worker-style scope whose payload exits leaving a detached child."""
    return subprocess.Popen(
        [
            "systemd-run", "--user", "--scope", "--quiet", "--collect",
            f"--slice={boundary.workers_unit}", "--unit", unit,
            "--", "bash", "-c", "setsid sleep 300 >/dev/null 2>&1 & exit 0",
        ],
        env=bus_env, start_new_session=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def _wait_for_scope(boundary, unit: str, *, populated: bool, timeout: float = 20.0):
    """Wait until *unit* exists with the expected cgroup population."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        group = _scope_group(boundary, unit)
        if group is not None and (_scope_populated(group) is populated):
            return _stranded_child(group)
        time.sleep(0.1)
    raise AssertionError(f"{unit}: cgroup never reached populated={populated}")


def _scope_group(boundary, unit: str):
    """The live cgroup of *unit* inside the fixture worker slice, if any."""
    directory = Path("/sys/fs/cgroup") / boundary.workers_group.lstrip("/")
    candidate = directory / unit
    return f"{boundary.workers_group}/{unit}" if candidate.is_dir() else None


def _scope_populated(group: str) -> bool:
    for line in (Path("/sys/fs/cgroup") / group.lstrip("/") / "cgroup.events").read_text(
        encoding="utf-8-sig"
    ).splitlines():
        key, _, value = line.partition(" ")
        if key == "populated":
            return value.strip() == "1"
    raise AssertionError(f"{group}: no population evidence")


def _stranded_child(group: str) -> subprocess.Popen:
    """A handle on the leftover process inside *group*, so its death is observable."""
    procs = (Path("/sys/fs/cgroup") / group.lstrip("/") / "cgroup.procs").read_text(
        encoding="utf-8-sig"
    ).split()
    assert procs, f"{group} holds no process"
    return _process_object(int(procs[0]))


def _process_object(pid: int):
    """A ``poll()``-able view of an unrelated live process (it is not our child)."""
    class _Observed:
        def __init__(self, pid: int) -> None:
            self.pid = pid

        def poll(self):
            return None if os.path.exists(f"/proc/{self.pid}") else 0

    return _Observed(pid)


def _slot_free(boundary, unit: str) -> bool:
    """Whether the worker slice's single slot is free again.

    The unit's own cgroup is the authority; ``--collect`` removes it once empty,
    exactly as in production.
    """
    group = _scope_group(boundary, unit)
    return True if group is None else not _scope_populated(group)


def _second_worker_starts(boundary, bus_env) -> bool:
    """Whether the slice admits another worker scope right now."""
    unit = "hermes-worker-kanban-t_next-run-21.scope"
    proc = subprocess.Popen(
        [
            "systemd-run", "--user", "--scope", "--quiet", "--collect",
            f"--slice={boundary.workers_unit}", "--unit", unit,
            "--", "sleep", "60",
        ],
        env=bus_env, start_new_session=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            group = _scope_group(boundary, unit)
            if group is not None and _scope_populated(group):
                return True
            if proc.poll() is not None:
                return False
            time.sleep(0.1)
        return False
    finally:
        subprocess.run(
            ["systemctl", "--user", "stop", unit], env=bus_env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30,
        )
        if proc.poll() is None:
            proc.wait(timeout=30)

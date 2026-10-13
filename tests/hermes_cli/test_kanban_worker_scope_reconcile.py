"""Reaping the worker scope an ended attempt leaves standing (issue #57).

Two invariants, per the repository's 1–2 tests-per-fix rule:

* ``test_scope_is_reaped_only_when_its_own_attempt_ended_and_its_worker_is_gone``
  pins the decision: a scope is stopped exactly when this board recorded that
  attempt, the attempt ended, and the worker identity recorded at spawn is
  provably gone — and in every other case it is left exactly as found and the
  refusal is reported.
* ``test_a_stranded_child_blocks_the_next_worker_until_it_is_reconciled`` pins the
  behaviour on REAL transient scopes: a stranded descendant of a finished attempt
  holds the slice's single slot, a second worker cannot start beside it, the
  reconciliation releases the slot, and the next worker really does start — while
  a live scope is refused and left running.

Why a scope at all: a transient ``systemd-run --scope`` unit stays ACTIVE while
anything at all remains in its cgroup (``--collect`` only removes it once the
cgroup is empty), and ``ConcurrencyHardMax=1`` on ``autonomous-workers.slice`` is
the sole admission authority. A worker-owned descendant that outlives the worker
kept the only slot after the run was over, so every queued card stalled while the
board showed no running task.
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
    return {
        "workers": {"cgroup": WORKERS_GROUP},
        "worker_scopes": list(cgroups),
        "misplaced_worker_scopes": [],
        "worker_slot_occupied": bool(cgroups),
    }


def _stub(monkeypatch, *cgroups: str, activation="now", stops=True) -> list[str]:
    """Install a boundary inventory, a recording stop, and the activation verdict.

    ``activation`` is what the manager would report: ``"now"`` (inside the
    attempt's window), an int epoch, or ``None`` for unreadable evidence.
    """
    stopped: list[str] = []
    monkeypatch.setattr(registry, "autonomous_boundary_inventory", lambda: _inventory(*cgroups))
    monkeypatch.setattr(
        registry, "_stop_systemd_unit", lambda unit: stopped.append(unit) or stops,
    )
    reported = int(time.time()) if activation == "now" else activation
    monkeypatch.setattr(kwr, "_scope_activation_epoch", lambda unit: reported)
    return stopped


def _insert_run(conn, task_id, run_id, **kw) -> None:
    """One remote-style ``task_runs`` row for a live scope to resolve against."""
    with kb.write_txn(conn):
        conn.execute(
            "INSERT INTO task_runs (id, task_id, profile, status, started_at, ended_at, "
            "outcome, worker_pid, worker_started_at, metadata) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                task_id,
                "implementer",
                kw.get("status", "done"),
                kw.get("started_at", int(time.time()) - 600),
                kw.get("ended_at", int(time.time()) - 300) if kw.get("ended", True) else None,
                "completed",
                kw.get("worker_pid"),
                kw.get("worker_started_at"),
                kb._json_or_null(kw.get("metadata") or {}),
            ),
        )


def _dead_pid() -> int:
    proc = subprocess.Popen(["true"])
    proc.wait()
    return proc.pid


def _metadata(*, preparation_id=PREPARATION_ID, pid, fingerprint, board=None, host_epoch=None) -> dict:
    """The shape ``_set_worker_pid`` plus a worker handoff leave on the run."""
    payload = {
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
    if board:
        payload["board"] = board
    return payload


def _reclaim_events(conn, task_id) -> list[dict]:
    return [
        kb._json_dict(row["payload"])
        for row in conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? AND kind = ?",
            (task_id, kwr.WORKER_SCOPE_RECLAIM_EVENT),
        )
    ]


def test_scope_is_reaped_only_when_its_own_attempt_ended_and_its_worker_is_gone(conn, monkeypatch):
    """The decision table: one exact reap, and every refusal fail-closed."""
    # --- reaped: the ended attempt's scope, by preparation id and by run id ---
    for index, form in enumerate(("preparation", "run")):
        run_id = 367 + index
        task_id = kb.create_task(conn, title=f"stranded {form}", assignee="implementer")
        unit = (
            _unit_name(task_id, preparation_id=PREPARATION_ID)
            if form == "preparation"
            else _unit_name(task_id, run_id=run_id)
        )
        cgroup = f"{WORKERS_GROUP}/{unit}"
        pid = _dead_pid()
        _insert_run(
            conn, task_id, run_id,
            metadata=_metadata(pid=pid, fingerprint=f"boot:1|{pid}") if form == "preparation" else None,
            worker_pid=None if form == "preparation" else pid,
            worker_started_at=None if form == "preparation" else "boot:1|1",
        )
        stopped = _stub(monkeypatch, cgroup)

        events = kwr.reconcile_orphan_worker_scopes(conn)

        assert stopped == [unit], form
        assert events[0]["reason"] == "run_ended_worker_gone"
        assert events[0]["outcome"] == "reaped"
        assert events[0]["slot_freed"] is True
        recorded = _reclaim_events(conn, task_id)
        assert recorded[0]["unit"] == unit
        # Bounded and sanitized: no command line, argument or credential field.
        assert set(recorded[0]) <= {
            "unit", "cgroup", "task_id", "run_id", "preparation_id", "pid", "board",
            "reason", "outcome", "slot_freed",
        }

    # --- refused: everything the fix must never stop ---
    alive = subprocess.Popen(["sleep", "60"])
    try:
        refusals = [
            ("run_not_terminal", dict(
                status="running", ended=False,
                metadata=_metadata(pid=_dead_pid(), fingerprint="boot:1|1")), "preparation", None, "now"),
            ("worker_alive", dict(metadata=_metadata(
                pid=alive.pid, fingerprint=kbd._process_fingerprint(alive.pid))), "preparation", None, "now"),
            ("newer_run_live", dict(metadata=_metadata(pid=_dead_pid(), fingerprint="boot:1|1")), "preparation", "newer", "now"),
            ("worker_identity_unknown", dict(metadata={"preparation_id": PREPARATION_ID}), "preparation", None, "now"),
            ("run_not_recorded", None, "preparation", None, "now"),
            ("scope_not_from_attempt", dict(worker_pid=_dead_pid()), "run", None, "late"),
            ("scope_not_from_attempt", dict(worker_pid=_dead_pid()), "run", None, None),
            ("stale_boot", dict(metadata=_metadata(
                pid=_dead_pid(), fingerprint="boot:1|1",
                host_epoch="00000000-0000-0000-0000-000000000000:1")), "preparation", None, "now"),
            ("foreign_board", dict(metadata=_metadata(
                pid=_dead_pid(), fingerprint="boot:1|1", board="another-board")), "preparation", None, "now"),
        ]
        for index, (reason, run_kw, form, extra_run, activation) in enumerate(refusals):
            run_id = 200 + index * 10
            task_id = kb.create_task(conn, title=f"refusal {index}", assignee="implementer")
            unit = (
                _unit_name(task_id, preparation_id=PREPARATION_ID)
                if form == "preparation"
                else _unit_name(task_id, run_id=run_id)
            )
            cgroup = f"{WORKERS_GROUP}/{unit}"
            if run_kw is not None:
                _insert_run(conn, task_id, run_id, **(run_kw or {}))
            if extra_run == "newer":
                _insert_run(conn, task_id, run_id + 1, status="running", ended=False)
            activation_value = (
                int(time.time()) + 86400 if activation == "late" else activation
            )
            stopped = _stub(monkeypatch, cgroup, activation=activation_value)

            events = kwr.reconcile_orphan_worker_scopes(conn)

            assert stopped == [], f"{reason}: a scope was stopped"
            assert events and events[0]["reason"] == reason, f"{reason}: {events}"
            assert events[0]["outcome"] == "refused"
            assert _reclaim_events(conn, task_id) == [], f"{reason}: refusal was recorded"

        # A live worker's own card also protects its scope from a terminal run row.
        task_id = kb.create_task(conn, title="card holds a live worker", assignee="implementer")
        unit = _unit_name(task_id, preparation_id=PREPARATION_ID)
        _insert_run(conn, task_id, 101, metadata=_metadata(pid=_dead_pid(), fingerprint="boot:1|1"))
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET worker_pid = ?, worker_started_at = ? WHERE id = ?",
                (alive.pid, kbd._process_fingerprint(alive.pid), task_id),
            )
        stopped = _stub(monkeypatch, f"{WORKERS_GROUP}/{unit}")
        assert kwr.reconcile_orphan_worker_scopes(conn)[0]["reason"] == "worker_alive"
        assert stopped == []
    finally:
        alive.terminate()
        alive.wait()

    # --- a misplaced survivor, an unknown suffix and a live scope are not touched ---
    for unit in (
        "hermes-worker-cron-job1-exec-e2.scope",
        "hermes-worker-kanban-t_abc-run-missing.scope",
        "hermes-worker-terminal-proc1234.scope",
    ):
        assert kwr._kanban_scope_owner(unit) is None
        stopped = _stub(monkeypatch, f"{WORKERS_GROUP}/{unit}")
        assert kwr.reconcile_orphan_worker_scopes(conn) == []
        assert stopped == []

    task_id = kb.create_task(conn, title="misplaced", assignee="implementer")
    unit = _unit_name(task_id, preparation_id=PREPARATION_ID)
    _insert_run(conn, task_id, 102, metadata=_metadata(pid=_dead_pid(), fingerprint="boot:1|1"))
    stopped = _stub(monkeypatch, f"/stub/app.slice/{unit}")
    assert kwr.reconcile_orphan_worker_scopes(conn)[0]["reason"] == "scope_not_in_worker_slice"
    assert stopped == []

    # --- a dry run and an unreadable boundary change nothing ---
    task_id = kb.create_task(conn, title="preview", assignee="implementer")
    _insert_run(conn, task_id, 103, metadata=_metadata(pid=_dead_pid(), fingerprint="boot:1|1"))
    unit = _unit_name(task_id, preparation_id=PREPARATION_ID)
    stopped = _stub(monkeypatch, f"{WORKERS_GROUP}/{unit}")
    events = kwr.reconcile_orphan_worker_scopes(conn, apply=False)
    assert [event["outcome"] for event in events] == ["would_reap"]
    assert stopped == [] and _reclaim_events(conn, task_id) == []

    monkeypatch.setattr(
        registry, "autonomous_boundary_inventory",
        lambda: (_ for _ in ()).throw(RuntimeError("boundary unreadable")),
    )
    assert kwr.reconcile_orphan_worker_scopes(conn) == []

    # --- a failed stop is reported and recorded, and is not retried blindly ---
    task_id = kb.create_task(conn, title="unstoppable", assignee="implementer")
    unit = _unit_name(task_id, preparation_id=PREPARATION_ID)
    _insert_run(conn, task_id, 104, metadata=_metadata(pid=_dead_pid(), fingerprint="boot:1|1"))
    stopped = _stub(monkeypatch, f"{WORKERS_GROUP}/{unit}", stops=False)
    events = kwr.reconcile_orphan_worker_scopes(conn)
    assert stopped == [unit] and events[0]["outcome"] == "failed" and events[0]["slot_freed"] is False
    assert [event["outcome"] for event in _reclaim_events(conn, task_id)] == ["failed"]

    # --- the slot is only called free on evidence ---
    monkeypatch.setattr(kwr, "CGROUP_ROOT", Path("/nonexistent-cgroup-root"))
    assert kwr._scope_slot_free("/stub/scope") is True, "a collected scope is a free slot"

    # --- the tick that closes a run reconciles in the same call ---
    # A worker that dies with its run still open leaves a populated scope; a later
    # sweep in the same reclaim phase ends that run. Reconciling first would refuse
    # ``run_not_terminal`` and defer recovery to the next tick.
    task_id = kb.create_task(conn, title="closed this tick", assignee="implementer")
    unit = _unit_name(task_id, preparation_id=PREPARATION_ID)
    _insert_run(
        conn, task_id, 105, status="running", ended=False,
        metadata=_metadata(pid=_dead_pid(), fingerprint="boot:1|1"),
    )
    stopped = _stub(monkeypatch, f"{WORKERS_GROUP}/{unit}")

    def _close_the_run(conn, **_kwargs):
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE task_runs SET status = 'timed_out', outcome = 'timed_out', ended_at = ? "
                "WHERE id = 105", (int(time.time()),),
            )
        return []

    monkeypatch.setattr(kbd, "detect_stale_running", _close_the_run)
    result = kbd.DispatchResult()
    kbd._run_reclaim_phase(
        conn, result, stale_timeout_seconds=1, failure_limit=3, reconcile_orphans=True,
    )
    assert stopped == [unit]
    assert [event["outcome"] for event in result.reconciled_worker_scopes] == ["reaped"]


@pytest.mark.platforms("linux")
@pytest.mark.live_system_guard_bypass  # creates and stops disposable transient scopes
def test_a_stranded_child_blocks_the_next_worker_until_it_is_reconciled(conn, monkeypatch):
    """The real regression, on real transient scopes with the production stop.

    A worker-style scope whose payload exits leaving a detached descendant holds
    the fixture slice's single ``ConcurrencyHardMax`` slot: a second worker cannot
    start beside it. The reconciliation reaps it, the descendant dies with its
    cgroup, the next worker really starts, and a LIVE scope is refused and left
    untouched.
    """
    from tests._fixtures import autonomous_slice as fixture

    fixture.require_user_bus()
    if _host_worker_scopes():
        pytest.skip("a worker scope is live in the real worker slice; the fixture cannot be validated beside it")
    task_id = kb.create_task(conn, title="stranded", assignee="implementer")
    unit = _unit_name(task_id, preparation_id=PREPARATION_ID)
    bus_env = fixture.user_bus_env()

    with fixture.disposable_boundary() as boundary:
        boundary.install(monkeypatch)
        # The runner blanks the environment, and the boundary readers query the user
        # manager as the runtime's own processes do: hand them the bus.
        for name in ("XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS"):
            if bus_env.get(name):
                monkeypatch.setenv(name, bus_env[name])
        # A stub replaces the host verdict for hermetic tests; this one needs the REAL
        # one, against the fixture topology.
        monkeypatch.setattr(
            registry, "autonomous_boundary_inventory",
            lambda: registry.check_autonomous_boundary(),
        )
        _insert_run(conn, task_id, 20, metadata=_metadata(pid=_dead_pid(), fingerprint="boot:1|1"))
        _start_scope(boundary, unit, bus_env)
        stranded = _wait_for_scope(boundary, unit)

        assert registry.check_autonomous_boundary()["worker_slot_occupied"]
        assert not _slot_free(boundary, unit), "the stranded child holds the slot"
        assert not _second_worker_starts(boundary, bus_env), "nothing may start in the held slot"

        events = kwr.reconcile_orphan_worker_scopes(conn)

        assert [event["outcome"] for event in events] == ["reaped"], events
        assert events[0]["slot_freed"] is True
        assert stranded.poll() is not None, "the stranded descendant died with its scope"
        assert _slot_free(boundary, unit)
        assert _second_worker_starts(boundary, bus_env, keep=True), "the next worker must actually start"

        # The worker that just started is LIVE: it is refused, not stopped.
        live = "hermes-worker-kanban-t_next-run-21.scope"
        assert _scope_group(boundary, live) is not None
        assert [event["reason"] for event in kwr.reconcile_orphan_worker_scopes(conn)] == ["run_not_recorded"]
        assert _scope_group(boundary, live) is not None, "a live scope stays untouched"


def _host_worker_scopes() -> list[str]:
    """Live worker scopes on the REAL host boundary, if this host has one."""
    from tools import autonomous_resources as resources
    from tests._fixtures import autonomous_slice as fixture

    completed = subprocess.run(
        ["systemctl", "--user", "show", "autonomous-workers.slice",
         "--property=ControlGroup", "--value"],
        capture_output=True, text=True, timeout=30, env=fixture.user_bus_env(),
        stdin=subprocess.DEVNULL,
    )
    group = completed.stdout.strip()
    if completed.returncode != 0 or not group.startswith("/"):
        return []
    return resources.live_worker_scope_cgroups(
        resources.CGROUP_ROOT, resources.user_manager_root(group),
    )


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


def _wait_for_scope(boundary, unit: str, timeout: float = 20.0):
    """Wait for *unit* to exist and report its leftover process."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        group = _scope_group(boundary, unit)
        if group is not None and _scope_populated(group):
            procs = (Path("/sys/fs/cgroup") / group.lstrip("/") / "cgroup.procs").read_text(
                encoding="utf-8-sig"
            ).split()
            assert procs, f"{group} holds no process"
            pid = int(procs[0])

            class _Observed:
                """``poll()``-able view of an unrelated process (it is not our child)."""

                def poll(self):
                    return None if os.path.exists(f"/proc/{pid}") else 0

            return _Observed()
        time.sleep(0.1)
    raise AssertionError(f"{unit}: cgroup never became populated")


def _scope_group(boundary, unit: str):
    """The live cgroup of *unit* inside the fixture worker slice, if any."""
    candidate = Path("/sys/fs/cgroup") / boundary.workers_group.lstrip("/") / unit
    return f"{boundary.workers_group}/{unit}" if candidate.is_dir() else None


def _scope_populated(group: str) -> bool:
    for line in (
        Path("/sys/fs/cgroup") / group.lstrip("/") / "cgroup.events"
    ).read_text(encoding="utf-8-sig").splitlines():
        key, _, value = line.partition(" ")
        if key == "populated":
            return value.strip() == "1"
    raise AssertionError(f"{group}: no population evidence")


def _slot_free(boundary, unit: str) -> bool:
    """Whether the slice's single slot is free: the unit's own cgroup decides."""
    group = _scope_group(boundary, unit)
    return True if group is None else not _scope_populated(group)


def _second_worker_starts(boundary, bus_env, *, keep: bool = False) -> bool:
    """Whether the slice admits another worker scope right now.

    ``keep=True`` leaves the started scope running (the fixture teardown still stops
    it), so a caller can assert what happens to a LIVE scope.
    """
    unit = "hermes-worker-kanban-t_next-run-21.scope"
    proc = subprocess.Popen(
        [
            "systemd-run", "--user", "--scope", "--quiet", "--collect",
            f"--slice={boundary.workers_unit}", "--unit", unit, "--", "sleep", "60",
        ],
        env=bus_env, start_new_session=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + 20
    started = False
    while time.monotonic() < deadline:
        group = _scope_group(boundary, unit)
        if group is not None and _scope_populated(group):
            started = True
            break
        if proc.poll() is not None:
            break
        time.sleep(0.1)
    if started and keep:
        return True
    if proc.poll() is None:
        subprocess.run(
            ["systemctl", "--user", "stop", unit], env=bus_env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30,
        )
        proc.wait(timeout=30)
    return started

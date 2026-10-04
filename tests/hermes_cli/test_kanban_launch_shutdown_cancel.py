"""A gateway that is going away must not grant Kanban work it will not outlive.

Issue #43: the embedded dispatcher kept launching workers straight through the
shutdown window. These are the contracts that break independently:

* the launch barriers (nothing is granted once a stop is observed);
* the grant boundary (a granted worker is never cancelled, whatever breaks after);
* the accounting boundary (only PROVEN shutdown excuses a launch failure).

``should_stop`` is driven by an explicit switch, and the accounting boundary is
exercised through the real dispatch path with a stand-in for the native spawn.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli.kanban_runtime import RuntimeIdentityError, prospective_identity


class _StopSwitch:
    """Cancellation predicate whose threshold the test moves explicitly."""

    def __init__(self, allow: int = 10**6) -> None:
        self.calls = 0
        self._allow = allow

    def arm(self, allow: int = 0) -> None:
        """Report "stopped" after ``allow`` further polls."""
        self._allow = self.calls + allow

    def __call__(self) -> bool:
        self.calls += 1
        return self.calls > self._allow


class _LaunchRecorder:
    """Stand-in for the native spawn that records grant/cancel side effects."""

    def __init__(self, switch: _StopSwitch) -> None:
        self.switch = switch
        self.spawn_calls = 0
        self.grants: list[tuple[int, str | None]] = []
        self.cancels = 0
        self.arm_on_spawn: int | None = None
        self.raise_on_spawn: BaseException | None = None

    def __call__(self, task, workspace, *, board=None, defer_grant=False, should_stop=None):
        self.spawn_calls += 1
        if self.arm_on_spawn is not None:
            # Arm BEFORE any raise, so a test can model "the drain began as this
            # launch was already failing".
            self.switch.arm(self.arm_on_spawn)
        if self.raise_on_spawn is not None:
            raise self.raise_on_spawn
        identity = prospective_identity()
        return kbd.WorkerLaunch(
            pid=identity.pid,
            runtime_identity=identity.as_dict(),
            preparation_id="prep-test",
            grant=lambda run_id, claim_lock: self.grants.append((run_id, claim_lock)),
            cancel=self._cancel,
        )

    def _cancel(self) -> None:
        self.cancels += 1


@pytest.fixture
def conn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    home = tmp_path / ".hermes"
    (home / "profiles" / "w").mkdir(parents=True)
    (home / "profiles" / "w" / "config.yaml").write_text("{}\n", encoding="utf-8")
    (home / "config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    # No real user manager is consulted unless a test asks for it.
    monkeypatch.setattr(kbd, "_user_manager_stopping", lambda: False)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    with kbc.connect() as open_conn:
        yield open_conn


@pytest.fixture
def switch() -> _StopSwitch:
    return _StopSwitch()


@pytest.fixture
def recorder(switch: _StopSwitch, monkeypatch: pytest.MonkeyPatch) -> _LaunchRecorder:
    stub = _LaunchRecorder(switch)
    monkeypatch.setattr(kbd, "_default_spawn", stub)
    return stub


def _row(conn, tid: str):
    return conn.execute(
        "SELECT status, consecutive_failures, current_run_id FROM tasks WHERE id = ?", (tid,),
    ).fetchone()


def _events(conn, tid: str, kind: str) -> list[dict]:
    rows = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = ? ORDER BY id",
        (tid, kind),
    ).fetchall()
    return [json.loads(row["payload"]) for row in rows if row["payload"]]


def _spawn_refused_phase(conn, tid: str) -> str | None:
    payloads = _events(conn, tid, "spawn_refused")
    return payloads[-1].get("phase") if payloads else None


@pytest.mark.parametrize("barrier", ["before_preparation", "after_preparation", "before_grant"])
def test_no_barrier_grants_work_once_a_stop_is_observed(
    conn, switch, monkeypatch, barrier,
):
    """before preparation / after preparation / before the grant are all barriers."""
    stub = _LaunchRecorder(switch)
    monkeypatch.setattr(kbd, "_default_spawn", stub)
    if barrier == "before_preparation":
        # The lane gate consumes the first poll; the launch's own barrier is next.
        switch.arm(1)
    elif barrier == "after_preparation":
        stub.arm_on_spawn = 0  # stop observed as the spawn returns, pre-claim
    else:
        stub.arm_on_spawn = 1  # stop observed at the pre-grant barrier
    tid = kb.create_task(conn, title=f"barrier {barrier}", assignee="w")

    result = kbd.dispatch_once(conn, failure_limit=3, should_stop=switch)

    assert stub.grants == [], "a stopping gateway must never grant"
    assert result.spawned == []
    row = _row(conn, tid)
    assert row["consecutive_failures"] == 0, "shutdown must not consume a retry"
    assert _spawn_refused_phase(conn, tid) is None
    if barrier == "before_preparation":
        assert stub.spawn_calls == 0, "preparation must not start after a drain"
    else:
        assert stub.spawn_calls == 1
        assert stub.cancels == 1, "the prepared worker must be cancelled"
    if barrier == "before_grant":
        # The card WAS claimed, so its run is closed and it needs an operator unblock.
        assert result.interrupted == [tid]
        assert result.cancelled == []
        assert row["status"] == "blocked"
        runs = conn.execute(
            "SELECT status, outcome FROM task_runs WHERE task_id = ?", (tid,),
        ).fetchall()
        assert [(r["status"], r["outcome"]) for r in runs] == [("interrupted", "interrupted")]
        assert [p["reason"] for p in _events(conn, tid, "blocked")] == [
            kbd.LAUNCH_STOPPED_BLOCK_REASON
        ]
    else:
        # Nothing was claimed: still queued, no run, nothing for the operator to unblock.
        assert result.cancelled == [tid]
        assert result.interrupted == []
        assert row["status"] == "ready", "an unclaimed card stays queued"
        assert row["current_run_id"] is None, "nothing may be claimed"


def test_a_granted_worker_is_never_cancelled_or_requeued(conn, switch, recorder, monkeypatch):
    """Past the grant the worker owns the card, whatever happens next."""
    tid = kb.create_task(conn, title="grant then drain", assignee="w")

    result = kbd.dispatch_once(conn, failure_limit=3, should_stop=switch)
    assert [entry[0] for entry in result.spawned] == [tid]
    assert len(recorder.grants) == 1 and recorder.cancels == 0

    # A drain arriving after the grant changes nothing, and a live claimed run is
    # not reclaimed either.
    switch.arm(0)
    assert kbd.dispatch_once(conn, failure_limit=3, should_stop=switch).interrupted == []
    assert recorder.cancels == 0, "a granted worker must never be cancelled"
    assert _row(conn, tid)["status"] == "running"

    # Post-grant bookkeeping failing (here: the durable PID write) is an
    # observability problem, not a launch failure.
    other = kb.create_task(conn, title="pid write fails", assignee="w")
    switch.arm(10**6)

    def _fail(*_args, **_kwargs):
        raise RuntimeError("disk I/O error")

    monkeypatch.setattr(kbd, "_set_worker_pid", _fail)
    result = kbd.dispatch_once(conn, failure_limit=3, should_stop=switch)
    assert [entry[0] for entry in result.spawned] == [other]
    assert len(recorder.grants) == 2 and recorder.cancels == 0
    assert result.interrupted == []
    assert _spawn_refused_phase(conn, other) is None
    row = _row(conn, other)
    assert (row["status"], row["consecutive_failures"]) == ("running", 0)


def test_only_proven_shutdown_excuses_a_launch_failure(conn, switch, monkeypatch):
    """Everything else fails closed and keeps its real phase."""
    cases: list[tuple[str, BaseException, bool, str | None, int]] = [
        # name, raised, user manager stopping, expected phase, expected failures
        ("ordinary failure", RuntimeError("systemd-run exploded"), False, "launch", 1),
        ("identity corruption", RuntimeIdentityError("not JSON"), False, "runtime_identity", 1),
        ("identity during shutdown", RuntimeIdentityError("not JSON"), True, "runtime_identity", 1),
        ("launch failure while the manager stops", RuntimeError("workspace gone"), True, "launch", 1),
    ]
    for name, raised, stopping, phase, failures in cases:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(kbd, "_user_manager_stopping", lambda stopping=stopping: stopping)
            stub = _LaunchRecorder(switch)
            stub.raise_on_spawn = raised
            patch.setattr(kbd, "_default_spawn", stub)
            tid = kb.create_task(conn, title=name, assignee="w")
            switch.arm(10**6)

            result = kbd.dispatch_once(conn, failure_limit=3, should_stop=switch)

            assert result.interrupted == [], f"{name}: must not be excused"
            assert _spawn_refused_phase(conn, tid) == phase, name
            row = _row(conn, tid)
            assert (row["status"], row["consecutive_failures"]) == ("ready", failures), name

    # The same evidence, sampled at the native launch failure site, converts only
    # the failure it was sampled beside.
    with pytest.MonkeyPatch.context() as patch:
        stub = _LaunchRecorder(switch)
        stub.raise_on_spawn = kbd.WorkerLaunchInterrupted(
            "launch interrupted: the user manager is stopping",
        )
        patch.setattr(kbd, "_default_spawn", stub)
        tid = kb.create_task(conn, title="native shutdown evidence", assignee="w")

        result = kbd.dispatch_once(conn, failure_limit=3, should_stop=switch)

        # Earlier cards in this test are ``ready`` again, so the stub raised for
        # them too; every assertion below is scoped to this card. Nothing was
        # claimed, so this is a cancellation, not an operator-recovery pause.
        assert tid in result.cancelled
        assert tid not in result.interrupted
        assert _spawn_refused_phase(conn, tid) is None
        assert _row(conn, tid)["consecutive_failures"] == 0

    # ...and that conversion is exactly this helper's job, so pin its branches.
    from hermes_cli import kanban_worker_runtime as kwr

    monkeypatch.setattr(kbd, "_user_manager_stopping", lambda: True)
    with pytest.raises(RuntimeIdentityError):
        kwr._launch_failure_or_shutdown(RuntimeIdentityError("bad identity"), "t_x")
    already = kwr.WorkerLaunchInterrupted("cancelled by drain")
    with pytest.raises(kwr.WorkerLaunchInterrupted) as kept:
        kwr._launch_failure_or_shutdown(already, "t_x")
    assert kept.value is already
    original = RuntimeError("systemd-run exploded")
    with pytest.raises(kwr.WorkerLaunchInterrupted) as converted:
        kwr._launch_failure_or_shutdown(original, "t_x")
    assert converted.value.__cause__ is original
    monkeypatch.setattr(kbd, "_user_manager_stopping", lambda: False)
    with pytest.raises(RuntimeError) as unchanged:
        kwr._launch_failure_or_shutdown(original, "t_x")
    assert unchanged.value is original


def test_the_stop_transition_cannot_land_between_check_and_grant():
    """The drain takes effect before the decision or after the grant, never between.

    The dispatcher's launch runs in a thread while the flags are set from the event
    loop, so the boundary is what makes "read the flag, then grant" atomic: a
    transition started while a grant decision is in flight must wait for it rather
    than land in the gap.
    """
    import threading

    from gateway.kanban_watchers import GatewayKanbanWatchersMixin

    runner = GatewayKanbanWatchersMixin()
    runner._running = True
    runner._draining = False
    runner._external_drain_active = False
    assert runner._kanban_shutdown_requested() is False

    landed = threading.Event()
    waiting = threading.Event()
    timed_acquires = []
    lock = threading.Lock()

    class ObservedLock:
        def acquire(self, *args, **kwargs):
            if "timeout" in kwargs:
                timed_acquires.append(kwargs["timeout"])
            if lock.locked():
                waiting.set()
            return lock.acquire(*args, **kwargs)

        def release(self):
            lock.release()

        def __enter__(self):
            self.acquire()
            return self

        def __exit__(self, *_args):
            self.release()

    runner._kanban_grant_lock_handle = ObservedLock()

    def _drain():
        runner._kanban_transition(_draining=True)
        landed.set()

    with runner._kanban_grant_guard():
        thread = threading.Thread(target=_drain, daemon=True)
        thread.start()
        assert waiting.wait(timeout=5.0), "the transition must reach the busy boundary"
        assert not landed.is_set(), "the drain must wait for the in-flight grant"

    assert landed.wait(timeout=5.0), "the transition must complete once the grant is done"
    thread.join(timeout=5.0)
    assert timed_acquires == [], "a delayed boundary must never time out and proceed unlocked"
    assert runner._kanban_shutdown_requested() is True
    # ...and a launch that starts now sees the stop immediately.
    assert runner._kanban_shutdown_requested() is True


@pytest.mark.parametrize("stdout,expected", [
    ("stopping\n", True),
    ("running\n", False),
    ("degraded\n", False),
    ("", False),
    (None, False),  # systemctl missing or the bus is inaccessible
])
def test_user_manager_evidence_requires_the_stopping_state(monkeypatch, stdout, expected):
    import subprocess

    from hermes_cli import gateway

    if stdout is None:
        def _run(*_args, **_kwargs):
            raise RuntimeError("systemctl is not available on this system")
    else:
        def _run(*args, **kwargs):
            return subprocess.CompletedProcess(args, 1, stdout, "")

    monkeypatch.setattr(gateway, "_run_systemctl", _run)
    assert kbd._user_manager_stopping() is expected


@pytest.mark.parametrize("engage", ["external_drain", "draining", "stopped"])
def test_a_gateway_that_is_draining_or_stopped_grants_nothing(engage):
    """The predicate the dispatcher polls: both drains, and a stopped gateway.

    The REVERSIBLE external drain (``.drain_request.json``) sets only
    ``_external_drain_active`` while advertising ``draining``, so omitting it
    would leave the queue granting through the quiesce window.
    """
    from gateway.kanban_watchers import GatewayKanbanWatchersMixin

    runner = GatewayKanbanWatchersMixin()
    runner._running = True
    runner._draining = False
    runner._external_drain_active = False
    assert runner._kanban_shutdown_requested() is False, "an ordinary tick must grant"

    if engage == "external_drain":
        runner._external_drain_active = True
    elif engage == "draining":
        runner._draining = True
    else:
        runner._running = False
    assert runner._kanban_shutdown_requested() is True

    # Every one of these states is recoverable; granting resumes with it.
    runner._running = True
    runner._draining = False
    runner._external_drain_active = False
    assert runner._kanban_shutdown_requested() is False


def test_native_grant_pipe_sends_one_complete_frame(monkeypatch):
    import os
    from hermes_cli import kanban_runtime as runtime
    from hermes_cli import kanban_worker_runtime as worker

    read_fd, write_fd = os.pipe()
    if os.name == "nt":
        worker._set_windows_pipe_nonblocking(write_fd)
    else:
        os.set_blocking(write_fd, False)
    frame = b'{"grant": true, "run_id": 23}\n'
    writes = []
    native_write = os.write

    def write_once(fd, data):
        writes.append(bytes(data))
        return native_write(fd, data)

    monkeypatch.setattr(worker.os, "write", write_once)
    try:
        worker._write_worker_grant(write_fd, frame)
    finally:
        os.close(write_fd)
    with os.fdopen(read_fd, "r") as reader:
        monkeypatch.setattr(runtime.sys, "stdin", reader)
        assert runtime._read_bootstrap_message() == {"grant": True, "run_id": 23}
    assert writes == [frame], "a completed grant must not be resent"


@pytest.mark.parametrize("write_error", [BlockingIOError, BrokenPipeError])
def test_partial_native_grant_failure_cancels_only_the_ungranted_worker(
    conn, switch, recorder, monkeypatch, write_error,
):
    import io
    from dataclasses import replace
    from hermes_cli import kanban_runtime as runtime
    from hermes_cli import kanban_worker_runtime as worker

    delivered = bytearray()
    frame = b'{"grant": true, "run_id": 23}\n'

    def partial_write(_fd, data):
        if delivered:
            raise write_error
        delivered.extend(data[:-1])
        return len(data) - 1

    def blocked_grant(_run_id, _claim_lock):
        with pytest.MonkeyPatch.context() as patch:
            clock = iter((0.0, 0.0, worker._GRANT_TIMEOUT_SECONDS + 1))
            patch.setattr(worker.os, "write", partial_write)
            patch.setattr(worker.time, "monotonic", lambda: next(clock))
            patch.setattr(worker.time, "sleep", lambda _seconds: None)
            worker._write_worker_grant(-1, frame)

    def spawn(*args, **kwargs):
        return replace(recorder(*args, **kwargs), grant=blocked_grant)

    monkeypatch.setattr(kbd, "_default_spawn", spawn)
    tid = kb.create_task(conn, title="grant pipe full", assignee="w")
    result = kbd.dispatch_once(conn, failure_limit=3, should_stop=switch)
    assert result.spawned == [] and recorder.cancels == 1
    row = _row(conn, tid)
    assert (row["status"], row["consecutive_failures"]) == ("ready", 1)
    # Even syntactically complete JSON is not a grant without the final newline.
    assert bytes(delivered) == frame[:-1]
    monkeypatch.setattr(runtime.sys, "stdin", io.StringIO(delivered.decode()))
    with pytest.raises(RuntimeIdentityError, match="incomplete worker bootstrap message"):
        runtime._read_bootstrap_message()


@pytest.mark.platforms("windows")
def test_windows_311_pipe_mode_supports_bounded_grants():
    import os
    from hermes_cli import kanban_worker_runtime as worker

    read_fd, write_fd = os.pipe()
    try:
        worker._set_windows_pipe_nonblocking(write_fd)
        frame = b'{"grant": true}\n'
        worker._write_worker_grant(write_fd, frame)
        assert os.read(read_fd, len(frame)) == frame
    finally:
        os.close(write_fd)
        os.close(read_fd)

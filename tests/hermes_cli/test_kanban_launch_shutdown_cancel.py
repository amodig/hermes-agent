"""A draining gateway must not grant Kanban work to a worker it will not outlive.

Issue #43: the embedded dispatcher kept launching workers straight through the
shutdown window. The fix threads a ``should_stop`` predicate from the runner /
daemon down into the native spawn, so cancellation is observed at four explicit
barriers (before preparation, after preparation / before claim, during bootstrap
waits, and immediately before the grant).

Contract:
* a cancelled launch charges NO failure and never becomes ``runtime_identity``;
* before the claim the queued card is left completely untouched;
* after the claim the prepared worker is cancelled and that exact run is closed
  as ``interrupted`` with the sticky ``gateway_stopping`` pause;
* after a successful grant nothing is ever cancelled;
* a genuine launch failure still fails closed and counts normally — except when
  shutdown is positively proven (the user manager reports ``stopping``).
"""

from __future__ import annotations

import json
import subprocess
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


def test_cancellation_before_preparation_never_starts_a_launch(conn, switch, recorder):
    """First poll is the lane gate; the second is the pre-preparation barrier."""
    switch.arm(1)
    tid = kb.create_task(conn, title="cancel early", assignee="w")

    result = kbd.dispatch_once(conn, failure_limit=3, should_stop=switch)

    assert recorder.spawn_calls == 0, "preparation must not start after a drain"
    assert result.interrupted == [tid]
    assert result.spawned == []
    assert _row(conn, tid)["status"] == "ready"
    assert _row(conn, tid)["consecutive_failures"] == 0
    assert _spawn_refused_phase(conn, tid) is None


def test_cancellation_after_preparation_cancels_the_prepared_worker(conn, switch, recorder):
    """The spawn completed its work, but the drain must stop the claim."""
    recorder.arm_on_spawn = 0
    tid = kb.create_task(conn, title="cancel late", assignee="w")

    result = kbd.dispatch_once(conn, failure_limit=3, should_stop=switch)

    assert recorder.spawn_calls == 1
    assert recorder.cancels == 1
    assert result.interrupted == [tid]
    assert result.spawned == []
    row = _row(conn, tid)
    assert row["status"] == "ready", "an unclaimed card must stay queued"
    assert row["current_run_id"] is None, "nothing may be claimed for a cancelled launch"
    assert row["consecutive_failures"] == 0
    assert _spawn_refused_phase(conn, tid) is None


def test_cancellation_before_grant_pauses_the_claimed_run(conn, switch, recorder):
    """One poll is allowed (post-claim barrier), the next one is the grant barrier."""
    recorder.arm_on_spawn = 1
    tid = kb.create_task(conn, title="cancel at grant", assignee="w")

    result = kbd.dispatch_once(conn, failure_limit=3, should_stop=switch)

    assert recorder.cancels == 1
    assert recorder.grants == [], "a draining gateway must not grant the run"
    assert result.interrupted == [tid]
    assert result.spawned == []
    assert result.auto_blocked == []

    row = _row(conn, tid)
    assert row["status"] == "blocked"
    assert row["consecutive_failures"] == 0, "shutdown must not consume a retry"

    runs = conn.execute(
        "SELECT id, status, outcome FROM task_runs WHERE task_id = ?", (tid,),
    ).fetchall()
    assert [(r["status"], r["outcome"]) for r in runs] == [("interrupted", "interrupted")]

    blocked = _events(conn, tid, "blocked")
    assert [payload["reason"] for payload in blocked] == [kbd.LAUNCH_STOPPED_BLOCK_REASON]


def test_granted_worker_survives_the_shutdown_that_follows(conn, switch, recorder):
    tid = kb.create_task(conn, title="grant then drain", assignee="w")

    result = kbd.dispatch_once(conn, failure_limit=3, should_stop=switch)
    assert result.spawned and result.spawned[0][0] == tid
    assert recorder.grants and recorder.cancels == 0

    # The drain arrives only after the grant; the worker already owns the card.
    switch.arm(0)
    later = kbd.dispatch_once(conn, failure_limit=3, should_stop=switch)
    assert later.interrupted == []
    assert recorder.cancels == 0, "a granted worker must never be cancelled"
    assert _row(conn, tid)["status"] == "running"


def test_genuine_launch_failure_still_fails_closed(conn, switch, recorder):
    recorder.raise_on_spawn = RuntimeError("systemd-run exploded")
    tid = kb.create_task(conn, title="real failure", assignee="w")

    result = kbd.dispatch_once(conn, failure_limit=3, should_stop=switch)

    assert result.interrupted == []
    assert _spawn_refused_phase(conn, tid) == "launch"
    row = _row(conn, tid)
    assert row["status"] == "ready"
    assert row["consecutive_failures"] == 1


def test_runtime_identity_failure_keeps_its_own_phase(conn, switch, recorder):
    recorder.raise_on_spawn = RuntimeIdentityError("runtime identity is not JSON")
    tid = kb.create_task(conn, title="identity failure", assignee="w")

    kbd.dispatch_once(conn, failure_limit=3, should_stop=switch)

    assert _spawn_refused_phase(conn, tid) == "runtime_identity"
    assert _row(conn, tid)["consecutive_failures"] == 1


def test_launch_interrupted_by_shutdown_leaves_the_card_queued(conn, switch, recorder):
    """What the native launch raises once it proves the user manager is stopping."""
    recorder.raise_on_spawn = kbd.WorkerLaunchInterrupted(
        "launch of t interrupted: the user manager is stopping",
    )
    tid = kb.create_task(conn, title="user manager stopping", assignee="w")

    result = kbd.dispatch_once(conn, failure_limit=3, should_stop=switch)

    assert result.interrupted == [tid]
    assert _spawn_refused_phase(conn, tid) is None
    row = _row(conn, tid)
    assert row["status"] == "ready"
    assert row["consecutive_failures"] == 0


def test_user_manager_stopping_does_not_exempt_a_dispatch_level_failure(
    conn, switch, recorder, monkeypatch,
):
    """The exemption belongs to the launch site, not to dispatch's handlers.

    Sampled next to a failure, positive infrastructure evidence says that failure
    was caused by shutdown. The same evidence sampled later, in a handler that
    also sees workspace, database and identity failures, says nothing about them.
    """
    recorder.raise_on_spawn = RuntimeError("workspace exploded")
    monkeypatch.setattr(kbd, "_user_manager_stopping", lambda: True)
    tid = kb.create_task(conn, title="unrelated failure", assignee="w")

    result = kbd.dispatch_once(conn, failure_limit=3, should_stop=switch)

    assert result.interrupted == []
    assert _spawn_refused_phase(conn, tid) == "launch"
    row = _row(conn, tid)
    assert row["status"] == "ready"
    assert row["consecutive_failures"] == 1


@pytest.mark.parametrize("stopping", [True, False])
def test_launch_failure_classification(monkeypatch, stopping):
    """Only a launch failure, and only on proven shutdown, becomes an interruption."""
    from hermes_cli import kanban_worker_runtime as kwr
    from hermes_cli.kanban_runtime import RuntimeIdentityError

    monkeypatch.setattr(kbd, "_user_manager_stopping", lambda: stopping)

    with pytest.raises(RuntimeIdentityError):
        kwr._launch_failure_or_shutdown(RuntimeIdentityError("bad identity"), "t_x")

    cancelled = kwr.WorkerLaunchInterrupted("cancelled by drain")
    with pytest.raises(kwr.WorkerLaunchInterrupted) as already:
        kwr._launch_failure_or_shutdown(cancelled, "t_x")
    assert already.value is cancelled

    original = RuntimeError("systemd-run exploded")
    if stopping:
        with pytest.raises(kwr.WorkerLaunchInterrupted) as converted:
            kwr._launch_failure_or_shutdown(original, "t_x")
        assert converted.value.__cause__ is original
    else:
        with pytest.raises(RuntimeError) as unchanged:
            kwr._launch_failure_or_shutdown(original, "t_x")
        assert unchanged.value is original


def test_concurrent_drain_does_not_exempt_a_genuine_failure(conn, switch, recorder):
    """A drain that starts as a real failure is raised must not excuse it.

    The runner's drain predicate is process state sampled AFTER the exception: if
    it could exempt a failure, a SIGTERM landing at the wrong moment would turn
    identity corruption into an uncharged "cancellation".
    """
    recorder.arm_on_spawn = 0
    recorder.raise_on_spawn = RuntimeIdentityError("runtime identity is not JSON")
    tid = kb.create_task(conn, title="failure then drain", assignee="w")

    result = kbd.dispatch_once(conn, failure_limit=3, should_stop=switch)

    assert result.interrupted == []
    assert _spawn_refused_phase(conn, tid) == "runtime_identity"
    row = _row(conn, tid)
    assert row["status"] == "ready"
    assert row["consecutive_failures"] == 1


def test_post_grant_bookkeeping_failure_keeps_the_granted_worker(
    conn, switch, recorder, monkeypatch,
):
    """Past the grant the worker owns the card, whatever the bookkeeping does."""
    tid = kb.create_task(conn, title="pid write fails", assignee="w")

    def _fail(*_args, **_kwargs):
        raise RuntimeError("disk I/O error")

    monkeypatch.setattr(kbd, "_set_worker_pid", _fail)

    result = kbd.dispatch_once(conn, failure_limit=3, should_stop=switch)

    assert [entry[0] for entry in result.spawned] == [tid]
    assert len(recorder.grants) == 1
    assert recorder.cancels == 0, "a granted worker must never be cancelled"
    assert result.interrupted == []
    assert _spawn_refused_phase(conn, tid) is None
    row = _row(conn, tid)
    assert row["status"] == "running"
    assert row["consecutive_failures"] == 0


@pytest.mark.parametrize("stdout,expected", [
    ("stopping\n", True),
    ("running\n", False),
    ("degraded\n", False),
    ("", False),
])
def test_user_manager_evidence_requires_the_stopping_state(monkeypatch, stdout, expected):
    from hermes_cli import gateway

    monkeypatch.setattr(
        gateway, "_run_systemctl",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 1, stdout, ""),
    )
    assert kbd._user_manager_stopping() is expected


def test_user_manager_probe_never_raises_on_failure(monkeypatch):
    from hermes_cli import gateway

    def _boom(*args, **kwargs):
        raise RuntimeError("systemctl is not available on this system")

    monkeypatch.setattr(gateway, "_run_systemctl", _boom)
    assert kbd._user_manager_stopping() is False

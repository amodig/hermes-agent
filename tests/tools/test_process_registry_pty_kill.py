"""PTY cleanup must finish independently of a detached descendant retaining the slave.

Kill and prune may not close a buffered PTY under a live reader, but deferring close
must also cancel that reader within a bounded interval. Output stays as reported by
the kill, completion is published once, and late teardown never resurrects a pruned
session. Real PTYs and processes: a fake cannot exercise the blocked buffer lock.
"""

import errno
import os
import shlex
import shutil
import sys
import threading
import time

import pytest

import tools.process_registry as module
from tools.process_registry import ProcessRegistry

_POSIX_PTY = pytest.mark.platforms("posix")

# The escapee exits on its own (it is reparented to init, outside what a test may signal).
# Before the fix the kill could only return once it had, so the kill deadline is shorter.
_ESCAPEE_LIFETIME_S = 8
_KILL_DEADLINE_S = 5


@pytest.fixture
def unscoped(monkeypatch):
    # No systemd scope: stopping one would reap the escapee and hide the hang.
    monkeypatch.setattr(module, "_is_supervised_gateway_process", lambda: False)


def _wait_for(predicate, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


def _terminate_owned(session):
    """Failure-path cleanup: stop the PTY child this test spawned (never the reparented escapee)."""
    if not session.exited and session._pty is not None:
        try:
            session._pty.terminate(force=True)
        except Exception:
            pass


def _spawn_with_escapee(registry, tmp_path, late_output=""):
    if shutil.which("setsid") is None:
        pytest.skip("setsid is required for the detached descendant fixture")
    pidfile = tmp_path / "escapee.pid"
    # Without ``late_output`` the escapee never writes to the PTY, so nothing ends the reader's
    # blocked read before the kill deadline. ``late_output`` is printed after the test has
    # killed the session.
    late = f"sleep 2; echo {late_output}; " if late_output else ""
    # The escapee's pid is read as the PPID of a grandchild, not as ``$$``: under
    # ``systemd-run --scope`` with environment expansion, ``$$`` reaches the shell as ``$``.
    session = registry.spawn_local(
        f"setsid sh -c 'sh -c \"echo \\$PPID\" > {pidfile}; {late}exec sleep {_ESCAPEE_LIFETIME_S}' & sleep 120",
        cwd=str(tmp_path), use_pty=True)
    try:
        assert _wait_for(lambda: pidfile.exists() and pidfile.read_text().strip(), 5)
    except BaseException:
        _terminate_owned(session)
        raise
    return session


def _kill_within_deadline(registry, session):
    result = {}
    killer = threading.Thread(
        target=lambda: result.update(registry.kill_process(session.id)), daemon=True)
    killer.start()
    killer.join(_KILL_DEADLINE_S)
    assert not killer.is_alive(), "kill_process blocked closing the PTY under a live reader"
    return result


def _wait_for_reader_close(session):
    # Closing must not wait for the detached descendant's slave lifetime.
    reader = session._reader_thread
    assert reader is not None
    assert _wait_for(lambda: not reader.is_alive(), _KILL_DEADLINE_S)
    assert session._pty.closed


@_POSIX_PTY
def test_kill_and_prune_release_reader_while_detached_descendant_holds_slave(
    tmp_path, monkeypatch, unscoped,
):
    pytest.importorskip("ptyprocess")
    registry = ProcessRegistry()
    stop, probe, alive, done = (tmp_path / name for name in ("stop", "probe", "alive", "done"))
    # Detach from the terminal and ignore shell hangup propagation, as a daemon
    # would. The file handshake still detects termination by the cleanup path.
    child = f"""
import os
import signal
from pathlib import Path
import time
os.setsid()
signal.signal(signal.SIGHUP, signal.SIG_IGN)
print("DETACHED-READY", flush=True)
deadline = time.monotonic() + 30
try:
    while not Path({str(stop)!r}).exists() and time.monotonic() < deadline:
        if Path({str(probe)!r}).exists():
            os.fstat(1)
            Path({str(alive)!r}).touch()
        time.sleep(0.02)
finally:
    Path({str(done)!r}).touch()
"""
    finish = registry._finish_reader
    release = threading.Event()

    def finish_after_prune(*args):
        # Hold reader teardown until pruning has removed the finished entry.
        release.wait(10)
        finish(*args)

    monkeypatch.setattr(registry, "_finish_reader", finish_after_prune)
    session = registry.spawn_local(
        f"{shlex.quote(sys.executable)} -c {shlex.quote(child)} & sleep 120",
        cwd=str(tmp_path), use_pty=True,
    )
    try:
        assert session._pty is not None, "test requires a real PTY, not pipe fallback"
        session.notify_on_complete = True
        assert _wait_for(lambda: "DETACHED-READY" in session.output_buffer, 5)
        master_fd = session._pty.fileno()
        result = _kill_within_deadline(registry, session)
        assert result["status"] == "killed"
        assert "DETACHED-READY" in result["output"]
        session.started_at = time.time() - module.FINISHED_TTL_SECONDS - 1

        def prune():
            with registry._lock:
                registry._prune_if_needed()

        pruner = threading.Thread(target=prune, daemon=True)
        pruner.start()
        pruner.join(_KILL_DEADLINE_S)
        assert not pruner.is_alive(), "pruning blocked closing the PTY under the registry lock"
        assert registry._lock.acquire(timeout=_KILL_DEADLINE_S), "registry lock was stranded"
        try:
            assert session.id not in registry._finished
        finally:
            registry._lock.release()
        release.set()
        _wait_for_reader_close(session)
        assert session._pty.fileobj.closed
        with pytest.raises(OSError) as closed:
            os.fstat(master_fd)
        assert closed.value.errno == errno.EBADF
        assert session.id not in registry._finished, "late reader resurrected a pruned session"
        assert registry.completion_queue.qsize() == 1
        event = registry.completion_queue.get_nowait()
        assert event["completion_reason"] == "killed"
        assert event["output"] == result["output"]
        probe.touch()
        assert _wait_for(alive.exists, 5), "cleanup signalled the detached slave holder"
        assert not done.exists(), "reader release depended on the detached child exiting"
    finally:
        release.set()
        stop.touch()
        _terminate_owned(session)
        _wait_for(done.exists, 5)


@_POSIX_PTY
def test_output_after_the_kill_is_not_added_to_the_killed_session(tmp_path, monkeypatch, unscoped):
    pytest.importorskip("ptyprocess")
    registry = ProcessRegistry()
    emit, emitted = registry._emit_output, []

    def recording_emit(session, text):
        emitted.append(text)
        emit(session, text)

    registry._emit_output = recording_emit
    session = _spawn_with_escapee(registry, tmp_path, late_output="LATE-ESCAPEE-OUTPUT")
    try:
        result = _kill_within_deadline(registry, session)
    finally:
        _terminate_owned(session)
    assert result["status"] == "killed"
    # Cancellation may discard the chunk before reading it; neither path keeps it.
    _wait_for_reader_close(session)
    assert "LATE-ESCAPEE-OUTPUT" not in session.output_buffer
    assert not any("LATE-ESCAPEE-OUTPUT" in text for text in emitted)


@_POSIX_PTY
def test_chunk_read_before_the_kill_is_not_added_after_it(tmp_path, unscoped):
    # The reader has read a chunk but not yet buffered it when the kill snapshots the output
    # and sets ``exited``. The chunk must not land in the killed session afterwards.
    pytest.importorskip("ptyprocess")
    registry = ProcessRegistry()
    entered, release = threading.Event(), threading.Event()
    ingest, emit, emitted = registry._ingest_output, registry._emit_output, []

    def paused_ingest(session, text, **kwargs):
        if "RACE-MARKER" in text:
            entered.set()
            release.wait(10)
        ingest(session, text, **kwargs)

    def recording_emit(session, text):
        emitted.append(text)
        emit(session, text)

    registry._ingest_output = paused_ingest
    registry._emit_output = recording_emit
    session = registry.spawn_local(
        "sleep 0.2; echo RACE-MARKER; exec sleep 15", cwd=str(tmp_path), use_pty=True)
    try:
        assert entered.wait(5), "reader never read the marker"
        result = registry.kill_process(session.id)
    finally:
        release.set()
    assert result["status"] == "killed"
    assert _wait_for(lambda: not session._reader_thread.is_alive(), 5)
    assert "RACE-MARKER" not in result.get("output", "")
    assert "RACE-MARKER" not in session.output_buffer
    assert not any("RACE-MARKER" in text for text in emitted)

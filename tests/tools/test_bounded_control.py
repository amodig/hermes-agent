"""``bounded_control``: strict surface, honest failures, and real-service semantics.

Two tiers:

* unit tier (fake process/manager): operations and arguments are refused before anything can
  run; worker/delegate contexts and unproven placement are refused without a fallback; an
  unsafe wrapper is refused; a readiness/identity mismatch means the payload is never granted;
  output stays bounded.
* real-service tier (``live_system_guard_bypass``): a disposable fixture boundary with two
  occupied worker slots proves the control operation runs beside them, that a race admits
  exactly one wrapper, and that native service teardown (not a reaper here) removes
  descendants after normal exit, withheld grant and caller death.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import pytest

import tools.bounded_control_tool as bc


# ---------------------------------------------------------------------------
# Unit tier: refusals before anything runs
# ---------------------------------------------------------------------------

def _error(args) -> str:
    payload = json.loads(bc._handle(args))
    assert "error" in payload, payload
    return payload["error"]


@pytest.mark.parametrize("args, expected", [
    ({"operation": "github_issue_list", "repo": "a/b", "number": 1}, "not applicable: number"),
    ({"operation": "github_status", "repo": "a/b"}, "not applicable: repo"),
    ({"operation": "github_issue_view", "repo": "a/b"}, "missing required parameter(s): number"),
    ({"operation": "github_issue_comment", "repo": "a/b", "number": 3}, "missing required parameter(s): body"),
    ({"operation": "github_issue_create", "repo": "a/b", "title": "x", "state": "open"}, "not applicable: state"),
    ({"operation": "github_issue_list", "repo": "--evil/x"}, "must not start with '-'"),
    ({"operation": "github_issue_list", "repo": "a/b/c"}, "must be OWNER/NAME"),
    ({"operation": "github_issue_list", "repo": "a/b", "limit": True}, "limit must be an integer"),
    ({"operation": "github_issue_list", "repo": "a/b", "limit": 0}, "limit must be between"),
    ({"operation": "github_issue_list", "repo": "a/b", "limit": 51}, "limit must be between"),
    ({"operation": "github_issue_list", "repo": "a/b", "state": "merged"}, "state must be one of"),
    ({"operation": "github_issue_create", "repo": "a/b", "title": "   "}, "title must not be blank"),
    ({"operation": "github_issue_create", "repo": "a/b", "title": "t" * 201}, "title must be at most"),
    ({"operation": "github_issue_comment", "repo": "a/b", "number": 1, "body": "x" * 20_001},
     "body must be at most"),
    ({"operation": "github_issue_comment", "repo": "a/b", "number": 1, "body": "\x00"}, "NUL"),
    ({"operation": "github_issue_view", "repo": "a/b", "number": -1}, "number must be between"),
    ({"operation": "shell"}, "operation must be one of"),
    ({"operation": "github_help", "command": "gh api user"}, "not applicable: command"),
])
def test_operations_and_arguments_are_refused_before_anything_runs(
    args, expected, monkeypatch,
):
    def _explode(*_a, **_kw):  # pragma: no cover - asserted not to run
        raise AssertionError("an invalid request must never reach execution")

    monkeypatch.setattr(bc, "_run_operation", _explode)
    assert expected in _error(args)


def test_worker_and_delegate_contexts_are_refused(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_deadbeef")
    assert "orchestrator-only" in _error({"operation": "github_status"})
    monkeypatch.delenv("HERMES_KANBAN_TASK")

    from tools import kanban_tools

    monkeypatch.setattr(
        kanban_tools, "_delegation_ctx",
        lambda name, default=None: True if name == "is_delegated_child_process_context" else default,
    )
    assert "delegate_task child" in _error({"operation": "github_status"})


def test_unproven_placement_refuses_instead_of_falling_back(monkeypatch):
    from tools.autonomous_resources import AutonomousResourceUnavailable

    def _outside(*, pid=None):
        raise AutonomousResourceUnavailable("launcher is outside autonomous.slice")

    monkeypatch.setattr(bc, "check_autonomous_placement", _outside)
    monkeypatch.setattr(bc, "_wrapper", lambda: __import__("pathlib").Path("/bin/true"))
    monkeypatch.setattr(bc.subprocess, "Popen", lambda *a, **k: pytest.fail("must not launch"))
    message = _error({"operation": "github_status"})
    assert "supervised process inside the autonomous aggregate" in message
    assert "must not fall back" in message


@pytest.mark.parametrize("shape", ["missing", "symlink", "group-writable"])
def test_unsafe_wrapper_is_refused(shape, tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / ".local" / "bin").mkdir(parents=True)
    wrapper = home / ".local" / "bin" / "hermes-github"
    if shape == "symlink":
        (home / ".local" / "bin" / "gh-real").write_text("#!/bin/sh\nexit 0\n")
        os.chmod(home / ".local" / "bin" / "gh-real", 0o755)
        wrapper.symlink_to(home / ".local" / "bin" / "gh-real")
    elif shape == "group-writable":
        wrapper.write_text("#!/bin/sh\nexit 0\n")
        os.chmod(wrapper, 0o775)
    monkeypatch.setattr(bc.Path, "home", classmethod(lambda cls: home))
    if shape == "missing":
        with pytest.raises(bc._Refuse) as excinfo:
            bc._wrapper()
        assert "refusing to fall back to plain gh" in str(excinfo.value)
        return
    with pytest.raises(bc._Refuse) as excinfo:
        bc._wrapper()
    assert "symlink" in str(excinfo.value) or "group/other writable" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Unit tier: the grant only follows a proven launch
# ---------------------------------------------------------------------------

class _RecordingStdin:
    def __init__(self):
        self.written = b""
        self.closed = False

    def write(self, data):
        self.written += data
        return len(data)

    def flush(self):
        return None

    def close(self):
        self.closed = True


class _BrokenStdin(_RecordingStdin):
    """The service died before the grant: the pipe is gone."""

    def write(self, data):
        raise BrokenPipeError("service exited before the grant")


class _FakeProc:
    """Real pipes for stdout/stderr plus a recording stdin, so the drains run for real."""

    def __init__(self, output: bytes, exit_code: int = 0, *, stderr_bytes: bytes = b"",
                 stdin=None):
        read_fd, write_fd = os.pipe()
        os.write(write_fd, output)
        os.close(write_fd)
        self.stdout = os.fdopen(read_fd, "rb", 0)
        self.stdin = stdin if stdin is not None else _RecordingStdin()
        self.stderr = None
        self.stderr_done = threading.Event()
        self._exit = exit_code
        if stderr_bytes:
            err_read, err_write = os.pipe()
            self.stderr = os.fdopen(err_read, "rb", 0)

            def _writer() -> None:
                try:
                    view = memoryview(stderr_bytes)
                    while view:
                        view = view[os.write(err_write, view[:65536]):]
                finally:
                    os.close(err_write)
                    self.stderr_done.set()

            threading.Thread(target=_writer, daemon=True).start()
        else:
            self.stderr_done.set()

    def poll(self):
        return self._exit

    def wait(self, timeout=None):
        return self._exit


def _stub_launch(monkeypatch, build, *, properties=None, memory=bc.MEMORY_BYTES):
    """Wire a fake manager/process around the real ``_run_operation`` flow."""
    monkeypatch.setattr(bc, "_wrapper", lambda: __import__("pathlib").Path("/bin/true"))
    monkeypatch.setattr(bc, "check_autonomous_placement", lambda pid=None: {
        "aggregate": {"unit": "fixture.slice", "cgroup": "/user.slice/fixture.slice"},
        "workers": {"unit": "fixture-workers.slice", "cgroup": "/user.slice/fixture.slice/fixture-workers.slice"},
        "worker_slot_occupied": True,
        "concurrency_hard_max": 2,
        "worker_scopes": ["/user.slice/fixture.slice/fixture-workers.slice/hermes-worker-a.scope"],
        "launcher_cgroup": "/user.slice/fixture.slice/launcher.scope",
        # `check_ancestors` reports kernel limits under the controller's own names.
        "aggregate_effective_limits": {"memory.high": memory, "memory.max": memory,
                                       "memory.swap.max": 0},
        "workers_effective_limits": {"memory.high": memory, "memory.max": memory,
                                     "memory.swap.max": 0},
    })
    monkeypatch.setattr(bc, "_resolve", lambda name: ["/usr/bin/" + name])
    monkeypatch.setattr(bc, "_unit_properties", lambda: properties)
    monkeypatch.setattr(bc, "check_ancestors", lambda *a, **k: {
        "effective_limits": {"memory.high": memory, "memory.max": memory,
                             "memory.swap.max": 0},
    })
    monkeypatch.setattr(bc.subprocess, "Popen", lambda argv, **k: build(argv))
    monkeypatch.setattr(bc, "live_contract", lambda: type("C", (), {"aggregate_unit": "fixture.slice"})())
    monkeypatch.setattr(
        "hermes_cli.kanban_runtime.process_start_time", lambda pid: 77, raising=False,
    )


def _properties(*, memory: int = bc.MEMORY_BYTES, **overrides):
    properties = {
        "LoadState": "loaded",
        "ActiveState": "active",
        "MainPID": "4242",
        "InvocationID": "inv-1",
        "ControlGroup": "/user.slice/fixture.slice/hermes-control.service",
        "MemoryHigh": str(memory),
        "MemoryMax": str(memory),
        "MemorySwapMax": "0",
    }
    properties.update(overrides)
    return properties


def _nonce_of(argv) -> str:
    """The nonce the parent injected into the gate command line."""
    return argv[argv.index("-c") + 2]


def _ready(argv, tail: bytes = b"", *, nonce=None, start_time: int = 77,
           **proc_kwargs) -> _FakeProc:
    line = json.dumps({"nonce": nonce if nonce is not None else _nonce_of(argv),
                       "pid": 4242, "start_time": start_time}).encode() + b"\n"
    return _FakeProc(line + tail, **proc_kwargs)


def test_readiness_mismatch_never_grants_the_payload(monkeypatch):
    _stub_launch(
        monkeypatch, lambda argv: _ready(argv, nonce="not-this-launch"),
        properties=_properties(MainPID="4242"),
    )
    result = bc._run_operation("github_status", {"operation": "github_status"})
    assert result["ok"] is False
    assert result["granted"] is False
    assert "refused before execution" in result["error"]
    assert "nonce" in result["error"]


def test_main_pid_mismatch_never_grants_the_payload(monkeypatch):
    recorded = {}
    _stub_launch(
        monkeypatch, lambda argv: recorded.setdefault("proc", _ready(argv)),
        properties=_properties(MainPID="9999"),
    )
    result = bc._run_operation("github_status", {"operation": "github_status"})
    assert result["ok"] is False
    assert result["granted"] is False
    assert "MainPID" in result["error"]
    assert recorded["proc"].stdin.written == b""


def test_lost_readiness_never_grants_the_payload(monkeypatch):
    _stub_launch(monkeypatch, lambda argv: _FakeProc(b""), properties=_properties())
    result = bc._run_operation("github_status", {"operation": "github_status"})
    assert result["granted"] is False
    assert "never announced readiness" in result["error"]


def test_memory_budget_clamps_down_to_a_lower_worker_bound(monkeypatch):
    """A slice already bounded below 512MiB must not get a wider control service."""
    fixture_bound = 128 * 1024 * 1024
    seen = []
    _stub_launch(
        monkeypatch,
        lambda argv: (seen.append(argv), _ready(argv))[1],
        properties=_properties(memory=fixture_bound),
        memory=fixture_bound,
    )
    result = bc._run_operation("github_status", {"operation": "github_status"})
    assert result["ok"] is True, result
    requested = {
        item.split("=", 1)[0]: item.split("=", 1)[1]
        for item in seen[0] if item.startswith(("MemoryHigh=", "MemoryMax="))
    }
    assert requested == {"MemoryHigh": str(fixture_bound), "MemoryMax": str(fixture_bound)}
    assert "MemorySwapMax=0" in seen[0]


def test_effective_ancestor_over_budget_is_refused(monkeypatch):
    """A finite ancestor wider than the requested budget refuses the launch."""
    monkeypatch.setattr(bc, "_unit_properties", lambda: _properties())
    monkeypatch.setattr(bc, "check_ancestors", lambda *a, **k: {
        "effective_limits": {"memory.high": bc.MEMORY_BYTES * 2,
                             "memory.max": bc.MEMORY_BYTES * 2,
                             "memory.swap.max": 0},
    })
    monkeypatch.setattr(
        "hermes_cli.kanban_runtime.process_start_time", lambda pid: 77, raising=False,
    )
    refusal = bc._verify_launch(
        {"nonce": "n", "pid": 4242, "start_time": 77}, nonce="n",
        placement={"aggregate": {"cgroup": "/user.slice/fixture.slice"}},
        memory_high=bc.MEMORY_BYTES, memory_max=bc.MEMORY_BYTES,
    )
    assert refusal is not None and "effective" in refusal, refusal


def test_grant_failure_is_a_refusal_not_an_exception(monkeypatch):
    """A service that dies before the grant must still be reaped and reported."""
    recorded = {}
    _stub_launch(
        monkeypatch,
        lambda argv: recorded.setdefault("proc", _ready(argv, stdin=_BrokenStdin())),
        properties=_properties(),
    )
    result = bc._run_operation("github_status", {"operation": "github_status"})
    assert result["ok"] is False
    assert result["granted"] is False
    assert "grant could not be delivered" in result["error"]
    assert result["exit_code"] is not None, result
    assert result["cleanup_complete"] is True


def test_stderr_is_drained_past_the_retained_cap(monkeypatch):
    """A noisy child must not block on a full stderr pipe; only the prefix is kept."""
    window = 200_000
    recorded = {}
    _stub_launch(
        monkeypatch,
        lambda argv: recorded.setdefault(
            "proc",
            _ready(argv, stderr_bytes=b"e" * window, exit_code=1),
        ),
        properties=_properties(),
    )
    result = bc._run_operation("github_status", {"operation": "github_status"})
    proc = recorded["proc"]
    assert result["ok"] is False and result["exit_code"] == 1
    assert len(result["error"]) <= bc.STDERR_CHARS, len(result["error"])
    assert proc.stderr_done.wait(10), "the stderr writer blocked: the pipe was not drained to EOF"


def test_bounded_output_and_grant_on_a_proven_launch(monkeypatch):
    recorded = {}
    _stub_launch(
        monkeypatch,
        lambda argv: recorded.setdefault(
            "proc", _ready(argv, (b"x" * 60_000) + b"\n")),
        properties=_properties(),
    )
    result = bc._run_operation("github_status", {"operation": "github_status"})
    proc = recorded["proc"]
    assert result["ok"] is True, result
    assert result["granted"] is True
    assert proc.stdin.written == b"1"
    assert result["truncated"] is True
    assert len(result["output"]) <= bc.OUTPUT_CHARS
    assert result["unit"] is None or result["unit"] == bc.CONTROL_UNIT


def test_pre_tool_call_hook_denial_prevents_execution(monkeypatch):
    """A denied mutation never reaches the wrapper: the standard guard runs first."""
    from hermes_cli import plugins as plugin_mod
    from model_tools import handle_function_call

    calls = []
    monkeypatch.setattr(bc, "_run_operation", lambda op, clean: calls.append(op))

    original = plugin_mod.PluginManager.invoke_hook

    def deny(self, hook_name, **kwargs):
        if hook_name == "pre_tool_call":
            return [{"action": "block", "message": "bounded control denied by policy"}]
        return original(self, hook_name, **kwargs)

    monkeypatch.setattr(plugin_mod.PluginManager, "invoke_hook", deny)

    out = handle_function_call(
        "bounded_control", {"operation": "github_issue_comment", "repo": "a/b",
                            "number": 1, "body": "hi"},
    )
    assert "denied by policy" in out or "block" in out.lower(), out
    assert calls == []


# ---------------------------------------------------------------------------
# Real-service tier: a disposable fixture boundary, two occupied worker slots
# ---------------------------------------------------------------------------

DRIVER = r'''
import json
import os
import sys

with open(os.environ["HERMES_TEST_PID"], "w", encoding="utf-8") as handle:
    handle.write(str(os.getpid()))

sys.path.insert(0, os.environ["HERMES_TEST_REPO"])
from tools import autonomous_resources as ar
from tools.autonomous_resources import _Contract

ar._LIVE_CONTRACT = _Contract(**json.loads(os.environ["HERMES_TEST_CONTRACT"]))
import tools.bounded_control_tool as bc

bc.CONTROL_UNIT = os.environ["HERMES_TEST_CONTROL_UNIT"]
if os.environ.get("HERMES_TEST_WITHHOLD"):
    bc._verify_launch = lambda *a, **k: "test withheld grant"
result = bc._handle(json.loads(os.environ["HERMES_TEST_ARGS"]))
with open(os.environ["HERMES_TEST_RESULT"], "w", encoding="utf-8") as handle:
    handle.write(result)
if os.environ.get("HERMES_TEST_HOLD"):
    import time
    time.sleep(float(os.environ["HERMES_TEST_HOLD"]))
'''

WRAPPER_TEMPLATE = """#!/bin/sh
set -eu
witness=__WITNESS__
printf 'start %s\\n' "$$" >> "$witness"
cgroup="$(cut -d: -f3 /proc/self/cgroup)"
printf 'cgroup %s\\n' "$cgroup" >> "$witness"
if [ -r "/sys/fs/cgroup${cgroup}/memory.max" ]; then
  printf 'memory-max %s\\n' "$(cat "/sys/fs/cgroup${cgroup}/memory.max")" >> "$witness"
fi
__SLEEP____DETACH__printf 'end %s\\n' "$$" >> "$witness"
"""


def _live_home(tmp_path: Path) -> Path:
    home = tmp_path / "home"
    (home / ".local" / "bin").mkdir(parents=True)
    (home / ".hermes").mkdir()
    return home


def _write_wrapper(home: Path, witness: Path, *, sleep: int = 0, detach: bool = False) -> None:
    """The fake wrapper is created with its witness/behaviour baked in.

    The payload environment is deliberately clean (no HERMES_TEST_*), so the script cannot
    read test knobs from the environment: everything it needs is written into it here.
    """
    body = WRAPPER_TEMPLATE.replace("__WITNESS__", f'"{witness}"')
    body = body.replace("__SLEEP__", f'sleep {sleep}\n' if sleep else "")
    body = body.replace(
        "__DETACH__", 'sleep 30 &\nprintf \'detached %s\\n\' "$!" >> "$witness"\n'
        if detach else "",
    )
    wrapper = home / ".local" / "bin" / "hermes-github"
    wrapper.write_text(body, encoding="utf-8")
    os.chmod(wrapper, 0o755)


def _driver_env(repo: Path, boundary, home: Path, unit: str, args, result_path: Path,
                **extra) -> dict:
    from tests._fixtures.autonomous_slice import user_bus_env

    env = dict(os.environ)
    env.update(user_bus_env())
    env.update({
        "HERMES_TEST_REPO": str(repo),
        "HERMES_TEST_CONTRACT": json.dumps(boundary.contract_kwargs()),
        "HERMES_TEST_CONTROL_UNIT": unit,
        "HERMES_TEST_ARGS": json.dumps(args),
        "HERMES_TEST_RESULT": str(result_path),
        "HERMES_TEST_PID": str(result_path) + ".pid",
        "HOME": str(home),
        "HERMES_HOME": str(home / ".hermes"),
        "TMPDIR": os.environ.get("TMPDIR", "/tmp"),
        **extra,
    })
    return env


def _launch_repo() -> Path:
    return Path(__file__).resolve().parents[2]


TERMINAL_STATES = ("inactive", "failed", "not-found")


def _await_terminal(unit: str, timeout: float = 60.0) -> str:
    """Wait for the transient unit to leave the manager, and report its final state."""
    from tests._fixtures.autonomous_slice import show

    deadline = time.monotonic() + timeout
    state = show(unit).get("ActiveState")
    while time.monotonic() < deadline and state not in TERMINAL_STATES:
        time.sleep(0.25)
        state = show(unit).get("ActiveState")
    return state


def _launch_in(boundary, argv, env, **kwargs):
    """Run *argv* as a real systemd scope inside the fixture aggregate."""
    return subprocess.Popen(
        boundary.launcher_argv(argv, unit_suffix=uuid.uuid4().hex[:8]),
        start_new_session=True, env=env, **kwargs,
    )

def _start_worker_sleeper(boundary, index: int):
    import subprocess
    import uuid
    from tests._fixtures.autonomous_slice import user_bus_env

    unit = f"hermes-worker-fixture{boundary.token}{index}-{uuid.uuid4().hex[:6]}.scope"
    return subprocess.Popen(
        ["systemd-run", "--user", "--scope", "--quiet", "--collect",
         f"--slice={boundary.workers_unit}", "--unit", unit, "--", "sleep", "120"],
        start_new_session=True, env=user_bus_env(),
    )


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="real systemd scopes")
@pytest.mark.live_system_guard_bypass
def test_control_runs_beside_two_occupied_worker_slots(tmp_path, monkeypatch):
    from tests._fixtures.autonomous_slice import (
        HIGH_BYTES, disposable_boundary, require_free_boundary, require_user_bus,
    )

    require_user_bus()
    require_free_boundary()
    home = _live_home(tmp_path)
    witness = tmp_path / "witness.txt"
    sleepers = []
    with disposable_boundary(concurrency_hard_max=2) as boundary:
        boundary.install(monkeypatch)
        try:
            _write_wrapper(home, witness)
            sleepers = [_start_worker_sleeper(boundary, i) for i in (1, 2)]
            # Both slots full: a third autonomous worker is refused.
            probe = _launch_in(boundary,
                [sys.executable, "-c",
                 "import os,sys;sys.path.insert(0,%r);"
                 "from tools.autonomous_resources import check_autonomous_boundary;"
                 "import json;print(json.dumps(check_autonomous_boundary(pid=os.getpid())))"
                 % str(_launch_repo())],
                env=_driver_env(_launch_repo(), boundary, home, "hermes-control-unused.service",
                                {}, tmp_path / "probe.json"),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            _out, err = probe.communicate(timeout=90)
            assert probe.returncode != 0, "a third worker must be refused while both slots are busy"
            assert b"AutonomousWorkerBusy" in err or b"already holds" in err, err

            # The typed control operation still runs, in its own sibling service.
            unit = f"hermes-control-{boundary.token}.service"
            result_path = tmp_path / "result.json"
            caller = _launch_in(boundary,
                [sys.executable, "-c", DRIVER],
                env=_driver_env(_launch_repo(), boundary, home, unit,
                                {"operation": "github_status"}, result_path),
                
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            _, caller_err = caller.communicate(timeout=180)
            assert caller.returncode == 0, caller_err
            result = json.loads(result_path.read_text(encoding="utf-8"))
            assert result["ok"] is True, result
            assert result["granted"] is True
            lines = witness.read_text(encoding="utf-8").splitlines()
            assert any(line.startswith("start ") for line in lines), lines
            cgroup = next(line.split(" ", 1)[1] for line in lines if line.startswith("cgroup "))
            assert f"/{unit}" in cgroup, cgroup
            memory = next(line.split(" ", 1)[1] for line in lines if line.startswith("memory-max "))
            # The fixture slice is bounded well below 512MiB, so the control service must
            # be clamped to the tighter of its high/max bounds rather than widening either.
            expected_memory = min(bc.MEMORY_BYTES, HIGH_BYTES)
            assert int(memory) == expected_memory, (memory, expected_memory)
            # The two worker sleepers are untouched by the control operation.
            assert all(sleeper.poll() is None for sleeper in sleepers)
        finally:
            for sleeper in sleepers:
                sleeper.kill()


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="real systemd scopes")
@pytest.mark.live_system_guard_bypass
def test_single_flight_withheld_grant_and_native_teardown(tmp_path, monkeypatch):
    from tests._fixtures.autonomous_slice import (
        disposable_boundary, require_free_boundary, require_user_bus, show,
    )

    require_user_bus()
    require_free_boundary()
    home = _live_home(tmp_path)
    witness = tmp_path / "witness.txt"
    unit = "hermes-control-fixture"
    with disposable_boundary(concurrency_hard_max=2) as boundary:
        boundary.install(monkeypatch)
        unit = f"hermes-control-{boundary.token}.service"
        try:
            # 1. A withheld grant never executes the wrapper.
            _write_wrapper(home, witness)
            withheld = tmp_path / "withheld.json"
            caller = _launch_in(boundary,
                [sys.executable, "-c", DRIVER],
                env=_driver_env(_launch_repo(), boundary, home, unit,
                                {"operation": "github_status"}, withheld,
                                HERMES_TEST_WITHHOLD="1"),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            _, err = caller.communicate(timeout=120)
            assert withheld.exists(), err.decode("utf-8", "replace")[-2000:]
            result = json.loads(withheld.read_text(encoding="utf-8"))
            assert result["granted"] is False and result["ok"] is False, result
            assert not witness.exists(), witness.read_text(encoding="utf-8")
            assert _await_terminal(unit) in TERMINAL_STATES

            # 2. A race admits exactly one wrapper; the loser never disturbs the winner.
            _write_wrapper(home, witness, sleep=5)
            racers = []
            for index in (1, 2):
                racers.append(_launch_in(boundary,
                    [sys.executable, "-c", DRIVER],
                    env=_driver_env(_launch_repo(), boundary, home, unit,
                                    {"operation": "github_issue_list", "repo": "owner/name"},
                                    tmp_path / f"race-{index}.json"),
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                ))
            for racer in racers:
                racer.communicate(timeout=180)
            results = [json.loads((tmp_path / f"race-{i}.json").read_text(encoding="utf-8"))
                       for i in (1, 2)]
            starts = [line for line in witness.read_text(encoding="utf-8").splitlines()
                      if line.startswith("start ")]
            assert len(starts) == 1, (starts, results)
            assert sum(1 for r in results if r["ok"]) == 1, results
            assert _await_terminal(unit) in TERMINAL_STATES

            # 3. A detached child of the payload is removed by the service's own teardown.
            _write_wrapper(home, witness, detach=True)
            witness.unlink()
            detached_result = tmp_path / "detached.json"
            caller = _launch_in(boundary,
                [sys.executable, "-c", DRIVER],
                env=_driver_env(_launch_repo(), boundary, home, unit,
                                {"operation": "github_status"}, detached_result),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            _, err = caller.communicate(timeout=180)
            assert detached_result.exists(), err.decode("utf-8", "replace")[-2000:]
            assert json.loads(detached_result.read_text(encoding="utf-8"))["ok"] is True
            lines = witness.read_text(encoding="utf-8").splitlines()
            child = int(next(line.split(" ", 1)[1] for line in lines
                             if line.startswith("detached ")))

            def _gone() -> bool:
                try:
                    os.kill(child, 0)
                except ProcessLookupError:
                    return True
                return False

            deadline = time.monotonic() + 15
            while not _gone() and time.monotonic() < deadline:
                time.sleep(0.2)
            assert _gone(), f"detached payload child {child} survived native service teardown"
            state = _await_terminal(unit)
            assert state in TERMINAL_STATES, state

            # 4. Caller death: the service's own RuntimeMaxSec bounds the payload.
            _write_wrapper(home, witness, sleep=120)
            witness.unlink()
            killed_result = tmp_path / "killed.json"
            caller = _launch_in(boundary,
                [sys.executable, "-c", DRIVER],
                env=_driver_env(_launch_repo(), boundary, home, unit,
                                {"operation": "github_status"}, killed_result),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            pid_path = Path(str(killed_result) + ".pid")
            deadline = time.monotonic() + 45
            while time.monotonic() < deadline and not (
                show(unit).get("LoadState") == "loaded"
                and show(unit).get("ActiveState") == "active"
                and show(unit).get("ControlGroup")
                and pid_path.exists()
            ):
                time.sleep(0.2)
            properties = show(unit)
            assert (properties.get("LoadState") == "loaded"
                    and properties.get("ActiveState") == "active"
                    and properties.get("ControlGroup")), properties
            assert pid_path.exists(), "caller never announced its pid"
            caller_pid = int(pid_path.read_text(encoding="utf-8"))
            os.kill(caller_pid, 9)
            caller.wait(timeout=30)
            state = _await_terminal(unit, timeout=60)
            assert state in TERMINAL_STATES, show(unit)
        finally:
            assert show(unit).get("ActiveState") in TERMINAL_STATES, show(unit)

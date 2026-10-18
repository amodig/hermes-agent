"""Hermes Kanban worker launch, bootstrap, and process-runtime management.

The dispatcher owns board scheduling; this module owns the subprocess boundary,
worker identity handoff, and immutable runtime generation cleanup.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Optional
from typing import TYPE_CHECKING
import uuid

from tools.autonomous_resources import CGROUP_ROOT, WORKER_SCOPE_PREFIX, WORKER_SCOPE_SUFFIX

if TYPE_CHECKING:
    from hermes_cli.kanban_db import Task


def _dispatcher():
    from hermes_cli import kanban_db_dispatch

    return kanban_db_dispatch


def _current_host_epoch() -> str:
    from hermes_cli.kanban_runtime import current_host_epoch

    return current_host_epoch()


def _kb():
    # Late-bound facade access: the facade imports this module through
    # ``kanban_db_dispatch``, so a module-level import here would make a
    # worker-runtime-first import order read a partially initialized facade.
    from hermes_cli import kanban_db

    return kanban_db

DEFAULT_LOG_ROTATE_BYTES = 2 * 1024 * 1024
DEFAULT_LOG_BACKUP_COUNT = 1

# Each bootstrap phase re-hashes the whole sealed payload before the worker may
# publish (pre-import) or confirm (post-import) its identity, so the work is
# proportional to payload bytes. A sealed payload carries interpreters and
# dependencies and is gigabytes; a fixed wall clock therefore only fits a warm,
# idle host. Budget the bytes instead, at a conservative cold-cache rate for a
# hash walk over many small files, plus a fixed allowance for interpreter start
# and the scope/identity gates.
_BOOTSTRAP_PAYLOAD_BYTES_PER_SECOND = 24 * 1024 * 1024
_BOOTSTRAP_FIXED_ALLOWANCE_SECONDS = 30.0
_BOOTSTRAP_PHASE_FLOOR_SECONDS = 60.0
# A ceiling on the derivation itself, so a pathological or corrupt payload cannot
# wedge a dispatcher tick. It is deliberately independent of the claim lease: the
# pre-import phase runs before any claim exists, so shrinking this budget to fit a
# short lease would reintroduce the timeout instead of protecting the run.
_BOOTSTRAP_PHASE_CEILING_SECONDS = 600.0
_GRANT_TIMEOUT_SECONDS = 1.0


def _payload_bytes(root: Path) -> int:
    """Bytes the worker re-hashes per phase, walked once per launch.

    Mirrors ``_payload_digest``'s member set (same excludes, same manifest and
    owner skip) so the budget counts what is actually read, and reuses that walk
    instead of maintaining a second one. The walk is metadata-only and stays
    small next to the hashing it sizes, so it is not cached.
    """
    from hermes_cli.kanban_runtime_generation import _MANIFEST, _OWNER, _members

    total = 0
    for name, path in _members(root):
        if name in {_MANIFEST, _OWNER} or name.startswith(f".{_OWNER}."):
            continue
        if path.is_file():
            with contextlib.suppress(OSError):
                total += path.stat().st_size
    return total


def _bootstrap_phase_timeout(generation_root: Optional[Path] = None) -> float:
    """Seconds each bootstrap phase may take, derived from what it must hash."""
    timeout = _BOOTSTRAP_PHASE_FLOOR_SECONDS
    if generation_root is not None:
        timeout = max(
            timeout,
            _payload_bytes(Path(generation_root)) / _BOOTSTRAP_PAYLOAD_BYTES_PER_SECOND
            + _BOOTSTRAP_FIXED_ALLOWANCE_SECONDS,
        )
    return float(min(timeout, _BOOTSTRAP_PHASE_CEILING_SECONDS))


def _set_windows_pipe_nonblocking(fd: int) -> None:
    """Python 3.11 lacks ``os.set_blocking`` support for Windows pipes."""
    import ctypes
    from ctypes import wintypes
    import msvcrt

    set_state = ctypes.WinDLL("kernel32", use_last_error=True).SetNamedPipeHandleState
    set_state.argtypes = (
        wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD),
        ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(wintypes.DWORD),
    )
    set_state.restype = wintypes.BOOL
    mode = wintypes.DWORD(1)  # PIPE_NOWAIT
    if not set_state(msvcrt.get_osfhandle(fd), ctypes.byref(mode), None, None):
        raise ctypes.WinError(ctypes.get_last_error())


def _write_worker_grant(fd: int, frame: bytes) -> None:
    """Bound final-grant IO on a nonblocking pipe while the stop/grant lock is held.

    The child requires the terminating newline: a partial send followed by
    cancellation/EOF cannot confer ownership. Once the complete frame is sent,
    return immediately; no later cleanup failure may revoke that grant.
    """
    deadline = time.monotonic() + _GRANT_TIMEOUT_SECONDS
    pending = memoryview(frame)
    while pending:
        try:
            written = os.write(fd, pending)
        except BlockingIOError:
            written = 0
        if written == len(pending):
            return
        pending = pending[written:]
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("worker bootstrap grant pipe timed out")
        time.sleep(min(0.01, remaining))


@dataclass(frozen=True)
class WorkerLaunch:
    """Child process identity verified before its Kanban grant."""

    pid: int
    runtime_identity: dict[str, Any]
    preparation_id: str
    grant: Optional[Callable[[int, Optional[str]], None]] = None
    cancel: Optional[Callable[[], None]] = None
    launcher_pid: Optional[int] = None


class WorkerLaunchInterrupted(RuntimeError):
    """A launch cancelled on purpose because the gateway is draining or stopping.

    Deliberately NOT a ``RuntimeIdentityError`` and NOT an ordinary spawn
    failure: nothing about the task, the runtime or the scope is wrong, so the
    caller must neither charge a retry nor report identity corruption.
    """


def _stop_requested(should_stop: Optional[Callable[[], bool]]) -> bool:
    """Whether a caller-supplied cancellation predicate now asks us to stop.

    A broken predicate must never wedge dispatch, so it reads as "not yet" and
    the ordinary failure paths still apply.
    """
    if should_stop is None:
        return False
    try:
        return bool(should_stop())
    except Exception:
        return False


def _check_not_stopping(should_stop: Optional[Callable[[], bool]], task_id: str) -> None:
    """Abort a launch that would hand work to a gateway which is going away."""
    if _stop_requested(should_stop):
        raise WorkerLaunchInterrupted(
            f"launch of {task_id} cancelled: gateway is draining or stopping"
        )


def _launch_failure_or_shutdown(exc: BaseException, task_id: str) -> None:
    """Re-raise a launch failure, or classify it as an infrastructure shutdown.

    Runs WHERE the launch failed, so the user-manager evidence is sampled next to
    the failure instead of after it has propagated through unrelated handlers —
    otherwise a caller could excuse the wrong exception just because a shutdown
    started concurrently. Runtime-identity corruption is never shutdown evidence,
    and neither is anything the user manager is not positively reporting as
    ``stopping``; both re-raise unchanged.
    """
    from hermes_cli.kanban_runtime import RuntimeIdentityError

    if isinstance(exc, (WorkerLaunchInterrupted, RuntimeIdentityError)):
        raise exc
    if _dispatcher()._user_manager_stopping():
        raise WorkerLaunchInterrupted(
            f"launch of {task_id} interrupted: the user manager is stopping ({exc})"
        ) from exc
    raise exc


# Bounded registry of recently-reaped worker exits, filled by the reap loop in
# ``dispatch_once`` and read by ``detect_crashed_workers`` to classify a dead-pid
# task. Entry: ``pid -> (raw_wait_status, reaped_at_epoch)``; raw status kept so
# both WIFEXITED/WEXITSTATUS and WIFSIGNALED can be consulted. Trimmed by age
# plus a total size cap.
_RECENT_WORKER_EXIT_TTL_SECONDS = 600
_RECENT_WORKER_EXITS_MAX = 4096
_recent_worker_exits: "dict[int, tuple[int, float]]" = {}
_worker_pid_aliases: "dict[int, int]" = {}
"""Map retained launcher PIDs to the verified Hermes worker PID."""
_worker_processes: "dict[int, Any]" = {}
"""Live ``Popen`` handles retained until the dispatcher reaps their children.

Without this registry a fenced worker can outlive ``_default_spawn``'s local
handle; ``Popen.__del__`` then warns while the worker is still legitimately
running.
"""
_worker_runtime_snapshots: "dict[int, Path]" = {}
"""Runtime generations retained until their worker's actual process exits."""


def _record_worker_exit(pid: int, raw_status: int) -> None:
    """Record a reaped worker exit under its verified Hermes PID."""
    launcher_pid = int(pid)
    snapshot = _worker_runtime_snapshots.pop(launcher_pid, None)
    if snapshot is not None:
        with contextlib.suppress(Exception):
            from hermes_cli.kanban_runtime_generation import cleanup_runtime_generation

            cleanup_runtime_generation(snapshot)
    worker_pid = _worker_pid_aliases.pop(launcher_pid, launcher_pid)
    process = _worker_processes.pop(launcher_pid, None)
    if process is not None and getattr(process, "returncode", None) is None:
        with contextlib.suppress(Exception):
            process.returncode = os.waitstatus_to_exitcode(int(raw_status))
    if worker_pid <= 0:
        return
    now = time.time()
    _recent_worker_exits[worker_pid] = (int(raw_status), now)
    if len(_recent_worker_exits) > _RECENT_WORKER_EXITS_MAX // 2:
        cutoff = now - _RECENT_WORKER_EXIT_TTL_SECONDS
        for _pid in [p for p, (_s, t) in _recent_worker_exits.items() if t < cutoff]:
            _recent_worker_exits.pop(_pid, None)
    if len(_recent_worker_exits) > _RECENT_WORKER_EXITS_MAX:
        # Drop oldest half.
        ordered = sorted(_recent_worker_exits.items(), key=lambda kv: kv[1][1])
        for _pid, _ in ordered[: len(ordered) // 2]:
            _recent_worker_exits.pop(_pid, None)


def _classify_worker_exit(pid: int) -> "tuple[str, Optional[int]]":
    """``(kind, code)`` for a reaped worker PID: ``clean_exit`` (rc 0 while
    still ``running`` = protocol violation), ``rate_limited``
    (``KANBAN_RATE_LIMIT_EXIT_CODE``, never counts as a failure),
    ``nonzero_exit``, ``signaled`` (``code`` is the signal), ``unknown`` (pid
    not in the reap registry; ``code`` None)."""
    entry = _recent_worker_exits.get(int(pid))
    if entry is None:
        return ("unknown", None)
    raw, _ = entry
    raw = int(raw)
    signal_number = raw & 0x7F
    if signal_number == 0:
        return _dispatcher()._exit_code_kind((raw >> 8) & 0xFF)
    if signal_number != 0x7F:
        return ("signaled", signal_number)
    return ("unknown", None)


def reap_worker_zombies() -> "list[int]":
    """Reap exited workers without blocking; poll retained handles on Windows."""
    from hermes_cli.kanban_runtime_generation import sweep_runtime_generations

    sweep_runtime_generations()
    reaped: "list[int]" = []
    if os.name != "nt":
        for launcher_pid in tuple(_worker_processes):
            try:
                pid, status = os.waitpid(launcher_pid, os.WNOHANG)
            except (ChildProcessError, OSError):
                continue
            if pid == 0:
                continue
            _record_worker_exit(pid, status)
            reaped.append(pid)
    else:
        for pid, process in list(_worker_processes.items()):
            try:
                returncode = process.poll()
            except Exception:
                continue
            if returncode is None:
                continue
            returncode = int(returncode)
            raw_status = returncode << 8 if returncode >= 0 else -returncode
            _record_worker_exit(pid, raw_status)
            reaped.append(pid)
    return reaped
def _module_hermes_argv() -> list[str]:
    """Interpreter-bound Hermes CLI invocation (``hermes_cli.main`` is the
    console-script target — there is no top-level ``hermes`` package)."""
    return [sys.executable, "-m", "hermes_cli.main"]



def _absolute_hermes_path(path: str) -> str:
    """Return an absolute filesystem path for a resolved Hermes shim."""
    expanded = os.path.expanduser(path)
    return expanded if os.path.isabs(expanded) else os.path.abspath(expanded)


def _looks_like_path(value: str) -> bool:
    """Return true when a command override is an explicit path, not a name."""
    expanded = os.path.expanduser(value)
    return (
        expanded.startswith("~")
        or os.path.isabs(expanded)
        or bool(os.path.dirname(expanded))
        or "\\" in expanded
        or bool(re.match(r"^[A-Za-z]:", expanded))
    )


def _is_windows_batch_shim(path: str) -> bool:
    """Return true for Windows shell/batch shims that should not be argv[0]."""
    return path.lower().endswith((".cmd", ".bat"))


def _path_search_names(command: str) -> list[str]:
    """Return executable names to try for an unqualified command."""
    if not _kb()._IS_WINDOWS or os.path.splitext(command)[1]:
        return [command]
    raw = os.environ.get("PATHEXT") or ".COM;.EXE;.BAT;.CMD"
    return [command + ext for ext in raw.split(";") if ext]


def _safe_which_no_cwd(command: str) -> Optional[str]:
    """Resolve a bare command from PATH without implicit current-dir search.

    On Windows ``shutil.which`` may search the current directory before PATH
    for bare names — unsafe for a dispatcher. Only explicit PATH entries are
    considered; empty / ``.`` entries are skipped.
    """
    for raw_dir in os.environ.get("PATH", "").split(os.pathsep):
        if not raw_dir or raw_dir == ".":
            continue
        directory = os.path.expanduser(raw_dir)
        for name in _path_search_names(command):
            candidate = os.path.join(directory, name)
            if os.path.isfile(candidate) and (_kb()._IS_WINDOWS or os.access(candidate, os.X_OK)):
                return candidate
    return None


def _hermes_path_argv(path: str) -> list[str]:
    """argv for a resolved Hermes executable path. Windows batch shims
    (``.cmd``/``.bat``) are unsafe as argv[0] because the argument vector
    includes task-derived values; prefer the module form."""
    if _kb()._IS_WINDOWS and _is_windows_batch_shim(path):
        return _module_hermes_argv()
    return [_absolute_hermes_path(path)]


def _resolve_hermes_argv() -> list[str]:
    """Prefer this interpreter's Hermes installation over any PATH shim.

    An explicit ``HERMES_BIN`` is resolved without implicit current-directory
    lookup and subsequently checked against the sealed installation.
    """
    import shutil
    import importlib.util

    env_bin = os.environ.get("HERMES_BIN", "").strip()
    if env_bin:
        if _looks_like_path(env_bin):
            return _hermes_path_argv(env_bin)
        resolved_env_bin = _safe_which_no_cwd(env_bin)
        if resolved_env_bin:
            return _hermes_path_argv(resolved_env_bin)
        return _module_hermes_argv()

    if importlib.util.find_spec("hermes_cli") is not None:
        return _module_hermes_argv()

    hermes_bin = _safe_which_no_cwd("hermes") if _kb()._IS_WINDOWS else shutil.which("hermes")
    if hermes_bin:
        return _hermes_path_argv(hermes_bin)
    return _module_hermes_argv()


def _validate_worker_entrypoint() -> None:
    """Only an entrypoint for this installation can use its sealed generation."""
    if not os.environ.get("HERMES_BIN", "").strip():
        return
    import sysconfig

    command = _dispatcher()._resolve_hermes_argv()
    canonical = {
        (Path(sysconfig.get_path("scripts")) / name).resolve()
        for name in ("hermes", "hermes.exe")
    }
    canonical.add(Path(__file__).resolve().parents[1] / "hermes")
    if len(command) == 1 and Path(command[0]).resolve() in canonical:
        return
    raise RuntimeError(
        "HERMES_BIN must select this installation's Hermes entrypoint for immutable "
        "Kanban workers; custom wrappers and other runtimes cannot be sealed. "
        "Unset HERMES_BIN or run the dispatcher from the intended Hermes installation."
    )


def _worker_terminal_timeout_env(
    max_runtime_seconds: Optional[int],
    current_timeout: Optional[str],
) -> Optional[str]:
    """Return a worker-scoped TERMINAL_TIMEOUT override, if needed.

    When ``max_runtime_seconds`` exceeds the terminal tool's default timeout,
    raise only the child's default so a long command isn't killed by the
    generic terminal default first.
    """
    if max_runtime_seconds is None:
        return None
    try:
        runtime = int(max_runtime_seconds)
    except (TypeError, ValueError):
        return None
    if runtime <= 0:
        return None

    desired = max(1, runtime - _dispatcher().KANBAN_TERMINAL_TIMEOUT_GRACE_SECONDS)
    try:
        existing = int(str(current_timeout).strip()) if current_timeout else 0
    except (TypeError, ValueError):
        existing = 0
    if existing >= desired:
        return None
    return str(desired)


@contextlib.contextmanager
def _worker_profile_scope(hermes_home: str, *, bind_home: bool = True):
    """Bind worker secrets; retain the dispatcher's passthrough policy for env building."""
    from agent.secret_scope import build_profile_secret_scope, reset_secret_scope, set_secret_scope
    from hermes_constants import get_process_hermes_home, reset_hermes_home_override, set_hermes_home_override
    from tools.terminal_scope import install_profile_terminal_scope, reset_terminal_scope
    from tui_gateway.launch_profile_policy import launch_secret_scope, launch_terminal_env

    home = Path(hermes_home)
    is_launch_home = home.resolve() == Path(get_process_hermes_home()).resolve()
    home_token = secret_token = terminal_token = None
    try:
        home_token = set_hermes_home_override(str(home)) if bind_home else None
        secret_token = set_secret_scope(
            launch_secret_scope(home) if is_launch_home else build_profile_secret_scope(home),
            profile_home=None if is_launch_home else str(home),
        )
        terminal_token = (
            install_profile_terminal_scope(home, env_overlay=launch_terminal_env() if is_launch_home else None)
            if bind_home else None
        )
        yield
    finally:
        if terminal_token is not None:
            reset_terminal_scope(terminal_token)
        if secret_token is not None:
            reset_secret_scope(secret_token)
        if home_token is not None:
            reset_hermes_home_override(home_token)


def _resolve_worker_cli_toolsets(hermes_home: Optional[str]) -> Optional[list[str]]:
    """Return the assigned profile's effective CLI toolsets for a worker.

    Resolved at dispatch time and passed as an explicit ``--toolsets`` pin so
    worker startup cannot fall back to a stale root/active-profile config or a
    profile whose top-level ``toolsets`` is only the kanban orchestrator
    surface. ``model_tools`` still appends the task-scoped kanban lifecycle
    tools when ``HERMES_KANBAN_TASK`` is set.
    """
    if not hermes_home:
        return None
    try:
        from hermes_cli.config import load_config
        from hermes_cli.tools_config import _get_platform_tools

        with _worker_profile_scope(hermes_home):
            cfg = load_config()
            toolsets = sorted(_get_platform_tools(cfg, "cli"))
        return toolsets or None
    except Exception as exc:
        _kb()._log.debug(
            "kanban worker: could not resolve CLI toolsets for HERMES_HOME=%r (%s)",
            hermes_home,
            exc,
        )
        return None


_retagged_workspace_roots: set[str] = set()


def _retag_legacy_worker_sessions(workspaces_root_path: str) -> None:
    """Reclaim pre-tag worker rows in state.db so they leave the session lists.

    Best-effort: the durable gate is ``state_meta`` in
    ``retag_kanban_worker_sessions``; the in-process set avoids reopening
    state.db on every spawn. A tick must never fail because a session DB was
    busy or missing.
    """
    if workspaces_root_path in _retagged_workspace_roots:
        return
    try:
        from hermes_state_registry import acquire, release_or_close

        db = acquire()
        try:
            db.retag_kanban_worker_sessions(workspaces_root_path)
        finally:
            release_or_close(db)
        _retagged_workspace_roots.add(workspaces_root_path)
    except Exception as exc:
        _kb()._log.debug("kanban worker: legacy session retag skipped (%s)", exc)


def _worker_argv(task: Task, profile_arg: str, hermes_home: Optional[str]) -> list[str]:
    """Build profile and task CLI arguments for the sealed runtime entrypoint."""
    cmd = [
        "-p", profile_arg,
        # A worker must NEVER boot the interactive TUI: its no-TTY bail-out
        # exits 0 without doing the task → "protocol violation" every attempt.
        "--cli",
        # Workers run under a profile-scoped HERMES_HOME and so see that
        # profile's shell-hook allowlist; pass --accept-hooks explicitly so
        # configured hooks still register.
        "--accept-hooks",
    ]
    # One `--skills X` pair per name: easier to read in `ps` and avoids quoting
    # ambiguity if a skill name contains unusual chars.
    for sk in task.skills or ():
        if sk:
            cmd.extend(["--skills", sk])
    if task.model_override:
        cmd.extend(["-m", task.model_override])
        # Pin the provider too so the worker resolves the model against the
        # intended backend (model X with provider Y is the classic board-stall).
        if task.provider_override:
            cmd.extend(["--provider", task.provider_override])
    # Independent of the model override — a task can run the profile's own
    # model at a different depth.
    if task.reasoning_effort:
        cmd.extend(["--reasoning", task.reasoning_effort])
    worker_toolsets = _dispatcher()._resolve_worker_cli_toolsets(hermes_home)
    if worker_toolsets:
        cmd.extend(["--toolsets", ",".join(worker_toolsets)])
    cmd.extend(["chat", "-q", f"work kanban task {task.id}"])
    return cmd


def _open_worker_log(task: Task, board: Optional[str]):
    """Append-mode per-task log (a re-run on unblock appends, never overwrites),
    rotated first. Anchored at the board root (not the shared kanban root) so
    `hermes kanban log` reads its own file and boards sharing task ids don't
    collide."""
    log_dir = _kb().worker_logs_dir(board=board)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{task.id}.log"
    dispatcher = _dispatcher()
    rotate_bytes, backup_count = dispatcher.worker_log_rotation_config()
    dispatcher._rotate_worker_log(log_path, rotate_bytes, backup_count)
    return open(log_path, "ab")
def _worker_scope_suffix(task: Task, *, preparation_id: Optional[str]) -> str:
    """Transient scope suffix for one worker launch.

    Shared by the launcher and by the refused-activation check, so the latter looks
    for the exact unit the launch would have created instead of guessing a name.
    """
    if task.current_run_id is None:
        # Pre-claim workers use the preparation id as their temporary scope;
        # the grant binds the eventual run before Kanban tools are available.
        return (
            f"kanban-{task.id}-preparation-{preparation_id}"
            if preparation_id
            else f"kanban-{task.id}-run-missing"
        )
    return f"kanban-{task.id}-run-{task.current_run_id}"


def _raise_if_launch_refused(scope_unit: str, returncode: Optional[int]) -> None:
    """Raise infrastructure when the launcher died because its scope was refused.

    Racing dispatchers can pass the capacity precheck before scopes activate;
    systemd admits only up to the slice's hard limit. A loser's launcher exits
    non-zero before bootstrap. That card never ran, so it must not spend a retry;
    the same exit code also describes a genuine bootstrap crash. A refused
    activation leaves OUR scope absent while the slice is full, whereas a real
    crash leaves our scope populated. If capacity has since freed, the evidence
    is ambiguous and the normal failure path handles it. Neither exit code nor
    stderr text is treated as proof of admission.
    """
    from tools import autonomous_resources as resources
    from tools.process_registry import _IS_LINUX, autonomous_boundary_inventory

    if not _IS_LINUX:
        return
    # INVENTORY, not admission: admission refuses on ANY populated worker scope --
    # including this launch's own -- so it cannot tell "the slot was held by someone
    # else" from "our scope exists and our child died inside it".
    try:
        report = autonomous_boundary_inventory()
    except resources.AutonomousResourceUnavailable as exc:
        raise resources.AutonomousResourceUnavailable(
            f"{exc}; worker launcher exited before bootstrap ({returncode})"
        ) from exc
    if any(path.endswith("/" + scope_unit) for path in report.get("worker_scopes", [])):
        return
    if report.get("worker_slot_occupied"):
        raise resources.AutonomousWorkerBusy(
            "autonomous worker slot occupied: the worker launcher was refused before "
            f"bootstrap ({returncode}); {scope_unit} was not created"
        )


def _restart_safe_worker_argv(
    task: Task, command: list[str], *, preparation_id: Optional[str] = None,
) -> list[str]:
    """Wrap a managed-gateway worker in the shared restart-safe scope."""
    from tools.process_registry import restart_safe_gateway_child_argv

    suffix = _worker_scope_suffix(task, preparation_id=preparation_id)
    scoped = restart_safe_gateway_child_argv(
        command, unit_suffix=suffix, require_restart_safe_scope=True, outlives_parent=True,
    )
    if task.current_run_id is None:
        if scoped.mode != "in_process" and not preparation_id:
            raise RuntimeError(
                "cannot create restart-safe systemd scope for Kanban worker: "
                "the claimed task has no current run id"
            )
        return scoped.argv
    return scoped.argv


#: Bounded cleanup event kind appended when the dispatcher reaps an orphaned worker scope.
WORKER_SCOPE_RECLAIM_EVENT = "worker_scope_reclaimed"

#: Inverse grammar of :func:`_worker_scope_suffix`. Only these two shapes name an
#: attempt this module can look up in ``task_runs``; every other suffix (a cron or
#: terminal scope, a ``-run-missing`` launch, a shape a later release adds) is left
#: exactly as found instead of guessed at.
_KANBAN_SCOPE_SUFFIX_GRAMMAR = (
    re.compile(r"^kanban-(?P<task>.+)-preparation-(?P<preparation>[0-9a-f]{32})$"),
    re.compile(r"^kanban-(?P<task>.+)-run-(?P<run>[0-9]+)$"),
)

_SCOPE_RUN_COLUMNS = (
    "id, task_id, status, started_at, ended_at, worker_pid, worker_started_at, metadata"
)

#: Slack around an attempt's own life when matching a LIVE scope to it, in seconds.
#: The launch creates the scope before the claim (pre-claim preparation is bounded
#: by the bootstrap phase timeouts) and a clock step is possible, so the window has
#: to be generous -- and still far smaller than the age of any historical or
#: imported run, which is what it exists to exclude.
_SCOPE_ATTEMPT_WINDOW_SECONDS = 600


def _scope_activation_epoch(unit_name: str) -> Optional[int]:
    """When the manager activated *unit_name*, or ``None`` when unreadable.

    A transient scope is created by exactly one launch, so its activation time is
    the only scope-side fact that proves a live scope belongs to the attempt being
    reconciled rather than to another board whose task/run identities happen to
    match (a board import preserves those identities, the unit name carries none).
    Read from the manager; the transient unit file it writes at activation is the
    fallback. Nothing here guesses: an unreadable activation fails closed.
    """
    from hermes_platform.resolver import locate_command
    from tools import process_registry as registry

    try:
        systemctl = locate_command("systemctl").command
    except Exception as exc:
        _kb()._log.debug("kanban worker scopes: systemctl cannot be resolved (%s)", exc)
        systemctl = ()
    if systemctl:
        try:
            completed = subprocess.run(
                [*systemctl, "--user", "--timestamp=unix", "show", unit_name,
                 "--property=ActiveEnterTimestamp", "--value"],
                capture_output=True, text=True, timeout=15, stdin=subprocess.DEVNULL,
                env=registry.systemd_user_bus_env(),
            )
            value = completed.stdout.strip().lstrip("@").strip()
            if value.isdigit():
                return int(value)
        except (OSError, subprocess.SubprocessError) as exc:
            _kb()._log.debug("kanban worker scopes: activation of %s unreadable (%s)", unit_name, exc)
    runtime_dir = registry.systemd_user_bus_env().get("XDG_RUNTIME_DIR")
    if not runtime_dir:
        return None
    try:
        return int((Path(runtime_dir) / "systemd" / "transient" / unit_name).stat().st_mtime)
    except OSError:
        return None


def _scope_activation_is_inside(unit_name: str, run) -> bool:
    """Whether *unit_name* was activated inside *run*'s own lifetime window.

    A worker scope cannot exist before its launch or, for a scope still standing,
    long after its attempt ended: an activation outside that window means the unit
    belongs to some other launch, so it is not this attempt's scope to stop.
    """
    activation = _scope_activation_epoch(unit_name)
    if activation is None:
        return False
    started = run["started_at"]
    if started is None:
        # Legacy row with no attempt start: nothing to compare against, so no proof.
        return False
    ended = int(run["ended_at"] or started)
    return (
        int(started) - _SCOPE_ATTEMPT_WINDOW_SECONDS
        <= activation
        <= ended + _SCOPE_ATTEMPT_WINDOW_SECONDS
    )


def _kanban_scope_owner(unit_name: str) -> Optional[dict[str, Any]]:
    """The attempt a Kanban worker scope unit names, or ``None`` if it names none.

    Inverse of :func:`_worker_scope_suffix`: reconciliation only ever acts on the
    exact unit a launch recorded, so the unit name has to answer "which attempt
    owns this scope" on its own. ``None`` means the caller must leave the unit
    alone -- stopping a scope kills every process in it, and an unrecognised
    suffix (another subsystem's scope, or a future shape) carries no evidence
    that a Hermes attempt owned it.
    """
    if not unit_name.startswith(WORKER_SCOPE_PREFIX) or not unit_name.endswith(WORKER_SCOPE_SUFFIX):
        return None
    suffix = unit_name[len(WORKER_SCOPE_PREFIX):-len(WORKER_SCOPE_SUFFIX)]
    for pattern in _KANBAN_SCOPE_SUFFIX_GRAMMAR:
        match = pattern.match(suffix)
        if match is None:
            continue
        owner: dict[str, Any] = {"task_id": match.group("task")}
        preparation_id = match.groupdict().get("preparation")
        if preparation_id is not None:
            owner["preparation_id"] = preparation_id
        else:
            owner["run_id"] = int(match.group("run"))
        return owner
    return None


def _orphan_scope_run(conn: sqlite3.Connection, owner: dict[str, Any]):
    """The ``task_runs`` row a live worker scope belongs to, or ``None``.

    A pre-claim launch names its ``preparation_id``, which is the only identity a
    worker has before the grant writes its run, so that form is matched against
    the metadata ``_set_worker_pid`` records: the ``LIKE`` only narrows the rows
    and exact JSON equality decides. No row means no attempt to prove finished,
    which is a reason to leave the scope alone.
    """
    if "run_id" in owner:
        return conn.execute(
            f"SELECT {_SCOPE_RUN_COLUMNS} FROM task_runs WHERE id = ? AND task_id = ?",
            (int(owner["run_id"]), owner["task_id"]),
        ).fetchone()
    preparation_id = str(owner["preparation_id"])
    for row in conn.execute(
        f"SELECT {_SCOPE_RUN_COLUMNS} FROM task_runs WHERE task_id = ? AND metadata LIKE ?",
        (owner["task_id"], f"%{preparation_id}%"),
    ):
        metadata = _kb()._json_dict(row["metadata"])
        if str(metadata.get("preparation_id") or "") == preparation_id:
            return row
    return None


def _run_worker_identity(run) -> "tuple[Optional[int], Any]":
    """``(pid, started_at)`` recorded for the worker of *run*.

    ``_set_worker_pid`` writes the run row's pair at spawn, and the worker's own
    handoff writes ``metadata["worker_pid"]`` beside the host epoch and start
    time; either is a spawn-time fingerprint the liveness check can compare, and
    two releases have left both shapes in the table. An unreadable identity is
    reported as unknown rather than assumed dead.
    """
    metadata = _kb()._json_dict(run["metadata"])
    identity = metadata.get("runtime_identity")
    identity = identity if isinstance(identity, dict) else {}
    pid = (
        metadata.get("worker_pid")
        or identity.get("worker_pid")
        or identity.get("pid")
        or run["worker_pid"]
    )
    if not pid:
        return None, None
    started_at: Any = run["worker_started_at"] or identity.get("worker_started_at")
    if started_at is None and metadata.get("host_epoch") and metadata.get("worker_start_time"):
        started_at = f"{metadata['host_epoch']}|{metadata['worker_start_time']}"
    if started_at is None:
        started_at = identity.get("start_time")
    return int(pid), started_at


def _scope_slot_free(cgroup: str) -> bool:
    """Whether *cgroup* no longer holds a process, directly or through a descendant.

    ``cgroup.events`` answers for the whole subtree, which is the evidence admission
    already uses, and a scope ``--collect`` already removed reads as free because its
    directory is gone. A cgroup that is still there but whose population cannot be
    read is reported as NOT free: the slot is only ever called free on evidence.
    """
    directory = CGROUP_ROOT / cgroup.lstrip("/")
    if not directory.is_dir():
        return True
    try:
        events = (directory / "cgroup.events").read_text(encoding="utf-8-sig")
    except OSError:
        return False
    for line in events.splitlines():
        key, _, value = line.partition(" ")
        if key == "populated":
            return value.strip() == "0"
    return False


def _record_scope_reclaim(
    conn: sqlite3.Connection, task_id: str, run_id: Optional[int], event: dict[str, Any],
) -> None:
    """Persist one bounded cleanup event against the attempt it belongs to.

    Only actions taken are recorded. A refusal changes nothing on the host, and
    writing one per live worker per tick would flood the board's own event trail;
    a failed stop IS recorded, because that is the state an operator has to act
    on.
    """
    with _kb().write_txn(conn):
        _kb()._append_event(conn, task_id, WORKER_SCOPE_RECLAIM_EVENT, event, run_id=run_id)


def _reconcile_worker_scope(
    conn: sqlite3.Connection,
    *,
    unit: str,
    cgroup: str,
    owner: dict[str, Any],
    workers_group: str,
    apply: bool,
    board: str = "",
) -> dict[str, Any]:
    """Decide one live scope: stop it, or say why not. Never stops on doubt."""
    task_id = str(owner["task_id"])

    def _event(
        reason: str, outcome: str, *, run_id: Optional[int] = None,
        pid: Optional[int] = None, slot_freed: bool = False, board: str = "",
    ) -> dict[str, Any]:
        event: dict[str, Any] = {
            "unit": unit,
            "cgroup": cgroup,
            "task_id": task_id,
            "run_id": run_id,
            "reason": reason,
            "outcome": outcome,
            "slot_freed": slot_freed,
        }
        if owner.get("preparation_id"):
            event["preparation_id"] = owner["preparation_id"]
        if pid:
            event["pid"] = pid
        if board:
            event["board"] = board
        return event

    if str(Path(cgroup).parent) != workers_group:
        # A survivor outside the worker slice is a placement fault for the
        # boundary check to report, not a slot this module may clear.
        return _event("scope_not_in_worker_slice", "refused")
    run = _orphan_scope_run(conn, owner)
    if run is None:
        return _event("run_not_recorded", "refused")
    run_id = int(run["id"])
    if run["ended_at"] is None or str(run["status"]) == "running":
        return _event("run_not_terminal", "refused", run_id=run_id)
    live_run = conn.execute(
        "SELECT id FROM task_runs WHERE task_id = ? AND ended_at IS NULL LIMIT 1", (task_id,),
    ).fetchone()
    if live_run is not None:
        return _event("newer_run_live", "refused", run_id=run_id)
    pid, started_at = _run_worker_identity(run)
    task_row = conn.execute(
        "SELECT worker_pid, worker_started_at FROM tasks WHERE id = ?", (task_id,),
    ).fetchone()
    if task_row is not None and task_row["worker_pid"]:
        if _dispatcher()._worker_alive(task_row["worker_pid"], task_row["worker_started_at"]):
            # The card itself still holds a live worker: this scope may be a
            # straggler's, but nothing here may prove whose, so it stays.
            return _event(
                "worker_alive", "refused", run_id=run_id, pid=int(task_row["worker_pid"]),
            )
    if not pid or pid == os.getpid():
        return _event("worker_identity_unknown", "refused", run_id=run_id)
    if _dispatcher()._worker_alive(pid, started_at):
        # A card can finish before its agent does: the run is closed while the
        # worker is still finalising its handoff. Its scope is still in use.
        return _event("worker_alive", "refused", run_id=run_id, pid=pid)
    run_epoch = str(_kb()._json_dict(run["metadata"]).get("host_epoch") or "")
    if run_epoch and _dispatcher()._instantiation_changed(run_epoch, _current_host_epoch()):
        # A transient scope cannot survive re-instantiation, so an attempt from a
        # previous boot cannot own a live one: the name match is a coincidence.
        return _event("stale_boot", "refused", run_id=run_id, pid=pid)
    recorded_board = str(_kb()._json_dict(run["metadata"]).get("board") or "")
    local_board = str(board or "")
    if recorded_board and local_board and recorded_board != local_board:
        # The attempt belongs to another board; its own dispatcher owns this scope.
        return _event("foreign_board", "refused", run_id=run_id, pid=pid, board=recorded_board)
    if not _scope_activation_is_inside(unit, run):
        # The unit name carries a task and a run/preparation identity but no board,
        # and board imports preserve those identities; the scope's own activation
        # inside this attempt's lifetime is what makes it this attempt's to stop.
        return _event("scope_not_from_attempt", "refused", run_id=run_id, pid=pid)
    if not apply:
        return _event(
            "run_ended_worker_gone", "would_reap", run_id=run_id, pid=pid, board=recorded_board,
        )

    from tools import process_registry as registry

    stopped = registry._stop_systemd_unit(unit)
    slot_freed = _scope_slot_free(cgroup) if stopped else False
    event = _event(
        "run_ended_worker_gone",
        "reaped" if stopped else "failed",
        run_id=run_id,
        pid=pid,
        slot_freed=slot_freed,
        board=recorded_board,
    )
    if not stopped:
        _kb()._log.warning(
            "kanban worker scopes: could not stop %s for task %s; the single worker "
            "slot stays held until it is stopped by hand", unit, task_id,
        )
    elif not slot_freed:
        _kb()._log.warning(
            "kanban worker scopes: %s still populates its cgroup after a stop; the "
            "worker slot is not free yet", unit,
        )
    else:
        _kb()._log.info(
            "kanban worker scopes: reaped orphaned scope %s (task %s, run %s) and freed "
            "the worker slot", unit, task_id, run_id,
        )
    _record_scope_reclaim(conn, task_id, run_id, event)
    return event


def reconcile_orphan_worker_scopes(
    conn: sqlite3.Connection, *, apply: bool = True, board: Optional[str] = None,
) -> "list[dict[str, Any]]":
    """Reap worker scopes left standing by an attempt that already ended (#57).

    A transient ``systemd-run --scope`` unit stays ACTIVE while anything at all
    remains in its cgroup -- ``--collect`` only removes it once the cgroup is
    empty -- so a worker-owned descendant that outlives the worker (the detached
    ``browser_harness`` daemon is the observed case) keeps its
    ``autonomous-workers.slice`` slot after the run is over. That slot is the
    native admission authority (``ConcurrencyHardMax=2``); two occupied slots
    stall queued cards until a scope empties. Nothing in the launch path tears
    the scope down, and a ``systemd-run`` client exits long before its worker
    does, so the dispatcher reconciles here instead: it runs where capacity is
    measured, which lets the same tick that frees a scope claim the next card.

    Stopping a scope kills every process in it, so every gate must pass on
    evidence: the unit must be a live ``hermes-worker-*.scope`` DIRECTLY in the
    worker slice, name an attempt recorded on this board, and that attempt's run
    must have ENDED while the worker identity recorded at spawn is provably gone
    (a recycled PID, a fingerprint from another boot epoch, or a PID that is
    simply absent all count as gone; an unreadable identity does not). The unit
    name carries no board and a board import preserves task and run identities, so
    the scope has to prove itself too: its own activation must fall inside that
    attempt's lifetime window, an attempt recorded against another instantiation
    or another board refuses, and neither an unreadable activation nor an
    unreadable population is ever read in this module's favour. A live worker, a
    closed run whose agent is still finalising its handoff, a newer live run for
    the card, a scope outside the worker slice, and a suffix this module cannot
    read all refuse instead: the scope is left exactly as found and reported, never
    stopped. The stop is therefore never the first signal a LIVE parent receives
    (``tools/AGENTS.md``): the recorded worker is already proven dead, recycled, or
    from a previous boot before it can run. Failures are fail-closed and reported,
    not retried with broader authority. One gap is deliberate: a launch that died
    during pre-claim preparation has no run row and no recorded identity, so its
    scope is refused as ``run_not_recorded`` rather than reaped on evidence that
    does not exist. That is an operator's stop of the exact unit named in the
    refusal, not a rule this module is entitled to widen -- a scope's only process
    could be the worker itself, and nothing in unit or cgroup metadata
    distinguishes the two.

    Returns bounded, sanitized events -- exact unit, cgroup, task, run, reason,
    outcome, and whether the slot came free -- for the dispatch result and the
    board's event trail. Nothing here reads command lines, arguments,
    credentials or browser data.
    """
    from tools import process_registry as registry

    try:
        inventory = registry.autonomous_boundary_inventory()
    except Exception as exc:
        # Unreadable boundary is not this scan's failure: no scope is stopped on
        # evidence that could not be gathered. The launch path refuses separately
        # with the operator-facing diagnostic.
        _kb()._log.debug("kanban worker scopes: boundary inventory unavailable (%s)", exc)
        return []
    workers_group = str((inventory.get("workers") or {}).get("cgroup") or "")
    try:
        local_board = _kb()._normalize_board_slug(board or _kb().get_current_board())
    except Exception:
        local_board = ""
    events: list[dict[str, Any]] = []
    for cgroup in inventory.get("worker_scopes") or []:
        unit = Path(str(cgroup)).name
        owner = _kanban_scope_owner(unit)
        if owner is None:
            continue  # not a Kanban worker scope: not ours to stop
        try:
            event = _reconcile_worker_scope(
                conn, unit=unit, cgroup=str(cgroup), owner=owner,
                workers_group=workers_group, apply=apply, board=local_board,
            )
        except Exception:
            # One unreadable scope must not cost the tick its other work.
            _kb()._log.debug("kanban worker scopes: reconciliation failed for %s", unit, exc_info=True)
            continue
        if event["outcome"] == "refused":
            if event["reason"] == "worker_identity_unknown":
                # A populated scope whose attempt is gone but whose worker identity
                # cannot be read is a slot only an operator can clear.
                _kb()._log.warning(
                    "kanban worker scopes: %s (task %s) holds the worker slot with an "
                    "unverifiable worker identity; stop that exact unit by hand",
                    unit, event["task_id"],
                )
            else:
                _kb()._log.debug(
                    "kanban worker scopes: leaving %s alone (%s)", unit, event["reason"],
                )
        events.append(event)
    return events


def _worker_project_plugins_enabled(env: dict[str, str]) -> bool:
    """Resolve only the discovery gate, without changing the dispatcher environment."""
    import codecs
    import io

    from dotenv.main import DotEnv
    from dotenv.variables import parse_variables
    from hermes_cli.env_loader import _sanitized_env_content
    from hermes_cli.managed_scope import get_managed_dir
    from utils import is_truthy_value

    values = dict(env)
    home = Path(values.get("HERMES_HOME") or Path.home() / ".hermes")
    managed = get_managed_dir()
    # Match load_hermes_dotenv's profile/1Password/managed ordering. The sealed
    # source excludes .env, so the CLI's installation-root fallback is absent.
    for path, override in (
        (home / ".env", True),
        (home / ".op.env", False),
        (managed / ".env" if managed is not None else None, True),
    ):
        if path is None or not path.is_file():
            continue
        if path.name == ".op.env" and values.get("OP_SERVICE_ACCOUNT_TOKEN"):
            continue
        raw = path.read_bytes()
        # .op.env is intentionally not sanitized by the child.
        content = _sanitized_env_content(raw) if path.name == ".env" else None
        if content is None:
            try:
                with io.TextIOWrapper(io.BytesIO(raw), encoding="utf-8-sig") as stream:
                    content = stream.read()
            except UnicodeDecodeError:
                content = raw.removeprefix(codecs.BOM_UTF8).decode("latin-1")
        parsed = DotEnv(None, stream=io.StringIO(content)).parse()
        resolved = {}
        for name, value in parsed:
            if value is not None:
                context = {**values, **resolved} if override else {**resolved, **values}
                value = "".join(atom.resolve(context) for atom in parse_variables(value))
            resolved[name] = value
        for name, value in resolved.items():
            if value is not None and (override or name not in values):
                values[name] = value
    return is_truthy_value(values.get("HERMES_ENABLE_PROJECT_PLUGINS"))


def _default_spawn(
    task: Task,
    workspace: str,
    *,
    board: Optional[str] = None,
    defer_grant: bool = False,
    should_stop: Optional[Callable[[], bool]] = None,
) -> WorkerLaunch | int:
    """Start a worker and verify its identity before granting Kanban access.

    ``should_stop`` is polled at every long step (before preparation, after
    runtime-generation preparation, before the scope is created, during the
    bootstrap waits) so a gateway that starts draining mid-launch hands nothing
    to a worker that would outlive it: :class:`WorkerLaunchInterrupted` unwinds
    and the prepared worker is cancelled.
    """
    _check_not_stopping(should_stop, task.id)
    if not task.assignee:
        raise ValueError(f"task {task.id} has no assignee")

    # Before ANY preparation: a host that cannot place an autonomous worker must not
    # burn a prepared runtime generation, a resolved workspace, or (via the caller)
    # a board claim on it. The scope wrapper re-checks at launch time, and the
    # bootstrap PID is verified again before the grant.
    from tools.process_registry import _IS_LINUX, require_autonomous_boundary

    if _IS_LINUX:
        require_autonomous_boundary(standalone_dispatch=True)

    from hermes_cli.profiles import normalize_profile_name, resolve_profile_env

    profile_arg = normalize_profile_name(task.assignee)

    from agent.secret_scope import is_multiplex_active
    from tools.environments.local import _is_routed_home, build_subprocess_env, strip_launch_profile_env

    try:
        profile_home = resolve_profile_env(profile_arg)
    except FileNotFoundError:
        profile_home = None
    routed = bool(profile_home) and _is_routed_home(profile_home)
    with (_worker_profile_scope(profile_home, bind_home=False) if profile_home else contextlib.nullcontext()):
        env = build_subprocess_env(
            scrub_secrets=is_multiplex_active() or routed,
            inherit_profile_home=True,
        )
    # The dispatcher is detached from every conversation; its worker must never
    # inherit routing mirrored by a previous gateway turn.
    from gateway.session_context import _VAR_MAP
    for key in _VAR_MAP:
        env.pop(key, None)

    # Inject HERMES_HOME so the worker reads the profile-scoped config.yaml:
    # without it the child's get_hermes_home() falls back to the DEFAULT
    # profile root because `hermes -p` applies its override before
    # hermes_constants is imported.
    if profile_home:
        env["HERMES_HOME"] = profile_home
        strip_launch_profile_env(env, profile_home)
    if task.tenant:
        env["HERMES_TENANT"] = task.tenant
    env["HERMES_KANBAN_TASK"] = task.id
    env["HERMES_KANBAN_WORKSPACE"] = workspace
    env["HERMES_SESSION_SOURCE"] = "kanban"
    if workspace and os.path.isabs(workspace) and os.path.isdir(workspace):
        env["TERMINAL_CWD"] = workspace
    if task.branch_name:
        env["HERMES_KANBAN_BRANCH"] = task.branch_name
    if task.current_run_id is not None:
        env["HERMES_KANBAN_RUN_ID"] = str(task.current_run_id)
    if task.claim_lock:
        env["HERMES_KANBAN_CLAIM_LOCK"] = task.claim_lock
    if task.goal_mode:
        env["HERMES_KANBAN_GOAL_MODE"] = "1"
        if task.goal_max_turns is not None:
            env["HERMES_KANBAN_GOAL_MAX_TURNS"] = str(int(task.goal_max_turns))
    for var in ("TERMINAL_TIMEOUT", "TERMINAL_MAX_FOREGROUND_TIMEOUT"):
        override = _dispatcher()._worker_terminal_timeout_env(
            task.max_runtime_seconds, env.get(var),
        )
        if override is not None:
            env[var] = override
    env["HERMES_KANBAN_DB"] = str(_kb().kanban_db_path(board=board))
    env["HERMES_KANBAN_WORKSPACES_ROOT"] = str(_kb().workspaces_root(board=board))
    _retag_legacy_worker_sessions(env["HERMES_KANBAN_WORKSPACES_ROOT"])
    env["HERMES_KANBAN_BOARD"] = _kb()._normalize_board_slug(board) or _kb().get_current_board()
    env["HERMES_PROFILE"] = profile_arg
    from agent.delegation_context import DELEGATED_CHILD_ENV_MARKER
    env.pop(DELEGATED_CHILD_ENV_MARKER, None)
    env.pop("HERMES_TUI", None)

    from hermes_cli.kanban_runtime import (
        assert_runtime_import_root,
        decode_identity,
        encode_identity,
        prospective_identity,
        verify_worker_ready,
    )
    from hermes_cli.kanban_runtime_generation import (
        cleanup_runtime_generation,
        prepare_runtime_generation,
        sweep_runtime_generations,
        write_runtime_generation_owner,
    )
    _validate_worker_entrypoint()
    expected_identity = prospective_identity()
    assert_runtime_import_root(identity=expected_identity)
    preparation_id = uuid.uuid4().hex
    preparation_path = (
        _kb().kanban_home() / "kanban" / "runtime-preparations"
        / f"{task.id}-{preparation_id}.json"
    )
    env["HERMES_KANBAN_BOOTSTRAP_PATH"] = str(preparation_path)
    env["HERMES_KANBAN_PREPARATION_ID"] = preparation_id
    env["HERMES_KANBAN_EXPECTED_RUNTIME"] = encode_identity(expected_identity)
    env["HERMES_KANBAN_BOOTSTRAP_WAIT"] = "1"
    env["HERMES_KANBAN_RUNTIME_FENCE"] = "1"

    dispatcher = _dispatcher()
    cli_args = dispatcher._worker_argv(task, profile_arg, env.get("HERMES_HOME"))
    worker_cwd = workspace if os.path.isdir(workspace) else os.getcwd()
    project_plugins_enabled = _worker_project_plugins_enabled(env)
    env["HERMES_ENABLE_PROJECT_PLUGINS"] = "1" if project_plugins_enabled else "0"
    generation = prepare_runtime_generation(
        expected_identity, workspace=worker_cwd, profile_home=env.get("HERMES_HOME"),
        project_plugins_enabled=project_plugins_enabled,
    )
    expected_identity = generation.identity
    # One walk per launch: both phases re-hash the same sealed payload.
    phase_timeout = _bootstrap_phase_timeout(generation.root)
    env.update(generation.env)
    env["HERMES_KANBAN_EXPECTED_RUNTIME"] = encode_identity(expected_identity)
    proc = None
    log_f = None
    actual = None

    def _close_resources() -> None:
        with contextlib.suppress(Exception):
            if proc is not None and proc.stdin is not None:
                proc.stdin.close()
        with contextlib.suppress(OSError):
            preparation_path.unlink()
        with contextlib.suppress(Exception):
            log_f.close()

    def _cancel() -> None:
        # A systemd scope launcher can exit while its worker survives. Signal
        # only the verified worker identity, and let owner-aware cleanup decide
        # when its generation is no longer in use.
        if actual is not None and proc is not None and actual.pid != proc.pid:
            with contextlib.suppress(Exception):
                from hermes_cli.kanban_runtime import process_start_time
                import signal

                if process_start_time(actual.pid) == actual.start_time:
                    os.kill(actual.pid, signal.SIGTERM)
        with contextlib.suppress(Exception):
            if proc is not None:
                proc.terminate()
        with contextlib.suppress(Exception):
            if proc is not None:
                proc.wait(timeout=2)
        if proc is not None and proc.poll() is not None:
            _worker_pid_aliases.pop(proc.pid, None)
            _worker_processes.pop(proc.pid, None)
            snapshot = _worker_runtime_snapshots.pop(proc.pid, None)
            with contextlib.suppress(Exception):
                cleanup_runtime_generation(snapshot or generation.root)
        elif proc is None:
            cleanup_runtime_generation(generation.root, force=True)
        _close_resources()

    try:
        # After the expensive runtime-generation preparation and before any
        # scope exists: a drain observed here costs one prepared generation and
        # nothing else.
        _check_not_stopping(should_stop, task.id)
        deferred_preparation_id = preparation_id if defer_grant else None
        worker_scope_unit = (
            f"{WORKER_SCOPE_PREFIX}"
            f"{_worker_scope_suffix(task, preparation_id=deferred_preparation_id)}"
            f"{WORKER_SCOPE_SUFFIX}"
        )
        cmd = dispatcher._restart_safe_worker_argv(
            task, generation.command_prefix + cli_args,
            preparation_id=deferred_preparation_id,
        )
        from tools.process_registry import systemd_user_bus_env
        env = systemd_user_bus_env(env)
        log_f = dispatcher._open_worker_log(task, board)
        proc = subprocess.Popen(  # noqa: S603 -- argv is a fixed list built above
            cmd,
            cwd=worker_cwd,
            stdin=subprocess.PIPE,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
            creationflags=subprocess.CREATE_NO_WINDOW if _kb()._IS_WINDOWS else 0,
        )
        # A fake process in legacy unit tests has only ``pid``. Keep those
        # tests focused on environment construction without weakening real
        # process fencing.
        if not hasattr(proc, "poll") or not hasattr(proc, "stdin"):
            _close_resources()
            return proc.pid
        _worker_processes[proc.pid] = proc
        _worker_runtime_snapshots[proc.pid] = generation.root
        log_f.close()
        deadline = time.monotonic() + phase_timeout
        payload = None
        while time.monotonic() < deadline:
            _check_not_stopping(should_stop, task.id)
            if preparation_path.is_file():
                try:
                    payload = json.loads(preparation_path.read_text(encoding="utf-8-sig"))
                except (OSError, json.JSONDecodeError):
                    payload = None
                if payload is not None:
                    break
            if proc.poll() is not None:
                _raise_if_launch_refused(worker_scope_unit, proc.returncode)
                raise RuntimeError(f"worker exited before bootstrap ({proc.returncode})")
            time.sleep(0.05)
        if payload is None:
            raise RuntimeError("worker bootstrap timed out")
        early_pid = decode_identity(payload.get("runtime_identity")).pid
        early = verify_worker_ready(
            payload, expected_identity, pid=early_pid, preparation_id=preparation_id,
        )
        actual = early
        _worker_pid_aliases[proc.pid] = early.pid
        if _IS_LINUX:
            # The worker must already be in its own scope inside the autonomous
            # worker slice: continuing its imports inside the gateway's cgroup
            # would put an autonomous worker next to the interactive session.
            from tools.autonomous_resources import check_autonomous_worker

            check_autonomous_worker(early.pid)
        write_runtime_generation_owner(
            generation.root, pid=early.pid, start_time=early.start_time,
        )
        if proc.stdin is None:
            raise RuntimeError("worker bootstrap pipe unavailable")
        proc.stdin.write(json.dumps({
            "continue_imports": True,
            "preparation_id": preparation_id,
            "runtime_identity": early.as_dict(),
        }, sort_keys=True).encode("utf-8") + b"\n")
        proc.stdin.flush()

        post_deadline = time.monotonic() + phase_timeout
        post_payload = None
        while time.monotonic() < post_deadline:
            _check_not_stopping(should_stop, task.id)
            if preparation_path.is_file():
                try:
                    candidate = json.loads(preparation_path.read_text(encoding="utf-8-sig"))
                except (OSError, json.JSONDecodeError):
                    candidate = None
                if isinstance(candidate, dict) and candidate.get("post_import") is True:
                    post_payload = candidate
                    break
            if proc.poll() is not None:
                raise RuntimeError(f"worker exited before post-import verification ({proc.returncode})")
            time.sleep(0.05)
        if post_payload is None:
            raise RuntimeError("worker post-import verification timed out")
        ready_pid = decode_identity(post_payload.get("runtime_identity")).pid
        post_actual = verify_worker_ready(
            post_payload, expected_identity, pid=ready_pid, preparation_id=preparation_id,
        )
        if post_actual.pid != early.pid or post_actual.start_time != early.start_time:
            raise RuntimeError("worker process identity changed during bootstrap")
        actual = post_actual
        _worker_pid_aliases[proc.pid] = actual.pid

        sweep_runtime_generations()
        grant_fd = proc.stdin.fileno()
        if os.name == "nt" and sys.version_info < (3, 12):
            _set_windows_pipe_nonblocking(grant_fd)
        else:
            os.set_blocking(grant_fd, False)
        # Encode the potentially large identity before entering the grant lock.
        grant_prefix = (json.dumps({
            "grant": True,
            "preparation_id": preparation_id,
            "runtime_identity": actual.as_dict(),
        }, sort_keys=True)[:-1] + ", ").encode("utf-8")
        def _grant(run_id: int, claim_lock: Optional[str]) -> None:
            if proc is None or proc.stdin is None:
                raise RuntimeError("worker bootstrap pipe unavailable")
            if _IS_LINUX:
                # Last barrier before the worker may touch Kanban: the claim is
                # already taken, so a boundary that drifted during bootstrap must
                # stop the grant here. The caller releases the ungranted claim and
                # keeps the task identity intact.
                from tools.autonomous_resources import check_autonomous_worker

                check_autonomous_worker(actual.pid)
            suffix = json.dumps({"run_id": run_id, "claim_lock": claim_lock})[1:].encode("utf-8")
            _write_worker_grant(grant_fd, grant_prefix + suffix + b"\n")
            # The buffered import handshake was already flushed. Only close an
            # empty, now-nonblocking pipe here; filesystem cleanup is child-owned.
            with contextlib.suppress(Exception):
                proc.stdin.close()

        if defer_grant:
            return WorkerLaunch(
                actual.pid, actual.as_dict(), preparation_id, grant=_grant, cancel=_cancel,
                launcher_pid=proc.pid,
            )
        _grant(int(task.current_run_id or 0), task.claim_lock)
        return WorkerLaunch(
            actual.pid, actual.as_dict(), preparation_id, launcher_pid=proc.pid,
        )
    except FileNotFoundError:
        _cancel()
        raise RuntimeError(
            "`hermes` executable not found on PATH. "
            "Install Hermes Agent or activate its venv before running the kanban dispatcher."
        )
    except Exception as exc:
        _cancel()
        # Classified here, where the scope route or bootstrap actually failed.
        _launch_failure_or_shutdown(exc, task.id)


def _pid_alive(pid: Optional[int]) -> bool:
    """Return True if ``pid`` is still running on this host.

    Uses ``gateway.status._pid_exists`` (OpenProcess on Windows, ``os.kill(pid, 0)``
    on POSIX). **DO NOT** call ``os.kill(pid, 0)`` directly on Windows — there
    ``sig=0`` is ``CTRL_C_EVENT`` broadcast to the console group, potentially
    killing unrelated processes.

    Zombies (exited, not yet reaped) still pass the existence check, so a
    worker would look "alive" forever between exit and reap. Linux: peek at
    ``/proc/<pid>/status`` and treat ``State: Z`` as dead; macOS: ask ``ps``
    for the BSD ``stat`` field and treat ``Z`` as dead.
    """
    if not pid or pid <= 0:
        return False
    from gateway.status import _pid_exists
    if not _pid_exists(int(pid)):
        return False
    if sys.platform == "linux":
        try:
            with open(f"/proc/{int(pid)}/status", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("State:"):
                        # "State:\tZ (zombie)" → dead
                        if "Z" in line.split(":", 1)[1]:
                            return False
                        break
        except (FileNotFoundError, PermissionError, OSError):
            # proc entry gone → already reaped; treat as dead.
            pass
    elif sys.platform == "darwin":
        try:
            proc = subprocess.run(
                ["ps", "-o", "stat=", "-p", str(int(pid))],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True, encoding='utf-8', errors='replace',
                timeout=1,
                check=False,
            )
            if proc.returncode != 0:
                return False
            if "Z" in (proc.stdout or "").strip():
                return False
        except (OSError, subprocess.SubprocessError, TimeoutError):
            # If the secondary probe fails, keep the kill(0) answer.
            pass
    return True
def _positive_int(value: Any, default: int, *, minimum: int = 1) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= minimum else default


def worker_log_rotation_config(kanban_cfg: Optional[dict] = None) -> tuple[int, int]:
    """Return ``(rotate_bytes, backup_count)`` for worker log rotation.
    Defaults: rotate at 2 MiB, keep one backup (``.log.1``); both overridable
    from ``config.yaml``.
    """
    if kanban_cfg is None:
        try:
            from hermes_cli.config import load_config

            kanban_cfg = (load_config().get("kanban") or {})
        except Exception:
            kanban_cfg = {}
    kanban_cfg = kanban_cfg or {}
    max_bytes = _positive_int(
        kanban_cfg.get("worker_log_rotate_bytes"),
        DEFAULT_LOG_ROTATE_BYTES,
        minimum=1,
    )
    backup_count = _positive_int(
        kanban_cfg.get("worker_log_backup_count"),
        DEFAULT_LOG_BACKUP_COUNT,
        minimum=0,
    )
    return max_bytes, backup_count


def _rotated_log_path(log_path: Path, generation: int) -> Path:
    return log_path.with_suffix(log_path.suffix + f".{generation}")


def _rotate_worker_log(
    log_path: Path,
    max_bytes: int,
    backup_count: int = DEFAULT_LOG_BACKUP_COUNT,
) -> None:
    """Rotate ``<log>`` when it exceeds ``max_bytes``: ``<log>`` → ``<log>.1``,
    older generations shift up to ``backup_count``.
    """
    try:
        if not log_path.exists() or log_path.stat().st_size <= max_bytes:
            return
        backup_count = _positive_int(
            backup_count, DEFAULT_LOG_BACKUP_COUNT, minimum=0,
        )
        if backup_count == 0:
            log_path.unlink()
            return
        oldest = _rotated_log_path(log_path, backup_count)
        with contextlib.suppress(OSError):
            if oldest.exists():
                oldest.unlink()
        for generation in range(backup_count - 1, 0, -1):
            src = _rotated_log_path(log_path, generation)
            if not src.exists():
                continue
            with contextlib.suppress(OSError):
                src.rename(_rotated_log_path(log_path, generation + 1))
        log_path.rename(_rotated_log_path(log_path, 1))
    except OSError:
        pass

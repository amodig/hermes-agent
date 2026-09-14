"""Hermes Kanban worker launch, bootstrap, and process-runtime management.

The dispatcher owns board scheduling; this module owns the subprocess boundary,
worker identity handoff, and runtime snapshot cleanup.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
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

if TYPE_CHECKING:
    from hermes_cli.kanban_db import Task


def _dispatcher():
    from hermes_cli import kanban_db_dispatch

    return kanban_db_dispatch

DEFAULT_LOG_ROTATE_BYTES = 2 * 1024 * 1024
DEFAULT_LOG_BACKUP_COUNT = 1


@dataclass(frozen=True)
class WorkerLaunch:
    """Child process identity verified before its Kanban grant."""

    pid: int
    runtime_identity: dict[str, Any]
    preparation_id: str
    grant: Optional[Callable[[int, Optional[str]], None]] = None
    cancel: Optional[Callable[[], None]] = None
    launcher_pid: Optional[int] = None


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
"""Filesystem snapshots retained until their launcher process is reaped."""


def _record_worker_exit(pid: int, raw_status: int) -> None:
    """Record a reaped worker exit under its verified Hermes PID."""
    launcher_pid = int(pid)
    snapshot = _worker_runtime_snapshots.pop(launcher_pid, None)
    if snapshot is not None:
        with contextlib.suppress(Exception):
            from hermes_cli.kanban_runtime import cleanup_runtime_snapshot

            cleanup_runtime_snapshot(snapshot)
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
    try:
        if os.WIFEXITED(raw):
            code = os.WEXITSTATUS(raw)
            if code == 0:
                return ("clean_exit", 0)
            if code == _kb.KANBAN_RATE_LIMIT_EXIT_CODE:
                return ("rate_limited", code)
            return ("nonzero_exit", code)
        if os.WIFSIGNALED(raw):
            return ("signaled", os.WTERMSIG(raw))
    except Exception:
        pass
    return ("unknown", None)


def reap_worker_zombies() -> "list[int]":
    """Reap exited workers without blocking; poll retained handles on Windows."""
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
def _trusted_module_hermes_argv(command: list[str]) -> list[str]:
    """Launch the module fallback from the dispatcher runtime root."""
    module_argv = _module_hermes_argv()
    if command[:3] != module_argv:
        return command
    runtime_root = str(Path(__file__).resolve().parents[1])
    bootstrap = (
        "import runpy,sys;"
        f"sys.path.insert(0,{runtime_root!r});"
        "runpy.run_module('hermes_cli.main',run_name='__main__')"
    )
    return [sys.executable, "-I", "-c", bootstrap, *command[3:]]



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
    if not _kb._IS_WINDOWS or os.path.splitext(command)[1]:
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
            if os.path.isfile(candidate) and (_kb._IS_WINDOWS or os.access(candidate, os.X_OK)):
                return candidate
    return None


def _hermes_path_argv(path: str) -> list[str]:
    """argv for a resolved Hermes executable path. Windows batch shims
    (``.cmd``/``.bat``) are unsafe as argv[0] because the argument vector
    includes task-derived values; prefer the module form."""
    if _kb._IS_WINDOWS and _is_windows_batch_shim(path):
        return _module_hermes_argv()
    return [_absolute_hermes_path(path)]


def _resolve_hermes_argv() -> list[str]:
    """Resolve the ``hermes`` invocation as argv for ``Popen``: ``$HERMES_BIN``
    (path-like -> absolute; bare names keep PATH semantics, never a
    same-directory file), then ``which("hermes")`` (Windows: safe PATH search,
    batch shims fall back to the module form), then ``sys.executable -m
    hermes_cli.main`` for shim-less environments (cron, systemd ``User=``,
    launchd). Mirrors ``gateway.run._resolve_hermes_bin``; local because
    ``hermes_cli`` sits below ``gateway`` in the dependency order.
    """
    import shutil

    env_bin = os.environ.get("HERMES_BIN", "").strip()
    if env_bin:
        if _looks_like_path(env_bin):
            return _hermes_path_argv(env_bin)
        resolved_env_bin = _safe_which_no_cwd(env_bin)
        if resolved_env_bin:
            return _hermes_path_argv(resolved_env_bin)
        return _module_hermes_argv()

    hermes_bin = _safe_which_no_cwd("hermes") if _kb._IS_WINDOWS else shutil.which("hermes")
    if hermes_bin:
        return _hermes_path_argv(hermes_bin)
    return _module_hermes_argv()


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
        from hermes_constants import reset_hermes_home_override, set_hermes_home_override
        from hermes_cli.config import load_config
        from hermes_cli.tools_config import _get_platform_tools

        token = set_hermes_home_override(hermes_home)
        try:
            cfg = load_config()
            toolsets = sorted(_get_platform_tools(cfg, "cli"))
        finally:
            reset_hermes_home_override(token)
        return toolsets or None
    except Exception as exc:
        _kb._log.debug(
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
        from hermes_state import SessionDB

        db = SessionDB()
        try:
            db.retag_kanban_worker_sessions(workspaces_root_path)
        finally:
            db.close()
        _retagged_workspace_roots.add(workspaces_root_path)
    except Exception as exc:
        _kb._log.debug("kanban worker: legacy session retag skipped (%s)", exc)


def _worker_argv(task: Task, profile_arg: str, hermes_home: Optional[str]) -> list[str]:
    """Build the ``hermes -p <profile> --cli ... chat -q ...`` worker command."""
    dispatcher = _dispatcher()
    cmd = dispatcher._trusted_module_hermes_argv(dispatcher._resolve_hermes_argv())
    cmd.extend([
        "-p", profile_arg,
        # A worker must NEVER boot the interactive TUI: its no-TTY bail-out
        # exits 0 without doing the task → "protocol violation" every attempt.
        "--cli",
        # Workers run under a profile-scoped HERMES_HOME and so see that
        # profile's shell-hook allowlist; pass --accept-hooks explicitly so
        # configured hooks still register.
        "--accept-hooks",
    ])
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
    if task.goal_mode:
        # The kanban goal-loop hook only runs in cli.py's fully-quiet branch.
        # Without -Q the worker gets one turn, prints text, exits rc=0, and the
        # dispatcher records a protocol violation.
        cmd.append("-Q")
    return cmd


def _open_worker_log(task: Task, board: Optional[str]):
    """Append-mode per-task log (a re-run on unblock appends, never overwrites),
    rotated first. Anchored at the board root (not the shared kanban root) so
    `hermes kanban log` reads its own file and boards sharing task ids don't
    collide."""
    log_dir = _kb.worker_logs_dir(board=board)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{task.id}.log"
    dispatcher = _dispatcher()
    rotate_bytes, backup_count = dispatcher.worker_log_rotation_config()
    dispatcher._rotate_worker_log(log_path, rotate_bytes, backup_count)
    return open(log_path, "ab")
def _restart_safe_worker_argv(
    task: Task, command: list[str], *, preparation_id: Optional[str] = None,
) -> list[str]:
    """Wrap a managed-gateway worker in the shared restart-safe scope."""
    from tools.process_registry import restart_safe_gateway_child_argv

    if task.current_run_id is None:
        # Pre-claim workers use the preparation id as their temporary scope;
        # the grant binds the eventual run before Kanban tools are available.
        suffix = (
            f"kanban-{task.id}-preparation-{preparation_id}"
            if preparation_id
            else f"kanban-{task.id}-run-missing"
        )
        scoped = restart_safe_gateway_child_argv(command, unit_suffix=suffix)
        if scoped is not command and not preparation_id:
            raise RuntimeError(
                "cannot create restart-safe systemd scope for Kanban worker: "
                "the claimed task has no current run id"
            )
        return scoped

    return restart_safe_gateway_child_argv(
        command,
        unit_suffix=f"kanban-{task.id}-run-{task.current_run_id}",
    )


def _default_spawn(
    task: Task,
    workspace: str,
    *,
    board: Optional[str] = None,
    defer_grant: bool = False,
) -> WorkerLaunch | int:
    """Start a worker and verify its identity before granting Kanban access."""
    if not task.assignee:
        raise ValueError(f"task {task.id} has no assignee")

    from hermes_cli.profiles import normalize_profile_name, resolve_profile_env

    profile_arg = normalize_profile_name(task.assignee)

    from agent.secret_scope import is_multiplex_active
    from tools.environments.local import build_subprocess_env

    env = build_subprocess_env(
        scrub_secrets=is_multiplex_active(),
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
    try:
        env["HERMES_HOME"] = resolve_profile_env(profile_arg)
    except FileNotFoundError:
        # No profile dir (isolated test fixtures) — the CLI resolves it from
        # HERMES_PROFILE (set below) instead.
        pass
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
    env["HERMES_KANBAN_DB"] = str(_kb.kanban_db_path(board=board))
    env["HERMES_KANBAN_WORKSPACES_ROOT"] = str(_kb.workspaces_root(board=board))
    _retag_legacy_worker_sessions(env["HERMES_KANBAN_WORKSPACES_ROOT"])
    env["HERMES_KANBAN_BOARD"] = _kb._normalize_board_slug(board) or _kb.get_current_board()
    env["HERMES_PROFILE"] = profile_arg
    env.pop("HERMES_TUI", None)

    from hermes_cli.kanban_runtime import (
        _sweep_runtime_snapshots,
        _write_runtime_snapshot_owner,
        assert_runtime_import_root,
        cleanup_runtime_snapshot,
        decode_identity,
        encode_identity,
        prospective_identity,
        verify_worker_ready,
    )
    expected_identity = prospective_identity()
    assert_runtime_import_root(identity=expected_identity)
    preparation_id = uuid.uuid4().hex
    preparation_path = (
        _kb.kanban_home() / "kanban" / "runtime-preparations"
        / f"{task.id}-{preparation_id}.json"
    )
    env["HERMES_KANBAN_BOOTSTRAP_PATH"] = str(preparation_path)
    env["HERMES_KANBAN_PREPARATION_ID"] = preparation_id
    env["HERMES_KANBAN_EXPECTED_RUNTIME"] = encode_identity(expected_identity)
    env["HERMES_KANBAN_BOOTSTRAP_WAIT"] = "1"
    env["HERMES_KANBAN_RUNTIME_FENCE"] = "1"

    dispatcher = _dispatcher()
    cmd = dispatcher._worker_argv(task, profile_arg, env.get("HERMES_HOME"))
    # A pre-claim worker cannot have a run id yet, so use its preparation id
    # as the temporary systemd scope. The durable grant binds the real run.
    cmd = dispatcher._restart_safe_worker_argv(
        task, cmd, preparation_id=preparation_id if defer_grant else None,
    )
    log_f = dispatcher._open_worker_log(task, board)
    proc = None
    snapshot_path: Optional[Path] = None

    def _close_resources() -> None:
        with contextlib.suppress(Exception):
            if proc is not None and proc.stdin is not None:
                proc.stdin.close()
        with contextlib.suppress(OSError):
            preparation_path.unlink()
        with contextlib.suppress(Exception):
            log_f.close()

    def _cancel() -> None:
        nonlocal snapshot_path
        with contextlib.suppress(Exception):
            if proc is not None:
                proc.terminate()
        with contextlib.suppress(Exception):
            if proc is not None:
                proc.wait(timeout=2)
        if proc is not None and proc.poll() is not None:
            _worker_pid_aliases.pop(proc.pid, None)
            _worker_processes.pop(proc.pid, None)
            snapshot = _worker_runtime_snapshots.pop(proc.pid, None) or snapshot_path
            with contextlib.suppress(Exception):
                cleanup_runtime_snapshot(snapshot)
            snapshot_path = None
        _close_resources()

    try:
        proc = subprocess.Popen(  # noqa: S603 -- argv is a fixed list built above
            cmd,
            cwd=workspace if os.path.isdir(workspace) else None,
            stdin=subprocess.PIPE,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
            creationflags=subprocess.CREATE_NO_WINDOW if _kb._IS_WINDOWS else 0,
        )
        # A fake process in legacy unit tests has only ``pid``. Keep those
        # tests focused on environment construction without weakening real
        # process fencing.
        if not hasattr(proc, "poll") or not hasattr(proc, "stdin"):
            _close_resources()
            return proc.pid
        log_f.close()
        deadline = time.monotonic() + 10.0
        payload = None
        while time.monotonic() < deadline:
            if preparation_path.is_file():
                try:
                    payload = json.loads(preparation_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    payload = None
                if payload is not None:
                    break
            if proc.poll() is not None:
                raise RuntimeError(f"worker exited before bootstrap ({proc.returncode})")
            time.sleep(0.05)
        if payload is None:
            raise RuntimeError("worker bootstrap timed out")
        early_pid = decode_identity(payload.get("runtime_identity")).pid
        early = verify_worker_ready(
            payload, expected_identity, pid=early_pid, preparation_id=preparation_id,
        )
        if proc.stdin is None:
            raise RuntimeError("worker bootstrap pipe unavailable")
        proc.stdin.write(json.dumps({
            "continue_imports": True,
            "preparation_id": preparation_id,
            "runtime_identity": early.as_dict(),
        }, sort_keys=True).encode("utf-8") + b"\n")
        proc.stdin.flush()

        post_deadline = time.monotonic() + 30.0
        post_payload = None
        while time.monotonic() < post_deadline:
            if preparation_path.is_file():
                try:
                    candidate = json.loads(preparation_path.read_text(encoding="utf-8"))
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
        actual = verify_worker_ready(
            post_payload, expected_identity, pid=ready_pid, preparation_id=preparation_id,
        )
        raw_snapshot = post_payload.get("runtime_snapshot")
        if isinstance(raw_snapshot, str) and raw_snapshot:
            snapshot_path = Path(raw_snapshot)
        _worker_processes[proc.pid] = proc
        _worker_pid_aliases[proc.pid] = actual.pid

        if snapshot_path is not None:
            _write_runtime_snapshot_owner(
                snapshot_path, pid=actual.pid, start_time=actual.start_time,
            )
            _worker_runtime_snapshots[proc.pid] = snapshot_path
        _sweep_runtime_snapshots()
        def _grant(run_id: int, claim_lock: Optional[str]) -> None:
            if proc is None or proc.stdin is None:
                raise RuntimeError("worker bootstrap pipe unavailable")
            proc.stdin.write(json.dumps({
                "grant": True,
                "preparation_id": preparation_id,
                "runtime_identity": actual.as_dict(),
                "run_id": run_id,
                "claim_lock": claim_lock,
            }, sort_keys=True).encode("utf-8") + b"\n")
            proc.stdin.flush()
            proc.stdin.close()
            with contextlib.suppress(OSError):
                preparation_path.unlink()

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
    except Exception:
        _cancel()
        raise

from hermes_cli import kanban_db as _kb
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

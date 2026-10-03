"""Dispatcher: crash/stale/orphan detection, failure accounting and the respawn circuit breaker, memory-aware concurrency caps, the one-shot ``dispatch_once`` pass, worker spawning (``_default_spawn``), worker-log rotation and the long-lived ``run_daemon`` loop.

Split out of ``hermes_cli.kanban_db``; origin-resident helpers are reached
late-bound via ``_kb`` (import-cycle breaking) so monkeypatching
``kanban_db.<name>`` keeps working.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shlex
import signal
import sqlite3
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from dataclasses import replace
from dataclasses import field
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Iterable
from typing import Mapping
from typing import Optional
from typing import TYPE_CHECKING
import uuid

from hermes_cli.quiet_single_query import KANBAN_WORKER_EXIT_TRAILER

if TYPE_CHECKING:
    from hermes_cli.kanban_db import Task


# After this many consecutive non-success attempts on a task/profile the
# dispatcher parks the task in ``blocked`` with a reason — prevents retry storms.
DEFAULT_FAILURE_LIMIT = 2

# Worker log files larger than this at spawn time are rotated.
DEFAULT_LOG_ROTATE_BYTES = 2 * 1024 * 1024   # 2 MiB
DEFAULT_LOG_BACKUP_COUNT = 1

# Keep a little wall-clock budget for the worker to observe a terminal timeout
# and make a terminal board call (kanban_block/kanban_complete/kanban_request_review)
# before max_runtime_seconds kills it.
KANBAN_TERMINAL_TIMEOUT_GRACE_SECONDS = 30

# A healthy worker is still alive for a while after kanban_complete /
# kanban_request_review returns (final assistant turn, session persistence), so
# a run's retained worker is only reaped once ended_at is at least this old
# (two default dispatch ticks).
TERMINAL_WORKER_REAP_GRACE_SECONDS = 120

# ---------------------------------------------------------------------------
# Respawn guard constants
# ---------------------------------------------------------------------------

# Patterns in last_failure_error that indicate a quota / auth blocker.
# These errors won't resolve by retrying immediately — auto-block instead.
# The auth family is a curated list, not an open `auth\w*` stem: that stem
# also matched ordinary English words like "author"/"authored"/"authoring"/
# "authoritative" in worker progress prose, parking a healthy card forever
# (#117009).
_RESPAWN_BLOCKER_RE = re.compile(
    r"\b(quota|rate[\s_\-]?limit|429|403|"
    r"auth|authenticat(?:e|es|ed|ing|ion)|authoriz(?:e|es|ed|ing|ation)|"
    r"authoris(?:e|es|ed|ing|ation)|authz|"
    r"unauthorized|forbidden|billing|subscription|"
    r"access[\s_]denied|permission[\s_]denied|"
    r"invalid[\s_]api[\s_]key)\b",
    re.IGNORECASE,
)

# Within this window a completed run counts as "recent proof"; don't re-spawn.
_RESPAWN_GUARD_SUCCESS_WINDOW = 3600  # 1 hour

# Cooldown after a rate-limited (quota-wall) requeue before re-spawning. Without
# it the task would re-spawn on the very next tick and bounce off the same quota
# wall, burning a worker slot every tick for hours. Overridable via
# ``HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS``.
DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS = 300  # 5 minutes

# Within this window a GitHub PR URL in a comment blocks re-spawn.
_RESPAWN_GUARD_PR_WINDOW = 86400  # 24 hours

_RESPAWN_GUARD_PR_URL_RE = re.compile(
    r"https?://github\.com/[^/\s]+/[^/\s]+/pull/\d+",
    re.IGNORECASE,
)


@dataclass
class DispatchResult:
    """Outcome of a single ``dispatch`` pass.

    ``kanban.default_assignee`` applied this tick before spawning (#27145). Surfaces the auto-assignment to
    telemetry / CLI / dashboard so the operator can see when the dispatcher is acting on the fallback rule
    ``kanban.max_in_progress_per_profile`` (#21582). Each entry is ``(task_id, assignee,
    current_running_count)``. NOT an operator-actionable failure — the task will be picked up on a
    subsequent tick when the assignee has capacity. Separate bucket so telemetry / dashboards can show "this
    profile is busy" vs
    the board's dispatch lock (issue #35240). A losing dispatcher does no DB writes this tick — the lock
    holder is making progress on the same board. This is the steady-state signal that a single-writer guard
    is
    """

    reclaimed: int = 0
    promoted: int = 0
    reconciled_orphans: list[str] = field(default_factory=list)
    """``running`` cards requeued by :func:`reconcile_orphaned_running` (broken
    claim bookkeeping, dead/gone worker)."""
    reaped_terminal_workers: list[str] = field(default_factory=list)
    """Task ids whose worker outlived its closed run and was terminated by
    :func:`reap_terminal_workers`."""
    spawned: list[tuple[str, str, str]] = field(default_factory=list)
    """``(task_id, assignee, workspace_path)`` triples."""
    skipped_unassigned: list[str] = field(default_factory=list)
    """Ready task ids with no assignee at all — operator-actionable (usually a
    misfiled task waiting for routing)."""
    auto_assigned_default: list[str] = field(default_factory=list)
    """Unassigned task ids that had ``kanban.default_assignee`` applied this
    tick before spawning, so telemetry/CLI/dashboard can show the dispatcher
    acting on the fallback rule rather than explicit assignments."""
    skipped_nonspawnable: list[str] = field(default_factory=list)
    """Ready task ids whose assignee names a control-plane lane (e.g. a Claude
    Code terminal like ``orion-cc``), not a Hermes profile. Expected steady-state
    on multi-lane setups, NOT operator-actionable; tracked apart so health
    telemetry can tell "stuck" from "correctly idle"."""
    skipped_per_profile_capped: list[tuple[str, str, int]] = field(default_factory=list)
    """``(task_id, assignee, current_running_count)`` deferred because the
    assignee is at ``kanban.max_in_progress_per_profile``. Picked up on a later
    tick; separate bucket so dashboards show "profile busy" vs "stuck"."""
    crashed: list[str] = field(default_factory=list)
    """Task ids reclaimed because their worker PID disappeared."""
    interrupted: list[str] = field(default_factory=list)
    """Task ids whose in-flight run the dispatcher ended as ``interrupted``:
    the card is sticky-blocked and waits for ``unblock_task``. Not a crash and not
    a breaker trip — no failure was charged."""
    cancelled: list[str] = field(default_factory=list)
    """Task ids whose LAUNCH the dispatcher cancelled before any run existed, so
    the card is still queued with no durable state change and nothing to unblock.
    Kept apart from ``interrupted`` so operator-recovery telemetry does not report
    a paused card for a launch that never started."""
    auto_blocked: list[str] = field(default_factory=list)
    """Task ids auto-blocked by the spawn-failure circuit breaker."""
    timed_out: list[str] = field(default_factory=list)
    """Task ids whose workers exceeded ``max_runtime_seconds``."""
    stale: list[str] = field(default_factory=list)
    """Task ids reclaimed for no heartbeat within ``dispatch_stale_timeout_seconds``."""
    respawn_guarded: list[tuple[str, str]] = field(default_factory=list)
    """``(task_id, reason)`` skipped by the respawn guard: ``"blocker_auth"``
    (quota/auth error — also auto-blocked), ``"recent_success"`` (completed run
    within guard window), ``"active_pr"`` (GitHub PR URL in a recent comment)."""
    rate_limited: list[str] = field(default_factory=list)
    """Task ids whose workers bailed on a provider rate-limit / quota wall
    (EX_TEMPFAIL sentinel exit) and were released to ``ready`` WITHOUT counting
    a failure — a long quota window must never trip the circuit breaker."""
    skipped_locked: bool = False
    """True when another process held the board's dispatch lock: this tick did
    no DB writes; the lock holder is making progress on the same board."""
    memory_pressure: Optional[str] = None
    """Memory pressure that restricted this tick: ``"critical"`` (no new
    workers), ``"elevated"`` (at most one), ``None`` (no restriction).
    Reclaim/promotion bookkeeping still ran; deferred tasks stay queued."""


def describe_suppression(results: Iterable[Optional["DispatchResult"]]) -> str:
    """Name task-level and tick-level holds in dispatcher health warnings."""
    counts: dict[str, int] = {}
    pressure: Optional[str] = None
    for res in results:
        if res is None:
            continue
        for _task_id, reason in res.respawn_guarded:
            counts[reason] = counts.get(reason, 0) + 1
        if res.rate_limited:
            counts["rate_limited"] = counts.get("rate_limited", 0) + len(res.rate_limited)
        if res.skipped_locked:
            counts["skipped_locked"] = counts.get("skipped_locked", 0) + 1
        if res.memory_pressure:
            pressure = res.memory_pressure
    parts = [f"{k}={v}" for k, v in sorted(counts.items())]
    if pressure:
        parts.append(f"memory_pressure={pressure}")
    return ", ".join(parts)



def _exit_code_kind(code: int) -> "tuple[str, int]":
    """Classify a worker exit regardless of whether a reaper or log observed it."""
    if code == 0:
        return ("clean_exit", 0)
    if code == _kb.KANBAN_RATE_LIMIT_EXIT_CODE:
        return ("rate_limited", code)
    if code == _kb.KANBAN_TERMINAL_PROVIDER_EXIT_CODE:
        return ("terminal_provider", code)
    return ("nonzero_exit", code)


_EXIT_TRAILER_RE = re.compile(
    r"^" + re.escape(KANBAN_WORKER_EXIT_TRAILER) + r"(\d+)\s*$", re.MULTILINE,
)


def _worker_log_exit_code(task_id: str, board: Optional[str] = None) -> Optional[int]:
    """Read the last durable CLI exit trailer when this process did not reap the worker."""
    try:
        raw = _kb.read_worker_log(task_id, tail_bytes=4000, board=board)
    except Exception:
        return None
    matches = _EXIT_TRAILER_RE.findall(raw or "")
    return int(matches[-1]) if matches else None

# ``worker_started_at`` value for a spawn whose fingerprint could not be captured. Distinct from the
# NULL legacy row (pre-fingerprint spawn): such a worker is held (its claim is never released beside
# the live PID) but NEVER signalled — missing process identity is refusal, not permission (#99558).
UNVERIFIED_WORKER_FINGERPRINT = "unverified"


def _process_fingerprint(pid: int) -> Optional[str]:
    """Restart-stable identity of a live process: ``"<instantiation epoch>|<start time>"``. The start
    time alone (``/proc/<pid>/stat`` field 22 on Linux) is clock ticks since THIS boot, so a row that
    survives a reboot could match an unrelated process with the same PID and the same tick value;
    ``gateway.drain_control.current_instantiation_epoch`` (``boot_id`` + PID-1 start) changes on every
    reboot / container recreate, so the composed value never survives one. ``None`` when unreadable."""
    from gateway.drain_control import current_instantiation_epoch
    from gateway.status import get_process_start_time
    start = get_process_start_time(int(pid))
    if start is None:
        return None
    return f"{current_instantiation_epoch()}|{start}"


def _worker_alive(pid: Optional[int], started_at) -> bool:
    """True when ``pid`` is live AND is still the worker we spawned. ``started_at`` is the fingerprint
    recorded by ``_set_worker_pid``; after a reboot (or any PID recycle) an unrelated process can own
    the number, so bare existence is never enough to extend a claim or to signal. A legacy row without
    a fingerprint keeps the existence answer: killing it is the pre-fingerprint behaviour and the row is
    rewritten with a fingerprint on its next spawn. An UNVERIFIED spawn also keeps the existence answer
    (a claim is never released beside a possibly-live worker) but ``_terminate_reclaimed_worker``
    refuses to signal it."""
    if not _kb._pid_alive(pid):
        return False
    if started_at == UNVERIFIED_WORKER_FINGERPRINT:
        return True
    return not _pid_recycled(pid, started_at)


def _pid_recycled(pid: Optional[int], started_at) -> bool:
    """True when a live ``pid`` is NOT the process fingerprinted at spawn (or the fingerprint can no
    longer be read). Signalling it would hit a stranger. ``None`` fingerprint = legacy row, never
    recycled; the UNVERIFIED marker is always foreign. An integer fingerprint (rows written before the
    boot witness was added) compares the start time only."""
    if started_at is None or not pid:
        return False
    if started_at == UNVERIFIED_WORKER_FINGERPRINT:
        return True
    if isinstance(started_at, str) and "|" in started_at:
        return _process_fingerprint(int(pid)) != started_at
    from gateway.status import _start_times_agree, get_process_start_time
    current = get_process_start_time(int(pid))
    if current is None:
        return True
    try:
        return not _start_times_agree(current, started_at)
    except (TypeError, ValueError):
        return True


def _kill_fn(signal_fn) -> Optional[Callable[[int, int], None]]:
    """``signal_fn`` test hook, else ``os.kill`` when the platform has one."""
    if signal_fn is not None:
        return signal_fn
    return os.kill if hasattr(os, "kill") else None


def _poll_worker_exit(pid: int, started_at: Optional[int] = None) -> bool:
    """Poll ~5 s (10 x 0.5 s) for ``pid`` to die; True once it is gone."""
    for _ in range(10):
        if not _worker_alive(pid, started_at):
            return True
        time.sleep(0.5)
    return False


def _sigkill(kill, pid: int) -> bool:
    """Best-effort SIGKILL; True when the signal was delivered."""
    try:
        # signal.SIGKILL doesn't exist on Windows; SIGTERM maps to TerminateProcess.
        kill(int(pid), getattr(signal, "SIGKILL", signal.SIGTERM))
        return True
    except (ProcessLookupError, OSError):
        return False


def _terminate_reclaimed_worker(
    pid: Optional[int],
    claim_lock: Optional[str],
    *,
    signal_fn=None,
    started_at=None,
) -> dict[str, Any]:
    """Best-effort host-local worker termination for reclaim paths. ``started_at`` is the spawn-time
    fingerprint: when the live process no longer matches it, the PID was recycled and nothing is
    signalled — the worker is gone, which is what the reclaim wanted (``terminated`` = True). An
    UNVERIFIED spawn (fingerprint capture failed) that is still live is never signalled either, but
    it is reported as surviving (``signal_refused``) so the reclaim holds the claim instead of
    spawning a duplicate beside it."""
    info: dict[str, Any] = {
        "prev_pid": int(pid) if pid else None,
        "host_local": False,
        "termination_attempted": False,
        "terminated": False,
        "sigkill": False,
    }
    if not pid or pid <= 0 or not claim_lock:
        return info
    if not str(claim_lock).startswith(_kb._host_prefix()):
        return info
    info["host_local"] = True
    if _kb._defer_post_commit(
        lambda: _terminate_reclaimed_worker(pid, claim_lock, signal_fn=signal_fn, started_at=started_at),
    ):
        return info

    kill = _kill_fn(signal_fn)
    if kill is None:
        return info
    if started_at == UNVERIFIED_WORKER_FINGERPRINT:
        # Never signal by bare number: a dead PID is "gone" (reclaim proceeds), a live one is held.
        info["signal_refused"] = True
        info["terminated"] = not _kb._pid_alive(pid)
        return info
    if _kb._pid_alive(pid) and _pid_recycled(pid, started_at):
        info["terminated"] = True
        info["pid_recycled"] = True
        return info

    info["termination_attempted"] = True
    try:
        kill(int(pid), signal.SIGTERM)
    except ProcessLookupError:
        # Already gone = successful termination. Leaving terminated=False would
        # make the reclaim guard misread a dead worker as alive and defer forever.
        info["terminated"] = True
        return info
    except OSError:
        return info

    if _poll_worker_exit(pid, started_at):
        info["terminated"] = True
        return info
    if _worker_alive(pid, started_at):
        if not _sigkill(kill, pid):
            return info
        info["sigkill"] = True
    info["terminated"] = not _worker_alive(pid, started_at)
    return info


def reap_terminal_workers(conn: sqlite3.Connection, *, signal_fn=None) -> list[str]:
    """End host-local workers that outlived their run (issue #111791) — a worker
    that called ``kanban_complete`` and then hung keeps its ``state.db`` sidecar
    fds open and no ``running``-only sweep can see it once ``tasks.worker_pid`` is
    cleared. Keys on the closed ``task_runs`` row's retained pid + spawn
    fingerprint: a legacy row (NULL fingerprint) or a recycled PID is never
    signalled; a pid that is simply gone or belongs to a proven older host epoch
    just has its evidence cleared, without probing the old host's PID. A run
    that ended less than ``TERMINAL_WORKER_REAP_GRACE_SECONDS`` ago is left
    alone so a worker still finalising after its own transition is not killed.
    One row's failure (signal, /proc probe) is logged and skips only that row.
    Returns the task ids whose worker was terminated."""
    rows = conn.execute(
        "SELECT id, task_id, worker_pid, worker_started_at, claim_lock, metadata FROM task_runs "
        "WHERE ended_at IS NOT NULL AND ended_at <= ? "
        "AND worker_pid IS NOT NULL AND worker_started_at IS NOT NULL",
        (int(time.time()) - TERMINAL_WORKER_REAP_GRACE_SECONDS,),
    ).fetchall()
    host_prefix = _kb._host_prefix()
    current_epoch = _kr.current_host_epoch()
    reaped: list[str] = []
    for row in rows:
        try:
            _reap_terminal_worker_row(conn, row, host_prefix, current_epoch, signal_fn, reaped)
        except Exception:
            _kb._log.debug(
                "kanban dispatch: terminal worker reap failed for run %s (task %s)",
                row["id"], row["task_id"], exc_info=True,
            )
    return reaped


def _reap_terminal_worker_row(
    conn, row, host_prefix: str, current_epoch: str, signal_fn, reaped: list[str],
) -> None:
    pid, fingerprint = int(row["worker_pid"]), row["worker_started_at"]
    if pid == os.getpid() or not str(row["claim_lock"] or "").startswith(host_prefix):
        return
    metadata = _kb._json_dict(row["metadata"])
    host_replaced = (
        metadata.get("reason") == HOST_RESTART_BLOCK_REASON
        or _instantiation_changed(str(metadata.get("host_epoch") or ""), current_epoch)
    )
    if not host_replaced and fingerprint == UNVERIFIED_WORKER_FINGERPRINT and _kb._pid_alive(pid):
        return  # unproven identity: never signalled; its evidence is cleared once the pid is gone
    alive = not host_replaced and _worker_alive(pid, fingerprint)
    termination = None
    if alive:
        termination = _terminate_reclaimed_worker(
            pid, row["claim_lock"], signal_fn=signal_fn, started_at=fingerprint)
        if not termination["terminated"]:
            return  # still alive: try again next tick
    with _kb.write_txn(conn):
        conn.execute(
            "UPDATE task_runs SET worker_pid = NULL, worker_started_at = NULL "
            "WHERE id = ? AND worker_pid = ? AND worker_started_at = ?",
            (row["id"], pid, fingerprint),
        )
        if alive:
            _kb._append_event(
                conn, row["task_id"], "terminal_worker_reaped",
                {"pid": pid, "worker_started_at": fingerprint, **termination}, run_id=row["id"],
            )
    if alive:
        reaped.append(row["task_id"])


def _worker_survived_termination(termination: dict) -> bool:
    """True when we tried to kill our own host-local worker and it is still alive.

    Reclaiming then would release the claim and spawn a second worker while the
    first still runs — the duplication loop. Only host-local workers we actually
    signalled count; a non-local lock or no-op attempt (no ``os.kill``) must fall
    through to the normal release path since we cannot manage that worker anyway.
    """
    return bool(
        termination.get("host_local")
        and (termination.get("termination_attempted") or termination.get("signal_refused"))
        and not termination.get("terminated")
    )


def _defer_reclaim_for_live_worker(
    conn: sqlite3.Connection,
    task_id: str,
    claim_lock: Optional[str],
    now: int,
    termination: dict,
    *,
    reason: str,
) -> None:
    """Hold a claim whose worker survived termination instead of releasing it.

    Extends ``claim_expires`` by ``RECLAIM_DEFER_GRACE_SECONDS`` so the task
    stays ``running`` (no duplicate spawn) and records ``reclaim_deferred``.
    The next tick retries the kill; not spawning a duplicate is what lets the
    throttled worker finally die.
    """
    grace = now + _kb.RECLAIM_DEFER_GRACE_SECONDS
    with _kb.write_txn(conn):
        cur = conn.execute(
            "UPDATE tasks SET claim_expires = ? "
            "WHERE id = ? AND status = 'running' AND claim_lock IS ?",
            (grace, task_id, claim_lock),
        )
        if cur.rowcount != 1:
            return
        run_id = _kb._current_run_id(conn, task_id)
        if run_id is not None:
            conn.execute("UPDATE task_runs SET claim_expires = ? WHERE id = ?", (grace, run_id))
        payload = {"reason": reason, "claim_lock": claim_lock, "claim_expires_now": grace}
        payload.update(termination)
        _kb._append_event(conn, task_id, "reclaim_deferred", payload, run_id=run_id)


def heartbeat_worker(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    note: Optional[str] = None,
    expected_run_id: Optional[int] = None,
) -> bool:
    """Record a ``heartbeat`` event + touch ``last_heartbeat_at``.

    Liveness signal orthogonal to the PID check: a worker whose forked child
    (train loop, crawl) is stuck can still have a live Python process.
    Returns False if the task is not running or its claim expired.
    """
    now = int(time.time())
    with _kb.write_txn(conn):
        sql = "UPDATE tasks SET last_heartbeat_at = ? WHERE id = ? AND status = 'running'"
        params: tuple = (now, task_id)
        if expected_run_id is not None:
            sql += " AND current_run_id = ?"
            params += (int(expected_run_id),)
        cur = conn.execute(sql, params)
        if cur.rowcount != 1:
            return False
        run_id = (
            int(expected_run_id)
            if expected_run_id is not None
            else _kb._current_run_id(conn, task_id)
        )
        if run_id is not None:
            conn.execute("UPDATE task_runs SET last_heartbeat_at = ? WHERE id = ?", (now, run_id))
        _kb._append_event(
            conn, task_id, "heartbeat",
            {"note": note} if note else None,
            run_id=run_id,
        )
    return True


def enforce_max_runtime(conn: sqlite3.Connection, *, signal_fn=None) -> list[str]:
    """Terminate workers whose per-task ``max_runtime_seconds`` has elapsed.

    SIGTERM, short grace, then SIGKILL. Emits ``timed_out`` and restores the
    task's source phase so the next tick re-spawns the same kind of worker —
    unless the circuit breaker already gave up, leaving it blocked. Host-local
    only (same reasoning as ``detect_crashed_workers``). ``signal_fn`` is a test hook.
    """
    timed_out: list[str] = []
    now = int(time.time())
    host_prefix = _kb._host_prefix()

    rows = conn.execute(
        "SELECT t.id, t.worker_pid, t.worker_started_at, "
        "       COALESCE(r.started_at, t.started_at) AS active_started_at, "
        "       t.max_runtime_seconds, t.claim_lock "
        "FROM tasks t "
        "LEFT JOIN task_runs r ON r.id = t.current_run_id "
        "WHERE t.status = 'running' AND t.max_runtime_seconds IS NOT NULL "
        "  AND COALESCE(r.started_at, t.started_at) IS NOT NULL "
        "  AND t.worker_pid IS NOT NULL"
    ).fetchall()
    for row in rows:
        lock = row["claim_lock"] or ""
        if not lock.startswith(host_prefix):
            continue
        # Runtime is per attempt: ``tasks.started_at`` records the FIRST start,
        # so retries must be measured from the active task_runs row.
        elapsed = now - int(row["active_started_at"])
        limit = int(row["max_runtime_seconds"])
        if elapsed < limit:
            continue

        pid = int(row["worker_pid"])
        tid = row["id"]
        started_at = _kb._row_get(row, "worker_started_at")
        if started_at == UNVERIFIED_WORKER_FINGERPRINT and _kb._pid_alive(pid):
            # Fingerprint capture failed at spawn: we cannot prove this live PID is our worker, so
            # it is neither signalled nor released beside (duplicate). It is reclaimed once it exits.
            _kb._log.warning("kanban: task %s worker pid %s exceeded max runtime but has no verified "
                             "identity; not signalled", tid, pid)
            continue
        # SIGTERM then SIGKILL after 5 s grace; workers wanting a cleaner
        # shutdown install their own SIGTERM handler. A recycled PID (fingerprint
        # mismatch) is never signalled: the worker is already gone.
        killed = False
        kill = _kill_fn(signal_fn)
        if kill is not None and not (_kb._pid_alive(pid) and _pid_recycled(pid, started_at)):
            with contextlib.suppress(ProcessLookupError, OSError):
                kill(pid, signal.SIGTERM)
            # Short polling wait — no time.sleep on the write txn.
            _poll_worker_exit(pid, started_at)
            if _worker_alive(pid, started_at):
                killed = _sigkill(kill, pid)

        error = f"elapsed {int(elapsed)}s > limit {limit}s"
        with _kb.write_txn(conn):
            retry_status = _kb._retry_status_for_run(conn, tid)
            cur = conn.execute(
                "UPDATE tasks SET status = ?, claim_lock = NULL, "
                "claim_expires = NULL, worker_pid = NULL, worker_started_at = NULL, "
                "last_heartbeat_at = NULL "
                "WHERE id = ? AND status = 'running' "
                "  AND worker_pid = ? AND claim_lock IS ?",
                (retry_status, tid, pid, row["claim_lock"]),
            )
            if cur.rowcount == 1:
                payload = {
                    "pid": pid,
                    "elapsed_seconds": int(elapsed),
                    "limit_seconds": limit,
                    "sigkill": killed,
                    "retry_status": retry_status,
                }
                run_id = _kb._end_run(
                    conn, tid, outcome="timed_out", status="timed_out",
                    error=error, metadata=payload,
                )
                _kb._append_event(conn, tid, "timed_out", payload, run_id=run_id)
                timed_out.append(tid)
        # Outside the write_txn above because ``_record_task_failure`` opens its
        # own. If the breaker trips this flips the task to ``blocked`` and emits
        # ``gave_up`` on top of the ``timed_out`` already emitted.
        if cur.rowcount == 1:
            _record_task_failure(
                conn, tid,
                error=error,
                outcome="timed_out",
                release_claim=False,
                end_run=False,
                event_payload_extra={"pid": pid, "sigkill": killed, "retry_status": retry_status},
            )
    return timed_out


# A running task with no heartbeat for this long is inactive regardless of
# ``dispatch_stale_timeout_seconds`` (spec: ">4h started + no commits in 1h").
_STALE_HEARTBEAT_GAP_SECONDS = 3600


def detect_stale_running(
    conn: sqlite3.Connection,
    *,
    stale_timeout_seconds: int = 0,
    signal_fn=None,
) -> list[str]:
    """Reclaim ``running`` tasks with no heartbeat progress; returns their ids.

    Stale = running longer than ``stale_timeout_seconds`` (active run's
    ``started_at``, else ``tasks.started_at``) AND ``last_heartbeat_at`` NULL or
    older than ``_STALE_HEARTBEAT_GAP_SECONDS``. Task returns to its source
    phase, run closes ``outcome='stale'``, a live host-local worker is killed.
    ``0`` disables the check; ``signal_fn`` is a test hook. Deliberately NOT
    counted via ``_record_task_failure``: an absent heartbeat is not a worker
    failure, and counting it would let long-running tasks trip the breaker.
    """
    if stale_timeout_seconds <= 0:
        return []

    now = int(time.time())
    reclaimed: list[str] = []

    rows = conn.execute(
        "SELECT t.id, t.worker_pid, t.worker_started_at, t.last_heartbeat_at, t.claim_lock, "
        "       COALESCE(r.started_at, t.started_at) AS active_started_at "
        "FROM tasks t "
        "LEFT JOIN task_runs r ON r.id = t.current_run_id "
        "WHERE t.status = 'running'"
    ).fetchall()

    for row in rows:
        if row["active_started_at"] is None:
            continue
        elapsed = now - int(row["active_started_at"])
        if elapsed < stale_timeout_seconds:
            continue

        last_hb = row["last_heartbeat_at"]
        hb_age = (now - int(last_hb)) if last_hb is not None else None
        if hb_age is not None and hb_age < _STALE_HEARTBEAT_GAP_SECONDS:
            continue

        pid = row["worker_pid"]
        tid = row["id"]
        lock = row["claim_lock"] or ""

        termination = _kb._terminate_reclaimed_worker(
            pid, lock, signal_fn=signal_fn, started_at=_kb._row_get(row, "worker_started_at"))

        # Never release a claim while our own worker is still alive: that would
        # spawn a duplicate beside it. Hold the claim and retry next tick.
        if _worker_survived_termination(termination):
            _defer_reclaim_for_live_worker(
                conn, tid, lock, now, termination,
                reason="heartbeat_stale_worker_alive",
            )
            continue

        with _kb.write_txn(conn):
            retry_status = _kb._retry_status_for_run(conn, tid)
            cur = conn.execute(
                "UPDATE tasks SET status = ?, claim_lock = NULL, "
                "claim_expires = NULL, worker_pid = NULL, worker_started_at = NULL, "
                "last_heartbeat_at = NULL "
                "WHERE id = ? AND status = 'running' "
                "  AND claim_lock IS ?",
                (retry_status, tid, row["claim_lock"]),
            )
            if cur.rowcount != 1:
                continue

            payload = {
                "elapsed_seconds": int(elapsed),
                "last_heartbeat_at": _kb._opt_int(last_hb),
                "heartbeat_age_seconds": _kb._opt_int(hb_age),
                "timeout_seconds": stale_timeout_seconds,
                "pid": int(pid) if pid else None,
                "retry_status": retry_status,
            }
            payload.update(termination)

            run_id = _kb._end_run(
                conn, tid,
                outcome="stale", status="stale",
                error=(
                    f"no heartbeat for {int(hb_age)}s "
                    if hb_age is not None
                    else "no heartbeat ever"
                ) + f" after {int(elapsed)}s running",
                metadata=payload,
            )
            _kb._append_event(conn, tid, "stale", payload, run_id=run_id)
            reclaimed.append(tid)

    return reclaimed


def reconcile_orphaned_running(conn: sqlite3.Connection) -> list[str]:
    """Requeue ``running`` cards with broken claim bookkeeping; returns their ids.

    A task ``running`` with NULL ``claim_lock``/``claim_expires`` (crash
    mid-claim, manual SQL, DB restore) is a zombie forever: ``release_stale_claims``
    needs ``claim_expires``, ``detect_crashed_workers`` needs a host-local lock +
    pid, ``detect_stale_running`` is off by default. Orphans go back to ``ready``
    with a comment, leaked run closed, ``reconciled`` event; a row with a live
    host-local PID is deferred so no duplicate spawns beside it.
    """
    now = int(time.time())
    reconciled: list[str] = []
    rows = conn.execute(
        "SELECT id, claim_lock, claim_expires, worker_pid, worker_started_at FROM tasks "
        "WHERE status = 'running' "
        "  AND (claim_lock IS NULL OR claim_expires IS NULL)"
    ).fetchall()
    for row in rows:
        tid = row["id"]
        pid = row["worker_pid"]
        if pid and _worker_alive(pid, _kb._row_get(row, "worker_started_at")):
            # Never requeue beside a live process. Retry next tick.
            _kb._log.debug(
                "kanban reconcile: task %s has broken claim bookkeeping but "
                "pid %s is alive on this host — deferring", tid, pid,
            )
            continue
        with _kb.write_txn(conn):
            retry_status = _kb._retry_status_for_run(conn, tid)
            cur = conn.execute(
                "UPDATE tasks SET status = ?, claim_lock = NULL, "
                "claim_expires = NULL, worker_pid = NULL, worker_started_at = NULL, "
                "last_heartbeat_at = NULL "
                "WHERE id = ? AND status = 'running' "
                "  AND claim_lock IS ? AND claim_expires IS ?",
                (retry_status, tid, row["claim_lock"], row["claim_expires"]),
            )
            if cur.rowcount != 1:
                continue
            payload = {
                "reason": "orphaned_running",
                "claim_lock": row["claim_lock"],
                "claim_expires": _kb._opt_int(row["claim_expires"]),
                "worker_pid": int(pid) if pid else None,
                "now": now,
            }
            run_id = _kb._end_run(
                conn, tid,
                outcome="reclaimed", status="reclaimed",
                error="orphaned running card (broken claim bookkeeping)",
                metadata=payload,
            )
            _kb._insert_comment(
                conn, tid, "dispatcher",
                "reconciliation: card was 'running' with no valid claim "
                "(dead/gone worker) — requeued to ready",
                now,
            )
            _kb._append_event(conn, tid, "reconciled", payload, run_id=run_id)
            reconciled.append(tid)
        _kb._log.info(
            "kanban reconcile: requeued orphaned running task %s "
            "(claim_lock=%r, worker_pid=%r)", tid, row["claim_lock"], pid,
        )
    return reconciled


def _error_fingerprint(error_text: str) -> str:
    """Normalize an error message (strip PIDs, timestamps) so same-root-cause errors group."""
    fp = re.sub(r'\bpid \d+\b', 'pid N', error_text[:80])
    fp = re.sub(r'\b\d{10,}\b', '<TS>', fp)
    return fp.lower().strip()


# ~96% of "clean exit without a terminal tool call" tasks complete on a later
# run, so a protocol violation gets a bounded retry before the breaker trips.
# The budget is a violation-only STREAK (``_protocol_violation_streak``),
# independent of ``consecutive_failures``: other failure kinds neither consume
# nor extend it. Per-task ``max_retries`` overrides it.
_PROTOCOL_VIOLATION_FAILURE_LIMIT = 3

# Closed runs to walk when counting the streak; it trips at a handful anyway.
_PROTOCOL_VIOLATION_SCAN_LIMIT = 50


def _protocol_violation_streak(conn: sqlite3.Connection, task_id: str) -> int:
    """Count the task's trailing run of clean-exit protocol violations.

    Walks closed runs newest-first (including the one ``detect_crashed_workers``
    just closed). ``rate_limited`` runs are neutral and skipped (a quota wall
    says nothing about the task); any other closed run breaks the streak, so
    the budget counts ONLY protocol violations. Violations are recognized by the
    ``protocol_violation`` run-metadata marker, with the error text as fallback
    for runs recorded before the marker existed.
    """
    streak = 0
    rows = conn.execute(
        "SELECT outcome, error, metadata FROM task_runs "
        "WHERE task_id = ? AND ended_at IS NOT NULL "
        "ORDER BY id DESC LIMIT ?",
        (task_id, _PROTOCOL_VIOLATION_SCAN_LIMIT),
    ).fetchall()
    for row in rows:
        outcome = row["outcome"] or ""
        if outcome == "rate_limited":
            continue
        if outcome == "crashed" and (
            _kb._json_dict(row["metadata"]).get("protocol_violation")
            or "protocol violation" in (row["error"] or "")
        ):
            streak += 1
            continue
        break
    return streak


_PROTOCOL_VIOLATION_ERROR = (
    # Worker subprocess returned 0 but its task is still ``running`` in the DB — it exited without calling
    # ``kanban_complete`` / ``kanban_block`` / ``kanban_request_review``. Overwhelmingly the work itself succeeded and only the
    # paperwork was skipped, so a retry usually completes; the corrective sentence below is surfaced to the
    # retry worker via the prior-attempt error in ``build_worker_context`` (guidance approach from #61817).
    # Keep this short: ``_record_task_failure`` caps the stored error at 500 chars and the worker's own
    # last output (``_worker_final_output``, up to 400 chars) is appended after it — a longer preamble
    # truncates away the worker's explanation, which is the part the board and the retry worker need.
    "worker exited cleanly (rc=0) without kanban_complete, kanban_block "
    "or kanban_request_review — protocol violation. "
    "If the prior run already did the work, verify it and "
    "report it via kanban_complete (or kanban_request_review); "
    "a run without a terminal kanban call counts as failed no "
    "matter what it did."
)


# Rich panel/rule chrome around the rendered response, and the CLI's own preamble lines.
_LOG_CHROME = re.compile(r"[─━═╭╮╰╯│┃┌┐└┘]+|☤\s*Hermes")


def _exit_summary_marker() -> str:
    """The CLI exit-summary header (``cli_session_mixin.show_exit_summary``), in the active language."""
    from agent.i18n import t
    return t("cli.session.exit_resume_hint")


def _log_noise_prefixes() -> tuple[str, ...]:
    from agent.i18n import t
    return ("session_id:", "Query:", t("cli.chat.initializing_agent"))


def _worker_final_output(task_id: str, board: Optional[str] = None) -> str:
    """Best-effort read of a dead worker's last printed text, for the board diagnostic.

    A ``chat -q`` worker's stdout/stderr are redirected to its per-task log
    (``_default_spawn``), so when it exits without a terminal board call the
    reason is usually sitting there: the model's own explanation of why it could
    not comply (#88603), or the rendered provider error (#46593). The reap used to
    discard it in favour of a canned message on every retry. Trims the CLI exit
    summary, rule lines and the ``session_id:`` trailer; returns "" (never raises)
    on a missing/empty log.

    ``board`` must come from the dispatching tick: ambient current-board resolution
    is wrong for every board but the one the dispatcher thread happens to call
    "current", so the log would silently not be found.
    """
    try:
        raw = _kb.read_worker_log(task_id, tail_bytes=4000, board=board)
    except Exception:
        return ""
    if not raw:
        return ""
    raw = _EXIT_TRAILER_RE.sub("", raw)
    cut = raw.rfind(_exit_summary_marker())
    if cut != -1:
        raw = raw[:cut]
    lines = []
    for ln in raw.splitlines():
        ln = _LOG_CHROME.sub("", ln).strip()
        if ln and not ln.startswith(_log_noise_prefixes()):
            lines.append(ln)
    return " ".join(lines)[-400:]


@dataclass
class _DeadWorker:
    """How ``detect_crashed_workers`` should book one dead worker."""

    kind: str
    code: Optional[int]
    error_text: str
    event_kind: str
    event_payload: dict
    protocol_violation: bool = False
    rate_limited: bool = False
    terminal_provider: bool = False
    """``KANBAN_TERMINAL_PROVIDER_EXIT_CODE``: the provider rejected the worker's
    credential/model — trips the breaker on this first occurrence."""

    @property
    def run_outcome(self) -> str:
        # A rate-limited requeue is recorded as ``rate_limited`` so board history
        # doesn't show a phantom crash for a quota wall.
        return "rate_limited" if self.rate_limited else "crashed"


def _classify_dead_worker(
    pid: int, claimer: Optional[str], *, task_id: Optional[str] = None, board: Optional[str] = None,
) -> _DeadWorker:
    """Map a dead worker's reaped exit status to its reclaim bookkeeping.

    A clean exit or a crash carries the worker's own last output (``worker_output``
    in the event payload, appended to the error text) so the board and the retry
    worker see WHY instead of a bare label; a rate-limited requeue does not need it.
    """
    dead = _classify_dead_worker_exit(pid, claimer, task_id=task_id, board=board)
    if task_id and not dead.rate_limited:
        worker_output = _worker_final_output(task_id, board=board)
        if worker_output:
            dead.error_text += f" Worker's last output: {worker_output!r}"
            dead.event_payload["worker_output"] = worker_output
    return dead


def _classify_dead_worker_exit(
    pid: int,
    claimer: Optional[str],
    *,
    task_id: Optional[str] = None,
    board: Optional[str] = None,
) -> _DeadWorker:
    """Exit status -> reclaim bookkeeping, before the worker's own words are folded in.

    The reap registry only knows children of THIS process; a per-tick dispatcher
    reads the exit trailer the worker left in its log instead, so the same death
    gets the same booking (protocol violation / rate-limit requeue / crash) as
    under the gateway-embedded dispatcher. A worker that never reached its exit
    epilogue (killed, OOM) leaves no trailer and stays a plain crash.
    """
    kind, code = _classify_worker_exit(pid)
    if kind == "unknown" and task_id:
        logged = _worker_log_exit_code(task_id, board=board)
        if logged is not None:
            kind, code = _exit_code_kind(logged)
    if kind == "clean_exit":
        # rc=0 while still ``running``: usually the work succeeded and only the
        # paperwork was skipped; the corrective sentence reaches the retry
        # worker via ``build_worker_context``.
        return _DeadWorker(
            kind, code, _PROTOCOL_VIOLATION_ERROR, "protocol_violation",
            # ``protocol_violation`` is the durable marker for
            # _protocol_violation_streak: _end_run copies this payload into the
            # run metadata.
            {"pid": pid, "claimer": claimer, "exit_code": code, "protocol_violation": True},
            protocol_violation=True,
        )
    if kind == "rate_limited":
        # Quota wall — NOT a task failure. Release to the source phase and do
        # NOT count a failure so a long quota window can't trip the breaker.
        return _DeadWorker(
            kind, code,
            f"pid {pid} exited rate-limited (quota wall) — requeued without counting a failure",
            "rate_limited",
            {"pid": pid, "claimer": claimer, "exit_code": code},
            rate_limited=True,
        )
    if kind == "terminal_provider":
        # The worker classified its own provider failure as unhealable (credential
        # revoked, model gone): every further spawn would hit the same wall, so
        # ``_account_crashes`` trips the breaker now instead of after ``failure_limit``.
        return _DeadWorker(
            kind, code,
            f"pid {pid} exited on a terminal provider error (exit {code}): the provider rejected "
            "this profile's credential or model — fix the configuration, then unblock.",
            "crashed",
            {"pid": pid, "claimer": claimer, "exit_kind": kind, "exit_code": code, "terminal_provider": True},
            terminal_provider=True,
        )
    if kind == "nonzero_exit":
        error_text = f"pid {pid} exited with code {code}"
    elif kind == "signaled":
        error_text = f"pid {pid} killed by signal {code}"
    else:
        error_text = f"pid {pid} not alive"
    event_payload = {"pid": pid, "claimer": claimer}
    if code is not None and kind != "unknown":
        event_payload["exit_kind"] = kind
        event_payload["exit_code"] = code
    return _DeadWorker(kind, code, error_text, "crashed", event_payload)


@dataclass
class _CrashSweep:
    """Everything ``detect_crashed_workers`` collects inside its reclaim txn."""

    crashed: list[str] = field(default_factory=list)
    rate_limited: list[str] = field(default_factory=list)
    # ``(task_id, pid, claimer, dead_worker)``: accounted after the txn via
    # ``_record_task_failure`` (needs its own write_txn).
    crash_details: list[tuple[str, int, str, _DeadWorker]] = field(default_factory=list)
    # Worker-exit observer payloads, fired only after every reclaim/accounting
    # txn has committed.
    exited_hook_payloads: list[dict] = field(default_factory=list)


def _reclaim_dead_workers(conn: sqlite3.Connection, board: Optional[str] = None) -> _CrashSweep:
    """Release every host-local ``running`` task whose worker PID is dead."""
    sweep = _CrashSweep()
    with _kb.write_txn(conn):
        rows = conn.execute(
            "SELECT id, worker_pid, worker_started_at, claim_lock, started_at, assignee "
            "FROM tasks "
            "WHERE status = 'running' AND worker_pid IS NOT NULL"
        ).fetchall()
        host_prefix = _kb._host_prefix()
        for row in rows:
            lock = row["claim_lock"] or ""
            if not lock.startswith(host_prefix):
                continue
            # Launch-window grace so a freshly-spawned worker isn't reclaimed
            # before its PID is visible on /proc.
            started_at = _kb._row_get(row, "started_at")
            if started_at is not None and time.time() - started_at < _kb._resolve_crash_grace_seconds():
                continue
            if _worker_alive(row["worker_pid"], _kb._row_get(row, "worker_started_at")):
                continue

            pid = int(row["worker_pid"])
            dead = _classify_dead_worker(pid, row["claim_lock"], task_id=row["id"], board=board)
            retry_status = _kb._retry_status_for_run(conn, row["id"])
            dead.event_payload["retry_status"] = retry_status
            cur = conn.execute(
                "UPDATE tasks SET status = ?, claim_lock = NULL, "
                "claim_expires = NULL, worker_pid = NULL, worker_started_at = NULL "
                "WHERE id = ? AND status = 'running' "
                "  AND worker_pid = ? AND claim_lock IS ?",
                (retry_status, row["id"], pid, row["claim_lock"]),
            )
            if cur.rowcount != 1:
                continue
            run_id = _kb._end_run(
                conn, row["id"],
                outcome=dead.run_outcome, status=dead.run_outcome,
                error=dead.error_text,
                metadata=dict(dead.event_payload),
            )
            _kb._append_event(conn, row["id"], dead.event_kind, dead.event_payload, run_id=run_id)
            sweep.exited_hook_payloads.append({
                "task_id": row["id"],
                "assignee": row["assignee"],
                "run_id": run_id,
                "worker_pid": pid,
                "exit_kind": dead.kind,
                "exit_code": dead.code,
                "outcome": dead.run_outcome,
                "retry_status": retry_status,
            })
            if dead.rate_limited or dead.protocol_violation:
                # Stamp last_failure_error WITHOUT touching ``consecutive_failures``:
                # a rate-limited requeue must show ``check_respawn_guard`` a quota
                # blocker; a below-budget protocol violation never reaches
                # ``_record_task_failure`` (which stamps this column), yet the
                # board UI and retry worker need the corrective message.
                conn.execute(
                    "UPDATE tasks SET last_failure_error = ? WHERE id = ?",
                    (dead.error_text[:500], row["id"]),
                )
            if dead.rate_limited:
                sweep.rate_limited.append(row["id"])
            else:
                sweep.crashed.append(row["id"])
                sweep.crash_details.append((row["id"], pid, row["claim_lock"], dead))
    return sweep


def _account_crashes(
    conn: sqlite3.Connection, crash_details: list, *, failure_limit: Optional[int] = None,
) -> list[str]:
    """Count each crash against the breaker; returns the task ids it tripped.

    Protocol violations get a BOUNDED violation-only budget independent of
    ``consecutive_failures`` (per-task ``max_retries`` takes precedence);
    systemic same-error crashes (>= 3 identical fingerprints this tick) and
    terminal provider errors (credential revoked, model gone — a retry cannot
    heal them) trip immediately. ``failure_limit`` is the dispatcher's configured
    ``kanban.failure_limit``; ``None`` keeps ``DEFAULT_FAILURE_LIMIT`` for
    direct callers.
    """
    auto_blocked: list[str] = []
    fp_counts: dict[str, int] = {}
    for _, _, _, dead in crash_details:
        fp = _error_fingerprint(dead.error_text)
        fp_counts[fp] = fp_counts.get(fp, 0) + 1
    for tid, pid, claimer, dead in crash_details:
        error_text = dead.error_text
        if dead.protocol_violation:
            streak = _protocol_violation_streak(conn, tid)
            trow = conn.execute("SELECT max_retries FROM tasks WHERE id = ?", (tid,)).fetchone()
            if trow is None:
                continue  # task deleted mid-loop
            task_override = _kb._row_get(trow, "max_retries")
            violation_limit = (
                int(task_override) if task_override is not None else _PROTOCOL_VIOLATION_FAILURE_LIMIT
            )
            if streak < violation_limit:
                # Below budget: already back at ``ready`` with the error stamped.
                # No ``_record_task_failure`` — must not consume the unified budget.
                continue
            # ``force_trip``: the decision (incl. per-task ``max_retries``) was
            # already made against the violation streak above.
            tripped = _record_task_failure(
                conn, tid,
                error=error_text,
                outcome="crashed",
                failure_limit=violation_limit,
                force_trip=True,
                release_claim=False,
                end_run=False,
                event_payload_extra={
                    "pid": pid,
                    "claimer": claimer,
                    "protocol_violations": streak,
                    "protocol_violation_limit": violation_limit,
                },
            )
        elif dead.terminal_provider:
            # A retry cannot heal a revoked credential or a missing model, so
            # the whole ``failure_limit`` budget would be spent on identical
            # failures. ``force_trip`` blocks now, sticky: ``recompute_ready``
            # must not auto-resume it before the operator fixes the provider.
            tripped = _record_task_failure(
                conn, tid,
                error=error_text,
                outcome="crashed",
                force_trip=True,
                release_claim=False,
                end_run=False,
                event_payload_extra={"pid": pid, "claimer": claimer, "terminal_provider": True},
            )
        else:
            is_systemic = fp_counts.get(_error_fingerprint(error_text), 0) >= 3
            extra = {"pid": pid, "claimer": claimer}
            if is_systemic:
                # Trips at 1, below any ``failure_limit``: hold it for an operator.
                extra["sticky"] = True
            tripped = _record_task_failure(
                conn, tid,
                error=error_text,
                outcome="crashed",
                failure_limit=1 if is_systemic else failure_limit,
                release_claim=False,
                end_run=False,
                event_payload_extra=extra,
            )
        if tripped:
            auto_blocked.append(tid)
    return auto_blocked


def detect_crashed_workers(
    conn: sqlite3.Connection, board: Optional[str] = None, *, failure_limit: Optional[int] = None,
) -> list[str]:
    """Reclaim ``running`` tasks whose worker PID is no longer alive.

    Restores the source phase immediately (no waiting for the claim TTL), for
    tasks claimed by *this host* only — other hosts' PIDs are meaningless.
    Clean exit while ``running`` is a protocol violation with a bounded
    violation-only retry budget; ``KANBAN_RATE_LIMIT_EXIT_CODE`` is a quota
    wall, released WITHOUT counting a failure and surfaced via the
    ``_last_rate_limited`` attribute (the return stays crashed-only).

    ``failure_limit`` forwards the dispatcher's configured ``kanban.failure_limit``
    to the breaker; ``None`` keeps ``DEFAULT_FAILURE_LIMIT`` for direct callers.
    """
    sweep = _reclaim_dead_workers(conn, board=board)
    # Outside the main txn: account each crash and maybe trip the breaker.
    auto_blocked = (
        _account_crashes(conn, sweep.crash_details, failure_limit=failure_limit)
        if sweep.crash_details else []
    )
    # Side-channel attributes keep the public ``list[str]`` return stable;
    # ``dispatch_once`` reads them to populate ``DispatchResult``. Rate-limited
    # requeues did NOT count a failure and are NOT crashes.
    detect_crashed_workers._last_auto_blocked = auto_blocked  # type: ignore[attr-defined]
    detect_crashed_workers._last_rate_limited = sweep.rate_limited  # type: ignore[attr-defined]
    # Fired only now, after the reclaim txn AND breaker accounting have
    # committed, so subscribers always observe fully durable board state.
    if sweep.exited_hook_payloads and _kb._kanban_observer_consumed("on_kanban_worker_exited"):
        _board = _kb.get_current_board()
        for hook_fields in sweep.exited_hook_payloads:
            hook_fields = dict(hook_fields)
            _kb._fire_kanban_lifecycle_hook(
                # Kanban worker-lifecycle, task-mutation, and dispatcher-tick observers (RFC #58548,
                # accepted as the design basis in the #64231 batch disposition; on_kanban_dispatch_tick is
                # the re-port of PR #56066). All five are observers only: return values are ignored, and
                # every fire site is fully best-effort, so a broken callback can never break dispatch or a
                # task mutation. Cost rule: every call site short-circuits on has_hook(), so when nothing
                # subscribes no payload is built and the hot paths (each dispatcher tick, each task write)
                # pay one dict probe. WHICH PROCESS: worker spawn/exit/stale-claim and the dispatch tick
                # fire in the DISPATCHER process (gateway-embedded dispatcher or ``hermes kanban
                # dispatch``); on_kanban_task_updated fires in whichever process committed the mutation
                # (CLI, worker, or the gateway-embedded dashboard API). Common kwargs (task-scoped hooks):
                # task_id: str, profile_name: str, board: str | None, assignee: str | None, run_id: int |
                # None. on_kanban_worker_spawned fires after ``spawn_fn`` returns AND the worker PID (when
                # one was reported) is durably persisted, per the RFC timing contract; like
                # kanban_task_claimed it runs inside the board's dispatch lock, so callbacks must stay fast.
                # Adds: worker_pid: int | None, workspace_path: str. Privacy: workspace_path is a filesystem
                # path and may reveal project layout or usernames.
                "on_kanban_worker_exited",
                hook_fields.pop("task_id"),
                board=_board,
                **hook_fields,
            )
    return sweep.crashed


# Typed block reasons for a card the dispatcher interrupted on purpose. Both
# hand the card back to the operator exactly like an explicit ``kanban_block``
# (sticky, no auto-recovery), and neither is a crash or a breaker trip.
HOST_RESTART_BLOCK_REASON = "host_restarted"
LAUNCH_STOPPED_BLOCK_REASON = "gateway_stopping"
_INTERRUPT_BLOCK_KIND = "needs_input"


def _instantiation_changed(recorded: str, current: str) -> bool:
    """Late-bound ``gateway.drain_control.instantiation_changed``; False when unavailable.

    Failing closed (False = "not proven changed") keeps legacy receipts and
    hosts without the gateway package on their existing reclaim behavior.
    """
    try:
        from gateway.drain_control import instantiation_changed
    except Exception:
        return False
    return bool(instantiation_changed(recorded, current))


def _user_manager_stopping() -> bool:
    """True only on POSITIVE native evidence that the user manager is stopping.

    Covers the user-manager shutdown race that can precede the gateway's own
    signal: the scope route then fails while no drain flag is set yet. A timeout,
    a missing ``systemctl``, an inaccessible bus, ``degraded``, and arbitrary
    error text are NOT evidence — those stay ordinary launch failures.
    """
    try:
        from hermes_cli.gateway import _run_systemctl

        result = _run_systemctl(
            ["is-system-running"], timeout=5, capture_output=True, text=True, check=False,
        )
    except Exception:
        return False
    return (result.stdout or "").strip().lower() == "stopping"


def _locally_owned_running_runs(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """``running`` cards whose CURRENT, unfinished run was claimed by this host.

    Ownership comes from the run's recorded claim prefix, not the task's: a
    half-cleaned claim can leave ``tasks.claim_lock`` NULL while the run still
    records ours (the orphan case ``reconcile_orphaned_running`` handles). A NULL
    or foreign run claim proves nothing about which host owns the worker, so it
    is skipped rather than guessed.
    """
    host_prefix = _kb._host_prefix()
    rows = conn.execute(
        "SELECT t.id AS task_id, t.current_run_id AS run_id, "
        "       r.claim_lock AS run_claim, r.metadata AS run_metadata "
        "FROM tasks t JOIN task_runs r ON r.id = t.current_run_id "
        "WHERE t.status = 'running' AND r.ended_at IS NULL"
    ).fetchall()
    return [
        row for row in rows
        if str(_kb._row_get(row, "run_claim") or "").startswith(host_prefix)
    ]


def _run_host_epoch(row: sqlite3.Row) -> str:
    """The ``host_epoch`` recorded on a run, or "" for legacy/unknown receipts."""
    return str(_kb._json_dict(_kb._row_get(row, "run_metadata")).get("host_epoch") or "")


def pause_interrupted_run(
    conn: sqlite3.Connection, task_id: str, run_id: int, *,
    reason: str, error: str, extra_payload: Optional[dict] = None,
) -> bool:
    """Atomically pause ONE interrupted card: sticky block, run closed as ``interrupted``.

    One txn, one compare-and-swap over (still ``running``, still that current run,
    observed claim/PID). Ownership must be THIS host's claim on that exact run —
    a NULL or foreign claim proves nothing and is refused rather than guessed.
    No failure is charged and none is reset, and the old PID is never probed or
    signalled, because after a reboot it may already belong to another process.
    The sticky ``blocked`` event keeps ``recompute_ready`` off the card until
    ``unblock_task``; the operator's block-loop columns are deliberately left
    alone so a repeat interruption cannot escalate the card to ``triage``.
    """
    host_prefix = _kb._host_prefix()
    with _kb.write_txn(conn):
        task = conn.execute(
            "SELECT status, claim_lock, worker_pid, current_run_id, "
            "       block_kind, block_recurrences "
            "FROM tasks WHERE id = ?", (task_id,),
        ).fetchone()
        if task is None or task["status"] != "running":
            return False
        if int(_kb._row_get(task, "current_run_id") or 0) != int(run_id):
            return False
        run = conn.execute(
            "SELECT claim_lock, ended_at FROM task_runs WHERE id = ? AND task_id = ?",
            (run_id, task_id),
        ).fetchone()
        if run is None or _kb._row_get(run, "ended_at") is not None:
            return False
        run_claim = str(_kb._row_get(run, "claim_lock") or "")
        if not run_claim.startswith(host_prefix):
            return False
        task_claim = _kb._row_get(task, "claim_lock")
        if task_claim is not None and str(task_claim) != run_claim:
            # The card carries a claim this run does not own: either a foreign
            # claimer, or a newer local claim from a rebuild of the same card.
            # Ownership is not ours to infer, so refuse rather than clear it.
            return False
        retry_status = _kb._retry_status_for_run(conn, task_id, run_id)
        # Deliberately NOT ``_route_block``: its unblock-loop breaker counts
        # repeated same-cause blocks and routes the card to ``triage`` past
        # ``BLOCK_RECURRENCE_LIMIT``. An infrastructure interruption is not an
        # operator block loop, and a second reboot must not move the card into
        # ``triage`` — ``unblock_task`` cannot release that status, while the
        # auto-decomposer scans exactly that lane and may rewrite the card. The
        # sticky ``blocked`` event alone is what the resume contract needs, so
        # the operator's block-loop columns are left untouched.
        event_kind = "blocked"
        payload = {
            "reason": reason,
            "kind": _INTERRUPT_BLOCK_KIND,
            "cause": _kb.normalized_block_cause(_INTERRUPT_BLOCK_KIND),
            "source_status": retry_status,
        }
        cur = conn.execute(
            "UPDATE tasks SET status = 'blocked', claim_lock = NULL, "
            "claim_expires = NULL, worker_pid = NULL, worker_started_at = NULL "
            "WHERE id = ? AND status = 'running' AND current_run_id = ? "
            "  AND claim_lock IS ? AND worker_pid IS ?",
            (task_id, run_id, task["claim_lock"], task["worker_pid"]),
        )
        if cur.rowcount != 1:
            return False
        provenance = {"reason": reason, "retry_status": retry_status, **(extra_payload or {})}
        payload.update({"run_id": run_id, **provenance})
        ended_run_id = _kb._end_run(
            conn, task_id, outcome="interrupted", status="interrupted", error=error,
            metadata=provenance,
        )
        _kb._append_event(
            conn, task_id, event_kind, payload, run_id=ended_run_id or run_id,
        )
    return True


def pause_host_interrupted_runs(conn: sqlite3.Connection) -> list[str]:
    """Pause locally owned ``running`` cards left behind by a replaced host instantiation.

    Must run BEFORE every other reclaim path: a PID that was alive before a
    reboot may already belong to an unrelated process, so nothing here may
    defer on, probe, or signal it. Returns the paused task ids.
    """
    current_epoch = _kr.current_host_epoch()
    if not current_epoch:
        # No readable identity (non-Linux, no /proc): never fail closed on a guess.
        return []
    paused: list[str] = []
    for row in _locally_owned_running_runs(conn):
        recorded_epoch = _run_host_epoch(row)
        if not recorded_epoch or not _instantiation_changed(recorded_epoch, current_epoch):
            continue
        if pause_interrupted_run(
            conn, row["task_id"], int(row["run_id"]),
            reason=HOST_RESTART_BLOCK_REASON,
            error=(
                f"host instantiation changed ({recorded_epoch} -> {current_epoch}); "
                "worker not signalled, card paused for the operator"
            ),
            extra_payload={
                "host_epoch": current_epoch,
                "recorded_host_epoch": recorded_epoch,
            },
        ):
            paused.append(row["task_id"])
        else:
            _kb._log.debug(
                "kanban reclaim: host-interruption pause for %s lost its compare-and-swap",
                row["task_id"],
            )
    return paused


def _record_task_failure(
    conn: sqlite3.Connection,
    task_id: str,
    error: str,
    *,
    outcome: str,
    failure_limit: int = None,
    force_trip: bool = False,
    release_claim: bool = False,
    end_run: bool = False,
    event_payload_extra: Optional[dict] = None,
    infrastructure: bool = False,
) -> bool:
    """Record a non-success outcome and maybe trip the circuit breaker; every
    non-success path funnels through here so ``consecutive_failures`` stays
    consistent. Returns True when the task was auto-blocked.

    ``release_claim=True, end_run=True``: spawn-failure path (task still
    running with an open run — restore source phase or ``blocked``, release
    claim, close run). Both False: timeout/crash path (caller already restored
    the phase and closed the run; only the counter moves, a trip flips to
    ``blocked`` + ``gave_up``). Threshold: per-task ``max_retries`` >
    ``failure_limit`` > ``DEFAULT_FAILURE_LIMIT``. ``force_trip`` trips
    unconditionally (caller applied its own bounded-retry policy).

    ``infrastructure=True``: the host refused the spawn (no restart-safe scope,
    #114720) — nothing about the card ran, so the run and event are recorded
    with ``infrastructure: true`` but ``consecutive_failures`` is left alone and
    the breaker never trips; the card stays retryable and
    :func:`check_respawn_guard` spaces the retries.
    """
    if failure_limit is None:
        failure_limit = DEFAULT_FAILURE_LIMIT
    error = error[:500]
    with _kb.write_txn(conn):
        row = conn.execute(
            "SELECT consecutive_failures, status, max_retries, current_run_id "
            "FROM tasks WHERE id = ?", (task_id,),
        ).fetchone()
        if row is None:
            return False
        retry_status = (
            _kb._retry_status_for_run(conn, task_id, row["current_run_id"])
            if release_claim
            else ("review" if row["status"] == "review" else "ready")
        )
        failures = int(row["consecutive_failures"]) + (0 if infrastructure else 1)

        # Per-task override wins over caller-supplied and default thresholds.
        task_override = _kb._row_get(row, "max_retries")
        if task_override is not None:
            effective_limit, limit_source = int(task_override), "task"
        else:
            effective_limit, limit_source = int(failure_limit), "dispatcher"

        if infrastructure or not (force_trip or failures >= effective_limit):
            if release_claim:
                # Spawn path: restore the claimed source phase + clear claim.
                conn.execute(
                    "UPDATE tasks SET status = ?, claim_lock = NULL, "
                    "claim_expires = NULL, worker_pid = NULL, worker_started_at = NULL, "
                    "consecutive_failures = ?, last_failure_error = ? "
                    "WHERE id = ? AND status = 'running'",
                    (retry_status, failures, error, task_id),
                )
            else:
                conn.execute(
                    "UPDATE tasks SET consecutive_failures = ?, "
                    "last_failure_error = ? WHERE id = ?",
                    (failures, error, task_id),
                )
            # Timeout/crash path's caller already emitted its own event.
            if end_run:
                detail = {"failures": failures, "retry_status": retry_status}
                if infrastructure:
                    detail["infrastructure"] = True
                run_id = _kb._end_run(
                    conn, task_id, outcome=outcome, status=outcome, error=error, metadata=detail,
                )
                _kb._append_event(conn, task_id, outcome, {"error": error, **detail}, run_id=run_id)
            return False

        # Spawn path (release_claim) is still running and also clears claim
        # state; the timeout/crash path already did.
        conn.execute(
            "UPDATE tasks SET status = 'blocked', "
            + ("claim_lock = NULL, claim_expires = NULL, worker_pid = NULL, worker_started_at = NULL, "
               if release_claim else "")
            + "consecutive_failures = ?, last_failure_error = ? "
            "WHERE id = ? AND status IN ('running', 'ready', 'review')",
            (failures, error, task_id),
        )
        payload = {
            "failures": failures,
            "effective_limit": effective_limit,
            "limit_source": limit_source,
            "error": error,
            "trigger_outcome": outcome,
            "retry_status": retry_status,
        }
        run_id = None
        if end_run:
            # Only the spawn path has an open run to close.
            run_id = _kb._end_run(
                conn, task_id, outcome="gave_up", status="gave_up", error=error,
                metadata={
                    "failures": failures,
                    "trigger_outcome": outcome,
                    "effective_limit": effective_limit,
                    "limit_source": limit_source,
                    "retry_status": retry_status,
                },
            )
        if force_trip:
            # The caller applied its own bounded policy, so the counter cannot
            # judge this block: ``recompute_ready`` holds it for an operator.
            payload["sticky"] = True
        if event_payload_extra:
            payload.update(event_payload_extra)
        _kb._append_event(conn, task_id, "gave_up", payload, run_id=run_id)
        return True


def _set_worker_pid(
    conn: sqlite3.Connection, task_id: str, pid: int, *,
    runtime_identity: Optional[Mapping[str, Any]] = None,
    preparation_id: Optional[str] = None,
) -> None:
    """Record the spawned child's PID only while its run still owns the claim."""
    started_at = _process_fingerprint(int(pid)) or UNVERIFIED_WORKER_FINGERPRINT
    with _kb.write_txn(conn):
        task_row = conn.execute(
            "SELECT current_run_id, claim_lock, status FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        run_id = task_row["current_run_id"] if task_row else None
        if (
            task_row is None
            or task_row["status"] != "running"
            or run_id is None
            or task_row["claim_lock"] is None
        ):
            return
        cur = conn.execute(
            "UPDATE tasks SET worker_pid = ?, worker_started_at = ? "
            "WHERE id = ? AND status = 'running' AND current_run_id = ? "
            "AND claim_lock IS ?",
            (int(pid), started_at, task_id, int(run_id), task_row["claim_lock"]),
        )
        if cur.rowcount != 1:
            return
        event = {"pid": int(pid)}
        row = conn.execute("SELECT metadata FROM task_runs WHERE id = ?", (run_id,)).fetchone()
        metadata = {}
        if row and row["metadata"]:
            try:
                parsed = json.loads(row["metadata"])
                metadata = parsed if isinstance(parsed, dict) else {}
            except (TypeError, json.JSONDecodeError):
                metadata = {}
        if runtime_identity is not None:
            identity = (
                runtime_identity.as_dict()
                if hasattr(runtime_identity, "as_dict")
                else dict(runtime_identity)
            )
            metadata["runtime_identity"] = identity
            event["runtime_identity"] = identity
        if preparation_id:
            metadata["preparation_id"] = str(preparation_id)
            event["preparation_id"] = str(preparation_id)
        conn.execute(
            "UPDATE task_runs SET worker_pid = ?, worker_started_at = ?, metadata = ? WHERE id = ? AND ended_at IS NULL",
            (int(pid), started_at, _kb._json_or_null(metadata or None), int(run_id)),
        )
        _kb._append_event(conn, task_id, "spawned", event, run_id=run_id)


def adopt_worker_pid(conn: sqlite3.Connection, task_id: str, run_id: int, pid: int) -> bool:
    """Register a worker only while its host-local run still owns the card."""
    started_at = _process_fingerprint(int(pid)) or UNVERIFIED_WORKER_FINGERPRINT
    with _kb.write_txn(conn):
        row = conn.execute(
            "SELECT status, current_run_id, worker_pid, claim_lock FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        if row is None or row["status"] != "running" or row["current_run_id"] != int(run_id):
            return False
        if not (row["claim_lock"] or "").startswith(_kb._host_prefix()):
            return False
        if row["worker_pid"] is not None:
            return int(row["worker_pid"]) == int(pid)
        conn.execute("UPDATE tasks SET worker_pid = ?, worker_started_at = ? WHERE id = ?",
                     (int(pid), started_at, task_id))
        conn.execute("UPDATE task_runs SET worker_pid = ?, worker_started_at = ? WHERE id = ?",
                     (int(pid), started_at, int(run_id)))
        _kb._append_event(conn, task_id, "worker_registered", {"pid": int(pid), "started_at": started_at},
                          run_id=int(run_id))
    return True


def _clear_failure_counter(conn: sqlite3.Connection, task_id: str) -> None:
    """Reset the unified consecutive-failures counter.

    Called from ``complete_task`` on success. NOT called on spawn success: a
    spawn proves the worker could start, not that the run will succeed, so
    timeouts and crashes must accumulate across spawn boundaries.
    """
    with _kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET consecutive_failures = 0, "
            "last_failure_error = NULL WHERE id = ?",
            (task_id,),
        )


PR_CONTINUATION_EVENT = "pr_continuation_authorized"


def _pr_urls_in(text: Optional[str]) -> list[str]:
    """Distinct GitHub PR URLs mentioned in one comment body (arrival order)."""
    if not text:
        return []
    return list(dict.fromkeys(_RESPAWN_GUARD_PR_URL_RE.findall(_kb._lossy_text(text))))


def _recorded_pr_urls(
    conn: sqlite3.Connection, task_ids: list[str],
) -> dict[str, list[str]]:
    """Sorted distinct GitHub PR URLs per task, from all of their comments.

    One query for many tasks: a held card's recovery command names its recorded
    set, so a board view must not pay a query per held card.
    """
    ids = list(dict.fromkeys(task_ids))
    if not ids:
        return {}
    marks = ",".join("?" * len(ids))
    found: dict[str, set[str]] = {task_id: set() for task_id in ids}
    for row in conn.execute(
        f"SELECT task_id, body FROM task_comments WHERE task_id IN ({marks})",
        tuple(ids),
    ).fetchall():
        found.setdefault(row["task_id"], set()).update(_pr_urls_in(row["body"]))
    return {task_id: sorted(urls) for task_id, urls in found.items()}


def _task_pr_urls(conn: sqlite3.Connection, task_id: str) -> list[str]:
    """Sorted distinct GitHub PR URLs recorded anywhere in the task's comments."""
    return _recorded_pr_urls(conn, [task_id]).get(task_id, [])


def _guard_state(
    conn: sqlite3.Connection, task_ids: list[str], *, now: Optional[int] = None,
) -> dict[str, dict[str, Any]]:
    """Every input the respawn guard reads, for many tasks, in a fixed number of queries.

    The guard's semantics live in :func:`_guard_reason`; this exists so a
    board-wide projection does not run a query per card. A missing task simply
    has no entry, which the evaluator treats as "no guard".
    """
    ids = list(dict.fromkeys(task_ids))
    if not ids:
        return {}
    now = int(time.time()) if now is None else int(now)
    marks = ",".join("?" * len(ids))
    state: dict[str, dict[str, Any]] = {}
    for row in conn.execute(
        f"SELECT id, status, claim_lock, version, goal_revision_id, last_failure_error, "
        f"assignee FROM tasks WHERE id IN ({marks})",
        tuple(ids),
    ).fetchall():
        state[row["id"]] = {
            "task_id": row["id"],
            "status": row["status"],
            "assignee": row["assignee"],
            "claim_lock": row["claim_lock"],
            "version": int(row["version"] or 1),
            "goal_revision_id": _kb._opt_int(row["goal_revision_id"]),
            "last_failure_error": row["last_failure_error"],
            "latest_outcome": None,
            "latest_metadata": {},
            "latest_ended_at": None,
            "last_completed_at": None,
            "last_requeue_at": None,
            "newest_authorization": None,
            "max_completed_run_id": None,
            "recent_pr_urls": [],
        }
    if not state:
        return state
    for row in conn.execute(
        "SELECT task_id, outcome, ended_at, metadata FROM ("
        "  SELECT task_id, outcome, ended_at, metadata, ROW_NUMBER() OVER ("
        "    PARTITION BY task_id ORDER BY ended_at DESC, id DESC) AS position"
        f"  FROM task_runs WHERE task_id IN ({marks}) AND ended_at IS NOT NULL"
        ") WHERE position = 1",
        tuple(ids),
    ).fetchall():
        entry = state[row["task_id"]]
        entry["latest_outcome"] = row["outcome"]
        entry["latest_ended_at"] = _kb._opt_int(row["ended_at"])
        entry["latest_metadata"] = _kb._json_dict(row["metadata"])
    # Native workers are prepared before claim: a refused launch has no run,
    # but must observe the same infrastructure cooldown as a post-claim refusal.
    for row in conn.execute(
        "SELECT task_id, kind, created_at, payload FROM ("
        "  SELECT task_id, kind, created_at, payload, ROW_NUMBER() OVER ("
        "    PARTITION BY task_id ORDER BY id DESC) AS position"
        f"  FROM task_events WHERE task_id IN ({marks}) AND kind IN ('spawn_refused', 'claimed')"
        ") WHERE position = 1 AND kind = 'spawn_refused'",
        tuple(ids),
    ).fetchall():
        entry = state[row["task_id"]]
        entry["latest_outcome"] = "spawn_failed"
        entry["latest_ended_at"] = int(row["created_at"])
        entry["latest_metadata"] = _kb._json_dict(row["payload"])
    for row in conn.execute(
        f"SELECT task_id, MAX(ended_at) AS last_at FROM task_runs "
        f"WHERE task_id IN ({marks}) AND outcome = 'completed' AND ended_at >= ? "
        f"GROUP BY task_id",
        (*ids, now - _RESPAWN_GUARD_SUCCESS_WINDOW),
    ).fetchall():
        state[row["task_id"]]["last_completed_at"] = _kb._opt_int(row["last_at"])
    for row in conn.execute(
        f"SELECT task_id, MAX(id) AS last_id FROM task_runs "
        f"WHERE task_id IN ({marks}) AND outcome = 'completed' GROUP BY task_id",
        tuple(ids),
    ).fetchall():
        state[row["task_id"]]["max_completed_run_id"] = _kb._opt_int(row["last_id"])
    requeue_kinds = ("status", "promoted", "promoted_manual", "unblocked", "reclaimed")
    for row in conn.execute(
        f"SELECT task_id, MAX(created_at) AS last_at FROM task_events "
        f"WHERE task_id IN ({marks}) "
        f"AND kind IN ({','.join('?' * len(requeue_kinds))}) GROUP BY task_id",
        (*ids, *requeue_kinds),
    ).fetchall():
        state[row["task_id"]]["last_requeue_at"] = _kb._opt_int(row["last_at"])
    for row in conn.execute(
        "SELECT task_id, id, payload FROM ("
        "  SELECT task_id, id, payload, ROW_NUMBER() OVER ("
        "    PARTITION BY task_id ORDER BY id DESC) AS position"
        f"  FROM task_events WHERE task_id IN ({marks}) AND kind = ?"
        ") WHERE position = 1",
        (*ids, PR_CONTINUATION_EVENT),
    ).fetchall():
        state[row["task_id"]]["newest_authorization"] = row
    for row in conn.execute(
        f"SELECT task_id, body FROM task_comments "
        f"WHERE task_id IN ({marks}) AND created_at >= ?",
        (*ids, now - _RESPAWN_GUARD_PR_WINDOW),
    ).fetchall():
        state[row["task_id"]]["recent_pr_urls"].extend(_pr_urls_in(row["body"]))
    return state


def _continuation_from_state(entry: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Usable ``pr_continuation_authorized`` for one guard-state entry, else None.

    The authorization covers the PR URL set recorded when it was written, and
    survives unsuccessful retries. It stops being usable once the effective
    goal is revised or a run newer than the authorization completes — an old
    success must not enable a later duplicate worker. A missing or malformed
    payload fails closed (no authorization).
    """
    if entry is None or entry["newest_authorization"] is None:
        return None
    row = entry["newest_authorization"]
    payload = _kb._json_dict(_kb._row_get(row, "payload"))
    urls = payload.get("pr_urls")

    def _int_key(name: str) -> Optional[int]:
        value = payload.get(name)
        if isinstance(value, bool) or not isinstance(value, int):
            return None
        return value

    goal_revision_id = _int_key("goal_revision_id")
    after_run_id = _int_key("after_run_id")
    through_comment_id = _int_key("through_comment_id")
    if (
        goal_revision_id is None
        or after_run_id is None
        or through_comment_id is None
        or not isinstance(urls, list)
        or not urls
        or not all(isinstance(url, str) and url for url in urls)
    ):
        return None
    if entry["goal_revision_id"] != goal_revision_id:
        return None
    newest_completed = entry["max_completed_run_id"]
    if newest_completed is not None and newest_completed > after_run_id:
        return None
    return {
        "event_id": int(row["id"]),
        "actor": payload.get("actor"),
        "reason": payload.get("reason"),
        "goal_revision_id": goal_revision_id,
        "through_comment_id": through_comment_id,
        "after_run_id": after_run_id,
        "pr_urls": sorted(dict.fromkeys(urls)),
    }


def _pr_continuation(
    conn: sqlite3.Connection, task_id: str,
) -> Optional[dict[str, Any]]:
    """Newest usable ``pr_continuation_authorized`` for ``task_id``, else None."""
    return _continuation_from_state(_guard_state(conn, [task_id]).get(task_id))


def _uncovered_recent_pr(entry: dict[str, Any]) -> bool:
    """True while a recent PR comment URL is outside the live authorized set."""
    continuation = _continuation_from_state(entry)
    authorized = set(continuation["pr_urls"]) if continuation else set()
    return any(url not in authorized for url in entry["recent_pr_urls"])


def _active_pr_guard_reason(
    conn: sqlite3.Connection, task_id: str, *, now: Optional[int] = None,
) -> Optional[str]:
    """``"active_pr"`` while a recent PR comment is not covered by a live
    ``continue_existing_pr`` authorization; None when the task has none.

    Keyed on the recorded URL set, not on wall-clock ordering: a later comment
    repeating an already-authorized URL stays permitted, a new URL guards even
    in the same second as the authorization.
    """
    entry = _guard_state(conn, [task_id], now=now).get(task_id)
    if entry is None or not _uncovered_recent_pr(entry):
        return None
    return "active_pr"


def _record_respawn_guard(
    conn: sqlite3.Connection, task_id: str, reason: str,
) -> None:
    """Append ``respawn_guarded`` once per guard episode (caller holds a write txn).

    An unchanged hold appends nothing further — every tick still reports the
    guard in :class:`DispatchResult`, but the event log stays bounded without a
    heartbeat row. Any intervening task activity starts a new episode.
    """
    row = conn.execute(
        "SELECT kind, payload FROM task_events WHERE task_id = ? ORDER BY id DESC LIMIT 1",
        (task_id,),
    ).fetchone()
    if (
        row is not None
        and _kb._row_get(row, "kind") == "respawn_guarded"
        and _kb._json_dict(_kb._row_get(row, "payload")).get("reason") == reason
    ):
        return
    _kb._append_event(conn, task_id, "respawn_guarded", {"reason": reason})


def _last_claim_rejected_reason(
    conn: sqlite3.Connection, task_id: str,
) -> Optional[str]:
    """Reason on the task's newest event when that event is a ``claim_rejected``."""
    row = conn.execute(
        "SELECT kind, payload FROM task_events WHERE task_id = ? ORDER BY id DESC LIMIT 1",
        (task_id,),
    ).fetchone()
    if row is None or _kb._row_get(row, "kind") != "claim_rejected":
        return None
    return _kb._json_dict(_kb._row_get(row, "payload")).get("reason")


def _guard_reason(
    entry: Optional[dict[str, Any]], *, lane: str, now: int,
) -> Optional[str]:
    """Guard reason for one guard-state entry, or None when it may be spawned.

    Priority: ``"rate_limit_cooldown"`` (latest run ``rate_limited`` within the
    cooldown; checked BEFORE ``blocker_auth`` because the requeue stamps a
    quota-flavored ``last_failure_error`` that would otherwise park the task
    forever — that path never increments ``consecutive_failures``),
    ``"blocker_auth"`` (quota/auth pattern; the breaker still trips eventually),
    then for the ready lane only ``"recent_success"`` (completed run within the
    window, unless a re-queue event arrived after it — a deliberate re-run — or a
    live ``continue_existing_pr`` authorization covers it) and ``"active_pr"`` (a
    recent comment URL outside the task's live authorized PR set). The review
    lane skips the last two: they are the *inputs* to a review handoff. Stale /
    dead claim locks are NOT a guard reason — the reclaim passes own those.

    The two rate-limit early-return branches still skip ``blocker_auth`` and
    ``recent_success``, but do apply the PR fence, so live diagnostics never
    report a task dispatchable that :func:`claim_task` would refuse.
    Authorization never bypasses an active cooldown.
    """
    if entry is None:
        return None

    def pr_hold() -> Optional[str]:
        """PR fence for the ready lane; the review lane treats PRs as inputs."""
        if lane == "review" or not _uncovered_recent_pr(entry):
            return None
        return "active_pr"

    if entry["latest_outcome"] == "spawn_failed" and entry["latest_metadata"].get("infrastructure"):
        cooldown = _kb._resolve_rate_limit_cooldown_seconds()
        ended_at = entry["latest_ended_at"]
        if ended_at is not None and (now - ended_at) < cooldown:
            return "infrastructure_cooldown"

    if entry["latest_outcome"] == "rate_limited":
        cooldown = _kb._resolve_rate_limit_cooldown_seconds()
        if cooldown <= 0:
            # Cooldown disabled — respawn immediately, skipping blocker_auth so
            # the stamped rate-limit text doesn't re-trap the task.
            return pr_hold()
        ended_at = entry["latest_ended_at"]
        if ended_at is not None and (now - ended_at) < cooldown:
            return "rate_limit_cooldown"
        # Cooldown elapsed — return early so blocker_auth doesn't catch the
        # stamped rate-limit text; this path intentionally retries forever
        # (spaced by the cooldown) until quota returns or a real run supersedes it.
        return pr_hold()

    err = _kb._lossy_text(entry["last_failure_error"])
    if err and entry["latest_outcome"] != "crashed" and _RESPAWN_BLOCKER_RE.search(err):
        return "blocker_auth"

    if lane == "review":
        return None

    completed_at = entry["last_completed_at"]
    if completed_at:
        requeued_after = entry["last_requeue_at"]
        if (requeued_after is None or requeued_after < completed_at) and (
            _continuation_from_state(entry) is None
        ):
            return "recent_success"

    return pr_hold()


def check_respawn_guard(
    conn: sqlite3.Connection, task_id: str, *, lane: str = "ready",
) -> Optional[str]:
    """Return a guard reason if ``task_id`` should NOT be re-spawned, else None.

    Called per ready/review row before any claim attempt; see
    :func:`_guard_reason` for the priority order.
    """
    now = int(time.time())
    return _guard_reason(
        _guard_state(conn, [task_id], now=now).get(task_id), lane=lane, now=now,
    )


_GUARD_RECOVERY: dict[str, str] = {
    "infrastructure_cooldown": (
        "The host could not provide restart-safe worker placement; repair the user service "
        "scope. Dispatch retries after the cooldown without charging the task's failure budget."
    ),
    "active_pr": (
        "A recent comment records an existing GitHub PR that no live continuation "
        "authorization covers, so the dispatcher will not start a worker for this "
        "card. Acknowledge the existing PR with the explicit 'continue_existing_pr' "
        "transition (only with the user's authorization) instead of opening a "
        "replacement PR."
    ),
    "rate_limit_cooldown": (
        "The last run hit a provider rate limit. The dispatcher retries by itself "
        "once the cooldown elapses; no action is required."
    ),
    "blocker_auth": (
        "The last run failed with a quota/auth error; immediate retries cannot help, "
        "and the circuit breaker will auto-block the card after the failure limit."
    ),
    "recent_success": (
        "A run already completed successfully inside the respawn guard window, so "
        "the dispatcher treats a new spawn as duplicate work. Re-queue the card "
        "deliberately, or authorize continuation of its recorded PR, to run it again."
    ),
}


def _guard_command(
    reason: str, task_id: str, version: int, *, board: Optional[str],
    pr_urls: Optional[list[str]] = None,
) -> str:
    """The runnable recovery command for one guard reason, or ``""``.

    Only a reason with a single safe command gets one; the rest are guidance
    only. Surfaces offering a copy action paste this field verbatim, so it must
    never carry explanation prose.

    The acknowledgment names the PR URLs it authorizes. A comment does not bump
    the card's version, so a URL that lands after this command was printed would
    otherwise be authorized without the operator ever seeing it; the transition
    requires the set to match exactly.
    """
    if reason != "active_pr":
        return ""
    board_flag = f"--board {board} " if board else ""
    authorized = " ".join(
        f"--authorized-pr {shlex.quote(url)}" for url in (pr_urls or [])
    )
    return (
        f"hermes kanban {board_flag}update {task_id} --expected-version {version} "
        f"--transition continue_existing_pr {authorized} "
        "--reason 'Explicitly authorized: continue the existing PR; do not open another PR'"
    )


def respawn_guard_alert(holds: list[tuple[str, str]], ticks: int) -> str:
    """One bounded alert line for a run of guard-held ticks.

    ``holds`` is ``[(task_label, reason), ...]`` for the latest tick. Reason
    counts are aggregated and at most five task labels are named, so a wide
    board produces one actionable line instead of a per-task log storm.
    """
    counts = Counter(reason for _, reason in holds)
    reasons = ", ".join(f"{reason}={count}" for reason, count in sorted(counts.items()))
    labels = ", ".join(label for label, _ in holds[:5])
    return (
        f"respawn guard held ready work for {ticks} consecutive ticks "
        f"(reasons: {reasons}; tasks: {labels}). Run `hermes kanban diagnostics` "
        f"for the per-task recovery action."
    )


def _guard_hold(
    entry: Optional[dict[str, Any]], *, now: int,
) -> Optional[tuple[dict[str, Any], str]]:
    """``(entry, reason)`` when the guard holds this card, else None.

    Mirrors the dispatch order: `_dispatch_lane_task` classifies an unassigned
    card (`skipped_unassigned`) or one with no real profile
    (`skipped_nonspawnable`) BEFORE it consults the guard, so the guard is not
    what stops those cards and its recovery command would not make them spawn.
    `profile_exists` also rejects an empty name, so the unassigned case must not
    reach it; the stranded-in-ready diagnostic names the assignee cause instead.
    """
    if (
        entry is None
        or entry["claim_lock"] is not None
        or entry["status"] not in ("ready", "review")
    ):
        return None
    assignee = (entry["assignee"] or "").strip()
    profile_exists = _profile_exists_fn()
    if not assignee or (profile_exists is not None and not profile_exists(assignee)):
        return None
    reason = _guard_reason(
        entry, lane="review" if entry["status"] == "review" else "ready", now=now,
    )
    if reason is None:
        return None
    return entry, reason


def _guard_payload(
    entry: dict[str, Any], reason: str, *, board: str, pr_urls: list[str],
) -> dict[str, str]:
    """The projection payload for one held card.

    ``board`` is the slug the caller resolved before opening its connection; an
    empty value means the caller did not resolve one and the command carries no
    ``--board``.
    """
    return {
        "reason": reason,
        "recovery": _GUARD_RECOVERY.get(
            reason, f"The dispatcher respawn guard is holding this task ({reason}).",
        ),
        "command": _guard_command(
            reason, entry["task_id"], entry["version"], board=board, pr_urls=pr_urls,
        ),
    }


def get_dispatch_guard(
    conn: sqlite3.Connection, task_id: str, *, board: Optional[str] = None,
) -> Optional[dict[str, str]]:
    """Live dispatch hold for a ready/review task, else None.

    A projection of the guard as it stands *now* — never inferred from
    historical ``respawn_guarded`` events, which a later authorization leaves
    stale. Returns ``{"reason", "recovery", "command"}`` — operator guidance plus
    the runnable command (``""`` when the reason has none) — or None for missing,
    claimed, or non-dispatchable tasks, including a card whose assignee is not a
    real profile (the dispatcher classifies those as nonspawnable first).
    """
    now = int(time.time())
    hold = _guard_hold(_guard_state(conn, [task_id], now=now).get(task_id), now=now)
    if hold is None:
        return None
    entry, reason = hold
    pr_urls = (
        _task_pr_urls(conn, entry["task_id"]) if reason == "active_pr" else []
    )
    # The caller resolves the board before opening the connection and passes it
    # here, so the command names the board the card actually lives on rather
    # than re-reading a pointer another process may have moved.
    return _guard_payload(entry, reason, board=board or "", pr_urls=pr_urls)


def get_dispatch_guards(
    conn: sqlite3.Connection, task_ids: list[str], *, board: Optional[str] = None,
) -> dict[str, dict[str, str]]:
    """:func:`get_dispatch_guard` for many tasks in a fixed number of queries.

    Board and fleet views project every card they render, so the per-task form
    would put one query set per card back into each request. Evaluation is
    shared with the single-task form; only the state fetch is batched.
    """
    now = int(time.time())
    holds: dict[str, tuple[dict[str, Any], str]] = {}
    for task_id, entry in _guard_state(conn, list(task_ids), now=now).items():
        hold = _guard_hold(entry, now=now)
        if hold is not None:
            holds[task_id] = hold
    if not holds:
        return {}
    pr_holds = [task_id for task_id, (_, reason) in holds.items() if reason == "active_pr"]
    recorded = _recorded_pr_urls(conn, pr_holds)
    return {
        task_id: _guard_payload(
            entry, reason, board=board or "", pr_urls=recorded.get(task_id, []),
        )
        for task_id, (entry, reason) in holds.items()
    }


def _is_handoff_event(kind: str, payload: Optional[str]) -> bool:
    """Only an ``assigned`` event that moves the card to a DIFFERENT profile is
    a handoff. A no-op re-assign (dev→dev via CLI/dashboard/``reassign
    --reclaim``), an unassign, or the dispatcher's own
    ``kanban.default_assignee`` write would otherwise lift ``active_pr`` for
    the very implementer that opened the PR. Events without ``from`` (written
    before it was recorded) are not trusted as handoffs — fail closed."""
    if kind != "assigned":
        return True
    data = _kb._json_or(payload, {})
    if not isinstance(data, dict) or data.get("source") == "kanban.default_assignee":
        return False
    to = data.get("assignee")
    return bool(to) and "from" in data and data["from"] != to


def _profile_exists_fn() -> Optional[Callable[[str], bool]]:
    """``hermes_cli.profiles.profile_exists``, or ``None`` when it cannot be
    imported (local import avoids a cycle; callers fall back to trusting the
    assignee).

    When ``kanban.dispatch_profiles`` is set (#110995) the returned predicate
    additionally requires the assignee to be listed, fail-closed — so a card
    assigned to ``default`` is only claimable by homes that opted into it.
    Foreign assignees land in the existing ``skipped_nonspawnable`` bucket.
    """
    try:
        from hermes_cli.profiles import normalize_profile_name, profile_exists
    except Exception:
        return None
    allowlist = _dispatch_profile_allowlist(normalize_profile_name)
    if allowlist is None:
        return profile_exists

    def _gated(name: str) -> bool:
        try:
            canon = normalize_profile_name(name)
        except ValueError:
            return False
        return canon in allowlist and bool(profile_exists(name))

    return _gated


def _dispatch_profile_allowlist(normalize_profile_name) -> Optional[frozenset]:
    """Per-home claim allowlist ``kanban.dispatch_profiles`` (#110995).

    On a shared board (one ``kanban.db`` mounted across several Hermes homes),
    every home's ``profile_exists`` returns True for ``default`` — the root
    profile every home has — so a card assigned to ``default`` is claimable by
    every home's dispatcher. A home opts out of foreign claims by declaring
    which assignees it may claim::

        kanban:
          dispatch_profiles: ["sage", "researcher"]   # or "sage,researcher"

    Returns ``None`` only when the key is absent from the user config (upstream
    behavior: any existing profile is claimable). A present value is
    fail-closed: an empty list, ``null`` or a bare ``dispatch_profiles:`` claims
    nothing. The user layer is read without the ``DEFAULT_CONFIG`` merge (whose
    ``None`` placeholder would make the key look present in every home), and a
    config read that raises also claims nothing — a corrupt config on a shared
    board must never widen this home's claim scope silently (#113620).
    """
    try:
        from hermes_cli.config_effective import load_user_config_effective
        kanban = (load_user_config_effective(fail_closed=True) or {}).get("kanban", {})
    except Exception as exc:
        _kb._log.warning(
            "kanban: could not read kanban.dispatch_profiles (%s: %s) — "
            "this home claims no cards until the config is readable",
            type(exc).__name__, exc,
        )
        return frozenset()
    if not isinstance(kanban, Mapping) or "dispatch_profiles" not in kanban:
        return None
    raw = kanban["dispatch_profiles"]
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        _kb._log.warning(
            "kanban: kanban.dispatch_profiles is present but empty — this home "
            "claims no cards; omit the key to allow any existing profile"
        )
        return frozenset()
    names = [str(n) for n in raw] if isinstance(raw, (list, tuple)) else str(raw).split(",")
    allowed = set()
    for n in names:
        try:
            allowed.add(normalize_profile_name(n))
        except ValueError:
            continue
    return frozenset(allowed)


def dispatch_profile_allowlist_summary() -> str:
    """Human-readable resolution of ``kanban.dispatch_profiles`` for this home.

    Surfaced by ``hermes kanban diagnostics`` so an operator on a shared board
    can see what a home believes it may claim (#113620): ``any`` (key absent),
    the sorted allowed names, or ``none (fail-closed: ...)``.
    """
    try:
        from hermes_cli.profiles import normalize_profile_name
    except Exception as exc:
        return f"none (fail-closed: profiles unavailable: {exc})"
    allowlist = _dispatch_profile_allowlist(normalize_profile_name)
    if allowlist is None:
        return "any"
    if allowlist:
        return ", ".join(sorted(allowlist))
    return ("none (fail-closed: kanban.dispatch_profiles is present but names no valid "
            "profile, or the config could not be read — omit the key to allow any)")


def _spawnable_ids(conn: sqlite3.Connection, status: str) -> list[str]:
    """Ids of ``status``+assigned+unclaimed tasks the dispatcher would spawn for.

    Lets health telemetry tell "stuck" (``0 spawned`` with spawnable work) from
    "correctly idle" (only control-plane lanes waiting on ``claim_task``), and
    lets a caller subtract the cards it already explains. Falls back to "any
    assigned" when ``profile_exists`` is unimportable.
    """
    rows = conn.execute(
        "SELECT id, assignee FROM tasks "
        "WHERE status = ? AND assignee IS NOT NULL AND claim_lock IS NULL "
        "ORDER BY priority DESC, created_at ASC",
        (status,),
    ).fetchall()
    if not rows:
        return []
    profile_exists = _profile_exists_fn()
    if profile_exists is None:
        # Can't introspect — assume spawnable, preserve legacy behavior.
        return [row["id"] for row in rows]
    return [row["id"] for row in rows if profile_exists(row["assignee"])]


def spawnable_lane_ids(conn: sqlite3.Connection) -> list[str]:
    """Spawnable ready ids plus review ids when review dispatch is on.

    Same gate as :func:`ready_nonempty` in the gateway dispatcher: the review
    column counts only while this process would claim from it.
    """
    ids = _spawnable_ids(conn, "ready")
    if review_dispatch_enabled():
        ids += _spawnable_ids(conn, "review")
    return ids


def has_spawnable_ready(conn: sqlite3.Connection) -> bool:
    """True iff a ready+assigned+unclaimed task maps to a real Hermes profile.

    Lets health telemetry tell "stuck" (``0 spawned`` with spawnable work) from
    "correctly idle" (only control-plane lanes waiting on ``claim_task``). Falls
    back to "any assigned" when ``profile_exists`` is unimportable.
    """
    return bool(_spawnable_ids(conn, "ready"))


def has_spawnable_review(conn: sqlite3.Connection) -> bool:
    """:func:`has_spawnable_ready` for the review column."""
    return bool(_spawnable_ids(conn, "review"))


def review_dispatch_enabled() -> bool:
    """Whether review tasks dispatch automatically. Default true (Hermes ships
    ``sdlc-review``); operators disable it for human-only review boards.
    """
    try:
        from hermes_cli.config import load_config
        return bool((load_config() or {}).get("kanban", {}).get("review_dispatch", True))
    except Exception:
        return True


# Memory-aware dispatch guard: an uncapped board once OOM'd a 1 GiB host. Two
# safeguards — a memory-DERIVED default cap when none is configured
# (``resolve_max_in_progress``) and a live memory-PRESSURE guard inside the
# tick (``_memory_pressure_level``) because a static cap can't see other
# tenants. Both fail open: non-Linux / read error → no cap / "unknown".

# Assumed per-worker footprint for the derived cap; deliberately conservative
# so the cap errs toward fewer workers on small VMs.
MEMORY_GUARD_MB_PER_WORKER = 512

# Derived default bounds: never below 2 (smallest VM must still progress),
# never above 8 (more fan-out must be explicit in config).
DERIVED_MAX_IN_PROGRESS_FLOOR = 2
DERIVED_MAX_IN_PROGRESS_CEILING = 8


def _system_memory_sample() -> dict:
    """Best-effort system memory snapshot (KiB values), ``{}`` when unknown.

    Local import keeps ``kanban_db`` importable without the gateway package.
    Module-level indirection is also the test seam — conftest patches this to
    ``{}`` so results don't depend on the CI runner's live memory.
    """
    try:
        from gateway.lifecycle_ledger import sample_memory
        return sample_memory() or {}
    except Exception:
        return {}


def derive_default_max_in_progress(sample: Optional[Mapping[str, Any]] = None) -> Optional[int]:
    """Memory-derived default for ``kanban.max_in_progress`` when unset:
    ``clamp(MemTotal / MEMORY_GUARD_MB_PER_WORKER, FLOOR, CEILING)``. Returns
    ``None`` (no cap) when total memory is unknown, so macOS/Windows dev
    machines are unaffected.
    """
    if sample is None:
        sample = _system_memory_sample()
    total_kib = sample.get("mem_total_kib")
    if isinstance(total_kib, bool) or not isinstance(total_kib, int) or total_kib <= 0:
        return None
    workers = (total_kib // 1024) // MEMORY_GUARD_MB_PER_WORKER
    return max(DERIVED_MAX_IN_PROGRESS_FLOOR, min(workers, DERIVED_MAX_IN_PROGRESS_CEILING))


def resolve_max_in_progress(configured: Optional[int]) -> Optional[int]:
    """Effective global concurrency cap: explicit config wins, else the
    memory-derived default. All config-parsing callers route through this so
    both paths agree.
    """
    if configured is not None:
        return configured
    return derive_default_max_in_progress()


def configured_max_in_progress() -> Optional[int]:
    """Read ``kanban.max_in_progress`` from config, or None when unset/invalid.

    Shared so every dispatch entry point agrees on "explicitly configured": a
    positive integer wins, anything else falls through to the derived default.
    """
    try:
        from hermes_cli.config import load_config_readonly
        raw = (load_config_readonly() or {}).get("kanban", {}).get("max_in_progress")
    except Exception:
        return None
    if raw is None:
        return None
    try:
        ival = int(raw)
    except (TypeError, ValueError):
        return None
    return ival if ival >= 1 else None


def count_running_tasks(conn: sqlite3.Connection) -> int:
    """Number of tasks in ``status='running'``.

    Used by the multi-board sweep to count OTHER boards' workers against the
    host-level budget — the memory-derived cap bounds the machine, not the
    board. Fails open to 0 so a broken board doesn't brick dispatch on healthy ones.
    """
    try:
        return int(
            conn.execute(
                "SELECT COUNT(*) FROM tasks WHERE status = 'running'"
            ).fetchone()[0]
        )
    except Exception:
        return 0


def count_running_tasks_other_boards(board: Optional[str] = None) -> int:
    """Total ``running`` tasks across every board EXCEPT ``board``.

    Caps bound the HOST, but each board's tick only sees its own DB; without
    this a derived cap of N gets multiplied by the number of active boards.
    Boards are matched by resolved DB path, so ``HERMES_KANBAN_DB`` (pins every
    board to one file) yields 0. Fails open per board.
    """
    try:
        current_path = str(_kb.kanban_db_path(board=board).expanduser().resolve())
    except Exception:
        current_path = None
    try:
        boards = _kb.list_boards(include_archived=False)
    except Exception:
        return 0
    total = 0
    for meta in boards:
        slug = meta.get("slug") or _kb.DEFAULT_BOARD
        try:
            path = _kb.kanban_db_path(board=slug).expanduser()
            resolved = str(path.resolve())
            if current_path is not None and resolved == current_path:
                continue
            if not path.exists():
                continue
            other = _kbc.connect(board=slug)
            try:
                total += count_running_tasks(other)
            finally:
                with contextlib.suppress(Exception):
                    other.close()
        except Exception:
            continue
    return total


def _memory_pressure_level(sample: Optional[Mapping[str, Any]] = None) -> str:
    """Classify system memory pressure: ok/elevated/critical/unknown.

    Reuses :func:`gateway.memory_status.classify_pressure` so "critical" matches
    the dashboard banner and lifecycle-ledger OOM heuristics. ``unknown``
    (non-Linux, read failure) imposes no restriction — never brick dispatch
    where /proc is unavailable.
    """
    if sample is None:
        sample = _system_memory_sample()
    if not sample:
        return "unknown"
    try:
        from gateway.memory_status import classify_pressure
        return classify_pressure(sample.get("mem_available_kib"), sample.get("mem_total_kib"))
    except Exception:
        return "unknown"


def _freeze_runtime_identity() -> None:
    """Freeze the parent code identity when dispatching starts."""
    from hermes_cli.kanban_runtime import runtime_identity

    runtime_identity()


def dispatch_once(
    conn: sqlite3.Connection,
    *,
    spawn_fn=None,
    ttl_seconds: Optional[int] = None,
    dry_run: bool = False,
    max_spawn: Optional[int] = None,
    max_in_progress: Optional[int] = None,
    failure_limit: int = DEFAULT_FAILURE_LIMIT,
    stale_timeout_seconds: int = 0,
    board: Optional[str] = None,
    default_assignee: Optional[str] = None,
    max_in_progress_per_profile: Optional[int] = None,
    reconcile_orphans: bool = True,
    should_stop: Optional[Callable[[], bool]] = None,
    grant_guard: Optional[Callable[[], Any]] = None,
) -> DispatchResult:
    """Run one dispatcher tick under the board's single-writer lock.

    Wraps :func:`_dispatch_once_locked` in the non-blocking :func:`_dispatch_tick_lock`
    so two dispatchers on one ``kanban.db`` never race a write tick on WAL
    frames. The loser returns an empty ``DispatchResult`` with
    ``skipped_locked=True`` and writes nothing; the lock is keyed on the
    resolved DB path so unrelated boards tick in parallel.

    ``should_stop`` lets the owning gateway/daemon cancel in-flight launches as
    soon as it starts draining, so no worker is handed a card by a process that
    is going away.
    """
    _freeze_runtime_identity()
    def _locked_tick() -> DispatchResult:
        return _dispatch_once_locked(
            conn,
            spawn_fn=spawn_fn,
            ttl_seconds=ttl_seconds,
            dry_run=dry_run,
            max_spawn=max_spawn,
            max_in_progress=max_in_progress,
            failure_limit=failure_limit,
            stale_timeout_seconds=stale_timeout_seconds,
            board=board,
            default_assignee=default_assignee,
            max_in_progress_per_profile=max_in_progress_per_profile,
            reconcile_orphans=reconcile_orphans,
            should_stop=should_stop,
            grant_guard=grant_guard,
        )

    try:
        db_path = _kb.kanban_db_path(board=board)
    except Exception:
        # Must not lose the tick — fall through to an unguarded dispatch.
        result = _locked_tick()
        _kb._fire_dispatch_tick_hook(result, board=board, dry_run=dry_run)
        return result
    with _kbc._dispatch_tick_lock(db_path) as held:
        if not held:
            result = DispatchResult(skipped_locked=True)
        else:
            result = _locked_tick()
            # Still under the dispatch lock: periodic PASSIVE WAL checkpoint.
            _kbc._maybe_checkpoint_wal(conn, db_path)
    # Lock released. Fire the tick observer strictly OUTSIDE the critical
    # section: a slow subscriber must never stall a sibling dispatcher's tick.
    _kb._fire_dispatch_tick_hook(result, board=board, dry_run=dry_run)
    return result


def _call_spawn_fn(spawn_fn, task: Task, workspace: str, board: Optional[str]) -> Optional[int]:
    """Back-compat: older spawn_fn signatures (and test stubs) accept only
    ``(task, workspace)``; pass ``board`` only when the callable supports it."""
    import inspect
    try:
        sig = inspect.signature(spawn_fn)
        if "board" in sig.parameters:
            return spawn_fn(task, workspace, board=board)
        return spawn_fn(task, workspace)
    except (TypeError, ValueError):
        return spawn_fn(task, workspace)


def _note_claim_hold(
    conn: sqlite3.Connection, task_id: str, result: "DispatchResult",
) -> None:
    """Surface a claim-time PR fence as a guard hold in ``DispatchResult``.

    The pre-claim guard can pass and a fresh unauthorized PR URL can then arrive
    while a deferred worker is being prepared. ``claim_task`` re-checks inside its
    own transaction, so that hold is observable from the ``claim_rejected`` event
    it writes — which is also its durable record, so this only reports the hold
    for the tick and appends nothing of its own. Ordinary CAS losers stay
    ordinary losers.
    """
    if _last_claim_rejected_reason(conn, task_id) == "active_pr":
        result.respawn_guarded.append((task_id, "active_pr"))


def _launch_failure_phase(exc: BaseException) -> str:
    """Accurate ``spawn_refused`` phase for one launch exception.

    Reporting every refusal as ``runtime_identity`` blamed the sealed runtime
    fingerprint for ordinary scope/launch failures, which sent operators after
    identity corruption that was not there.
    """
    try:
        from hermes_cli.kanban_runtime import RuntimeIdentityError
    except Exception:
        return "launch"
    return "runtime_identity" if isinstance(exc, RuntimeIdentityError) else "launch"


def _note_launch_cancelled(task_id: str, error: str, result: "DispatchResult") -> None:
    """Record a launch the dispatcher cancelled on purpose.

    The card was never claimed, so there is nothing to undo in the DB and nothing
    to charge: it lands in ``cancelled``, not ``interrupted``, because it needs no
    operator unblock and no run was closed.
    """
    result.cancelled.append(task_id)
    _kb._log.info(
        "kanban dispatcher: launch of %s cancelled for shutdown/drain (%s)",
        task_id, error[:200],
    )


def _pause_claimed_run_for_shutdown(
    conn: sqlite3.Connection, claimed, result: "DispatchResult", exc: BaseException,
) -> bool:
    """Cancel a claimed-but-ungranted launch and pause that exact run.

    The worker never received the grant, so no Kanban tool call can be in flight
    against this card; the run is closed as ``interrupted`` and the card waits
    for the operator instead of spending a retry on a shutdown.
    """
    if pause_interrupted_run(
        conn, claimed.id, int(claimed.current_run_id or 0),
        reason=LAUNCH_STOPPED_BLOCK_REASON,
        error=str(exc)[:500],
    ):
        result.interrupted.append(claimed.id)
        return True
    # The worker is already cancelled, so a lost compare-and-swap leaves a card
    # whose run has no live owner. Say so loudly: the next tick reclaims it as a
    # crash, and an operator needs the reason.
    _kb._log.warning(
        "kanban dispatcher: cancelled launch of %s could not pause its run %s; "
        "it will be reclaimed as a crash",
        claimed.id, claimed.current_run_id,
    )
    return False


def _record_granted_spawn(
    conn: sqlite3.Connection, claimed, launch: "WorkerLaunch", workspace: str,
    result: "DispatchResult", board: Optional[str], count_spawn,
) -> bool:
    """Bookkeeping for a worker that has ALREADY received its grant.

    A failure here is an observability problem, not a launch failure: the run is
    owned by a live, verified worker, so cancelling it would abandon work that is
    already running and charging a retry would schedule a duplicate beside it.
    Each step is isolated so a broken hook cannot skip the durable PID write.
    """
    def _step(label: str, action) -> None:
        try:
            action()
        except Exception as exc:
            _kb._log.warning(
                "kanban dispatcher: post-grant %s failed for %s; "
                "worker %s keeps its run: %s",
                label, claimed.id, launch.pid, exc,
            )

    _step("claimed hook", lambda: _kb._fire_task_hook(
        "kanban_task_claimed", claimed, claimed.id, claimed.current_run_id,
    ))
    if launch.pid:
        _step("worker pid", lambda: _set_worker_pid(
            conn, claimed.id, int(launch.pid), runtime_identity=launch.runtime_identity,
            preparation_id=launch.preparation_id,
        ))
    _step("spawned hook", lambda: _kb._fire_worker_spawned_hook(
        conn, claimed, workspace, launch.pid, board=board,
    ))
    result.spawned.append((claimed.id, claimed.assignee or "", workspace))
    count_spawn(claimed.assignee)
    return True


def _grant_boundary(grant_guard) -> Any:
    """Context manager holding the caller's grant boundary; a no-op without one.

    ``grant_guard`` comes from the owning gateway, which holds the same lock while
    it sets its drain/stop flags, so the two cannot interleave: the stop transition
    takes effect either before the guarded decision or after the grant. Callers
    with no gateway (the standalone daemon, the CLI) pass nothing and get the
    plain unconditioned check.
    """
    if grant_guard is None:
        return contextlib.nullcontext()
    return grant_guard()


def _dispatch_lane_task(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    assignee: str,
    result: "DispatchResult",
    *,
    lane: str,
    dry_run: bool,
    ttl_seconds: Optional[int],
    board: Optional[str],
    failure_limit: int,
    spawn_fn,
    per_profile_cap: Optional[int],
    per_profile_running: dict[str, int],
    should_stop=None,
    grant_guard=None,
) -> bool:
    """Guard, verify, claim, resolve, and spawn one ready/review row.

    ``should_stop`` is polled before the workspace/preparation work, after it
    (before the claim), and immediately before the grant. A launch cancelled by
    a drain charges no failure; a cancellation landing after the claim also
    pauses that exact run for the operator. Nothing after a successful grant is
    ever cancelled — that worker owns the card and survives a gateway restart.
    """
    task_id = row["id"]
    profile_exists = _profile_exists_fn()
    if profile_exists is not None and not profile_exists(assignee):
        result.skipped_nonspawnable.append(task_id)
        # Per-task diagnostic so ``show``/``tail`` name the missing profile instead of leaving
        # the card in ``ready`` with zero board evidence (#122422). Unlike a respawn guard the
        # condition never expires on its own, so write it once: a repeat only when something
        # else happened on the card since (reassign, comment) — not one row per tick forever,
        # and not one row per foreign home per tick on a shared board (#101015).
        if not dry_run:
            with _kb.write_txn(conn):
                last = conn.execute(
                    "SELECT kind, payload FROM task_events WHERE task_id = ? "
                    "ORDER BY created_at DESC, id DESC LIMIT 1", (task_id,)).fetchone()
                if (last is None or last["kind"] != "skipped_nonspawnable"
                        or last["payload"] != _kb._json_or_null({"assignee": assignee})):
                    _kb._append_event(conn, task_id, "skipped_nonspawnable", {"assignee": assignee})
        return False
    if per_profile_cap is not None:
        current = per_profile_running.get(assignee, 0)
        if current >= per_profile_cap:
            result.skipped_per_profile_capped.append((task_id, assignee, current))
            return False
    guard_reason = check_respawn_guard(conn, task_id, lane=lane)
    if guard_reason is not None:
        result.respawn_guarded.append((task_id, guard_reason))
        if not dry_run:
            with _kb.write_txn(conn):
                _record_respawn_guard(conn, task_id, guard_reason)
        return False

    def _count_spawn(name: str) -> None:
        if per_profile_cap is not None and name:
            per_profile_running[name] = per_profile_running.get(name, 0) + 1

    if dry_run:
        result.spawned.append((task_id, assignee, ""))
        _count_spawn(assignee)
        return True

    use_default_spawn = spawn_fn is None
    launch: WorkerLaunch | None = None
    claimed = None
    workspace = None
    resolved_branch_name = None

    if use_default_spawn:
        preflight = _kb.get_task(conn, task_id)
        if preflight is None:
            return False
        spawn_task = preflight
        handoff_error = _kb._parent_handoff_start_error(
            conn, task_id, phase=lane,
        )
        if handoff_error is not None:
            _kb._record_parent_handoff_start_error(
                conn, task_id, handoff_error,
            )
            return False
        if lane == "review":
            spawn_task = replace(
                preflight,
                skills=list(dict.fromkeys([*(preflight.skills or []), "sdlc-review"])),
            )
        try:
            _check_not_stopping(should_stop, task_id)
            if spawn_task.workspace_kind == "worktree":
                workspace, resolved_branch_name = _kbw._resolve_worktree_workspace(spawn_task, board=board)
            else:
                workspace = _kbw.resolve_workspace(spawn_task, board=board)
            launch = _default_spawn(
                spawn_task, str(workspace), board=board, defer_grant=True,
                should_stop=should_stop,
            )
            if not isinstance(launch, WorkerLaunch):
                raise RuntimeError("default worker spawn did not return a fenced launch")
            # Everything expensive is done; a drain that arrived meanwhile must
            # not reach the claim.
            _check_not_stopping(should_stop, task_id)
            claim = _kb.claim_review_task if lane == "review" else _kb.claim_task
            claimed = claim(
                conn,
                task_id,
                ttl_seconds=ttl_seconds,
                runtime_identity=launch.runtime_identity,
                worker_pid=launch.pid,
                worker_start_time=launch.runtime_identity["start_time"],
                preparation_id=launch.preparation_id,
                expected_task=preflight,
                fire_hook=False,
            )
            if claimed is None:
                if launch.cancel:
                    launch.cancel()
                _note_claim_hold(conn, task_id, result)
                return False
        except WorkerLaunchInterrupted as exc:
            if launch is not None and launch.cancel:
                launch.cancel()
            _note_launch_cancelled(task_id, str(exc), result)
            return False
        except Exception as exc:
            if launch is not None and launch.cancel:
                launch.cancel()
            from tools.process_registry import RestartSafeScopeUnavailable

            infrastructure = isinstance(exc, RestartSafeScopeUnavailable)
            with _kb.write_txn(conn):
                _kb._append_event(
                    conn,
                    task_id,
                    "spawn_refused",
                    {
                        "phase": _launch_failure_phase(exc), "error": str(exc)[:1000],
                        **({"infrastructure": True} if infrastructure else {}),
                    },
                )
            if _record_task_failure(
                conn,
                task_id,
                str(exc),
                outcome="spawn_failed",
                failure_limit=failure_limit,
                infrastructure=infrastructure,
            ):
                result.auto_blocked.append(task_id)
            return False
    else:
        claim = _kb.claim_review_task if lane == "review" else _kb.claim_task
        claimed = claim(conn, task_id, ttl_seconds=ttl_seconds)
        if claimed is None:
            _note_claim_hold(conn, task_id, result)
            return False
        try:
            if claimed.workspace_kind == "worktree":
                workspace, resolved_branch_name = _kbw._resolve_worktree_workspace(claimed, board=board)
            else:
                workspace = _kbw.resolve_workspace(claimed, board=board)
        except Exception as exc:
            if _record_task_failure(
                conn, claimed.id, f"workspace: {exc}",
                outcome="spawn_failed", failure_limit=failure_limit, release_claim=True, end_run=True,
            ):
                result.auto_blocked.append(claimed.id)
            return False

    try:
        assert claimed is not None
        assert workspace is not None
        _kbw.set_workspace_path(conn, claimed.id, str(workspace))
        if claimed.workspace_kind == "worktree":
            _kbw.set_branch_name(
                conn, claimed.id,
                resolved_branch_name or (claimed.branch_name or "").strip() or f"wt/{claimed.id}",
            )
        _kbw._maybe_emit_scratch_tip(conn, claimed.id, claimed.workspace_kind)
        if lane == "review":
            claimed.skills = list(dict.fromkeys([*(claimed.skills or []), "sdlc-review"]))
        if launch is None:
            launch = _call_spawn_fn(spawn_fn, claimed, str(workspace), board)
        if isinstance(launch, WorkerLaunch):
            pid = launch.pid
            launch_identity = launch.runtime_identity
            preparation_id = launch.preparation_id
            if launch.grant:
                # Final barrier, held under the caller's grant boundary so the stop
                # transition cannot land between the decision and the grant: the
                # drain either takes effect before the decision (launch cancelled)
                # or after the grant (that worker is already final and survives).
                # It is a fence, not a revoker: revoking would mean cancelling a
                # worker that may already be past its bootstrap, which the contract
                # forbids.
                with _grant_boundary(grant_guard):
                    _check_not_stopping(should_stop, claimed.id)
                    launch.grant(int(claimed.current_run_id), claimed.claim_lock)
                return _record_granted_spawn(
                    conn, claimed, launch, str(workspace), result, board, _count_spawn,
                )
        else:
            pid = launch
            launch_identity = None
            preparation_id = None
        if pid:
            _set_worker_pid(
                conn, claimed.id, int(pid), runtime_identity=launch_identity,
                preparation_id=preparation_id,
            )
        _kb._fire_worker_spawned_hook(conn, claimed, str(workspace), pid, board=board)
        result.spawned.append((claimed.id, claimed.assignee or "", str(workspace)))
        _count_spawn(claimed.assignee)
        return True
    except WorkerLaunchInterrupted as exc:
        if launch is not None and launch.cancel:
            launch.cancel()
        _pause_claimed_run_for_shutdown(conn, claimed, result, exc)
        return False
    except Exception as exc:
        if launch is not None and launch.cancel:
            launch.cancel()
        from tools.process_registry import RestartSafeScopeUnavailable

        # The host refused the spawn (no restart-safe scope): nothing about the
        # card ran, so it must not spend the card's retry budget (#114720).
        infrastructure = isinstance(exc, RestartSafeScopeUnavailable)
        if infrastructure:
            _kb._log.warning("kanban dispatcher: spawn of %s deferred, host cannot place the worker: %s", claimed.id, exc)
        if _record_task_failure(
            conn, claimed.id, str(exc),
            outcome="spawn_failed", failure_limit=failure_limit, release_claim=True, end_run=True,
            infrastructure=infrastructure,
        ):
            result.auto_blocked.append(claimed.id)
        return False


def _apply_default_assignee(
    conn: sqlite3.Connection, task_id: str, assignee: str, *, dry_run: bool,
) -> bool:
    """Persist ``kanban.default_assignee`` on an unassigned ready row.

    Mutating the row keeps board state honest: the task is legitimately owned
    by the default, not "unassigned but secretly routed". ``dry_run`` reports
    without writing. Returns False when the write failed.
    """
    if dry_run:
        return True
    try:
        assignee = _kb._canonical_assignee(assignee)
        if not assignee:
            return False
        with _kb.write_txn(conn):
            row = conn.execute(
                "SELECT lifecycle_contract FROM tasks WHERE id = ? "
                "AND (assignee IS NULL OR assignee = '')",
                (task_id,),
            ).fetchone()
            _kb._validate_lifecycle_role_identity(
                conn,
                _kb.safe_decode_contract(_kb._row_get(row, "lifecycle_contract")),
                assignee,
                task_id=task_id,
            )
            conn.execute(
                "UPDATE tasks SET assignee = ? WHERE id = ? "
                "AND (assignee IS NULL OR assignee = '')",
                (assignee, task_id),
            )
            _kb._append_event(
                conn, task_id, "assigned",
                {"assignee": assignee, "source": "kanban.default_assignee"},
            )
    except Exception:
        _kb._log.debug(
            "kanban dispatch: failed to apply default_assignee=%r to task %s",
            assignee,
            task_id,
            exc_info=True,
        )
        return False
    return True


def _run_reclaim_phase(
    conn: sqlite3.Connection,
    result: DispatchResult,
    *,
    stale_timeout_seconds: int,
    failure_limit: int,
    reconcile_orphans: bool,
    board: Optional[str] = None,
) -> None:
    """Reclaim stale/orphaned/crashed/timed-out running tasks, then promote."""
    reap_worker_zombies()
    # Host re-instantiation outranks every path below: an old PID may already be
    # reused, so nothing after this may probe, defer on, or signal it.
    result.interrupted = pause_host_interrupted_runs(conn)
    result.reaped_terminal_workers = reap_terminal_workers(conn)
    result.reclaimed = _kb.release_stale_claims(conn, failure_limit=failure_limit)
    if reconcile_orphans:
        result.reconciled_orphans = reconcile_orphaned_running(conn)
    result.stale = detect_stale_running(conn, stale_timeout_seconds=stale_timeout_seconds)
    result.crashed = detect_crashed_workers(conn, board=board, failure_limit=failure_limit)
    # Side-channel attributes (see detect_crashed_workers); rate-limited tasks
    # went back to ``ready`` and the respawn guard defers them until quota clears.
    result.auto_blocked.extend(getattr(detect_crashed_workers, "_last_auto_blocked", []))
    result.rate_limited.extend(getattr(detect_crashed_workers, "_last_rate_limited", []))
    result.timed_out = enforce_max_runtime(conn)
    result.promoted = _kb.recompute_ready(conn, failure_limit=failure_limit)


def _tick_spawn_budget(
    conn: sqlite3.Connection,
    result: DispatchResult,
    *,
    max_spawn: Optional[int],
    max_in_progress: Optional[int],
    board: Optional[str],
) -> tuple[bool, Optional[int]]:
    """``(may_spawn, spawn_budget)`` for this tick; ``budget None`` = uncapped.

    ``max_spawn`` is a live per-board concurrency cap (running + this tick's
    spawns), not a per-tick budget — a per-tick reading would grow concurrency
    by N every tick. ``max_in_progress`` is a HOST-level cap: running workers on
    every other board count against the same budget, else N boards multiply the
    cap by N — exactly the fan-out the memory-derived default exists to prevent.
    """
    # Count already-running tasks so max_spawn enforces concurrency, not a
    # per-tick budget: "running" tasks stay running until the worker makes a terminal
    # board call (kanban_complete/kanban_block/kanban_request_review) or the TTL reclaims them.
    running_count = 0
    spawn_budget: Optional[int] = None
    if max_spawn is not None or max_in_progress is not None:
        running_count = count_running_tasks(conn)

    # Both ready and review loops consume from the same budget.
    if max_spawn is not None:
        if running_count >= max_spawn:
            return False, None
        spawn_budget = max_spawn - running_count

    if max_in_progress is not None:
        total_running = running_count + count_running_tasks_other_boards(board)
        if total_running >= max_in_progress:
            return False, None
        remaining = max_in_progress - total_running
        if spawn_budget is None or spawn_budget > remaining:
            spawn_budget = remaining

    # Memory-pressure guard: a static cap can't see the host's actual state.
    # critical -> spawn nothing this tick; elevated -> at most one new worker.
    # Reclaim/promotion already ran, so bookkeeping stays live; deferred tasks
    # wait for a later tick. "unknown" imposes no restriction.
    pressure = _memory_pressure_level()
    if pressure == "critical":
        result.memory_pressure = pressure
        _kb._log.warning(
            "kanban dispatch: system memory pressure is critical; "
            "spawning no new workers this tick (deferred, not dropped)"
        )
        return False, None
    if pressure == "elevated":
        result.memory_pressure = pressure
        if spawn_budget is None or spawn_budget > 1:
            _kb._log.warning(
                "kanban dispatch: system memory pressure is elevated; "
                "limiting to at most 1 new worker this tick"
            )
            spawn_budget = 1
    return True, spawn_budget


def _lane_rows(conn: sqlite3.Connection, status: str) -> list[sqlite3.Row]:
    """Unclaimed rows of one lane in dispatch order."""
    return conn.execute(
        "SELECT id, assignee FROM tasks "
        f"WHERE status = '{status}' AND claim_lock IS NULL "
        "ORDER BY priority DESC, created_at ASC"
    ).fetchall()


def _any_spawnable_review(
    conn: sqlite3.Connection,
    review_rows: list[sqlite3.Row],
    *,
    per_profile_cap: Optional[int] = None,
    per_profile_running: Optional[dict[str, int]] = None,
) -> bool:
    """Mirror review dispatch gates before reserving ready-lane capacity.

    Unavailable profile metadata retains the historic fail-open behavior. A
    review row that :func:`_dispatch_lane_task` would refuse this tick — its
    assignee already at the per-profile cap, or respawn-guarded — cannot
    consume the reservation, so it must not withhold capacity from an
    otherwise ready task (one such row would pin ``ready_budget`` to 0).
    """
    if not review_rows:
        return False
    profile_exists = _profile_exists_fn()
    running = per_profile_running or {}
    for row in review_rows:
        assignee = row["assignee"]
        if not assignee:
            continue
        if profile_exists is not None and not profile_exists(assignee):
            continue
        if per_profile_cap is not None and running.get(assignee, 0) >= per_profile_cap:
            continue
        if check_respawn_guard(conn, row["id"], lane="review") is None:
            return True
    return False


def _resolve_default_assignee(default_assignee: Optional[str]) -> Optional[str]:
    """``kanban.default_assignee`` when it names a real profile this home may
    claim (``kanban.dispatch_profiles`` gated, same predicate as the spawn
    gate). Otherwise ``None`` so an unassigned shared-board card is never
    written to. When the profiles module isn't importable trust the
    operator's config: the downstream check still buckets a missing profile
    as nonspawnable."""
    name = (default_assignee or "").strip() or None
    if name:
        profile_exists = _profile_exists_fn()
        if profile_exists is not None and not profile_exists(name):
            return None
    return name


# The dispatch lock has been released here. Fire the tick observer strictly OUTSIDE the single-writer
# critical section (#56066 sweeper finding / #64231 disposition): a slow subscriber must never extend the
# lock hold and stall a sibling dispatcher's tick.
def _dispatch_once_locked(
    conn: sqlite3.Connection,
    *,
    spawn_fn=None,
    ttl_seconds: Optional[int] = None,
    dry_run: bool = False,
    max_spawn: Optional[int] = None,
    max_in_progress: Optional[int] = None,
    failure_limit: int = DEFAULT_FAILURE_LIMIT,
    stale_timeout_seconds: int = 0,
    board: Optional[str] = None,
    default_assignee: Optional[str] = None,
    max_in_progress_per_profile: Optional[int] = None,
    reconcile_orphans: bool = True,
    should_stop: Optional[Callable[[], bool]] = None,
    grant_guard: Optional[Callable[[], Any]] = None,
) -> DispatchResult:
    """One dispatcher tick: reclaim stale/crashed running tasks, promote
    todo -> ready, then atomically claim each spawnable ready/review row and
    call ``spawn_fn(task, workspace_path, board) -> Optional[int]``, recording
    the PID so later ticks catch crashes before the TTL. Cap semantics:
    :func:`_tick_spawn_budget`."""
    result = DispatchResult()
    _run_reclaim_phase(
        conn, result, stale_timeout_seconds=stale_timeout_seconds,
        failure_limit=failure_limit, reconcile_orphans=reconcile_orphans, board=board,
    )
    may_spawn, spawn_budget = _tick_spawn_budget(
        conn, result, max_spawn=max_spawn, max_in_progress=max_in_progress, board=board,
    )
    if not may_spawn:
        return result

    ready_rows = _lane_rows(conn, "ready")
    # Review rows are enumerated up front so the budget split can see whether
    # review work exists at all.
    review_rows = _lane_rows(conn, "review") if review_dispatch_enabled() else []
    # Per-profile cap. Deferred tasks go to skipped_per_profile_capped, not
    # skipped_unassigned — "busy, retry later" differs from "needs routing".
    # Resolved BEFORE the review reservation so the reservation can see which
    # review rows the lane loop would refuse this tick.
    per_profile_cap = max_in_progress_per_profile if (
        # Per-profile concurrency cap (#21582): when set, track how many workers each assignee already has
        # in flight, and refuse to spawn when this would push that assignee past the cap. Prevents fan-out
        # workloads from melting a single profile's local model / API quota / browser pool while leaving
        # other profiles idle.
        isinstance(max_in_progress_per_profile, int)
        and max_in_progress_per_profile > 0
    ) else None
    per_profile_running: dict[str, int] = {}
    if per_profile_cap is not None:
        for prow in conn.execute(
            "SELECT assignee, COUNT(*) AS n FROM tasks "
            "WHERE status = 'running' AND assignee IS NOT NULL "
            "GROUP BY assignee"
        ):
            per_profile_running[prow["assignee"]] = int(prow["n"])
    # Review-lane reservation: the ready loop runs first and would otherwise
    # consume the ENTIRE shared budget, starving reviews under a sustained ready
    # backlog. When spawnable review work exists and there is any budget, hold
    # one slot back.
    ready_budget = spawn_budget
    if spawn_budget is not None and spawn_budget > 0 and _any_spawnable_review(
        conn, review_rows,
        per_profile_cap=per_profile_cap, per_profile_running=per_profile_running,
    ):
        ready_budget = max(spawn_budget - 1, 0)
    lane_kwargs: dict[str, Any] = dict(
        dry_run=dry_run, ttl_seconds=ttl_seconds, board=board,
        failure_limit=failure_limit, spawn_fn=spawn_fn,
        per_profile_cap=per_profile_cap, per_profile_running=per_profile_running,
        should_stop=should_stop, grant_guard=grant_guard,
    )
    default_assignee = _resolve_default_assignee(default_assignee)
    spawned = 0
    for row in ready_rows:
        if _stop_requested(should_stop):
            break
        if ready_budget is not None and spawned >= ready_budget:
            break
        row_assignee = row["assignee"]
        if not row_assignee:
            # Honour kanban.default_assignee so an unassigned task doesn't
            # park in 'ready' forever.
            if not default_assignee or not _apply_default_assignee(
                conn, row["id"], default_assignee, dry_run=dry_run,
            ):
                result.skipped_unassigned.append(row["id"])
                continue
            row_assignee = default_assignee
            result.auto_assigned_default.append(row["id"])
        if _dispatch_lane_task(conn, row, row_assignee, result, lane="ready", **lane_kwargs):
            spawned += 1

    # A review agent (sdlc-review) approves (→ done) or requests changes
    # (→ ready/todo). Review spawns share max_spawn with ready tasks. The loop
    # checks the FULL shared ``spawn_budget`` — the reservation above caps the
    # ready lane, it grants no extra capacity here.
    for row in review_rows:
        if _stop_requested(should_stop):
            break
        if spawn_budget is not None and spawned >= spawn_budget:
            break
        if not row["assignee"]:
            result.skipped_unassigned.append(row["id"])
            continue
        if _dispatch_lane_task(conn, row, row["assignee"], result, lane="review", **lane_kwargs):
            spawned += 1
    return result





def _pid_alive(pid: Optional[int]) -> bool:
    from hermes_cli.kanban_worker_runtime import _pid_alive as worker_pid_alive

    return worker_pid_alive(pid)


def _worker_terminal_timeout_env(
    max_runtime_seconds: Optional[int],
    current_timeout: Optional[str],
) -> Optional[str]:
    from hermes_cli.kanban_worker_runtime import _worker_terminal_timeout_env

    return _worker_terminal_timeout_env(max_runtime_seconds, current_timeout)

# ---------------------------------------------------------------------------
# Long-lived dispatcher daemon
# ---------------------------------------------------------------------------

def run_daemon(
    *,
    interval: float = 60.0,
    max_spawn: Optional[int] = None,
    failure_limit: int = DEFAULT_FAILURE_LIMIT,
    stop_event=None,
    on_tick=None,
) -> None:
    """Run the dispatcher in a loop until interrupted.

    Calls :func:`dispatch_once` every ``interval`` seconds; exits cleanly on
    SIGINT / SIGTERM so it is systemd-friendly. ``stop_event`` and ``on_tick``
    are test hooks. Each tick resolves ``kanban.max_in_progress`` exactly like
    the gateway dispatcher and ``hermes kanban dispatch`` — the standalone
    daemon must not be the one uncapped entry point.
    """
    _freeze_runtime_identity()
    import threading

    if stop_event is None:
        stop_event = threading.Event()

    def _handle(_signum, _frame):
        stop_event.set()

    # Install handlers only on the main thread — tests call this inline from
    # worker threads and signal() would raise there.
    if threading.current_thread() is threading.main_thread():
        for sig_name in ("SIGINT", "SIGTERM"):
            sig = getattr(signal, sig_name, None)
            if sig is not None:
                with contextlib.suppress(ValueError, OSError):
                    signal.signal(sig, _handle)

    while not stop_event.is_set():
        try:
            # Re-resolved every tick (config load is mtime-cached) so operator
            # edits apply without a restart.
            max_in_progress = resolve_max_in_progress(configured_max_in_progress())
            with contextlib.closing(_kbc.connect()) as conn:
                res = dispatch_once(
                    conn,
                    max_spawn=max_spawn,
                    max_in_progress=max_in_progress,
                    failure_limit=failure_limit,
                    should_stop=stop_event.is_set,
                )
            if on_tick is not None:
                with contextlib.suppress(Exception):
                    on_tick(res)
        except Exception:
            # Don't let any single tick kill the daemon.
            import traceback
            traceback.print_exc()
        stop_event.wait(timeout=interval)


# Late-bound origin namespace (see module docstring); imported LAST so this
# module is fully populated before ``kanban_db`` imports from it.
from hermes_cli import kanban_db as _kb  # noqa: E402
from hermes_cli import kanban_db_connect as _kbc  # noqa: E402
from hermes_cli import kanban_db_workspace as _kbw  # noqa: E402

from hermes_cli import kanban_worker_runtime as _kwr  # noqa: E402
from hermes_cli import kanban_runtime as _kr  # noqa: E402

WorkerLaunch = _kwr.WorkerLaunch
WorkerLaunchInterrupted = _kwr.WorkerLaunchInterrupted
_stop_requested = _kwr._stop_requested
_check_not_stopping = _kwr._check_not_stopping
_recent_worker_exits = _kwr._recent_worker_exits
_worker_pid_aliases = _kwr._worker_pid_aliases
_worker_processes = _kwr._worker_processes
_worker_runtime_snapshots = _kwr._worker_runtime_snapshots
_record_worker_exit = _kwr._record_worker_exit
_classify_worker_exit = _kwr._classify_worker_exit
_pid_alive = _kwr._pid_alive
_positive_int = _kwr._positive_int
worker_log_rotation_config = _kwr.worker_log_rotation_config
_rotated_log_path = _kwr._rotated_log_path
_rotate_worker_log = _kwr._rotate_worker_log
reap_worker_zombies = _kwr.reap_worker_zombies
_module_hermes_argv = _kwr._module_hermes_argv
_resolve_hermes_argv = _kwr._resolve_hermes_argv
_worker_terminal_timeout_env = _kwr._worker_terminal_timeout_env
_resolve_worker_cli_toolsets = _kwr._resolve_worker_cli_toolsets
_retagged_workspace_roots = _kwr._retagged_workspace_roots
_retag_legacy_worker_sessions = _kwr._retag_legacy_worker_sessions
_worker_argv = _kwr._worker_argv
_open_worker_log = _kwr._open_worker_log
_restart_safe_worker_argv = _kwr._restart_safe_worker_argv


def _default_spawn(
    task: "Task",
    workspace: str,
    *,
    board: Optional[str] = None,
    defer_grant: bool = False,
    should_stop: Optional[Callable[[], bool]] = None,
) -> WorkerLaunch | int:
    return _kwr._default_spawn(
        task, workspace, board=board, defer_grant=defer_grant, should_stop=should_stop,
    )

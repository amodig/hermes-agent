"""Embedded kanban dispatcher: settings resolution and per-tick board work.

``GatewayKanbanWatchersMixin._kanban_dispatcher_watcher`` owns the loop,
the singleton lock and the health telemetry; everything that only needs the
``kanban_db`` module and the resolved settings lives here.
"""

from __future__ import annotations

import contextlib
import os
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from gateway.kanban_watchers_common import _board_slugs, _positive_int_setting, logger


def _kbc():
    from hermes_cli import kanban_db_connect
    return kanban_db_connect


def _kbd():
    from hermes_cli import kanban_db_dispatch
    return kanban_db_dispatch

_CORRUPT_DB_MARKERS = ("file is not a database", "database disk image is malformed")


@dataclass
class _DispatcherSettings:
    """``kanban.*`` dispatch settings, read once at boot (restart to apply)."""

    interval: float
    max_spawn: Any
    max_in_progress: Optional[int]
    failure_limit: int
    stale_timeout_seconds: int
    reconcile_orphans: bool
    default_assignee: Optional[str]
    max_in_progress_per_profile: Optional[int]


def _resolve_dispatcher_settings(kanban_cfg: dict, kb: Any) -> _DispatcherSettings:
    """Parse and log the dispatcher settings in their established order."""
    try:
        interval = float(kanban_cfg.get("dispatch_interval_seconds", 60) or 60)
    except (ValueError, TypeError):
        logger.warning("kanban dispatcher: invalid dispatch_interval_seconds=%r, using default 60",
                       kanban_cfg.get("dispatch_interval_seconds"))
        interval = 60.0
    interval = max(interval, 1.0)  # sanity floor — tighter than this is a footgun

    max_spawn = kanban_cfg.get("max_spawn")
    if max_spawn is not None:
        logger.info("kanban dispatcher: max_spawn=%s", max_spawn)

    # Cap simultaneously running tasks so slow workers don't pile up and time
    # out. Explicit config wins; otherwise a memory-derived default (unbounded
    # fan-out swap-thrashes small hosts), or None where total memory can't be read.
    max_in_progress = _positive_int_setting(kanban_cfg, "max_in_progress")
    effective_max_in_progress = _kbd().resolve_max_in_progress(max_in_progress)
    if max_in_progress is None and effective_max_in_progress is not None:
        logger.info(
            "kanban dispatcher: kanban.max_in_progress unset; using "
            "memory-derived default max_in_progress=%d "
            "(set kanban.max_in_progress in config.yaml to override)",
            effective_max_in_progress,
        )

    raw_failure_limit = kanban_cfg.get("failure_limit", kb.DEFAULT_FAILURE_LIMIT)
    try:
        failure_limit = int(raw_failure_limit)
    except (TypeError, ValueError):
        logger.warning("kanban dispatcher: invalid kanban.failure_limit=%r; using default %d",
                       raw_failure_limit, kb.DEFAULT_FAILURE_LIMIT)
        failure_limit = kb.DEFAULT_FAILURE_LIMIT
    if failure_limit < 1:
        logger.warning("kanban dispatcher: kanban.failure_limit=%r is below 1; using default %d",
                       raw_failure_limit, kb.DEFAULT_FAILURE_LIMIT)
        failure_limit = kb.DEFAULT_FAILURE_LIMIT

    # 0 disables stale detection.
    raw_stale = kanban_cfg.get("dispatch_stale_timeout_seconds", 0)
    try:
        stale_timeout_seconds = int(raw_stale or 0)
    except (TypeError, ValueError):
        logger.warning("kanban dispatcher: invalid kanban.dispatch_stale_timeout_seconds=%r; "
                       "disabling stale detection", raw_stale)
        stale_timeout_seconds = 0

    # Fallback profile for tasks created without an assignee (e.g. via the
    # dashboard). Empty (the schema default) keeps skipping them.
    # When set, the dispatcher applies it to unassigned ready tasks instead of skipping them indefinitely
    # (#27145). Empty string (the schema default) means "no fallback, keep skipping" — backward-compatible
    # with existing installs.
    default_assignee = (kanban_cfg.get("default_assignee") or "").strip() or None
    if default_assignee:
        logger.info("kanban dispatcher: default_assignee=%r (unassigned ready tasks "
                    "will route to this profile)", default_assignee)

    return _DispatcherSettings(
        interval=interval,
        max_spawn=max_spawn,
        max_in_progress=effective_max_in_progress,
        failure_limit=failure_limit,
        stale_timeout_seconds=stale_timeout_seconds,
        # Requeue 'running' cards with broken claim bookkeeping (zombie-card
        # reconciliation); false keeps orphans frozen for manual forensics.
        reconcile_orphans=bool(kanban_cfg.get("reconcile_orphans", True)),
        default_assignee=default_assignee,
        # Per-profile concurrency cap: no single profile's local model / API
        # quota / browser pool gets overwhelmed by a fan-out.
        max_in_progress_per_profile=_positive_int_setting(kanban_cfg, "max_in_progress_per_profile"),
    )


class _KanbanDispatcher:
    """Per-tick board work for the embedded dispatcher (runs in worker threads).

    Boards are enumerated every tick so a board created mid-run is picked up
    without a restart. Corrupt-looking board DBs are quarantined per
    fingerprint and retried after ``CORRUPT_BOARD_RETRY_AFTER_SECONDS``:
    transient WAL/open races can look like "malformed" for one tick.
    """

    CORRUPT_BOARD_RETRY_AFTER_SECONDS = 300

    def __init__(self, kb: Any, settings: _DispatcherSettings, should_stop=None,
                 grant_guard=None) -> None:
        self.kb = kb
        self.settings = settings
        # Called on every tick so a drain/stop that starts mid-tick cancels the
        # launches still in flight instead of granting them to a dying gateway.
        self.should_stop = should_stop
        # Held across the final stop re-check and the grant, so a drain cannot
        # land between the decision and the grant (see the gateway mixin).
        self.grant_guard = grant_guard
        self.disabled_corrupt_boards: dict[str, tuple[tuple[str, int | None, int | None], float]] = {}
        _kbd()._freeze_runtime_identity()

    def _board_slugs(self) -> list[str]:
        """Visit each resolved DB once, keeping the pinned board's identity."""
        from hermes_cli import kanban_db as _kb

        with _kb.pin_first_board_resolution():
            slugs = _board_slugs(self.kb)
            if os.environ.get("HERMES_KANBAN_DB", "").strip():
                # Enumeration starts with default, not necessarily the board
                # whose DB the gateway pins. Use the canonical active slug.
                slugs.insert(0, self.kb.get_current_board())
            seen_db_paths: set[str] = set()
            unique: list[str] = []
            for slug in slugs:
                try:
                    resolved = str(self.kb.kanban_db_path(slug).expanduser().resolve())
                except Exception:
                    resolved = f"slug:{slug}"
                if resolved not in seen_db_paths:
                    seen_db_paths.add(resolved)
                    unique.append(slug)
            return unique

    def board_db_fingerprint(self, slug: str) -> tuple[str, int | None, int | None]:
        from hermes_cli import kanban_db as _kb
        with _kb.pin_first_board_resolution():
            path = self.kb.kanban_db_path(slug)
        try:
            resolved = str(path.expanduser().resolve())
        except Exception:
            resolved = str(path)
        try:
            stat = path.stat()
        except OSError:
            return (resolved, None, None)
        return (resolved, stat.st_mtime_ns, stat.st_size)

    def is_corrupt_board_db_error(self, exc: Exception) -> bool:
        if isinstance(exc, _kbc().KanbanDbCorruptError):
            return True
        return isinstance(exc, sqlite3.DatabaseError) and any(m in str(exc).lower() for m in _CORRUPT_DB_MARKERS)

    def _quarantine_lifted(self, slug: str, fingerprint: tuple) -> bool:
        """Return False while *slug* stays quarantined; lift (and log) otherwise."""
        disabled_entry = self.disabled_corrupt_boards.get(slug)
        if disabled_entry is None:
            return True
        disabled_fingerprint, disabled_at = disabled_entry
        age = time.monotonic() - disabled_at
        if disabled_fingerprint == fingerprint and age < self.CORRUPT_BOARD_RETRY_AFTER_SECONDS:
            return False
        if disabled_fingerprint == fingerprint:
            logger.info("kanban dispatcher: board %s database fingerprint unchanged "
                        "after %.0fs quarantine; retrying dispatch", slug, age)
        else:
            logger.info("kanban dispatcher: board %s database changed; retrying dispatch", slug)
        self.disabled_corrupt_boards.pop(slug, None)
        return True

    def tick_once_for_board(self, slug: str) -> Optional[object]:
        """Run one dispatch_once for a specific board.

        The per-board DB is opened explicitly so boards never share a
        connection or claim across each other.
        """
        conn = None
        fingerprint = self.board_db_fingerprint(slug)
        if not self._quarantine_lifted(slug, fingerprint):
            return None
        kwargs = {k: v for k, v in asdict(self.settings).items() if k != "interval"}
        try:
            # No explicit init_db(): connect() runs the migration once per
            # process (see the matching note in the notifier collector).
            # Pin-first: the tick is machine flow — on a box whose env pins
            # HERMES_KANBAN_DB every enumerated slug must resolve to the pinned
            # file, or the dispatcher reads per-slug DBs nobody writes.
            from hermes_cli import kanban_db as _kb
            with _kb.pin_first_board_resolution():
                conn = _kbc().connect(board=slug)
                return _kbd().dispatch_once(
                    conn, board=slug, should_stop=self.should_stop,
                    grant_guard=self.grant_guard, **kwargs,
                )
        except Exception as exc:
            if self.is_corrupt_board_db_error(exc):
                self.disabled_corrupt_boards[slug] = (fingerprint, time.monotonic())
                logger.error(
                    "kanban dispatcher: board %s database %s is not a valid "
                    "SQLite database; pausing dispatch for this board until "
                    "the file changes, the gateway restarts, or the "
                    "quarantine timer expires. Move or restore the file, "
                    "then run `hermes kanban init` if you need a fresh board.",
                    slug, fingerprint[0],
                )
                return None
            logger.exception("kanban dispatcher: tick failed on board %s", slug)
            return None
        finally:
            if conn is not None:
                with contextlib.suppress(Exception):
                    conn.close()

    def tick_once(self) -> list[tuple[str, Optional[object]]]:
        """Run one dispatch_once per board. Returns (slug, result) pairs.

        Stops enumerating boards the moment the caller asks us to stop: every
        later board's tick would only prepare workers for a gateway that is
        going away.
        """
        results: list[tuple[str, Optional[object]]] = []
        for slug in self._board_slugs():
            # ``_stop_requested`` reads a broken predicate as "keep going", so a
            # faulty shutdown check can never wedge the dispatcher loop.
            if _kbd()._stop_requested(self.should_stop):
                break
            results.append((slug, self.tick_once_for_board(slug)))
        return results

    def spawnable_ids(self) -> list[str]:
        """Board-qualified ids of tasks on ANY board the dispatcher would spawn for.

        Control-plane lanes (e.g. ``orion-cc``) are pulled by terminals via
        ``claim_task`` and never spawnable — a queue full of those is
        "correctly idle", not "stuck". The review column is probed only when
        review dispatch is on (same gate as the dispatcher): a task waiting
        for a human reviewer is idle, not stuck.
        """
        kbd = _kbd()
        found: list[str] = []
        from hermes_cli import kanban_db as _kb
        with _kb.pin_first_board_resolution():
            for slug in self._board_slugs():
                conn = None
                try:
                    conn = _kbc().connect(board=slug)
                    found += [f"{slug}/{task_id}" for task_id in kbd.spawnable_lane_ids(conn)]
                except Exception:
                    continue
                finally:
                    if conn is not None:
                        with contextlib.suppress(Exception):
                            conn.close()
        return found

    def ready_nonempty(self) -> bool:
        """:meth:`spawnable_ids` as a bool."""
        return bool(self.spawnable_ids())

    def auto_decompose_tick(self, auto_decompose_per_tick: int) -> int:
        """Auto-decompose up to N triage tasks across all boards into ready workgraphs.

        Runs before dispatch fans out; the per-tick cap keeps a bulk triage
        load from burst-spending the aux LLM. Returns the number decomposed.
        """
        try:
            from hermes_cli import kanban_decompose as _decomp
        except Exception as exc:  # pragma: no cover
            logger.warning("kanban auto-decompose: import failed (%s); skipping", exc)
            return 0
        attempted = 0
        successes = 0
        from hermes_cli import kanban_db as _kb
        with _default_profile_secret_scope(), _kb.pin_first_board_resolution():
            for slug in self._board_slugs():
                if attempted >= auto_decompose_per_tick:
                    break
                # Pin the board via env for the call: the decomposer connects
                # with no board kwarg (same pattern as the dashboard specify endpoint).
                prev_env = os.environ.get("HERMES_KANBAN_BOARD")
                try:
                    os.environ["HERMES_KANBAN_BOARD"] = slug
                    try:
                        triage_ids = _decomp.list_triage_ids()
                    except Exception as exc:
                        logger.debug("kanban auto-decompose: list_triage_ids failed on board %s (%s)", slug, exc)
                        triage_ids = []
                    for tid in triage_ids:
                        if attempted >= auto_decompose_per_tick:
                            break
                        attempted += 1
                        successes += self._decompose_one(_decomp, slug, tid)
                finally:
                    if prev_env is None:
                        os.environ.pop("HERMES_KANBAN_BOARD", None)
                    else:
                        os.environ["HERMES_KANBAN_BOARD"] = prev_env
        return successes

    @staticmethod
    def _decompose_one(_decomp: Any, slug: str, tid: str) -> int:
        """Decompose one triage task; returns 1 on success, 0 otherwise."""
        try:
            outcome = _decomp.decompose_task(tid, author="auto-decomposer")
        except Exception:
            logger.exception("kanban auto-decompose: decompose_task crashed on %s", tid)
            return 0
        if not outcome.ok:
            # Common no-op reasons (no aux client) must not spam logs every tick.
            logger.debug("kanban auto-decompose [%s]: %s skipped: %s", slug, tid, outcome.reason)
            return 0
        if outcome.fanout and outcome.child_ids:
            logger.info("kanban auto-decompose [%s]: %s → %d children", slug, tid, len(outcome.child_ids))
        else:
            logger.info("kanban auto-decompose [%s]: %s → single task (no fanout)", slug, tid)
        return 1


def guarded_holds(results: Optional[list]) -> list[tuple[str, str]]:
    """``(board/task, reason)`` for every guard hold in one gateway tick.

    Board-qualified so an operator can act on the reported task without
    guessing which board's dispatcher produced it.
    """
    holds: list[tuple[str, str]] = []
    for slug, res in (results or []):
        for task_id, reason in (getattr(res, "respawn_guarded", None) or []):
            holds.append((f"{slug}/{task_id}", reason))
    return holds


def guarded_alert(holds: list[tuple[str, str]], ticks: int) -> str:
    """Bounded alert text for a run of guard-held ticks (dispatcher-owned wording)."""
    return _kbd().respawn_guard_alert(holds, ticks)


@contextlib.contextmanager
def _default_profile_secret_scope():
    """Install the gateway launch profile's secret scope while multiplexing is on.

    The tick runs via ``_to_thread_process_service`` in a fresh context, so no
    per-turn scope exists and ``get_secret`` fails closed. The decomposer's aux
    LLM reads ``auxiliary.*`` from ``get_hermes_home()``, so its credentials come
    from that same home. No-op for single-profile gateways.
    """
    from agent.secret_scope import (
        build_profile_secret_scope, is_multiplex_active, reset_secret_scope, set_secret_scope)
    from hermes_constants import get_hermes_home

    if not is_multiplex_active():
        yield
        return
    token = set_secret_scope(
        build_profile_secret_scope(Path(get_hermes_home())), profile_home=str(get_hermes_home()))
    try:
        yield
    finally:
        reset_secret_scope(token)


def _log_spawn_results(results: Optional[list]) -> bool:
    """Log per-board spawn summaries; returns whether any board spawned.

    An interruption-only tick (a paused card, no spawn) still logs: that line is
    the operator's only in-gateway signal that the dispatcher acted.
    """
    any_spawned = False
    for slug, res in (results or []):
        if res is None:
            continue
        spawned = getattr(res, "spawned", None)
        interrupted = len(getattr(res, "interrupted", ()) or ())
        cancelled = len(getattr(res, "cancelled", ()) or ())
        if spawned:
            any_spawned = True
        if not spawned and not interrupted and not cancelled:
            # Quiet by default: an idle gateway stays silent.
            continue
        logger.info(
            "kanban dispatcher [%s]: spawned=%d reclaimed=%d "
            "crashed=%d timed_out=%d interrupted=%d cancelled=%d promoted=%d auto_blocked=%d",
            slug, len(spawned or ()), res.reclaimed,
            len(res.crashed) if hasattr(res.crashed, "__len__") else 0,
            len(res.timed_out) if hasattr(res.timed_out, "__len__") else 0,
            interrupted,
            cancelled,
            res.promoted,
            len(res.auto_blocked) if hasattr(res.auto_blocked, "__len__") else 0,
        )
    return any_spawned

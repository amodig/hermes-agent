"""External drain-control marker contract (dashboard → gateway).

No control channel exists into a running gateway, so begin/cancel-drain writes
(or removes) ``{HERMES_HOME}/.drain_request.json`` and a gateway watcher reacts;
an ACTIVE marker means ``gateway_state -> "draining"``.  Two lenient staleness
signals (either suffices): epoch mismatch (HERMES_HOME is a durable volume on
Hermes Cloud, so a marker survives the restart a drain-gated action ends in and
would park the fresh gateway in ``draining`` forever) and expiry (same-epoch
orphan past :data:`DRAIN_REQUEST_MAX_AGE_SECONDS`; re-writing refreshes it).
Reading never raises: a malformed file reads as ``{}`` — still drain-active
(fail-safe toward quiescing).  Staleness rejects only on a *definite* verdict.
"""
from __future__ import annotations

import contextlib
import functools
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from gateway.memory_status import _parse_iso
from hermes_constants import get_hermes_home
from utils import atomic_json_write

_log = logging.getLogger(__name__)

_DRAIN_REQUEST_FILENAME = ".drain_request.json"
# Drain-gated lifecycle actions complete in minutes; an hour bounds the wedge a leaked
# marker can cause. Long drains refresh the marker instead of raising this.
# Max-age fallback for a same-epoch orphaned marker (#85433). Long-running drains refresh the marker via
# write_drain_request() (idempotent re-write bumps ``requested_at``) rather than raising this bound.
DRAIN_REQUEST_MAX_AGE_SECONDS = 3600.0
# Dedup for the expired-marker warning (the watcher re-reads every second); keyed by
# ``requested_at`` so a keep-alive re-write that later expires logs again.
_expiry_logged_for: Optional[str] = None

# Canonical ``/proc/sys/kernel/random/boot_id`` shape (8-4-4-4-12 lowercase hex).
_BOOT_ID_RE = re.compile(
    r"\A[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\Z"
)


@functools.lru_cache(maxsize=1)
def current_instantiation_epoch() -> str:
    """Identity of THIS container / VM instantiation ("<boot_id>:<pid1_start>").

    Stable for the life of PID 1 (a gateway-only respawn keeps honouring an
    in-flight drain) but changes when the machine is recreated: boot_id on a VM
    reboot, PID 1's start time on ``docker restart``.  ``""`` when neither is
    readable (non-Linux, no ``/proc``) disables the epoch check — never fail-closed.
    """
    boot_id = pid1_start = ""
    with contextlib.suppress(OSError):
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()
    with contextlib.suppress(OSError, IndexError):
        # "<pid> (<comm>) <state> ...": comm may contain spaces/parens, so split on the
        # LAST ')'. starttime is field 22 (1-indexed) = tail index 19.
        pid1_start = Path("/proc/1/stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split()[19]
    return f"{boot_id}:{pid1_start}" if (boot_id or pid1_start) else ""


def _is_boot_id(value: str) -> bool:
    """A canonical ``/proc/sys/kernel/random/boot_id`` value (8-4-4-4-12 hex).

    The grouping is part of the check: a 32-hex string with hyphens in the wrong
    places is malformed provenance, and malformed provenance must never prove a
    host change.
    """
    return bool(_BOOT_ID_RE.match(value))


def split_instantiation_epoch(value: Any) -> tuple[Optional[str], Optional[str]]:
    """Validated ``(boot_id, pid1_start)`` components of an epoch string.

    Either component is ``None`` when it is missing or malformed: a receipt
    written before this contract, a non-Linux host, or a partially readable
    ``/proc``. "Unknown" is never "changed".
    """
    boot_id, _, pid1_start = str(value or "").strip().partition(":")
    return (
        boot_id.lower() if _is_boot_id(boot_id) else None,
        pid1_start if pid1_start.isdigit() and pid1_start else None,
    )


def instantiation_changed(recorded: Any, current: Any) -> bool:
    """True ONLY when readable components positively prove a new instantiation.

    A differing valid boot UUID proves a reboot; within the SAME valid boot,
    differing valid PID 1 start ticks prove a new container instantiation
    (``docker restart`` keeps the kernel boot id). Anything unreadable,
    malformed or incomparable is unknown and must not pause a task, so receipts
    that predate host epochs keep their existing reclaim behavior.
    """
    recorded_boot, recorded_pid1 = split_instantiation_epoch(recorded)
    current_boot, current_pid1 = split_instantiation_epoch(current)
    if recorded_boot is None or current_boot is None:
        return False
    if recorded_boot != current_boot:
        return True
    return (
        recorded_pid1 is not None
        and current_pid1 is not None
        and recorded_pid1 != current_pid1
    )


def drain_request_path(home: Optional[Path] = None) -> Path:
    """Absolute path to the drain-request marker, respecting HERMES_HOME."""
    return Path(home if home is not None else get_hermes_home()) / _DRAIN_REQUEST_FILENAME


def write_drain_request(
    *, principal: str = "drain-control", suppress_notification: bool = False, home: Optional[Path] = None
) -> dict[str, Any]:
    """Write the begin-drain marker atomically; returns the payload.

    Re-writing refreshes ``requested_at`` (keep-alive past the max-age).
    ``suppress_notification`` skips ONLY the home-channel "gateway shutting down"
    broadcast (the per-session interrupt ping is never suppressed); which drains
    are quiet is the caller's policy.  Stamped with the instantiation epoch so a
    copy surviving a machine restart on the durable volume reads as stale.
    """
    payload = {
        "action": "drain", "requested_at": datetime.now(timezone.utc).isoformat(), "principal": principal,
        "epoch": current_instantiation_epoch(), "suppress_notification": bool(suppress_notification),
    }
    atomic_json_write(drain_request_path(home), payload)
    return payload


def clear_drain_request(*, home: Optional[Path] = None) -> bool:
    """Remove the drain marker (cancel-drain, idempotent). Returns True if one existed."""
    path = drain_request_path(home)
    try:
        path.unlink()
        return True
    except OSError as e:
        if not isinstance(e, FileNotFoundError):
            _log.warning("drain-control: failed to remove %s: %s", path, e)
        return False


def _marker_is_expired(body: dict[str, Any]) -> bool:
    """True iff ``requested_at`` parses AND is older than the max-age.

    Missing/unparseable and future-dated (clock skew) timestamps are honoured.
    Logged once per marker, not per poll — the operator's breadcrumb for a leak.

    See #85433.
    """
    global _expiry_logged_for
    raw = body.get("requested_at")
    requested_at = _parse_iso(raw)
    if requested_at is None:
        return False
    age = (datetime.now(timezone.utc) - requested_at).total_seconds()
    if age <= DRAIN_REQUEST_MAX_AGE_SECONDS:
        return False
    if _expiry_logged_for != raw:
        _expiry_logged_for = raw
        _log.warning(
            "drain-control: ignoring expired drain marker (requested_at=%s, age=%.0fs > max %.0fs, principal=%s) "
            "— the drain that wrote it was never cancelled; treating as stale so the gateway keeps accepting turns.",
            raw, age, DRAIN_REQUEST_MAX_AGE_SECONDS, body.get("principal"),
        )
    return True


def _active_drain_body(home: Optional[Path]) -> Optional[dict[str, Any]]:
    """Marker body if present AND not stale (definite epoch mismatch or expired), else None."""
    body = read_drain_request(home=home)
    if body is None:
        return None
    current, marker_epoch = current_instantiation_epoch(), body.get("epoch")
    if (current and marker_epoch and marker_epoch != current) or _marker_is_expired(body):
        return None
    return body


def drain_requested(*, home: Optional[Path] = None) -> bool:
    """True iff an active (present, same-epoch, unexpired) begin-drain marker exists.

    A marker whose ``epoch`` does not match the current instantiation epoch is treated as absent: it
    survived a container/VM restart (HERMES_HOME is a durable Fly volume on Hermes Cloud) and the lifecycle
    action that triggered the drain has already completed — honouring it would wedge the freshly-restarted
    gateway in ``draining`` (NS-570). A marker whose ``requested_at`` is older than
    :data:`DRAIN_REQUEST_MAX_AGE_SECONDS` is likewise treated as absent: it is a same-epoch orphan whose
    drain-gated action completed without a restart and was never cancelled (#85433). Both staleness checks
    are lenient (see :func:`_marker_epoch_is_stale` / :func:`_marker_is_expired`): a legacy/corrupt marker
    with no epoch and no timestamp, or an environment without ``/proc``, still reads as drain-active.
    """
    return _active_drain_body(home) is not None


def drain_notification_suppressed(*, home: Optional[Path] = None) -> bool:
    """True iff an ACTIVE marker asks to suppress the shutdown broadcast.

    Same activeness rule as :func:`drain_requested`, so an orphan can never silence
    a fresh gateway; a marker without the field reads False (fail toward louder).

    "Active" means exactly what :func:`drain_requested` means — a marker present AND stamped with the
    current instantiation epoch AND not past its max-age. A stale (other-epoch) marker that survived a
    machine restart on the durable HERMES_HOME volume, or an expired same-epoch orphan (#85433), is ignored
    here just as it is for drain state (NS-570): we must never let an orphaned marker's flag silence a
    *fresh* gateway's legitimate shutdown broadcast.
    """
    body = _active_drain_body(home)
    return bool(body and body.get("suppress_notification"))


def read_drain_request(*, home: Optional[Path] = None) -> Optional[dict[str, Any]]:
    """Return the marker payload, ``{}`` if present but unparseable, ``None`` if absent. Never raises."""
    path = drain_request_path(home)
    try:
        raw = path.read_text(encoding="utf-8-sig")
    except OSError as e:
        if not isinstance(e, FileNotFoundError):
            _log.warning("drain-control: failed to read %s: %s", path, e)
        return None
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}

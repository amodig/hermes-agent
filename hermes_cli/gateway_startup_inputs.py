"""Startup file metadata and non-secret adapter routes, never a live reload."""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
import stat

import psutil

from hermes_cli.managed_scope import get_managed_dir
from hermes_constants import PROFILE_ID_RE, get_hermes_home, get_process_hermes_home, named_profile_is_live
from utils import atomic_json_write

_SNAPSHOT_NAME = "gateway_startup_inputs.json"
_pending: dict | None = None
_context: tuple | None = None
_started = False
_published = False
_log = logging.getLogger(__name__)


def startup_input_paths(home: Path, project_env: Path, secondary_homes=(), *, managed_dirs=()) -> dict[Path, str]:
    """The single inventory shared by the startup writer and installed verifier."""
    paths = {project_env: "dotenv"}
    for profile in (home, *secondary_homes):
        paths[profile / ".env"] = "dotenv"
        paths[profile / ".op.env"] = "dotenv"
        paths[profile / "config.yaml"] = "configuration"
        paths[profile / "gateway.json"] = "configuration"
    for managed in (*managed_dirs, get_managed_dir()):
        if managed is not None:
            paths[managed / ".env"] = "dotenv"
            paths[managed / "config.yaml"] = "configuration"
    return paths


def _stat_identity(info: os.stat_result) -> tuple:
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _file_metadata(path: Path) -> dict | None:
    try:
        link = path.lstat()
    except FileNotFoundError:
        return None
    info = path.stat()  # A dangling symlink is not an absent optional input.
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"{path}: startup input is not a regular file")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if (_stat_identity(path.lstat()) != _stat_identity(link)
            or _stat_identity(path.stat()) != _stat_identity(info)):
        raise ValueError(f"{path}: startup input changed while being read")
    return {
        "size": info.st_size, "mtime_ns": info.st_mtime_ns,
        "ctime_ns": info.st_ctime_ns, "device": info.st_dev, "inode": info.st_ino,
        "link_mtime_ns": link.st_mtime_ns, "link_ctime_ns": link.st_ctime_ns,
        "link_inode": link.st_ino, "sha256": digest,
    }


def _capture_input(path: Path, kind: str) -> dict:
    if kind == "dotenv" and path.is_file():
        from hermes_cli.env_loader import _sanitize_env_file_if_needed

        _sanitize_env_file_if_needed(path)
    return {"kind": kind, "state": _file_metadata(path)}


def _authority_selection(home: Path) -> dict:
    managed = get_managed_dir()
    return {"home": str(home.absolute()),
            "managed_dir": str(managed.absolute()) if managed is not None else None}


def capture_startup_inputs(home: Path, project_env: Path, secondary_homes=(), *,
                           pid: int | None = None, start_time: float | None = None) -> dict:
    pid = os.getpid() if pid is None else pid
    managed = get_managed_dir()
    return {
        "pid": pid,
        "start_time": psutil.Process(pid).create_time() if start_time is None else start_time,
        "managed_dirs": [str(managed.absolute())] if managed is not None else [],
        "authority": _authority_selection(home),
        "inputs": {str(path.absolute()): _capture_input(path, kind)
                   for path, kind in startup_input_paths(home, project_env, secondary_homes).items()},
    }


def _require_matching_file(path: Path, kind: str, recorded: dict | None) -> None:
    if _file_metadata(path) != recorded:
        raise ValueError(
            f"loaded {kind} state cannot be established: {path}: "
            "file or file set changed after process start")


def write_startup_inputs(home: Path, snapshot: dict) -> None:
    """Publish once at startup, after ownership is claimed; never called by status ticks."""
    for name, entry in snapshot["inputs"].items():
        _require_matching_file(Path(name), entry["kind"], entry["state"])
    atomic_json_write(home / _SNAPSHOT_NAME, snapshot, mode=0o600)


def require_unchanged_startup_inputs(home: Path, project_env: Path, secondary_homes=(), *,
                                     pid: int, start_time: float, check_authority: bool = True) -> dict:
    snapshot = json.loads((home / _SNAPSHOT_NAME).read_text(encoding="utf-8"))
    if (not isinstance(snapshot, dict) or snapshot.get("pid") != pid
            or snapshot.get("start_time") != start_time or not isinstance(snapshot.get("inputs"), dict)):
        raise ValueError("startup input snapshot does not identify the serving gateway PID/start time")
    if check_authority and snapshot.get("authority") != _authority_selection(home):
        raise ValueError("startup authority selection differs from the reconstructed home/managed layer")
    managed_dirs = snapshot.get("managed_dirs")
    if not isinstance(managed_dirs, list) or any(not isinstance(p, str) for p in managed_dirs):
        raise ValueError("startup input snapshot has no managed-layer inventory")
    for path, kind in startup_input_paths(
        home, project_env, secondary_homes, managed_dirs=map(Path, managed_dirs),
    ).items():
        entry = snapshot["inputs"].get(str(path.absolute()))
        if not isinstance(entry, dict) or entry.get("kind") != kind or "state" not in entry:
            raise ValueError(f"{path}: startup input snapshot has no presence record")
        _require_matching_file(path, kind, entry["state"])
    return snapshot


def begin_gateway_startup_inputs(home: Path, project_env: Path) -> None:
    """Normalize dotenv, then freeze inputs before loading; imports never publish."""
    global _started, _pending, _context
    if _started:
        return
    _started = True
    try:
        profiles = home / "profiles"
        secondary = tuple(
            path for path in sorted(profiles.iterdir())
            if path.name != "default" and PROFILE_ID_RE.match(path.name) and named_profile_is_live(path)
        ) if profiles.is_dir() else ()
        _context = (home, project_env, secondary)
        _pending = capture_startup_inputs(*_context)
    except (OSError, ValueError, psutil.Error):
        _log.warning("Gateway startup input metadata unavailable; installed verification will refuse", exc_info=True)


def extend_gateway_startup_inputs() -> None:
    """Freeze newly selected layers before consumption and retain the current authority."""
    global _pending
    if _pending is None or _published:
        return
    try:
        home = get_process_hermes_home()
        _pending["authority"] = _authority_selection(home)
        managed = get_managed_dir()
        if managed is not None and str(managed.absolute()) not in _pending["managed_dirs"]:
            _pending["managed_dirs"].append(str(managed.absolute()))
        paths = startup_input_paths(*_context)
        paths.update(startup_input_paths(home, _context[1]))
        for path, kind in paths.items():
            key = str(path.absolute())
            if key not in _pending["inputs"]:
                _pending["inputs"][key] = _capture_input(path, kind)
    except (OSError, ValueError):
        _pending = None
        _log.warning("Gateway startup input metadata unavailable; installed verification will refuse", exc_info=True)


def record_gateway_startup_route(platform: str, route: dict) -> None:
    """Adapters supply only non-secret route fields after resolving their scoped inputs."""
    if _pending is not None and not _published:
        routes = _pending.setdefault("routes", {}).setdefault(str(get_hermes_home().absolute()), {})
        routes[platform] = route


def publish_gateway_startup_inputs() -> None:
    """Only the startup PID-claim winner publishes, at most once per process."""
    global _published
    if _pending is None or _published:
        return
    _published = True
    try:
        write_startup_inputs(_context[0], _pending)
    except (OSError, ValueError):
        _log.warning("Gateway startup inputs changed or could not be recorded; installed verification will refuse", exc_info=True)

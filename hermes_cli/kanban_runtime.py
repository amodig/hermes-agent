"""Frozen Hermes runtime identity and Kanban worker bootstrap fence.

The identity separates code provenance from process liveness.  Code fields must
match exactly across a parent and child; PID and process start time fence the
actual worker claim against PID reuse.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Mapping, Optional
import uuid


RUNTIME_IDENTITY_PROTOCOL = 1
_BOOTSTRAP_TIMEOUT_SECONDS = 10.0
_IDENTITY_ROOTS = ("hermes_cli", "tools", "agent", "gateway", "plugins")


class RuntimeIdentityError(RuntimeError):
    """Raised when a worker's runtime cannot be trusted."""


@dataclass(frozen=True)
class RuntimeIdentity:
    protocol: int
    code_sha: str
    version: str
    module_root: str
    fingerprint: str
    pid: int
    start_time: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol,
            "code_sha": self.code_sha,
            "version": self.version,
            "module_root": self.module_root,
            "fingerprint": self.fingerprint,
            "pid": self.pid,
            "start_time": self.start_time,
        }

    @classmethod
    def from_value(cls, value: Mapping[str, Any]) -> "RuntimeIdentity":
        try:
            identity = cls(
                protocol=int(value["protocol"]),
                code_sha=str(value["code_sha"]),
                version=str(value["version"]),
                module_root=str(value["module_root"]),
                fingerprint=str(value["fingerprint"]),
                pid=int(value["pid"]),
                start_time=int(value["start_time"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeIdentityError("malformed runtime identity") from exc
        if identity.protocol != RUNTIME_IDENTITY_PROTOCOL:
            raise RuntimeIdentityError(f"unsupported runtime identity protocol {identity.protocol}")
        if identity.pid <= 0 or identity.start_time <= 0:
            raise RuntimeIdentityError("runtime identity must include a live pid and start time")
        return identity


def _module_root(module_root: Optional[os.PathLike[str] | str] = None) -> Path:
    return Path(module_root or Path(__file__).resolve().parents[1]).resolve()


def process_start_time(pid: Optional[int] = None) -> int:
    """Return a PID-reuse-resistant process start marker on every platform."""
    pid = int(pid or os.getpid())
    if sys.platform.startswith("linux"):
        try:
            fields = Path(f"/proc/{pid}/stat").read_text(encoding="ascii").split()
            # Linux procfs field 22 is process start time in clock ticks.
            return int(fields[21])
        except (OSError, IndexError, ValueError):
            pass
    try:
        import psutil

        created = psutil.Process(pid).create_time()
    except Exception as exc:
        if pid == os.getpid():
            return _PROCESS_START_TIME
        raise RuntimeIdentityError(
            f"process start time is unavailable for pid {pid}"
        ) from exc
    # Microseconds are stable across the parent/child psutil calls while
    # avoiding float-noise from the platform API.
    return int(round(float(created) * 1_000_000))


_PROCESS_START_TIME = int(time.time_ns())


def _git_sha(root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError, TypeError):
        return "unknown"
    sha = result.stdout.strip()
    return sha or "unknown"


def _version(root: Path) -> str:
    try:
        text = (root / "hermes_cli" / "__init__.py").read_text(encoding="utf-8")
    except OSError:
        return "unknown"
    marker = '__version__ = "'
    start = text.find(marker)
    if start < 0:
        return "unknown"
    start += len(marker)
    end = text.find('"', start)
    return text[start:end] if end > start else "unknown"


def _fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    relatives: list[Path] = []
    for directory in _IDENTITY_ROOTS:
        base = root / directory
        if not base.is_dir():
            raise RuntimeIdentityError(f"runtime identity directory is missing: {directory}")
        relatives.extend(
            path.relative_to(root)
            for path in base.rglob("*.py")
            if path.is_file()
            and not any(part.startswith(".") or part == "__pycache__" for part in path.parts)
        )
    for relative in sorted(set(relatives), key=lambda value: value.as_posix()):
        path = root / relative
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


_FROZEN_RUNTIME_IDENTITY: Optional[RuntimeIdentity] = None


def runtime_identity(
    module_root: Optional[os.PathLike[str] | str] = None,
    *,
    pid: Optional[int] = None,
    start_time: Optional[int] = None,
) -> RuntimeIdentity:
    global _FROZEN_RUNTIME_IDENTITY
    if module_root is None and pid is None and start_time is None:
        if _FROZEN_RUNTIME_IDENTITY is None:
            _FROZEN_RUNTIME_IDENTITY = runtime_identity(
                _module_root(), pid=os.getpid(), start_time=process_start_time(),
            )
        return _FROZEN_RUNTIME_IDENTITY
    root = _module_root(module_root)
    process_pid = int(pid or os.getpid())
    return RuntimeIdentity(
        protocol=RUNTIME_IDENTITY_PROTOCOL,
        code_sha=_git_sha(root),
        version=_version(root),
        module_root=str(root),
        fingerprint=_fingerprint(root),
        pid=process_pid,
        start_time=int(start_time or process_start_time(process_pid)),
    )


def code_identity(value: RuntimeIdentity | Mapping[str, Any]) -> tuple[Any, ...]:
    identity = value if isinstance(value, RuntimeIdentity) else RuntimeIdentity.from_value(value)
    return (
        identity.protocol,
        identity.code_sha,
        identity.version,
        identity.module_root,
        identity.fingerprint,
    )


def same_code_identity(expected: RuntimeIdentity | Mapping[str, Any], actual: RuntimeIdentity | Mapping[str, Any]) -> bool:
    try:
        return code_identity(expected) == code_identity(actual)
    except RuntimeIdentityError:
        return False


def same_runtime_identity(expected: RuntimeIdentity | Mapping[str, Any], actual: RuntimeIdentity | Mapping[str, Any]) -> bool:
    try:
        left = expected if isinstance(expected, RuntimeIdentity) else RuntimeIdentity.from_value(expected)
        right = actual if isinstance(actual, RuntimeIdentity) else RuntimeIdentity.from_value(actual)
    except RuntimeIdentityError:
        return False
    return left == right


def assert_runtime_import_root(
    module_root: Optional[os.PathLike[str] | str] = None,
    *,
    expected: RuntimeIdentity | Mapping[str, Any] | None = None,
    identity: RuntimeIdentity | Mapping[str, Any] | None = None,
) -> RuntimeIdentity:
    """Reject mixed checkout imports before a worker can touch the board.

    Callers that already fingerprinted the same root may pass ``identity`` to
    reuse it while retaining the import-root check.
    """
    root = _module_root(module_root)
    observed = identity or runtime_identity(root, pid=os.getpid(), start_time=process_start_time())
    if expected is not None and not same_code_identity(expected, observed):
        raise RuntimeIdentityError("runtime code changed during startup")
    prefixes = ("hermes_cli", "gateway", "tools", "agent")
    mixed: list[str] = []
    for name, module in tuple(sys.modules.items()):
        if not name.startswith(prefixes):
            continue
        location = getattr(module, "__file__", None)
        if not location:
            continue
        try:
            Path(location).resolve().relative_to(root)
        except ValueError:
            mixed.append(f"{name}={location}")
    if mixed:
        raise RuntimeIdentityError(
            "mixed runtime import roots: " + ", ".join(sorted(mixed)[:8])
        )
    return observed

def prospective_identity(module_root: Optional[os.PathLike[str] | str] = None) -> RuntimeIdentity:
    """Return the frozen code identity with the caller's live process marker."""
    if module_root is None:
        frozen = runtime_identity()
        return RuntimeIdentity(
            protocol=frozen.protocol,
            code_sha=frozen.code_sha,
            version=frozen.version,
            module_root=frozen.module_root,
            fingerprint=frozen.fingerprint,
            pid=os.getpid(),
            start_time=process_start_time(),
        )
    return runtime_identity(module_root, pid=max(os.getpid(), 1), start_time=process_start_time())
def encode_identity(identity: RuntimeIdentity | Mapping[str, Any]) -> str:
    value = identity.as_dict() if isinstance(identity, RuntimeIdentity) else RuntimeIdentity.from_value(identity).as_dict()
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def decode_identity(value: str | bytes | Mapping[str, Any]) -> RuntimeIdentity:
    if isinstance(value, Mapping):
        return RuntimeIdentity.from_value(value)
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise RuntimeIdentityError("runtime identity is not JSON") from exc
    if not isinstance(parsed, Mapping):
        raise RuntimeIdentityError("runtime identity must be an object")
    return RuntimeIdentity.from_value(parsed)




def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    os.replace(temporary, path)


def verify_worker_ready(
    payload: Mapping[str, Any],
    expected: RuntimeIdentity | Mapping[str, Any],
    *,
    pid: int,
    preparation_id: str,
) -> RuntimeIdentity:
    if payload.get("ready") is not True or str(payload.get("preparation_id")) != str(preparation_id):
        raise RuntimeIdentityError("worker bootstrap preparation mismatch")
    actual = decode_identity(payload.get("runtime_identity"))
    if actual.pid != int(pid) or not same_code_identity(expected, actual):
        raise RuntimeIdentityError("worker runtime identity mismatch")
    if process_start_time(pid) != actual.start_time:
        raise RuntimeIdentityError("worker process start time mismatch")
    return actual


def worker_bootstrap_from_env() -> Optional[dict[str, Any]]:
    """Publish a worker identity and wait for the parent grant, if requested.

    The normal CLI has no bootstrap side effect. Dispatcher workers opt in with
    ``HERMES_KANBAN_BOOTSTRAP_PATH`` and remain unable to use Kanban tools until
    the parent grants the matching preparation.
    """
    path_raw = os.environ.get("HERMES_KANBAN_BOOTSTRAP_PATH", "").strip()
    if not path_raw:
        return None
    preparation_id = os.environ.get("HERMES_KANBAN_PREPARATION_ID", "").strip()
    expected_raw = os.environ.get("HERMES_KANBAN_EXPECTED_RUNTIME", "").strip()
    if not preparation_id or not expected_raw:
        raise RuntimeIdentityError("worker bootstrap environment is incomplete")
    expected = decode_identity(expected_raw)
    actual = runtime_identity()
    payload = {
        "ready": same_code_identity(expected, actual),
        "preparation_id": preparation_id,
        "runtime_identity": actual.as_dict(),
    }
    _atomic_write(Path(path_raw), payload)
    if not payload["ready"]:
        raise RuntimeIdentityError("worker runtime identity mismatch")
    if os.environ.get("HERMES_KANBAN_BOOTSTRAP_WAIT", "1").lower() in {"0", "false", "no", "off"}:
        os.environ["HERMES_KANBAN_RUNTIME_GRANTED"] = "1"
        return payload
    grant_queue: queue.Queue[Optional[str]] = queue.Queue(maxsize=1)

    def _read_grant() -> None:
        try:
            grant_queue.put(sys.stdin.readline())
        except (OSError, ValueError) as exc:
            grant_queue.put(None)

    threading.Thread(target=_read_grant, daemon=True).start()
    try:
        grant_line = grant_queue.get(timeout=_BOOTSTRAP_TIMEOUT_SECONDS)
    except queue.Empty as exc:
        raise RuntimeIdentityError("worker bootstrap grant timed out") from exc
    try:
        grant = json.loads(grant_line or "")
    except (json.JSONDecodeError, TypeError) as exc:
        raise RuntimeIdentityError("malformed worker bootstrap grant") from exc
    if (
        grant.get("grant") is not True
        or str(grant.get("preparation_id")) != preparation_id
        or not same_runtime_identity(grant.get("runtime_identity", {}), actual)
    ):
        raise RuntimeIdentityError("worker bootstrap grant mismatch")
    if grant.get("run_id") is not None:
        os.environ["HERMES_KANBAN_RUN_ID"] = str(grant["run_id"])
    if grant.get("claim_lock"):
        os.environ["HERMES_KANBAN_CLAIM_LOCK"] = str(grant["claim_lock"])
    os.environ["HERMES_KANBAN_RUNTIME_GRANTED"] = "1"
    return payload


def runtime_identity_json(module_root: Optional[os.PathLike[str] | str] = None) -> str:
    return encode_identity(runtime_identity(module_root))

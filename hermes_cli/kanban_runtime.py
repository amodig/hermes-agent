"""Frozen Hermes runtime identity and Kanban worker bootstrap fence.

The identity separates code provenance from process liveness.  Code fields must
match exactly across a parent and child; PID and process start time fence the
actual worker claim against PID reuse.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Any, Mapping, Optional
import uuid


RUNTIME_IDENTITY_PROTOCOL = 1
_BOOTSTRAP_TIMEOUT_SECONDS = 10.0
_IDENTITY_ROOTS = (
    "hermes_cli",
    "tools",
    "agent",
    "gateway",
    "plugins",
    "providers",
    "cron",
    "acp_adapter",
    "tui_gateway",
)
_RUNTIME_RESOURCE_ROOTS = (
    ("skills", "HERMES_BUNDLED_SKILLS"),
    ("optional-skills", "HERMES_OPTIONAL_SKILLS"),
    ("locales", "HERMES_BUNDLED_LOCALES"),
    ("optional-mcps", "HERMES_OPTIONAL_MCPS"),
    ("plugins", "HERMES_BUNDLED_PLUGINS"),
)

_IDENTITY_ASSETS = ("skills/devops/sdlc-review/SKILL.md",)
_RUNTIME_DEPENDENCY_ROOT_NAMES = frozenset({"site-packages", "dist-packages"})
_BOOTSTRAP_INPUT_ENV = (
    "HERMES_KANBAN_BOOTSTRAP_PATH",
    "HERMES_KANBAN_PREPARATION_ID",
    "HERMES_KANBAN_EXPECTED_RUNTIME",
    "HERMES_KANBAN_BOOTSTRAP_WAIT",
)


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
    def _valid(value: str) -> str:
        value = value.strip()
        return value if len(value) == 40 and all(char in "0123456789abcdef" for char in value) else ""

    try:
        marker = root / ".git"
        if marker.is_dir():
            git_dir = marker
        else:
            text = marker.read_text(encoding="utf-8").strip()
            prefix, separator, value = text.partition(":")
            if prefix != "gitdir" or not separator:
                raise OSError("invalid gitdir marker")
            git_dir = Path(value.strip())
            if not git_dir.is_absolute():
                git_dir = (root / git_dir).resolve()

        head = (git_dir / "HEAD").read_text(encoding="ascii").strip()
        if head.startswith("ref: "):
            ref = head[5:].strip()
            common_dir = git_dir
            commondir_file = git_dir / "commondir"
            if commondir_file.is_file():
                common = Path(commondir_file.read_text(encoding="utf-8").strip())
                if not common.is_absolute():
                    common = (git_dir / common).resolve()
                common_dir = common
            candidates = [git_dir / ref]
            if common_dir != git_dir:
                candidates.append(common_dir / ref)
            for candidate in candidates:
                try:
                    sha = _valid(candidate.read_text(encoding="ascii"))
                except OSError:
                    continue
                if sha:
                    return sha
            for packed in (git_dir / "packed-refs", common_dir / "packed-refs"):
                if not packed.is_file():
                    continue
                for line in packed.read_text(encoding="ascii").splitlines():
                    if line and not line.startswith(("#", "^")):
                        sha, _, packed_ref = line.partition(" ")
                        if packed_ref == ref:
                            return _valid(sha) or "unknown"
            return "unknown"
        return _valid(head) or "unknown"
    except (OSError, UnicodeError, ValueError):
        pass
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
    return sha if len(sha) == 40 else "unknown"


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
def _identity_root_module_names(root: Path) -> frozenset[str]:
    return frozenset(path.stem for path in root.glob("*.py") if path.is_file())


def _pin_runtime_import_root(root: Path) -> None:
    """Put the verified checkout before the worker workspace on ``sys.path``."""
    root = root.resolve()
    retained: list[str] = []
    for entry in sys.path:
        try:
            if Path(entry or os.getcwd()).resolve() == root:
                continue
        except OSError:
            pass
        retained.append(entry)
    sys.path[:] = [str(root), *retained]
_FROZEN_IMPORT_ROOT: Optional[Path] = None

_RUNTIME_SNAPSHOT_OWNER_FILE = ".hermes-kanban-runtime-owner.json"
_RUNTIME_SNAPSHOT_ORPHAN_GRACE_SECONDS = 3600


def _sweep_runtime_snapshots() -> None:
    """Remove snapshots left by workers from an older gateway process."""
    temp_root = Path(tempfile.gettempdir()).resolve()
    try:
        snapshots = temp_root.glob("hermes-kanban-runtime-*")
    except OSError:
        return
    now = time.time()
    for snapshot in snapshots:
        try:
            if not snapshot.is_dir():
                continue
            owner_path = snapshot / _RUNTIME_SNAPSHOT_OWNER_FILE
            try:
                owner = json.loads(owner_path.read_text(encoding="utf-8"))
                pid = int(owner["pid"])
                start_time = int(owner["start_time"])
            except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
                owner = None
                pid = 0
                start_time = 0
            if owner is not None:
                try:
                    if pid > 0 and process_start_time(pid) == start_time:
                        continue
                except (RuntimeIdentityError, OSError, ValueError):
                    pass
                _remove_frozen_import_root(snapshot)
                continue
            if now - snapshot.stat().st_mtime < _RUNTIME_SNAPSHOT_ORPHAN_GRACE_SECONDS:
                continue
            _remove_frozen_import_root(snapshot)
        except OSError:
            continue


def _write_runtime_snapshot_owner(
    snapshot: Path,
    *,
    pid: Optional[int] = None,
    start_time: Optional[int] = None,
) -> None:
    owner_pid = int(pid or os.getpid())
    owner_start_time = (
        process_start_time(owner_pid) if start_time is None else int(start_time)
    )
    _atomic_write(
        snapshot / _RUNTIME_SNAPSHOT_OWNER_FILE,
        {"pid": owner_pid, "start_time": owner_start_time},
    )


def _is_frozen_import_location(location: str) -> bool:
    root = _FROZEN_IMPORT_ROOT
    if root is None:
        return False
    normalized_location = str(location).replace("\\", "/")
    normalized_root = str(root).replace("\\", "/")
    return normalized_location.startswith(normalized_root + "/")


def _remove_frozen_import_root(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


def cleanup_runtime_snapshot(path: Optional[os.PathLike[str] | str]) -> None:
    """Remove a worker snapshot path previously emitted by this module."""
    if not path:
        return
    snapshot = Path(path).resolve()
    if (
        snapshot.parent != Path(tempfile.gettempdir()).resolve()
        or not snapshot.name.startswith("hermes-kanban-runtime-")
    ):
        return
    _remove_frozen_import_root(snapshot)

def _runtime_dependency_roots() -> tuple[Path, ...]:
    """Return active interpreter package roots that can change during an update."""
    roots: list[Path] = []
    seen: set[Path] = set()
    for entry in sys.path:
        if not entry:
            continue
        try:
            candidate = Path(entry)
            if candidate.name not in _RUNTIME_DEPENDENCY_ROOT_NAMES:
                continue
            candidate = candidate.resolve()
        except (OSError, RuntimeError, TypeError):
            continue
        if candidate.is_dir() and candidate not in seen:
            roots.append(candidate)
            seen.add(candidate)
    return tuple(roots)


def _copy_runtime_dependency_file(source: str, destination: str) -> str:
    # Dependency installers replace non-source artifacts atomically. Hardlinks keep that
    # snapshot cheap; Python sources are copied because an in-place edit must not leak through.
    if Path(source).suffix in {".py", ".pyi"}:
        shutil.copy2(source, destination)
    else:
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
    return destination

def _pin_loaded_dependency_imports(
    source_roots: tuple[Path, ...], destinations: list[Path],
) -> None:
    """Point already-imported dependency packages at their frozen submodule roots."""
    pairs = tuple(zip(source_roots, destinations))
    if not pairs:
        return

    def snapshot_path(location: object) -> Optional[Path]:
        try:
            resolved = Path(location).resolve()
        except (OSError, RuntimeError, TypeError):
            return None
        for source, destination in pairs:
            try:
                relative = resolved.relative_to(source)
            except ValueError:
                continue
            return destination / relative
        return None

    for module in tuple(sys.modules.values()):
        package_path = getattr(module, "__path__", None)
        if package_path is None:
            continue
        try:
            entries = tuple(package_path)
        except (TypeError, ValueError):
            continue
        mapped_paths: list[object] = []
        changed = False
        for entry in entries:
            mapped = snapshot_path(entry)
            mapped_paths.append(str(mapped) if mapped is not None else entry)
            changed = changed or mapped is not None
        if not changed:
            continue
        try:
            module.__path__ = mapped_paths
            spec = getattr(module, "__spec__", None)
            if spec is not None and spec.submodule_search_locations is not None:
                spec.submodule_search_locations = mapped_paths
            for attribute in ("__file__", "__cached__"):
                mapped = snapshot_path(getattr(module, attribute, None))
                if mapped is not None:
                    setattr(module, attribute, str(mapped))
            if spec is not None:
                mapped = snapshot_path(getattr(spec, "origin", None))
                if mapped is not None:
                    spec.origin = str(mapped)
        except (AttributeError, TypeError):
            pass




def _freeze_runtime_import_root(
    root: Path, *, include_dependencies: bool = False,
) -> Path:
    """Serve future runtime imports from a pre-grant source snapshot."""
    global _FROZEN_IMPORT_ROOT
    if _FROZEN_IMPORT_ROOT is not None:
        return _FROZEN_IMPORT_ROOT
    root = root.resolve()
    members = _runtime_snapshot_members(root)
    dependency_roots = _runtime_dependency_roots() if include_dependencies else ()
    dependency_destinations: list[Path] = []
    snapshot_root = Path(tempfile.mkdtemp(prefix="hermes-kanban-runtime-")).resolve()
    try:
        for name, path in sorted(members.items()):
            destination = snapshot_root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
        for index, source in enumerate(dependency_roots):
            destination = snapshot_root / ".third-party" / str(index)
            shutil.copytree(source, destination, copy_function=_copy_runtime_dependency_file)
            dependency_destinations.append(destination)
    except Exception:
        _remove_frozen_import_root(snapshot_root)
        raise
    for directory, env_var in _RUNTIME_RESOURCE_ROOTS:
        source = _resource_root(root, directory, env_var)
        if not source.is_dir():
            continue
        destination = snapshot_root / directory
        destination.mkdir(parents=True, exist_ok=True)
        os.environ[env_var] = str(destination)

    _FROZEN_IMPORT_ROOT = snapshot_root
    atexit.register(_remove_frozen_import_root, snapshot_root)
    _pin_runtime_import_root(snapshot_root)
    for destination in reversed(dependency_destinations):
        sys.path.insert(1, str(destination))
    _pin_loaded_dependency_imports(tuple(dependency_roots), dependency_destinations)
    for name, module in tuple(sys.modules.items()):
        package_path = getattr(module, "__path__", None)
        if package_path is None or not any(
            name == prefix or name.startswith(f"{prefix}.") for prefix in _IDENTITY_ROOTS
        ):
            continue
        try:
            module.__path__ = [str(snapshot_root.joinpath(*name.split(".")))]
        except (AttributeError, TypeError):
            pass
    return snapshot_root





def _resource_root(root: Path, directory: str, env_var: str) -> Path:
    override = os.environ.get(env_var, "").strip()
    if override:
        candidate = Path(override).resolve()
        if candidate.is_dir():
            return candidate
    return root / directory


def _identity_asset_path(root: Path, asset: str) -> Path:
    if asset.startswith("skills/"):
        path = _resource_root(root, "skills", "HERMES_BUNDLED_SKILLS")
    else:
        path = root
    path = path / Path(asset).relative_to("skills") if asset.startswith("skills/") else path / asset
    if path.is_file():
        return path
    raise RuntimeIdentityError(f"runtime identity asset is missing: {asset}")

def _runtime_snapshot_members(root: Path) -> dict[str, Path]:
    root = root.resolve()
    members: dict[str, Path] = {}

    for path in root.glob("*.py"):
        if path.is_file():
            members[path.name] = path

    def add_tree(base: Path, destination_root: str) -> None:
        for path in base.rglob("*"):
            relative = path.relative_to(base)
            if path.is_file() and not any(
                part.startswith(".") or part == "__pycache__" for part in relative.parts
            ):
                members[(Path(destination_root) / relative).as_posix()] = path

    for directory in _IDENTITY_ROOTS:
        if directory == "plugins":
            continue
        base = root / directory
        if not base.is_dir():
            raise RuntimeIdentityError(f"runtime identity directory is missing: {directory}")
        add_tree(base, directory)

    for directory, env_var in _RUNTIME_RESOURCE_ROOTS:
        base = _resource_root(root, directory, env_var)
        if not base.is_dir():
            if directory == "plugins":
                raise RuntimeIdentityError(f"runtime identity directory is missing: {directory}")
            continue
        add_tree(base, directory)

    for asset in _IDENTITY_ASSETS:
        members[asset] = _identity_asset_path(root, asset)
    return members


def _fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    for name, path in sorted(_runtime_snapshot_members(root).items()):
        digest.update(name.encode("utf-8"))
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
    prefixes = _IDENTITY_ROOTS
    root_modules = _identity_root_module_names(root)
    mixed: list[str] = []
    for name, module in tuple(sys.modules.items()):
        if not (
            name in root_modules
            or any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes)
        ):
            continue
        location = getattr(module, "__file__", None)
        if not location:
            continue
        if _is_frozen_import_location(str(location)):
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


def _read_bootstrap_message() -> dict[str, Any]:
    grant_queue: queue.Queue[Optional[str]] = queue.Queue(maxsize=1)

    def _read_line() -> None:
        try:
            grant_queue.put(sys.stdin.readline())
        except (OSError, ValueError):
            grant_queue.put(None)

    threading.Thread(target=_read_line, daemon=True).start()
    try:
        grant_line = grant_queue.get(timeout=_BOOTSTRAP_TIMEOUT_SECONDS)
    except queue.Empty as exc:
        raise RuntimeIdentityError("worker bootstrap message timed out") from exc
    try:
        message = json.loads(grant_line or "")
    except (json.JSONDecodeError, TypeError) as exc:
        raise RuntimeIdentityError("malformed worker bootstrap message") from exc
    if not isinstance(message, dict):
        raise RuntimeIdentityError("worker bootstrap message must be an object")
    return message

def worker_bootstrap_from_env() -> Optional[dict[str, Any]]:
    """Publish the pre-import identity and wait for the import release.

    Dispatcher workers first prove that the early checkout matches the
    dispatcher's snapshot.  The parent then releases startup imports; the
    post-import handshake performs the final identity check before Kanban
    tools are granted.
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
        "phase": "pre_import",
        "preparation_id": preparation_id,
        "runtime_identity": actual.as_dict(),
    }
    _atomic_write(Path(path_raw), payload)
    if not payload["ready"]:
        raise RuntimeIdentityError("worker runtime identity mismatch")
    _pin_runtime_import_root(Path(actual.module_root))
    if os.environ.get("HERMES_KANBAN_BOOTSTRAP_WAIT", "1").lower() in {"0", "false", "no", "off"}:
        os.environ["HERMES_KANBAN_RUNTIME_GRANTED"] = "1"
        for key in _BOOTSTRAP_INPUT_ENV:
            os.environ.pop(key, None)
        return payload

    message = _read_bootstrap_message()
    if message.get("grant") is True:
        if (
            str(message.get("preparation_id")) != preparation_id
            or not same_runtime_identity(message.get("runtime_identity", {}), actual)
        ):
            raise RuntimeIdentityError("worker bootstrap grant mismatch")
        if message.get("run_id") is not None:
            os.environ["HERMES_KANBAN_RUN_ID"] = str(message["run_id"])
        if message.get("claim_lock"):
            os.environ["HERMES_KANBAN_CLAIM_LOCK"] = str(message["claim_lock"])
        os.environ["HERMES_KANBAN_RUNTIME_GRANTED"] = "1"
        for key in _BOOTSTRAP_INPUT_ENV:
            os.environ.pop(key, None)
        return payload
    if (
        message.get("continue_imports") is not True
        or str(message.get("preparation_id")) != preparation_id
        or not same_runtime_identity(message.get("runtime_identity", {}), actual)
    ):
        raise RuntimeIdentityError("worker bootstrap import handshake mismatch")
    return payload


def worker_bootstrap_post_import(*, wait_for_grant: bool = True) -> Optional[dict[str, Any]]:
    """Verify imported worker command modules before granting Kanban access."""
    path_raw = os.environ.get("HERMES_KANBAN_BOOTSTRAP_PATH", "").strip()
    if not path_raw:
        return None
    preparation_id = os.environ.get("HERMES_KANBAN_PREPARATION_ID", "").strip()
    expected_raw = os.environ.get("HERMES_KANBAN_EXPECTED_RUNTIME", "").strip()
    if not preparation_id or not expected_raw:
        raise RuntimeIdentityError("worker bootstrap environment is incomplete")
    expected = decode_identity(expected_raw)
    # Snapshot every runtime source file before the final grant. Lazy turn and
    # tool imports then resolve from this immutable filesystem snapshot, not a mutable checkout.
    snapshot_root = _freeze_runtime_import_root(_module_root(), include_dependencies=True)
    if _fingerprint(snapshot_root) != expected.fingerprint:
        raise RuntimeIdentityError("runtime snapshot changed during startup")
    # ``main.py`` and ``cli.py`` defer these imports to keep ordinary CLI
    # startup cheap. Workers must load them before the final identity check;
    # otherwise an update can replace their source after this handshake and
    # before the lazy import.
    from cli import main as _cli_main  # noqa: F401
    from run_agent import AIAgent as _worker_agent  # noqa: F401
    from agent.agent_init import init_agent as _init_agent  # noqa: F401
    import agent.credits_tracker as _credits_tracker  # noqa: F401

    actual = assert_runtime_import_root(expected=expected)
    payload = {
        "ready": True,
        "phase": "post_import",
        "post_import": True,
        "preparation_id": preparation_id,
        "runtime_identity": actual.as_dict(),
        "runtime_snapshot": str(snapshot_root),
    }
    _atomic_write(Path(path_raw), payload)
    if wait_for_grant:
        _finish_worker_bootstrap_grant(expected, preparation_id, actual)
    return payload

def _finish_worker_bootstrap_grant(
    expected: RuntimeIdentity,
    preparation_id: str,
    actual: RuntimeIdentity,
) -> None:
    grant = _read_bootstrap_message()
    final = assert_runtime_import_root(expected=expected)
    if (
        grant.get("grant") is not True
        or str(grant.get("preparation_id")) != preparation_id
        or not same_runtime_identity(grant.get("runtime_identity", {}), actual)
        or not same_runtime_identity(final, actual)
    ):
        raise RuntimeIdentityError("worker bootstrap grant mismatch")
    if grant.get("run_id") is not None:
        os.environ["HERMES_KANBAN_RUN_ID"] = str(grant["run_id"])
    if grant.get("claim_lock"):
        os.environ["HERMES_KANBAN_CLAIM_LOCK"] = str(grant["claim_lock"])
    os.environ["HERMES_KANBAN_RUNTIME_GRANTED"] = "1"
    for key in _BOOTSTRAP_INPUT_ENV:
        os.environ.pop(key, None)


def worker_bootstrap_after_constructor() -> None:
    """Grant Kanban access only after the worker agent constructor completes."""
    path_raw = os.environ.get("HERMES_KANBAN_BOOTSTRAP_PATH", "").strip()
    if not path_raw:
        return
    preparation_id = os.environ.get("HERMES_KANBAN_PREPARATION_ID", "").strip()
    expected_raw = os.environ.get("HERMES_KANBAN_EXPECTED_RUNTIME", "").strip()
    if not preparation_id or not expected_raw:
        raise RuntimeIdentityError("worker bootstrap environment is incomplete")
    expected = decode_identity(expected_raw)
    actual = assert_runtime_import_root(expected=expected)
    _finish_worker_bootstrap_grant(expected, preparation_id, actual)

def runtime_identity_json(module_root: Optional[os.PathLike[str] | str] = None) -> str:
    return encode_identity(runtime_identity(module_root))

if not os.environ.get("HERMES_KANBAN_BOOTSTRAP_PATH"):
    _sweep_runtime_snapshots()

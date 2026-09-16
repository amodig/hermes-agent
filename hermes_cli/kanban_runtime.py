"""Frozen Hermes runtime identity and Kanban worker bootstrap fence.

The identity separates code provenance from process liveness.  Code fields must
match exactly across a parent and child; PID and process start time fence the
actual worker claim against PID reuse.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, replace
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
    generation: str = ""
    dependency_fingerprint: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol,
            "code_sha": self.code_sha,
            "version": self.version,
            "module_root": self.module_root,
            "fingerprint": self.fingerprint,
            "pid": self.pid,
            "start_time": self.start_time,
            "generation": self.generation,
            "dependency_fingerprint": self.dependency_fingerprint,
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
                generation=str(value.get("generation", "")),
                dependency_fingerprint=str(value.get("dependency_fingerprint", "")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeIdentityError("malformed runtime identity") from exc
        if identity.protocol != RUNTIME_IDENTITY_PROTOCOL:
            raise RuntimeIdentityError(f"unsupported runtime identity protocol {identity.protocol}")
        if identity.pid <= 0 or identity.start_time <= 0:
            raise RuntimeIdentityError("runtime identity must include a live pid and start time")
        if bool(identity.generation) != bool(identity.dependency_fingerprint):
            raise RuntimeIdentityError("generation identity must attest its dependencies")
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


def _fingerprint(root: Path) -> str:
    """Comparable deployed code/resources, not build inputs or sealed payloads."""
    from importlib.metadata import PathDistribution

    from hermes_constants import _packaged_dir
    from hermes_cli.kanban_runtime_generation import _digest_members, _members, _TREE_EXCLUDES

    code_suffixes = (".py", ".pyi", ".so", ".pyd", ".dll", ".dylib")
    # Match pyproject.toml's package discovery/data and setup.py's root modules.
    # Source-only manifests, package docs and native build inputs are not shipped.
    # Full generation digests independently seal those bytes and dependencies.
    package_data = {
        "hermes_cli": ("observability/schemas/*.json", "data/*.json", "local_runtime/*.json"),
        "gateway": ("assets/**/*",),
        "plugins": ("**/plugin.yaml", "**/plugin.yml"),
    }
    data_files = {
        member
        for name, patterns in package_data.items()
        for pattern in patterns
        for member in (root / name).glob(pattern)
        if member.is_file()
    }
    paths = {name: root / name for name in _IDENTITY_ROOTS}
    # A packaged module root may be shared with third-party distributions. RECORD
    # owns its root modules; source/editable trees use setup.py's discovery rule.
    wheel_metadata = next(
        (marker.parent for marker in root.glob("hermes_agent-*.dist-info/WHEEL") if marker.is_file()),
        None,
    )
    installed_modules = None
    if wheel_metadata is not None:
        files = PathDistribution(wheel_metadata).files
        if files is None:
            raise RuntimeIdentityError("installed runtime has no file inventory")
        installed_modules = {str(member) for member in files if len(member.parts) == 1}
    for path in root.iterdir():
        if (
            path.is_file() and path.suffix in code_suffixes and path.name != "setup.py"
            and (installed_modules is None or path.name in installed_modules)
        ):
            paths[path.name] = path

    # Logical names normalize Nix's relocated bundles. Reuse the import-safe
    # skills/catalog resolver, with a default rooted in the tree being attested.
    # Locale/plugin helpers bind __file__ (plugins also imports runtime state),
    # so mirror only their path selection here, without importing either loader.
    for name, variable in _RUNTIME_RESOURCE_ROOTS:
        default = root / name
        if name == "locales":
            # agent.i18n._locales_dir: strip, require a directory, no expanduser.
            override = os.getenv(variable, "").strip()
            path = Path(override) if override and Path(override).is_dir() else default
        elif name == "plugins":
            # hermes_cli.plugins.get_bundled_plugins_dir: raw override, no fallback
            # on a missing path, no stripping or user expansion.
            path = Path(os.getenv(variable) or default)
        else:
            path = _packaged_dir(variable, default, name)
        if path.is_dir():
            paths[f"bundled/{name}"] = path

    def members():
        for name, path in sorted(paths.items()):
            if path.is_file():
                yield name, path
            elif path.is_dir():
                # These exact frontend destinations and dependencies are build
                # outputs; Nix also omits skill index caches from its bundles.
                excludes = ("node_modules",)
                if name in {"bundled/skills", "bundled/optional-skills"}:
                    excludes += ("index-cache",)
                for relative, member in _members(path, source=True, exclude=excludes):
                    if name == "hermes_cli" and relative.split("/", 1)[0] in {"web_dist", "tui_dist"}:
                        continue
                    if name == "acp_adapter" and "/" in relative:
                        continue  # setuptools declares this package, not acp_adapter.*.
                    if (
                        member.is_file() and member.name not in _TREE_EXCLUDES
                        and (
                            name.startswith("bundled/") or member.suffix in code_suffixes
                            or member.name == "py.typed" or member in data_files
                        )
                    ):
                        yield f"{name}/{relative}", member

    return _digest_members(members())


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
    generation = os.environ.get("HERMES_KANBAN_RUNTIME_GENERATION")
    if generation and root == Path(generation).resolve() / "source":
        from hermes_cli.kanban_runtime_generation import generation_manifest
        frozen = RuntimeIdentity.from_value(generation_manifest(generation)["identity"])
        return replace(frozen, pid=process_pid, start_time=int(start_time or process_start_time(process_pid)))
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
        identity.generation,
        identity.dependency_fingerprint,
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
    """Reject mutable or mixed imports before the worker can touch the board."""
    root = _module_root(module_root)
    observed = identity or runtime_identity(root, pid=os.getpid(), start_time=process_start_time())
    if expected is not None and not same_code_identity(expected, observed):
        raise RuntimeIdentityError("runtime code changed during startup")
    generation = os.environ.get("HERMES_KANBAN_RUNTIME_GENERATION")
    if generation:
        from hermes_cli.kanban_runtime_generation import generation_manifest, _is_stdlib
        generation_root = Path(generation).resolve()
        sealed = RuntimeIdentity.from_value(generation_manifest(generation, verify=True)["identity"])
        if not same_code_identity(sealed, observed):
            raise RuntimeIdentityError("runtime generation identity mismatch")
        for module in tuple(sys.modules.values()):
            location = getattr(module, "__file__", None)
            if not location or str(location).startswith("<"):
                continue
            path = Path(location).resolve()
            if not path.is_relative_to(generation_root) and not _is_stdlib(path):
                raise RuntimeIdentityError(f"mutable runtime import: {location}")
        return observed
    prefixes = _IDENTITY_ROOTS
    root_modules = {path.stem for path in root.glob("*.py") if path.is_file()}
    for name, module in tuple(sys.modules.items()):
        if name not in root_modules and not any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes):
            continue
        location = getattr(module, "__file__", None)
        if location and not Path(location).resolve().is_relative_to(root):
            raise RuntimeIdentityError(f"mixed runtime import roots: {name}={location}")
    return observed

def prospective_identity(module_root: Optional[os.PathLike[str] | str] = None) -> RuntimeIdentity:
    """Return the frozen code identity with the caller's live process marker."""
    if module_root is None:
        frozen = runtime_identity()
        return replace(frozen, pid=os.getpid(), start_time=process_start_time())
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
    if not expected.generation or not os.environ.get("HERMES_KANBAN_RUNTIME_GENERATION"):
        raise RuntimeIdentityError("worker bootstrap requires a prepared runtime generation")
    actual = assert_runtime_import_root(expected=expected)
    payload = {
        "ready": same_code_identity(expected, actual),
        "phase": "pre_import",
        "preparation_id": preparation_id,
        "runtime_identity": actual.as_dict(),
    }
    _atomic_write(Path(path_raw), payload)
    if not payload["ready"]:
        raise RuntimeIdentityError("worker runtime identity mismatch")
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
    generation_root = os.environ.get("HERMES_KANBAN_RUNTIME_GENERATION")
    if not generation_root or not expected.generation:
        raise RuntimeIdentityError("worker bootstrap requires a prepared runtime generation")
    actual = assert_runtime_import_root(expected=expected)
    payload = {
        "ready": True,
        "phase": "post_import",
        "post_import": True,
        "preparation_id": preparation_id,
        "runtime_identity": actual.as_dict(),
        "runtime_generation": generation_root,
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
    from hermes_cli.kanban_runtime_generation import sweep_runtime_generations
    sweep_runtime_generations()

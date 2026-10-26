"""The one verification executor: rootless Podman, ordinary containers, private stack.

Every engine argv is constructed here from the frozen dispatcher grant and validated
policy.  A recipe's payload is repository code, so it never runs until the boundary is
attested: a private network namespace keeper, fresh per-call engine storage, copied
source/workspace/scratch binds, synthetic identity and resolver files, dropped
capabilities, a read-only root, disabled cgroups, and a per-container conmon deadline.
Payload containers wait behind an executor-owned release marker; the marker is created
only after every container's image, argv, environment, mounts, namespace, ports, security
flags and processes have been checked.

Cleanup is owned by an on-disk journal so a dead worker can be reconciled by the
dispatcher: exact container IDs, conmon tuples and namespace inodes are recorded before
release, and nothing outside the journal's own private engine store is ever touched.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

PODMAN = "/usr/bin/podman"

#: Pinned runtime dependency for the namespace keeper (never selected by a recipe).
KEEPER_IMAGE = "docker.io/library/python@sha256:7dff173021016c8829ae58d5ce112ffc388073ebe67f351aeb6ba2909e35f847"
KEEPER_ENTRYPOINT = "/usr/local/bin/python3"

#: `--cap-drop=ALL` as expanded by rootless Podman for the container's user; the
#: boundary is only attested when exactly these remain in `CapDrop` and `CapAdd` is empty.
DEFAULT_ROOTLESS_CAPS = frozenset({
    "CAP_CHOWN", "CAP_DAC_OVERRIDE", "CAP_FOWNER", "CAP_FSETID", "CAP_KILL",
    "CAP_NET_BIND_SERVICE", "CAP_SETFCAP", "CAP_SETGID", "CAP_SETPCAP",
    "CAP_SETUID", "CAP_SYS_CHROOT",
})

VISIBLE_OUTPUT_BYTES = 64 * 1024
LOG_CAP_BYTES = 1024 * 1024
STARTUP_DEADLINE_SECONDS = 60.0
READINESS_PER_EXEC_SECONDS = 5.0
CONTAINER_TIMER_GRACE_SECONDS = 30.0  # create/start latency admitted before a payload timer
CLEANUP_GRACE_SECONDS = 10.0
JOURNAL_PROTOCOL = 1
MAX_REAP_JOURNALS = 64
#: Admission needs room for a private engine store and one image load; the running
#: check only cancels when the owned filesystem is about to run dry. Neither is an
#: aggregate disk quota.
ADMISSION_FREE_BYTES = 1024 * 1024 * 1024
MONITOR_FREE_BYTES = 256 * 1024 * 1024

_CONTAINER_ID_RE = re.compile(r"^[0-9a-f]{64}$")
_TAG_PREFIX = "localhost/worker-execution:"


class VerificationFailure(RuntimeError):
    """The boundary could not be attested or the engine refused; never a pass."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _record(record: Any, key: str, default: Any = None) -> Any:
    if isinstance(record, Mapping):
        return record.get(key, default)
    for entry in record or ():
        if isinstance(entry, (tuple, list)) and len(entry) == 2 and entry[0] == key:
            return entry[1]
    return default


def _bounded(text: Any, limit: int = 300) -> str:
    return " ".join(str(text).split())[:limit]


def remove_tree(path: Path) -> bool:
    """Remove one executor-owned directory tree, including image-layer ownership.

    ``vfs`` extracts image layers with their in-image ownership, which rootless Podman
    maps to subordinate UIDs on disk; a plain ``rmtree`` then cannot unlink them. The
    fallback runs in the same user namespace, so it removes exactly this tree without
    touching anything outside it.
    """
    for attempt in range(5):
        shutil.rmtree(path, ignore_errors=True)
        if not path.exists():
            return True
        try:
            result = subprocess.run(  # noqa: S603 -- fixed executable, owned path
                [PODMAN, "unshare", "rm", "-rf", str(path)], stdin=subprocess.DEVNULL,
                capture_output=True, timeout=120.0,
            )
        except (OSError, subprocess.SubprocessError):
            result = None
        if result is not None and result.returncode == 0 and not path.exists():
            return True
        # A shared rootless helper can hold a mount reference for a moment after the
        # last container is removed; retry briefly, then report the failure.
        time.sleep(0.5 * (attempt + 1))
    return not path.exists()


def _sha256_file(path: Path, *, limit: int = 8 * 1024 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    read = 0
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            read += len(chunk)
            if read > limit:
                raise VerificationFailure("image_archive_too_large", "image archive exceeds the verification bound")
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _free_bytes(path: Path) -> int:
    statvfs = os.statvfs(path)
    return int(statvfs.f_bavail) * int(statvfs.f_frsize)


def _require_space(path: Path) -> int:
    free = _free_bytes(path)
    if free < MONITOR_FREE_BYTES:
        raise VerificationFailure("insufficient_disk", "the task-owned filesystem is low on space")
    return free


def _process_start_time(pid: int) -> Optional[int]:
    try:
        text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8", errors="strict")
    except OSError:
        return None
    tail = text.rsplit(")", 1)[1].split()
    with contextlib.suppress(IndexError, ValueError):
        return int(tail[19])
    return None


def _process_state(pid: int) -> Optional[str]:
    try:
        text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8", errors="strict")
    except OSError:
        return None
    with contextlib.suppress(IndexError):
        return text.rsplit(")", 1)[1].split()[0]
    return None


def _process_identity(pid: Any, start: Any) -> str:
    """``match``, ``gone`` or ``unknown`` for one recorded process identity."""
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return "gone"
    if isinstance(start, bool) or not isinstance(start, int):
        return "unknown"
    if not Path(f"/proc/{pid}").exists():
        return "gone"
    observed = _process_start_time(pid)
    if observed is None:
        return "unknown"
    if observed != start:
        return "gone"
    state = _process_state(pid)
    if state is None:
        return "unknown"
    return "gone" if state in {"Z", "X", "x"} else "match"


def _process_running(pid: Any, start: Any) -> bool:
    """True only for a live (non-zombie) process whose start time matches *start*.

    A killed container init or conmon can linger as a zombie until its parent reaps
    it; a zombie executes nothing, holds no namespace and must not fail teardown. An
    unreadable process entry is NOT evidence of absence: it counts as running.
    """
    identity = _process_identity(pid, start)
    return identity != "gone"  # an unreadable identity counts as alive


def _process_cgroup(pid: int) -> Optional[str]:
    try:
        for line in Path(f"/proc/{pid}/cgroup").read_text(encoding="utf-8").splitlines():
            if line.startswith("0::"):
                return line[3:].strip()
    except OSError:
        return None
    return None


def _namespace_inode(pid: int, name: str) -> Optional[int]:
    try:
        return int(os.stat(f"/proc/{pid}/ns/{name}").st_ino)
    except OSError:
        return None


def _wait_status(pid: int) -> tuple[Optional[str], Optional[str]]:
    try:
        text = Path(f"/proc/{pid}/status").read_text(encoding="utf-8")
    except OSError:
        return None, None
    state = name = None
    for line in text.splitlines():
        if line.startswith("State:"):
            state = line.split()[1]
        elif line.startswith("Name:"):
            name = line.split(":", 1)[1].strip()
    return state, name


def _cmdline(pid: int) -> str:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return ""
    return raw.replace(b"\x00", b" ").decode("utf-8", "replace").strip()


class BoundedCollector:
    """A 64 KiB visible window over container output that may be far larger."""

    def __init__(self, limit: int = VISIBLE_OUTPUT_BYTES) -> None:
        self.limit = limit
        self.total = 0
        self.truncated = False
        self._window = bytearray()

    def feed(self, chunk: bytes) -> None:
        if not chunk:
            return
        self.total += len(chunk)
        self._window.extend(chunk)
        if len(self._window) > self.limit:
            self.truncated = True
            del self._window[: len(self._window) - self.limit]

    def text(self) -> str:
        return bytes(self._window).decode("utf-8", "replace")


class Engine:
    """One invocation's private engine store and fixed command prefix."""

    def __init__(self, run_root: Path) -> None:
        self.run_root = run_root
        self.worker_cgroup = ""
        self.storage = run_root / "storage"
        self.runroot = run_root / "runroot"
        self.tmpdir = run_root / "engine-tmp"
        self.hooks = run_root / "empty-hooks"
        self.home = run_root / "home"
        self.config = run_root / "config"
        self.cache = run_root / "cache"
        self.auth = run_root / "auth.json"
        self.logs = run_root / "logs"

    def prepare(self) -> None:
        for directory in (self.storage, self.runroot, self.tmpdir, self.hooks, self.home,
                          self.config, self.cache, self.logs):
            directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            os.chmod(directory, 0o700)
            if directory.is_symlink():
                raise VerificationFailure("engine_store_insecure", "engine directory must not be a symlink")
        self.auth.write_text("{}\n", encoding="utf-8")
        os.chmod(self.auth, 0o600)
        # Private engine configuration: nothing here reads the user's containers config.
        (self.config / "containers.conf").write_text(
            "[engine]\nlabel = false\n", encoding="utf-8",
        )
        (self.config / "storage.conf").write_text(
            "[storage]\n"
            f'driver = "vfs"\n'
            f'graphroot = "{self.storage}"\n'
            f'runroot = "{self.runroot}"\n',
            encoding="utf-8",
        )
        (self.config / "registries.conf").write_text(
            "[registries.insecure]\nregistries = []\n", encoding="utf-8",
        )
        for name in ("containers.conf", "storage.conf", "registries.conf"):
            os.chmod(self.config / name, 0o600)

    @property
    def prefix(self) -> list[str]:
        return [
            PODMAN,
            "--root", str(self.storage),
            "--runroot", str(self.runroot),
            "--tmpdir", str(self.tmpdir),
            "--storage-driver", "vfs",
            "--cgroup-manager", "cgroupfs",
            "--events-backend", "none",
            "--hooks-dir", str(self.hooks),
        ]

    def env(self) -> dict[str, str]:
        uid = os.getuid() if hasattr(os, "getuid") else -1  # windows-footgun: ok — hasattr-guarded; rootless Podman is POSIX-only
        runtime_dir = os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{uid}"
        return {
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "HOME": str(self.home),
            "XDG_CONFIG_HOME": str(self.config),
            "XDG_CACHE_HOME": str(self.cache),
            "XDG_RUNTIME_DIR": runtime_dir,
            "TMPDIR": str(self.tmpdir),
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "REGISTRY_AUTH_FILE": str(self.auth),
            "CONTAINERS_CONF": str(self.config / "containers.conf"),
            "CONTAINERS_STORAGE_CONF": str(self.config / "storage.conf"),
            "CONTAINERS_REGISTRIES_CONF": str(self.config / "registries.conf"),
            "HTTP_PROXY": "", "HTTPS_PROXY": "", "NO_PROXY": "",
        }

    def run(
        self,
        args: list[str],
        *,
        timeout: float = 60.0,
        check: bool = False,
        deadline: Optional[float] = None,
        supervise: Optional[Callable[[], None]] = None,
        supervise_after: bool = True,
    ) -> subprocess.CompletedProcess:
        """Run one engine command under live identity/placement supervision.

        The child is started with ``Popen`` and its PID start-time and cgroup are read
        before waiting: an engine command that is not in the authenticated worker's
        cgroup is killed and refused, never trusted. While it runs, the absolute
        invocation deadline and the caller's supervision callback (claim ownership,
        free space) are polled, so a slow load cannot outlive its authority.
        """
        if supervise is not None:
            supervise()
        try:
            process = subprocess.Popen(  # noqa: S603 -- fixed executable and constructed argv
                [*self.prefix, *args], cwd=str(self.run_root), env=self.env(),
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise VerificationFailure("engine_unavailable", _bounded(exc))
        start = _process_start_time(process.pid)
        cgroup = _process_cgroup(process.pid)
        if start is None or cgroup is None or not _cgroup_contained(cgroup, self.worker_cgroup):
            with contextlib.suppress(Exception):
                process.kill()
                process.wait(timeout=5)
            raise VerificationFailure("attestation_failed", "engine command is outside the authenticated worker cgroup")
        command_deadline = time.monotonic() + max(0.1, timeout)

        def _stop(reason: str) -> None:
            with contextlib.suppress(Exception):
                process.kill()
                process.wait(timeout=5)
            raise VerificationFailure(
                reason, f"engine command stopped: {reason} ({' '.join(args[:3])})",
            )

        stdout = stderr = b""
        while True:
            now = time.monotonic()
            limits = [command_deadline]
            if deadline is not None:
                limits.append(deadline)
            remaining = min(limits) - now
            if remaining <= 0:
                _stop("engine_timeout" if now >= command_deadline else "deadline_exhausted")
            try:
                stdout, stderr = process.communicate(timeout=min(0.25, remaining))
                break
            except subprocess.TimeoutExpired:
                if supervise is not None:
                    try:
                        supervise()
                    except BaseException:
                        with contextlib.suppress(Exception):
                            process.kill()
                            process.wait(timeout=5)
                        raise
        # A command that finished inside the poll interval still must not carry work
        # past cancellation: re-check supervision before returning it, unless the caller
        # must first take ownership of the created resource (_create owns its id first).
        if supervise is not None and supervise_after:
            supervise()
        result = subprocess.CompletedProcess(
            [*self.prefix, *args], process.returncode, stdout, stderr,
        )
        if check and result.returncode != 0:
            raise VerificationFailure(
                "engine_failed", _bounded(stderr.decode("utf-8", "replace")) or "engine command failed",
            )
        return result

    def text(
        self,
        args: list[str],
        *,
        timeout: float = 60.0,
        deadline: Optional[float] = None,
        supervise: Optional[Callable[[], None]] = None,
    ) -> str:
        result = self.run(args, timeout=timeout, check=True, deadline=deadline, supervise=supervise)
        return result.stdout.decode("utf-8", "replace").strip()


def _verify_local_rootless(
    engine: Engine, *, deadline: Optional[float] = None, supervise: Optional[Callable[[], None]] = None,
) -> None:
    result = engine.run(
        ["info", "--format", "{{.Host.Security.Rootless}}"], timeout=60.0,
        deadline=deadline, supervise=supervise,
    )
    if result.returncode != 0:
        raise VerificationFailure("engine_unavailable", "podman is not usable")
    if result.stdout.strip().decode("utf-8", "replace").lower() != "true":
        raise VerificationFailure("engine_not_rootless", "verification requires rootless Podman")


class ImageStore:
    """The grant's persistent, owner-only archive store (never the ordinary engine store)."""

    def __init__(self, authority: Any) -> None:
        record = authority.image_store
        if record is None:
            raise VerificationFailure("image_store_unavailable", "no image store is bound to this grant")
        self.path = Path(str(_record(record, "path")))
        self.expected_inventory_digest = _record(record, "inventory_digest")
        if self.path.is_symlink() or not self.path.is_dir():
            raise VerificationFailure("image_store_unavailable", "image store is not a real directory")
        info = os.lstat(self.path)
        if int(info.st_dev) != int(_record(record, "device", -1)) or int(info.st_ino) != int(_record(record, "inode", -1)):
            raise VerificationFailure("image_store_changed", "image store identity changed")
        if info.st_mode & 0o077:
            raise VerificationFailure("image_store_insecure", "image store must be owner-only")
        self.inventory = self._load_inventory()

    def _load_inventory(self) -> dict[str, Any]:
        path = self.path / "inventory.json"
        try:
            info = os.lstat(path)
            if not stat.S_ISREG(info.st_mode) or path.is_symlink() or info.st_size > 1024 * 1024:
                raise VerificationFailure("image_store_insecure", "image store inventory is not a bounded regular file")
            raw = path.read_bytes()
        except OSError as exc:
            raise VerificationFailure("image_store_unavailable", _bounded(exc))
        digest = "sha256:" + hashlib.sha256(raw).hexdigest()
        if self.expected_inventory_digest and digest != self.expected_inventory_digest:
            raise VerificationFailure("image_store_changed", "image store inventory digest changed")
        try:
            inventory = json.loads(raw)
        except ValueError:
            raise VerificationFailure("image_store_malformed", "image store inventory is not JSON")
        if not isinstance(inventory, Mapping) or not isinstance(inventory.get("images"), Mapping):
            raise VerificationFailure("image_store_malformed", "image store inventory is malformed")
        return dict(inventory)

    def entry(self, reference: str) -> dict[str, Any]:
        record = self.inventory["images"].get(reference)
        if not isinstance(record, Mapping):
            raise VerificationFailure("image_not_provisioned", "the required image is not provisioned locally")
        archive = str(record.get("archive") or "")
        tag = str(record.get("tag") or "")
        image_id = str(record.get("image_id") or "")
        digest = str(record.get("sha256") or "")
        if not archive or "/" in archive or not re.match(r"^[0-9a-f]{64}\.oci\.tar$", archive):
            raise VerificationFailure("image_store_malformed", "archive name is malformed")
        if not tag.startswith(_TAG_PREFIX) or not re.match(r"^[a-z0-9._-]{1,64}$", tag[len(_TAG_PREFIX):]):
            raise VerificationFailure("image_store_malformed", "archive tag is malformed")
        if not re.match(r"^[0-9a-f]{64}$", image_id):
            raise VerificationFailure("image_store_malformed", "archive image id is malformed")
        if not re.match(r"^sha256:[0-9a-f]{64}$", digest):
            raise VerificationFailure("image_store_malformed", "archive digest is malformed")
        return {"archive": self.path / archive, "tag": tag, "image_id": image_id, "sha256": digest}


def _load_images(
    engine: Engine,
    store: ImageStore,
    references: list[str],
    *,
    deadline: Optional[float] = None,
    claim_check: Optional[Callable[[], None]] = None,
    space_check: Optional[Callable[[], int]] = None,
) -> dict[str, dict[str, Any]]:
    loaded: dict[str, dict[str, Any]] = {}
    for reference in references:
        if reference in loaded:
            continue
        entry = store.entry(reference)
        archive = entry["archive"]
        try:
            info = os.lstat(archive)
        except OSError as exc:
            raise VerificationFailure("image_store_unavailable", _bounded(exc))
        if not stat.S_ISREG(info.st_mode) or archive.is_symlink():
            raise VerificationFailure("image_store_insecure", "image archive must be a regular file")
        if _sha256_file(archive) != entry["sha256"]:
            raise VerificationFailure("image_archive_changed", "image archive bytes do not match the inventory")
        loaded[reference] = entry
    def _supervise() -> None:
        if claim_check is not None:
            claim_check()
        if space_check is not None:
            space_check()

    for reference, entry in loaded.items():
        _supervise()
        timeout = 600.0
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise VerificationFailure("deadline_exhausted", "the verification deadline elapsed while loading images")
            timeout = max(1.0, min(timeout, remaining))
        result = engine.run(
            ["load", "-i", str(entry["archive"])], timeout=timeout,
            deadline=deadline, supervise=_supervise,
        )
        if result.returncode != 0:
            raise VerificationFailure(
                "image_load_failed", _bounded(result.stderr.decode("utf-8", "replace")) or "image load failed",
            )
    for reference, entry in loaded.items():
        image_id = engine.text(
            ["image", "inspect", "--format", "{{.Id}}", entry["tag"]], timeout=60.0,
            deadline=deadline, supervise=_supervise,
        )
        if image_id != entry["image_id"].lower():
            raise VerificationFailure("image_identity_mismatch", "imported image identity does not match the inventory")
    return loaded


def _run_root_record(path: Path) -> dict[str, Any]:
    info = os.lstat(path)
    return {"path": str(path), "device": int(info.st_dev), "inode": int(info.st_ino)}


def _journal_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    try:
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _journal_read(path: Path) -> Optional[dict[str, Any]]:
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    if len(raw) > 1024 * 1024:
        return None
    try:
        payload = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(payload, dict) or payload.get("protocol") != JOURNAL_PROTOCOL:
        return None
    return payload


def _write_synthetic_identity(run_root: Path) -> dict[str, Path]:
    files = {
        "passwd": run_root / "etc-passwd",
        "group": run_root / "etc-group",
        "hosts": run_root / "etc-hosts",
        "resolv": run_root / "etc-resolv.conf",
    }
    files["passwd"].write_text(
        "root:x:0:0:root:/root:/bin/sh\nworker:x:1000:1000:worker:/scratch/home:/bin/sh\n",
        encoding="utf-8",
    )
    files["group"].write_text("root:x:0:\nworker:x:1000:\n", encoding="utf-8")
    files["hosts"].write_text("127.0.0.1 localhost\n::1 localhost\n", encoding="utf-8")
    files["resolv"].write_text("", encoding="utf-8")
    for path in files.values():
        os.chmod(path, 0o444)
    return files


class Container:
    def __init__(self, name: str, kind: str, image_reference: str, image_tag: str) -> None:
        self.name = name
        self.kind = kind
        self.image_reference = image_reference
        self.image_tag = image_tag
        self.id: Optional[str] = None
        self.pid: Optional[int] = None
        self.pid_start: Optional[int] = None
        self.conmon_pid: Optional[int] = None
        self.conmon_start: Optional[int] = None
        self.net_inode: Optional[int] = None
        self.pid_inode: Optional[int] = None
        self.conmon_path: Optional[Path] = None
        self.expiry_epoch: Optional[float] = None
        self.exit_code: Optional[int] = None
        self.timed_out = False
        self.log_path: Optional[Path] = None

    def record(self) -> dict[str, Any]:
        return {
            "name": self.name, "kind": self.kind, "id": self.id,
            "pid": self.pid, "pid_start": self.pid_start,
            "conmon_pid": self.conmon_pid, "conmon_start": self.conmon_start,
            "conmon_path": str(self.conmon_path) if self.conmon_path else None,
            "net_inode": self.net_inode, "pid_inode": self.pid_inode,
            "expiry_epoch": self.expiry_epoch,
            "log_path": str(self.log_path) if self.log_path else None,
        }


class _Spec:
    """One container's fixed, executor-generated creation arguments."""

    def __init__(
        self,
        *,
        name: str,
        kind: str,
        image_reference: str,
        image_tag: str,
        entrypoint: list[str],
        command: list[str],
        env: dict[str, str],
        mounts: list[tuple[Path, str, bool]],
        network: str,
        timeout_seconds: int,
        log_path: Path,
        user: str = "1000:1000",
        ulimits: tuple[str, ...] = (),
    ) -> None:
        self.name = name
        self.kind = kind
        self.image_reference = image_reference
        self.image_tag = image_tag
        self.entrypoint = entrypoint
        self.command = command
        self.env = env
        self.mounts = mounts
        self.network = network
        self.timeout_seconds = timeout_seconds
        self.log_path = log_path
        self.user = user
        self.ulimits = ulimits


def _create_args(engine: Engine, spec: _Spec) -> list[str]:
    args = [
        "create", "--name", spec.name,
        "--network", spec.network,
        "--pull=never",
        "--cgroups=disabled",
        "--read-only",
        "--read-only-tmpfs=false",
        "--restart=no",
        "--no-healthcheck",
        "--image-volume=ignore",
        "--http-proxy=false",
        "--unsetenv-all",
        "--userns=keep-id:uid=1000,gid=1000",
        "--pid=private",
        "--ipc=private",
        "--uts=private",
        "--no-hosts",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        f"--user={spec.user}",
        "--stop-signal=SIGKILL",
        "--stop-timeout=2",
        f"--timeout={spec.timeout_seconds}",
        f"--conmon-pidfile={engine.run_root / (spec.name + '.conmon.pid')}",
        f"--cidfile={engine.run_root / (spec.name + '.cid')}",
        "--ulimit=nofile=1024:1024",
        "--ulimit=core=0:0",
        "--ulimit=fsize=268435456:268435456",
        "--log-driver=k8s-file",
        f"--log-opt=path={spec.log_path}",
        "--log-opt=max-size=1mb",
        "--entrypoint", spec.entrypoint[0],
    ]
    if len(spec.entrypoint) > 1:
        args.extend(spec.entrypoint[1:])
    for key, value in sorted(spec.env.items()):
        args.extend(["--env", f"{key}={value}"])
    for source, destination, writable in spec.mounts:
        mode = "rw" if writable else "ro"
        args.extend(["-v", f"{source}:{destination}:{mode},z"])
    args.append(spec.image_tag)
    args.extend(spec.command)
    return args


def _gate_command(argv: list[str]) -> list[str]:
    return [
        "-c",
        "until [ -e /scratch/release ]; do sleep 0.2; done; exec \"$@\"",
        "worker-execution-gate",
        *argv,
    ]


def _inspect_json(
    engine: Engine,
    container_id: str,
    *,
    deadline: Optional[float] = None,
    supervise: Optional[Callable[[], None]] = None,
) -> dict[str, Any]:
    try:
        raw = engine.text(
            ["inspect", container_id], timeout=30.0, deadline=deadline, supervise=supervise,
        )
    except VerificationFailure:
        raise VerificationFailure("attestation_failed", "container could not be inspected")
    try:
        payload = json.loads(raw)
    except ValueError:
        raise VerificationFailure("attestation_failed", "container inspect output is not JSON")
    if not isinstance(payload, list) or len(payload) != 1 or not isinstance(payload[0], dict):
        raise VerificationFailure("attestation_failed", "container inspect output is malformed")
    return payload[0]


def _attest(
    engine: Engine,
    container: Container,
    spec: _Spec,
    *,
    image_id: str,
    keeper_id: Optional[str],
    keeper_net_inode: Optional[int],
    host_pid_inode: Optional[int],
    worker_cgroup: str,
    expect_waiting_gate: bool,
    deadline: Optional[float] = None,
    supervise: Optional[Callable[[], None]] = None,
) -> None:
    data = _inspect_json(
        engine, container.id or "", deadline=deadline, supervise=supervise,
    )
    if str(data.get("Id", "")).lower() != (container.id or "").lower():
        raise VerificationFailure("attestation_failed", "container id does not match the created id")
    state = data.get("State") or {}
    if state.get("Status") != "running":
        raise VerificationFailure("attestation_failed", "container is not running")
    if str(data.get("Image", "")).lower() != image_id.lower():
        raise VerificationFailure("attestation_failed", "container image does not match the provisioned image")
    config = data.get("Config") or {}
    host = data.get("HostConfig") or {}
    if list(config.get("Entrypoint") or []) != list(spec.entrypoint):
        raise VerificationFailure("attestation_failed", "container entrypoint changed")
    if list(config.get("Cmd") or []) != list(spec.command):
        raise VerificationFailure("attestation_failed", "container command changed")
    if str(config.get("User", "")) != spec.user:
        raise VerificationFailure("attestation_failed", "container user changed")
    if bool(config.get("Tty")) or bool(config.get("OpenStdin")):
        raise VerificationFailure("attestation_failed", "container has a tty or open stdin")
    expected_env = {f"{key}={value}" for key, value in spec.env.items()}
    actual_env = set(config.get("Env") or [])
    hostname = f"HOSTNAME={(container.id or '')[:12]}"
    if actual_env - expected_env - {hostname}:
        raise VerificationFailure("attestation_failed", "container environment contains unexpected values")
    if not expected_env <= actual_env:
        raise VerificationFailure("attestation_failed", "container environment is missing generated values")
    healthcheck = config.get("Healthcheck") or {}
    if list(healthcheck.get("Test") or []) != ["NONE"]:
        raise VerificationFailure("attestation_failed", "container healthcheck is not disabled")
    if config.get("StopSignal") != "SIGKILL":
        raise VerificationFailure("attestation_failed", "container stop signal changed")
    if host.get("ReadonlyRootfs") is not True:
        raise VerificationFailure("attestation_failed", "container root filesystem is writable")
    if host.get("CapAdd"):
        raise VerificationFailure("attestation_failed", "container adds capabilities")
    if set(host.get("CapDrop") or []) != set(DEFAULT_ROOTLESS_CAPS):
        raise VerificationFailure("attestation_failed", "container capability drop does not match the boundary")
    if "no-new-privileges" not in (host.get("SecurityOpt") or []):
        raise VerificationFailure("attestation_failed", "container does not disable new privileges")
    if host.get("PortBindings"):
        raise VerificationFailure("attestation_failed", "container publishes ports")
    if bool(host.get("Privileged")) or bool(host.get("PublishAllPorts")):
        raise VerificationFailure("attestation_failed", "container is privileged or publishes ports")
    if (host.get("RestartPolicy") or {}).get("Name") not in (None, "", "no"):
        raise VerificationFailure("attestation_failed", "container restart policy changed")
    if host.get("AutoRemove"):
        raise VerificationFailure("attestation_failed", "container would auto-remove evidence")
    log_config = host.get("LogConfig") or {}
    if log_config.get("Type") != "k8s-file":
        raise VerificationFailure("attestation_failed", "container log driver changed")
    observed_log = str(log_config.get("Path") or "")
    if spec.log_path and observed_log and Path(observed_log).resolve() != spec.log_path.resolve():
        raise VerificationFailure("attestation_failed", "container log path changed")
    for ulimit in host.get("Ulimits") or []:
        if ulimit.get("Name") == "RLIMIT_NOFILE" and (ulimit.get("Soft"), ulimit.get("Hard")) != (1024, 1024):
            raise VerificationFailure("attestation_failed", "container file descriptor limit changed")
    expected_mounts = {(str(source.resolve()), destination, writable) for source, destination, writable in spec.mounts}
    actual_mounts = {
        (str(Path(str(mount.get("Source"))).resolve()), str(mount.get("Destination")), bool(mount.get("RW")))
        for mount in data.get("Mounts") or []
    }
    if actual_mounts != expected_mounts:
        raise VerificationFailure("attestation_failed", "container mounts do not match the generated set")
    network = data.get("NetworkSettings") or {}
    expected_network = "none" if keeper_id is None else f"container:{keeper_id}"
    if str(host.get("NetworkMode")) != expected_network:
        raise VerificationFailure("attestation_failed", "container network mode changed")
    sandbox = str(network.get("SandboxKey") or "")
    pid = int(state.get("Pid") or 0)
    if pid <= 0:
        raise VerificationFailure("attestation_failed", "container has no live process")
    container.pid = pid
    container.pid_start = _process_start_time(pid)
    container.conmon_pid = int(state.get("ConmonPid") or 0) or None
    if not container.conmon_pid:
        raise VerificationFailure("attestation_failed", "container has no conmon process")
    container.conmon_start = _process_start_time(container.conmon_pid)
    if container.conmon_start is None:
        raise VerificationFailure("attestation_failed", "conmon process identity is unreadable")
    container.conmon_path = engine.run_root / (spec.name + ".conmon.pid")
    try:
        recorded = int(container.conmon_path.read_bytes().decode("utf-8").strip())
    except (OSError, ValueError):
        raise VerificationFailure("attestation_failed", "conmon pidfile is missing or malformed")
    if recorded != container.conmon_pid:
        raise VerificationFailure("attestation_failed", "conmon pidfile names another process")
    container.net_inode = _namespace_inode(pid, "net")
    container.pid_inode = _namespace_inode(pid, "pid")
    if container.pid_inode is None:
        raise VerificationFailure("attestation_failed", "container PID namespace is unreadable")
    if host_pid_inode is not None and container.pid_inode == host_pid_inode:
        raise VerificationFailure("attestation_failed", "container shares the host PID namespace")
    if container.pid_start is None or container.net_inode is None:
        raise VerificationFailure("attestation_failed", "container process identity is unreadable")
    if not sandbox:
        raise VerificationFailure("attestation_failed", "container has no network sandbox")
    try:
        sandbox_inode = os.stat(sandbox).st_ino
    except OSError:
        raise VerificationFailure("attestation_failed", "container network sandbox is unreadable")
    if int(sandbox_inode) != int(container.net_inode):
        raise VerificationFailure("attestation_failed", "container sandbox key does not name its own namespace")
    if keeper_net_inode is not None and container.net_inode != keeper_net_inode:
        raise VerificationFailure("attestation_failed", "container does not share the keeper network namespace")
    if keeper_net_inode is None:
        # The keeper's own namespace must differ from the host's.
        if container.net_inode == _namespace_inode(1, "net"):
            raise VerificationFailure("attestation_failed", "keeper does not own a private network namespace")
        if container.net_inode == _namespace_inode(os.getpid(), "net"):
            raise VerificationFailure("attestation_failed", "keeper shares the host network namespace")
    cgroup = _process_cgroup(pid)
    if cgroup is None or not _cgroup_contained(cgroup, worker_cgroup):
        raise VerificationFailure("attestation_failed", "container process is outside the authenticated worker cgroup")
    conmon_cgroup = _process_cgroup(container.conmon_pid) if container.conmon_pid else None
    if container.conmon_pid and (conmon_cgroup is None or not _cgroup_contained(conmon_cgroup, worker_cgroup)):
        raise VerificationFailure("attestation_failed", "conmon is outside the authenticated worker cgroup")
    if expect_waiting_gate:
        current = _cmdline(pid)
        if "worker-execution-gate" not in current or "exec" not in current:
            raise VerificationFailure("attestation_failed", "payload was released before attestation")


def _cgroup_contained(path: str, parent: str) -> bool:
    if path == parent:
        return True
    return path.startswith(parent.rstrip("/") + "/")


def _container_env(extra: Optional[dict[str, str]] = None) -> dict[str, str]:
    env = {
        "HOME": "/scratch/home",
        "TMPDIR": "/scratch/tmp",
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "CI": "1",
    }
    if extra:
        env.update(extra)
    return env


def _service_env(services: list[Mapping[str, Any]]) -> dict[str, str]:
    extra: dict[str, str] = {}
    seen: set[str] = set()
    for service in services:
        name = str(service.get("id"))
        folded = name.upper().replace("-", "_")
        if folded in seen:
            raise VerificationFailure("policy_malformed", "service ids collide after normalization")
        seen.add(folded)
        extra[f"VERIFY_SERVICE_{folded}_HOST"] = "127.0.0.1"
        extra[f"VERIFY_SERVICE_{folded}_PORT"] = str(int(service.get("port")))
    return extra


def _mounts_for(
    *, source_dir: Path, workspace: Optional[Path], scratch: Path, data: Optional[Path],
    identity: dict[str, Path],
) -> list[tuple[Path, str, bool]]:
    mounts: list[tuple[Path, str, bool]] = [(source_dir, "/source", False)]
    if workspace is not None:
        mounts.append((workspace, "/workspace", True))
    mounts.append((scratch, "/scratch", True))
    if data is not None:
        mounts.append((data, "/data", True))
    mounts.extend([
        (identity["passwd"], "/etc/passwd", False),
        (identity["group"], "/etc/group", False),
        (identity["hosts"], "/etc/hosts", False),
        (identity["resolv"], "/etc/resolv.conf", False),
    ])
    return mounts


def _stage_workspace(run_root: Path, source_dir: Path) -> Path:
    workspace = run_root / "workspace"
    shutil.copytree(source_dir, workspace, symlinks=True, copy_function=shutil.copy2)
    for root, directories, files in os.walk(workspace):
        os.chmod(root, 0o755)
        for name in files:
            path = Path(root) / name
            if not path.is_symlink():
                with contextlib.suppress(OSError):
                    os.chmod(path, 0o755 if os.stat(path).st_mode & 0o111 else 0o644)
    return workspace


def _make_scratch(run_root: Path, name: str) -> Path:
    scratch = run_root / f"scratch-{name}"
    (scratch / "home").mkdir(mode=0o777, parents=True, exist_ok=True)
    (scratch / "tmp").mkdir(mode=0o777, parents=True, exist_ok=True)
    os.chmod(scratch, 0o700)
    os.chmod(scratch / "home", 0o777)
    os.chmod(scratch / "tmp", 0o777)
    return scratch


def _attest_expiry(
    container: Container, spec: "_Spec", deadline: float, *, keeper: bool = False,
) -> None:
    """Every container's own conmon deadline must fit the invocation deadline.

    Podman's ``--timeout`` counts running time from start, while the value was computed
    before ``create``; creation/start latency therefore pushes the real expiry later.
    Refuse release when that latency would let a payload outlive the invocation.
    """
    started = time.monotonic()
    expiry = started + int(spec.timeout_seconds)
    permitted = deadline + CONTAINER_TIMER_GRACE_SECONDS + (60.0 if keeper else 0.0)
    if expiry > permitted:
        raise VerificationFailure(
            "isolation_unavailable",
            f"{container.name} container deadline exceeds the invocation deadline",
        )
    container.expiry_epoch = float(started) + (time.time() - time.monotonic()) + int(spec.timeout_seconds)


def _start(
    engine: Engine,
    container: Container,
    *,
    deadline: Optional[float] = None,
    supervise: Optional[Callable[[], None]] = None,
) -> None:
    result = engine.run(
        ["start", container.id or ""], timeout=120.0, deadline=deadline, supervise=supervise,
    )
    if result.returncode != 0:
        raise VerificationFailure("engine_failed", _bounded(result.stderr.decode("utf-8", "replace")) or "container start failed")


def _wait_ready(
    engine: Engine,
    services: list[tuple[Mapping[str, Any], Container]],
    *,
    deadline: float,
    claim_check: Optional[Callable[[], None]],
    startup_deadline: float,
    space_check: Optional[Callable[[], int]] = None,
) -> None:
    pending = list(services)
    while pending:
        now = time.monotonic()
        if now > min(deadline, startup_deadline):
            raise VerificationFailure("readiness_timeout", "declared services did not become ready")
        if claim_check is not None:
            claim_check()
        if space_check is not None:
            space_check()
        still: list[tuple[Mapping[str, Any], Container]] = []
        for service, container in pending:
            argv = [str(item) for item in service.get("ready_argv") or ()]
            now = time.monotonic()
            shared = min(deadline, startup_deadline)
            if shared - now <= 0:
                raise VerificationFailure("readiness_timeout", "declared services did not become ready")
            budget = max(0.5, min(READINESS_PER_EXEC_SECONDS, shared - now))
            try:
                result = engine.run(
                    ["exec", container.id or "", *argv], timeout=budget,
                    deadline=shared, supervise=lambda: claim_check() if claim_check else None,
                )
            except VerificationFailure as exc:
                # A slow engine exec is a not-ready probe, not a service failure: the
                # startup deadline still bounds the wait, and cancellation re-raises.
                if exc.code != "engine_timeout":
                    raise
                still.append((service, container))
                continue
            if result.returncode != 0:
                still.append((service, container))
        if not still:
            return
        pending = still
        time.sleep(1.0)


def _release(marker: Path) -> None:
    """Create the gate marker atomically with native interruption."""
    from tools.interrupt import run_if_not_interrupted

    def _create() -> None:
        fd = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(fd)

    if not run_if_not_interrupted(_create):
        raise VerificationFailure("cancelled", "verification was interrupted before release")


def _collect_log(engine: Engine, container: Container) -> tuple[str, bool]:
    collector = BoundedCollector()
    if container.id is None:
        return "", False
    result = engine.run(["logs", container.id], timeout=30.0)
    collector.feed(result.stdout)
    collector.feed(result.stderr)
    return collector.text(), collector.truncated


def _deadline_seconds(deadline: float) -> int:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise VerificationFailure("deadline_exhausted", "the verification deadline already elapsed")
    return max(1, int(remaining) + 1)


def _cgroup_limit(cgroup: str, name: str) -> Optional[int]:
    """Smallest finite *name* along the ancestor chain, or None when unreadable/unlimited.

    The effective ceiling is the minimum along the tree, not the first value found: a
    child may be unlimited while its slice is capped.
    """
    current = cgroup.strip("/")
    parts = current.split("/") if current else []
    observed: list[int] = []
    while True:
        path = Path("/sys/fs/cgroup") / ("/".join(parts) if parts else "") / name
        try:
            text = path.read_bytes().decode("utf-8").strip()
        except OSError:
            text = ""
        if text and text != "max":
            try:
                observed.append(int(text))
            except ValueError:
                return None
        if not parts:
            break
        parts.pop()
    return min(observed) if observed else None


def _inherited_limits(worker_cgroup: str) -> dict[str, int]:
    """The finite ancestor limits this worker's containers inherit.

    ``--cgroups=disabled`` puts payload processes in the caller's cgroup, so the only
    aggregate ceilings are the ones already on that chain. A host without a finite
    process bound must refuse instead of trusting per-UID ``nproc``.
    """
    memory = _cgroup_limit(worker_cgroup, "memory.max")
    pids = _cgroup_limit(worker_cgroup, "pids.max")
    if memory is None or pids is None:
        raise VerificationFailure(
            "isolation_unavailable",
            "the worker cgroup has no finite inherited memory/process limit",
        )
    return {"memory_max_bytes": memory, "pids_max": pids}


def _claim_check(grant: Any) -> Callable[[], None]:
    from hermes_cli import worker_execution

    last = [0.0]

    def check() -> None:
        now = time.monotonic()
        if now - last[0] < 5.0:
            return
        last[0] = now
        try:
            worker_execution._reauthorize_live_claim(grant)
        except Exception as exc:
            raise VerificationFailure("claim_lost", _bounded(exc))

    return check


def run_verification(
    *,
    authority: Any,
    recipe: Mapping[str, Any],
    source_dir: Path,
    run_root: Path,
    journal_path: Path,
    grant: Any,
) -> dict[str, Any]:
    """Provision, attest, release, execute and tear down one granted recipe."""
    timeout_seconds = int(recipe.get("timeout_seconds") or 0)
    deadline = time.monotonic() + timeout_seconds
    started_epoch = time.time()
    deadline_epoch = started_epoch + timeout_seconds
    worker_cgroup = _process_cgroup(os.getpid()) or ""
    if not worker_cgroup:
        raise VerificationFailure("isolation_unavailable", "the worker cgroup is unreadable")
    engine = Engine(run_root)
    engine.worker_cgroup = worker_cgroup
    images: dict[str, dict[str, Any]] = {}

    claim_check = _claim_check(grant)
    from tools.interrupt import is_interrupted

    def supervise() -> None:
        if time.monotonic() >= deadline:
            raise VerificationFailure("deadline_exhausted", "the verification deadline elapsed")
        if is_interrupted():
            raise VerificationFailure("cancelled", "verification was interrupted")
        claim_check()
        _require_space(run_root)
        if time.monotonic() >= deadline:
            raise VerificationFailure("deadline_exhausted", "the verification deadline elapsed")

    journal: dict[str, Any] = {
        "protocol": JOURNAL_PROTOCOL,
        "invocation": run_root.name,
        "run_root": _run_root_record(run_root),
        "engine_root": str(engine.storage),
        "run_root_path": str(run_root),
        "worker_pid": os.getpid(),
        "worker_start": _process_start_time(os.getpid()),
        "worker_cgroup": worker_cgroup,
        "host_epoch": _host_epoch(),
        "deadline_epoch": time.time() + timeout_seconds,
        "phase": "provisioning",
        "recipe": str(recipe.get("id")),
        "containers": [],
        "creates": [],
    }
    _journal_write(journal_path, journal)

    keeper: Optional[Container] = None
    containers: list[Container] = []
    result: dict[str, Any] = {
        "ok": False,
        "accepted": False,
        "exit_code": None,
        "timed_out": False,
        "cleanup_ok": False,
        "stdout": "",
        "stderr": "",
        "output_combined": True,
        "truncated": False,
    }
    keeper_id: Optional[str] = None
    limits: dict[str, int] = {}
    host_pid_inode: Optional[int] = None
    try:
        # Everything that can allocate is inside the journal's protection: a failure
        # here still reaches the cleanup path, and the reaper can finish the job if
        # this process dies first.
        engine.prepare()
        _verify_local_rootless(engine, deadline=deadline, supervise=supervise)
        if _free_bytes(run_root) < ADMISSION_FREE_BYTES:
            raise VerificationFailure("insufficient_disk", "verification requires at least 1 GiB free")
        limits = _inherited_limits(worker_cgroup)
        host_pid_inode = _namespace_inode(os.getpid(), "pid")
        store = ImageStore(authority)
        services = list(recipe.get("services") or ())
        references = [KEEPER_IMAGE, str(recipe.get("image"))] + [str(service.get("image")) for service in services]
        images = _load_images(
            engine, store, references, deadline=deadline, claim_check=supervise,
        )
        _require_space(run_root)
        keeper = Container("keeper", "keeper", KEEPER_IMAGE, images[KEEPER_IMAGE]["tag"])
        identity = _write_synthetic_identity(run_root)
        workspace = _stage_workspace(run_root, Path(source_dir))
        keeper_scratch = _make_scratch(run_root, "keeper")
        keeper_seconds = _deadline_seconds(deadline) + int(CLEANUP_GRACE_SECONDS) + 30
        keeper_spec = _Spec(
            name="keeper", kind="keeper", image_reference=KEEPER_IMAGE,
            image_tag=images[KEEPER_IMAGE]["tag"],
            entrypoint=[KEEPER_ENTRYPOINT],
            command=["-I", "-S", "-c", f"import time; time.sleep({keeper_seconds})"],
            env={"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": "/scratch/home"},
            mounts=[(keeper_scratch, "/scratch", True)],
            network="none", timeout_seconds=keeper_seconds,
            log_path=engine.logs / "keeper.log",
        )
        journal["creates"].append(keeper_spec.name)
        _journal_write(journal_path, journal)
        keeper.id, keeper_error = _create(engine, keeper_spec, deadline=deadline, supervise=supervise)
        keeper.log_path = keeper_spec.log_path
        if keeper.id:
            journal["containers"].append(keeper.record())
            _journal_write(journal_path, journal)
        if keeper_error is not None:
            raise keeper_error
        journal["phase"] = "keeper-started"
        _journal_write(journal_path, journal)
        _start(engine, keeper, deadline=deadline, supervise=supervise)
        _attest_expiry(keeper, keeper_spec, deadline, keeper=True)
        _attest(
            engine, keeper, keeper_spec, image_id=images[KEEPER_IMAGE]["image_id"],
            keeper_id=None, keeper_net_inode=None,
            host_pid_inode=host_pid_inode, worker_cgroup=worker_cgroup,
            expect_waiting_gate=False, deadline=deadline, supervise=supervise,
        )
        keeper_id = keeper.id
        journal["keeper"] = keeper.record()
        _journal_write(journal_path, journal)

        service_containers: list[tuple[Mapping[str, Any], Container]] = []
        service_specs: list[tuple[Mapping[str, Any], Container, _Spec]] = []
        for index, service in enumerate(services):
            name = f"service-{index}-{service['id']}"
            scratch = _make_scratch(run_root, f"service-{index}")
            data = run_root / f"data-{index}"
            data.mkdir(mode=0o700)
            container = Container(name, "service", str(service["image"]), images[str(service["image"])]["tag"])
            spec = _Spec(
                name=name, kind="service", image_reference=str(service["image"]),
                image_tag=images[str(service["image"])]["tag"],
                entrypoint=["/bin/sh"], command=_gate_command([str(item) for item in service["argv"]]),
                env=_container_env(_service_env(services)),
                mounts=_mounts_for(
                    source_dir=source_dir, workspace=None, scratch=scratch, data=data, identity=identity,
                ),
                network=f"container:{keeper_id}",
                timeout_seconds=0,
                log_path=engine.logs / f"{name}.log",
            )
            journal["creates"].append(spec.name)
            _journal_write(journal_path, journal)
            container.id, container_error = _create(
                engine, spec, deadline=deadline, supervise=supervise,
            )
            container.log_path = spec.log_path
            containers.append(container)
            service_containers.append((service, container))
            service_specs.append((service, container, spec))
            if container.id:
                journal["containers"].append(container.record())
                _journal_write(journal_path, journal)
            if container_error is not None:
                raise container_error
            supervise()
            _start(engine, container, deadline=deadline, supervise=supervise)
            _attest_expiry(container, spec, deadline)

        verifier = Container("verifier", "verifier", str(recipe["image"]), images[str(recipe["image"])]["tag"])
        verifier_scratch = _make_scratch(run_root, "verifier")
        verifier_spec = _Spec(
            name="verifier", kind="verifier", image_reference=str(recipe["image"]),
            image_tag=images[str(recipe["image"])]["tag"],
            entrypoint=["/bin/sh"], command=_gate_command([str(item) for item in recipe["argv"]]),
            env=_container_env(_service_env(services)),
            mounts=_mounts_for(
                source_dir=source_dir, workspace=workspace, scratch=verifier_scratch, data=None, identity=identity,
            ),
            network=f"container:{keeper_id}",
            timeout_seconds=0,
            log_path=engine.logs / "verifier.log",
        )
        journal["creates"].append(verifier_spec.name)
        _journal_write(journal_path, journal)
        verifier.id, verifier_error = _create(
            engine, verifier_spec, deadline=deadline, supervise=supervise,
        )
        verifier.log_path = verifier_spec.log_path
        containers.append(verifier)
        if verifier.id:
            journal["containers"].append(verifier.record())
            _journal_write(journal_path, journal)
        if verifier_error is not None:
            raise verifier_error
        supervise()
        _start(engine, verifier, deadline=deadline, supervise=supervise)
        _attest_expiry(verifier, verifier_spec, deadline)

        journal["phase"] = "started"
        _journal_write(journal_path, journal)

        # Attest every container before any payload is released.
        for service, container, spec in service_specs:
            _attest(
                engine, container, spec, image_id=images[str(service["image"])]["image_id"],
                keeper_id=keeper_id, keeper_net_inode=keeper.net_inode,
                host_pid_inode=host_pid_inode, worker_cgroup=worker_cgroup,
                expect_waiting_gate=True, deadline=deadline, supervise=supervise,
            )
        _attest(
            engine, verifier, verifier_spec, image_id=images[str(recipe["image"])]["image_id"],
            keeper_id=keeper_id, keeper_net_inode=keeper.net_inode,
            host_pid_inode=host_pid_inode, worker_cgroup=worker_cgroup,
            expect_waiting_gate=True, deadline=deadline, supervise=supervise,
        )
        # Persist the identity tuples the reaper needs before anything is released.
        journal["containers"] = [container.record() for container in ([keeper] if keeper is not None else []) + containers]
        journal["phase"] = "attested"
        _journal_write(journal_path, journal)

        # Services must run before readiness can succeed; the verifier stays gated
        # until every declared service has answered its readiness check.
        for index, (_service, _container) in enumerate(service_containers):
            supervise()
            _release(run_root / f"scratch-service-{index}" / "release")
        journal["phase"] = "services-released"
        _journal_write(journal_path, journal)
        _wait_ready(
            engine, service_containers, deadline=deadline, claim_check=supervise,
            startup_deadline=min(deadline, time.monotonic() + STARTUP_DEADLINE_SECONDS),
        )
        supervise()
        journal["phase"] = "released"
        _journal_write(journal_path, journal)
        supervise()
        _release(verifier_scratch / "release")

        exit_code, timed_out = _wait_exit(
            engine, verifier, deadline=deadline, claim_check=supervise,
        )
        result["exit_code"] = exit_code
        result["timed_out"] = timed_out
        for container in [*containers, *([keeper] if keeper is not None else [])]:
            text, truncated = _collect_log(engine, container)
            if container is verifier:
                result["stdout"] = text
            else:
                result["stderr"] += text
            result["truncated"] = result["truncated"] or truncated
        result["stderr"] = result["stderr"][-VISIBLE_OUTPUT_BYTES:]
        result["ok"] = True
        result["accepted"] = bool(exit_code == 0 and not timed_out)
    except VerificationFailure as exc:
        result["error"] = exc.code
        result["message"] = _bounded(exc.message)
        for container in containers:
            with contextlib.suppress(Exception):
                text, truncated = _collect_log(engine, container)
                result["stdout"] += text
                result["truncated"] = result["truncated"] or truncated
    finally:
        cleanup_ok, cleanup_reasons = _cleanup(
            engine, ([keeper] if keeper is not None else []) + containers,
            creates=tuple(str(name) for name in journal.get("creates") or ()),
        )
        result["cleanup_ok"] = cleanup_ok
        result["cleanup_detail"] = cleanup_reasons
        if not cleanup_ok:
            result["accepted"] = False
            result["ok"] = False
        journal["phase"] = "cleaned" if cleanup_ok else "cleanup-failed"
        with contextlib.suppress(Exception):
            _journal_write(journal_path, journal)
        with contextlib.suppress(OSError):
            if cleanup_ok:
                journal_path.unlink()
        result.update({
            "task_id": getattr(grant, "task_id", None),
            "run_id": getattr(grant, "run_id", None),
            "preparation_id": getattr(grant, "preparation_id", None),
            "recipe": str(recipe.get("id")),
            "policy_digest": authority.policy_digest,
            "grant_digest": authority.bound_digest,
            "runtime_identity": getattr(getattr(grant, "runtime_identity", None), "as_dict", lambda: {})(),
            "images": {
                reference: {"image_id": entry["image_id"], "archive_sha256": entry["sha256"]}
                for reference, entry in images.items()
            },
            "resources": {
                "limits": limits,
                "keeper_id": keeper_id,
                "containers": [container.record() for container in ([keeper] if keeper is not None else []) + containers],
                "worker_cgroup": worker_cgroup,
                "namespaces": {
                    container.name: container.net_inode
                    for container in ([keeper] if keeper is not None else []) + containers
                },
            },
            "timestamps": {
                "started": started_epoch,
                "finished": time.time(),
                "deadline_epoch": deadline_epoch,
            },
        })
    return result


def _recover_cid(engine: Engine, spec: _Spec) -> Optional[str]:
    """The exact id Podman wrote for a create that was interrupted or ambiguous."""
    try:
        text = (engine.run_root / (spec.name + ".cid")).read_bytes().decode("utf-8").strip()
    except OSError:
        return None
    return text.lower() if _CONTAINER_ID_RE.match(text) else None


def _create(
    engine: Engine,
    spec: _Spec,
    *,
    deadline: Optional[float] = None,
    supervise: Optional[Callable[[], None]] = None,
) -> tuple[Optional[str], Optional[BaseException]]:
    """Create one container, returning its exact id even when the wait is aborted."""
    # Podman's --timeout bounds running time from start, and the value is frozen at
    # create. The remaining budget is bound here so the conmon expiry lands on the
    # invocation deadline plus at most the create/start latency (attested below).
    if spec.kind != "keeper" and deadline is not None:
        spec.timeout_seconds = _deadline_seconds(deadline)
    try:
        result = engine.run(
            _create_args(engine, spec), timeout=120.0, deadline=deadline, supervise=supervise,
            supervise_after=False,
        )
    except BaseException as exc:
        return _recover_cid(engine, spec), exc
    if result.returncode != 0:
        return _recover_cid(engine, spec), VerificationFailure(
            "engine_failed", _bounded(result.stderr.decode("utf-8", "replace")) or "container create failed",
        )
    container_id = result.stdout.decode("ascii", "replace").strip()
    if not _CONTAINER_ID_RE.match(container_id):
        return None, VerificationFailure("attestation_failed", "container create returned no exact id")
    return container_id, None


def _wait_exit(
    engine: Engine,
    container: Container,
    *,
    deadline: float,
    claim_check: Optional[Callable[[], None]],
    space_check: Optional[Callable[[], int]] = None,
) -> tuple[Optional[int], bool]:
    last_space = 0.0
    while time.monotonic() < deadline:
        if claim_check is not None:
            claim_check()
        if space_check is not None and time.monotonic() - last_space > 2.0:
            space_check()
            last_space = time.monotonic()
        data = _inspect_json(
            engine, container.id or "", deadline=deadline, supervise=claim_check,
        )
        state = data.get("State") or {}
        if state.get("Status") != "running":
            return int(state.get("ExitCode") or 0), False
        time.sleep(0.2)
    return None, True


def _engine_step(engine: Engine, argv: list[str], *, timeout: float) -> Optional[subprocess.CompletedProcess]:
    """Run one cleanup command; an engine that cannot start is a failed step, not a crash."""
    try:
        return engine.run(argv, timeout=timeout)
    except VerificationFailure:
        return None


def _cleanup(
    engine: Engine, containers: list[Container], *, creates: tuple[str, ...] = (),
) -> tuple[bool, list[str]]:
    ok = True
    reasons: list[str] = []
    try:
        listed = engine.run(["ps", "-a", "--no-trunc", "--format", "{{.ID}}"], timeout=30.0)
    except VerificationFailure as exc:
        # An engine that cannot start proves nothing was allocated only when this
        # invocation never asked for a container; otherwise the journal must survive.
        if not containers and not creates and remove_tree(engine.run_root):
            return True, [f"engine_unavailable_no_allocation:{exc.code}"]
        return False, [f"engine_unavailable:{exc.code}"]
    if listed.returncode != 0:
        if not containers and not creates and remove_tree(engine.run_root):
            return True, ["listing_failed_no_allocation"]
        # An unreadable inventory is not an empty one: never claim cleanup on it.
        ok = False
        reasons.append("listing_failed")
        present: set[str] = set()
    else:
        present = {
            line.strip().lower() for line in listed.stdout.decode("utf-8", "replace").splitlines() if line.strip()
        }
    # Children share the keeper's network namespace, so they must go first.
    for container in reversed(containers):
        if not container.id or container.id.lower() not in present:
            continue
        result = _engine_step(engine, ["rm", "-f", "-t", "2", container.id], timeout=60.0)
        if result is None:
            ok = False
            reasons.append(f"rm_failed:{container.name}:engine_unavailable")
        elif result.returncode != 0:
            ok = False
            detail = result.stderr.decode("utf-8", "replace").strip().replace("\n", " | ")[:200]
            reasons.append(f"rm_failed:{container.name}:{detail}")
    for container in reversed(containers):
        if not _terminate_recorded(container.pid, container.pid_start):
            ok = False
            reasons.append(f"kill_failed:{container.name}")
        if container.conmon_pid:
            _terminate_recorded(container.conmon_pid, container.conmon_start)
    listed = _engine_step(engine, ["ps", "-a", "--no-trunc", "--format", "{{.ID}}"], timeout=30.0)
    if listed is None or listed.returncode != 0:
        ok = False
        reasons.append("listing_failed_after_rm")
        remaining: set[str] = set()
    else:
        remaining = {
            line.strip().lower() for line in listed.stdout.decode("utf-8", "replace").splitlines() if line.strip()
        }
    for container in containers:
        if container.id and container.id.lower() in remaining:
            ok = False
            reasons.append(f"still_listed:{container.name}")
    known = {container.id.lower() for container in containers if container.id}
    for container_id in sorted(remaining - known):
        ok = False
        reasons.append(f"unaccounted_container:{container_id}")
    # Teardown is asynchronous at the kernel level; give the recorded processes a
    # bounded grace window, then judge. A process or namespace that survives it is
    # evidence, never a retry.
    grace_deadline = time.monotonic() + CLEANUP_GRACE_SECONDS
    while True:
        residual = False
        for container in containers:
            if container.pid and _process_running(container.pid, container.pid_start):
                residual = True
                reasons.append(f"pid_alive:{container.name}")
            if container.conmon_pid and _process_running(container.conmon_pid, container.conmon_start):
                residual = True
                reasons.append(f"conmon_alive:{container.name}")
            if container.net_inode is not None and _namespace_inodes_in_use(
                container.net_inode, worker_cgroup=engine.worker_cgroup,
            ):
                residual = True
                reasons.append(f"namespace_alive:{container.name}:{container.net_inode}")
            if container.pid_inode is not None and _namespace_inodes_in_use(
                container.pid_inode, name="pid", worker_cgroup=engine.worker_cgroup,
            ):
                residual = True
                reasons.append(f"pid_namespace_alive:{container.name}:{container.pid_inode}")
        if not residual:
            break
        if time.monotonic() >= grace_deadline:
            ok = False
            reasons.append("residual_process_or_namespace")
            break
        time.sleep(0.2)
    if ok:
        removed = remove_tree(engine.run_root)
        ok = removed
        if not removed:
            reasons.append("store_not_removed")
    return ok, reasons


def _terminate_recorded(pid: Any, start: Any) -> bool:
    """SIGKILL one recorded container process only while its start time still matches.

    The container engine is the primary killer (conmon's hard deadline). This is the
    fallback for an engine process that was itself killed: the recorded init is
    identity-checked, signalled, and the private PID namespace tears its children
    down with it.
    """
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return True
    if isinstance(start, bool) or not isinstance(start, int):
        return False
    identity = _process_identity(pid, start)
    if identity == "gone":
        return True
    if identity != "match":
        # Never signal a process whose identity is not proven.
        return False
    try:
        os.kill(pid, getattr(signal, "SIGKILL", signal.SIGTERM))
    except ProcessLookupError:
        return True
    except OSError:
        return False
    return True


def _namespace_inodes_in_use(inode: int, *, name: str = "net", worker_cgroup: str = "") -> bool:
    """Any process of this host UID still inside a recorded namespace blocks cleanup.

    An unreadable namespace link is uncertainty, not absence: it fails closed when the
    process belongs to this worker's cgroup. A process outside it (for example the
    user manager, whose namespace links are not readable) is unrelated host evidence.
    """
    if inode <= 0:
        return False
    try:
        entries = list(os.scandir("/proc"))
    except OSError:
        return True
    for entry in entries:
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            info = os.stat(f"/proc/{pid}")
        except OSError:
            continue
        if int(info.st_uid) != (os.getuid() if hasattr(os, "getuid") else -1):  # windows-footgun: ok — hasattr-guarded; POSIX-only /proc scan
            continue
        try:
            observed = int(os.stat(f"/proc/{pid}/ns/{name}").st_ino)
        except FileNotFoundError:
            continue
        except OSError:
            if worker_cgroup:
                process_cgroup = _process_cgroup(pid)
                if process_cgroup is not None and not _cgroup_contained(process_cgroup, worker_cgroup):
                    continue
            return True
        if observed != inode:
            continue
        state = _process_state(pid)
        if state in {"Z", "X", "x"}:
            # A zombie still appears in /proc and holds a namespace reference until
            # its parent reaps it, but it executes nothing and owns no children.
            continue
        return True
    return False


def _host_epoch() -> str:
    """The current boot identity, or ``"unknown"`` when it cannot be established."""
    try:
        value = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()
    except OSError:
        return "unknown"
    return value or "unknown"


def cleanup_journal(journal_path: Path, *, home: Optional[Path] = None) -> tuple[bool, str]:
    """Tear down one recorded invocation using only its own private engine store."""
    payload = _journal_read(journal_path)
    if payload is None:
        return False, "journal_unreadable"
    root = Path(str(payload.get("run_root_path") or ""))
    record = payload.get("run_root") or {}
    recorded_epoch = payload.get("host_epoch")
    current_epoch = _host_epoch()
    if (
        not isinstance(recorded_epoch, str) or not recorded_epoch
        or recorded_epoch == "unknown"
        or not current_epoch or current_epoch == "unknown"
    ):
        # Process identity is boot-relative: without two known boot identities nothing
        # can be concluded, so keep the journal and never signal a possibly reused PID.
        return False, "boot_identity_unknown"
    stale_boot = recorded_epoch != current_epoch
    try:
        info = os.lstat(root)
    except FileNotFoundError:
        # The store is gone, but recorded processes may still run (start times are
        # boot-relative, so a prior boot proves nothing survives).
        if not stale_boot:
            for record in payload.get("containers") or []:
                if _process_identity(record.get("pid"), record.get("pid_start")) == "match":
                    return False, "root_absent_processes_alive"
                if _process_identity(record.get("conmon_pid"), record.get("conmon_start")) == "match":
                    return False, "root_absent_processes_alive"
        with contextlib.suppress(OSError):
            journal_path.unlink()
        return True, "absent"
    except OSError:
        return False, "root_unreadable"
    if (
        not stat.S_ISDIR(info.st_mode) or root.is_symlink()
        or int(info.st_dev) != int(record.get("device", -1))
        or int(info.st_ino) != int(record.get("inode", -1))
    ):
        return False, "path_identity_changed"
    if payload.get("phase") == "cleaned":
        with contextlib.suppress(OSError):
            journal_path.unlink()
        return True, "already_cleaned"
    engine = Engine(root)
    ok = True
    try:
        listed = engine.run(["ps", "-a", "--no-trunc", "--format", "{{.ID}}"], timeout=60.0)
    except VerificationFailure as exc:
        return False, f"engine_unavailable:{exc.code}"
    if listed.returncode != 0:
        return False, "listing_failed"
    present = {
        line.strip().lower() for line in listed.stdout.decode("utf-8", "replace").splitlines() if line.strip()
    }
    # Creation intent is journaled before `create`, and the engine writes a CID file
    # per container. Both are identity-checked evidence from this invocation's own
    # store, so an interrupted create stays reclaimable; a missing CID is unknown
    # allocation, never "nothing created".
    recorded = {str(record.get("id") or "").lower() for record in payload.get("containers") or []}
    recovered: list[tuple[str, str]] = []
    for name in [str(item) for item in payload.get("creates") or []]:
        try:
            candidate = (root / f"{name}.cid").read_bytes().decode("utf-8").strip().lower()
        except OSError:
            candidate = ""
        if _CONTAINER_ID_RE.match(candidate) and candidate not in recorded:
            recovered.append((name, candidate))
            recorded.add(candidate)
    failures: list[str] = []
    # Children share the keeper's network namespace: remove them first.
    for name, candidate in reversed(recovered):
        if candidate not in present:
            continue
        result = _engine_step(engine, ["rm", "-f", "-t", "2", candidate], timeout=60.0)
        if result is None or result.returncode != 0:
            ok = False
            failures.append(f"rm:{name}")
    for record in reversed(payload.get("containers") or []):
        container_id = str(record.get("id") or "")
        if not _CONTAINER_ID_RE.match(container_id):
            ok = False
            failures.append(f"invalid_id:{record.get('name')}")
            continue
        if container_id.lower() not in present:
            continue  # already gone (engine deadline or an earlier pass)
        result = _engine_step(engine, ["rm", "-f", "-t", "2", container_id], timeout=60.0)
        if result is None or result.returncode != 0:
            ok = False
            failures.append(f"rm:{record.get('name')}")
    worker_cgroup = str(payload.get("worker_cgroup") or "")
    if not stale_boot:
        for record in payload.get("containers") or []:
            if not _terminate_recorded(record.get("pid"), record.get("pid_start")):
                ok = False
                failures.append(f"unverified_pid:{record.get('name')}")
            if not _terminate_recorded(record.get("conmon_pid"), record.get("conmon_start")):
                ok = False
                failures.append(f"unverified_conmon:{record.get('name')}")
    # Kernel teardown is asynchronous: wait out the recorded processes and namespace
    # references before judging, exactly like the in-process cleanup path.
    grace_deadline = time.monotonic() + CLEANUP_GRACE_SECONDS
    while True:
        reasons: list[str] = []
        for record in payload.get("containers") or []:
            if stale_boot:
                continue
            if _process_running(record.get("pid"), record.get("pid_start")):
                reasons.append(f"pid:{record.get('name')}")
            if _process_running(record.get("conmon_pid"), record.get("conmon_start")):
                reasons.append(f"conmon:{record.get('name')}")
            inode = record.get("net_inode")
            if isinstance(inode, int) and _namespace_inodes_in_use(inode, worker_cgroup=worker_cgroup):
                reasons.append(f"netns:{record.get('name')}")
            pid_inode = record.get("pid_inode")
            if isinstance(pid_inode, int) and _namespace_inodes_in_use(
                pid_inode, name="pid", worker_cgroup=worker_cgroup,
            ):
                reasons.append(f"pidns:{record.get('name')}")
        if not reasons:
            break
        if time.monotonic() >= grace_deadline:
            return False, "leftover_processes:" + ",".join(sorted(set(reasons)))
        time.sleep(0.2)
    final = _engine_step(engine, ["ps", "-a", "--no-trunc", "--format", "{{.ID}}"], timeout=60.0)
    if final is None or final.returncode != 0:
        return False, "listing_failed_after_rm"
    leftovers = {
        line.strip().lower() for line in final.stdout.decode("utf-8", "replace").splitlines() if line.strip()
    }
    if leftovers:
        # Never delete the store or the journal while an owned container remains
        # outside exact-ID reclamation.
        return False, "unaccounted_containers:" + ",".join(sorted(leftovers))
    if ok:
        if not remove_tree(root):
            return False, "store_not_removed"
        with contextlib.suppress(OSError):
            journal_path.unlink()
        return True, "cleaned"
    return False, "leftover_processes:" + ",".join(sorted(set(failures)))


def reap_journals(
    journal_dir: Path, *, max_journals: int = MAX_REAP_JOURNALS,
) -> tuple[list[dict[str, Any]], int]:
    """Reconcile one directory; return ``(results, journals_examined)``.

    Callers carry the budget across directories so a dispatcher pass stays bounded.
    """
    results: list[dict[str, Any]] = []
    if not journal_dir.is_dir() or journal_dir.is_symlink():
        return results, 0
    processed = 0
    for path in sorted(journal_dir.glob("*.json")):
        if processed >= max_journals:
            break
        payload = _journal_read(path)
        if payload is None:
            results.append({"journal": path.name, "ok": False, "reason": "journal_unreadable"})
            processed += 1
            continue
        worker_pid = payload.get("worker_pid")
        worker_start = payload.get("worker_start")
        owner_alive = (
            isinstance(worker_pid, int) and isinstance(worker_start, int)
            and _process_start_time(worker_pid) == worker_start
        )
        epoch = payload.get("host_epoch")
        if epoch != _host_epoch():
            # A changed boot epoch proves every recorded process is gone, but never
            # authorizes deleting a path whose identity changed.
            owner_alive = False
        if owner_alive:
            continue
        deadline = payload.get("deadline_epoch")
        if isinstance(deadline, (int, float)) and time.time() < float(deadline):
            continue
        try:
            ok, reason = cleanup_journal(path)
        except Exception as exc:  # one bad journal must not stop the sweep
            ok, reason = False, f"cleanup_error:{_bounded(exc, 120)}"
        results.append({"journal": path.name, "ok": ok, "reason": reason})
        processed += 1
    return results, processed

"""Accepted-source inventory, synthetic Git and typed read/list/diff inspection.

Nothing here executes repository code: Git runs from a private synthetic directory with
fixed argv and a minimal environment, and files are read through held directory fds.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import posixpath
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

import json
import os
import posixpath
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

TOOL_INSPECT = "kanban_worker_inspect"
TOOL_VERIFY = "kanban_worker_verify"
EXECUTION_TOOLS = frozenset({TOOL_INSPECT, TOOL_VERIFY})
from hermes_cli.worker_execution_policy import _minimal_git_env

#: The frozen plugin identity that must supply the mandatory pre-execution approval.
POLICY_PLUGIN_ID = "cto-consequence-guard"
POLICY_PLUGIN_MODULE = "__init__.py"
POLICY_PLUGIN_MANIFEST = "plugin.yaml"
REQUIRED_HOOK = "pre_worker_execution"
from hermes_cli.worker_execution_common import (
    CONFIG_KEY, EXECUTION_TOOLS, GIT_TIMEOUT_SECONDS, LIST_DEFAULT_LIMIT, MAX_ARGV_BYTES,
    MAX_ARGV_ELEMENTS, MAX_GIT_OUTPUT_BYTES, MAX_INVENTORY_BYTES, MAX_INVENTORY_ENTRIES,
    MAX_PAGE_BYTES, MAX_PAGE_LINES, MAX_POLICY_BYTES, MAX_RECIPES, MAX_SCAN_STATUS_BYTES,
    MAX_SERVICES_PER_RECIPE, MAX_TIMEOUT_SECONDS, PAGE_DEFAULT_LINES, POLICY_PLUGIN_ID,
    POLICY_PLUGIN_MANIFEST, POLICY_PLUGIN_MODULE, POLICY_PROTOCOL, REQUIRED_HOOK,
    RESERVED_BASENAME_ALLOWED, RESERVED_BASENAME_PREFIX, RESERVED_BASENAMES,
    RESERVED_COMPONENTS, TOOL_INSPECT, TOOL_VERIFY, ExecutionAuthority, ExecutionGrant,
    FrozenDict, WorkerExecutionError, WorkerExecutionRefused, WorkerExecutionUnavailable,
    _DIGEST_RE, _GIT, _IMAGE_RE, _OBJECT_ID_RE, _SERVICE_ID_RE, _VERIFY_LOCK, _bounded,
    _freeze_mapping, _freeze_recipes, _record_value, _refuse, canonical_bytes, digest_of,
    sha256_hex,
)

# ---------------------------------------------------------------------------
# Safe source traversal
# ---------------------------------------------------------------------------

_O_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_O_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
_O_NONBLOCK = getattr(os, "O_NONBLOCK", 0)


class SafeRoot:
    """Component-wise, symlink-refusing traversal under one held directory FD."""

    def __init__(self, path: str) -> None:
        self.path = path
        try:
            self.fd = os.open(path, os.O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW | _O_CLOEXEC)
        except OSError as exc:
            raise WorkerExecutionRefused("workspace_unavailable", _bounded(exc))
        info = os.fstat(self.fd)
        self.device = int(info.st_dev)
        self.inode = int(info.st_ino)

    def close(self) -> None:
        with contextlib.suppress(OSError):
            os.close(self.fd)

    def __enter__(self) -> "SafeRoot":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    def _walk(self, parts: list[str], *, final_dir: bool) -> int:
        fd = self._open_base()
        try:
            for index, part in enumerate(parts):
                last = index == len(parts) - 1
                flags = os.O_RDONLY | _O_NOFOLLOW | _O_CLOEXEC
                if last and not final_dir:
                    # A non-regular entry (FIFO/device) must fail on fstat, never block
                    # the open itself.
                    flags |= _O_NONBLOCK
                if not last or final_dir:
                    flags |= _O_DIRECTORY
                next_fd = os.open(part, flags, dir_fd=fd)
                os.close(fd)
                fd = next_fd
            return fd
        except OSError:
            os.close(fd)
            raise

    def _open_base(self) -> int:
        """A fresh descriptor for the SAME directory inode this root was opened on.

        Re-opening ``self.path`` would follow a pathname swapped after the identity
        check. ``/proc/self/fd/<n>`` names the held inode instead; the duplicate is
        then re-verified, so every traversal stays anchored to the granted directory.
        """
        fd = os.open(f"/proc/self/fd/{self.fd}", os.O_RDONLY | _O_DIRECTORY | _O_CLOEXEC)
        info = os.fstat(fd)
        if (int(info.st_dev), int(info.st_ino)) != (self.device, self.inode):
            os.close(fd)
            raise WorkerExecutionRefused("workspace_changed", "granted workspace root identity changed")
        return fd

    def current_identity(self) -> tuple[int, int]:
        info = os.fstat(self.fd)
        return int(info.st_dev), int(info.st_ino)

    def open_dir(self, parts: list[str]) -> int:
        return self._walk(parts, final_dir=True)

    def open_file(self, parts: list[str]) -> int:
        return self._walk(parts, final_dir=False)

    def lstat(self, parts: list[str]) -> os.stat_result:
        if not parts:
            return os.fstat(self.fd)
        parent = self.open_dir(parts[:-1]) if len(parts) > 1 else self._open_base()
        try:
            return os.stat(parts[-1], dir_fd=parent, follow_symlinks=False)
        finally:
            os.close(parent)

    def readlink(self, parts: list[str]) -> str:
        parent = self.open_dir(parts[:-1]) if len(parts) > 1 else self._open_base()
        try:
            return os.readlink(parts[-1], dir_fd=parent)
        finally:
            os.close(parent)


def _split_relative(path: str, label: str = "path") -> list[str]:
    text = str(path or "").strip()
    if not text:
        return []
    if "\x00" in text or "\\" in text or os.path.isabs(text):
        _refuse("invalid_path", f"{label} must be a relative POSIX path")
    parts = text.split("/")
    if any(part in ("", ".", "..") for part in parts):
        _refuse("invalid_path", f"{label} must not contain empty, '.' or '..' components")
    return parts


def _validate_source_name(name: str) -> None:
    """A repository-controlled path must stay a plain relative POSIX path."""
    parts = name.split("/")
    if (
        not name or "\\" in name or "\x00" in name or name.startswith("/")
        or any(part in ("", ".", "..") for part in parts)
    ):
        _refuse("git_unsupported", "the repository names an unsafe source path")


def _reserved(parts: list[str]) -> bool:
    if not parts:
        return False
    for part in parts[:-1]:
        if part in RESERVED_COMPONENTS:
            return True
    name = parts[-1]
    if name in RESERVED_COMPONENTS:
        return True
    if name in RESERVED_BASENAME_ALLOWED:
        return False
    if name in RESERVED_BASENAMES:
        return True
    return name.startswith(RESERVED_BASENAME_PREFIX)


@dataclass
class SourceEntry:
    path: str
    kind: str  # "file" | "link" | "deleted"
    executable: bool
    size: int
    digest: str

    def digest_line(self) -> bytes:
        return (
            f"{self.path}\x00{self.kind}\x00{'1' if self.executable else '0'}\x00{self.digest}\n"
        ).encode("utf-8", "surrogateescape")


@dataclass
class SourceSnapshot:
    head: str
    branch: Optional[str]
    detached: bool
    entries: list[SourceEntry]
    source_digest: str
    index_digest: str
    staged_dir: Optional[Path]

    def entry(self, path: str) -> Optional[SourceEntry]:
        for item in self.entries:
            if item.path == path:
                return item
        return None


class SyntheticGit:
    """A private, generated Git directory with no hooks, includes or live config."""

    def __init__(self, workspace: str, git: Mapping[str, Any]) -> None:
        self.workspace = workspace
        self.git_dir = Path(str(git["git_dir"]))
        self.common_dir = Path(str(git["common_dir"]))
        self.root = Path(tempfile.mkdtemp(prefix="synthetic-git-", dir=_scratch_dir()))
        self.path = self.root / "git"
        try:
            self.path.mkdir(mode=0o700)
            self._write_metadata()
        except BaseException:
            shutil.rmtree(self.root, ignore_errors=True)
            raise

    def _scratch_dir(self) -> Path:
        return self.root

    def _copy_regular(self, source: Path, target: Path, *, limit: int = 1024 * 1024) -> bool:
        """Copy one bounded regular Git metadata file, validating the opened inode.

        The descriptor is opened first (no symlink following) and every check runs on
        that descriptor, so a pathname swap between check and open cannot substitute a
        different file.
        """
        try:
            fd = os.open(source, os.O_RDONLY | _O_NOFOLLOW | _O_NONBLOCK | _O_CLOEXEC)
        except FileNotFoundError:
            return False
        except OSError as exc:
            _refuse("git_metadata_unsafe", _bounded(exc))
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
                _refuse("git_metadata_unsafe", f"Git metadata {source.name} is not a bounded regular file")
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            with os.fdopen(fd, "rb", closefd=False) as reader, open(target, "xb") as writer:
                shutil.copyfileobj(reader, writer, 1024 * 1024)
        finally:
            os.close(fd)
        return True

    def _write_metadata(self) -> None:
        self._copy_regular(self.git_dir / "HEAD", self.path / "HEAD")
        self._copy_regular(self.git_dir / "index", self.path / "index", limit=64 * 1024 * 1024)
        self._copy_refs(self.common_dir / "refs", self.path / "refs")
        for name in ("packed-refs", "shallow"):
            with contextlib.suppress(WorkerExecutionError):
                self._copy_regular(self.common_dir / name, self.path / name, limit=8 * 1024 * 1024)
        self._write_config()

    def _copy_refs(self, source: Path, target: Path) -> None:
        try:
            info = os.lstat(source)
        except OSError:
            return
        if not stat.S_ISDIR(info.st_mode) or source.is_symlink():
            _refuse("git_metadata_unsafe", "Git refs storage is not a real directory")
        for directory, _dirnames, filenames in os.walk(source, followlinks=False):
            base = Path(directory)
            for name in filenames:
                candidate = base / name
                relative = candidate.relative_to(source)
                self._copy_regular(candidate, target / relative, limit=4096)

    def _write_config(self) -> None:
        format_lines = self._format_scalars()
        # Only validated scalar constants reach the generated config: a repository
        # value with quotes/newlines must not be able to inject sections, includes or
        # filter definitions into the trusted synthetic file.
        version = format_lines.get("core.repositoryformatversion") or "0"
        if not re.fullmatch(r"0|1", version):
            _refuse("git_unsupported", "Git repository format version is not supported")
        config: list[str] = [
            "[core]",
            "\trepositoryformatversion = " + version,
            "\tbare = false",
            "\thooksPath = /dev/null",
            "\tfsmonitor = false",
            "\tpager = cat",
            "\tuntrackedCache = false",
            "\tprecomposeUnicode = false",
            "[protocol]",
            "\tallow = never",
            "[credential]",
            "\thelper =",
            "[submodule]",
            "\trecurse = false",
            "[diff]",
            "\texternal =",
            "\tnoprefix = false",
            "[gc]",
            "\tauto = 0",
            "[maintenance]",
            "\tauto = false",
        ]
        extensions = {key: value for key, value in format_lines.items() if key.startswith("extensions.")}
        if extensions:
            config.append("[extensions]")
            for key, value in sorted(extensions.items()):
                name = key.split(".", 1)[1]
                if not re.fullmatch(r"[a-z][a-z0-9-]{0,31}", name):
                    _refuse("git_unsupported", "Git extension name is not supported for inspection")
                if name not in {"objectformat"}:
                    _refuse("git_unsupported", f"Git extension {name} is not supported for inspection")
                if value.lower() not in {"sha1", "sha256"}:
                    _refuse("git_unsupported", "Git object format is not supported for inspection")
                config.append(f"\t{name} = {value.lower()}")
        body = "\n".join(config) + "\n"
        config_path = self.path / "config"
        config_path.write_text(body, encoding="utf-8")
        self.config_path = config_path

    def _format_scalars(self) -> dict[str, str]:
        source = self.git_dir / "config"
        if not source.is_file() or source.is_symlink():
            source = self.common_dir / "config"
        if not source.is_file() or source.is_symlink():
            return {}
        try:
            info = os.lstat(source)
            if not stat.S_ISREG(info.st_mode) or info.st_size > 1024 * 1024:
                _refuse("git_metadata_unsafe", "Git config is not a bounded regular file")
            fd = os.open(source, os.O_RDONLY | _O_NOFOLLOW | _O_NONBLOCK | _O_CLOEXEC)
            opened = os.fstat(fd)
            if not stat.S_ISREG(opened.st_mode) or opened.st_size > 1024 * 1024:
                os.close(fd)
                _refuse("git_metadata_unsafe", "Git config is not a bounded regular file")
        except OSError as exc:
            _refuse("git_metadata_unsafe", _bounded(exc))
        try:
            env = _minimal_git_env(None)
            result = subprocess.run(  # noqa: S603 -- fixed absolute executable
                [_GIT, "config", "--no-includes", "--file", f"/proc/self/fd/{fd}",
                 "--null", "--get-regexp", r"^(core\.repositoryformatversion|extensions\..*)$"],
                cwd=str(self.root), env=env, stdin=subprocess.DEVNULL,
                capture_output=True, timeout=GIT_TIMEOUT_SECONDS, pass_fds=(fd,),
            )
        except (OSError, subprocess.SubprocessError) as exc:
            _refuse("git_unavailable", _bounded(exc))
        finally:
            os.close(fd)
        if result.returncode not in (0, 1):
            _refuse("git_unavailable", "Git config could not be read")
        values: dict[str, str] = {}
        for item in result.stdout.split(b"\x00"):
            if not item.strip():
                continue
            try:
                text = item.decode("utf-8", "strict")
            except UnicodeDecodeError:
                _refuse("git_unsupported", "Git config is not valid UTF-8")
            if "\n" in text:
                key, _, value = text.partition("\n")
            else:
                key, _, value = text.partition(" ")
            key, value = key.strip(), value.strip()
            if not re.fullmatch(r"(core\.repositoryformatversion|extensions\.[a-z][a-z0-9-]{0,31})", key):
                continue
            if any(ch in value for ch in "\x00\n\r") or len(value) > 64:
                _refuse("git_unsupported", "Git config value is not a supported scalar")
            values[key] = value
        return values

    def object_dir(self) -> Path:
        objects = self.common_dir / "objects"
        try:
            info = os.lstat(objects)
        except OSError as exc:
            raise WorkerExecutionRefused("git_unavailable", _bounded(exc))
        if not stat.S_ISDIR(info.st_mode) or objects.is_symlink():
            _refuse("git_metadata_unsafe", "Git object storage is not a real directory")
        alternates = objects / "info" / "alternates"
        if alternates.exists() or alternates.is_symlink():
            _refuse("git_metadata_unsafe", "Git object alternates are not supported")
        entries = 0
        for directory, dirnames, filenames in os.walk(objects, followlinks=False):
            base = Path(directory)
            try:
                base_info = os.lstat(base)
            except OSError as exc:
                _refuse("git_metadata_unsafe", _bounded(exc))
            if not stat.S_ISDIR(base_info.st_mode) or base.is_symlink():
                _refuse("git_metadata_unsafe", "Git object storage contains a non-directory")
            for name in [*dirnames, *filenames]:
                candidate = base / name
                entries += 1
                if entries > 1_000_000:
                    _refuse("git_unsupported", "Git object storage is too large for inspection")
                try:
                    entry_info = os.lstat(candidate)
                except OSError as exc:
                    _refuse("git_metadata_unsafe", _bounded(exc))
                if stat.S_ISLNK(entry_info.st_mode):
                    _refuse("git_metadata_unsafe", "Git object storage contains a symlink")
                if not (stat.S_ISDIR(entry_info.st_mode) or stat.S_ISREG(entry_info.st_mode)):
                    _refuse("git_metadata_unsafe", "Git object storage contains a non-regular entry")
            dirnames[:] = [name for name in dirnames if not (base / name).is_symlink()]
        return objects

    def run(self, args: list[str], *, cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
        from hermes_cli._subprocess_compat import harden_git_argv

        objects = self.object_dir()
        env = _minimal_git_env(str(objects))
        argv = [
            _GIT,
            f"--git-dir={self.path}",
            f"--work-tree={self.workspace}",
            *harden_git_argv(args),
        ]
        try:
            return subprocess.run(  # noqa: S603 -- fixed executable, validated argv
                argv, cwd=str(cwd or self.root), env=env, stdin=subprocess.DEVNULL,
                capture_output=True, timeout=GIT_TIMEOUT_SECONDS,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise WorkerExecutionRefused("git_unavailable", _bounded(exc))

    def close(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)


def _scratch_dir() -> str:
    base = os.environ.get("TMPDIR") or tempfile.gettempdir()
    parent = Path(base) / "hermes-worker-inspect"
    parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    return str(parent)


def _git_text(git: SyntheticGit, args: list[str]) -> str:
    result = git.run(args)
    if result.returncode != 0:
        _refuse("git_unavailable", "Git inspection command failed")
    try:
        return result.stdout.decode("utf-8", "strict")
    except UnicodeDecodeError:
        _refuse("git_unsupported", "Git output is not valid UTF-8")


def _git_tracked(git: SyntheticGit) -> dict[str, tuple[str, str]]:
    """``path -> (mode, object id)`` from the index; no worktree access."""
    raw = git.run(["ls-files", "--stage", "-z"])
    if raw.returncode != 0:
        _refuse("git_unavailable", "Git index could not be read")
    tracked: dict[str, tuple[str, str]] = {}
    for item in raw.stdout.split(b"\x00"):
        if not item:
            continue
        try:
            header, _, path = item.partition(b"\t")
            fields = header.decode("ascii").split(" ")
            name = path.decode("utf-8", "strict")
        except UnicodeDecodeError:
            _refuse("git_unsupported", "Git index contains a non-UTF-8 path")
        if len(fields) != 3:
            _refuse("git_unsupported", "Git index entry is malformed")
        _validate_source_name(name)
        tracked[name] = (fields[0], fields[1])
    return tracked


def _index_digest(git: SyntheticGit) -> str:
    try:
        info = os.lstat(git.path / "index")
    except OSError:
        return "sha256:" + sha256_hex(b"missing-index")
    with open(git.path / "index", "rb") as handle:
        return "sha256:" + sha256_hex(handle.read())


def _scan_source(
    authority: ExecutionAuthority,
    *,
    materialize: Optional[Path] = None,
) -> SourceSnapshot:
    """Build one bounded inventory from tracked paths plus granted extra source files."""
    if authority.git is None:
        raise WorkerExecutionRefused("workspace_not_a_repository", "granted workspace is not a Git repository")
    git = SyntheticGit(authority.workspace_path, authority.git)
    try:
        head_result = git.run(["rev-parse", "HEAD"])
        if head_result.returncode != 0:
            raise WorkerExecutionRefused("workspace_not_a_repository", "granted workspace has no HEAD commit")
        try:
            head = head_result.stdout.decode("ascii").strip()
        except UnicodeDecodeError:
            raise WorkerExecutionRefused("git_unsupported", "Git HEAD is malformed")
        if not _OBJECT_ID_RE.match(head):
            raise WorkerExecutionRefused("git_unsupported", "Git HEAD is not an object id")
        symbolic = git.run(["symbolic-ref", "--quiet", "HEAD"])
        branch = None
        if symbolic.returncode == 0:
            try:
                branch = symbolic.stdout.decode("utf-8", "strict").strip().removeprefix("refs/heads/")
            except UnicodeDecodeError:
                raise WorkerExecutionRefused("git_unsupported", "Git branch is malformed")
        tracked = _git_tracked(git)
        extras = [list(_split_relative(item, "extra_source_paths")) for item in authority.extra_source_paths]
        reserved_tracked = [path for path in tracked if _reserved(path.split("/"))]
        reserved_extras = [parts for parts in extras if _reserved(parts)]
        if reserved_tracked or reserved_extras:
            raise WorkerExecutionRefused(
                "reserved_source_paths",
                f"the granted source set contains {len(reserved_tracked) + len(reserved_extras)} reserved path(s); "
                "verification cannot run against a silently reduced tree",
            )
        for path, (mode, _oid) in tracked.items():
            if mode == "160000":
                raise WorkerExecutionRefused("submodule_source", "the granted source set contains a submodule gitlink")
            if mode not in {"100644", "100755", "120000"}:
                raise WorkerExecutionRefused("unsupported_source_mode", "the granted source set contains an unsupported file mode")
        accepted: dict[str, str] = {}
        for path, (mode, _oid) in tracked.items():
            accepted[path] = "link" if mode == "120000" else "file"
        for parts in extras:
            name = "/".join(parts)
            if name in accepted:
                continue
            accepted[name] = "file"
        entries: list[SourceEntry] = []
        link_targets: dict[str, tuple[str, str]] = {}
        total_bytes = 0
        if len(accepted) > MAX_INVENTORY_ENTRIES:
            raise WorkerExecutionRefused("source_too_large", "the granted source set exceeds the entry bound")
        with SafeRoot(authority.workspace_path) as root:
            for name in sorted(accepted):
                parts = name.split("/")
                try:
                    info = root.lstat(parts)
                except FileNotFoundError:
                    entries.append(SourceEntry(name, "deleted", False, 0, "deleted"))
                    continue
                except OSError as exc:
                    raise WorkerExecutionRefused("source_unreadable", _bounded(exc))
                # The worktree entry's ACTUAL type decides how it is hashed and copied;
                # the index only selects the path. A tracked symlink replaced by a file
                # (or the reverse) is therefore represented, never silently dropped.
                if stat.S_ISLNK(info.st_mode):
                    target = root.readlink(parts)
                    parent = "/".join(parts[:-1])
                    joined = posixpath.normpath(posixpath.join(parent, target)) if parent else posixpath.normpath(target)
                    normalized = joined.replace(os.sep, "/")
                    if (
                        os.path.isabs(target) or "\x00" in target
                        or normalized in ("", ".", "..") or normalized.startswith("../")
                    ):
                        raise WorkerExecutionRefused("unsafe_symlink", "the granted source set contains an escaping symlink")
                    if normalized not in accepted:
                        raise WorkerExecutionRefused("unsafe_symlink", "the granted source set contains an unaccepted symlink target")
                    if normalized == name:
                        raise WorkerExecutionRefused("unsafe_symlink", "the granted source set contains a symlink cycle")
                    # The digest binds the exact link text that is copied; normalization
                    # is only for containment and graph validation.
                    digest = "sha256:" + sha256_hex(target.encode("utf-8", "surrogateescape"))
                    link_targets[name] = (target, normalized)
                    entries.append(SourceEntry(name, "link", False, len(target), digest))
                    if materialize is not None:
                        target_path = materialize / name
                        target_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                        os.symlink(target, target_path)
                    continue
                if not stat.S_ISREG(info.st_mode):
                    raise WorkerExecutionRefused("unsupported_source_type", "the granted source set contains a non-regular entry")
                if int(info.st_nlink) != 1:
                    raise WorkerExecutionRefused("hardlinked_source", "the granted source set contains a hard-linked file")
                try:
                    fd = root.open_file(parts)
                except OSError as exc:
                    raise WorkerExecutionRefused("source_unreadable", _bounded(exc))
                try:
                    opened = os.fstat(fd)
                    if not stat.S_ISREG(opened.st_mode) or int(opened.st_nlink) != 1:
                        raise WorkerExecutionRefused("hardlinked_source", "the granted source set changed during inspection")
                    digest = hashlib.sha256()
                    written = None
                    if materialize is not None:
                        target_path = materialize / name
                        target_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                        written = open(target_path, "xb")
                    try:
                        while True:
                            chunk = os.read(fd, 1024 * 1024)
                            if not chunk:
                                break
                            total_bytes += len(chunk)
                            if total_bytes > MAX_INVENTORY_BYTES:
                                raise WorkerExecutionRefused("source_too_large", "the granted source set exceeds the byte bound")
                            digest.update(chunk)
                            if written is not None:
                                written.write(chunk)
                    finally:
                        if written is not None:
                            written.close()
                    after = os.fstat(fd)
                    if (after.st_size, after.st_mtime_ns, after.st_ino) != (opened.st_size, opened.st_mtime_ns, opened.st_ino):
                        raise WorkerExecutionRefused("source_changed", "source changed during inspection")
                finally:
                    os.close(fd)
                executable = bool(opened.st_mode & 0o111)
                entries.append(SourceEntry(name, "file", executable, int(opened.st_size), "sha256:" + digest.hexdigest()))
                if materialize is not None:
                    with contextlib.suppress(OSError):
                        os.chmod(materialize / name, 0o755 if executable else 0o644)
        _validate_link_graph(link_targets, accepted)
        digests = sorted(entry.digest_line() for entry in entries)
        source_digest = "sha256:" + sha256_hex(b"".join(digests) + b"HEAD\x00" + head.encode("ascii"))
        return SourceSnapshot(
            head=head,
            branch=branch,
            detached=branch is None,
            entries=entries,
            source_digest=source_digest,
            index_digest=_index_digest(git),
            staged_dir=materialize,
        )
    finally:
        git.close()


def _validate_link_graph(
    link_targets: Mapping[str, tuple[str, str]], accepted: Mapping[str, str],
) -> None:
    """Every accepted link chain must terminate in an accepted non-link entry."""
    for name in sorted(link_targets):
        seen = {name}
        current = link_targets[name][1]
        for _ in range(64):
            if current not in accepted:
                raise WorkerExecutionRefused("unsafe_symlink", "the granted source set contains an unaccepted symlink target")
            if current not in link_targets:
                break
            if current in seen:
                raise WorkerExecutionRefused("unsafe_symlink", "the granted source set contains a symlink cycle")
            seen.add(current)
            current = link_targets[current][1]
        else:
            raise WorkerExecutionRefused("unsafe_symlink", "the granted source set contains a symlink chain")


def _porcelain_status(authority: ExecutionAuthority) -> list[dict[str, str]]:
    if authority.git is None:
        return []
    git = SyntheticGit(authority.workspace_path, authority.git)
    try:
        raw = git.run(["status", "--porcelain=v1", "-z", "--no-renames", "--untracked-files=normal"])
        if raw.returncode != 0:
            _refuse("git_unavailable", "Git status failed")
        if len(raw.stdout) > MAX_SCAN_STATUS_BYTES:
            _refuse("source_too_large", "Git status exceeds the bounded output")
        try:
            text = raw.stdout.decode("utf-8", "strict")
        except UnicodeDecodeError:
            _refuse("git_unsupported", "Git status contains a non-UTF-8 path")
        records: list[dict[str, str]] = []
        for item in text.split("\x00"):
            if not item:
                continue
            code, _, path = item[:2], item[2:3], item[3:]
            parts = path.split("/")
            if _reserved(parts):
                _refuse("reserved_source_paths", "Git status names a reserved path")
            records.append({"status": code, "path": path})
        return records
    finally:
        git.close()


# ---------------------------------------------------------------------------
# Inspection operations
# ---------------------------------------------------------------------------

_INSPECT_KEYS = {
    "summary": frozenset({"operation"}),
    "read": frozenset({"operation", "path", "start_line", "line_count", "expected_source_digest"}),
    "list": frozenset({"operation", "path", "offset", "limit", "expected_source_digest"}),
    "diff": frozenset({
        "operation", "base", "head", "mode", "path", "start_line", "line_count",
        "expected_source_digest", "expected_diff_digest",
    }),
}


def _bounded_int(value: Any, label: str, *, default: int, minimum: int, maximum: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _refuse("invalid_arguments", f"{label} must be an integer >= {minimum}")
    if value > maximum:
        _refuse("invalid_arguments", f"{label} exceeds {maximum}")
    return value


def _validate_inspect_args(args: Any) -> tuple[str, dict[str, Any]]:
    if not isinstance(args, Mapping):
        _refuse("invalid_arguments", "arguments must be a JSON object")
    operation = args.get("operation")
    if not isinstance(operation, str) or operation not in _INSPECT_KEYS:
        _refuse("invalid_arguments", "operation must be summary, read, list or diff")
    unknown = set(args) - _INSPECT_KEYS[operation]
    if unknown:
        _refuse("invalid_arguments", f"unknown argument(s): {sorted(unknown)}")
    normalized: dict[str, Any] = {"operation": operation}
    if operation == "summary":
        return operation, normalized
    if operation == "diff":
        has_pair = "base" in args or "head" in args
        has_mode = "mode" in args
        if has_pair == has_mode:
            _refuse("invalid_arguments", "diff requires either base/head or mode, not both")
        if has_pair:
            for key in ("base", "head"):
                value = args.get(key)
                if not isinstance(value, str) or not _OBJECT_ID_RE.match(value):
                    _refuse("invalid_arguments", f"{key} must be a full lowercase object id")
            normalized["base"] = args["base"]
            normalized["head"] = args["head"]
        else:
            mode = args.get("mode")
            if mode not in ("working", "staged"):
                _refuse("invalid_arguments", "mode must be working or staged")
            normalized["mode"] = mode
    if args.get("path") is not None:
        normalized["path"] = "/".join(_split_relative(args["path"]))
    if "start_line" in args:
        normalized["start_line"] = _bounded_int(
            args["start_line"], "start_line", default=1, minimum=1, maximum=10**9,
        )
    if "line_count" in args:
        normalized["line_count"] = _bounded_int(
            args["line_count"], "line_count", default=PAGE_DEFAULT_LINES, minimum=1, maximum=MAX_PAGE_LINES,
        )
    if "offset" in args:
        normalized["offset"] = _bounded_int(args["offset"], "offset", default=0, minimum=0, maximum=10**9)
    if "limit" in args:
        normalized["limit"] = _bounded_int(
            args["limit"], "limit", default=LIST_DEFAULT_LIMIT, minimum=1, maximum=MAX_PAGE_LINES,
        )
    for key in ("expected_source_digest", "expected_diff_digest"):
        if key in args:
            value = args[key]
            if not isinstance(value, str) or not _DIGEST_RE.match(value):
                _refuse("invalid_arguments", f"{key} must be sha256:<64 lowercase hex>")
            if key == "expected_diff_digest" and operation != "diff":
                _refuse("invalid_arguments", "expected_diff_digest is only valid for diff")
            normalized[key] = value
    if operation == "diff" and "expected_diff_digest" not in normalized:
        pass
    return operation, normalized


def _line_page(data: bytes, start_line: int, line_count: int) -> tuple[list[str], Optional[int]]:
    lines = data.splitlines(keepends=True)
    start = max(0, start_line - 1)
    window = lines[start:start + line_count]
    rendered: list[str] = []
    used = 0
    for line in window:
        if used + len(line) > MAX_PAGE_BYTES:
            if not rendered:
                _refuse("page_too_large", "a single line exceeds the page bound")
            break
        try:
            rendered.append(line.decode("utf-8", "strict"))
        except UnicodeDecodeError:
            _refuse("not_utf8", "requested source is not valid UTF-8")
        used += len(line)
    next_line = start + len(rendered) + 1 if start + len(rendered) < len(lines) else None
    return rendered, next_line


def _read_file_page(
    root: SafeRoot, parts: list[str], entry: SourceEntry, start_line: int, line_count: int,
) -> tuple[str, Optional[int]]:
    """One pass over the opened file: hash every byte AND collect the requested lines."""
    try:
        fd = root.open_file(parts)
    except OSError as exc:
        raise WorkerExecutionRefused("source_unreadable", _bounded(exc))
    digest = hashlib.sha256()
    wanted_start = max(0, start_line - 1)
    wanted_end = wanted_start + line_count
    page: list[bytes] = []
    page_bytes = 0
    completed_lines = 0
    pending = bytearray()
    last_chunk = b""
    binary = False
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or int(info.st_nlink) != 1:
            _refuse("unsupported_source_type", "requested source is not a single regular file")
        if info.st_size > MAX_INVENTORY_BYTES:
            _refuse("source_too_large", "requested source exceeds the byte bound")
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            last_chunk = chunk
            if not binary and b"\x00" in chunk:
                binary = True
            if completed_lines >= wanted_end:
                # Past the requested page: only the digest and the line count matter.
                completed_lines += chunk.count(b"\n")
                continue
            pending.extend(chunk)
            while True:
                newline = pending.find(b"\n")
                if newline < 0:
                    break
                line = bytes(pending[: newline + 1])
                del pending[: newline + 1]
                if wanted_start <= completed_lines < wanted_end:
                    page_bytes += len(line)
                    if page_bytes > MAX_PAGE_BYTES:
                        _refuse("page_too_large", "the requested page exceeds the byte bound")
                    page.append(line)
                completed_lines += 1
                if completed_lines >= wanted_end:
                    # Count the rest of this chunk, then stop buffering lines.
                    completed_lines += pending.count(b"\n")
                    pending.clear()
                    break
        if pending and wanted_start <= completed_lines < wanted_end:
            page_bytes += len(pending)
            if page_bytes > MAX_PAGE_BYTES:
                _refuse("page_too_large", "the requested page exceeds the byte bound")
            page.append(bytes(pending))
        total_lines = completed_lines + (1 if last_chunk and not last_chunk.endswith(b"\n") else 0)
        after = os.fstat(fd)
        if (after.st_size, after.st_mtime_ns, after.st_ino) != (info.st_size, info.st_mtime_ns, info.st_ino):
            raise WorkerExecutionRefused("source_changed", "requested source changed during inspection")
    finally:
        os.close(fd)
    if binary:
        raise WorkerExecutionRefused("binary_content", "requested source is binary")
    if "sha256:" + digest.hexdigest() != entry.digest:
        raise WorkerExecutionRefused("source_changed", "requested source changed during inspection")
    rendered: list[str] = []
    for line in page:
        try:
            rendered.append(line.decode("utf-8", "strict"))
        except UnicodeDecodeError:
            _refuse("not_utf8", "requested source is not valid UTF-8")
    consumed = wanted_start + len(rendered)
    next_line = consumed + 1 if consumed < total_lines else None
    return "".join(rendered), next_line


def _diff_preflight(git: SyntheticGit, scope: list[str], path: Optional[str]) -> None:
    argv = ["diff", "--raw", "-z", "--no-renames", *scope]
    if path:
        argv.extend(["--", path])
    result = git.run(argv)
    if result.returncode != 0:
        _refuse("git_unavailable", "Git diff preflight failed")
    items = result.stdout.split(b"\x00")
    index = 0
    while index < len(items):
        header = items[index]
        if not header:
            index += 1
            continue
        if not header.startswith(b":"):
            _refuse("git_unsupported", "Git diff preflight output is malformed")
        try:
            fields = header[1:].decode("ascii").split(" ")
        except UnicodeDecodeError:
            _refuse("git_unsupported", "Git diff preflight output is malformed")
        if len(fields) != 5:
            _refuse("git_unsupported", "Git diff preflight output is malformed")
        old_mode, new_mode, _old_oid, _new_oid, status_code = fields
        path_bytes = items[index + 1] if index + 1 < len(items) else b""
        index += 2
        try:
            name = path_bytes.decode("utf-8", "strict")
        except UnicodeDecodeError:
            _refuse("not_utf8", "the diff names a non-UTF-8 path")
        _validate_source_name(name)
        if _reserved(name.split("/")):
            _refuse("reserved_source_paths", "the diff touches a reserved path")
        letter = status_code[:1]
        if letter not in {"A", "M", "D", "T"}:
            _refuse("unsupported_diff", f"diff status {status_code!r} is not supported")
        for mode in (old_mode, new_mode):
            if mode == "160000":
                _refuse("submodule_source", "the diff touches a submodule gitlink")
            if mode not in {"000000", "100644", "100755", "120000"}:
                _refuse("unsupported_source_mode", f"diff mode {mode!r} is not supported")
        if len(result.stdout) > MAX_SCAN_STATUS_BYTES:
            _refuse("diff_too_large", "the diff preflight exceeds the bounded output")


def _diff_text(git: SyntheticGit, scope: list[str], path: Optional[str], target: Path) -> int:
    from hermes_cli._subprocess_compat import harden_git_argv

    objects = git.object_dir()
    env = _minimal_git_env(str(objects))
    argv = [
        _GIT,
        f"--git-dir={git.path}",
        f"--work-tree={git.workspace}",
        *harden_git_argv([
            "diff", "--no-color", "--unified=3", "--no-renames", "--ignore-submodules=all", *scope,
        ]),
    ]
    if path:
        argv.extend(["--", path])
    drained = 0
    with open(target, "xb") as sink:
        try:
            process = subprocess.Popen(  # noqa: S603 -- fixed executable, validated argv
                argv, cwd=str(git.root), env=env, stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            )
        except OSError as exc:
            raise WorkerExecutionRefused("git_unavailable", _bounded(exc))
        import select

        try:
            assert process.stdout is not None
            fd = process.stdout.fileno()
            os.set_blocking(fd, False)
            deadline = time.monotonic() + GIT_TIMEOUT_SECONDS
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    process.kill()
                    process.wait()
                    raise WorkerExecutionRefused("git_timeout", "the diff exceeded its time bound")
                ready, _, _ = select.select([fd], [], [], min(0.25, remaining))
                if not ready:
                    if process.poll() is not None:
                        # Drain whatever is buffered, then finish.
                        with contextlib.suppress(BlockingIOError):
                            chunk = os.read(fd, 1024 * 1024)
                            while chunk:
                                drained += len(chunk)
                                if drained > MAX_GIT_OUTPUT_BYTES:
                                    raise WorkerExecutionRefused("diff_too_large", "the diff exceeds the drained output bound")
                                sink.write(chunk)
                                chunk = os.read(fd, 1024 * 1024)
                        break
                    continue
                chunk = os.read(fd, 1024 * 1024)
                if not chunk:
                    break
                drained += len(chunk)
                if drained > MAX_GIT_OUTPUT_BYTES:
                    process.kill()
                    process.wait()
                    raise WorkerExecutionRefused("diff_too_large", "the diff exceeds the drained output bound")
                sink.write(chunk)
        finally:
            if process.poll() is None:
                process.kill()
            with contextlib.suppress(Exception):
                process.wait(timeout=GIT_TIMEOUT_SECONDS)
    if process.returncode != 0:
        _refuse("git_unavailable", "Git diff failed")
    return drained


def _inspect_summary(grant: Any, authority: ExecutionAuthority) -> dict[str, Any]:
    snapshot = _scan_source(authority)
    return {
        "ok": True,
        "operation": "summary",
        "task_id": grant.task_id,
        "run_id": grant.run_id,
        "profile": grant.profile,
        "preparation_id": grant.preparation_id,
        "policy_digest": authority.policy_digest,
        "git": {
            "head": snapshot.head,
            "branch": snapshot.branch,
            "detached": snapshot.detached,
            "status": _porcelain_status(authority),
        },
        "source_digest": snapshot.source_digest,
        "source_entries": len(snapshot.entries),
        "review": _review_summary(authority),
    }


def _review_summary(authority: ExecutionAuthority) -> Optional[dict[str, Any]]:
    review = authority.review
    if review is None:
        return None
    return {
        "base_sha": str(_record_value(review, "base", "")),
        "head_sha": str(_record_value(review, "head", "")),
    }


def _inspect_read(args: Mapping[str, Any], authority: ExecutionAuthority) -> dict[str, Any]:
    parts = _split_relative(args.get("path"), "path")
    if not parts:
        _refuse("invalid_arguments", "read requires a relative source path")
    if _reserved(parts):
        _refuse("reserved_source_paths", "read refuses a reserved path")
    snapshot = _scan_source(authority)
    name = "/".join(parts)
    entry = snapshot.entry(name)
    if entry is None:
        _refuse("path_not_in_source", "requested path is not in the granted source set")
    if entry.kind == "deleted":
        _refuse("path_deleted", "requested path is deleted in the granted source set")
    if entry.kind == "link":
        _refuse("path_not_regular", "requested path is a symlink")
    _check_expected(args, "expected_source_digest", snapshot.source_digest)
    start_line = int(args.get("start_line", 1))
    line_count = int(args.get("line_count", PAGE_DEFAULT_LINES))
    with SafeRoot(authority.workspace_path) as root:
        text, next_line = _read_file_page(root, parts, entry, start_line, line_count)
    return {
        "ok": True,
        "operation": "read",
        "path": name,
        "start_line": start_line,
        "next_line": next_line,
        "text": text,
        "head": snapshot.head,
        "source_digest": snapshot.source_digest,
    }


def _inspect_list(args: Mapping[str, Any], authority: ExecutionAuthority) -> dict[str, Any]:
    prefix_parts = _split_relative(args.get("path"), "path")
    if prefix_parts and _reserved(prefix_parts):
        _refuse("reserved_source_paths", "list refuses a reserved path")
    snapshot = _scan_source(authority)
    _check_expected(args, "expected_source_digest", snapshot.source_digest)
    prefix = "/".join(prefix_parts)
    visible = [
        entry for entry in snapshot.entries
        if not prefix or entry.path == prefix or entry.path.startswith(prefix + "/")
    ]
    visible.sort(key=lambda entry: entry.path)
    offset = int(args.get("offset", 0))
    limit = int(args.get("limit", LIST_DEFAULT_LIMIT))
    window = visible[offset:offset + limit]
    next_offset = offset + len(window) if offset + len(window) < len(visible) else None
    return {
        "ok": True,
        "operation": "list",
        "path": prefix,
        "entries": [{"path": entry.path, "type": entry.kind} for entry in window],
        "next_offset": next_offset,
        "total": len(visible),
        "head": snapshot.head,
        "source_digest": snapshot.source_digest,
    }


def _diff_scope(authority: ExecutionAuthority, args: Mapping[str, Any]) -> tuple[list[str], str]:
    role = str(getattr(authority, "role", "unknown"))
    if role == "review":
        if "mode" in args:
            _refuse("diff_not_granted", "commit-diff is the granted capability for a review claim")
    elif role == "implementation":
        if "mode" not in args:
            _refuse("diff_not_granted", "implementation claims have no historical commit-diff capability")
        return (["--cached"] if args["mode"] == "staged" else []), args["mode"]
    else:
        _refuse("diff_not_granted", "this claim has no diff capability")
    review = authority.review
    if review is None:
        _refuse("diff_not_granted", "no immutable candidate handoff is bound to this claim")
    base = str(_record_value(review, "base", ""))
    head = str(_record_value(review, "head", ""))
    if args.get("base", "").lower() != base.lower() or args.get("head", "").lower() != head.lower():
        _refuse("diff_not_granted", "only the granted review base/head pair may be diffed")
    return [base, head], "commit"


def _inspect_diff(args: Mapping[str, Any], authority: ExecutionAuthority) -> dict[str, Any]:
    scope, mode = _diff_scope(authority, args)
    path = args.get("path")
    if path is not None and _reserved(str(path).split("/")):
        _refuse("reserved_source_paths", "diff refuses a reserved path")
    if authority.git is None:
        _refuse("workspace_not_a_repository", "granted workspace is not a Git repository")
    snapshot = _scan_source(authority)
    if mode == "commit" and snapshot.head.lower() != str(_record_value(authority.review, "head", "")).lower():
        _refuse("source_changed", "the granted workspace HEAD is not the reviewed head")
    _check_expected(args, "expected_source_digest", snapshot.source_digest)
    git = SyntheticGit(authority.workspace_path, authority.git)
    patch_path = Path(tempfile.mkdtemp(prefix="worker-diff-", dir=_scratch_dir())) / "patch.diff"
    try:
        if mode == "commit":
            for commit in scope:
                typed = git.run(["cat-file", "-t", commit])
                if typed.returncode != 0 or typed.stdout.decode("ascii", "replace").strip() != "commit":
                    _refuse("diff_not_granted", "the granted review pair is not two commits")
        _diff_preflight(git, scope, path)
        # The inventory and HEAD identity must be stable across the two Git passes.
        _diff_text(git, scope, path, patch_path)
        after = _scan_source(authority)
        if (
            after.source_digest != snapshot.source_digest
            or after.head.lower() != snapshot.head.lower()
            or after.index_digest != snapshot.index_digest
        ):
            raise WorkerExecutionRefused("source_changed", "source or index changed during diff rendering")
        data = patch_path.read_bytes()
    finally:
        shutil.rmtree(patch_path.parent, ignore_errors=True)
        git.close()
    diff_digest = "sha256:" + sha256_hex(data)
    if "expected_diff_digest" in args and args["expected_diff_digest"] != diff_digest:
        raise WorkerExecutionRefused("source_changed", "the rendered diff does not match the expected digest")
    start_line = int(args.get("start_line", 1))
    line_count = int(args.get("line_count", PAGE_DEFAULT_LINES))
    rendered, next_line = _line_page(data, start_line, line_count)
    return {
        "ok": True,
        "operation": "diff",
        "mode": mode,
        "base": args.get("base"),
        "head": args.get("head"),
        "path": path,
        "patch": "".join(rendered),
        "next_line": next_line,
        "diff_digest": diff_digest,
        "head": snapshot.head,
        "source_digest": snapshot.source_digest,
    }


def _check_expected(args: Mapping[str, Any], key: str, actual: str) -> None:
    expected = args.get(key)
    if isinstance(expected, str) and expected != actual:
        raise WorkerExecutionRefused("source_changed", "source digest does not match the expected digest")



def _materialize_source(authority: ExecutionAuthority, run_root: Path) -> tuple[SourceSnapshot, Path]:
    source_dir = run_root / "source"
    source_dir.mkdir(mode=0o700)
    snapshot = _scan_source(authority, materialize=source_dir)
    return snapshot, source_dir



"""Shared primitives for granted worker execution: errors, digests, frozen authority.

The public entry points live in :mod:`hermes_cli.worker_execution`; policy parsing is
in :mod:`hermes_cli.worker_execution_policy` and source inspection in
:mod:`hermes_cli.worker_execution_source`.
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

#: The frozen plugin identity that must supply the mandatory pre-execution approval.
POLICY_PLUGIN_ID = "cto-consequence-guard"
POLICY_PLUGIN_MODULE = "__init__.py"
POLICY_PLUGIN_MANIFEST = "plugin.yaml"
REQUIRED_HOOK = "pre_worker_execution"
POLICY_PROTOCOL = 1
CONFIG_KEY = "worker_execution"

#: Implementation bounds. Deliberately constants: a second configuration surface would
#: let a task widen its own budget.
MAX_RECIPES = 32
MAX_SERVICES_PER_RECIPE = 4
MAX_POLICY_BYTES = 64 * 1024
MAX_ARGV_ELEMENTS = 64
MAX_ARGV_BYTES = 16 * 1024
MAX_TIMEOUT_SECONDS = 3600
MAX_PAGE_LINES = 1000
MAX_PAGE_BYTES = 256 * 1024
MAX_GIT_OUTPUT_BYTES = 256 * 1024 * 1024
MAX_INVENTORY_ENTRIES = 20_000
MAX_INVENTORY_BYTES = 256 * 1024 * 1024
GIT_TIMEOUT_SECONDS = 15.0
MAX_SCAN_STATUS_BYTES = 4 * 1024 * 1024
PAGE_DEFAULT_LINES = 200
LIST_DEFAULT_LIMIT = 200

#: Source-selection boundaries. This is not a promise to detect secrets committed under
#: arbitrary names: it refuses the structural places where live state and credentials
#: of this installation live.
RESERVED_COMPONENTS = frozenset({".git", ".hermes", ".ssh", ".aws", ".gnupg"})
RESERVED_BASENAMES = frozenset({".env"})
RESERVED_BASENAME_PREFIX = ".env."
RESERVED_BASENAME_ALLOWED = frozenset({".env.example", ".env.sample"})

_SERVICE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")
_IMAGE_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*(?::[0-9]+)?/[a-z0-9][a-z0-9._/-]*@sha256:[0-9a-f]{64}$")
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_OBJECT_ID_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")

_GIT = "/usr/bin/git"
_VERIFY_LOCK = threading.Lock()

class WorkerExecutionError(RuntimeError):
    """A typed refusal. ``code`` is stable and machine-readable."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class WorkerExecutionUnavailable(WorkerExecutionError):
    """This process holds no usable execution authority (a refusal, never emptiness)."""


class WorkerExecutionRefused(WorkerExecutionError):
    """Authority exists but the requested operation is not permitted, or inputs are unsafe."""


def _refuse(code: str, message: str) -> None:
    raise WorkerExecutionRefused(code, message)


def _bounded(text: Any, limit: int = 300) -> str:
    return " ".join(str(text).split())[:limit]


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False,
    ).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_of(value: Any) -> str:
    return "sha256:" + sha256_hex(canonical_bytes(value))

@dataclass(frozen=True)
class ExecutionGrant:
    """The dispatcher-prepared authority section, before run/claim binding."""

    policy_digest: str
    task_id: str
    profile: str
    board: str
    preparation_id: str
    workspace: dict[str, Any]
    git: Optional[dict[str, Any]]
    image_store: Optional[dict[str, Any]]
    inspect: bool
    role: str
    extra_source_paths: tuple[str, ...]
    recipes: tuple[dict[str, Any], ...]
    review: Optional[dict[str, str]]
    policy_plugin: dict[str, Any]
    runtime: dict[str, str]
    execution_parent: dict[str, Any]
    journal_dir: dict[str, Any]
    runtime_parent: dict[str, Any]

    def _body(self) -> dict[str, Any]:
        return {
            "protocol": 1,
            "policy_digest": self.policy_digest,
            "task_id": self.task_id,
            "profile": self.profile,
            "board": self.board,
            "preparation_id": self.preparation_id,
            "workspace": self.workspace,
            "git": self.git,
            "image_store": self.image_store,
            "inspect": self.inspect,
            "role": self.role,
            "extra_source_paths": list(self.extra_source_paths),
            "recipes": [dict(recipe) for recipe in self.recipes],
            "review": self.review,
            "policy_plugin": self.policy_plugin,
            "runtime": self.runtime,
            "execution_parent": self.execution_parent,
            "journal_dir": self.journal_dir,
            "runtime_parent": self.runtime_parent,
        }

    def bind(self, run_id: int, claim_lock: Optional[str]) -> dict[str, Any]:
        """Add the actual claim and the integrity digest at grant time."""
        body = self._body()
        body["run_id"] = int(run_id)
        body["claim_lock"] = str(claim_lock or "")
        body["digest"] = digest_of(body)
        return body


class FrozenDict(Mapping):
    """An immutable mapping that still supports ``Mapping`` reads (``get``, iteration).

    The grant is deep-frozen when retained: lists become tuples and mappings become
    this read-only view. Callers read it exactly like the JSON section they came from,
    but no later mutation can change retained authority.
    """

    __slots__ = ("_data",)

    def __init__(self, pairs: Any = ()) -> None:
        object.__setattr__(self, "_data", {str(key): _freeze_value(item) for key, item in dict(pairs).items()})

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"FrozenDict({self._data!r})"

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Mapping):
            return dict(self._data) == {str(k): v for k, v in other.items()}
        return NotImplemented

    def __hash__(self) -> int:
        return hash(tuple(sorted((key, repr(value)) for key, value in self._data.items())))


def _freeze_mapping(value: Any) -> Optional[FrozenDict]:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise WorkerExecutionUnavailable("worker_execution_malformed", "worker execution grant section is malformed")
    return FrozenDict(value.items())


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return FrozenDict(value.items())
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise WorkerExecutionUnavailable("worker_execution_malformed", "worker execution grant value is malformed")


def _freeze_recipes(value: Any) -> tuple[FrozenDict, ...]:
    if not isinstance(value, (list, tuple)):
        raise WorkerExecutionUnavailable("worker_execution_malformed", "worker execution recipes are malformed")
    frozen = []
    for recipe in value:
        if not isinstance(recipe, Mapping):
            raise WorkerExecutionUnavailable("worker_execution_malformed", "worker execution recipe is malformed")
        frozen.append(FrozenDict(recipe.items()))
    return tuple(frozen)


@dataclass(frozen=True)
class ExecutionAuthority:
    """Frozen authority retained by this worker process."""

    policy_digest: str
    task_id: str
    run_id: int
    preparation_id: str
    workspace_path: str
    workspace_device: int
    workspace_inode: int
    git: Optional[dict[str, Any]]
    image_store: Optional[dict[str, Any]]
    inspect: bool
    role: str
    extra_source_paths: tuple[str, ...]
    recipes: tuple[dict[str, Any], ...]
    review: Optional[dict[str, Any]]
    policy_plugin: dict[str, Any]
    runtime: dict[str, Any]
    execution_parent: dict[str, Any]
    journal_dir: dict[str, Any]
    runtime_parent: dict[str, Any]
    runtime_identity: Any
    bound_digest: str

    def verify_static_identities(self) -> None:
        """Re-check every bound filesystem identity before any read or launch."""
        if not sys.platform.startswith("linux"):
            raise WorkerExecutionUnavailable("worker_execution_unsupported", "worker execution is Linux-only")
        path = Path(self.workspace_path)
        try:
            info = os.lstat(path)
        except OSError as exc:
            raise WorkerExecutionUnavailable("workspace_changed", f"granted workspace is unavailable: {_bounded(exc)}")
        if (
            not stat.S_ISDIR(info.st_mode) or path.is_symlink()
            or int(info.st_dev) != self.workspace_device or int(info.st_ino) != self.workspace_inode
        ):
            raise WorkerExecutionUnavailable("workspace_changed", "granted workspace identity changed")
        if self.git is not None:
            for key, device_key, inode_key in (
                ("git_dir", "git_device", "git_inode"),
                ("common_dir", "common_device", "common_inode"),
            ):
                directory = Path(str(self.git.get(key, "")))
                try:
                    info = os.lstat(directory)
                except OSError as exc:
                    raise WorkerExecutionUnavailable("git_changed", f"granted Git directory is unavailable: {_bounded(exc)}")
                if (
                    not stat.S_ISDIR(info.st_mode) or directory.is_symlink()
                    or int(info.st_dev) != int(self.git.get(device_key, -1))
                    or int(info.st_ino) != int(self.git.get(inode_key, -1))
                ):
                    raise WorkerExecutionUnavailable("git_changed", "granted Git directory identity changed")
        for record, label in (
            (self.execution_parent, "execution parent"),
            (self.journal_dir, "journal directory"),
            (self.runtime_parent, "runtime parent"),
        ):
            if record is None:
                raise WorkerExecutionUnavailable("worker_execution_malformed", f"{label} is missing from the grant")
            directory = Path(str(_record_value(record, "path", "")))
            try:
                info = os.lstat(directory)
            except OSError as exc:
                raise WorkerExecutionUnavailable("execution_directory_changed", f"{label} is unavailable: {_bounded(exc)}")
            if (
                not stat.S_ISDIR(info.st_mode) or directory.is_symlink()
                or int(info.st_dev) != int(_record_value(record, "device", -1))
                or int(info.st_ino) != int(_record_value(record, "inode", -1))
            ):
                raise WorkerExecutionUnavailable("execution_directory_changed", f"{label} identity changed")
        if self.image_store is not None:
            store = Path(str(_record_value(self.image_store, "path", "")))
            try:
                info = os.lstat(store)
            except OSError as exc:
                raise WorkerExecutionUnavailable("image_store_changed", f"image store is unavailable: {_bounded(exc)}")
            if (
                not stat.S_ISDIR(info.st_mode) or store.is_symlink()
                or int(info.st_dev) != int(_record_value(self.image_store, "device", -1))
                or int(info.st_ino) != int(_record_value(self.image_store, "inode", -1))
            ):
                raise WorkerExecutionUnavailable("image_store_changed", "image store identity changed")

    def recipe(self, recipe_id: str) -> dict[str, Any]:
        for recipe in self.recipes:
            if recipe.get("id") == recipe_id:
                return recipe
        raise WorkerExecutionRefused("recipe_not_granted", "recipe is not in the granted policy")

    def recipe_ids(self) -> tuple[str, ...]:
        return tuple(str(recipe.get("id")) for recipe in self.recipes)


def _record_value(record: Mapping[str, Any], key: str, default: Any = None) -> Any:
    """Read a frozen mapping value (tuples of key/value pairs) or a plain mapping."""
    if isinstance(record, Mapping):
        return record.get(key, default)
    for entry in record:
        if isinstance(entry, (tuple, list)) and len(entry) == 2 and entry[0] == key:
            return entry[1]
    return default


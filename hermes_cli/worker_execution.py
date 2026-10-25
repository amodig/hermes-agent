"""Explicit, operator-granted worker source inspection and disposable verification.

Authority for both tools is the dispatcher's final runtime grant, retained in-process
by :func:`hermes_cli.worker_startup._retain_worker_grant`.  The dispatcher resolves the
operator's ``kanban.worker_execution`` policy through its own profile config and binds
the resolved workspace, Git identity, image store, review handoff and policy plugin
into that grant; a worker can select an already-granted recipe, never construct one.

Two capabilities live here:

* ``kanban_worker_inspect`` — typed, read-only source/Git inspection that never runs
  repository code (no hooks, filters, textconv drivers, fsmonitor, pager or network).
* ``kanban_worker_verify`` — one explicitly granted recipe executed by
  :mod:`hermes_cli.worker_execution_podman` inside a fresh rootless private stack.

Startup metadata (``kanban_worker_startup``) is unchanged; an absent execution section
leaves it usable and these two capabilities unavailable.
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


# ---------------------------------------------------------------------------
# Policy schema (strict; no unknown keys anywhere)
# ---------------------------------------------------------------------------

_POLICY_KEYS = frozenset({"protocol", "image_store", "workspaces"})
_WORKSPACE_KEYS = frozenset({"path", "inspect", "extra_source_paths", "recipes"})
_RECIPE_KEYS = frozenset({"image", "argv", "timeout_seconds", "services"})
_SERVICE_KEYS = frozenset({"id", "image", "argv", "ready_argv", "port"})


@dataclass(frozen=True)
class ServicePolicy:
    id: str
    image: str
    argv: tuple[str, ...]
    ready_argv: tuple[str, ...]
    port: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "image": self.image, "argv": list(self.argv),
            "ready_argv": list(self.ready_argv), "port": self.port,
        }


@dataclass(frozen=True)
class RecipePolicy:
    id: str
    image: str
    argv: tuple[str, ...]
    timeout_seconds: int
    services: tuple[ServicePolicy, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "image": self.image, "argv": list(self.argv),
            "timeout_seconds": self.timeout_seconds,
            "services": [service.as_dict() for service in self.services],
        }


@dataclass(frozen=True)
class WorkspacePolicy:
    path: str
    inspect: bool
    extra_source_paths: tuple[str, ...]
    recipes: tuple[RecipePolicy, ...]


@dataclass(frozen=True)
class WorkerExecutionPolicy:
    image_store: Optional[str]
    workspaces: tuple[WorkspacePolicy, ...]
    digest: str

    def recipes(self) -> dict[str, RecipePolicy]:
        return {recipe.id: recipe for workspace in self.workspaces for recipe in workspace.recipes}


def _strict_keys(value: Mapping[str, Any], expected: frozenset, label: str) -> None:
    if not isinstance(value, Mapping):
        _refuse("policy_malformed", f"{label} must be a mapping")
    extra = set(value) - expected
    missing = expected - set(value)
    if extra:
        _refuse("policy_malformed", f"{label} has unknown keys: {sorted(extra)}")
    if missing:
        _refuse("policy_malformed", f"{label} is missing keys: {sorted(missing)}")


def _policy_argv(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        _refuse("policy_malformed", f"{label} must be a nonempty list")
    if len(value) > MAX_ARGV_ELEMENTS:
        _refuse("policy_malformed", f"{label} has too many elements")
    argv: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item or "\x00" in item:
            _refuse("policy_malformed", f"{label} elements must be nonempty strings")
        argv.append(item)
    if sum(len(item.encode("utf-8")) for item in argv) > MAX_ARGV_BYTES:
        _refuse("policy_malformed", f"{label} exceeds the argument byte bound")
    if not argv[0].startswith("/"):
        _refuse("policy_malformed", f"{label}[0] must be an absolute in-image executable")
    return tuple(argv)


def _policy_image(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _IMAGE_RE.match(value):
        _refuse("policy_malformed", f"{label} must be a fully qualified digest reference")
    return value


def _policy_relative_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _refuse("policy_malformed", f"{label} must be a nonempty relative path")
    text = value.strip()
    if os.path.isabs(text) or "\\" in text or "\x00" in text:
        _refuse("policy_malformed", f"{label} must be a relative POSIX path")
    parts = text.split("/")
    if any(part in ("", ".", "..") for part in parts):
        _refuse("policy_malformed", f"{label} must not contain empty, '.' or '..' components")
    return text


def parse_policy(section: Any) -> WorkerExecutionPolicy:
    """Validate one ``kanban.worker_execution`` section into a frozen policy."""
    _strict_keys(section, _POLICY_KEYS, "kanban.worker_execution")
    protocol = section["protocol"]
    if isinstance(protocol, bool) or protocol != POLICY_PROTOCOL:
        _refuse("policy_malformed", f"worker_execution protocol must be {POLICY_PROTOCOL}")
    image_store = section["image_store"]
    if image_store is not None:
        if not isinstance(image_store, str) or not os.path.isabs(image_store):
            _refuse("policy_malformed", "image_store must be null or an absolute path")
        image_store = os.path.normpath(image_store)
    workspaces_raw = section["workspaces"]
    if not isinstance(workspaces_raw, list):
        _refuse("policy_malformed", "workspaces must be a list")
    seen_paths: set[str] = set()
    seen_recipe_ids: set[str] = set()
    workspaces: list[WorkspacePolicy] = []
    for index, entry in enumerate(workspaces_raw):
        label = f"workspaces[{index}]"
        _strict_keys(entry, _WORKSPACE_KEYS, label)
        raw_path = entry["path"]
        if not isinstance(raw_path, str) or not os.path.isabs(raw_path):
            _refuse("policy_malformed", f"{label}.path must be an absolute path")
        path = os.path.normpath(raw_path)
        if path in seen_paths:
            _refuse("policy_malformed", f"{label}.path is duplicated")
        seen_paths.add(path)
        inspect = entry["inspect"]
        if not isinstance(inspect, bool):
            _refuse("policy_malformed", f"{label}.inspect must be a boolean")
        extras_raw = entry["extra_source_paths"]
        if not isinstance(extras_raw, list):
            _refuse("policy_malformed", f"{label}.extra_source_paths must be a list")
        extras = tuple(_policy_relative_path(item, f"{label}.extra_source_paths") for item in extras_raw)
        if len(set(extras)) != len(extras):
            _refuse("policy_malformed", f"{label}.extra_source_paths contains duplicates")
        recipes_raw = entry["recipes"]
        if not isinstance(recipes_raw, Mapping):
            _refuse("policy_malformed", f"{label}.recipes must be a mapping")
        recipes: list[RecipePolicy] = []
        for recipe_id, recipe in recipes_raw.items():
            if not isinstance(recipe_id, str) or not _SERVICE_ID_RE.match(recipe_id):
                _refuse("policy_malformed", f"{label}.recipes key {recipe_id!r} is not a valid id")
            if recipe_id in seen_recipe_ids:
                total = len(seen_recipe_ids)
                if total >= MAX_RECIPES:
                    _refuse("policy_malformed", "too many recipes")
            _strict_keys(recipe, _RECIPE_KEYS, f"{label}.recipes.{recipe_id}")
            image = _policy_image(recipe["image"], f"{label}.recipes.{recipe_id}.image")
            argv = _policy_argv(recipe["argv"], f"{label}.recipes.{recipe_id}.argv")
            timeout = recipe["timeout_seconds"]
            if isinstance(timeout, bool) or not isinstance(timeout, int) or not 1 <= timeout <= MAX_TIMEOUT_SECONDS:
                _refuse("policy_malformed", f"{label}.recipes.{recipe_id}.timeout_seconds is out of range")
            services_raw = recipe["services"]
            if not isinstance(services_raw, list) or len(services_raw) > MAX_SERVICES_PER_RECIPE:
                _refuse("policy_malformed", f"{label}.recipes.{recipe_id}.services is invalid")
            services: list[ServicePolicy] = []
            ids: set[str] = set()
            ports: set[int] = set()
            normalized: set[str] = set()
            for service_index, service in enumerate(services_raw):
                slabel = f"{label}.recipes.{recipe_id}.services[{service_index}]"
                _strict_keys(service, _SERVICE_KEYS, slabel)
                service_id = service["id"]
                if not isinstance(service_id, str) or not _SERVICE_ID_RE.match(service_id):
                    _refuse("policy_malformed", f"{slabel}.id is invalid")
                if service_id in ids:
                    _refuse("policy_malformed", f"{slabel}.id is duplicated")
                ids.add(service_id)
                folded = service_id.upper().replace("-", "_")
                if folded in normalized:
                    _refuse("policy_malformed", f"{slabel}.id normalizes onto another service")
                normalized.add(folded)
                s_image = _policy_image(service["image"], f"{slabel}.image")
                s_argv = _policy_argv(service["argv"], f"{slabel}.argv")
                s_ready = _policy_argv(service["ready_argv"], f"{slabel}.ready_argv")
                port = service["port"]
                if isinstance(port, bool) or not isinstance(port, int) or not 1024 <= port <= 65535:
                    _refuse("policy_malformed", f"{slabel}.port is out of range")
                if port in ports:
                    _refuse("policy_malformed", f"{slabel}.port is duplicated")
                ports.add(port)
                services.append(ServicePolicy(service_id, s_image, s_argv, s_ready, port))
            recipes.append(RecipePolicy(recipe_id, image, argv, int(timeout), tuple(services)))
            seen_recipe_ids.add(recipe_id)
        if len(seen_recipe_ids) > MAX_RECIPES:
            _refuse("policy_malformed", "too many recipes")
        workspaces.append(WorkspacePolicy(path, inspect, extras, tuple(recipes)))
    if image_store is None and any(workspace.recipes for workspace in workspaces):
        _refuse("policy_malformed", "image_store is required when recipes are configured")
    digest = digest_of(section)
    encoded = canonical_bytes(section)
    if len(encoded) > MAX_POLICY_BYTES:
        _refuse("policy_malformed", "worker_execution policy exceeds the serialized bound")
    return WorkerExecutionPolicy(image_store, tuple(workspaces), digest)


def load_policy() -> Optional[WorkerExecutionPolicy]:
    """Read ``kanban.worker_execution`` from the dispatcher's own process home.

    The override is pinned to :func:`get_process_hermes_home` so an active profile
    scope (or an inherited environment value the worker will later carry) can never
    substitute another home's policy. Only the dispatcher process reads this; the
    worker consumes the frozen grant.
    """
    from hermes_cli.config import load_config_readonly
    from hermes_constants import get_process_hermes_home, reset_hermes_home_override, set_hermes_home_override

    token = set_hermes_home_override(get_process_hermes_home())
    try:
        config = load_config_readonly()
    except Exception as exc:  # a broken config must grant nothing
        raise WorkerExecutionUnavailable("policy_malformed", f"worker execution config is unavailable: {_bounded(exc)}")
    finally:
        reset_hermes_home_override(token)
    kanban = config.get("kanban") if isinstance(config, Mapping) else None
    if not isinstance(kanban, Mapping) or CONFIG_KEY not in kanban:
        return None
    section = kanban[CONFIG_KEY]
    if section is None:
        return None
    return parse_policy(section)


# ---------------------------------------------------------------------------
# Dispatcher-side resolution
# ---------------------------------------------------------------------------


def capture_git_identity(workspace: str) -> Optional[dict[str, Any]]:
    """Resolve the workspace's Git directories with fixed argv and a minimal environment.

    Called only by the dispatcher before the final grant. Returns ``None`` when the
    workspace is not a Git worktree; inspection then reports an explicit unavailability.
    """
    if not Path(_GIT).is_file():
        return None
    try:
        output = _run_git_probe(["-C", workspace, "rev-parse", "--absolute-git-dir", "--git-common-dir"])
    except WorkerExecutionError:
        return None
    lines = [line for line in output.splitlines() if line.strip()]
    if len(lines) != 2:
        return None
    git_dir = Path(lines[0].strip())
    common = Path(lines[1].strip())
    if not common.is_absolute():
        common = Path(workspace) / common
    try:
        git_dir = git_dir.resolve(strict=True)
        common = common.resolve(strict=True)
        git_stat = os.lstat(git_dir)
        common_stat = os.lstat(common)
    except OSError:
        return None
    if not stat.S_ISDIR(git_stat.st_mode) or not stat.S_ISDIR(common_stat.st_mode):
        return None
    if Path(git_dir).is_symlink() or Path(common).is_symlink():
        return None
    return {
        "git_dir": str(git_dir),
        "common_dir": str(common),
        "git_device": int(git_stat.st_dev),
        "git_inode": int(git_stat.st_ino),
        "common_device": int(common_stat.st_dev),
        "common_inode": int(common_stat.st_ino),
    }


def _run_git_probe(args: list[str]) -> str:
    env = _minimal_git_env(None)
    try:
        result = subprocess.run(  # noqa: S603 -- fixed absolute executable
            [_GIT, *args], cwd="/", env=env, stdin=subprocess.DEVNULL,
            capture_output=True, text=True, timeout=GIT_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise WorkerExecutionRefused("git_unavailable", f"git probe failed: {_bounded(exc)}")
    if result.returncode != 0:
        raise WorkerExecutionRefused("git_unavailable", "git probe failed")
    return result.stdout


def _minimal_git_env(object_dir: Optional[str]) -> dict[str, str]:
    from hermes_cli._subprocess_compat import noninteractive_git_env

    base = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "HOME": "/nonexistent",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "TZ": "UTC",
    }
    env = noninteractive_git_env(base)
    env.update({
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_NO_LAZY_FETCH": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_ALLOW_PROTOCOL": "never",
        "GIT_PROTOCOL_FROM_USER": "0",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_PAGER": "cat",
        "GIT_EDITOR": "true",
        "GIT_ASKPASS": "/bin/true",
        "SSH_ASKPASS": "/bin/true",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_SYSTEM": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_ATTR_NOSYSTEM": "1",
        "GIT_CEILING_DIRECTORIES": "/",
    })
    if object_dir:
        env["GIT_OBJECT_DIRECTORY"] = object_dir
    return env


def _resolve_workspace(policy_entry: WorkspacePolicy) -> Optional[dict[str, Any]]:
    """Exact canonical path plus held directory identity, never a prefix or label."""
    try:
        resolved = Path(policy_entry.path).resolve(strict=True)
        info = os.lstat(resolved)
    except OSError:
        return None
    if not stat.S_ISDIR(info.st_mode) or Path(policy_entry.path).is_symlink():
        return None
    if str(resolved) != policy_entry.path:
        return None
    return {"path": str(resolved), "device": int(info.st_dev), "inode": int(info.st_ino)}


def _resolve_image_store(policy: WorkerExecutionPolicy) -> Optional[dict[str, Any]]:
    if policy.image_store is None:
        return None
    path = Path(policy.image_store)
    try:
        info = os.lstat(path)
    except OSError:
        _refuse("policy_malformed", "image_store does not exist")
    if path.is_symlink() or not stat.S_ISDIR(info.st_mode):
        _refuse("policy_malformed", "image_store must be a real directory")
    if info.st_mode & 0o077:
        _refuse("policy_malformed", "image_store must be owner-only")
    if int(info.st_uid) != os.getuid():  # windows-footgun: ok — POSIX-only ownership check
        _refuse("policy_malformed", "image_store must be owned by this user")
    inventory = path / "inventory.json"
    try:
        inv_stat = os.lstat(inventory)
    except OSError:
        if policy.recipes():
            _refuse("policy_malformed", "image_store has no inventory.json")
        inv_digest = None
    else:
        if not stat.S_ISREG(inv_stat.st_mode) or inventory.is_symlink():
            _refuse("policy_malformed", "image_store inventory.json must be a regular file")
        if inv_stat.st_size > 1024 * 1024:
            _refuse("policy_malformed", "image_store inventory.json is too large")
        with open(inventory, "rb") as handle:
            inv_digest = "sha256:" + sha256_hex(handle.read())
    for archive in sorted(path.glob("*.oci.tar")):
        if archive.is_symlink():
            _refuse("policy_malformed", "image_store must not contain symlinked archives")
    return {
        "path": str(path),
        "device": int(info.st_dev),
        "inode": int(info.st_ino),
        "inventory_digest": inv_digest,
    }


def _live_state_roots(profile_home: Optional[str], db_path: Optional[str]) -> list[Path]:
    roots: list[Path] = []
    for candidate in (profile_home, db_path):
        if isinstance(candidate, str) and candidate:
            roots.append(Path(candidate).resolve().parent if candidate.endswith(".db") else Path(candidate).resolve())
    with contextlib.suppress(Exception):
        from hermes_cli import kanban_db

        roots.append(Path(kanban_db.kanban_home()).resolve())
    return roots


def _policy_plugin_digests(profile_home: Optional[str]) -> Optional[dict[str, Any]]:
    if not profile_home:
        return None
    plugins_root = Path(profile_home) / "plugins"
    if not plugins_root.is_dir() or plugins_root.is_symlink():
        return None
    for directory in sorted(plugins_root.iterdir()):
        manifest = directory / POLICY_PLUGIN_MANIFEST
        module = directory / POLICY_PLUGIN_MODULE
        try:
            if directory.is_symlink() or not manifest.is_file() or not module.is_file():
                continue
            text = manifest.read_text(encoding="utf-8-sig", errors="strict")
        except OSError:
            return None
        if not re.search(r"^name:\s*" + re.escape(POLICY_PLUGIN_ID) + r"\s*$", text, re.MULTILINE):
            continue
        module_bytes = module.read_bytes()
        manifest_bytes = manifest.read_bytes()
        return {
            "id": POLICY_PLUGIN_ID,
            "module_sha256": sha256_hex(module_bytes),
            "manifest_sha256": sha256_hex(manifest_bytes),
        }
    return None


def resolve_review_context(db_path: str, task_id: str) -> tuple[str, Optional[dict[str, str]]]:
    """``(role, review_pair)`` for the claimed task.

    ``role`` is ``review`` when the card is being reviewed (separate-card role cards
    and same-card reviews, whose contract stays ``kind: code``), otherwise
    ``implementation``. The pair comes only from the native immutable handoff of the
    reviewed candidate — for a same-card review the candidate is the card itself. An
    unreadable board yields ``unknown``: metadata startup still works, commit diffs
    and working diffs are both refused.
    """
    from hermes_cli import kanban_db
    from hermes_cli.worker_startup import _open_board

    try:
        connection = _open_board(db_path)
    except Exception:
        return "unknown", None
    try:
        row = connection.execute(
            "SELECT status, lifecycle_contract FROM tasks WHERE id = ?", (task_id,),
        ).fetchone()
        if row is None:
            return "unknown", None
        status = str(row["status"] or "")
        raw = row["lifecycle_contract"]
        contract: dict[str, Any] = {}
        if raw:
            try:
                contract = json.loads(raw) if isinstance(raw, str) else dict(raw)
            except (TypeError, ValueError):
                return ("review" if status == "review" else "unknown"), None
        if not isinstance(contract, Mapping):
            return ("review" if status == "review" else "unknown"), None
        kind = contract.get("kind")
        # A typed role card (review or validation, which dispatches through the ready
        # lane) is a review claim; a same-card review keeps `kind: code` and is
        # recognized by its review status.
        if kind in ("review", "validation"):
            role = "review"
            candidate_id = contract.get("candidate_task_id")
        elif status == "review":
            role = "review"
            candidate_id = task_id
        else:
            return "implementation", None
        if not isinstance(candidate_id, str) or not candidate_id:
            return role, None
        handoff = kanban_db.latest_handoff(connection, candidate_id)
        base = str(handoff.get("base_sha") or "").strip()
        head = str(handoff.get("head_sha") or "").strip()
        if not _OBJECT_ID_RE.match(base) or not _OBJECT_ID_RE.match(head):
            return role, None
        return role, {"base": base, "head": head}
    except Exception:
        return "unknown", None
    finally:
        with contextlib.suppress(Exception):
            connection.execute("ROLLBACK")
        connection.close()


def resolve_execution_grant(
    *,
    profile_home: Optional[str],
    workspace: str,
    task_id: str,
    profile: str,
    board: str,
    db_path: str,
    preparation_id: str,
    runtime_identity: Any,
) -> tuple[Optional["ExecutionGrant"], Optional[str]]:
    """Resolve and validate configured authority; ``(None, reason)`` grants nothing."""
    try:
        policy = load_policy()
    except WorkerExecutionError as exc:
        return None, f"{exc.code}: {exc.message}"
    if policy is None:
        return None, None
    canonical_workspace = os.path.normpath(workspace)
    matches = []
    for entry in policy.workspaces:
        if entry.path != canonical_workspace and os.path.realpath(entry.path) != os.path.realpath(canonical_workspace):
            continue
        resolved = _resolve_workspace(entry)
        if resolved is None:
            continue
        matches.append((entry, resolved))
    if not matches:
        return None, "workspace_not_granted"
    if len(matches) > 1:
        return None, "workspace_ambiguous"
    entry, resolved_workspace = matches[0]
    try:
        image_store = _resolve_image_store(policy)
        git = capture_git_identity(resolved_workspace["path"])
    except WorkerExecutionError as exc:
        return None, f"{exc.code}: {exc.message}"
    # The image store must live outside the workspace, profile and live state roots.
    if image_store is not None:
        store = Path(image_store["path"])
        forbidden = [Path(resolved_workspace["path"]).resolve()]
        if profile_home:
            forbidden.append(Path(profile_home).resolve())
        forbidden.extend(_live_state_roots(profile_home, db_path))
        for root in forbidden:
            if store == root or store.is_relative_to(root):
                return None, "image_store_inside_protected_root"
    plugin = _policy_plugin_digests(profile_home)
    if plugin is None:
        return None, "policy_plugin_unavailable"
    role, review = resolve_review_context(db_path, task_id)
    if review is not None and git is not None:
        head = _workspace_head_from_identity(git)
        if head is not None and head.lower() != review["head"].lower():
            # The reviewer is entitled to metadata but not to a stale commit-diff pair.
            review = None
    from hermes_constants import get_process_hermes_home

    parent = Path(tempfile.gettempdir()) / "hermes-worker-execution"
    try:
        parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(parent, 0o700)
        run_parent = parent / preparation_id
        run_parent.mkdir(mode=0o700, exist_ok=True)
        journal_dir = get_process_hermes_home() / "worker-executions" / preparation_id
        journal_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        parent_info = os.lstat(parent)
        run_parent_info = os.lstat(run_parent)
        journal_info = os.lstat(journal_dir)
    except OSError as exc:
        return None, f"execution_directories_unavailable: {_bounded(exc)}"
    for path in (parent, run_parent, journal_dir):
        if path.is_symlink() or os.lstat(path).st_mode & 0o077:
            return None, "execution_directory_insecure"
    identity = runtime_identity
    grant = ExecutionGrant(
        policy_digest=policy.digest,
        task_id=task_id,
        profile=profile,
        board=board,
        preparation_id=preparation_id,
        workspace=resolved_workspace,
        git=git,
        image_store=image_store,
        inspect=entry.inspect,
        role=role,
        extra_source_paths=tuple(entry.extra_source_paths),
        recipes=tuple({"id": recipe.id, **recipe.as_dict()} for recipe in entry.recipes),
        review=review,
        policy_plugin=plugin,
        runtime={
            "generation": str(getattr(identity, "generation", "") or ""),
            "dependency_fingerprint": str(getattr(identity, "dependency_fingerprint", "") or ""),
            "code_sha": str(getattr(identity, "code_sha", "") or ""),
        },
        execution_parent={
            "path": str(run_parent), "device": int(run_parent_info.st_dev),
            "inode": int(run_parent_info.st_ino),
        },
        journal_dir={
            "path": str(journal_dir), "device": int(journal_info.st_dev),
            "inode": int(journal_info.st_ino),
        },
        runtime_parent={
            "path": str(parent), "device": int(parent_info.st_dev),
            "inode": int(parent_info.st_ino),
        },
    )
    return grant, None


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


def _workspace_head_from_identity(git: Mapping[str, Any]) -> Optional[str]:
    """Read HEAD through the captured Git directory (no repository code execution)."""
    head_path = Path(str(git.get("git_dir", ""))) / "HEAD"
    try:
        info = os.lstat(head_path)
        if not stat.S_ISREG(info.st_mode) or info.st_size > 4096:
            return None
        raw = head_path.read_text(encoding="utf-8-sig", errors="strict").strip()
    except OSError:
        return None
    if not raw.startswith("ref:"):
        return raw if _OBJECT_ID_RE.match(raw) else None
    ref = raw[4:].strip()
    for base in (Path(str(git.get("git_dir", ""))), Path(str(git.get("common_dir", "")))):
        candidate = base / ref
        try:
            if candidate.is_file() and not candidate.is_symlink():
                value = candidate.read_text(encoding="utf-8-sig", errors="strict").strip()
                if _OBJECT_ID_RE.match(value):
                    return value
        except OSError:
            continue
        packed = base / "packed-refs"
        try:
            if packed.is_file() and not packed.is_symlink():
                for line in packed.read_text(encoding="utf-8-sig", errors="strict").splitlines():
                    if line.startswith(("#", "^")):
                        continue
                    fields = line.split(" ", 1)
                    if len(fields) == 2 and fields[1].strip() == ref and _OBJECT_ID_RE.match(fields[0]):
                        return fields[0]
        except OSError:
            continue
    return None


def parse_grant_section(section: Any, grant: Any) -> "ExecutionAuthority":
    """Deep-freeze and cross-check the retained dispatcher grant section."""
    if not isinstance(section, Mapping):
        raise WorkerExecutionUnavailable("worker_execution_malformed", "worker execution grant is missing")
    body = dict(section)
    digest = body.pop("digest", None)
    if not isinstance(digest, str) or digest != digest_of(body):
        raise WorkerExecutionUnavailable("worker_execution_malformed", "worker execution grant digest does not match")
    if body.get("protocol") != 1:
        raise WorkerExecutionUnavailable("worker_execution_malformed", "worker execution protocol is unsupported")
    expected = {
        "task_id": grant.task_id,
        "profile": grant.profile,
        "board": grant.board,
        "preparation_id": grant.preparation_id,
        "run_id": grant.run_id,
        "claim_lock": grant.claim_lock,
    }
    for key, value in expected.items():
        if body.get(key) != value:
            raise WorkerExecutionUnavailable("worker_execution_grant_mismatch", f"worker execution {key} does not match the grant")
    workspace = body.get("workspace")
    if not isinstance(workspace, Mapping) or workspace.get("path") != grant.workspace:
        raise WorkerExecutionUnavailable("worker_execution_grant_mismatch", "worker execution workspace does not match the grant")
    authority = ExecutionAuthority(
        policy_digest=str(body["policy_digest"]),
        task_id=grant.task_id,
        run_id=grant.run_id,
        preparation_id=grant.preparation_id,
        workspace_path=grant.workspace,
        workspace_device=int(workspace["device"]),
        workspace_inode=int(workspace["inode"]),
        git=_freeze_mapping(body.get("git")),
        image_store=_freeze_mapping(body.get("image_store")),
        inspect=bool(body.get("inspect")),
        role=str(body.get("role") or "unknown"),
        extra_source_paths=tuple(str(item) for item in body.get("extra_source_paths") or ()),
        recipes=_freeze_recipes(body.get("recipes")),
        review=_freeze_mapping(body.get("review")),
        policy_plugin=_freeze_mapping(body.get("policy_plugin")),
        runtime=_freeze_mapping(body.get("runtime")),
        execution_parent=_freeze_mapping(body.get("execution_parent")),
        journal_dir=_freeze_mapping(body.get("journal_dir")),
        runtime_parent=_freeze_mapping(body.get("runtime_parent")),
        runtime_identity=grant.runtime_identity,
        bound_digest=digest,
    )
    authority.verify_static_identities()
    return authority


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


def _authorize_execution() -> tuple[Any, ExecutionAuthority]:
    from hermes_cli import worker_startup

    try:
        grant = worker_startup._authorize()
    except worker_startup.WorkerStartupUnavailable as exc:
        raise WorkerExecutionUnavailable("worker_execution_unavailable", _bounded(exc))
    authority = getattr(grant, "execution", None)
    if not isinstance(authority, ExecutionAuthority):
        raise WorkerExecutionUnavailable("worker_execution_unavailable", "no worker execution grant for this process")
    authority.verify_static_identities()
    return grant, authority


def _execution_context_allowed() -> bool:
    """Availability also refuses delegated children, like the startup capability."""
    from agent.delegation_context import (
        is_delegated_child_process_context,
        is_dispatcher_owned_worker_context,
    )

    return not is_delegated_child_process_context() and is_dispatcher_owned_worker_context()


def worker_inspect_available() -> bool:
    try:
        if not _execution_context_allowed():
            return False
        authority = getattr(_retained_grant_quiet(), "execution", None)
        return isinstance(authority, ExecutionAuthority) and authority.inspect
    except Exception:
        return False


def worker_verify_available() -> bool:
    try:
        if not _execution_context_allowed():
            return False
        authority = getattr(_retained_grant_quiet(), "execution", None)
        return isinstance(authority, ExecutionAuthority) and bool(authority.recipes)
    except Exception:
        return False


def _retained_grant_quiet() -> Any:
    from hermes_cli import worker_startup

    return worker_startup._retained_grant()


def worker_execution_policy_summary() -> dict[str, Any]:
    """What the mandatory pre-execution callback may authorize, from the grant only.

    The native handler called :func:`_authorize_execution` (including the full runtime
    provenance check) immediately before invoking this hook and re-checks the final
    arguments afterwards, so the callback reads the retained authority's summary
    instead of repeating the expensive sealed-payload verification.
    """
    from hermes_cli import worker_startup

    grant = worker_startup._retained_grant()
    authority = getattr(grant, "execution", None)
    if not isinstance(authority, ExecutionAuthority):
        raise WorkerExecutionUnavailable("worker_execution_unavailable", "no worker execution grant for this process")
    authority.verify_static_identities()
    return {
        "policy_digest": authority.policy_digest,
        "inspect": authority.inspect,
        "recipes": list(authority.recipe_ids()),
    }


# ---------------------------------------------------------------------------
# Live claim re-authorization
# ---------------------------------------------------------------------------


def _reauthorize_live_claim(grant: Any) -> None:
    """Revalidate the active claim in a NEW read-only board transaction."""
    from hermes_cli import worker_startup

    try:
        connection = worker_startup._open_board(grant.db_path)
    except Exception as exc:
        raise WorkerExecutionUnavailable("claim_unavailable", _bounded(exc))
    try:
        worker_startup._read_startup(connection, grant)
    except worker_startup.WorkerStartupUnavailable as exc:
        raise WorkerExecutionUnavailable("claim_lost", _bounded(exc))
    finally:
        with contextlib.suppress(Exception):
            connection.execute("ROLLBACK")
        connection.close()


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
        self.path.mkdir(mode=0o700)
        self._write_metadata()

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
    base = os.environ.get("TMPDIR") or "/tmp"
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


# ---------------------------------------------------------------------------
# Mandatory required-hook approval
# ---------------------------------------------------------------------------


def _args_digest(args: Mapping[str, Any]) -> str:
    return digest_of(args)


def _original_args_equal(args: Mapping[str, Any], original_args: Optional[Mapping[str, Any]]) -> bool:
    if original_args is None:
        return True
    try:
        return canonical_bytes(args) == canonical_bytes(original_args)
    except (TypeError, ValueError):
        return False


def _require_execution_approval(
    authority: ExecutionAuthority,
    *,
    tool_name: str,
    args: Mapping[str, Any],
    operation: Optional[str],
    recipe: Optional[str],
) -> None:
    from hermes_cli.plugins_dispatch import dispatch_required_worker_execution

    ok, reason = dispatch_required_worker_execution(
        owner=authority.policy_plugin,
        tool_name=tool_name,
        args=dict(args),
        policy_digest=authority.policy_digest,
        args_digest=_args_digest(args),
        operation=operation,
        recipe=recipe,
    )
    if not ok:
        raise WorkerExecutionRefused("authorization_refused", reason)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def _verify_args(args: Any) -> dict[str, str]:
    if not isinstance(args, Mapping):
        _refuse("invalid_arguments", "arguments must be a JSON object")
    unknown = set(args) - {"recipe", "source_digest"}
    missing = {"recipe", "source_digest"} - set(args)
    if unknown:
        _refuse("invalid_arguments", f"unknown argument(s): {sorted(unknown)}")
    if missing:
        _refuse("invalid_arguments", f"missing argument(s): {sorted(missing)}")
    recipe = args["recipe"]
    if not isinstance(recipe, str) or not _SERVICE_ID_RE.match(recipe):
        _refuse("invalid_arguments", "recipe must be a granted recipe id")
    digest = args["source_digest"]
    if not isinstance(digest, str) or not _DIGEST_RE.match(digest):
        _refuse("invalid_arguments", "source_digest must be sha256:<64 lowercase hex>")
    return {"recipe": recipe, "source_digest": digest}


def _materialize_source(authority: ExecutionAuthority, run_root: Path) -> tuple[SourceSnapshot, Path]:
    source_dir = run_root / "source"
    source_dir.mkdir(mode=0o700)
    snapshot = _scan_source(authority, materialize=source_dir)
    return snapshot, source_dir


def worker_verify(args: dict, *, original_args: dict | None = None) -> str:
    """Execute one granted recipe inside a fresh private stack; JSON text result."""
    try:
        if not _original_args_equal(args, original_args):
            raise WorkerExecutionRefused("arguments_changed", "final arguments differ from the request")
        normalized = _verify_args(args)
        grant, authority = _authorize_execution()
        recipe = authority.recipe(normalized["recipe"])
        if authority.image_store is None:
            _refuse("image_store_unavailable", "no image store is bound to this grant")
        _require_execution_approval(
            authority, tool_name=TOOL_VERIFY, args=normalized,
            operation="verify", recipe=normalized["recipe"],
        )
        if not _original_args_equal(args, original_args):
            raise WorkerExecutionRefused("arguments_changed", "arguments changed during authorization")
        if not _VERIFY_LOCK.acquire(blocking=False):
            raise WorkerExecutionRefused("verification_busy", "another verification is running in this worker")
        try:
            _reauthorize_live_claim(grant)
            invocation = uuid.uuid4().hex
            run_root = Path(str(_record_value(authority.execution_parent, "path", ""))) / invocation
            run_root.mkdir(mode=0o700)
            journal_path = Path(str(_record_value(authority.journal_dir, "path", ""))) / f"{invocation}.json"
            try:
                snapshot, source_dir = _materialize_source(authority, run_root)
                if snapshot.source_digest != normalized["source_digest"]:
                    raise WorkerExecutionRefused("source_changed", "the granted source does not match the requested digest")
                from hermes_cli import worker_execution_podman as backend

                result = backend.run_verification(
                    authority=authority,
                    recipe=recipe,
                    source_dir=source_dir,
                    run_root=run_root,
                    journal_path=journal_path,
                    grant=grant,
                )
            except BaseException:
                # Ownership transfers only once the backend's journal exists; before
                # that (materialization, digest refusal, engine/cgroup/journal setup
                # failures) this call removes its own allocation.
                if not journal_path.exists():
                    from hermes_cli.worker_execution_podman import remove_tree

                    remove_tree(run_root)
                raise
            # Revalidate the claim and source identity at finish; anything that changed
            # makes the evidence unusable rather than silently accepted.
            _reauthorize_live_claim(grant)
            authority.verify_static_identities()
            final = _scan_source(authority)
            if final.source_digest != snapshot.source_digest or final.head != snapshot.head:
                raise WorkerExecutionRefused("source_changed", "the granted source changed during verification")
            from tools.interrupt import is_interrupted

            if is_interrupted():
                result["cancelled"] = True
                result["accepted"] = False
            result["source_digest"] = snapshot.source_digest
            result["head"] = snapshot.head
            result["invocation_digest"] = _args_digest(normalized)
            result["original_args_digest"] = _args_digest(args)
            if not result.get("cleanup_ok"):
                result["accepted"] = False
            return json.dumps(result, sort_keys=True)
        finally:
            _VERIFY_LOCK.release()
    except WorkerExecutionError as exc:
        return json.dumps({
            "ok": False, "accepted": False, "error": exc.code, "message": _bounded(exc.message),
        }, sort_keys=True)
    except Exception as exc:  # a boundary failure is a refusal, never a pass
        return json.dumps({
            "ok": False, "accepted": False, "error": "worker_execution_failed",
            "message": _bounded(f"{type(exc).__name__}: {exc}"),
        }, sort_keys=True)


def worker_inspect(args: dict, *, original_args: dict | None = None) -> str:
    """Typed, read-only source/Git inspection; JSON text result."""
    try:
        if not _original_args_equal(args, original_args):
            raise WorkerExecutionRefused("arguments_changed", "final arguments differ from the request")
        operation, normalized = _validate_inspect_args(args)
        grant, authority = _authorize_execution()
        if not authority.inspect:
            raise WorkerExecutionRefused("inspect_not_granted", "source inspection is not granted to this workspace")
        _require_execution_approval(
            authority, tool_name=TOOL_INSPECT, args=normalized,
            operation=operation, recipe=None,
        )
        if not _original_args_equal(args, original_args):
            raise WorkerExecutionRefused("arguments_changed", "arguments changed during authorization")
        _reauthorize_live_claim(grant)
        if operation == "summary":
            result = _inspect_summary(grant, authority)
        elif operation == "read":
            result = _inspect_read(normalized, authority)
        elif operation == "list":
            result = _inspect_list(normalized, authority)
        else:
            result = _inspect_diff(normalized, authority)
        # Discard the buffered result if this worker lost the claim or its granted
        # filesystem identity meanwhile.
        _reauthorize_live_claim(grant)
        authority.verify_static_identities()
        return json.dumps(result, sort_keys=True)
    except WorkerExecutionError as exc:
        return json.dumps({
            "ok": False, "error": exc.code, "message": _bounded(exc.message),
        }, sort_keys=True)
    except Exception as exc:
        return json.dumps({
            "ok": False, "error": "inspection_failed",
            "message": _bounded(f"{type(exc).__name__}: {exc}"),
        }, sort_keys=True)

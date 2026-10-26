"""Operator ``kanban.worker_execution`` policy and the retained dispatcher grant.

Parsing is strict: unknown keys, unpinned images, non-canonical paths and duplicate
workspaces are refused. Resolution happens once in the dispatcher; a worker receives a
frozen :class:`~hermes_cli.worker_execution_common.ExecutionAuthority`, never policy text.
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
    lexical = Path(policy.image_store)
    # Walk the CONFIGURED path: a symlink at the leaf or any ancestor is refused, so a
    # resolved target cannot substitute itself for the operator's declared store.
    current = Path(lexical.anchor)
    for part in lexical.parts[1:]:
        current = current / part
        try:
            component = os.lstat(current)
        except OSError:
            _refuse("policy_malformed", "image_store does not exist")
        if stat.S_ISLNK(component.st_mode):
            _refuse("policy_malformed", "image_store must not be or have a symlinked component")
    try:
        path = lexical.resolve(strict=True)
    except OSError:
        _refuse("policy_malformed", "image_store does not exist")
    try:
        info = os.lstat(path)
    except OSError:
        _refuse("policy_malformed", "image_store does not exist")
    if not stat.S_ISDIR(info.st_mode):
        _refuse("policy_malformed", "image_store must be a real directory")
    if info.st_mode & 0o077:
        _refuse("policy_malformed", "image_store must be owner-only")
    uid = os.getuid() if hasattr(os, "getuid") else -1  # windows-footgun: ok — hasattr-guarded; grant resolution is POSIX-only
    if int(info.st_uid) != uid:
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
            text = manifest.read_bytes().decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
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
        elif kind == "code" and str(contract.get("review_mode") or "").casefold() == "same_card":
            # A same-card review keeps ``kind: code`` and is recognized by its review
            # lane. A claimed run reads ``running``, so consult the recorded resume
            # status instead of requiring the literal ``review`` status at spawn time.
            if status != "review" and kanban_db._resume_status_from_events(connection, task_id) != "review":
                return "implementation", None
            role = "review"
            candidate_id = task_id
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


def _workspace_head_from_identity(git: Mapping[str, Any]) -> Optional[str]:
    """Read HEAD through the captured Git directory (no repository code execution)."""
    head_path = Path(str(git.get("git_dir", ""))) / "HEAD"
    try:
        info = os.lstat(head_path)
        if not stat.S_ISREG(info.st_mode) or info.st_size > 4096:
            return None
        raw = head_path.read_bytes().decode("utf-8").strip()
    except OSError:
        return None
    if not raw.startswith("ref:"):
        return raw if _OBJECT_ID_RE.match(raw) else None
    ref = raw[4:].strip()
    for base in (Path(str(git.get("git_dir", ""))), Path(str(git.get("common_dir", "")))):
        candidate = base / ref
        try:
            if candidate.is_file() and not candidate.is_symlink():
                value = candidate.read_bytes().decode("utf-8").strip()
                if _OBJECT_ID_RE.match(value):
                    return value
        except OSError:
            continue
        packed = base / "packed-refs"
        try:
            if packed.is_file() and not packed.is_symlink():
                for line in packed.read_bytes().decode("utf-8").splitlines():
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



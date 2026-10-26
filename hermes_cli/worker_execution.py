"""Explicit, operator-granted worker source inspection and disposable verification.

Authority for both tools is the dispatcher's final runtime grant, retained in-process
by :func:`hermes_cli.worker_startup._retain_worker_grant`.  The dispatcher resolves the
operator's ``kanban.worker_execution`` policy through its own profile config and binds
the resolved workspace, Git identity, image store, review handoff and policy plugin
into that grant; a worker can select an already-granted recipe, never construct one.

Handlers live here; policy parsing is in :mod:`hermes_cli.worker_execution_policy`,
source inspection in :mod:`hermes_cli.worker_execution_source`, shared frozen types in
:mod:`hermes_cli.worker_execution_common` and the container boundary in
:mod:`hermes_cli.worker_execution_podman`.
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
from hermes_cli.worker_execution_policy import parse_grant_section
from hermes_cli.worker_execution_source import (
    _inspect_diff, _inspect_list, _inspect_read, _inspect_summary, _materialize_source,
    _scan_source, _validate_inspect_args,
)

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

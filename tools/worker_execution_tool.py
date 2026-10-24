"""Native registration for the granted worker source-inspection/verification tools.

Both tools are runtime-owned `kanban` tools: the handlers compare the model's original
arguments with the final (possibly hook-rewritten) ones, then re-authorize the retained
dispatcher grant, the live claim and the mandatory pre-execution approval. Availability
is advisory only; final authorization happens inside each handler.

Only the exact handler objects registered here may own these names
(:data:`tools.registry.RESERVED_NATIVE_EXECUTION_TOOLS`), so a profile plugin, a
same-toolset replacement or an ``override=True`` registration cannot take them over.
"""

from __future__ import annotations

from typing import Any, Optional

from tools.registry import no_cache_check_fn, registry

INSPECT_TOOL = "kanban_worker_inspect"
VERIFY_TOOL = "kanban_worker_verify"

_INSPECT_DESCRIPTION = (
    "Read this dispatcher-owned worker's granted source and Git state without executing "
    "repository code. Operations: summary (HEAD, branch, porcelain status, complete source "
    "digest), read (bounded UTF-8 source page), list (bounded source inventory) and diff "
    "(the granted review commit pair, or working/staged changes for implementation claims). "
    "Use expected digests to detect a concurrent change."
)
_VERIFY_DESCRIPTION = (
    "Run one explicitly granted verification recipe inside a disposable private stack "
    "(fresh rootless containers, private network namespace, copied source). Requires the "
    "recipe id and the complete source digest reported by kanban_worker_inspect. Returns a "
    "structured evidence receipt; a recipe failure is evidence, not a retry signal."
)

_INSPECT_SCHEMA = {
    "name": INSPECT_TOOL,
    "description": _INSPECT_DESCRIPTION,
    "parameters": {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["summary", "read", "list", "diff"],
                "description": "Inspection operation.",
            },
            "path": {
                "type": "string",
                "description": "Relative source path (read/diff) or directory prefix (list).",
            },
            "start_line": {"type": "integer", "minimum": 1, "description": "First returned line (default 1)."},
            "line_count": {
                "type": "integer", "minimum": 1, "maximum": 1000,
                "description": "Maximum returned lines (default 200).",
            },
            "offset": {"type": "integer", "minimum": 0, "description": "Inventory offset (default 0)."},
            "limit": {
                "type": "integer", "minimum": 1, "maximum": 1000,
                "description": "Maximum inventory entries (default 200).",
            },
            "base": {"type": "string", "description": "Full commit id of the granted review base."},
            "head": {"type": "string", "description": "Full commit id of the granted review head."},
            "mode": {
                "type": "string", "enum": ["working", "staged"],
                "description": "Uncommitted changes instead of a review commit pair.",
            },
            "expected_source_digest": {
                "type": "string", "description": "Refuse (source_changed) when the source digest moved.",
            },
            "expected_diff_digest": {
                "type": "string", "description": "Refuse (source_changed) when the rendered diff moved.",
            },
        },
        "required": ["operation"],
        "additionalProperties": False,
    },
}

_VERIFY_SCHEMA = {
    "name": VERIFY_TOOL,
    "description": _VERIFY_DESCRIPTION,
    "parameters": {
        "type": "object",
        "properties": {
            "recipe": {"type": "string", "description": "Id of a granted verification recipe."},
            "source_digest": {
                "type": "string",
                "description": "Complete source digest from kanban_worker_inspect (sha256:<64 hex>).",
            },
        },
        "required": ["recipe", "source_digest"],
        "additionalProperties": False,
    },
}


def _inspect(args: dict, *, original_args: Optional[dict] = None) -> str:
    from hermes_cli.worker_execution import worker_inspect

    return worker_inspect(args, original_args=original_args)


def _verify(args: dict, *, original_args: Optional[dict] = None) -> str:
    from hermes_cli.worker_execution import worker_verify

    return worker_verify(args, original_args=original_args)


def _inspect_available() -> bool:
    try:
        from hermes_cli.worker_execution import worker_inspect_available

        return bool(worker_inspect_available())
    except Exception:
        return False


def _verify_available() -> bool:
    try:
        from hermes_cli.worker_execution import worker_verify_available

        return bool(worker_verify_available())
    except Exception:
        return False


registry.register(
    name=INSPECT_TOOL,
    toolset="kanban",
    schema=_INSPECT_SCHEMA,
    handler=_inspect,
    check_fn=no_cache_check_fn(_inspect_available),
    description="Inspect granted worker source and Git state",
    emoji="🔎",
)

registry.register(
    name=VERIFY_TOOL,
    toolset="kanban",
    schema=_VERIFY_SCHEMA,
    handler=_verify,
    check_fn=no_cache_check_fn(_verify_available),
    description="Run a granted verification recipe in a private stack",
    emoji="🧪",
)

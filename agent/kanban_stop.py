"""Turn-end guard for kanban workers.

Kanban workers must end with ``kanban_complete``, ``kanban_request_changes``,
or ``kanban_block``. Models
("Let me write the report now") and stop with ``finish_reason=stop`` and no
tool calls. Hermes treats that as a clean exit → ``rc=0`` → dispatcher
``protocol_violation``.

This module is policy-only: when a kanban worker tries to finish without a
terminal board tool, return a bounded synthetic nudge so the conversation
loop continues instead of exiting.
"""

from __future__ import annotations

import os
from typing import Any, Iterable, Optional


_TERMINAL_KANBAN_TOOLS = frozenset({
    "kanban_complete",
    "kanban_request_changes",
    "kanban_block",
})

_DEFAULT_MAX_ATTEMPTS = 2


def kanban_stop_nudge_enabled() -> bool:
    """Return whether the kanban stop-guard is active for this process.

    On when ``HERMES_KANBAN_TASK`` is set (dispatcher-spawned worker), unless
    ``HERMES_KANBAN_STOP_NUDGE`` explicitly disables it.
    """
    env = os.environ.get("HERMES_KANBAN_STOP_NUDGE")
    if env is not None and env.strip().lower() in {"0", "false", "no", "off"}:
        return False
    task = (os.environ.get("HERMES_KANBAN_TASK") or "").strip()
    return bool(task)


def _tool_call_name(tc: Any) -> str:
    if isinstance(tc, dict):
        fn = tc.get("function")
        if isinstance(fn, dict):
            return str(fn.get("name") or "")
        return str(tc.get("name") or "")
    fn = getattr(tc, "function", None)
    if fn is not None:
        return str(getattr(fn, "name", "") or "")
    return str(getattr(tc, "name", "") or "")


def session_called_kanban_terminal(messages: Iterable[dict] | None) -> bool:
    """True if this conversation already invoked a terminal kanban tool."""
    if not messages:
        return False
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role == "assistant":
            for tc in msg.get("tool_calls") or []:
                if _tool_call_name(tc) in _TERMINAL_KANBAN_TOOLS:
                    return True
        elif role == "tool":
            name = str(msg.get("name") or "")
            if name in _TERMINAL_KANBAN_TOOLS:
                return True
    return False


def build_kanban_stop_nudge(
    *,
    messages: Iterable[dict] | None = None,
    attempts: int = 0,
    max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
    task_id: Optional[str] = None,
) -> Optional[str]:
    """Return a synthetic follow-up when a kanban worker exits without a terminal tool.

    Returns ``None`` when the guard should not fire (not a kanban worker,
    already completed/blocked, or nudge budget exhausted).
    """
    if not kanban_stop_nudge_enabled():
        return None
    if attempts >= max_attempts:
        return None
    if session_called_kanban_terminal(messages):
        return None

    tid = (task_id or os.environ.get("HERMES_KANBAN_TASK") or "").strip() or "this task"
    return (
        "[System: You are a Hermes kanban worker. A plain-text reply is NOT a "
        "terminal state for the board.\n\n"
        f"Task `{tid}` is still `running`. Ending now without a board tool "
        "causes a protocol violation (clean exit with no terminal kanban "
        "tool).\n\n"
        "Do this immediately in your next response — do not narrate intent:\n"
        "1. Finish any remaining deliverable (write the required file(s) now).\n"
        "2. Call `kanban_complete(summary=..., artifacts=[...])` if the work "
        "is done, `kanban_request_changes(reason=...)` for a review retry, "
        "OR `kanban_block(reason=...)` if you are blocked.\n\n"
        "Never end a turn with only a promise of future action. Repeated "
        "protocol violations will block this task and require manual intervention.]"
    )


def terminal_call_audit(messages: Iterable[dict] | None) -> dict[str, Any]:
    """Summarize terminal-tool evidence for a protocol-violation run."""
    observed: list[str] = []
    if messages:
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            if msg.get("role") == "assistant":
                observed.extend(
                    name for name in (
                        _tool_call_name(tc) for tc in msg.get("tool_calls") or []
                    ) if name
                )
            elif msg.get("role") == "tool":
                name = str(msg.get("name") or "")
                if name:
                    observed.append(name)
    terminal = [name for name in observed if name in _TERMINAL_KANBAN_TOOLS]
    return {
        "terminal_call_seen": bool(terminal),
        "terminal_tools": terminal,
        "observed_tools": observed,
        "required_tools": sorted(_TERMINAL_KANBAN_TOOLS),
    }


def persist_kanban_protocol_violation(
    *,
    messages: Iterable[dict] | None,
    final_output: Optional[str],
    session_id: Optional[str] = None,
    task_id: Optional[str] = None,
    run_id: Optional[int] = None,
) -> bool:
    """Save final-output and terminal-call evidence on the open worker run."""
    tid = (task_id or os.environ.get("HERMES_KANBAN_TASK") or "").strip()
    if not tid or session_called_kanban_terminal(messages):
        return False
    raw_run_id = run_id if run_id is not None else os.environ.get("HERMES_KANBAN_RUN_ID")
    try:
        worker_run_id = int(raw_run_id)
    except (TypeError, ValueError):
        return False
    metadata = {
        "protocol_violation": True,
        "final_assistant_output": final_output or "",
        "terminal_call_audit": terminal_call_audit(messages),
    }
    try:
        from tools.kanban_tools import _stamp_worker_session_metadata
        metadata = _stamp_worker_session_metadata(tid, metadata) or metadata
        if session_id:
            metadata["worker_session_id"] = session_id
        from hermes_cli import kanban_db
        conn = kanban_db.connect(board=os.environ.get("HERMES_KANBAN_BOARD"))
        try:
            return kanban_db.record_protocol_violation(
                conn,
                tid,
                run_id=worker_run_id,
                metadata=metadata,
                summary=final_output,
            )
        finally:
            conn.close()
    except Exception:
        return False


__all__ = [
    "build_kanban_stop_nudge",
    "kanban_stop_nudge_enabled",
    "session_called_kanban_terminal",
    "terminal_call_audit",
    "persist_kanban_protocol_violation",
]

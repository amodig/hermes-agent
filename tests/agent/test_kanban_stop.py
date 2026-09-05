"""Tests for the kanban worker turn-end stop guard."""

from __future__ import annotations

import json
import pytest

from agent.kanban_stop import (
    build_kanban_stop_nudge,
    kanban_stop_nudge_enabled,
    persist_kanban_protocol_violation,
    session_called_kanban_terminal,
    terminal_call_audit,
)

from hermes_cli import kanban_db as kb


@pytest.fixture
def clear_kanban_env(monkeypatch):
    for var in ("HERMES_KANBAN_TASK", "HERMES_KANBAN_STOP_NUDGE"):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch






def test_env_can_disable(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_abc")
    clear_kanban_env.setenv("HERMES_KANBAN_STOP_NUDGE", "0")
    assert kanban_stop_nudge_enabled() is False
    assert build_kanban_stop_nudge(messages=[]) is None


def test_nudge_when_no_terminal_tool(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_46be8aa5")
    messages = [
        {"role": "user", "content": "work kanban task"},
        {
            "role": "assistant",
            "content": "Let me write the comprehensive recipe.",
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {"name": "kanban_heartbeat", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "name": "kanban_heartbeat", "tool_call_id": "1", "content": "ok"},
    ]
    nudge = build_kanban_stop_nudge(messages=messages, attempts=0)
    assert nudge is not None
    assert "kanban_complete" in nudge
    assert "kanban_block" in nudge
    assert "t_46be8aa5" in nudge
    assert "protocol violation" in nudge.lower() or "protocol" in nudge.lower()


def test_no_nudge_after_kanban_complete(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_abc")
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {"name": "kanban_complete", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "name": "kanban_complete", "tool_call_id": "1", "content": "done"},
    ]
    assert session_called_kanban_terminal(messages) is True
    assert build_kanban_stop_nudge(messages=messages) is None


def test_request_changes_is_a_terminal_kanban_call(clear_kanban_env):
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_review")
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "kanban_request_changes",
                        "arguments": '{"reason":"fix"}',
                    },
                }
            ],
        },
        {"role": "tool", "name": "kanban_request_changes", "content": "builder"},
    ]
    assert session_called_kanban_terminal(messages) is True
    assert build_kanban_stop_nudge(messages=messages) is None
    assert terminal_call_audit(messages)["terminal_tools"] == [
        "kanban_request_changes",
        "kanban_request_changes",
    ]


def test_protocol_violation_evidence_preserves_worker_run_data(
    clear_kanban_env, tmp_path,
):
    db_path = tmp_path / "kanban.db"
    clear_kanban_env.setenv("HERMES_KANBAN_DB", str(db_path))
    clear_kanban_env.setenv("HERMES_KANBAN_TASK", "t_evidence")
    clear_kanban_env.setenv("HERMES_SESSION_ID", "worker-session-1")
    kb.init_db()
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="evidence", assignee="worker")
        run = kb.claim_task(conn, tid, claimer="worker:1")
        assert run is not None
        clear_kanban_env.setenv("HERMES_KANBAN_TASK", tid)
        clear_kanban_env.setenv("HERMES_KANBAN_RUN_ID", str(run.current_run_id))
        final_output = "final answer without a terminal call"
        assert persist_kanban_protocol_violation(
            messages=[{"role": "assistant", "content": final_output}],
            final_output=final_output,
            session_id="worker-session-1",
        )
        row = conn.execute(
            "SELECT summary, metadata FROM task_runs WHERE id = ?", (run.current_run_id,),
        ).fetchone()
        assert row["summary"] == final_output
        metadata = json.loads(row["metadata"])
        assert metadata["protocol_violation"] is True
        assert metadata["worker_session_id"] == "worker-session-1"
        assert metadata["final_assistant_output"] == final_output
        assert metadata["terminal_call_audit"]["terminal_call_seen"] is False






# ── Integration: agent nudge + dispatcher bounded retry ──────────────
# These tests verify the two layers compose correctly: the agent-side
# nudge fires first (up to 2 attempts), and if the worker still exits
# without a terminal call, the dispatcher's bounded retry (streak of 3)
# handles it.  See also tests/hermes_cli/test_kanban_core_functionality.py
# for the dispatcher-side streak tests.





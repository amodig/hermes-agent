"""Regression tests for atomic Kanban goal updates (issue #4)."""

from __future__ import annotations

import argparse
import json

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    kb.init_db()
    return home


def _run_cli(*argv: str) -> int:
    root = argparse.ArgumentParser(prog="hermes")
    subparsers = root.add_subparsers(dest="command")
    kc.build_parser(subparsers)
    args = root.parse_args(["kanban", *argv])
    return kc.kanban_command(args)


def test_update_requeues_triage_card_and_show_exposes_goal_revision(
    kanban_home, capsys, monkeypatch
):
    monkeypatch.setenv("HERMES_PROFILE", "cto")
    with kb.connect_closing() as conn:
        task_id = kb.create_task(
            conn, title="rough goal", body="draft", triage=True
        )

    assert (
        _run_cli(
            "update",
            task_id,
            "--title",
            "approved goal",
            "--body",
            "ship it",
            "--goal-mode",
            "--expected-version",
            "1",
            "--reason",
            "authorize implementation",
            "--transition",
            "triage_to_ready",
            "--json",
        )
        == 0
    )
    updated = json.loads(capsys.readouterr().out)
    assert updated["task"]["status"] == "ready"
    assert updated["task"]["version"] == 2
    assert updated["effective_goal"]["title"] == "approved goal"
    assert updated["effective_goal"]["author"] == "cto"
    assert updated["effective_goal"]["prior_version"] == 1

    assert _run_cli("show", task_id, "--json") == 0
    shown = json.loads(capsys.readouterr().out)
    goal = shown["effective_goal"]
    assert goal["version"] == 2
    assert goal["body"] == "ship it"
    assert goal["reason"] == "authorize implementation"
    assert isinstance(goal["timestamp"], int)


def test_update_rejects_stale_version_without_mutation(kanban_home):
    with kb.connect_closing() as conn:
        task_id = kb.create_task(conn, title="original", triage=True)
        kb.update_task(
            conn,
            task_id,
            title="first revision",
            expected_version=1,
            reason="first approval",
        )
        before = kb.get_task(conn, task_id)
        events_before = len(kb.list_events(conn, task_id))
        with pytest.raises(kb.TaskUpdateConflict, match="expected version 1"):
            kb.update_task(
                conn,
                task_id,
                title="stale revision",
                expected_version=1,
                reason="stale approval",
            )
        after = kb.get_task(conn, task_id)

    assert after.title == before.title
    assert after.version == before.version
    with kb.connect_closing() as conn:
        assert len(kb.list_events(conn, task_id)) == events_before


def test_triage_to_ready_requires_completed_parents(kanban_home):
    with kb.connect_closing() as conn:
        open_parent = kb.create_task(conn, title="open parent")
        blocked_child = kb.create_task(
            conn, title="blocked child", parents=[open_parent], triage=True
        )
        kb.update_task(
            conn,
            blocked_child,
            expected_version=1,
            reason="try to ready blocked child",
            transition="triage_to_ready",
        )
        assert kb.get_task(conn, blocked_child).status == "todo"

        done_parent = kb.create_task(conn, title="done parent")
        assert kb.complete_task(
            conn, done_parent, summary="finished", result="done"
        )
        ready_child = kb.create_task(
            conn, title="ready child", parents=[done_parent], triage=True
        )
        kb.update_task(
            conn,
            ready_child,
            expected_version=1,
            reason="ready after parent completion",
            transition="triage_to_ready",
        )
        assert kb.get_task(conn, ready_child).status == "ready"


def test_tool_update_matches_cli_contract(kanban_home, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_PROFILE", "cto")
    with kb.connect_closing() as conn:
        cli_task = kb.create_task(conn, title="cli", triage=True)
        tool_task = kb.create_task(conn, title="tool", triage=True)

    assert _run_cli(
        "update",
        cli_task,
        "--body",
        "same body",
        "--expected-version",
        "1",
        "--reason",
        "same approval",
        "--transition",
        "triage_to_ready",
        "--json",
    ) == 0
    cli_payload = json.loads(capsys.readouterr().out)

    # Importing the tool registry is intentional: this exercises the public
    # handler instead of reaching into the database implementation.
    from tools import kanban_tools

    tool_payload = json.loads(
        kanban_tools._handle_update(
            {
                "task_id": tool_task,
                "body": "same body",
                "expected_version": 1,
                "reason": "same approval",
                "transition": "triage_to_ready",
            }
        )
    )
    assert tool_payload["ok"] is True
    assert tool_payload["status"] == "ready"
    assert tool_payload["version"] == 2
    assert tool_payload["effective_goal"]["body"] == "same body"
    assert tool_payload["effective_goal"]["reason"] == "same approval"
    assert cli_payload["task"]["status"] == tool_payload["status"]
    assert cli_payload["task"]["version"] == tool_payload["version"]
    for key in ("body", "reason", "version", "author", "prior_version"):
        assert cli_payload["effective_goal"][key] == tool_payload["effective_goal"][key]

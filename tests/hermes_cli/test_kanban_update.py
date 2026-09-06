"""Regression coverage for atomic Kanban goal revisions and updates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from hermes_cli import kanban as kanban_cli
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)
    return home


def _run_cli(*argv: str) -> int:
    """Invoke the real ``hermes kanban`` parser/dispatch path."""
    root = argparse.ArgumentParser()
    subparsers = root.add_subparsers(dest="cmd")
    kanban_cli.build_parser(subparsers)
    namespace = root.parse_args(["kanban", *argv])
    return kanban_cli.kanban_command(namespace)


def _goal_revision_rows(conn, task_id: str) -> list[tuple]:
    rows = conn.execute(
        "SELECT version, title, body, goal_mode, author, reason, prior_version "
        "FROM task_goal_revisions WHERE task_id = ? ORDER BY id",
        (task_id,),
    ).fetchall()
    return [tuple(row) for row in rows]


def test_cli_update_requeues_triage_and_show_exposes_effective_goal(
    kanban_home, capsys, monkeypatch
):
    monkeypatch.setenv("HERMES_PROFILE", "cto")
    with kbc.connect_closing() as conn:
        task_id = kb.create_task(
            conn, title="rough goal", body="draft", triage=True, created_by="user"
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
    assert updated["task"]["revision"] == 2
    assert updated["effective_goal"]["title"] == "approved goal"
    assert updated["effective_goal"]["body"] == "ship it"
    assert updated["effective_goal"]["goal_mode"] is True
    assert updated["effective_goal"]["author"] == "cto"
    assert updated["effective_goal"]["prior_version"] == 1

    with kbc.connect_closing() as conn:
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.version == task.revision == 2
        event = [e for e in kb.list_events(conn, task_id) if e.kind == "goal_revised"][
            -1
        ]
        assert event.payload["actor"] == "cto"
        assert event.payload["reason"] == "authorize implementation"
        assert event.payload["old"]["version"] == 1
        assert event.payload["new"]["version"] == 2
        context = kb.build_worker_context(conn, task_id)
    assert "approved goal" in context
    assert "ship it" in context

    assert _run_cli("show", task_id, "--json") == 0
    shown = json.loads(capsys.readouterr().out)
    goal = shown["effective_goal"]
    assert goal["version"] == 2
    assert goal["body"] == "ship it"
    assert goal["reason"] == "authorize implementation"
    assert isinstance(goal["timestamp"], int)


def test_stale_update_rejects_atomically_without_mutating_any_surface(kanban_home):
    with kbc.connect_closing() as conn:
        task_id = kb.create_task(conn, title="original", body="opening", triage=True)
        assert kb.update_task(
            conn,
            task_id,
            title="first revision",
            expected_version=1,
            reason="first approval",
        )
        before_task = kb.get_task(conn, task_id)
        before_goal = kb.get_effective_goal(conn, task_id)
        before_events = [
            (event.kind, event.payload, event.run_id)
            for event in kb.list_events(conn, task_id)
        ]
        before_revisions = _goal_revision_rows(conn, task_id)

        with pytest.raises(kb.TaskUpdateConflict, match="expected version 1"):
            kb.update_task(
                conn,
                task_id,
                title="stale revision",
                body="stale body",
                assignee="worker",
                model="model-x",
                provider="provider-x",
                goal_mode=True,
                expected_version=1,
                reason="stale approval",
                transition="triage_to_ready",
            )

        after_task = kb.get_task(conn, task_id)
        after_goal = kb.get_effective_goal(conn, task_id)
        after_events = [
            (event.kind, event.payload, event.run_id)
            for event in kb.list_events(conn, task_id)
        ]
        after_revisions = _goal_revision_rows(conn, task_id)

    assert after_task == before_task
    assert after_goal == before_goal
    assert after_events == before_events
    assert after_revisions == before_revisions


def test_triage_transition_recomputes_ready_or_todo_from_parents(kanban_home):
    with kbc.connect_closing() as conn:
        open_parent = kb.create_task(conn, title="open parent")
        blocked_child = kb.create_task(
            conn, title="blocked child", parents=[open_parent], triage=True
        )
        assert kb.get_task(conn, blocked_child).status == "triage"
        assert kb.update_task(
            conn,
            blocked_child,
            expected_version=1,
            reason="try to ready blocked child",
            transition="triage_to_ready",
        )
        assert kb.get_task(conn, blocked_child).status == "todo"

        done_parent = kb.create_task(conn, title="done parent")
        assert kb.complete_task(conn, done_parent, summary="finished", result="done")
        ready_child = kb.create_task(
            conn, title="ready child", parents=[done_parent], triage=True
        )
        assert kb.update_task(
            conn,
            ready_child,
            expected_version=1,
            reason="ready after parent completion",
            transition="triage_to_ready",
        )
        assert kb.get_task(conn, ready_child).status == "ready"


def test_claimed_update_is_rejected_and_run_history_links_and_comments_survive(
    kanban_home,
):
    with kbc.connect_closing() as conn:
        task_id = kb.create_task(conn, title="implementation")
        child_id = kb.create_task(conn, title="review", parents=[task_id])
        kb.add_comment(conn, task_id, "operator", "keep this context")
        claimed = kb.claim_task(conn, task_id, claimer="worker:1")
        assert claimed is not None

        before_runs = kb.list_runs(conn, task_id)
        before_events = [
            (event.kind, event.payload, event.run_id)
            for event in kb.list_events(conn, task_id)
        ]
        before_comments = kb.list_comments(conn, task_id)
        before_parents = kb.parent_ids(conn, task_id)
        before_children = kb.child_ids(conn, task_id)

        with pytest.raises(kb.TaskUpdateConflict, match="currently claimed"):
            kb.update_task(
                conn,
                task_id,
                title="operator overwrite",
                expected_version=claimed.version,
                reason="unsafe revision",
            )

        after = kb.get_task(conn, task_id)
        assert after is not None
        assert after.status == "running"
        assert after.version == claimed.version
        assert after.current_run_id == claimed.current_run_id
        assert after.claim_lock == claimed.claim_lock
        after_events = [
            (event.kind, event.payload, event.run_id)
            for event in kb.list_events(conn, task_id)
        ]
        assert after_events == before_events
        assert kb.list_comments(conn, task_id) == before_comments
        assert kb.parent_ids(conn, task_id) == before_parents
        assert kb.child_ids(conn, task_id) == before_children == [child_id]


def test_non_goal_fields_do_not_create_a_goal_revision(kanban_home):
    with kbc.connect_closing() as conn:
        task_id = kb.create_task(conn, title="rough", body="draft", triage=True)
        initial_goal = kb.get_effective_goal(conn, task_id)
        assert initial_goal is not None

        assert kb.update_task(
            conn,
            task_id,
            assignee="worker",
            expected_version=1,
            reason="route to worker",
        )
        task = kb.get_task(conn, task_id)
        goal_after_assignment = kb.get_effective_goal(conn, task_id)
        assert task is not None
        assert task.version == 2
        assert goal_after_assignment == initial_goal
        assert task.goal_revision_id == initial_goal["id"]

        assert kb.update_task(
            conn,
            task_id,
            body="authorized implementation",
            expected_version=2,
            reason="authorize implementation",
        )
        task = kb.get_task(conn, task_id)
        goal_after_body = kb.get_effective_goal(conn, task_id)
        assert task is not None
        assert task.version == 3
        assert goal_after_body["version"] == 2
        assert goal_after_body["body"] == "authorized implementation"
        assert goal_after_body["prior_version"] == 1


def test_tool_update_matches_cli_and_tool_show_effective_goal(
    kanban_home, monkeypatch, capsys
):
    monkeypatch.setenv("HERMES_PROFILE", "cto")
    with kbc.connect_closing() as conn:
        cli_task = kb.create_task(conn, title="cli", triage=True)
        tool_task = kb.create_task(conn, title="tool", triage=True)

    assert (
        _run_cli(
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
        )
        == 0
    )
    cli_payload = json.loads(capsys.readouterr().out)

    from tools import kanban_tools

    tool_payload = json.loads(
        kanban_tools._handle_update({
            "task_id": tool_task,
            "body": "same body",
            "expected_version": 1,
            "reason": "same approval",
            "transition": "triage_to_ready",
        })
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

    shown = json.loads(kanban_tools._handle_show({"task_id": tool_task}))
    assert shown["task"]["version"] == 2
    assert shown["effective_goal"] == tool_payload["effective_goal"]

"""Regression coverage for effective Kanban goal revisions at handoff gates."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "cto")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)
    kb.init_db()
    return home


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _git_repo(tmp_path: Path) -> tuple[Path, str, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(
        ["git", "init", "-b", "main", str(repo)],
        check=True,
        capture_output=True,
    )
    _git(repo, "config", "user.email", "kanban@example.com")
    _git(repo, "config", "user.name", "Kanban Test")
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "base")
    base = _git(repo, "rev-parse", "HEAD")
    branch = "task/goal-revision"
    _git(repo, "checkout", "-b", branch)
    return repo, base, branch


def _commit(repo: Path, path: str = "changed.py") -> str:
    target = repo / path
    target.write_text("value = 1\n", encoding="utf-8")
    _git(repo, "add", path)
    _git(repo, "commit", "-m", "implementation")
    return _git(repo, "rev-parse", "HEAD")


def _git_handoff_task(
    tmp_path: Path, *, plan_only: bool = False, include_tester: bool = True
) -> tuple[str, str, str]:
    repo, base, branch = _git_repo(tmp_path)
    body = (
        "Plan-only: no implementation is permitted."
        if plan_only
        else "Implement the requested change and report verification."
    )
    with kbc.connect_closing() as conn:
        task_id = kb.create_task(
            conn,
            title="Implementation handoff",
            body=body,
            assignee="implementer",
            workspace_kind="worktree",
            workspace_path=str(repo),
            branch_name=branch,
        )
        reviewer = kb.create_task(
            conn,
            title="Review implementation",
            assignee="reviewer",
            parents=[task_id],
        )
        if include_tester:
            tester = kb.create_task(
                conn,
                title="Test implementation",
                assignee="tester",
                parents=[task_id],
            )
        assert kb.get_task(conn, reviewer).status == "todo"
        if include_tester:
            assert kb.get_task(conn, tester).status == "todo"
    head = _commit(repo)
    return task_id, base, head


def test_plan_only_gate_then_authorization_revision_allows_completion(
    kanban_home, monkeypatch
):
    from tools import kanban_tools as kt

    captured: list[tuple[str, str]] = []
    monkeypatch.setenv("HERMES_PROFILE", "worker")
    monkeypatch.setattr(kt, "_goal_judge_available", lambda: True)

    def judge(goal, last_response, **_kwargs):
        captured.append((goal, last_response))
        if len(captured) == 1:
            assert "plan-only" in goal.lower()
            return "continue", "authorization is still required", False, None, False
        assert "explicit authorization to implement" in goal
        assert "plan-only" not in goal.lower()
        assert "implemented the authorized change" in last_response
        return "done", "verified", False, None, False

    monkeypatch.setattr(kt, "judge_goal", judge)
    with kbc.connect_closing() as conn:
        task_id = kb.create_task(
            conn,
            title="Plan the change",
            body="Plan-only: wait for authorization; do not implement.",
            assignee="worker",
            goal_mode=True,
        )

    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    first = json.loads(
        kt._handle_complete({
            "task_id": task_id,
            "summary": "I have a plan but cannot implement it yet",
        })
    )
    assert "error" in first
    with kbc.connect_closing() as conn:
        before = kb.get_task(conn, task_id)
        assert before is not None
        assert before.status == "ready"
        assert before.version == 1
        assert kb.get_effective_goal(conn, task_id)["version"] == 1
        assert kb.update_task(
            conn,
            task_id,
            title="Implement the change",
            body="Implement the change and report concrete verification.",
            expected_version=1,
            reason="explicit authorization to implement",
            author="cto",
        )

    second = json.loads(
        kt._handle_complete({
            "task_id": task_id,
            "summary": "implemented the authorized change; verification passed",
        })
    )
    assert second["ok"] is True
    assert len(captured) == 2
    with kbc.connect_closing() as conn:
        task = kb.get_task(conn, task_id)
        goal = kb.get_effective_goal(conn, task_id)
    assert task is not None and task.status == "done"
    assert goal["version"] == 2
    assert goal["reason"] == "explicit authorization to implement"


def test_request_review_judge_reads_authorized_effective_revision(
    kanban_home, monkeypatch
):
    from tools import kanban_tools as kt

    captured: list[str] = []
    monkeypatch.setenv("HERMES_PROFILE", "worker")
    monkeypatch.setattr(kt, "_goal_judge_available", lambda: True)

    def judge(goal, last_response, **_kwargs):
        captured.append(goal)
        return "done", "verified", False, None, False

    monkeypatch.setattr(kt, "judge_goal", judge)
    with kbc.connect_closing() as conn:
        task_id = kb.create_task(
            conn,
            title="Plan the review",
            body="Plan-only: no implementation is permitted.",
            assignee="worker",
            goal_mode=True,
        )
        assert kb.update_task(
            conn,
            task_id,
            title="Review the authorized implementation",
            body="Review the authorized implementation and report verification.",
            expected_version=1,
            reason="explicit authorization to implement",
            author="cto",
        )

    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    output = json.loads(
        kt._handle_request_review({
            "task_id": task_id,
            "summary": "implemented and verified the authorized change",
        })
    )
    assert output["ok"] is True
    assert output["status"] == "review"
    assert len(captured) == 1
    assert "Review the authorized implementation" in captured[0]
    assert "explicit authorization to implement" in captured[0]
    assert "plan-only" not in captured[0].lower()
    with kbc.connect_closing() as conn:
        assert kb.get_task(conn, task_id).status == "review"


def test_specifier_refuses_to_overwrite_newer_goal_revision(kanban_home, monkeypatch):
    from hermes_cli import kanban_specify

    with kbc.connect_closing() as conn:
        task_id = kb.create_task(
            conn,
            title="Plan the change",
            body="Plan-only: wait for authorization; do not implement.",
            triage=True,
        )
        assert kb.update_task(
            conn,
            task_id,
            title="Implement the change",
            body="Authorized implementation contract.",
            expected_version=1,
            reason="explicit authorization to implement",
            author="cto",
        )

    old_gate = {
        "title": "Plan the change",
        "body": "Plan-only: wait for authorization; do not implement.",
    }
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(old_gate)))]
    )
    prompt_messages: list[dict] = []

    def call_llm(**kwargs):
        prompt_messages.extend(kwargs["messages"])
        return response

    monkeypatch.setattr("agent.auxiliary_client.call_llm", call_llm)
    outcome = kanban_specify.specify_task(task_id)
    assert outcome.ok is False
    assert "kanban_update" in outcome.reason
    assert prompt_messages
    assert "Authorized implementation contract." in prompt_messages[-1]["content"]
    with kbc.connect_closing() as conn:
        task = kb.get_task(conn, task_id)
        goal = kb.get_effective_goal(conn, task_id)
    assert task is not None
    assert task.status == "triage"
    assert task.title == "Implement the change"
    assert goal["version"] == 2
    assert goal["body"] == "Authorized implementation contract."


def test_plan_only_nonempty_patch_is_rejected_before_reviewer_or_tester_release(
    kanban_home, tmp_path
):
    repo, base, branch = _git_repo(tmp_path)
    with kbc.connect_closing() as conn:
        parent = kb.create_task(
            conn,
            title="Implementation handoff",
            body="Plan-only: no implementation is permitted.",
            assignee="implementer",
            workspace_kind="worktree",
            workspace_path=str(repo),
            branch_name=branch,
        )
        reviewer = kb.create_task(
            conn, title="Review implementation", assignee="reviewer", parents=[parent]
        )
        tester = kb.create_task(
            conn, title="Test implementation", assignee="tester", parents=[parent]
        )
        assert kb.get_task(conn, reviewer).status == "todo"
        assert kb.get_task(conn, tester).status == "todo"
        head = _commit(repo)

        with pytest.raises(kb.CompletionContractError, match="plan-only"):
            kb.complete_task(
                conn,
                parent,
                summary="implemented",
                metadata={"base_sha": base, "head_sha": head},
            )

        parent_after = kb.get_task(conn, parent)
        reviewer_after = kb.get_task(conn, reviewer)
        tester_after = kb.get_task(conn, tester)
        event = conn.execute(
            "SELECT kind, payload FROM task_events WHERE task_id = ? "
            "ORDER BY id DESC LIMIT 1",
            (parent,),
        ).fetchone()

    assert parent_after is not None and parent_after.status == "ready"
    assert parent_after.version == 1
    assert reviewer_after is not None and reviewer_after.status == "todo"
    assert tester_after is not None and tester_after.status == "todo"
    assert event["kind"] == "completion_blocked_contract"
    assert "changed.py" in event["payload"]


def test_specifier_output_becomes_effective_goal_revision(kanban_home, monkeypatch):
    from hermes_cli import kanban_specify

    with kbc.connect_closing() as conn:
        task_id = kb.create_task(
            conn,
            title="Rough implementation idea",
            body="Draft notes.",
            triage=True,
        )
        initial = kb.get_effective_goal(conn, task_id)
        assert initial is not None

    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=json.dumps({
                        "title": "Implement the authorized change",
                        "body": "Concrete implementation contract.",
                    })
                )
            )
        ]
    )
    monkeypatch.setattr("agent.auxiliary_client.call_llm", lambda **_: response)

    outcome = kanban_specify.specify_task(task_id, author="specifier")
    assert outcome.ok is True

    with kbc.connect_closing() as conn:
        task = kb.get_task(conn, task_id)
        goal = kb.get_effective_goal(conn, task_id)
        revisions = conn.execute(
            "SELECT version, title, body, prior_version "
            "FROM task_goal_revisions WHERE task_id = ? ORDER BY version",
            (task_id,),
        ).fetchall()
        events = kb.list_events(conn, task_id)

    assert task is not None
    assert task.status == "ready"
    assert task.version == 2
    assert task.goal_revision_id == goal["id"]
    assert goal["version"] == 2
    assert goal["title"] == "Implement the authorized change"
    assert goal["body"] == "Concrete implementation contract."
    assert [
        (row["version"], row["title"], row["body"], row["prior_version"])
        for row in revisions
    ] == [
        (1, "Rough implementation idea", "Draft notes.", None),
        (2, "Implement the authorized change", "Concrete implementation contract.", 1),
    ]
    assert any(event.kind == "goal_revised" for event in events)


@pytest.mark.parametrize("operation", ["complete", "request_review"])
@pytest.mark.parametrize("race", ["goal", "review_child"])
def test_handoff_revalidates_goal_and_review_graph_at_commit(
    kanban_home, tmp_path, monkeypatch, operation, race
):
    task_id, base, head = _git_handoff_task(tmp_path, include_tester=race == "goal")
    original_prepare = kb._prepare_completion_handoff
    raced = False

    def prepare(conn, prepared_task_id, metadata, *args, **kwargs):
        nonlocal raced
        prepared = original_prepare(conn, prepared_task_id, metadata, *args, **kwargs)
        if raced:
            return prepared
        raced = True
        with kbc.connect_closing() as race_conn:
            if race == "goal":
                assert kb.update_task(
                    race_conn,
                    task_id,
                    title="Plan-only after handoff preparation",
                    body="Plan-only: no implementation is permitted.",
                    expected_version=1,
                    reason="concurrent goal revision",
                    author="cto",
                )
            else:
                late_child = kb.create_task(
                    race_conn, title="Late tester", assignee="tester"
                )
                kb.link_tasks(race_conn, task_id, late_child)
        return prepared

    monkeypatch.setattr(kb, "_prepare_completion_handoff", prepare)
    metadata = {"base_sha": base, "head_sha": head}
    with kbc.connect_closing() as conn:
        before = kb.get_task(conn, task_id)
        assert before is not None and before.status == "ready"
        with pytest.raises(kb.CompletionContractError):
            if operation == "complete":
                kb.complete_task(
                    conn,
                    task_id,
                    summary="Implementation is ready.",
                    metadata=metadata,
                )
            else:
                kb.request_review(
                    conn,
                    task_id,
                    summary="Implementation is ready.",
                    metadata=metadata,
                )
        after = kb.get_task(conn, task_id)
        events = kb.list_events(conn, task_id)
        blocked_events = [
            event for event in events if event.kind == "completion_blocked_contract"
        ]
        goal_version = (
            kb.get_effective_goal(conn, task_id)["version"] if race == "goal" else None
        )
        child_count = len(kb.child_ids(conn, task_id))

    assert raced is True
    assert after is not None
    assert after.status == before.status == "ready"
    assert "completed" not in {event.kind for event in events}
    assert "review_requested" not in {event.kind for event in events}
    assert blocked_events
    if race == "goal":
        assert after.version == 2
        assert goal_version == 2
    else:
        assert child_count == 2


def test_request_review_propagates_completion_contract_error(
    kanban_home, tmp_path, monkeypatch
):
    task_id, base, head = _git_handoff_task(tmp_path, plan_only=True)
    metadata = {"base_sha": base, "head_sha": head}

    with kbc.connect_closing() as conn:
        with pytest.raises(kb.CompletionContractError, match="plan-only"):
            kb.request_review(
                conn,
                task_id,
                summary="Implementation is ready.",
                metadata=metadata,
            )
        task = kb.get_task(conn, task_id)
        events = kb.list_events(conn, task_id)

    assert task is not None and task.status == "ready"
    assert any(event.kind == "completion_blocked_contract" for event in events)

    from tools import kanban_tools as kt

    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    output = json.loads(
        kt._handle_request_review({
            "task_id": task_id,
            "summary": "Implementation is ready.",
            "metadata": metadata,
        })
    )
    assert "error" in output
    assert "No task state changed" in output["error"]
    assert "kanban_update" in output["error"]

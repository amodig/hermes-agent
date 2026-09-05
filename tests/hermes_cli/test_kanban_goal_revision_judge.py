"""Regression coverage for effective Kanban goal revisions (issue #2)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "cto")
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
    branch = "task/issue-2"
    _git(repo, "checkout", "-b", branch)
    return repo, base, branch


def test_completion_judge_uses_authorized_effective_revision(
    kanban_home, monkeypatch
):
    from tools import kanban_tools as kt

    captured: list[str] = []
    monkeypatch.setattr(kt, "_goal_judge_available", lambda: True)

    def judge(goal, last_response, **_kwargs):
        captured.append(goal)
        assert "explicit authorization to implement" in goal
        assert "plan-only" not in goal.lower()
        assert "implemented the authorized change" in last_response
        return "done", "verified", False, None, False

    monkeypatch.setattr(kt, "judge_goal", judge)
    with kb.connect_closing() as conn:
        task_id = kb.create_task(
            conn,
            title="Plan the change",
            body="Plan-only: wait for authorization; do not implement.",
            assignee="worker",
            goal_mode=True,
        )
        kb.update_task(
            conn,
            task_id,
            title="Implement the change",
            body="Implement the change and report concrete verification.",
            expected_version=1,
            reason="explicit authorization to implement",
            author="cto",
        )

    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    out = json.loads(
        kt._handle_complete({
            "task_id": task_id,
            "summary": "implemented the authorized change; verification passed",
        })
    )
    assert out["ok"] is True
    assert captured
    with kb.connect_closing() as conn:
        assert kb.get_task(conn, task_id).status == "done"


def test_specifier_refuses_to_demote_newer_goal_revision(kanban_home, monkeypatch):
    from hermes_cli import kanban_specify

    with kb.connect_closing() as conn:
        task_id = kb.create_task(
            conn,
            title="Plan the change",
            body="Plan-only: wait for authorization; do not implement.",
            triage=True,
        )
        kb.update_task(
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
    monkeypatch.setattr("agent.auxiliary_client.call_llm", lambda **_kwargs: response)

    outcome = kanban_specify.specify_task(task_id)
    assert outcome.ok is False
    assert "kanban_update" in outcome.reason
    with kb.connect_closing() as conn:
        task = kb.get_task(conn, task_id)
        goal = kb.get_effective_goal(conn, task_id)
    assert task.status == "triage"
    assert task.title == "Implement the change"
    assert goal["version"] == 2
    assert goal["body"] == "Authorized implementation contract."


def test_plan_only_completion_rejects_nonempty_patch_for_reviewer(
    kanban_home, tmp_path
):
    repo, base, branch = _git_repo(tmp_path)
    with kb.connect_closing() as conn:
        parent = kb.create_task(
            conn,
            title="Implementation handoff",
            body="Plan-only: no implementation is permitted.",
            assignee="implementer",
            workspace_kind="worktree",
            workspace_path=str(repo),
            branch_name=branch,
        )
        kb.create_task(
            conn,
            title="Review implementation",
            assignee="reviewer",
            parents=[parent],
        )
        (repo / "changed.py").write_text("value = 1\n", encoding="utf-8")
        _git(repo, "add", "changed.py")
        _git(repo, "commit", "-m", "implementation")
        head = _git(repo, "rev-parse", "HEAD")

        with pytest.raises(kb.CompletionContractError, match="plan-only/no implementation"):
            kb.complete_task(
                conn,
                parent,
                summary="implemented",
                metadata={"base_sha": base, "head_sha": head},
            )
        assert kb.get_task(conn, parent).status == "ready"
        event = conn.execute(
            "SELECT kind FROM task_events WHERE task_id = ? ORDER BY id DESC LIMIT 1",
            (parent,),
        ).fetchone()
    assert event["kind"] == "completion_blocked_contract"

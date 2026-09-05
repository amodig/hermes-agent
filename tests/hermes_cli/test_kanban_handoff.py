"""Regression coverage for immutable Kanban task handoffs."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
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


def _repo(tmp_path: Path) -> tuple[Path, str, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-b", "main", str(repo)], check=True, capture_output=True)
    _git(repo, "config", "user.email", "kanban@example.com")
    _git(repo, "config", "user.name", "Kanban Test")
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "base")
    base = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-b", "task/immutable")
    return repo, base, "task/immutable"


def _parent_with_reviewer(conn, repo: Path, branch: str) -> tuple[str, str]:
    parent = kb.create_task(
        conn,
        title="implementation",
        assignee="implementer",
        workspace_kind="worktree",
        workspace_path=str(repo),
        branch_name=branch,
    )
    child = kb.create_task(
        conn,
        title="review implementation",
        assignee="reviewer",
        parents=[parent],
    )
    return parent, child


def _commit_change(repo: Path) -> str:
    changed = repo / "src" / "changed.py"
    changed.parent.mkdir()
    changed.write_text("value = 1\n", encoding="utf-8")
    _git(repo, "add", "src/changed.py")
    _git(repo, "commit", "-m", "implementation")
    return _git(repo, "rev-parse", "HEAD")


def test_completion_rejects_dependents_without_recorded_head(kanban_home, tmp_path):
    repo, base, branch = _repo(tmp_path)
    with kb.connect() as conn:
        parent, _ = _parent_with_reviewer(conn, repo, branch)
        _commit_change(repo)
        with pytest.raises(kb.HandoffValidationError, match="head_sha is required"):
            kb.complete_task(conn, parent, summary="implemented", metadata={"base_sha": base})
        event = conn.execute(
            "SELECT kind FROM task_events WHERE task_id = ? ORDER BY id DESC LIMIT 1",
            (parent,),
        ).fetchone()
        assert event["kind"] == "completion_blocked_handoff"


def test_completion_records_git_changed_files(kanban_home, tmp_path):
    repo, base, branch = _repo(tmp_path)
    with kb.connect() as conn:
        parent, _ = _parent_with_reviewer(conn, repo, branch)
        head = _commit_change(repo)
        assert kb.complete_task(
            conn,
            parent,
            summary="implemented",
            metadata={"base_sha": base, "head_sha": head, "changed_files": []},
        )
        row = conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'completed' "
            "ORDER BY id DESC LIMIT 1",
            (parent,),
        ).fetchone()
        payload = json.loads(row["payload"])
        assert payload["changed_files"] == ["src/changed.py"]


def test_reviewer_context_contains_exact_parent_handoff(kanban_home, tmp_path):
    repo, base, branch = _repo(tmp_path)
    with kb.connect() as conn:
        parent, child = _parent_with_reviewer(conn, repo, branch)
        head = _commit_change(repo)
        kb.complete_task(
            conn,
            parent,
            summary="implemented",
            metadata={"base_sha": base, "head_sha": head},
        )
        context = kb.build_worker_context(conn, child)
        assert base in context
        assert head in context
        assert branch in context
        assert "src/changed.py" in context
        assert "moving branch tip" in context


def test_claim_rejects_parent_head_movement_with_explicit_event(kanban_home, tmp_path):
    repo, base, branch = _repo(tmp_path)
    with kb.connect() as conn:
        parent, child = _parent_with_reviewer(conn, repo, branch)
        head = _commit_change(repo)
        kb.complete_task(
            conn,
            parent,
            summary="implemented",
            metadata={"base_sha": base, "head_sha": head},
        )
        (repo / "src" / "changed.py").write_text("value = 2\n", encoding="utf-8")
        _git(repo, "add", "src/changed.py")
        _git(repo, "commit", "post-approval change")
        assert kb.claim_task(conn, child) is None
        row = conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'handoff_head_moved' "
            "ORDER BY id DESC LIMIT 1",
            (child,),
        ).fetchone()
        payload = json.loads(row["payload"])
        assert payload["expected_head_sha"] == head
        assert payload["actual_head_sha"] != head

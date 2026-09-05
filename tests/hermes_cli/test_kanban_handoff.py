"""Regression coverage for immutable Kanban task handoffs."""

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

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
    monkeypatch.setenv("HERMES_PROFILE", "test-orchestrator")
    return home


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _repo(tmp_path: Path) -> tuple[Path, str, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(
        ["git", "init", "-b", "main", str(repo)], check=True, capture_output=True
    )
    _git(repo, "config", "user.email", "kanban@example.com")
    _git(repo, "config", "user.name", "Kanban Test")
    (repo / "README.md").write_text("base\n")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "base")
    base = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-b", "task/immutable")
    return repo, base, "task/immutable"


def _lane(conn, repo: Path, branch: str) -> tuple[str, str]:
    parent = kb.create_task(
        conn,
        title="implementation",
        assignee="implementer",
        workspace_kind="worktree",
        workspace_path=str(repo),
        branch_name=branch,
    )
    child = kb.create_task(conn, title="review", assignee="reviewer", parents=[parent])
    return parent, child


def _commit(repo: Path, path: str = "src/changed.py") -> str:
    target = repo / path
    target.parent.mkdir(exist_ok=True)
    target.write_text("value = 1\n")
    _git(repo, "add", path)
    _git(repo, "commit", "-m", "implementation")
    return _git(repo, "rev-parse", "HEAD")


def _legacy_complete(conn, task_id: str, metadata: dict) -> None:
    with kb.write_txn(conn):
        run_id = kb._end_run(
            conn,
            task_id,
            outcome="completed",
            status="completed",
            summary="legacy implementation",
            metadata=metadata,
        )
        conn.execute(
            "UPDATE tasks SET status = 'done', completed_at = 1, claim_lock = NULL, "
            "claim_expires = NULL, worker_pid = NULL WHERE id = ?",
            (task_id,),
        )
        kb._append_event(conn, task_id, "completed", metadata, run_id=run_id)


def test_completion_records_and_enforces_exact_head(kanban_home, tmp_path):
    repo, base, branch = _repo(tmp_path)
    with kbc.connect_closing() as conn:
        parent, child = _lane(conn, repo, branch)
        head = _commit(repo)
        with pytest.raises(kb.HandoffValidationError, match="head_sha is required"):
            kb.complete_task(conn, parent, metadata={"base_sha": base})
        assert kb.complete_task(
            conn, parent, metadata={"base_sha": base, "head_sha": head}
        )
        handoff = kb.latest_handoff(conn, parent)
        assert handoff["changed_files"] == ["src/changed.py"]
        context = kb.build_worker_context(conn, child)
        assert base in context and head in context and "moving branch tip" in context
        (repo / "src/changed.py").write_text("value = 2\n")
        _git(repo, "add", "src/changed.py")
        _git(repo, "commit", "-m", "moved")
        assert kb.claim_task(conn, child) is None
        event = [
            e for e in kb.list_events(conn, child) if e.kind == "handoff_head_moved"
        ][-1]
        assert event.payload["expected_head_sha"] == head


def test_legacy_handoff_requeues_and_recompletes_same_lane(kanban_home, tmp_path):
    from tools import kanban_tools as kt

    repo, base, branch = _repo(tmp_path)
    with kbc.connect_closing() as conn:
        parent, reviewer = _lane(conn, repo, branch)
        tester = kb.create_task(
            conn, title="test", assignee="tester", parents=[reviewer]
        )
        kb.claim_task(conn, parent)
        kb.add_comment(conn, parent, "operator", "preserve legacy context")
        changed = repo / "src/legacy.py"
        changed.parent.mkdir()
        changed.write_text("value = 1\n")
        patch = tmp_path / "legacy.patch"
        patch.write_text(_git(repo, "diff", "--binary"))
        _legacy_complete(conn, parent, {"changed_files": ["src/legacy.py"]})
        kb.recompute_ready(conn)
        assert kb.claim_task(conn, reviewer) is None
        assert kb.claim_task(conn, reviewer) is None
        assert (
            len([
                e
                for e in kb.list_events(conn, reviewer)
                if e.kind == "handoff_unverifiable"
            ])
            == 1
        )

    stale = json.loads(
        kt._handle_requeue_handoff({
            "board": "default",
            "task_id": parent,
            "expected_version": 99,
            "reason": "repair",
        })
    )
    assert "update conflict" in stale["error"]
    repaired = json.loads(
        kt._handle_requeue_handoff({
            "board": "default",
            "task_id": parent,
            "expected_version": 1,
            "reason": "repair",
            "base_sha": base,
            "branch_name": branch,
            "workspace_path": str(repo),
            "patch_artifact": str(patch),
            "patch_sha256": hashlib.sha256(patch.read_bytes()).hexdigest(),
        })
    )
    assert repaired["ok"] and repaired["status"] == "ready"

    with kbc.connect_closing() as conn:
        assert kb.get_task(conn, reviewer).status == "todo"
        assert kb.parent_ids(conn, reviewer) == [parent]
        assert len(kb.list_comments(conn, parent)) == 1
        assert len(kb.list_runs(conn, parent)) == 1
        run = kb.claim_task(conn, parent)
        head = _commit(repo, "src/legacy.py")
        assert kb.complete_task(
            conn,
            parent,
            metadata={"base_sha": base, "head_sha": head},
            expected_run_id=run.current_run_id,
        )
        handoff = kb.latest_handoff(conn, parent)
        assert handoff["provenance"]["kind"] == "legacy_handoff_recompletion"
        review_run = kb.claim_task(conn, reviewer)
        assert review_run is not None
        assert kb.complete_task(
            conn, reviewer, expected_run_id=review_run.current_run_id
        )
        assert kb.get_task(conn, tester).status == "ready"
        reviewed = kb.latest_handoff(conn, reviewer)
        assert reviewed["head_sha"] == head
        assert reviewed["handoff_provenance"]["kind"] == "reviewed_parent"

    shown = json.loads(kt._handle_show({"board": "default", "task_id": parent}))
    assert shown["active_handoff"]["head_sha"] == head


def test_legacy_requeue_rejects_tampered_or_missing_patch(kanban_home, tmp_path):
    repo, base, branch = _repo(tmp_path)
    with kbc.connect_closing() as conn:
        parent, _ = _lane(conn, repo, branch)
        kb.claim_task(conn, parent)
        patch = tmp_path / "legacy.patch"
        patch.write_text("")
        metadata = {
            "base_sha": base,
            "branch_name": branch,
            "workspace_path": str(repo),
            "patch_artifact": str(patch),
            "patch_sha256": hashlib.sha256(patch.read_bytes()).hexdigest(),
        }
        _legacy_complete(conn, parent, metadata)
        patch.write_text("tampered\n")
        with pytest.raises(kb.HandoffValidationError, match="does not match"):
            kb.requeue_legacy_handoff(conn, parent, expected_version=1, reason="repair")
        patch.unlink()
        with pytest.raises(kb.HandoffValidationError, match="does not exist"):
            kb.requeue_legacy_handoff(conn, parent, expected_version=1, reason="repair")

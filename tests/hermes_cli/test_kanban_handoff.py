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


def _lane(
    conn, repo: Path, branch: str, *, validation_required: bool = False
) -> tuple[str, str]:
    parent = kb.create_task(
        conn,
        title="implementation",
        assignee="implementer",
        workspace_kind="worktree",
        workspace_path=str(repo),
        branch_name=branch,
        lifecycle_contract={
            "kind": "code",
            "review_mode": "separate_card",
            "reviewer": "reviewer",
            "validation_required": validation_required,
        },
    )
    child = kb.create_task(
        conn,
        title="review",
        assignee="reviewer",
        parents=[parent],
        lifecycle_contract={"kind": "review", "candidate_task_id": parent},
    )
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
        implementation_run = kb.claim_task(conn, parent)
        assert implementation_run is not None
        head = _commit(repo)
        with pytest.raises(kb.HandoffValidationError, match="head_sha is required"):
            kb.complete_task(conn, parent, metadata={"base_sha": base})
        assert kb.complete_task(
            conn,
            parent,
            expected_run_id=implementation_run.current_run_id,
            metadata={"base_sha": base, "head_sha": head},
        )
        handoff = kb.latest_handoff(conn, parent)
        assert handoff["changed_files"] == ["src/changed.py"]
        context = kb.build_worker_context(conn, child)
        assert base in context and head in context
        assert kb.get_task(conn, child).status == "review"
        (repo / "src/changed.py").write_text("value = 2\n")
        _git(repo, "add", "src/changed.py")
        _git(repo, "commit", "-m", "moved")
        assert kb.claim_review_task(conn, child) is None
        event = [
            e for e in kb.list_events(conn, child) if e.kind == "handoff_head_moved"
        ][-1]
        assert event.payload["expected_head_sha"] == head


def test_review_claim_aborts_after_parent_head_moves(kanban_home, tmp_path):
    repo, base, branch = _repo(tmp_path)
    with kbc.connect_closing() as conn:
        parent, reviewer = _lane(conn, repo, branch)
        head = _commit(repo)
        assert kb.claim_task(conn, parent) is not None
        assert kb.complete_task(
            conn, parent, metadata={"base_sha": base, "head_sha": head}
        )
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET lifecycle_contract = NULL, status = 'review' "
                "WHERE id = ?", (reviewer,)
            )
        _commit(repo, "src/moved.py")

        assert kb.claim_review_task(conn, reviewer) is None
        assert kb.get_task(conn, reviewer).status == "review"
        assert any(
            event.kind == "handoff_head_moved"
            for event in kb.list_events(conn, reviewer)
        )


def test_legacy_handoff_requeues_and_recompletes_same_lane(kanban_home, tmp_path):
    from tools import kanban_tools as kt

    repo, base, branch = _repo(tmp_path)
    with kbc.connect_closing() as conn:
        parent, reviewer = _lane(conn, repo, branch, validation_required=True)
        tester = kb.create_task(
            conn,
            title="test",
            assignee="tester",
            parents=[reviewer],
            lifecycle_contract={"kind": "validation", "candidate_task_id": parent},
        )
        kb.claim_task(conn, parent)
        kb.add_comment(conn, parent, "operator", "preserve legacy context")
        changed = repo / "src/legacy.py"
        changed.parent.mkdir()
        changed.write_text("value = 1\n")
        patch = tmp_path / "legacy.patch"
        patch.write_text(_git(repo, "diff", "--binary"))
        _legacy_complete(conn, parent, {"changed_files": ["src/legacy.py"]})
        # Historical role cards have no contract or typed edge requirements.
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET lifecycle_contract = NULL WHERE id = ?", (reviewer,)
            )
            conn.execute(
                "UPDATE task_links SET requirement = NULL "
                "WHERE parent_id = ? OR child_id = ?", (reviewer, reviewer)
            )
        kb.recompute_ready(conn)
        assert kb.get_task(conn, reviewer).status == "ready"
        assert kb.bind_lifecycle_contract(
            conn,
            reviewer,
            {"kind": "review", "candidate_task_id": parent},
            expected_version=kb.get_task(conn, reviewer).version,
            reason="classify the historical review before dispatch",
            author="operator",
        )
        parent_version = kb.get_task(conn, parent).version
        assert kb.claim_review_task(conn, reviewer) is None
        assert kb.claim_review_task(conn, reviewer) is None
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
            "expected_version": parent_version + 1,
            "reason": "repair",
        })
    )
    assert "update conflict" in stale["error"]
    repaired = json.loads(
        kt._handle_requeue_handoff({
            "board": "default",
            "task_id": parent,
            "expected_version": parent_version,
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
        review_run = kb.claim_review_task(conn, reviewer)
        assert review_run is not None
        assert kb.complete_task(
            conn,
            reviewer,
            expected_run_id=review_run.current_run_id,
            verdict="APPROVE",
            metadata={"reviewed_head_sha": head},
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


def _rework_graph(conn, repo: Path, base: str, branch: str) -> tuple[str, str, str, str, str]:
    implementation = kb.create_task(
        conn,
        title="implementation",
        assignee="implementer",
        workspace_kind="worktree",
        workspace_path=str(repo),
        branch_name=branch,
        lifecycle_contract={
            "kind": "code",
            "review_mode": "separate_card",
            "reviewer": "reviewer",
            "validation_required": True,
        },
    )
    reviewer = kb.create_task(
        conn,
        title="review",
        assignee="reviewer",
        parents=[implementation],
        lifecycle_contract={"kind": "review", "candidate_task_id": implementation},
    )
    tester = kb.create_task(
        conn,
        title="validation",
        assignee="tester",
        parents=[reviewer],
        initial_status="blocked",
        lifecycle_contract={"kind": "validation", "candidate_task_id": implementation},
    )
    descendant = kb.create_task(
        conn, title="publish", assignee="publisher", parents=[tester]
    )

    implementation_run = kb.claim_task(conn, implementation, claimer="implementer:1")
    assert implementation_run is not None
    first_head = _commit(repo, "src/rejected.py")
    assert kb.complete_task(
        conn,
        implementation,
        expected_run_id=implementation_run.current_run_id,
        metadata={"base_sha": base, "head_sha": first_head},
    )
    reviewer_run = kb.claim_review_task(conn, reviewer, claimer="reviewer:1")
    assert reviewer_run is not None
    assert kb.complete_task(
        conn,
        reviewer,
        expected_run_id=reviewer_run.current_run_id,
        verdict="REQUEST_CHANGES",
        metadata={"reviewed_head_sha": first_head},
    )
    kb.add_comment(conn, implementation, "operator", "rework is required")
    return implementation, reviewer, tester, descendant, first_head


def test_typed_same_card_rework_refuses_active_implementation(kanban_home, tmp_path):
    repo, base, _branch = _repo(tmp_path)
    with kbc.connect_closing() as conn:
        implementation = kb.create_task(
            conn,
            title="typed same-card implementation",
            assignee="implementer",
            initial_status="blocked",
            workspace_kind="dir",
            workspace_path=str(repo),
            lifecycle_contract={
                "kind": "code",
                "review_mode": "same_card",
                "reviewer": "reviewer",
                "validation_required": False,
            },
        )
        assert kb.unblock_task(conn, implementation)
        head = _commit(repo)
        implementation_run = kb.claim_task(conn, implementation, claimer="implementer:1")
        assert implementation_run is not None
        assert kb.complete_task(
            conn,
            implementation,
            expected_run_id=implementation_run.current_run_id,
            metadata={"base_sha": base, "head_sha": head},
        )
        review_run = kb.claim_review_task(conn, implementation, claimer="reviewer:1")
        assert review_run is not None
        assert kb.complete_task(
            conn,
            implementation,
            expected_run_id=review_run.current_run_id,
            verdict="REQUEST_CHANGES",
            summary="repair required",
            metadata={"reviewed_head_sha": head},
        )
        awaiting_rework = kb.get_task(conn, implementation)
        assert awaiting_rework is not None
        assert awaiting_rework.status == "ready"

        active = kb.claim_task(conn, implementation, claimer="implementer:2")
        assert active is not None
        with pytest.raises(ValueError, match="is claimed"):
            kb.rework_review_graph(
                conn,
                implementation,
                expected_implementation_version=active.version,
                reason="repair the rejected implementation",
            )

        unchanged = kb.get_task(conn, implementation)
        assert unchanged is not None
        assert unchanged.status == "running"
        assert unchanged.current_run_id == active.current_run_id
        assert unchanged.claim_lock == active.claim_lock


def test_typed_same_card_rework_rejects_newer_review_handoff(kanban_home, tmp_path):
    repo, base, _branch = _repo(tmp_path)
    with kbc.connect_closing() as conn:
        implementation = kb.create_task(
            conn,
            title="stale same-card implementation",
            assignee="implementer",
            initial_status="blocked",
            workspace_kind="dir",
            workspace_path=str(repo),
            lifecycle_contract={
                "kind": "code",
                "review_mode": "same_card",
                "reviewer": "reviewer",
                "validation_required": False,
            },
        )
        assert kb.unblock_task(conn, implementation)
        first_run = kb.claim_task(conn, implementation, claimer="implementer:1")
        assert first_run is not None
        first_head = _commit(repo, "src/first.py")
        assert kb.complete_task(
            conn,
            implementation,
            expected_run_id=first_run.current_run_id,
            metadata={"base_sha": base, "head_sha": first_head},
        )
        review_run = kb.claim_review_task(conn, implementation, claimer="reviewer:1")
        assert review_run is not None
        assert kb.complete_task(
            conn,
            implementation,
            expected_run_id=review_run.current_run_id,
            verdict="REQUEST_CHANGES",
            summary="repair required",
            metadata={"reviewed_head_sha": first_head},
        )

        second_run = kb.claim_task(conn, implementation, claimer="implementer:2")
        assert second_run is not None
        second_head = _commit(repo, "src/second.py")
        assert kb.complete_task(
            conn,
            implementation,
            expected_run_id=second_run.current_run_id,
            metadata={"base_sha": base, "head_sha": second_head},
        )
        assert kb.get_task(conn, implementation).status == "review"

        task = kb.get_task(conn, implementation)
        assert task is not None
        with pytest.raises(ValueError, match="evidence is stale"):
            kb.rework_review_graph(
                conn,
                implementation,
                expected_implementation_version=task.version,
                reason="reject stale rework request",
            )

        assert kb.get_task(conn, implementation).status == "review"

def test_typed_same_card_rework_rechecks_evidence_inside_transaction(
    kanban_home, tmp_path, monkeypatch
):
    repo, base, _branch = _repo(tmp_path)
    with kbc.connect_closing() as conn:
        implementation = kb.create_task(
            conn,
            title="racing same-card implementation",
            assignee="implementer",
            initial_status="blocked",
            workspace_kind="dir",
            workspace_path=str(repo),
            lifecycle_contract={
                "kind": "code",
                "review_mode": "same_card",
                "reviewer": "reviewer",
                "validation_required": False,
            },
        )
        assert kb.unblock_task(conn, implementation)
        first_run = kb.claim_task(conn, implementation, claimer="implementer:1")
        assert first_run is not None
        first_head = _commit(repo, "src/first.py")
        assert kb.complete_task(
            conn,
            implementation,
            expected_run_id=first_run.current_run_id,
            metadata={"base_sha": base, "head_sha": first_head},
        )
        review_run = kb.claim_review_task(conn, implementation, claimer="reviewer:1")
        assert review_run is not None
        assert kb.complete_task(
            conn,
            implementation,
            expected_run_id=review_run.current_run_id,
            verdict="REQUEST_CHANGES",
            summary="repair required",
            metadata={"reviewed_head_sha": first_head},
        )
        task = kb.get_task(conn, implementation)
        assert task is not None

        expected_version = task.version

        import hermes_cli.kanban_db_lifecycle_rework as rework

        original_implementation_routing = rework._implementation_routing
        raced = False
        second_review = None

        def racing_implementation_routing(connection, task_id):
            nonlocal raced, second_review
            routing = original_implementation_routing(connection, task_id)
            if not raced:
                raced = True
                second_run = kb.claim_task(conn, implementation, claimer="implementer:2")
                assert second_run is not None
                second_head = _commit(repo, "src/second.py")
                assert kb.complete_task(
                    conn,
                    implementation,
                    expected_run_id=second_run.current_run_id,
                    metadata={"base_sha": base, "head_sha": second_head},
                )
                second_review = kb.claim_review_task(
                    conn, implementation, claimer="reviewer:2"
                )
                assert second_review is not None
                assert kb.complete_task(
                    conn,
                    implementation,
                    expected_run_id=second_review.current_run_id,
                    verdict="REQUEST_CHANGES",
                    summary="second repair required",
                    metadata={"reviewed_head_sha": second_head},
                )
            return routing

        monkeypatch.setattr(
            rework, "_implementation_routing", racing_implementation_routing
        )
        with pytest.raises(ValueError, match="evidence changed"):
            kb.rework_review_graph(
                conn,
                implementation,
                expected_implementation_version=expected_version,
                reason="reject stale rework request",
            )
        current = kb.get_task(conn, implementation)
        assert current is not None
        assert raced
        assert second_review is not None
        assert current.status == "ready"
        assert current.current_run_id is None
        assert current.version == expected_version


def test_separate_reviewer_requires_explicit_approval(kanban_home, tmp_path):
    repo, base, branch = _repo(tmp_path)
    with kbc.connect_closing() as conn:
        implementation = kb.create_task(
            conn,
            title="implementation",
            assignee="implementer",
            workspace_kind="worktree",
            workspace_path=str(repo),
            branch_name=branch,
            lifecycle_contract={
                "kind": "code",
                "review_mode": "separate_card",
                "reviewer": "reviewer",
                "validation_required": True,
            },
        )
        reviewer = kb.create_task(
            conn,
            title="review",
            assignee="reviewer",
            parents=[implementation],
            lifecycle_contract={"kind": "review", "candidate_task_id": implementation},
        )
        tester = kb.create_task(
            conn,
            title="validation",
            assignee="tester",
            parents=[reviewer],
            lifecycle_contract={"kind": "validation", "candidate_task_id": implementation},
        )
        implementation_run = kb.claim_task(conn, implementation, claimer="implementer:1")
        assert implementation_run is not None
        head = _commit(repo, "src/approved.py")
        assert kb.complete_task(
            conn,
            implementation,
            expected_run_id=implementation_run.current_run_id,
            metadata={"base_sha": base, "head_sha": head},
        )
        reviewer_run = kb.claim_review_task(conn, reviewer, claimer="reviewer:1")
        assert reviewer_run is not None
        with pytest.raises(kb.LifecycleEvidenceError):
            kb.complete_task(
                conn,
                reviewer,
                expected_run_id=reviewer_run.current_run_id,
                metadata={"reviewed_head_sha": head},
            )
        assert kb.get_task(conn, reviewer).current_run_id == reviewer_run.current_run_id
        assert kb.get_task(conn, tester).status == "todo"
        assert kb.claim_task(conn, tester) is None
        assert kb.complete_task(
            conn,
            reviewer,
            expected_run_id=reviewer_run.current_run_id,
            verdict="APPROVE",
            metadata={"reviewed_head_sha": head},
        )
        assert kb.get_task(conn, tester).status == "ready"


def test_rework_review_accepts_todo_tester_for_second_rejection(kanban_home, tmp_path):
    repo, base, branch = _repo(tmp_path)
    with kbc.connect_closing() as conn:
        implementation, reviewer, tester, _descendant, first_head = _rework_graph(
            conn, repo, base, branch
        )
        ids = (implementation, reviewer, tester)
        versions = tuple(kb.get_task(conn, task_id).version for task_id in ids)
        kb.rework_review_graph(
            conn,
            *ids,
            expected_implementation_version=versions[0],
            expected_reviewer_version=versions[1],
            expected_tester_version=versions[2],
            reason="first repair",
        )
        assert kb.get_task(conn, tester).status == "todo"

        implementation_run = kb.claim_task(conn, implementation, claimer="implementer:2")
        assert implementation_run is not None
        second_head = _commit(repo, "src/repaired.py")
        assert kb.complete_task(
            conn,
            implementation,
            expected_run_id=implementation_run.current_run_id,
            metadata={"base_sha": base, "head_sha": second_head},
        )
        reviewer_run = kb.claim_review_task(conn, reviewer, claimer="reviewer:2")
        assert reviewer_run is not None
        assert kb.complete_task(
            conn,
            reviewer,
            expected_run_id=reviewer_run.current_run_id,
            verdict="REQUEST_CHANGES",
            metadata={"reviewed_head_sha": second_head},
        )
        versions = tuple(kb.get_task(conn, task_id).version for task_id in ids)
        result = kb.rework_review_graph(
            conn,
            *ids,
            expected_implementation_version=versions[0],
            expected_reviewer_version=versions[1],
            expected_tester_version=versions[2],
            reason="second repair",
        )
        assert result["status"] == "ready"
        assert kb.get_task(conn, tester).status == "todo"
        assert first_head != second_head


@pytest.mark.parametrize("block_kind", ["needs_input", "capability"])
def test_rework_preserves_sticky_blocked_descendant(
    kanban_home, tmp_path, block_kind
):
    repo, base, branch = _repo(tmp_path)
    with kbc.connect_closing() as conn:
        implementation, reviewer, tester, descendant, _head = _rework_graph(
            conn, repo, base, branch
        )
        sticky = kb.create_task(conn, title="owner approval", assignee="publisher")
        assert kb.block_task(
            conn, sticky, reason="owner approval required", kind=block_kind
        )
        kb.link_tasks(conn, tester, sticky)
        before = kb.get_task(conn, sticky)
        assert before is not None
        versions = tuple(
            kb.get_task(conn, task_id).version
            for task_id in (implementation, reviewer, tester)
        )
        result = kb.rework_review_graph(
            conn,
            implementation,
            reviewer,
            tester,
            expected_implementation_version=versions[0],
            expected_reviewer_version=versions[1],
            expected_tester_version=versions[2],
            reason="preserve owner gate",
        )
        after = kb.get_task(conn, sticky)
        assert after is not None
        assert result["status"] == "ready"
        assert after.status == "blocked"
        assert after.block_kind == block_kind
        assert after.block_recurrences == before.block_recurrences
        assert kb.get_task(conn, descendant).status == "todo"
        assert any(event.kind == "blocked" for event in kb.list_events(conn, sticky))
        assert any(
            event.kind == "acceptance_invalidated"
            for event in kb.list_events(conn, sticky)
        )


def test_rework_review_reuses_cards_and_requires_new_head_approval(
    kanban_home, tmp_path
):
    """A rejected separate-card review is reworked in place, never duplicated."""
    repo, base, branch = _repo(tmp_path)
    kb.create_board("rework")
    with kb.scoped_current_board("rework"), kbc.connect_closing() as conn:
        implementation, reviewer, tester, descendant, first_head = _rework_graph(
            conn, repo, base, branch
        )
        assert kb.get_task(conn, tester).status == "blocked"

        ids = (implementation, reviewer, tester)
        versions = tuple(kb.get_task(conn, task_id).version for task_id in ids)
        before_runs = {
            task_id: len(kb.list_runs(conn, task_id)) for task_id in ids
        }
        result = kb.rework_review_graph(
            conn,
            *ids,
            expected_implementation_version=versions[0],
            expected_reviewer_version=versions[1],
            expected_tester_version=versions[2],
            reason="repair the rejected candidate",
            author="operator",
        )
        assert result["rejected_head_sha"] == first_head
        assert result["status"] == "ready"
        assert {entry["id"] for entry in result["invalidated"]} == {
            implementation,
            reviewer,
            tester,
            descendant,
        }
        assert kb.parent_ids(conn, reviewer) == [implementation]
        assert kb.parent_ids(conn, tester) == [reviewer]
        assert kb.parent_ids(conn, descendant) == [tester]
        assert kb.get_task(conn, implementation).workspace_path == str(repo)
        assert kb.get_task(conn, implementation).status == "ready"
        assert kb.get_task(conn, reviewer).status == "todo"
        assert kb.get_task(conn, tester).status == "todo"
        assert kb.get_task(conn, descendant).status == "todo"
        assert {
            task_id: len(kb.list_runs(conn, task_id)) for task_id in ids
        } == before_runs
        assert len(kb.list_comments(conn, implementation)) == 1
        assert len(
            [event for event in kb.list_events(conn, implementation)
             if event.kind == "review_rework_requested"]
        ) == 1

        # A different request with stale CAS versions must not apply rework twice.
        with pytest.raises(ValueError):
            kb.rework_review_graph(
                conn,
                *ids,
                expected_implementation_version=versions[0],
                expected_reviewer_version=versions[1],
                expected_tester_version=versions[2],
                reason="retry",
            )
        assert len(
            [event for event in kb.list_events(conn, implementation)
             if event.kind == "review_rework_requested"]
        ) == 1

        implementation_run = kb.claim_task(
            conn, implementation, claimer="implementer:2"
        )
        assert implementation_run is not None
        second_head = _commit(repo, "src/repaired.py")
        assert kb.complete_task(
            conn,
            implementation,
            expected_run_id=implementation_run.current_run_id,
            metadata={"base_sha": base, "head_sha": second_head},
        )
        reviewer_run = kb.claim_review_task(conn, reviewer, claimer="reviewer:2")
        assert reviewer_run is not None
        with pytest.raises(kb.HandoffValidationError):
            kb.complete_task(
                conn,
                reviewer,
                expected_run_id=reviewer_run.current_run_id,
                verdict="APPROVE",
                metadata={"head_sha": first_head},
            )
        assert kb.get_task(conn, tester).status == "todo"
        assert kb.claim_task(conn, tester) is None
        assert kb.complete_task(
            conn,
            reviewer,
            expected_run_id=reviewer_run.current_run_id,
            verdict="APPROVE",
            metadata={"reviewed_head_sha": second_head},
        )
        assert kb.get_task(conn, tester).status == "ready"
        tester_run = kb.claim_task(conn, tester, claimer="tester:1")
        assert tester_run is not None
        assert kb.complete_task(
            conn,
            tester,
            expected_run_id=tester_run.current_run_id,
            verdict="PASS",
            metadata={"reviewed_head_sha": second_head},
            summary="validated",
        )
        assert kb.get_task(conn, tester).status == "done"
        assert kb.get_task(conn, descendant).status == "ready"
        assert kb.latest_handoff(conn, implementation)["head_sha"] == second_head
        assert first_head != second_head

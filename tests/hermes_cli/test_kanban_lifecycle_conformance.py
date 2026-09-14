"""Hermetic end-to-end contract proof for the typed Kanban lifecycle.

This file is intentionally runnable with ``python -m unittest`` as well as the
canonical pytest runner.  It exercises the real SQLite kernel, a real Git
handoff, the dispatcher/recovery boundary, CLI parsing, dashboard projection,
notifier formatting, and the frozen worker identity protocol.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sqlite3
import sys
import subprocess
import time
import tempfile
import unittest
from types import SimpleNamespace

from gateway import kanban_watchers_notifier as notifier
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_runtime as runtime
from hermes_cli.kanban_lifecycle import get_lifecycle_state
from hermes_cli.kanban_parser import build_parser
from hermes_cli.kanban_runtime import (
    RuntimeIdentityError,
    _fingerprint,
    assert_runtime_import_root,
    code_identity,
    process_start_time,
    runtime_identity,
    same_code_identity,
)
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
RUNTIME_ROOT = Path(os.environ.get("HERMES_CONFORMANCE_RUNTIME_ROOT", ROOT)).resolve()
FIXTURE = ROOT / "tests" / "fixtures" / "kanban_lifecycle_v0.json"


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


class KanbanLifecycleConformance(unittest.TestCase):
    def setUp(self) -> None:
        self.home = tempfile.TemporaryDirectory(prefix="kanban-conformance-home-")
        self.env = patch.dict(
            "os.environ",
            {
                "HERMES_HOME": self.home.name,
                "HERMES_KANBAN_HOME": self.home.name,
                "HERMES_PROFILE": "cto",
            },
            clear=False,
        )
        self.env.start()
        self.conn = sqlite3.connect(":memory:")
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(kb.SCHEMA_SQL)
        kb._ensure_lifecycle_schema(self.conn)
        kb._ensure_goal_revision_schema(self.conn)

    def tearDown(self) -> None:
        self.conn.close()
        self.env.stop()
        self.home.cleanup()

    def _task(self, task_id: str):
        task = kb.get_task(self.conn, task_id)
        self.assertIsNotNone(task)
        return task

    def _move(self, trace: list[str], alias: str, operation) -> None:
        before = self._task(self.ids[alias]).status
        result = operation()
        self.assertNotEqual(result, False)
        after = self._task(self.ids[alias]).status
        trace.append(f"{alias}:{before}->{after}")

    def _load_fixture(self) -> dict:
        fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
        self.assertEqual(fixture["fixture_version"], 0)
        return fixture

    def test_separate_card_fixture_releases_acceptance_in_order(self) -> None:
        fixture = self._load_fixture()
        with tempfile.TemporaryDirectory(prefix="kanban-conformance-git-") as raw_repo:
            repo = Path(raw_repo)
            _git(repo, "init", "-q")
            _git(repo, "config", "user.email", "conformance@example.invalid")
            _git(repo, "config", "user.name", "Kanban Conformance")
            (repo / "README").write_text("base\n", encoding="utf-8")
            _git(repo, "add", "README")
            _git(repo, "commit", "-qm", "base")
            base_sha = _git(repo, "rev-parse", "HEAD")
            (repo / "lifecycle.py").write_text("print('proof')\n", encoding="utf-8")
            _git(repo, "add", "lifecycle.py")
            _git(repo, "commit", "-qm", "implementation")
            head_sha = _git(repo, "rev-parse", "HEAD")

            implementation_spec = fixture["tasks"][0]
            implementation = kb.create_task(
                self.conn,
                title=implementation_spec["title"],
                assignee=implementation_spec["assignee"],
                initial_status=implementation_spec["status"],
                workspace_kind="scratch",
                workspace_path=str(repo),
                lifecycle_contract=implementation_spec["lifecycle_contract"],
            )
            review_spec = fixture["tasks"][1]
            review = kb.create_task(
                self.conn,
                title=review_spec["title"],
                assignee=review_spec["assignee"],
                initial_status=review_spec["status"],
                lifecycle_contract={"kind": "review", "candidate_task_id": implementation},
            )
            validation_spec = fixture["tasks"][2]
            validation = kb.create_task(
                self.conn,
                title=validation_spec["title"],
                assignee=validation_spec["assignee"],
                initial_status=validation_spec["status"],
                lifecycle_contract={"kind": "validation", "candidate_task_id": implementation},
            )
            self.ids = {"implementation": implementation, "review": review, "validation": validation}
            for edge in fixture["edges"]:
                parent = self.ids[edge["parent"]]
                child = self.ids[edge["child"]]
                kb.link_tasks(self.conn, parent, child, requirement=edge["requirement"])

            trace: list[str] = []
            self._move(trace, "implementation", lambda: kb.unblock_task(self.conn, implementation))
            impl_run = kb.claim_task(self.conn, implementation, claimer="implementer:conformance")
            self.assertIsNotNone(impl_run)
            trace.append("implementation:ready->running")
            self.assertTrue(
                kb.complete_task(
                    self.conn,
                    implementation,
                    expected_run_id=impl_run.current_run_id,
                    summary="Implementation evidence",
                    metadata={
                        "base_sha": base_sha,
                        "head_sha": head_sha,
                        "changed_files": ["lifecycle.py"],
                    },
                )
            )
            trace.append("implementation:running->done")
            self.assertEqual(self._task(review).status, "review")
            trace.append("review:blocked->review")

            review_run = kb.claim_review_task(self.conn, review, claimer="reviewer:conformance")
            self.assertIsNotNone(review_run, msg=f"review status={self._task(review).status} dependencies={kb.evaluate_dependencies(self.conn, review)}")
            trace.append("review:review->running")
            self.assertTrue(
                kb.complete_task(
                    self.conn,
                    review,
                    expected_run_id=review_run.current_run_id,
                    verdict="APPROVE",
                    summary="Reviewed exact implementation head",
                    metadata={"reviewed_head_sha": head_sha},
                )
            )
            trace.append("review:running->done")
            self.assertEqual(self._task(validation).status, "ready")
            trace.append("validation:blocked->ready")

            validation_run = kb.claim_task(self.conn, validation, claimer="tester:conformance")
            self.assertIsNotNone(validation_run)
            trace.append("validation:ready->running")
            self.assertTrue(
                kb.complete_task(
                    self.conn,
                    validation,
                    expected_run_id=validation_run.current_run_id,
                    verdict="PASS",
                    summary="Validation passed on reviewed head",
                    metadata={"head_sha": head_sha},
                )
            )
            trace.append("validation:running->done")
            acceptance = get_lifecycle_state(self.conn, implementation)
            self.assertEqual(acceptance["acceptance"], "accepted")
            self.assertEqual(acceptance["review_verdict"], "APPROVE")
            self.assertEqual(acceptance["validation_verdict"], "PASS")
            trace.append("implementation:acceptance=pending->accepted")
            self.assertEqual(trace[-1], fixture["expected_trace"][-1])

    def test_archived_negative_role_evidence_is_pending(self) -> None:
        with tempfile.TemporaryDirectory(prefix="kanban-conformance-archive-") as raw_repo:
            repo = Path(raw_repo)
            _git(repo, "init", "-q")
            _git(repo, "config", "user.email", "conformance@example.invalid")
            _git(repo, "config", "user.name", "Kanban Conformance")
            (repo / "README").write_text("base\n", encoding="utf-8")
            _git(repo, "add", "README")
            _git(repo, "commit", "-qm", "base")
            base_sha = _git(repo, "rev-parse", "HEAD")
            (repo / "lifecycle.py").write_text("print('proof')\n", encoding="utf-8")
            _git(repo, "add", "lifecycle.py")
            _git(repo, "commit", "-qm", "implementation")
            head_sha = _git(repo, "rev-parse", "HEAD")

            implementation = kb.create_task(
                self.conn,
                title="Archived negative verdict implementation",
                assignee="implementer",
                initial_status="blocked",
                workspace_kind="dir",
                workspace_path=str(repo),
                lifecycle_contract={
                    "kind": "code",
                    "review_mode": "separate_card",
                    "reviewer": "reviewer",
                    "validation_required": False,
                },
            )
            review = kb.create_task(
                self.conn,
                title="Archived negative verdict review",
                assignee="reviewer",
                initial_status="blocked",
                lifecycle_contract={"kind": "review", "candidate_task_id": implementation},
            )
            kb.link_tasks(self.conn, implementation, review, requirement="phase_finished")
            self.assertTrue(kb.unblock_task(self.conn, implementation))
            implementation_run = kb.claim_task(
                self.conn, implementation, claimer="implementer:conformance"
            )
            self.assertIsNotNone(implementation_run)
            self.assertTrue(
                kb.complete_task(
                    self.conn,
                    implementation,
                    expected_run_id=implementation_run.current_run_id,
                    summary="Implementation evidence",
                    metadata={
                        "base_sha": base_sha,
                        "head_sha": head_sha,
                        "changed_files": ["lifecycle.py"],
                    },
                )
            )

            review_run = kb.claim_review_task(self.conn, review, claimer="reviewer:conformance")
            self.assertIsNotNone(review_run)
            self.assertTrue(
                kb.complete_task(
                    self.conn,
                    review,
                    expected_run_id=review_run.current_run_id,
                    verdict="REQUEST_CHANGES",
                    summary="Changes are required",
                )
            )
            self.assertEqual(get_lifecycle_state(self.conn, review)["acceptance"], "rejected")
            self.assertEqual(
                get_lifecycle_state(self.conn, implementation)["acceptance"], "rejected"
            )

            self.assertTrue(kb.archive_task(self.conn, review))
            self.assertEqual(get_lifecycle_state(self.conn, review)["acceptance"], "pending")
            self.assertEqual(
                get_lifecycle_state(self.conn, implementation)["acceptance"], "pending"
            )


    def test_cross_phase_verdicts_are_malformed(self) -> None:
        candidate = kb.create_task(
            self.conn,
            title="cross-phase candidate",
            assignee="implementer",
            initial_status="blocked",
            workspace_kind="dir",
            lifecycle_contract={
                "kind": "code",
                "review_mode": "separate_card",
                "reviewer": "reviewer",
                "validation_required": True,
            },
        )
        review = kb.create_task(
            self.conn,
            title="cross-phase review",
            assignee="reviewer",
            initial_status="blocked",
            parents=[candidate],
            lifecycle_contract={"kind": "review", "candidate_task_id": candidate},
        )
        validation = kb.create_task(
            self.conn,
            title="cross-phase validation",
            assignee="tester",
            initial_status="blocked",
            parents=[review],
            lifecycle_contract={"kind": "validation", "candidate_task_id": candidate},
        )
        with kb.write_txn(self.conn):
            self.conn.execute(
                "UPDATE tasks SET status = 'done' WHERE id IN (?, ?, ?)",
                (candidate, review, validation),
            )
            for task_id, phase, verdict in (
                (review, "review", "PASS"),
                (validation, "validation", "APPROVE"),
            ):
                self.conn.execute(
                    "INSERT INTO task_runs "
                    "(task_id, status, started_at, ended_at, outcome, metadata) "
                    "VALUES (?, 'done', 1, 2, 'completed', ?)",
                    (
                        task_id,
                        json.dumps(
                            {
                                "lifecycle": {
                                    "schema": 1,
                                    "phase": phase,
                                    "verdict": verdict,
                                }
                            }
                        ),
                    ),
                )

        review_state = get_lifecycle_state(self.conn, review)
        validation_state = get_lifecycle_state(self.conn, validation)
        assert review_state["review_verdict"] is None
        assert validation_state["validation_verdict"] is None
        assert review_state["acceptance"] == "pending"
        assert validation_state["acceptance"] == "pending"
        assert "verdict_malformed" in review_state["diagnostics"]
        assert "verdict_malformed" in validation_state["diagnostics"]
        assert get_lifecycle_state(self.conn, candidate)["acceptance"] == "pending"

    def test_missing_candidate_verdicts_are_not_accepted(self) -> None:
        candidate = kb.create_task(
            self.conn,
            title="missing lifecycle candidate",
            assignee="implementer",
            initial_status="blocked",
            lifecycle_contract={
                "kind": "code",
                "review_mode": "separate_card",
                "reviewer": "reviewer",
                "validation_required": True,
            },
        )
        for kind, assignee, phase, verdict in (
            ("review", "reviewer", "review", "APPROVE"),
            ("validation", "tester", "validation", "PASS"),
        ):
            role = kb.create_task(
                self.conn,
                title=f"missing {phase} candidate role",
                assignee=assignee,
                initial_status="blocked",
                lifecycle_contract={"kind": kind, "candidate_task_id": candidate},
            )
            with kb.write_txn(self.conn):
                kb._synthesize_ended_run(
                    self.conn,
                    role,
                    outcome="completed",
                    metadata={
                        "lifecycle": {
                            "schema": 1,
                            "phase": phase,
                            "candidate_task_id": candidate,
                            "candidate_run_id": 99,
                            "head_sha": "missing-head",
                            "goal_revision_ids": {candidate: 1},
                            "task_goal_revision_id": 1,
                            "verdict": verdict,
                        }
                    },
                )
                self.conn.execute("UPDATE tasks SET status = 'done' WHERE id = ?", (role,))

            state = get_lifecycle_state(self.conn, role)
            self.assertEqual(state["acceptance"], "stale")
            self.assertEqual(state["diagnostics"], ["candidate_missing"])

    def test_separate_card_validator_preserves_implementer_identity(self) -> None:
        with tempfile.TemporaryDirectory(prefix="kanban-conformance-identity-") as raw_repo:
            repo = Path(raw_repo)
            _git(repo, "init", "-q")
            _git(repo, "config", "user.email", "conformance@example.invalid")
            _git(repo, "config", "user.name", "Kanban Conformance")
            (repo / "README").write_text("base\n", encoding="utf-8")
            _git(repo, "add", "README")
            _git(repo, "commit", "-qm", "base")
            base_sha = _git(repo, "rev-parse", "HEAD")
            (repo / "lifecycle.py").write_text("print('proof')\n", encoding="utf-8")
            _git(repo, "add", "lifecycle.py")
            _git(repo, "commit", "-qm", "implementation")
            head_sha = _git(repo, "rev-parse", "HEAD")

            implementation = kb.create_task(
                self.conn,
                title="reassigned implementation",
                assignee="bob",
                initial_status="blocked",
                workspace_kind="dir",
                workspace_path=str(repo),
                lifecycle_contract={
                    "kind": "code",
                    "review_mode": "separate_card",
                    "reviewer": "alice",
                    "validation_required": True,
                },
            )
            review = kb.create_task(
                self.conn,
                title="identity review",
                assignee="alice",
                initial_status="blocked",
                lifecycle_contract={"kind": "review", "candidate_task_id": implementation},
            )
            kb.link_tasks(self.conn, implementation, review, requirement="phase_finished")
            self.assertTrue(kb.unblock_task(self.conn, implementation))
            implementation_run = kb.claim_task(self.conn, implementation, claimer="bob")
            self.assertIsNotNone(implementation_run)
            self.assertTrue(
                kb.complete_task(
                    self.conn,
                    implementation,
                    expected_run_id=implementation_run.current_run_id,
                    metadata={
                        "base_sha": base_sha,
                        "head_sha": head_sha,
                        "changed_files": ["lifecycle.py"],
                    },
                )
            )
            self.assertTrue(kb.assign_task(self.conn, implementation, "charlie"))
            with self.assertRaises(kb.LifecycleContractError):
                kb.create_task(
                    self.conn,
                    title="identity validation",
                    assignee="bob",
                    initial_status="blocked",
                    parents=(review,),
                    lifecycle_contract={
                        "kind": "validation",
                        "candidate_task_id": implementation,
                    },
                )

    def test_orphan_validator_blocks_candidate_reassignment(self) -> None:
        implementation = kb.create_task(
            self.conn,
            title="orphan validator candidate",
            assignee="implementer",
            initial_status="blocked",
            lifecycle_contract={
                "kind": "code",
                "review_mode": "same_card",
                "reviewer": "reviewer",
                "validation_required": True,
            },
        )
        kb.create_task(
            self.conn,
            title="orphan validator",
            assignee="tester",
            initial_status="blocked",
            lifecycle_contract={"kind": "validation", "candidate_task_id": implementation},
        )
        with self.assertRaises(kb.LifecycleContractError):
            kb.assign_task(self.conn, implementation, "tester")

    def test_malformed_parent_contract_is_dependency_blocker(self) -> None:
        parent = kb.create_task(
            self.conn,
            title="malformed dependency parent",
            initial_status="blocked",
        )
        child = kb.create_task(
            self.conn,
            title="malformed dependency child",
            initial_status="blocked",
            parents=(parent,),
        )
        unrelated = kb.create_task(
            self.conn,
            title="unrelated ready task",
            initial_status="blocked",
        )
        with kb.write_txn(self.conn):
            self.conn.execute(
                "UPDATE tasks SET lifecycle_contract = ? WHERE id = ?",
                ("{\"kind\": \"corrupt\"}", parent),
            )
        projection = kb.evaluate_dependencies(self.conn, child)
        self.assertFalse(projection["satisfied"])
        self.assertEqual(projection["blockers"][0]["code"], "lifecycle_unclassified")
        self.assertEqual(kb.recompute_ready(self.conn), 1)
        self.assertEqual(self._task(unrelated).status, "ready")

    def test_legacy_null_contracts_remain_schedulable(self) -> None:
        legacy_review = kb.create_task(
            self.conn,
            title="legacy verdictless review",
            assignee="reviewer",
            initial_status="blocked",
        )
        with kb.write_txn(self.conn):
            self.conn.execute(
                "UPDATE tasks SET lifecycle_contract = NULL, status = 'review' WHERE id = ?",
                (legacy_review,),
            )
        review_run = kb.claim_review_task(self.conn, legacy_review, claimer="reviewer")
        self.assertIsNotNone(review_run)
        self.assertTrue(
            kb.complete_task(
                self.conn,
                legacy_review,
                expected_run_id=review_run.current_run_id,
                summary="legacy review completed without a verdict",
            )
        )
        self.assertEqual(
            get_lifecycle_state(self.conn, legacy_review)["acceptance"],
            "unclassified",
        )

        parent = kb.create_task(
            self.conn,
            title="legacy parent",
            initial_status="blocked",
        )
        child = kb.create_task(
            self.conn,
            title="legacy dependent",
            assignee="worker",
            initial_status="blocked",
            parents=(parent,),
        )
        with kb.write_txn(self.conn):
            self.conn.execute(
                "UPDATE tasks SET lifecycle_contract = NULL WHERE id IN (?, ?)",
                (parent, child),
            )
            self.conn.execute(
                "UPDATE task_links SET requirement = NULL WHERE parent_id = ? AND child_id = ?",
                (parent, child),
            )
            self.conn.execute(
                "UPDATE tasks SET status = 'done' WHERE id = ?", (parent,),
            )
        self.assertEqual(
            kb.evaluate_dependencies(self.conn, child),
            {"satisfied": True, "blockers": []},
        )
        self.assertEqual(kb.recompute_ready(self.conn), 1)
        child_run = kb.claim_task(self.conn, child, claimer="worker")
        self.assertIsNotNone(child_run)
        self.assertTrue(
            kb.complete_task(
                self.conn,
                child,
                expected_run_id=child_run.current_run_id,
                summary="legacy dependent completed",
            )
        )
        self.assertEqual(get_lifecycle_state(self.conn, child)["acceptance"], "unclassified")

    def test_legacy_triage_root_can_be_decomposed(self) -> None:
        root = kb.create_task(
            self.conn,
            title="legacy triage root",
            initial_status="blocked",
        )
        with kb.write_txn(self.conn):
            self.conn.execute(
                "UPDATE tasks SET lifecycle_contract = NULL, status = 'triage' WHERE id = ?",
                (root,),
            )
        child_ids = kb.decompose_triage_task(
            self.conn,
            root,
            root_assignee="orchestrator",
            children=[{
                "title": "legacy triage child",
                "assignee": "worker",
                "lifecycle_contract": {"kind": "general"},
            }],
            auto_promote=False,
        )
        self.assertEqual(len(child_ids or []), 1)
        edge = self.conn.execute(
            "SELECT requirement FROM task_links WHERE parent_id = ? AND child_id = ?",
            (child_ids[0], root),
        ).fetchone()
        self.assertIsNotNone(edge)
        self.assertEqual(edge["requirement"], "phase_finished")
        self.assertEqual(self._task(root).status, "todo")

    def test_review_card_requires_separate_card_candidate(self) -> None:
        implementation = kb.create_task(
            self.conn,
            title="same-card candidate",
            assignee="implementer",
            lifecycle_contract={
                "kind": "code",
                "review_mode": "same_card",
                "reviewer": "reviewer",
                "validation_required": False,
            },
        )
        with self.assertRaises(kb.LifecycleContractError):
            kb.create_task(
                self.conn,
                title="invalid downstream review",
                assignee="reviewer",
                initial_status="blocked",
                lifecycle_contract={"kind": "review", "candidate_task_id": implementation},
            )

    def test_force_promotion_overrides_unfinished_parent(self) -> None:
        parent = kb.create_task(
            self.conn,
            title="unfinished parent",
            initial_status="blocked",
        )
        child = kb.create_task(
            self.conn,
            title="manually promoted child",
            initial_status="blocked",
            parents=(parent,),
        )
        refused, reason = kb.promote_task(self.conn, child, actor="operator")
        self.assertFalse(refused)
        self.assertIn("unsatisfied lifecycle dependencies", reason or "")
        promoted, reason = kb.promote_task(
            self.conn, child, actor="operator", reason="override", force=True,
        )
        self.assertTrue(promoted)
        self.assertIsNone(reason)
        self.assertEqual(self._task(child).status, "ready")
        claimed = kb.claim_task(self.conn, child, claimer="operator")
        self.assertIsNotNone(claimed)


    def test_force_promotion_survives_preclaim_event(self) -> None:
        parent = kb.create_task(
            self.conn,
            title="unfinished parent",
            initial_status="blocked",
        )
        child = kb.create_task(
            self.conn,
            title="manually promoted child",
            initial_status="blocked",
            parents=(parent,),
        )
        promoted, reason = kb.promote_task(
            self.conn, child, actor="operator", reason="override", force=True,
        )
        self.assertTrue(promoted)
        self.assertIsNone(reason)
        with kb.write_txn(self.conn):
            kb._append_event(
                self.conn, child, "respawn_guarded", {"reason": "recent_success"},
            )
        self.assertIsNotNone(kb.claim_task(self.conn, child, claimer="operator"))

    def test_force_promotion_cannot_bypass_missing_role_edges(self) -> None:
        candidate = kb.create_task(
            self.conn,
            title="orphaned role candidate",
            assignee="implementer",
            initial_status="blocked",
            lifecycle_contract={
                "kind": "code",
                "review_mode": "separate_card",
                "reviewer": "reviewer",
                "validation_required": True,
            },
        )
        review = kb.create_task(
            self.conn,
            title="orphaned review",
            assignee="reviewer",
            initial_status="blocked",
            lifecycle_contract={"kind": "review", "candidate_task_id": candidate},
        )
        validation = kb.create_task(
            self.conn,
            title="orphaned validation",
            assignee="tester",
            initial_status="blocked",
            lifecycle_contract={"kind": "validation", "candidate_task_id": candidate},
        )

        promoted, reason = kb.promote_task(
            self.conn, review, actor="operator", reason="override", force=True,
        )
        self.assertFalse(promoted)
        self.assertIn("candidate_edge_missing", reason or "")
        self.assertEqual(self._task(review).status, "blocked")
        promoted, reason = kb.promote_task(
            self.conn, validation, actor="operator", reason="override", force=True,
        )
        self.assertFalse(promoted)
        self.assertIn("candidate_edge_missing", reason or "")
        self.assertEqual(self._task(validation).status, "blocked")

        with kb.write_txn(self.conn):
            self.conn.execute(
                "UPDATE tasks SET status = 'review' WHERE id = ?", (review,),
            )
            kb._append_event(
                self.conn,
                review,
                "promoted_manual",
                {"actor": "operator", "reason": "override", "forced": True},
            )
        self.assertIsNone(kb.claim_review_task(self.conn, review, claimer="reviewer"))
        self.assertEqual(self._task(review).status, "todo")

        with kb.write_txn(self.conn):
            self.conn.execute(
                "UPDATE tasks SET status = 'ready' WHERE id = ?", (validation,),
            )
            kb._append_event(
                self.conn,
                validation,
                "promoted_manual",
                {"actor": "operator", "reason": "override", "forced": True},
            )
        self.assertIsNone(kb.claim_task(self.conn, validation, claimer="tester"))
        self.assertEqual(self._task(validation).status, "todo")

    def test_blocked_same_card_review_keeps_reviewer_identity(self) -> None:
        task_id = kb.create_task(
            self.conn,
            title="blocked same-card review",
            assignee="builder",
            initial_status="blocked",
            lifecycle_contract={
                "kind": "code",
                "review_mode": "same_card",
                "reviewer": "reviewer",
                "validation_required": False,
            },
        )
        with kb.write_txn(self.conn):
            kb._append_event(
                self.conn,
                task_id,
                "blocked",
                {"source_status": "review", "resume_status": "review"},
            )

        with self.assertRaises(kb.LifecycleContractError):
            kb.assign_task(self.conn, task_id, "builder")
        self.assertTrue(kb.assign_task(self.conn, task_id, "reviewer"))


    def test_deletion_protects_review_parent_of_validation(self) -> None:
        implementation = kb.create_task(
            self.conn,
            title="deletion candidate",
            assignee="bob",
            lifecycle_contract={
                "kind": "code",
                "review_mode": "separate_card",
                "reviewer": "alice",
                "validation_required": True,
            },
        )
        review = kb.create_task(
            self.conn,
            title="deletion review",
            assignee="alice",
            lifecycle_contract={"kind": "review", "candidate_task_id": implementation},
        )
        validation = kb.create_task(
            self.conn,
            title="deletion validation",
            assignee="tester",
            parents=(review,),
            lifecycle_contract={
                "kind": "validation",
                "candidate_task_id": implementation,
            },
        )
        kb.link_tasks(self.conn, implementation, review, requirement="phase_finished")
        with self.assertRaises(kb.LifecycleContractError):
            kb.delete_task(self.conn, review)
        self.assertIsNotNone(kb.get_task(self.conn, review))
        self.assertIsNotNone(kb.get_task(self.conn, validation))


    def test_deletion_protects_leaf_review_role(self) -> None:
        implementation = kb.create_task(
            self.conn,
            title="leaf deletion candidate",
            assignee="bob",
            lifecycle_contract={
                "kind": "code",
                "review_mode": "separate_card",
                "reviewer": "alice",
                "validation_required": False,
            },
        )
        review = kb.create_task(
            self.conn,
            title="leaf deletion review",
            assignee="alice",
            lifecycle_contract={"kind": "review", "candidate_task_id": implementation},
        )
        kb.link_tasks(self.conn, implementation, review, requirement="phase_finished")

        with self.assertRaises(kb.LifecycleContractError):
            kb.delete_task(self.conn, review)
        self.assertIsNotNone(kb.get_task(self.conn, implementation))
        self.assertIsNotNone(kb.get_task(self.conn, review))

    def test_deletion_protects_terminal_validation_role(self) -> None:
        implementation = kb.create_task(
            self.conn,
            title="terminal validation candidate",
            assignee="bob",
            lifecycle_contract={
                "kind": "code",
                "review_mode": "same_card",
                "reviewer": "alice",
                "validation_required": True,
            },
        )
        validation = kb.create_task(
            self.conn,
            title="terminal validation",
            assignee="tester",
            lifecycle_contract={
                "kind": "validation",
                "candidate_task_id": implementation,
            },
        )
        kb.link_tasks(self.conn, implementation, validation, requirement="review_approved")
        self.assertTrue(kb.archive_task(self.conn, validation))

        with self.assertRaises(kb.LifecycleContractError):
            kb.delete_archived_task(self.conn, validation)
        self.assertIsNotNone(kb.get_task(self.conn, implementation))
        self.assertIsNotNone(kb.get_task(self.conn, validation))

    def test_archived_lifecycle_graph_can_be_purged_atomically(self) -> None:
        implementation = kb.create_task(
            self.conn,
            title="purge candidate",
            assignee="bob",
            initial_status="blocked",
            lifecycle_contract={
                "kind": "code",
                "review_mode": "separate_card",
                "reviewer": "alice",
                "validation_required": True,
            },
        )
        review = kb.create_task(
            self.conn,
            title="purge review",
            assignee="alice",
            initial_status="blocked",
            lifecycle_contract={"kind": "review", "candidate_task_id": implementation},
        )
        validation = kb.create_task(
            self.conn,
            title="purge validation",
            assignee="tester",
            initial_status="blocked",
            lifecycle_contract={
                "kind": "validation",
                "candidate_task_id": implementation,
            },
        )
        kb.link_tasks(self.conn, implementation, review, requirement="phase_finished")
        kb.link_tasks(self.conn, review, validation, requirement="review_approved")
        for task_id in (implementation, review, validation):
            self.assertTrue(kb.archive_task(self.conn, task_id))

        self.assertTrue(kb.delete_archived_task(self.conn, implementation))
        for task_id in (implementation, review, validation):
            self.assertIsNone(kb.get_task(self.conn, task_id))
        self.assertEqual(
            self.conn.execute(
                "SELECT COUNT(*) FROM task_links WHERE parent_id IN (?, ?, ?) "
                "OR child_id IN (?, ?, ?)",
                (implementation, review, validation, implementation, review, validation),
            ).fetchone()[0],
            0,
        )

    def test_archived_lifecycle_graph_purge_requires_every_node_archived(self) -> None:
        implementation = kb.create_task(
            self.conn,
            title="partial purge candidate",
            assignee="bob",
            initial_status="blocked",
            lifecycle_contract={
                "kind": "code",
                "review_mode": "separate_card",
                "reviewer": "alice",
                "validation_required": True,
            },
        )
        review = kb.create_task(
            self.conn,
            title="partial purge review",
            assignee="alice",
            initial_status="blocked",
            lifecycle_contract={"kind": "review", "candidate_task_id": implementation},
        )
        validation = kb.create_task(
            self.conn,
            title="partial purge validation",
            assignee="tester",
            initial_status="blocked",
            lifecycle_contract={
                "kind": "validation",
                "candidate_task_id": implementation,
            },
        )
        kb.link_tasks(self.conn, implementation, review, requirement="phase_finished")
        kb.link_tasks(self.conn, review, validation, requirement="review_approved")
        self.assertTrue(kb.archive_task(self.conn, implementation))
        self.assertTrue(kb.archive_task(self.conn, review))

        with self.assertRaises(kb.LifecycleContractError):
            kb.delete_archived_task(self.conn, implementation)
        for task_id in (implementation, review, validation):
            self.assertIsNotNone(kb.get_task(self.conn, task_id))



    def test_archived_lifecycle_graph_stops_at_external_required_edges(self) -> None:
        implementation = kb.create_task(
            self.conn,
            title="boundary purge candidate",
            assignee="bob",
            initial_status="blocked",
            lifecycle_contract={
                "kind": "code",
                "review_mode": "separate_card",
                "reviewer": "alice",
                "validation_required": True,
            },
        )
        review = kb.create_task(
            self.conn,
            title="boundary purge review",
            assignee="alice",
            initial_status="blocked",
            lifecycle_contract={"kind": "review", "candidate_task_id": implementation},
        )
        validation = kb.create_task(
            self.conn,
            title="boundary purge validation",
            assignee="tester",
            initial_status="blocked",
            lifecycle_contract={
                "kind": "validation",
                "candidate_task_id": implementation,
            },
        )
        downstream = kb.create_task(
            self.conn,
            title="boundary downstream task",
            assignee="owner",
            initial_status="blocked",
            lifecycle_contract={"kind": "general"},
        )
        kb.link_tasks(self.conn, implementation, review, requirement="phase_finished")
        kb.link_tasks(self.conn, review, validation, requirement="review_approved")
        kb.link_tasks(self.conn, validation, downstream, requirement="validation_passed")
        for task_id in (implementation, review, validation, downstream):
            self.assertTrue(kb.archive_task(self.conn, task_id))

        with self.assertRaises(kb.LifecycleContractError):
            kb.delete_archived_task(self.conn, implementation)
        for task_id in (implementation, review, validation, downstream):
            self.assertIsNotNone(kb.get_task(self.conn, task_id))

    def test_worker_command_imports_stay_pinned_after_final_grant(self) -> None:
        with tempfile.TemporaryDirectory(prefix="kanban-conformance-worker-import-") as raw_root:
            root = Path(raw_root)
            for directory in runtime._IDENTITY_ROOTS:
                package = root / directory
                package.mkdir(parents=True)
                (package / "module.py").write_text("value = 1\n", encoding="utf-8")
            skill = root / "skills" / "devops" / "sdlc-review" / "SKILL.md"
            skill.parent.mkdir(parents=True)
            skill.write_text("review instructions\n", encoding="utf-8")
            (root / "cli.py").write_text(
                "def main():\n    return 1\n",
                encoding="utf-8",
            )
            (root / "run_agent.py").write_text(
                "class AIAgent:\n    pass\n",
                encoding="utf-8",
            )
            receipt = root / "receipt.json"
            script = (
                "import importlib, json, os, sys; "
                "from pathlib import Path; "
                "from hermes_cli import kanban_runtime as runtime; "
                "root = Path(sys.argv[1]); receipt = Path(sys.argv[2]); "
                "runtime._module_root = lambda module_root=None: root; "
                "runtime._FROZEN_RUNTIME_IDENTITY = None; "
                "sys.path.insert(0, str(root)); "
                "sys.modules.pop('hermes_cli.kanban_runtime', None); "
                "sys.modules.pop('hermes_cli', None); "
                "expected = runtime.runtime_identity(root, pid=os.getpid(), "
                "start_time=runtime.process_start_time()); "
                "os.environ['HERMES_KANBAN_BOOTSTRAP_PATH'] = str(root / 'preparation.json'); "
                "os.environ['HERMES_KANBAN_PREPARATION_ID'] = 'race-preparation'; "
                "os.environ['HERMES_KANBAN_EXPECTED_RUNTIME'] = runtime.encode_identity(expected); "
                "runtime._read_bootstrap_message = lambda: {"
                "'grant': True, 'preparation_id': 'race-preparation', "
                "'runtime_identity': expected.as_dict()}; "
                "runtime.worker_bootstrap_post_import(); "
                "(root / 'cli.py').write_text('def main():\\n    return 2\\n', encoding='utf-8'); "
                "result = importlib.import_module('cli').main(); "
                "receipt.write_text(json.dumps({'result': result}), encoding='utf-8')"
            )
            env = os.environ.copy()
            env["PYTHONPATH"] = str(ROOT)
            completed = subprocess.run(
                [sys.executable, "-c", script, str(root), str(receipt)],
                text=True,
                capture_output=True,
                env=env,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(json.loads(receipt.read_text(encoding="utf-8"))["result"], 1)

    def test_embedded_dispatcher_freezes_identity_before_first_tick(self) -> None:
        with tempfile.TemporaryDirectory(prefix="kanban-conformance-dispatcher-") as raw_root:
            root = Path(raw_root)
            for directory in ("hermes_cli", "tools", "agent", "gateway", "plugins", "providers", "cron"):
                package = root / directory
                package.mkdir(parents=True)
                (package / "module.py").write_text("value = 1\n", encoding="utf-8")
            skill = root / "skills" / "devops" / "sdlc-review" / "SKILL.md"
            skill.parent.mkdir(parents=True)
            skill.write_text("review instructions\n", encoding="utf-8")

            from gateway.kanban_watchers_dispatcher import _KanbanDispatcher

            with patch.object(runtime, "_FROZEN_RUNTIME_IDENTITY", None), patch.object(
                runtime, "_module_root", return_value=root,
            ):
                _KanbanDispatcher(object(), object())
                frozen = runtime.runtime_identity()
                (root / "hermes_cli" / "module.py").write_text("value = 2\n", encoding="utf-8")
                expected = runtime.prospective_identity()

            self.assertEqual(expected.fingerprint, frozen.fingerprint)
            self.assertNotEqual(frozen.fingerprint, _fingerprint(root))

    def test_runtime_identity_fingerprints_review_skill(self) -> None:
        with tempfile.TemporaryDirectory(prefix="kanban-conformance-runtime-") as raw_root:
            root = Path(raw_root)
            for directory in ("hermes_cli", "tools", "agent", "gateway", "plugins", "providers", "cron"):
                package = root / directory
                package.mkdir(parents=True)
                (package / "module.py").write_text("value = 1\n", encoding="utf-8")
            skill = root / "skills" / "devops" / "sdlc-review" / "SKILL.md"
            skill.parent.mkdir(parents=True)
            skill.write_text("review instructions v1\n", encoding="utf-8")

            before = _fingerprint(root)
            skill.write_text("review instructions v2\n", encoding="utf-8")
            self.assertNotEqual(before, _fingerprint(root))
            bundled_root = root / "packaged-skills"
            bundled_skill = bundled_root / "devops" / "sdlc-review" / "SKILL.md"
            bundled_skill.parent.mkdir(parents=True)
            bundled_skill.write_text("packaged review instructions v1\n", encoding="utf-8")
            skill.unlink()
            with patch.dict(os.environ, {"HERMES_BUNDLED_SKILLS": str(bundled_root)}):
                before = _fingerprint(root)
                bundled_skill.write_text("packaged review instructions v2\n", encoding="utf-8")
                self.assertNotEqual(before, _fingerprint(root))

    def test_completion_rechecks_candidate_head_at_commit_boundary(self) -> None:
        with tempfile.TemporaryDirectory(prefix="kanban-conformance-race-") as raw_repo:
            repo = Path(raw_repo)
            _git(repo, "init", "-q")
            _git(repo, "config", "user.email", "conformance@example.invalid")
            _git(repo, "config", "user.name", "Kanban Conformance")
            (repo / "README").write_text("base\n", encoding="utf-8")
            _git(repo, "add", "README")
            _git(repo, "commit", "-qm", "base")
            base_sha = _git(repo, "rev-parse", "HEAD")
            (repo / "lifecycle.py").write_text("print('proof')\n", encoding="utf-8")
            _git(repo, "add", "lifecycle.py")
            _git(repo, "commit", "-qm", "implementation")
            head_sha = _git(repo, "rev-parse", "HEAD")

            implementation = kb.create_task(
                self.conn,
                title="Implement race fixture",
                assignee="implementer",
                initial_status="blocked",
                workspace_kind="scratch",
                workspace_path=str(repo),
                lifecycle_contract={
                    "kind": "code",
                    "review_mode": "separate_card",
                    "reviewer": "reviewer",
                    "validation_required": False,
                },
            )
            review = kb.create_task(
                self.conn,
                title="Review race fixture",
                assignee="reviewer",
                initial_status="blocked",
                lifecycle_contract={"kind": "review", "candidate_task_id": implementation},
            )
            kb.link_tasks(self.conn, implementation, review, requirement="phase_finished")
            self.assertTrue(kb.unblock_task(self.conn, implementation))
            implementation_run = kb.claim_task(self.conn, implementation, claimer="implementer:race")
            self.assertIsNotNone(implementation_run)
            self.assertTrue(
                kb.complete_task(
                    self.conn,
                    implementation,
                    expected_run_id=implementation_run.current_run_id,
                    summary="Implementation evidence",
                    metadata={
                        "base_sha": base_sha,
                        "head_sha": head_sha,
                        "changed_files": ["lifecycle.py"],
                    },
                )
            )
            review_run = kb.claim_review_task(self.conn, review, claimer="reviewer:race")
            self.assertIsNotNone(review_run)

            original_stamp = kb._stamp_lifecycle_metadata
            raced = False

            def stamp(conn, task_id, metadata, *, phase, run_id, verdict):
                nonlocal raced
                prepared = original_stamp(
                    conn,
                    task_id,
                    metadata,
                    phase=phase,
                    run_id=run_id,
                    verdict=verdict,
                )
                if not raced:
                    raced = True
                    (repo / "race.py").write_text("print('moved')\n", encoding="utf-8")
                    _git(repo, "add", "race.py")
                    _git(repo, "commit", "-qm", "candidate advanced")
                return prepared

            with patch.object(kb, "_stamp_lifecycle_metadata", side_effect=stamp):
                with self.assertRaises(kb.HandoffValidationError):
                    kb.complete_task(
                        self.conn,
                        review,
                        expected_run_id=review_run.current_run_id,
                        verdict="APPROVE",
                        summary="Reviewed exact implementation head",
                        metadata={"reviewed_head_sha": head_sha},
                    )
            self.assertEqual(self._task(review).status, "running")

    def test_identity_claim_and_runtime_surfaces(self) -> None:
        identity = runtime_identity(RUNTIME_ROOT)
        self.assertEqual(identity.protocol, 1)
        self.assertEqual(identity, runtime_identity(RUNTIME_ROOT, pid=identity.pid, start_time=identity.start_time))
        self.assertEqual(code_identity(identity), code_identity(identity.as_dict()))
        self.assertTrue(same_code_identity(identity, identity.as_dict()))
        self.assertEqual(process_start_time(identity.pid), identity.start_time)
        for mixed_name in (
            "hermes_cli._conformance_mixed_root",
            "providers._conformance_mixed_root",
        ):
            sys.modules[mixed_name] = SimpleNamespace(__file__="/tmp/mixed-hermes-runtime.py")
            try:
                with self.assertRaises(RuntimeIdentityError):
                    assert_runtime_import_root(RUNTIME_ROOT)
            finally:
                sys.modules.pop(mixed_name, None)

        implementation = kb.create_task(
            self.conn,
            title="identity proof",
            assignee="implementer",
            initial_status="blocked",
        )
        self.assertTrue(kb.unblock_task(self.conn, implementation))
        claimed = kb.claim_task(
            self.conn,
            implementation,
            runtime_identity=identity,
            worker_pid=identity.pid,
            worker_start_time=identity.start_time,
            preparation_id="conformance-preparation",
        )
        self.assertIsNotNone(claimed)
        run = self.conn.execute(
            "SELECT metadata, worker_pid FROM task_runs WHERE id = ?", (claimed.current_run_id,)
        ).fetchone()
        metadata = json.loads(run["metadata"])
        self.assertEqual(run["worker_pid"], identity.pid)
        self.assertEqual(metadata["runtime_identity"], identity.as_dict())
        self.assertEqual(metadata["preparation_id"], "conformance-preparation")

        root_parser = argparse.ArgumentParser()
        subparsers = root_parser.add_subparsers(dest="command")
        build_parser(subparsers)
        args = root_parser.parse_args(
            [
                "kanban",
                "create",
                "typed",
                "--lifecycle-contract",
                '{"kind":"code","review_mode":"same_card","reviewer":"reviewer","validation_required":false}',
            ]
        )
        self.assertEqual(json.loads(args.lifecycle_contract)["kind"], "code")

        from plugins.kanban.dashboard import plugin_api
        task_dict = plugin_api._task_dict(self._task(implementation), conn=self.conn)
        self.assertEqual(task_dict["lifecycle"]["acceptance"], "not_applicable")
        self.assertIn("dependencies", task_dict)

        event = SimpleNamespace(
            kind="acceptance_changed",
            payload={"phase": "review", "result": "APPROVE"},
        )
        notice = SimpleNamespace(head="Lifecycle", task_id=implementation, title="identity proof")
        message, wake, detail = notifier._fmt_acceptance_changed(event, notice)
        self.assertIn("review result: APPROVE", message)
        self.assertIsNone(wake)
        self.assertIsNone(detail)
    def test_default_dispatch_fences_child_before_claim(self) -> None:
        task_id = kb.create_task(
            self.conn,
            title="default spawn identity proof",
            assignee="implementer",
            initial_status="blocked",
        )
        self.assertTrue(kb.unblock_task(self.conn, task_id))
        with tempfile.TemporaryDirectory(prefix="kanban-worker-proof-") as raw:
            receipt = Path(raw) / "grant.json"
            script = (
                "import json, os, sys; "
                "from pathlib import Path; "
                "import hermes_cli.main; "
                "from hermes_cli.kanban_runtime import worker_bootstrap_post_import; "
                "worker_bootstrap_post_import(); "
                "Path(sys.argv[1]).write_text(json.dumps({"
                "'run': os.environ.get('HERMES_KANBAN_RUN_ID'), "
                "'claim': os.environ.get('HERMES_KANBAN_CLAIM_LOCK')}))"
            )
            with patch.object(kbd, "_profile_exists_fn", return_value=None), patch.object(
                kbd,
                "_restart_safe_worker_argv",
                side_effect=lambda task, command, preparation_id=None: command,
            ), patch.object(
                kbd,
                "_worker_argv",
                return_value=[sys.executable, "-c", script, str(receipt)],
            ), patch.dict("os.environ", {"PYTHONPATH": str(RUNTIME_ROOT)}, clear=False):
                result = kbd.dispatch_once(
                    self.conn,
                    max_spawn=1,
                    reconcile_orphans=False,
                )
            self.assertEqual([entry[0] for entry in result.spawned], [task_id])
            self.assertEqual(self._task(task_id).status, "running")
            deadline = time.monotonic() + 5
            while not receipt.exists() and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertTrue(receipt.exists())
            granted = json.loads(receipt.read_text(encoding="utf-8"))
            self.assertEqual(granted["run"], str(self._task(task_id).current_run_id))
            self.assertTrue(granted["claim"])
            pid = self._task(task_id).worker_pid
            self.assertTrue(kb.reclaim_task(self.conn, task_id, reason="conformance cleanup", signal_fn=lambda pid, sig: os.kill(pid, sig)))
            deadline = time.monotonic() + 2
            while pid and kbd._pid_alive(pid) and time.monotonic() < deadline:
                time.sleep(0.02)
            kbd.reap_worker_zombies()


    def test_default_dispatch_rejects_changed_execution_snapshot(self) -> None:
        task_id = kb.create_task(
            self.conn,
            title="reject changed worker configuration",
            assignee="implementer",
            initial_status="blocked",
        )
        self.assertTrue(kb.unblock_task(self.conn, task_id))
        identity = runtime_identity(RUNTIME_ROOT)
        cancelled: list[bool] = []

        def fake_default_spawn(task, workspace, *, board=None, defer_grant=False):
            self.assertTrue(defer_grant)
            self.assertTrue(kb.assign_task(self.conn, task.id, "replacement"))
            return kbd.WorkerLaunch(
                identity.pid,
                identity.as_dict(),
                "race-preparation",
                cancel=lambda: cancelled.append(True),
            )

        with patch.object(kbd, "_profile_exists_fn", return_value=None), patch.object(
            kbd, "_default_spawn", side_effect=fake_default_spawn,
        ):
            result = kbd.dispatch_once(
                self.conn,
                max_spawn=1,
                reconcile_orphans=False,
            )

        current = self._task(task_id)
        self.assertEqual(result.spawned, [])
        self.assertEqual(current.status, "ready")
        self.assertEqual(current.assignee, "replacement")
        self.assertIsNone(current.current_run_id)
        self.assertEqual(cancelled, [True])
        rejected = [
            event for event in kb.list_events(self.conn, task_id)
            if event.kind == "claim_rejected"
        ]
        self.assertEqual(rejected[-1].payload["reason"], "dispatch_snapshot_changed")

    def test_default_dispatch_rejects_child_identity_mismatch(self) -> None:
        task_id = kb.create_task(
            self.conn,
            title="reject mixed runtime worker",
            assignee="implementer",
            initial_status="blocked",
        )
        self.assertTrue(kb.unblock_task(self.conn, task_id))
        script = (
            "import os; "
            "os.environ['HERMES_KANBAN_EXPECTED_RUNTIME'] = '{}'; "
            "from hermes_cli.kanban_runtime import worker_bootstrap_from_env; "
            "worker_bootstrap_from_env()"
        )
        with patch.object(kbd, "_profile_exists_fn", return_value=None), patch.object(
            kbd,
            "_restart_safe_worker_argv",
            side_effect=lambda task, command, preparation_id=None: command,
        ), patch.object(
            kbd,
            "_worker_argv",
            return_value=[sys.executable, "-c", script],
        ), patch.dict("os.environ", {"PYTHONPATH": str(RUNTIME_ROOT)}, clear=False):
            result = kbd.dispatch_once(
                self.conn,
                max_spawn=1,
                reconcile_orphans=False,
            )
        self.assertEqual(result.spawned, [])
        rejected = self._task(task_id)
        self.assertEqual(rejected.status, "ready")
        self.assertIsNone(rejected.current_run_id)
        self.assertIsNone(rejected.worker_pid)
        event = self.conn.execute(
            "SELECT 1 FROM task_events WHERE task_id = ? AND kind = 'spawn_refused'",
            (task_id,),
        ).fetchone()
        self.assertIsNotNone(event)

    def test_default_dispatch_rejects_invalid_handoff_before_spawn(self) -> None:
        task_id = kb.create_task(
            self.conn,
            title="reject review without handoff",
            assignee="builder",
            initial_status="blocked",
            lifecycle_contract={
                "kind": "code",
                "review_mode": "same_card",
                "reviewer": "reviewer",
                "validation_required": False,
            },
        )
        with kb.write_txn(self.conn):
            self.conn.execute(
                "UPDATE tasks SET status = 'review', assignee = 'reviewer' WHERE id = ?",
                (task_id,),
            )
        with patch.object(kbd, "_profile_exists_fn", return_value=None), patch.object(
            kbd,
            "_default_spawn",
            side_effect=AssertionError("invalid handoff must be rejected before spawn"),
        ):
            result = kbd.dispatch_once(
                self.conn,
                max_spawn=1,
                reconcile_orphans=False,
            )

        self.assertEqual(result.spawned, [])
        self.assertEqual(self._task(task_id).status, "review")
        self.assertTrue(
            any(
                event.kind == "handoff_unverifiable"
                for event in kb.list_events(self.conn, task_id)
            )
        )


    def test_preclaim_spawn_failures_trip_the_dispatcher_breaker(self) -> None:
        task_id = kb.create_task(
            self.conn,
            title="retry failed bootstrap",
            assignee="implementer",
            initial_status="blocked",
        )
        self.assertTrue(kb.unblock_task(self.conn, task_id))
        with patch.object(kbd, "_profile_exists_fn", return_value=None), patch.object(
            kbd, "_default_spawn", side_effect=RuntimeError("bootstrap unavailable"),
        ):
            first = kbd.dispatch_once(
                self.conn,
                max_spawn=1,
                failure_limit=2,
                reconcile_orphans=False,
            )
            second = kbd.dispatch_once(
                self.conn,
                max_spawn=1,
                failure_limit=2,
                reconcile_orphans=False,
            )

        self.assertEqual(first.auto_blocked, [])
        self.assertEqual(second.auto_blocked, [task_id])
        current = self._task(task_id)
        self.assertEqual(current.status, "blocked")
        self.assertEqual(current.consecutive_failures, 2)
        self.assertTrue(
            any(event.kind == "gave_up" for event in kb.list_events(self.conn, task_id))
        )


    def test_dispatcher_and_recovery_use_the_same_dependency_boundary(self) -> None:
        task_id = kb.create_task(
            self.conn,
            title="dispatcher proof",
            assignee="implementer",
            initial_status="blocked",
        )
        self.assertTrue(kb.unblock_task(self.conn, task_id))
        with patch.object(kbd, "_profile_exists_fn", return_value=None):
            result = kbd.dispatch_once(
                self.conn,
                spawn_fn=lambda task, workspace: 424242,
                max_spawn=1,
                reconcile_orphans=False,
            )
        self.assertEqual([entry[0] for entry in result.spawned], [task_id])
        running = kb.get_task(self.conn, task_id)
        self.assertEqual(running.status, "running")
        self.assertEqual(kb._retry_status_for_run(self.conn, task_id), "ready")
        self.assertTrue(kb.reclaim_task(self.conn, task_id, reason="conformance cleanup"))
        self.assertIn(kb.get_task(self.conn, task_id).status, {"ready", "todo"})


if __name__ == "__main__":
    unittest.main()

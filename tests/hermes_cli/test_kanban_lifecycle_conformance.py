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
import shutil
import sqlite3
import sys
import subprocess
import time
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from gateway import kanban_watchers_notifier as notifier
from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_runtime as runtime
from hermes_cli.kanban_lifecycle import evaluate_dependencies, get_lifecycle_state, lifecycle_metadata
from hermes_cli import kanban_runtime_generation as generations
from hermes_cli.kanban_parser import build_parser
from hermes_cli.kanban_runtime import (
    RuntimeIdentityError,
    _fingerprint,
    _git_sha,
    process_start_time,
    assert_runtime_import_root,
    code_identity,
    runtime_identity,
    same_code_identity,
)
from tests.hermes_cli import kanban_conformance_fixture as MODULE


class KanbanLifecycleConformance(MODULE.KanbanConformanceFixture):

    def test_separate_card_fixture_releases_acceptance_in_order(self) -> None:
        fixture = json.loads(MODULE.FIXTURE.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory(prefix="kanban-conformance-git-") as raw_repo:
            repo = Path(raw_repo)
            MODULE._git(repo, "init", "-q")
            MODULE._git(repo, "config", "user.email", "conformance@example.invalid")
            MODULE._git(repo, "config", "user.name", "Kanban Conformance")
            (repo / "README").write_text("base\n", encoding="utf-8")
            MODULE._git(repo, "add", "README")
            MODULE._git(repo, "commit", "-qm", "base")
            base_sha = MODULE._git(repo, "rev-parse", "HEAD")
            (repo / "lifecycle.py").write_text("print('proof')\n", encoding="utf-8")
            MODULE._git(repo, "add", "lifecycle.py")
            MODULE._git(repo, "commit", "-qm", "implementation")
            head_sha = MODULE._git(repo, "rev-parse", "HEAD")

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
            ids = {"implementation": implementation, "review": review, "validation": validation}
            for edge in fixture["edges"]:
                parent = ids[edge["parent"]]
                child = ids[edge["child"]]
                kb.link_tasks(self.conn, parent, child, requirement=edge["requirement"])

            # The v0 fixture parks every card explicitly. Release the manual
            # holds before testing automatic dependency-driven scheduling.
            for role in (review, validation):
                self.assertTrue(kb.unblock_task(self.conn, role))
                self.assertFalse(kb.evaluate_dependencies(self.conn, role)["satisfied"])
                self.assertEqual(self._task(role).status, "todo")
            self.assertTrue(kb.unblock_task(self.conn, implementation))
            impl_run = kb.claim_task(self.conn, implementation, claimer="implementer:conformance")
            self.assertIsNotNone(impl_run)
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
            implementation_events = kb.list_events(self.conn, implementation)
            self.assertNotIn(
                "completed", {event.kind for event in implementation_events}
            )
            self.assertIn(
                "review_requested", {event.kind for event in implementation_events}
            )
            self.assertEqual(
                get_lifecycle_state(self.conn, implementation)["acceptance"],
                "pending",
            )
            self.assertEqual(self._task(implementation).status, "done")
            self.assertEqual(self._task(review).status, "review")
            self.assertIsNone(kb.claim_task(self.conn, validation, claimer="tester:too-early"))

            review_run = kb.claim_review_task(self.conn, review, claimer="reviewer:conformance")
            self.assertIsNotNone(review_run, msg=f"review status={self._task(review).status} dependencies={kb.evaluate_dependencies(self.conn, review)}")
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
            self.assertEqual(self._task(review).status, "done")
            self.assertEqual(self._task(validation).status, "ready")

            validation_run = kb.claim_task(self.conn, validation, claimer="tester:conformance")
            self.assertIsNotNone(validation_run)
            with patch.object(kb, "_fire_task_hook") as fire_hook:
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
            candidate_completion_calls = [
                call
                for call in fire_hook.call_args_list
                if (
                    len(call.args) >= 3
                    and call.args[0] == "kanban_task_completed"
                    and call.args[2] == implementation
                )
            ]
            self.assertEqual(len(candidate_completion_calls), 1)
            self.assertEqual(candidate_completion_calls[0].args[1].id, implementation)
            self.assertEqual(
                candidate_completion_calls[0].kwargs["summary"],
                "Validation passed on reviewed head",
            )
            self.assertEqual(self._task(validation).status, "done")
            acceptance = get_lifecycle_state(self.conn, implementation)
            self.assertEqual(acceptance["acceptance"], "accepted")
            self.assertEqual(acceptance["review_verdict"], "APPROVE")
            self.assertEqual(acceptance["validation_verdict"], "PASS")

    def test_same_card_review_waits_for_validation_before_terminal_completion(self) -> None:
        with tempfile.TemporaryDirectory(prefix="kanban-conformance-validation-") as raw_repo:
            repo = Path(raw_repo)
            MODULE._git(repo, "init", "-q")
            MODULE._git(repo, "config", "user.email", "conformance@example.invalid")
            MODULE._git(repo, "config", "user.name", "Kanban Conformance")
            (repo / "README").write_text("base\n", encoding="utf-8")
            MODULE._git(repo, "add", "README")
            MODULE._git(repo, "commit", "-qm", "base")
            base_sha = MODULE._git(repo, "rev-parse", "HEAD")
            (repo / "lifecycle.py").write_text("print('proof')\n", encoding="utf-8")
            MODULE._git(repo, "add", "lifecycle.py")
            MODULE._git(repo, "commit", "-qm", "implementation")
            head_sha = MODULE._git(repo, "rev-parse", "HEAD")

            implementation = kb.create_task(
                self.conn,
                title="same-card validation implementation",
                assignee="implementer",
                initial_status="blocked",
                workspace_kind="dir",
                workspace_path=str(repo),
                lifecycle_contract={
                    "kind": "code",
                    "review_mode": "same_card",
                    "reviewer": "reviewer",
                    "validation_required": True,
                },
            )
            validation = kb.create_task(
                self.conn,
                title="same-card validation",
                assignee="tester",
                lifecycle_contract={
                    "kind": "validation",
                    "candidate_task_id": implementation,
                },
            )
            kb.link_tasks(
                self.conn, implementation, validation, requirement="review_approved"
            )
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
            review_run = kb.claim_review_task(
                self.conn, implementation, claimer="reviewer:conformance"
            )
            self.assertIsNotNone(review_run)
            with patch.object(kb, "_fire_task_hook") as fire_hook:
                self.assertTrue(
                    kb.complete_task(
                        self.conn,
                        implementation,
                        expected_run_id=review_run.current_run_id,
                        verdict="APPROVE",
                        summary="Review approved",
                        metadata={"reviewed_head_sha": head_sha},
                    )
                )
                fire_hook.assert_not_called()

            events = kb.list_events(self.conn, implementation)
            self.assertNotIn("completed", {event.kind for event in events})
            self.assertIn("validation_requested", {event.kind for event in events})
            self.assertEqual(
                get_lifecycle_state(self.conn, implementation)["acceptance"],
                "pending",
            )
            self.assertEqual(self._task(validation).status, "ready")
    def test_same_card_acceptance_fires_candidate_hook_once(self) -> None:
        with tempfile.TemporaryDirectory(prefix="kanban-conformance-same-card-") as raw_repo:
            repo = Path(raw_repo)
            MODULE._git(repo, "init", "-q")
            MODULE._git(repo, "config", "user.email", "conformance@example.invalid")
            MODULE._git(repo, "config", "user.name", "Kanban Conformance")
            (repo / "README").write_text("base\n", encoding="utf-8")
            MODULE._git(repo, "add", "README")
            MODULE._git(repo, "commit", "-qm", "base")
            base_sha = MODULE._git(repo, "rev-parse", "HEAD")
            (repo / "lifecycle.py").write_text("print('proof')\n", encoding="utf-8")
            MODULE._git(repo, "add", "lifecycle.py")
            MODULE._git(repo, "commit", "-qm", "implementation")
            head_sha = MODULE._git(repo, "rev-parse", "HEAD")

            implementation = kb.create_task(
                self.conn,
                title="same-card implementation",
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
            review_run = kb.claim_review_task(
                self.conn, implementation, claimer="reviewer:conformance"
            )
            self.assertIsNotNone(review_run)
            with patch.object(kb, "_fire_task_hook") as fire_hook:
                self.assertTrue(
                    kb.complete_task(
                        self.conn,
                        implementation,
                        expected_run_id=review_run.current_run_id,
                        verdict="APPROVE",
                        summary="Review approved",
                        metadata={"reviewed_head_sha": head_sha},
                    )
                )

            candidate_completion_calls = [
                call
                for call in fire_hook.call_args_list
                if (
                    len(call.args) >= 3
                    and call.args[0] == "kanban_task_completed"
                    and call.args[2] == implementation
                )
            ]
            self.assertEqual(len(candidate_completion_calls), 1)
            self.assertEqual(
                candidate_completion_calls[0].args[3],
                implementation_run.current_run_id,
            )

            completed = self._task(implementation)
            contract = completed.lifecycle_contract
            self.assertEqual(completed.status, "done")
            self.assertEqual(completed.assignee, "reviewer")
            self.assertEqual(
                get_lifecycle_state(self.conn, implementation)["acceptance"], "accepted",
            )
            before_rejected_updates = tuple(self.conn.iterdump())
            with self.assertRaises(kb.LifecycleContractError):
                kb.update_task(
                    self.conn, implementation,
                    assignee="implementer",
                    lifecycle_contract=contract,
                    expected_version=completed.version,
                    reason="cannot reassign genuine review work",
                )
            self.assertEqual(tuple(self.conn.iterdump()), before_rejected_updates)
            with self.assertRaises(kb.LifecycleContractError):
                kb.update_task(
                    self.conn, implementation,
                    body="revised implementation goal",
                    lifecycle_contract={**contract, "reviewer": "other-reviewer"},
                    expected_version=completed.version,
                    reason="cannot change classified contract while reopening",
                )
            self.assertEqual(tuple(self.conn.iterdump()), before_rejected_updates)
            with self.assertRaises(kb.LifecycleContractError):
                kb.update_task(
                    self.conn, implementation,
                    body="revised implementation goal",
                    assignee="reviewer",
                    lifecycle_contract=contract,
                    expected_version=completed.version,
                    reason="cannot assign implementation to its reviewer",
                )
            self.assertEqual(tuple(self.conn.iterdump()), before_rejected_updates)
            self.assertTrue(
                kb.update_task(
                    self.conn, implementation,
                    body="revised implementation goal",
                    assignee="replacement-builder",
                    model="replacement-model",
                    provider="replacement-provider",
                    lifecycle_contract=contract,
                    expected_version=completed.version,
                    reason="reopen completed code for an explicit goal revision",
                )
            )
            reopened = self._task(implementation)
            self.assertEqual(reopened.status, "ready")
            self.assertEqual(reopened.assignee, "replacement-builder")
            self.assertEqual(reopened.model_override, "replacement-model")
            self.assertEqual(reopened.provider_override, "replacement-provider")
            self.assertEqual(reopened.lifecycle_contract, contract)
            self.assertNotEqual(reopened.goal_revision_id, completed.goal_revision_id)
            self.assertIsNone(reopened.candidate_run_id)
            self.assertIsNone(reopened.completed_at)
            self.assertIsNone(reopened.result)
            self.assertEqual(
                get_lifecycle_state(self.conn, implementation)["acceptance"], "stale",
            )
            self.assertTrue(kb.evaluate_dependencies(self.conn, implementation)["satisfied"])
            self.assertIsNone(
                kb.claim_review_task(self.conn, implementation, claimer="reviewer:conformance")
            )
            implementation_run = kb.claim_task(
                self.conn, implementation, claimer="replacement-builder:conformance"
            )
            self.assertIsNotNone(implementation_run)
            self.assertEqual(implementation_run.assignee, "replacement-builder")


    def test_archived_negative_role_evidence_is_pending(self) -> None:
        with tempfile.TemporaryDirectory(prefix="kanban-conformance-archive-") as raw_repo:
            repo = Path(raw_repo)
            MODULE._git(repo, "init", "-q")
            MODULE._git(repo, "config", "user.email", "conformance@example.invalid")
            MODULE._git(repo, "config", "user.name", "Kanban Conformance")
            (repo / "README").write_text("base\n", encoding="utf-8")
            MODULE._git(repo, "add", "README")
            MODULE._git(repo, "commit", "-qm", "base")
            base_sha = MODULE._git(repo, "rev-parse", "HEAD")
            (repo / "lifecycle.py").write_text("print('proof')\n", encoding="utf-8")
            MODULE._git(repo, "add", "lifecycle.py")
            MODULE._git(repo, "commit", "-qm", "implementation")
            head_sha = MODULE._git(repo, "rev-parse", "HEAD")

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

            self.conn.execute(
                "CREATE TEMP TRIGGER reject_archive_acceptance BEFORE INSERT ON task_events "
                "WHEN NEW.kind = 'acceptance_changed' "
                "BEGIN SELECT RAISE(ABORT, 'reject acceptance event'); END"
            )
            with self.assertRaises(sqlite3.IntegrityError):
                kb.archive_task(self.conn, review)
            self.assertEqual(kb.get_task(self.conn, review).status, "done")
            self.assertFalse(any(
                event.kind == "archived" for event in kb.list_events(self.conn, review)
            ))
            self.conn.execute("DROP TRIGGER reject_archive_acceptance")

            self.assertTrue(kb.archive_task(self.conn, review))
            self.assertEqual(get_lifecycle_state(self.conn, review)["acceptance"], "pending")
            self.assertEqual(
                get_lifecycle_state(self.conn, implementation)["acceptance"], "pending"
            )
            transitions = [
                event.payload for event in kb.list_events(self.conn, implementation)
                if event.kind == "acceptance_changed"
            ]
            self.assertEqual(
                [(event["old"], event["new"]) for event in transitions],
                [("pending", "rejected"), ("rejected", "pending")],
            )
            self.assertEqual(transitions[-1]["source_task_id"], review)


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
            MODULE._git(repo, "init", "-q")
            MODULE._git(repo, "config", "user.email", "conformance@example.invalid")
            MODULE._git(repo, "config", "user.name", "Kanban Conformance")
            (repo / "README").write_text("base\n", encoding="utf-8")
            MODULE._git(repo, "add", "README")
            MODULE._git(repo, "commit", "-qm", "base")
            base_sha = MODULE._git(repo, "rev-parse", "HEAD")
            (repo / "lifecycle.py").write_text("print('proof')\n", encoding="utf-8")
            MODULE._git(repo, "add", "lifecycle.py")
            MODULE._git(repo, "commit", "-qm", "implementation")
            head_sha = MODULE._git(repo, "rev-parse", "HEAD")

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
                    summary="Implemented lifecycle.py at the recorded candidate head",
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
            parents=(parent,),
        )
        unrelated = kb.create_task(
            self.conn,
            title="unrelated ready task",
        )
        with kb.write_txn(self.conn):
            self.conn.execute(
                "UPDATE tasks SET lifecycle_contract = ? WHERE id = ?",
                ("{\"kind\": \"corrupt\"}", parent),
            )
        projection = kb.evaluate_dependencies(self.conn, child)
        self.assertFalse(projection["satisfied"])
        self.assertEqual(projection["blockers"][0]["code"], "lifecycle_unclassified")
        kb.recompute_ready(self.conn)
        self.assertEqual(self._task(child).status, "todo")
        self.assertIsNone(kb.claim_task(self.conn, child, claimer="worker"))
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

    def _completed_candidate(
        self, *, parents=(), review_mode="same_card", verdict="APPROVE",
    ) -> tuple[str, str]:
        """Historical evidence using the same envelopes as live completion."""
        candidate = kb.create_task(
            self.conn, title="completed candidate", assignee="builder", parents=parents,
            lifecycle_contract={
                "kind": "code", "review_mode": review_mode,
                "reviewer": "reviewer", "validation_required": False,
            },
        )
        review = candidate
        if review_mode == "separate_card":
            review = kb.create_task(
                self.conn, title="completed review", assignee="reviewer", parents=[candidate],
                lifecycle_contract={"kind": "review", "candidate_task_id": candidate},
            )
        with kb.write_txn(self.conn):
            run_id = kb._synthesize_ended_run(self.conn, candidate, outcome="completed")
            implementation = lifecycle_metadata(
                self.conn, candidate, phase="implementation", run_id=run_id,
                verdict=None, head_sha="a" * 40,
            )
            self.conn.execute(
                "UPDATE task_runs SET metadata = ? WHERE id = ?",
                (json.dumps({"lifecycle": implementation}), run_id),
            )
            self.conn.execute(
                "UPDATE tasks SET candidate_run_id = ? WHERE id = ?", (run_id, candidate),
            )
            kb._synthesize_ended_run(
                self.conn, review, outcome="completed",
                metadata={"lifecycle": lifecycle_metadata(
                    self.conn, review, phase="review", run_id=run_id,
                    verdict=verdict, head_sha="a" * 40,
                )},
            )
            self.conn.execute(
                "UPDATE tasks SET status = 'done', completed_at = 123456, result = 'obsolete' "
                "WHERE id IN (?, ?)", (candidate, review),
            )
        self.assertEqual(
            get_lifecycle_state(self.conn, candidate)["acceptance"],
            "accepted" if verdict == "APPROVE" else "rejected",
        )
        return candidate, review

    def test_link_rebind_retracts_completed_branch_atomically(self) -> None:
        upstream, gate = self._completed_candidate(
            review_mode="separate_card", verdict="REQUEST_CHANGES",
        )
        child = kb.create_task(self.conn, title="historical general child")
        sibling = kb.create_task(self.conn, title="unrelated sibling")
        for task_id in (child, sibling):
            self.assertTrue(kb.complete_task(self.conn, task_id, result="historical output"))
        with kb.write_txn(self.conn):
            self.conn.executemany(
                "INSERT INTO task_links (parent_id, child_id, requirement) VALUES (?, ?, NULL)",
                ((gate, child), (gate, sibling)),
            )
        candidate, review = self._completed_candidate(
            parents=[child], review_mode="separate_card",
        )
        bridge = kb.create_task(self.conn, title="nested bridge", parents=[review])
        self.assertTrue(kb.complete_task(self.conn, bridge, result="bridge output"))
        nested, _ = self._completed_candidate(parents=[bridge])
        untouched, _ = self._completed_candidate(parents=[sibling])
        ready = kb.create_task(self.conn, title="ready descendant", parents=[nested])
        in_review = kb.create_task(
            self.conn, title="review descendant", parents=[nested], assignee="builder",
        )
        self.assertTrue(kb.request_review(self.conn, in_review, reviewer="reviewer"))
        running = kb.create_task(
            self.conn, title="running descendant", parents=[nested], assignee="worker",
        )
        claimed = kb.claim_task(self.conn, running)
        self.assertIsNotNone(claimed)
        with patch.object(kbd, "_process_fingerprint", return_value="424242:start"):
            kbd._set_worker_pid(self.conn, running, 424242)
        affected = (child, candidate, review, bridge, nested, ready, in_review, running)
        before = {tid: self._task(tid) for tid in (upstream, gate, sibling, untouched, *affected)}
        tables = ("tasks", "task_links", "task_runs", "task_events", "task_comments")
        snapshot = {
            table: [tuple(row) for row in self.conn.execute(f"SELECT * FROM {table} ORDER BY rowid")]
            for table in tables
        }

        def rebind():
            kb.link_tasks(
                self.conn, gate, child, requirement="review_approved",
                expected_parent_version=before[gate].version,
                expected_child_version=before[child].version,
                reason="require review approval for historical output", author="operator",
            )

        self.conn.execute(
            "CREATE TEMP TRIGGER reject_link_acceptance BEFORE INSERT ON task_events "
            "WHEN NEW.kind = 'acceptance_changed' "
            "BEGIN SELECT RAISE(ABORT, 'reject acceptance event'); END"
        )
        with patch.object(kb, "_terminate_reclaimed_worker") as terminate:
            with self.assertRaises(sqlite3.IntegrityError):
                rebind()
            terminate.assert_not_called()
        self.assertEqual(snapshot, {
            table: [tuple(row) for row in self.conn.execute(f"SELECT * FROM {table} ORDER BY rowid")]
            for table in tables
        })
        self.conn.execute("DROP TRIGGER reject_link_acceptance")

        def terminate_after_commit(pid, claim_lock, *, started_at):
            self.assertFalse(self.conn.in_transaction)
            self.assertEqual(
                (pid, claim_lock, started_at),
                (424242, claimed.claim_lock, "424242:start"),
            )
            self.assertEqual(kb.latest_run(self.conn, running).outcome, "reclaimed")
            for tid in (candidate, nested):
                changes = [e for e in kb.list_events(self.conn, tid) if e.kind == "acceptance_changed"]
                self.assertEqual(len(changes), 1)

        with patch.object(kb, "_terminate_reclaimed_worker", side_effect=terminate_after_commit) as terminate:
            rebind()
            terminate.assert_called_once()
        dependencies = kb.evaluate_dependencies(self.conn, child)
        self.assertFalse(dependencies["satisfied"])
        self.assertEqual(dependencies["blockers"][0]["code"], "verdict_conflict")
        self.assertEqual(self._task(gate).version, before[gate].version + 1)
        for tid in affected:
            task = self._task(tid)
            self.assertEqual(task.status, "todo")
            self.assertEqual(task.version, before[tid].version + (2 if tid == child else 1))
            self.assertEqual((
                task.completed_at, task.result, task.candidate_run_id, task.current_run_id,
                task.claim_lock, task.claim_expires, task.worker_pid,
            ), (None,) * 7)
            with self.assertRaises(kb.TaskUpdateConflict):
                kb.update_task(
                    self.conn, tid, expected_version=before[tid].version,
                    reason="stale editor", body="must not overwrite retracted work",
                )
        for tid in (candidate, nested):
            acceptance = get_lifecycle_state(self.conn, tid)["acceptance"]
            self.assertIn(acceptance, {"pending", "stale"})
            changes = [e.payload for e in kb.list_events(self.conn, tid) if e.kind == "acceptance_changed"]
            self.assertEqual(
                [(e["old"], e["new"], e["source_task_id"]) for e in changes],
                [("accepted", acceptance, child)],
            )
        for tid in (upstream, sibling, untouched):
            self.assertEqual(self._task(tid), before[tid])
        self.assertEqual(get_lifecycle_state(self.conn, untouched)["acceptance"], "accepted")
        self.assertFalse(kb.complete_task(
            self.conn, running, expected_run_id=claimed.current_run_id, result="late worker result",
        ))

    def test_new_typed_link_reclaims_produced_work_but_satisfied_links_do_not(self) -> None:
        gate = kb.create_task(self.conn, title="new unfinished dependency")
        other_gate = kb.create_task(self.conn, title="existing unfinished dependency")
        satisfied = kb.create_task(self.conn, title="completed dependency")
        self.assertTrue(kb.complete_task(self.conn, satisfied, summary="Dependency work finished"))
        for status in ("done", "review"):
            with self.subTest(status=status):
                if status == "done":
                    child, _ = self._completed_candidate()
                else:
                    child = kb.create_task(
                        self.conn, title=f"{status} child", assignee="worker",
                    )
                    self.assertTrue(kb.request_review(self.conn, child, reviewer="reviewer"))
                # Existing historical blockers must not make a satisfied new link
                # retract unrelated work, nor make an identical link non-idempotent.
                with kb.write_txn(self.conn):
                    self.conn.execute(
                        "INSERT INTO task_links (parent_id, child_id, requirement) VALUES (?, ?, NULL)",
                        (other_gate, child),
                    )
                before = self._task(child)
                kb.link_tasks(self.conn, satisfied, child)
                self.assertEqual(self._task(child), before)
                events = kb.list_events(self.conn, child)
                kb.link_tasks(self.conn, satisfied, child)
                self.assertEqual(self._task(child), before)
                self.assertEqual(kb.list_events(self.conn, child), events)
                with patch.object(kb, "_terminate_reclaimed_worker") as terminate:
                    kb.link_tasks(self.conn, gate, child, requirement="phase_finished")
                    terminate.assert_not_called()
                task = self._task(child)
                self.assertEqual(task.status, "todo")
                self.assertEqual(task.version, before.version + 1)
                self.assertEqual((
                    task.completed_at, task.result, task.candidate_run_id, task.current_run_id,
                    task.claim_lock, task.claim_expires, task.worker_pid,
                ), (None,) * 7)
                if status == "done":
                    acceptance = get_lifecycle_state(self.conn, child)["acceptance"]
                    self.assertIn(acceptance, {"pending", "stale"})
                    changes = [
                        e.payload for e in kb.list_events(self.conn, child)
                        if e.kind == "acceptance_changed"
                    ]
                    self.assertEqual(
                        [(e["old"], e["new"]) for e in changes], [("accepted", acceptance)],
                    )

    def test_running_dependency_links_preserve_claim_and_require_run_ownership(self) -> None:
        gate = kb.create_task(self.conn, title="unfinished dependency")
        child = kb.create_task(self.conn, title="owned worker", assignee="worker")
        claimed = kb.claim_task(self.conn, child)
        self.assertIsNotNone(claimed)
        with patch.object(kbd, "_process_fingerprint", return_value="424242:start"):
            kbd._set_worker_pid(self.conn, child, 424242)
        before = self._task(child)
        run_before = kb.latest_run(self.conn, child)
        snapshot = tuple(self.conn.iterdump())
        with patch.object(kb, "_terminate_reclaimed_worker") as terminate:
            for run_id in (None, claimed.current_run_id + 1):
                with self.subTest(run_id=run_id), self.assertRaises(ValueError):
                    kb.link_tasks(
                        self.conn, gate, child, requirement="phase_finished",
                        expected_child_run_id=run_id,
                    )
                self.assertEqual(tuple(self.conn.iterdump()), snapshot)
            kb.link_tasks(
                self.conn, gate, child, requirement="phase_finished",
                expected_child_run_id=claimed.current_run_id,
            )
            self.assertFalse(kb.evaluate_dependencies(self.conn, child)["satisfied"])
            self.assertEqual(self._task(child), before)
            self.assertEqual(kb.latest_run(self.conn, child), run_before)
            linked_snapshot = tuple(self.conn.iterdump())
            kb.link_tasks(
                self.conn, gate, child, requirement="phase_finished",
                expected_child_run_id=claimed.current_run_id,
            )
            self.assertEqual(tuple(self.conn.iterdump()), linked_snapshot)
            # Ownership permits adding a dependency before a block handoff,
            # not rebinding an existing edge underneath an active claim.
            with kb.write_txn(self.conn):
                self.conn.execute(
                    "UPDATE task_links SET requirement = NULL WHERE parent_id = ? AND child_id = ?",
                    (gate, child),
                )
            legacy_snapshot = tuple(self.conn.iterdump())
            with self.assertRaises(kb.TaskUpdateConflict):
                kb.link_tasks(
                    self.conn, gate, child, requirement="phase_finished",
                    expected_child_run_id=claimed.current_run_id,
                    expected_parent_version=self._task(gate).version,
                    expected_child_version=before.version, reason="cannot rebind a claim",
                )
            self.assertEqual(tuple(self.conn.iterdump()), legacy_snapshot)
            terminate.assert_not_called()

    def test_archived_children_reject_changed_dependencies(self) -> None:
        parent = kb.create_task(self.conn, title="completed dependency")
        self.assertTrue(kb.complete_task(self.conn, parent, summary="Dependency work finished"))
        archived = kb.create_task(self.conn, title="archived child", parents=[parent])
        self.assertTrue(kb.archive_task(self.conn, archived))
        candidate, _ = self._completed_candidate(parents=[archived])
        before = {task_id: self._task(task_id) for task_id in (parent, archived, candidate)}
        requirement = self.conn.execute(
            "SELECT requirement FROM task_links WHERE parent_id = ? AND child_id = ?",
            (parent, archived),
        ).fetchone()["requirement"]
        kb.link_tasks(self.conn, parent, archived, requirement=requirement)
        self.assertEqual({task_id: self._task(task_id) for task_id in before}, before)
        for rebind in (False, True):
            with self.subTest(rebind=rebind):
                gate = kb.create_task(self.conn, title="unfinished dependency")
                if rebind:
                    with kb.write_txn(self.conn):
                        self.conn.execute(
                            "INSERT INTO task_links (parent_id, child_id, requirement) VALUES (?, ?, NULL)",
                            (gate, archived),
                        )
                events = kb.list_events(self.conn, archived)
                with self.assertRaises(kb.LifecycleContractError):
                    kb.link_tasks(
                        self.conn, gate, archived, requirement="phase_finished",
                        expected_parent_version=self._task(gate).version,
                        expected_child_version=before[archived].version,
                        reason="archived dependency edit",
                    )
                self.assertEqual({task_id: self._task(task_id) for task_id in before}, before)
                self.assertEqual(kb.list_events(self.conn, archived), events)
                edge = self.conn.execute(
                    "SELECT requirement FROM task_links WHERE parent_id = ? AND child_id = ?",
                    (gate, archived),
                ).fetchone()
                self.assertEqual(None if edge is None else dict(edge), {"requirement": None} if rebind else None)

    def test_dependency_invalidation_stops_at_unchanged_archived_boundary(self) -> None:
        parent = kb.create_task(self.conn, title="completed branch")
        self.assertTrue(kb.complete_task(self.conn, parent, summary="Branch work finished"))
        archived = kb.create_task(self.conn, title="archived boundary", parents=[parent])
        self.assertTrue(kb.archive_task(self.conn, archived))
        candidate, _ = self._completed_candidate(parents=[archived])
        running = kb.create_task(
            self.conn, title="running beyond archive", parents=[archived], assignee="worker",
        )
        kb.recompute_ready(self.conn)
        claim = kb.claim_task(self.conn, running)
        self.assertIsNotNone(claim)
        kbd._set_worker_pid(self.conn, running, 424242)
        before = {task_id: self._task(task_id) for task_id in (archived, candidate, running)}
        events = {task_id: kb.list_events(self.conn, task_id) for task_id in before}
        gate = kb.create_task(self.conn, title="unfinished dependency")
        with patch.object(kb, "_terminate_reclaimed_worker") as terminate:
            kb.link_tasks(self.conn, gate, parent, requirement="phase_finished")
            terminate.assert_not_called()
        self.assertEqual(self._task(parent).status, "todo")
        self.assertEqual({task_id: self._task(task_id) for task_id in before}, before)
        self.assertEqual({task_id: kb.list_events(self.conn, task_id) for task_id in before}, events)
        self.assertTrue(evaluate_dependencies(self.conn, candidate)["satisfied"])
        self.assertEqual(get_lifecycle_state(self.conn, candidate)["acceptance"], "accepted")
        self.assertEqual(self._task(running).claim_lock, claim.claim_lock)

    def test_binding_repairs_legacy_edges_and_clears_terminal_fields(self) -> None:
        repo_home = tempfile.TemporaryDirectory(prefix="kanban-binding-git-")
        self.addCleanup(repo_home.cleanup)
        repo = Path(repo_home.name)
        MODULE._git(repo, "init", "-q")
        MODULE._git(repo, "config", "user.email", "conformance@example.invalid")
        MODULE._git(repo, "config", "user.name", "Kanban Conformance")
        (repo / "README").write_text("legacy\n", encoding="utf-8")
        MODULE._git(repo, "add", "README")
        MODULE._git(repo, "commit", "-qm", "legacy")
        branch = MODULE._git(repo, "branch", "--show-current")
        head = MODULE._git(repo, "rev-parse", "HEAD")

        candidate = kb.create_task(
            self.conn,
            title="legacy validation candidate",
            assignee="implementer",
            initial_status="blocked",
            lifecycle_contract={
                "kind": "code",
                "review_mode": "same_card",
                "reviewer": "reviewer",
                "validation_required": True,
            },
        )
        contract = self._task(candidate).lifecycle_contract
        with kb.write_txn(self.conn):
            self.conn.execute(
                "UPDATE tasks SET lifecycle_contract = NULL, status = 'done', "
                "completed_at = 123456, result = 'legacy result' WHERE id = ?",
                (candidate,),
            )
        self.assertTrue(
            kb.bind_lifecycle_contract(
                self.conn, candidate, contract,
                expected_version=self._task(candidate).version,
                reason="classify completed historical implementation",
                author="operator",
            )
        )
        rebound_candidate = self._task(candidate)
        self.assertEqual(rebound_candidate.status, "ready")
        self.assertEqual(rebound_candidate.assignee, "implementer")
        self.assertEqual(rebound_candidate.lifecycle_contract, contract)
        self.assertIsNone(rebound_candidate.candidate_run_id)
        self.assertIsNone(rebound_candidate.completed_at)
        self.assertIsNone(rebound_candidate.result)
        self.assertTrue(kb.evaluate_dependencies(self.conn, candidate)["satisfied"])
        self.assertEqual(get_lifecycle_state(self.conn, candidate)["acceptance"], "stale")
        with kb.write_txn(self.conn):
            candidate_goal = int(self._task(candidate).goal_revision_id)
            handoff = {
                "base_sha": head,
                "head_sha": head,
                "branch_name": branch,
                "workspace_path": str(repo),
                "changed_files": [],
            }
            implementation_run = kb._synthesize_ended_run(
                self.conn,
                candidate,
                outcome="completed",
                summary="implementation complete",
                metadata=handoff,
            )
            implementation_lifecycle = {
                "schema": 1,
                "phase": "implementation",
                "candidate_task_id": candidate,
                "candidate_run_id": implementation_run,
                "head_sha": head,
                "goal_revision_ids": {candidate: candidate_goal},
                "task_goal_revision_id": candidate_goal,
                "verdict": None,
            }
            self.conn.execute(
                "UPDATE task_runs SET metadata = ? WHERE id = ?",
                (
                    json.dumps({**handoff, "lifecycle": implementation_lifecycle}, sort_keys=True),
                    implementation_run,
                ),
            )
            review_run = kb._synthesize_ended_run(
                self.conn,
                candidate,
                outcome="completed",
                summary="review approved",
                metadata=handoff,
            )
            review_lifecycle = {
                **implementation_lifecycle,
                "phase": "review",
                "candidate_run_id": implementation_run,
                "verdict": "APPROVE",
            }
            self.conn.execute(
                "UPDATE task_runs SET metadata = ? WHERE id = ?",
                (
                    json.dumps({**handoff, "lifecycle": review_lifecycle}, sort_keys=True),
                    review_run,
                ),
            )
            self.conn.execute(
                "UPDATE tasks SET status = 'done', candidate_run_id = ?, completed_at = 123456 "
                "WHERE id = ?",
                (implementation_run, candidate),
            )

        validation = kb.create_task(
            self.conn,
            title="historical validation card",
            assignee="tester",
            initial_status="blocked",
        )
        downstream = kb.create_task(
            self.conn,
            title="validation dependent",
            assignee="worker",
            initial_status="blocked",
            lifecycle_contract={"kind": "general"},
        )
        with kb.write_txn(self.conn):
            self.conn.execute(
                "UPDATE tasks SET lifecycle_contract = NULL, status = 'done', "
                "completed_at = 123456, result = 'stale' WHERE id = ?",
                (validation,),
            )
            self.conn.execute(
                "UPDATE tasks SET status = 'done', completed_at = 123456, "
                "result = 'obsolete' WHERE id = ?",
                (downstream,),
            )
            self.conn.execute(
                "INSERT INTO task_links (parent_id, child_id, requirement) VALUES (?, ?, NULL)",
                (candidate, validation),
            )
            self.conn.execute(
                "INSERT INTO task_links (parent_id, child_id, requirement) VALUES (?, ?, NULL)",
                (validation, downstream),
            )
        self.assertTrue(
            kb.bind_lifecycle_contract(
                self.conn,
                validation,
                {"kind": "validation", "candidate_task_id": candidate},
                expected_version=self._task(validation).version,
                reason="classify historical validation card",
                author="operator",
            )
        )

        rebound = self._task(validation)
        self.assertEqual(rebound.status, "ready")
        self.assertIsNone(rebound.completed_at)
        self.assertIsNone(rebound.result)
        edge = self.conn.execute(
            "SELECT requirement FROM task_links WHERE parent_id = ? AND child_id = ?",
            (validation, downstream),
        ).fetchone()
        self.assertEqual(edge["requirement"], "validation_passed")
        candidate_edge = self.conn.execute(
            "SELECT requirement FROM task_links WHERE parent_id = ? AND child_id = ?",
            (candidate, validation),
        ).fetchone()
        self.assertEqual(candidate_edge["requirement"], "review_approved")
        self.assertEqual(self._task(downstream).status, "todo")
        self.assertIsNone(self._task(downstream).completed_at)
        self.assertIsNone(self._task(downstream).result)

        validation_run = kb.claim_task(self.conn, validation, claimer="tester")
        self.assertIsNotNone(validation_run)
        self.assertTrue(
            kb.complete_task(
                self.conn,
                validation,
                result="failed validation",
                verdict="FAIL",
                expected_run_id=validation_run.current_run_id,
            )
        )
        self.assertEqual(self._task(validation).status, "done")
        dependency_state = kb.evaluate_dependencies(self.conn, downstream)
        self.assertFalse(dependency_state["satisfied"])
        self.assertEqual(dependency_state["blockers"][0]["requirement"], "validation_passed")
        self.assertEqual(dependency_state["blockers"][0]["code"], "verdict_conflict")

        _, rejected_review = self._completed_candidate(
            review_mode="separate_card", verdict="REQUEST_CHANGES",
        )
        _, approved_review = self._completed_candidate(review_mode="separate_card")
        for gate, requirement, satisfied in (
            (validation, "validation_passed", False),
            (rejected_review, "review_approved", False),
            (approved_review, "review_approved", True),
        ):
            with self.subTest(incoming_requirement=requirement, satisfied=satisfied):
                child = kb.create_task(self.conn, title="historical completed general task")
                self.assertTrue(kb.complete_task(self.conn, child, result="historical output"))
                descendant, _ = self._completed_candidate()
                with kb.write_txn(self.conn):
                    self.conn.execute(
                        "UPDATE tasks SET lifecycle_contract = NULL WHERE id = ?", (child,),
                    )
                    self.conn.executemany(
                        "INSERT INTO task_links (parent_id, child_id, requirement) VALUES (?, ?, NULL)",
                        ((gate, child), (child, descendant)),
                    )
                before_child, before_descendant = self._task(child), self._task(descendant)
                self.assertTrue(kb.bind_lifecycle_contract(
                    self.conn, child, {"kind": "general"},
                    expected_version=before_child.version,
                    reason="classify historical general work",
                ))
                rebound_child = self._task(child)
                self.assertEqual(rebound_child.version, before_child.version + 1)
                self.assertEqual(rebound_child.status, "done" if satisfied else "todo")
                self.assertEqual(
                    rebound_child.result, before_child.result if satisfied else None,
                )
                self.assertEqual(
                    rebound_child.completed_at, before_child.completed_at if satisfied else None,
                )
                self.assertEqual(evaluate_dependencies(self.conn, child)["satisfied"], satisfied)
                edge = self.conn.execute(
                    "SELECT requirement FROM task_links WHERE parent_id = ? AND child_id = ?",
                    (gate, child),
                ).fetchone()
                self.assertEqual(edge["requirement"], requirement)
                self.assertEqual(
                    get_lifecycle_state(self.conn, descendant)["acceptance"],
                    "accepted" if satisfied else "stale",
                )
                rebound_descendant = self._task(descendant)
                self.assertGreater(rebound_descendant.version, before_descendant.version)
                self.assertEqual(rebound_descendant.status, "done" if satisfied else "todo")
                self.assertEqual(
                    rebound_descendant.candidate_run_id,
                    before_descendant.candidate_run_id if satisfied else None,
                )

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
    def test_validation_edges_require_explicit_candidate_validation(self) -> None:
        same_card_candidate = kb.create_task(
            self.conn,
            title="same-card candidate without validation",
            assignee="implementer",
            initial_status="blocked",
            lifecycle_contract={
                "kind": "code",
                "review_mode": "same_card",
                "reviewer": "reviewer",
                "validation_required": False,
            },
        )
        same_card_validation = kb.create_task(
            self.conn,
            title="invalid same-card validation",
            assignee="tester",
            initial_status="blocked",
            lifecycle_contract={
                "kind": "validation",
                "candidate_task_id": same_card_candidate,
            },
        )
        with self.assertRaises(kb.LifecycleContractError):
            kb.link_tasks(
                self.conn,
                same_card_candidate,
                same_card_validation,
                requirement="review_approved",
            )
        with self.assertRaises(kb.LifecycleContractError):
            kb.create_task(
                self.conn,
                title="invalid same-card validation parent",
                assignee="tester",
                initial_status="blocked",
                parents=[same_card_candidate],
                lifecycle_contract={
                    "kind": "validation",
                    "candidate_task_id": same_card_candidate,
                },
            )

        separate_candidate = kb.create_task(
            self.conn,
            title="separate-card candidate without validation",
            assignee="implementer",
            initial_status="blocked",
            lifecycle_contract={
                "kind": "code",
                "review_mode": "separate_card",
                "reviewer": "reviewer",
                "validation_required": False,
            },
        )
        review = kb.create_task(
            self.conn,
            title="valid separate-card review",
            assignee="reviewer",
            initial_status="blocked",
            parents=[separate_candidate],
            lifecycle_contract={"kind": "review", "candidate_task_id": separate_candidate},
        )
        separate_validation = kb.create_task(
            self.conn,
            title="invalid separate-card validation",
            assignee="tester",
            initial_status="blocked",
            lifecycle_contract={
                "kind": "validation",
                "candidate_task_id": separate_candidate,
            },
        )
        with self.assertRaises(kb.LifecycleContractError):
            kb.link_tasks(
                self.conn,
                review,
                separate_validation,
                requirement="review_approved",
            )
        with self.assertRaises(kb.LifecycleContractError):
            kb.create_task(
                self.conn,
                title="invalid separate-card validation parent",
                assignee="tester",
                initial_status="blocked",
                parents=[review],
                lifecycle_contract={
                    "kind": "validation",
                    "candidate_task_id": separate_candidate,
                },
            )

    def test_duplicate_review_edge_reports_the_existing_child(self) -> None:
        candidate = kb.create_task(
            self.conn,
            title="reviewed candidate",
            assignee="implementer",
            initial_status="blocked",
            lifecycle_contract={
                "kind": "code",
                "review_mode": "separate_card",
                "reviewer": "reviewer",
                "validation_required": False,
            },
        )
        review = kb.create_task(
            self.conn,
            title="existing review",
            assignee="reviewer",
            initial_status="blocked",
            parents=[candidate],
            lifecycle_contract={"kind": "review", "candidate_task_id": candidate},
        )
        kb.link_tasks(self.conn, candidate, review, requirement="phase_finished")
        linked = self.conn.execute(
            "SELECT requirement FROM task_links WHERE parent_id = ? AND child_id = ?",
            (candidate, review),
        ).fetchone()
        self.assertEqual(linked["requirement"], "phase_finished")

        before = tuple(self.conn.iterdump())
        with self.assertRaises(kb.LifecycleContractError) as raised:
            kb.create_task(
                self.conn,
                title="duplicate review",
                assignee="reviewer",
                initial_status="blocked",
                parents=[candidate],
                lifecycle_contract={"kind": "review", "candidate_task_id": candidate},
            )
        self.assertIn("no legal lifecycle requirement", str(raised.exception))
        self.assertIn("duplicate lifecycle review edge", str(raised.exception))
        self.assertIn(candidate, str(raised.exception))
        self.assertIn(review, str(raised.exception))
        self.assertEqual(tuple(self.conn.iterdump()), before)

        second = kb.create_task(
            self.conn,
            title="unlinked duplicate review",
            assignee="reviewer",
            initial_status="blocked",
            lifecycle_contract={"kind": "review", "candidate_task_id": candidate},
        )
        before = tuple(self.conn.iterdump())
        with self.assertRaises(kb.LifecycleContractError) as raised:
            kb.link_tasks(self.conn, candidate, second)
        self.assertIn("no legal lifecycle requirement", str(raised.exception))
        self.assertIn("duplicate lifecycle review edge", str(raised.exception))
        self.assertIn(candidate, str(raised.exception))
        self.assertIn(review, str(raised.exception))
        self.assertEqual(tuple(self.conn.iterdump()), before)

    def test_illegal_topology_reports_every_rejection_not_ambiguity(self) -> None:
        first = kb.create_task(
            self.conn,
            title="separate-card candidate",
            assignee="implementer",
            initial_status="blocked",
            lifecycle_contract={
                "kind": "code",
                "review_mode": "separate_card",
                "reviewer": "reviewer",
                "validation_required": False,
            },
        )
        second = kb.create_task(
            self.conn,
            title="same-card candidate",
            assignee="implementer",
            initial_status="blocked",
            lifecycle_contract={
                "kind": "code",
                "review_mode": "same_card",
                "reviewer": "reviewer",
                "validation_required": False,
            },
        )
        before = tuple(self.conn.iterdump())
        with self.assertRaises(kb.LifecycleContractError) as raised:
            kb.link_tasks(self.conn, first, second)
        message = str(raised.exception)
        self.assertIn("no legal lifecycle requirement", message)
        self.assertIn("illegal lifecycle edge", message)
        self.assertIn(f"{first} -> {second}", message)
        self.assertEqual(tuple(self.conn.iterdump()), before)

    def test_default_dispatch_grants_worker_before_claimed_hook(self) -> None:
        task_id = kb.create_task(
            self.conn,
            title="grant before claimed hook",
            assignee="implementer",
            initial_status="blocked",
        )
        self.assertTrue(kb.unblock_task(self.conn, task_id))
        identity = runtime_identity(MODULE.RUNTIME_ROOT)
        order: list[str] = []

        def fake_default_spawn(task, workspace, *, board=None, defer_grant=False, should_stop=None):
            self.assertTrue(defer_grant)

            def grant(_run_id, _claim_lock):
                order.append("grant")

            return kbd.WorkerLaunch(
                identity.pid,
                identity.as_dict(),
                "claimed-hook-order",
                grant=grant,
            )

        def fire_task_hook(event, _task, _task_id, _run_id, **_fields):
            if event == "kanban_task_claimed":
                order.append("claimed")

        with patch.object(kbd, "_profile_exists_fn", return_value=None), patch.object(
            kbd, "_default_spawn", side_effect=fake_default_spawn,
        ), patch.object(kb, "_fire_task_hook", side_effect=fire_task_hook):
            result = kbd.dispatch_once(
                self.conn,
                max_spawn=1,
                reconcile_orphans=False,
            )

        self.assertEqual([entry[0] for entry in result.spawned], [task_id])
        self.assertEqual(order, ["grant", "claimed"])

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

    def test_ancestor_reopen_restarts_same_card_reviews_as_implementation(self) -> None:
        repo = Path(self.home.name) / "ancestor-review-workspace"
        repo.mkdir()
        MODULE._git(repo, "init", "-q")
        MODULE._git(repo, "config", "user.email", "conformance@example.invalid")
        MODULE._git(repo, "config", "user.name", "Kanban Conformance")
        (repo / "README").write_text("base\n", encoding="utf-8")
        MODULE._git(repo, "add", "README")
        MODULE._git(repo, "commit", "-qm", "base")
        base_sha = MODULE._git(repo, "rev-parse", "HEAD")
        (repo / "implementation.py").write_text("print('proof')\n", encoding="utf-8")
        MODULE._git(repo, "add", "implementation.py")
        MODULE._git(repo, "commit", "-qm", "implementation")
        head_sha = MODULE._git(repo, "rev-parse", "HEAD")
        for phase in ("review", "blocked", "running"):
            with self.subTest(phase=phase):
                parent = kb.create_task(self.conn, title=f"{phase} ancestor")
                self.assertTrue(kb.complete_task(self.conn, parent, summary="Ancestor work finished"))
                child = kb.create_task(
                    self.conn, title=f"{phase} descendant", assignee="builder",
                    parents=[parent], workspace_kind="dir", workspace_path=str(repo),
                    lifecycle_contract={
                        "kind": "code", "review_mode": "same_card",
                        "reviewer": "reviewer", "validation_required": False,
                    },
                )
                kb.recompute_ready(self.conn)
                implementation = kb.claim_task(self.conn, child, claimer="builder:conformance")
                self.assertIsNotNone(implementation)
                self.assertTrue(kb.complete_task(
                    self.conn, child, expected_run_id=implementation.current_run_id,
                    summary="Implementation evidence",
                    metadata={"base_sha": base_sha, "head_sha": head_sha, "changed_files": ["implementation.py"]},
                ))
                if phase != "review":
                    review = kb.claim_review_task(self.conn, child, claimer="reviewer:conformance")
                    self.assertIsNotNone(review)
                    if phase == "blocked":
                        self.assertTrue(kb.block_task(
                            self.conn, child, kind="needs_input", reason="human decision needed",
                            expected_run_id=review.current_run_id,
                        ))
                before = self._task(child)
                self.assertEqual(before.assignee, "reviewer")
                with patch.object(kb, "_terminate_reclaimed_worker"):
                    with kb.write_txn(self.conn):
                        self.conn.execute(
                            "UPDATE tasks SET status = 'todo', completed_at = NULL WHERE id = ?", (parent,),
                        )
                        kb.invalidate_descendants_for_parent_reopen(self.conn, parent, author="operator")
                invalidated = self._task(child)
                self.assertIsNone(invalidated.candidate_run_id)
                self.assertEqual(invalidated.assignee, "builder")
                self.assertEqual(invalidated.version, before.version + 1)
                self.assertEqual(invalidated.status, "blocked" if phase == "blocked" else "todo")
                kb.recompute_ready(self.conn)
                self.assertTrue(kb.complete_task(self.conn, parent, summary="Reopened ancestor work finished"))
                kb.recompute_ready(self.conn)
                if phase == "blocked":
                    self.assertEqual(self._task(child).status, "blocked")
                    self.assertTrue(kb.unblock_task(self.conn, child))
                self.assertEqual(self._task(child).status, "ready")
                self.assertIsNone(kb.claim_review_task(self.conn, child, claimer="reviewer:conformance"))
                replacement = kb.claim_task(self.conn, child, claimer="builder:conformance")
                self.assertIsNotNone(replacement)
                self.assertEqual(replacement.assignee, "builder")

    def test_blocked_same_card_review_keeps_reviewer_identity(self) -> None:
        repo = Path(self.home.name) / "workspace"
        repo.mkdir()
        MODULE._git(repo, "init", "-q")
        MODULE._git(repo, "config", "user.email", "conformance@example.invalid")
        MODULE._git(repo, "config", "user.name", "Kanban Conformance")
        (repo / "README").write_text("base\n", encoding="utf-8")
        MODULE._git(repo, "add", "README")
        MODULE._git(repo, "commit", "-qm", "base")
        base_sha = MODULE._git(repo, "rev-parse", "HEAD")
        (repo / "lifecycle.py").write_text("print('proof')\n", encoding="utf-8")
        MODULE._git(repo, "add", "lifecycle.py")
        MODULE._git(repo, "commit", "-qm", "implementation")
        head_sha = MODULE._git(repo, "rev-parse", "HEAD")
        task_id = kb.create_task(
            self.conn,
            title="blocked same-card review",
            assignee="builder",
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
        self.assertTrue(kb.unblock_task(self.conn, task_id))
        implementation_run = kb.claim_task(self.conn, task_id, claimer="builder:conformance")
        self.assertIsNotNone(implementation_run)
        self.assertTrue(
            kb.complete_task(
                self.conn, task_id,
                expected_run_id=implementation_run.current_run_id,
                summary="Implementation evidence",
                metadata={
                    "base_sha": base_sha,
                    "head_sha": head_sha,
                    "changed_files": ["lifecycle.py"],
                },
            )
        )
        review_run = kb.claim_review_task(self.conn, task_id, claimer="reviewer:conformance")
        self.assertIsNotNone(review_run)
        self.assertTrue(
            kb.block_task(
                self.conn, task_id, kind="needs_input",
                reason="human decision needed for review",
                expected_run_id=review_run.current_run_id,
            )
        )
        blocked = self._task(task_id)
        self.assertEqual(blocked.status, "blocked")
        self.assertEqual(blocked.assignee, "reviewer")
        self.assertEqual(blocked.candidate_run_id, implementation_run.current_run_id)
        before_rejected_updates = tuple(self.conn.iterdump())
        with self.assertRaises(kb.LifecycleContractError):
            kb.assign_task(self.conn, task_id, "builder")
        self.assertEqual(tuple(self.conn.iterdump()), before_rejected_updates)
        with self.assertRaises(kb.LifecycleContractError):
            kb.update_task(
                self.conn, task_id,
                assignee="builder",
                lifecycle_contract=blocked.lifecycle_contract,
                expected_version=blocked.version,
                reason="cannot reroute blocked review without a goal revision",
            )
        self.assertEqual(tuple(self.conn.iterdump()), before_rejected_updates)
        self.assertTrue(kb.assign_task(self.conn, task_id, "reviewer"))
        blocked = self._task(task_id)
        self.assertTrue(
            kb.update_task(
                self.conn, task_id,
                title="revised implementation goal after human input",
                expected_version=blocked.version,
                reason="reopen blocked review for implementation",
            )
        )
        reopened = self._task(task_id)
        self.assertEqual(reopened.status, "ready")
        self.assertEqual(reopened.assignee, "builder")
        self.assertEqual(reopened.lifecycle_contract, blocked.lifecycle_contract)
        self.assertEqual(reopened.version, blocked.version + 1)
        self.assertNotEqual(reopened.goal_revision_id, blocked.goal_revision_id)
        self.assertIsNone(reopened.candidate_run_id)
        self.assertIsNone(reopened.completed_at)
        self.assertIsNone(reopened.result)
        self.assertIsNone(kb.claim_review_task(self.conn, task_id, claimer="reviewer:conformance"))
        resumed = kb.claim_task(self.conn, task_id, claimer="builder:conformance")
        self.assertIsNotNone(resumed)
        self.assertEqual(resumed.assignee, "builder")
        (repo / "lifecycle.py").write_text("print('revised proof')\n", encoding="utf-8")
        MODULE._git(repo, "add", "lifecycle.py")
        MODULE._git(repo, "commit", "-qm", "revised implementation")
        revised_head = MODULE._git(repo, "rev-parse", "HEAD")
        self.assertTrue(
            kb.complete_task(
                self.conn, task_id,
                expected_run_id=resumed.current_run_id,
                summary="Revised implementation evidence",
                metadata={
                    "base_sha": head_sha,
                    "head_sha": revised_head,
                    "changed_files": ["lifecycle.py"],
                },
            )
        )
        pending_review = self._task(task_id)
        self.assertEqual(pending_review.status, "review")
        self.assertEqual(pending_review.assignee, "reviewer")
        self.assertEqual(pending_review.candidate_run_id, resumed.current_run_id)
        self.assertEqual(get_lifecycle_state(self.conn, task_id)["acceptance"], "pending")
        self.assertIsNone(kb.claim_task(self.conn, task_id, claimer="builder:conformance"))
        fresh_review = kb.claim_review_task(self.conn, task_id, claimer="reviewer:conformance")
        self.assertIsNotNone(fresh_review)
        self.assertTrue(
            kb.block_task(
                self.conn, task_id, kind="needs_input",
                reason="human decision needed for revised review",
                expected_run_id=fresh_review.current_run_id,
            )
        )
        self.assertEqual(self._task(task_id).status, "blocked")
        self.assertTrue(kb.unblock_task(self.conn, task_id))
        self.assertEqual(self._task(task_id).status, "review")
        self.assertEqual(self._task(task_id).assignee, "reviewer")
        fresh_review = kb.claim_review_task(self.conn, task_id, claimer="reviewer:conformance")
        self.assertIsNotNone(fresh_review)
        self.assertTrue(
            kb.complete_task(
                self.conn, task_id,
                expected_run_id=fresh_review.current_run_id,
                verdict="APPROVE",
                summary="Revised implementation approved",
                metadata={"reviewed_head_sha": revised_head},
            )
        )
        self.assertEqual(self._task(task_id).status, "done")
        self.assertEqual(get_lifecycle_state(self.conn, task_id)["acceptance"], "accepted")


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
        code_downstream = kb.create_task(
            self.conn,
            title="boundary downstream code task",
            assignee="owner",
            initial_status="blocked",
            lifecycle_contract={
                "kind": "code",
                "review_mode": "same_card",
                "reviewer": "alice",
                "validation_required": False,
            },
        )
        kb.link_tasks(self.conn, implementation, review, requirement="phase_finished")
        kb.link_tasks(self.conn, review, validation, requirement="review_approved")
        kb.link_tasks(self.conn, validation, downstream, requirement="validation_passed")
        kb.link_tasks(self.conn, validation, code_downstream, requirement="validation_passed")
        for task_id in (implementation, review, validation, downstream, code_downstream):
            self.assertTrue(kb.archive_task(self.conn, task_id))

        with self.assertRaises(kb.LifecycleContractError):
            kb.delete_archived_task(self.conn, implementation)
        for task_id in (implementation, review, validation, downstream, code_downstream):
            self.assertIsNotNone(kb.get_task(self.conn, task_id))

    def test_archived_lifecycle_graph_can_purge_requested_general_boundary(self) -> None:
        implementation = kb.create_task(
            self.conn,
            title="general boundary candidate",
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
            title="general boundary review",
            assignee="alice",
            initial_status="blocked",
            lifecycle_contract={"kind": "review", "candidate_task_id": implementation},
        )
        validation = kb.create_task(
            self.conn,
            title="general boundary validation",
            assignee="tester",
            initial_status="blocked",
            lifecycle_contract={
                "kind": "validation",
                "candidate_task_id": implementation,
            },
        )
        downstream = kb.create_task(
            self.conn,
            title="general boundary downstream",
            assignee="owner",
            initial_status="blocked",
            lifecycle_contract={"kind": "general"},
        )
        kb.link_tasks(self.conn, implementation, review, requirement="phase_finished")
        kb.link_tasks(self.conn, review, validation, requirement="review_approved")
        kb.link_tasks(self.conn, validation, downstream, requirement="validation_passed")
        task_ids = (implementation, review, validation, downstream)
        for task_id in task_ids:
            self.assertTrue(kb.archive_task(self.conn, task_id))

        with self.assertRaises(kb.LifecycleContractError):
            kb.delete_archived_task(self.conn, implementation)
        self.assertTrue(
            kb.delete_archived_task(
                self.conn,
                downstream,
                requested_task_ids=(downstream, implementation),
            )
        )
        for task_id in task_ids:
            self.assertIsNone(kb.get_task(self.conn, task_id))

    def test_archived_lifecycle_graph_closes_converging_requested_candidates(self) -> None:
        def _candidate_graph(prefix: str, assignee: str) -> tuple[str, str, str]:
            candidate = kb.create_task(
                self.conn,
                title=f"{prefix} candidate",
                assignee=assignee,
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
                title=f"{prefix} review",
                assignee="alice",
                initial_status="blocked",
                lifecycle_contract={"kind": "review", "candidate_task_id": candidate},
            )
            validation = kb.create_task(
                self.conn,
                title=f"{prefix} validation",
                assignee="tester",
                initial_status="blocked",
                lifecycle_contract={"kind": "validation", "candidate_task_id": candidate},
            )
            kb.link_tasks(self.conn, candidate, review, requirement="phase_finished")
            kb.link_tasks(self.conn, review, validation, requirement="review_approved")
            return candidate, review, validation

        candidate_a, review_a, validation_a = _candidate_graph("converging A", "bob")
        candidate_b, review_b, validation_b = _candidate_graph("converging B", "carol")
        boundary = kb.create_task(
            self.conn,
            title="converging general boundary",
            assignee="owner",
            initial_status="blocked",
            lifecycle_contract={"kind": "general"},
        )
        kb.link_tasks(self.conn, validation_a, boundary, requirement="validation_passed")
        kb.link_tasks(self.conn, validation_b, boundary, requirement="validation_passed")
        task_ids = (
            candidate_a, review_a, validation_a,
            candidate_b, review_b, validation_b, boundary,
        )
        for task_id in task_ids:
            self.assertTrue(kb.archive_task(self.conn, task_id))

        with self.assertRaises(kb.LifecycleContractError):
            kb.delete_archived_task(self.conn, candidate_a)
        self.assertTrue(
            kb.delete_archived_task(
                self.conn,
                candidate_a,
                requested_task_ids=(candidate_a, candidate_b, boundary),
            )
        )
        for task_id in task_ids:
            self.assertIsNone(kb.get_task(self.conn, task_id))

    def test_archived_lifecycle_graph_can_purge_through_typed_validation_edge(
        self,
    ) -> None:
        implementation = kb.create_task(
            self.conn,
            title="connected purge candidate",
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
            title="connected purge review",
            assignee="alice",
            initial_status="blocked",
            lifecycle_contract={"kind": "review", "candidate_task_id": implementation},
        )
        validation = kb.create_task(
            self.conn,
            title="connected purge validation",
            assignee="tester",
            initial_status="blocked",
            lifecycle_contract={
                "kind": "validation",
                "candidate_task_id": implementation,
            },
        )
        downstream = kb.create_task(
            self.conn,
            title="connected downstream candidate",
            assignee="owner",
            initial_status="blocked",
            lifecycle_contract={
                "kind": "code",
                "review_mode": "separate_card",
                "reviewer": "alice",
                "validation_required": False,
            },
        )
        downstream_review = kb.create_task(
            self.conn,
            title="connected downstream review",
            assignee="alice",
            initial_status="blocked",
            lifecycle_contract={"kind": "review", "candidate_task_id": downstream},
        )
        kb.link_tasks(self.conn, implementation, review, requirement="phase_finished")
        kb.link_tasks(self.conn, review, validation, requirement="review_approved")
        kb.link_tasks(self.conn, validation, downstream, requirement="validation_passed")
        for task_id in (implementation, review, validation, downstream, downstream_review):
            self.assertTrue(kb.archive_task(self.conn, task_id))

        with self.assertRaises(kb.LifecycleContractError):
            kb.delete_archived_task(self.conn, implementation)
        for task_id in (implementation, review, validation, downstream, downstream_review):
            self.assertIsNotNone(kb.get_task(self.conn, task_id))
        with self.assertRaises(kb.LifecycleContractError):
            kb.delete_archived_task(
                self.conn,
                implementation,
                requested_task_ids=(implementation, downstream_review),
            )
        for task_id in (implementation, review, validation, downstream, downstream_review):
            self.assertIsNotNone(kb.get_task(self.conn, task_id))

        self.assertTrue(
            kb.delete_archived_task(
                self.conn,
                implementation,
                requested_task_ids=(implementation, downstream),
            )
        )
        for task_id in (implementation, review, validation, downstream, downstream_review):
            self.assertIsNone(kb.get_task(self.conn, task_id))

    def test_worker_generation_survives_source_and_dependency_replacement(self) -> None:
        with tempfile.TemporaryDirectory(prefix="kanban-conformance-generation-") as raw:
            base = Path(raw)
            source = base / "install"
            MODULE._make_runtime_fixture(source)
            first = MODULE._prepare_fixture_generation(source)
            preparation = base / "preparation.json"
            receipt = base / "first.json"
            release = base / "release"
            env = {
                **os.environ,
                **first.env,
                "HERMES_KANBAN_BOOTSTRAP_PATH": str(preparation),
                "HERMES_KANBAN_PREPARATION_ID": "generation-proof",
                "HERMES_KANBAN_EXPECTED_RUNTIME": runtime.encode_identity(first.identity),
                "HERMES_TEST_RUNTIME_RECEIPT": str(receipt),
                "HERMES_TEST_RUNTIME_RELEASE": str(release),
            }
            child = subprocess.Popen(first.command_prefix, stdin=subprocess.PIPE, env=env)
            try:
                early = MODULE._wait_for_receipt(preparation)
                actual = runtime.verify_worker_ready(
                    early, first.identity, pid=child.pid, preparation_id="generation-proof",
                )
                child.stdin.write(json.dumps({
                    "continue_imports": True, "preparation_id": "generation-proof",
                    "runtime_identity": actual.as_dict(),
                }).encode() + b"\n")
                child.stdin.flush()
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline:
                    ready = MODULE._wait_for_receipt(preparation)
                    if ready.get("post_import"):
                        break
                    time.sleep(0.02)
                self.assertTrue(ready.get("post_import"))
                self.assertFalse(receipt.exists())
                child.stdin.write(json.dumps({
                    "grant": True, "preparation_id": "generation-proof",
                    "runtime_identity": actual.as_dict(), "run_id": 23, "claim_lock": "old-claim",
                }).encode() + b"\n")
                child.stdin.close()
                for name in ("fixture_early.py", "fixture_lazy.py"):
                    (source / name).write_text("value = 2\n", encoding="utf-8")
                for name in ("third_party_early", "third_party_dynamic"):
                    dependency = base / "site-packages" / name
                    (dependency / "__init__.py").write_text("value = 2\n", encoding="utf-8")
                    (dependency / "data.bin").write_bytes(b"dependency-v2")
                for directory, _ in runtime._RUNTIME_RESOURCE_ROOTS:
                    (source / directory / "marker.txt").write_text(f"{directory}:v2\n", encoding="utf-8")
                generations.cleanup_runtime_generation(first.root)
                generations.sweep_runtime_generations()
                self.assertTrue(first.root.exists())
                release.touch()
                observed = MODULE._wait_for_receipt(receipt)
                self.assertEqual(child.wait(timeout=10), 0)
                self.assertEqual(observed["early"], [1, 1])
                self.assertEqual(observed["lazy"], [1, 1])
                self.assertEqual(observed["dependency_data"], "dependency-v1")
                self.assertEqual(observed["run"], "23")
                self.assertEqual(observed["claim"], "old-claim")
                for location in observed["locations"]:
                    self.assertTrue(Path(location).is_relative_to(first.root))
                for directory, _ in runtime._RUNTIME_RESOURCE_ROOTS:
                    self.assertEqual(observed["resources"][directory], f"{directory}:v1\n")
                second = MODULE._prepare_fixture_generation(source)
                try:
                    self.assertFalse(same_code_identity(first.identity, second.identity))
                    self.assertNotEqual(first.identity.dependency_fingerprint, second.identity.dependency_fingerprint)
                    next_receipt = base / "second.json"
                    next_env = {**os.environ, **second.env, "HERMES_TEST_RUNTIME_RECEIPT": str(next_receipt)}
                    completed = subprocess.run(second.command_prefix, env=next_env, check=False, timeout=10)
                    self.assertEqual(completed.returncode, 0)
                    current = MODULE._wait_for_receipt(next_receipt)
                    self.assertEqual(current["early"], [2, 2])
                    self.assertEqual(current["lazy"], [2, 2])
                    self.assertEqual(current["dependency_data"], "dependency-v2")
                finally:
                    generations.cleanup_runtime_generation(second.root)
            finally:
                if child.poll() is None:
                    child.kill()
                    child.wait(timeout=10)
                generations.cleanup_runtime_generation(first.root)
            self.assertFalse(first.root.exists())

    @unittest.skipUnless(os.name == "posix", "executable modes are POSIX-specific")
    def test_generation_preserves_executable_resources(self) -> None:
        with tempfile.TemporaryDirectory(prefix="kanban-conformance-mode-") as raw:
            source = Path(raw) / "install"
            MODULE._make_runtime_fixture(source)
            prepared = MODULE._prepare_fixture_generation(source)
            try:
                helper = Path(prepared.env["HERMES_BUNDLED_SKILLS"]) / "helper"
                completed = subprocess.run([str(helper)], capture_output=True, text=True, check=True)
                self.assertEqual(completed.stdout, "sealed-helper")
            finally:
                generations.cleanup_runtime_generation(prepared.root, force=True)

    def test_generation_sweep_preserves_live_owner_and_rejects_pid_reuse(self) -> None:
        with tempfile.TemporaryDirectory(prefix="kanban-conformance-owner-") as raw:
            source = Path(raw) / "install"
            MODULE._make_runtime_fixture(source)
            prepared = MODULE._prepare_fixture_generation(source)
            try:
                generations.sweep_runtime_generations()
                self.assertTrue(prepared.root.exists())
                generations.write_runtime_generation_owner(
                    prepared.root, pid=os.getpid(), start_time=process_start_time() + 1,
                )
                generations.sweep_runtime_generations()
                self.assertFalse(prepared.root.exists())
            finally:
                generations.cleanup_runtime_generation(prepared.root, force=True)

    def test_sealed_main_never_runs_install_recovery(self) -> None:
        with tempfile.TemporaryDirectory(prefix="kanban-conformance-recovery-") as raw:
            source = Path(raw) / "install"
            MODULE._make_runtime_fixture(source)
            for name in ("main.py", "_subprocess_compat.py", "_startup_fast.py",
                         "_parser.py", "_early_recovery.py"):
                shutil.copy2(MODULE.RUNTIME_ROOT / "hermes_cli" / name, source / "hermes_cli" / name)
            for name in ("hermes_bootstrap.py", "hermes_constants.py", "hermes_constants_scratch.py"):
                shutil.copy2(MODULE.RUNTIME_ROOT / name, source / name)
            shutil.copytree(
                MODULE.RUNTIME_ROOT / "pm", source / "pm",
                ignore=shutil.ignore_patterns("__pycache__"),
            )
            recovery_marker = Path(raw) / "recovery-mutated-install"
            # Keep the real interrupted-pull guard; record any dependency
            # recovery invocation without allowing an installer to run.
            with (source / "hermes_cli" / "_early_recovery.py").open("a", encoding="utf-8") as handle:
                handle.write(
                    f"\ndef recover_if_needed(*args, **kwargs):\n"
                    f"    Path({str(recovery_marker)!r}).touch()\n"
                )
            prepared = MODULE._prepare_fixture_generation(source)
            try:
                completed = subprocess.run(
                    prepared.command_prefix + ["runtime-identity", "--json"],
                    env={**os.environ, **prepared.env}, capture_output=True, text=True, timeout=10,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertTrue(same_code_identity(prepared.identity, json.loads(completed.stdout)))
                self.assertFalse(recovery_marker.exists())
            finally:
                generations.cleanup_runtime_generation(prepared.root)

    def test_early_recovery_defers_to_live_installation_lock(self) -> None:
        script = """
import importlib
import sys
from pathlib import Path
from hermes_cli import _early_recovery as recovery
from hermes_cli import kanban_runtime_generation as generations
from pm import client, paths
generations._runtime_storage_root = lambda: Path(sys.argv[2])
root = Path(sys.argv[1])
paths.repo_root = lambda: root
engine = importlib.import_module("pm.install")
# This child runs the real PM mutation entrypoint and kernel locks, but never
# downloads or builds dependencies. Only the package's build is a test double.
client.sync_venv = engine.sync_venv
package = engine.get_package("venv")
package.expected_stamp = lambda *args, **kwargs: "conformance-repair"
def install(*args, **kwargs):
    (root / "installer-ran").touch()
package.apply = install
assert recovery.recover_if_needed(project_root=root, argv=[]) is (sys.argv[3] == "recover")
"""
        for marker in (".update-incomplete", ".lazy-refresh-incomplete"):
            with self.subTest(marker=marker), tempfile.TemporaryDirectory(
                prefix="kanban-conformance-recovery-lock-",
            ) as raw:
                source = Path(raw)
                (source / "pyproject.toml").write_text(
                    '[project]\nname = "recovery-fixture"\nversion = "1"\ndependencies = []\n',
                    encoding="utf-8",
                )
                (source / marker).write_text("pid=0\n", encoding="utf-8")
                env = {
                    **os.environ, "PYTHONPATH": str(MODULE.RUNTIME_ROOT), "TMPDIR": raw,
                    "HOME": raw, "USERPROFILE": raw,
                    "HERMES_RUNTIME_DIR": str(source / "tools"),
                }
                env.pop("HERMES_INSTALL_ROOT", None)
                command = [sys.executable, "-c", script, raw, str(source / "runtime-storage")]
                # Other test files share the interpreter installation, not this fixture's locks.
                with patch.object(generations, "_runtime_storage_root", lambda: source / "runtime-storage"), generations.installation_mutation_lock(source):
                    deferred = subprocess.run(
                        command + ["defer"], env=env, capture_output=True, text=True, timeout=10,
                    )
                self.assertEqual(deferred.returncode, 0, deferred.stderr)
                self.assertFalse((source / "installer-ran").exists())
                self.assertEqual((source / marker).read_text(encoding="utf-8"), "pid=0\n")
                recovered = subprocess.run(
                    command + ["recover"], env=env, capture_output=True, text=True, timeout=10,
                )
                self.assertEqual(recovered.returncode, 0, recovered.stderr)
                self.assertTrue((source / "installer-ran").exists())
                self.assertFalse((source / marker).exists())

    def test_reaped_launcher_exit_uses_verified_worker_pid(self) -> None:
        launcher_pid = 2_147_482_999
        worker_pid = launcher_pid - 1
        rate_limit = kb.KANBAN_RATE_LIMIT_EXIT_CODE
        kbd._worker_pid_aliases[launcher_pid] = worker_pid
        kbd._worker_processes[launcher_pid] = SimpleNamespace(returncode=None)
        try:
            kbd._record_worker_exit(launcher_pid, rate_limit << 8)
            self.assertEqual(kbd._classify_worker_exit(worker_pid), ("rate_limited", rate_limit))
        finally:
            kbd._worker_pid_aliases.pop(launcher_pid, None)
            kbd._worker_processes.pop(launcher_pid, None)
            kbd._recent_worker_exits.pop(worker_pid, None)

    @unittest.skipUnless(os.name == "posix", "waitpid is POSIX-specific")
    def test_worker_reaper_does_not_consume_unregistered_children(self) -> None:
        registered_pid = 2_147_482_998
        kbd._worker_processes[registered_pid] = SimpleNamespace(returncode=None)
        calls: list[int] = []

        def waitpid(pid: int, options: int) -> tuple[int, int]:
            calls.append(pid)
            self.assertEqual(options, os.WNOHANG)
            self.assertEqual(pid, registered_pid)
            return registered_pid, 0

        try:
            with patch.object(kbd.os, "waitpid", side_effect=waitpid):
                self.assertEqual(kbd.reap_worker_zombies(), [registered_pid])
            self.assertEqual(calls, [registered_pid])
        finally:
            kbd._worker_processes.pop(registered_pid, None)
            kbd._recent_worker_exits.pop(registered_pid, None)

    def test_runtime_sha_reads_packed_worktree_refs_from_common_git_dir(self) -> None:
        with tempfile.TemporaryDirectory(prefix="kanban-conformance-packed-ref-") as raw_root:
            root = Path(raw_root) / "worktree"
            common = Path(raw_root) / "repo.git"
            git_dir = common / "worktrees" / "worktree"
            root.mkdir()
            git_dir.mkdir(parents=True)
            (root / ".git").write_text(f"gitdir: {git_dir}\n", encoding="utf-8")
            (git_dir / "HEAD").write_text("ref: refs/heads/feature\n", encoding="ascii")
            (git_dir / "commondir").write_text("../..\n", encoding="ascii")
            sha = "a" * 40
            (common / "packed-refs").write_text(f"{sha} refs/heads/feature\n", encoding="ascii")
            self.assertEqual(_git_sha(root), sha)

    def test_embedded_dispatcher_freezes_identity_before_first_tick(self) -> None:
        with tempfile.TemporaryDirectory(prefix="kanban-conformance-dispatcher-") as raw_root:
            root = Path(raw_root)
            package = root / "hermes_cli"
            package.mkdir()
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
            package = root / "hermes_cli"
            package.mkdir()
            (package / "module.py").write_text("value = 1\n", encoding="utf-8")
            skill = root / "skills" / "devops" / "sdlc-review" / "SKILL.md"
            skill.parent.mkdir(parents=True)
            skill.write_text("review instructions v1\n", encoding="utf-8")
            locale = root / "locales" / "en.yaml"
            locale.parent.mkdir(parents=True)
            locale.write_text("locale: v1\n", encoding="utf-8")
            plugin_manifest = root / "plugins" / "sample" / "plugin.yaml"
            plugin_manifest.parent.mkdir(parents=True)
            plugin_manifest.write_text("name: v1\n", encoding="utf-8")

            before = _fingerprint(root)
            skill.write_text("review instructions v2\n", encoding="utf-8")
            self.assertNotEqual(before, _fingerprint(root))
            before = _fingerprint(root)
            locale.write_text("locale: v2\n", encoding="utf-8")
            self.assertNotEqual(before, _fingerprint(root))
            before = _fingerprint(root)
            plugin_manifest.write_text("name: v2\n", encoding="utf-8")
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

    def test_generation_fences_packaged_resource_roots(self) -> None:
        with tempfile.TemporaryDirectory(prefix="kanban-conformance-resources-") as raw:
            source = Path(raw) / "install"
            MODULE._make_runtime_fixture(source)
            overrides = {}
            for directory, env_var in runtime._RUNTIME_RESOURCE_ROOTS:
                external = Path(raw) / f"packaged-{directory}"
                shutil.move(str(source / directory), external)
                overrides[env_var] = str(external)
            with patch.dict(os.environ, overrides), patch.object(
                generations, "_runtime_import_roots", return_value=[Path(raw) / "site-packages"],
            ):
                prepared = generations.prepare_runtime_generation(runtime_identity(source))
            try:
                for directory, env_var in runtime._RUNTIME_RESOURCE_ROOTS:
                    original = Path(overrides[env_var]) / "marker.txt"
                    original.write_text("replaced\n", encoding="utf-8")
                    copied = Path(prepared.env[env_var]) / "marker.txt"
                    self.assertEqual(copied.read_text(encoding="utf-8"), f"{directory}:v1\n")
                    self.assertTrue(copied.is_relative_to(prepared.root))
            finally:
                generations.cleanup_runtime_generation(prepared.root, force=True)

    def test_completion_rechecks_candidate_head_at_commit_boundary(self) -> None:
        with tempfile.TemporaryDirectory(prefix="kanban-conformance-race-") as raw_repo:
            repo = Path(raw_repo)
            MODULE._git(repo, "init", "-q")
            MODULE._git(repo, "config", "user.email", "conformance@example.invalid")
            MODULE._git(repo, "config", "user.name", "Kanban Conformance")
            (repo / "README").write_text("base\n", encoding="utf-8")
            MODULE._git(repo, "add", "README")
            MODULE._git(repo, "commit", "-qm", "base")
            base_sha = MODULE._git(repo, "rev-parse", "HEAD")
            (repo / "lifecycle.py").write_text("print('proof')\n", encoding="utf-8")
            MODULE._git(repo, "add", "lifecycle.py")
            MODULE._git(repo, "commit", "-qm", "implementation")
            head_sha = MODULE._git(repo, "rev-parse", "HEAD")

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
                    MODULE._git(repo, "add", "race.py")
                    MODULE._git(repo, "commit", "-qm", "candidate advanced")
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
        identity = runtime_identity(MODULE.RUNTIME_ROOT)
        self.assertEqual(identity, runtime_identity(MODULE.RUNTIME_ROOT, pid=identity.pid, start_time=identity.start_time))
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
                    assert_runtime_import_root(MODULE.RUNTIME_ROOT)
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
        stale_event = SimpleNamespace(
            kind="acceptance_changed",
            payload={
                "old": "accepted",
                "new": "stale",
                "phase": "validation",
                "result": "PASS",
            },
        )
        stale_message, stale_wake, stale_detail = notifier._fmt_acceptance_changed(
            stale_event, notice
        )
        self.assertIn("acceptance: accepted → stale", stale_message)
        self.assertIn("(validation result: PASS)", stale_message)
        self.assertNotEqual(stale_message, "ℹ️ Lifecycle validation result: PASS")
        self.assertIsNone(stale_wake)
        self.assertIsNone(stale_detail)
    def test_default_dispatch_fences_child_before_claim(self) -> None:
        task_id = kb.create_task(
            self.conn, title="default spawn identity proof",
            assignee="implementer", initial_status="blocked",
        )
        self.assertTrue(kb.unblock_task(self.conn, task_id))
        with tempfile.TemporaryDirectory(prefix="kanban-worker-proof-") as raw:
            source = Path(raw) / "install"
            MODULE._make_runtime_fixture(source)
            receipt = Path(raw) / "grant.json"
            with patch.object(kbd, "_profile_exists_fn", return_value=None), patch.object(
                kbd, "_restart_safe_worker_argv",
                side_effect=lambda task, command, preparation_id=None: command,
            ), patch.object(
                generations, "prepare_runtime_generation",
                side_effect=lambda expected, workspace=None, profile_home=None, project_plugins_enabled=None: MODULE._prepare_fixture_generation(
                    source, workspace=workspace, profile_home=profile_home,
                    project_plugins_enabled=project_plugins_enabled,
                ),
            ), patch.dict(os.environ, {
                "HERMES_TEST_RUNTIME_RECEIPT": str(receipt), "HERMES_BIN": "",
            }):
                result = kbd.dispatch_once(self.conn, max_spawn=1, reconcile_orphans=False)
            self.assertEqual([entry[0] for entry in result.spawned], [task_id])
            claimed = self._task(task_id)
            process = kbd._worker_processes[claimed.worker_pid]
            try:
                granted = MODULE._wait_for_receipt(receipt)
                self.assertEqual(granted["run"], str(claimed.current_run_id))
                self.assertEqual(granted["claim"], claimed.claim_lock)
                self.assertEqual(granted["granted"], "1")
                metadata = json.loads(self.conn.execute(
                    "SELECT metadata FROM task_runs WHERE id = ?", (claimed.current_run_id,),
                ).fetchone()["metadata"])
                self.assertEqual(metadata["runtime_identity"], granted["identity"])
                process.wait(timeout=10)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=10)
                kbd._record_worker_exit(process.pid, process.returncode << 8)


    def test_default_dispatch_rejects_changed_execution_snapshot(self) -> None:
        task_id = kb.create_task(
            self.conn,
            title="reject changed worker configuration",
            assignee="implementer",
            initial_status="blocked",
        )
        self.assertTrue(kb.unblock_task(self.conn, task_id))
        identity = runtime_identity(MODULE.RUNTIME_ROOT)
        cancelled: list[bool] = []

        def fake_default_spawn(task, workspace, *, board=None, defer_grant=False, should_stop=None):
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

    # #39: an authorized existing-PR continuation dispatches the SAME
    # coordinator — one claimed run, a real post-grant child, no duplicate
    # worker, and no loss of graph/upstream evidence. (No docstring: unittest's
    # verbose runner interleaves it with the "... ok" trailer, which the
    # lifecycle verification receipt parses by line.)
    def test_existing_pr_continuation_starts_worker(self) -> None:
        upstream = kb.create_task(
            self.conn, title="upstream review", assignee="reviewer",
            initial_status="blocked",
        )
        self.assertTrue(kb.unblock_task(self.conn, upstream))
        upstream_run = kb.claim_task(self.conn, upstream, claimer="reviewer:upstream")
        self.assertIsNotNone(upstream_run)
        self.assertTrue(kb.complete_task(
            self.conn, upstream, expected_run_id=upstream_run.current_run_id,
            summary="upstream acceptance evidence",
        ))
        coordinator = kb.create_task(
            self.conn, title="repair the existing PR", assignee="implementer",
            parents=[upstream], triage=True,
        )
        self.assertEqual(self._task(coordinator).status, "triage")
        # Native triage requeue onto a satisfied typed dependency.
        self.assertTrue(kb.update_task(
            self.conn, coordinator, expected_version=self._task(coordinator).version,
            reason="ready for publication work", transition="triage_to_ready",
        ))
        self.assertEqual(self._task(coordinator).status, "ready")
        pr_url = "https://github.com/example/repo/pull/4242"
        kb.add_comment(
            self.conn, coordinator, author="worker",
            body=f"Published {pr_url}; continue published-head checks.",
        )
        self.assertEqual(kbd.check_respawn_guard(self.conn, coordinator), "active_pr")
        # An ordinary goal revision still does not acknowledge the PR.
        self.assertTrue(kb.update_task(
            self.conn, coordinator, expected_version=self._task(coordinator).version,
            reason="Polish the PR with review loops",
            body="Polish the PR with review loops.",
        ))
        self.assertEqual(kbd.check_respawn_guard(self.conn, coordinator), "active_pr")
        self.assertTrue(kb.update_task(
            self.conn, coordinator, expected_version=self._task(coordinator).version,
            reason="Explicitly authorized: continue the existing PR",
            transition="continue_existing_pr",
            authorized_pr_urls=[pr_url],
        ))
        self.assertIsNone(kbd.check_respawn_guard(self.conn, coordinator))
        authorization = kb._pr_continuation(self.conn, coordinator)
        self.assertEqual(authorization["pr_urls"], [pr_url])
        graph_before = (
            kb.parent_ids(self.conn, coordinator), kb.child_ids(self.conn, coordinator),
        )
        upstream_result_before = self._task(upstream).result

        with tempfile.TemporaryDirectory(prefix="kanban-pr-continuation-") as raw:
            source = Path(raw) / "install"
            MODULE._make_runtime_fixture(source)
            receipt = Path(raw) / "grant.json"
            release = Path(raw) / "release"
            with patch.object(kbd, "_profile_exists_fn", return_value=None), patch.object(
                kbd, "_restart_safe_worker_argv",
                side_effect=lambda task, command, preparation_id=None: command,
            ), patch.object(
                generations, "prepare_runtime_generation",
                side_effect=lambda expected, workspace=None, profile_home=None, project_plugins_enabled=None: MODULE._prepare_fixture_generation(
                    source, workspace=workspace, profile_home=profile_home,
                    project_plugins_enabled=project_plugins_enabled,
                ),
            ), patch.dict(os.environ, {
                "HERMES_TEST_RUNTIME_RECEIPT": str(receipt),
                "HERMES_TEST_RUNTIME_RELEASE": str(release),
                "HERMES_BIN": "",
            }):
                result = kbd.dispatch_once(self.conn, max_spawn=1, reconcile_orphans=False)
                self.assertEqual([entry[0] for entry in result.spawned], [coordinator])
                claimed = self._task(coordinator)
                self.assertEqual(claimed.status, "running")
                process = kbd._worker_processes[claimed.worker_pid]
                try:
                    # The live claim owns the card: a second tick must not
                    # launch a duplicate worker for the same PR continuation.
                    second = kbd.dispatch_once(
                        self.conn, max_spawn=1, reconcile_orphans=False,
                    )
                    self.assertEqual(second.spawned, [])
                    self.assertEqual(
                        self._task(coordinator).current_run_id, claimed.current_run_id,
                    )
                    release.write_text("go\n", encoding="utf-8")
                    granted = MODULE._wait_for_receipt(receipt)
                    self.assertEqual(granted["granted"], "1")
                    self.assertEqual(granted["task"], coordinator)
                    self.assertEqual(granted["run"], str(claimed.current_run_id))
                    self.assertEqual(granted["claim"], claimed.claim_lock)
                    metadata = json.loads(self.conn.execute(
                        "SELECT metadata FROM task_runs WHERE id = ?",
                        (claimed.current_run_id,),
                    ).fetchone()["metadata"])
                    self.assertEqual(metadata["runtime_identity"], granted["identity"])
                    self.assertEqual(metadata["pr_continuation"]["pr_urls"], [pr_url])
                    self.assertEqual(
                        metadata["pr_continuation"]["event_id"],
                        authorization["event_id"],
                    )
                    claimed_event = [
                        event for event in kb.list_events(self.conn, coordinator)
                        if event.kind == "claimed"
                    ][-1]
                    self.assertEqual(
                        claimed_event.payload["pr_continuation_event_id"],
                        authorization["event_id"],
                    )
                    process.wait(timeout=10)
                finally:
                    if process.poll() is None:
                        process.kill()
                        process.wait(timeout=10)
                    kbd._record_worker_exit(process.pid, process.returncode << 8)

        # Exactly one run/owner, and the original graph plus upstream immutable
        # evidence are untouched.
        self.assertEqual(len(kb.list_runs(self.conn, coordinator)), 1)
        self.assertEqual(
            (kb.parent_ids(self.conn, coordinator), kb.child_ids(self.conn, coordinator)),
            graph_before,
        )
        self.assertEqual(self._task(upstream).result, upstream_result_before)
        self.assertEqual(self._task(upstream).status, "done")

    def test_existing_pr_continuation_fences_a_new_unauthorized_url_at_claim(self) -> None:
        # A PR URL that arrives while the deferred worker is being prepared
        # cancels the launch: no grant, no run, and the hold is reported as
        # ``active_pr`` rather than as a spawn failure.
        task_id = kb.create_task(
            self.conn, title="PR continuation race", assignee="implementer",
            initial_status="blocked",
        )
        self.assertTrue(kb.unblock_task(self.conn, task_id))
        pr_url = "https://github.com/example/repo/pull/7"
        kb.add_comment(self.conn, task_id, author="worker", body=f"Published {pr_url}")
        self.assertTrue(kb.update_task(
            self.conn, task_id, expected_version=self._task(task_id).version,
            reason="Explicitly authorized: continue the existing PR",
            transition="continue_existing_pr",
            authorized_pr_urls=[pr_url],
        ))
        identity = runtime_identity(MODULE.RUNTIME_ROOT)
        cancelled: list[bool] = []

        def fake_default_spawn(task, workspace, *, board=None, defer_grant=False, should_stop=None):
            self.assertTrue(defer_grant)
            # A brand-new, unauthorized PR appears after the pre-claim guard
            # passed and before the claim transaction opens.
            kb.add_comment(
                self.conn, task.id, author="worker",
                body="Also opened https://github.com/example/repo/pull/8",
            )
            return kbd.WorkerLaunch(
                identity.pid,
                identity.as_dict(),
                "pr-race-preparation",
                cancel=lambda: cancelled.append(True),
            )

        with patch.object(kbd, "_profile_exists_fn", return_value=None), patch.object(
            kbd, "_default_spawn", side_effect=fake_default_spawn,
        ):
            result = kbd.dispatch_once(self.conn, max_spawn=1, reconcile_orphans=False)

        current = self._task(task_id)
        self.assertEqual(result.spawned, [])
        self.assertEqual(result.respawn_guarded, [(task_id, "active_pr")])
        self.assertEqual(cancelled, [True])
        self.assertEqual(current.status, "ready")
        self.assertIsNone(current.current_run_id)
        self.assertIsNone(current.worker_pid)
        self.assertEqual(kb.list_runs(self.conn, task_id), [])
        rejected = [
            event for event in kb.list_events(self.conn, task_id)
            if event.kind == "claim_rejected"
        ]
        self.assertEqual(rejected[-1].payload["reason"], "active_pr")
        # Re-authorizing now covers BOTH recorded URLs; two competing claims
        # then produce exactly one run and one owner.
        self.assertTrue(kb.update_task(
            self.conn, task_id, expected_version=self._task(task_id).version,
            reason="Explicitly authorized: continue both recorded PRs",
            transition="continue_existing_pr",
            authorized_pr_urls=[pr_url, pr_url.replace("/7", "/8")],
        ))
        self.assertEqual(
            kb._pr_continuation(self.conn, task_id)["pr_urls"], [pr_url, pr_url.replace("/7", "/8")],
        )
        winner = kb.claim_task(self.conn, task_id, claimer="claimer:a")
        self.assertIsNotNone(winner)
        self.assertIsNone(kb.claim_task(self.conn, task_id, claimer="claimer:b"))
        runs = kb.list_runs(self.conn, task_id)
        self.assertEqual([run.claim_lock for run in runs], [winner.claim_lock])
        self.assertEqual(self._task(task_id).current_run_id, winner.current_run_id)

    def test_default_dispatch_rejects_child_identity_mismatch(self) -> None:
        task_id = kb.create_task(
            self.conn,
            title="reject mixed runtime worker",
            assignee="implementer",
            initial_status="blocked",
        )
        self.assertTrue(kb.unblock_task(self.conn, task_id))
        with tempfile.TemporaryDirectory(prefix="kanban-worker-mismatch-") as raw:
            source = Path(raw) / "install"
            MODULE._make_runtime_fixture(source)
            (source / "hermes_cli" / "main.py").write_text(
                "import os\n"
                "os.environ['HERMES_KANBAN_EXPECTED_RUNTIME'] = '{}'\n"
                "from hermes_cli.kanban_runtime import worker_bootstrap_from_env\n"
                "worker_bootstrap_from_env()\n",
                encoding="utf-8",
            )
            with patch.object(kbd, "_profile_exists_fn", return_value=None), patch.object(
                kbd, "_restart_safe_worker_argv",
                side_effect=lambda task, command, preparation_id=None: command,
            ), patch.object(
                generations, "prepare_runtime_generation",
                side_effect=lambda expected, workspace=None, profile_home=None, project_plugins_enabled=None: MODULE._prepare_fixture_generation(
                    source, workspace=workspace, profile_home=profile_home,
                    project_plugins_enabled=project_plugins_enabled,
                ),
            ), patch.dict(os.environ, {"HERMES_BIN": ""}):
                result = kbd.dispatch_once(self.conn, max_spawn=1, reconcile_orphans=False)
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


def _handoff_phase_hint(user_prompt: str) -> str:
    """The phase marker the runtime must put in the judge prompt."""
    for phase in ("implementation", "review", "validation"):
        if f"Lifecycle phase being judged now: {phase}" in user_prompt:
            return phase
    return "generic"


class _PhaseJudge:
    """Deterministic auxiliary-model double keyed on the judge's phase marker.

    ``pairs`` lists the ``(phase, verdict)`` each successive judge call must
    produce. A call whose prompt carries a different phase than expected returns
    ``continue``, so a handoff that reaches the judge without its phase — the
    reported regression — surfaces as a rejected transition instead of a pass.
    """

    def __init__(self, *pairs: tuple[Optional[str], str]):
        self.pairs = list(pairs)
        self.calls: list[tuple[str, str, str, str]] = []

    def __call__(self, call_llm, system_prompt: str, user_prompt: str, timeout) -> str:
        expected, verdict = self.pairs[min(len(self.calls), len(self.pairs) - 1)]
        phase = _handoff_phase_hint(user_prompt)
        self.calls.append((phase, system_prompt, user_prompt, verdict))
        ok = phase == ("generic" if expected is None else expected)
        return json.dumps({
            "verdict": verdict if ok else "continue",
            "reason": "scripted" if ok else f"judge saw phase {phase!r}, expected {expected!r}",
        })


class KanbanPhaseAwareHandoff(MODULE.KanbanConformanceFixture):
    GUARD_REASON = "same-card review runs must submit a review verdict"
    GOAL = (
        "Land the phase-aware handoff fix.\n"
        "Acceptance: implemented and locally verified, independently reviewed by the "
        "reviewer profile, then validated by the tester profile."
    )

    def setUp(self) -> None:
        super().setUp()
        for name in ("HERMES_KANBAN_TASK", "HERMES_KANBAN_RUN_ID", "HERMES_DELEGATED_CHILD_CONTEXT"):
            os.environ.pop(name, None)
        kb._INITIALIZED_PATHS.clear()
        kb.init_db()

    # --- fixtures ---------------------------------------------------------
    def _repo(self) -> tuple[Path, str]:
        scratch = tempfile.TemporaryDirectory(prefix="kanban-phase-repo-")
        self.addCleanup(scratch.cleanup)
        repo = Path(scratch.name)
        MODULE._git(repo, "init", "-q", "-b", "main")
        MODULE._git(repo, "config", "user.email", "phase@example.invalid")
        MODULE._git(repo, "config", "user.name", "Phase Conformance")
        (repo / "README.md").write_text("base\n", encoding="utf-8")
        MODULE._git(repo, "add", "README.md")
        MODULE._git(repo, "commit", "-qm", "base")
        base = MODULE._git(repo, "rev-parse", "HEAD")
        MODULE._git(repo, "checkout", "-q", "-b", "task/phase-aware")
        return repo, base

    def _commit(self, repo: Path, n: int) -> str:
        (repo / "phase.py").write_text(f"value = {n}\n", encoding="utf-8")
        MODULE._git(repo, "add", "phase.py")
        MODULE._git(repo, "commit", "-qm", f"implementation {n}")
        return MODULE._git(repo, "rev-parse", "HEAD")

    def _same_card_graph(self, repo: Path, *, goal_mode: bool = True,
                         validation_required: bool = True):
        """Implementation card + tester validation child gated on review_approved."""
        with kbc.connect_closing() as conn:
            implementation = kb.create_task(
                conn,
                title="Phase-aware handoff",
                body=self.GOAL,
                assignee="implementer",
                workspace_kind="dir",
                workspace_path=str(repo),
                goal_mode=goal_mode,
                lifecycle_contract={
                    "kind": "code",
                    "review_mode": "same_card",
                    "reviewer": "reviewer",
                    "validation_required": validation_required,
                },
            )
            validation = None
            if validation_required:
                validation = kb.create_task(
                    conn,
                    title="Phase-aware validation",
                    assignee="tester",
                    lifecycle_contract={"kind": "validation", "candidate_task_id": implementation},
                )
                kb.link_tasks(conn, implementation, validation, requirement="review_approved")
            run = kb.claim_task(conn, implementation, claimer="implementer:1")
            self.assertIsNotNone(run)
        return implementation, validation, run

    def _implementation_metadata(self, base: str, head: str) -> dict:
        return {"base_sha": base, "head_sha": head, "changed_files": ["phase.py"]}

    def _handoff(self, surface: str, tid: str, run, base: str, head: str, *, verdict=None,
                 metadata: dict | None = None, summary: str | None = None):
        """Drive one implementation handoff through a tool or CLI surface."""
        summary = summary or (
            "Implemented the phase-aware handoff and ran the focused suite: 12 passed. "
            "Independent review and tester validation are still pending."
        )
        metadata = self._implementation_metadata(base, head) if metadata is None else metadata
        env = {
            "HERMES_KANBAN_TASK": tid,
            "HERMES_KANBAN_RUN_ID": str(run.current_run_id),
            "HERMES_PROFILE": "implementer",
        }
        with patch.dict(os.environ, env):
            if surface == "tool_request_review":
                from tools import kanban_tools as tools

                return json.loads(tools._handle_request_review(
                    {"summary": summary, "metadata": metadata}))
            if surface == "tool_complete":
                from tools import kanban_tools as tools

                return json.loads(tools._handle_complete(
                    {"summary": summary, "metadata": metadata, "verdict": verdict}))
            if surface == "cli_request_review":
                return kc.run_slash(
                    f"request-review {tid} --summary {summary!r} "
                    f"--metadata {json.dumps(metadata)!r}")
            if surface == "cli_complete":
                return kc.run_slash(
                    f"complete {tid} --summary {summary!r} "
                    f"--metadata {json.dumps(metadata)!r}")
        raise AssertionError(f"unknown surface {surface!r}")

    def _state(self, tid: str) -> dict:
        with kbc.connect_closing() as conn:
            task = kb.get_task(conn, tid)
            state = get_lifecycle_state(conn, tid)
            run = kb.latest_run(conn, tid)
            events = kb.list_events(conn, tid)
        return {
            "task": task,
            "lifecycle": state,
            "run": run,
            "events": events,
            "phase": None if task is None else self._phase(tid),
        }

    def _phase(self, tid: str):
        with kbc.connect_closing() as conn:
            return kb.handoff_phase(conn, kb.get_task(conn, tid))

    def _judge(self, *pairs: tuple[Optional[str], str]):
        """Patch the auxiliary model reply, keeping the real prompt builder."""
        judge = _PhaseJudge(*pairs)
        return judge, patch.multiple(
            "hermes_cli.goals",
            _call_goal_judge_llm=judge,
        )

    def _claims(self, tid: str) -> bool:
        with kbc.connect_closing() as conn:
            return kb.claim_task(conn, tid, claimer="tester:conformance") is not None

    # --- surface 1: phase-scoped judgement --------------------------------
    def test_handoff_is_judged_in_its_own_lifecycle_phase(self) -> None:
        # Invariant: a handoff is judged against the phase it belongs to, so a
        # locally verified implementation candidate reaches review (never final
        # acceptance) while a handoff judged without evidence, or from a review
        # run, is rejected with the card untouched.
        for surface in ("tool_request_review", "tool_complete",
                        "cli_request_review", "cli_complete"):
            with self.subTest(surface=surface):
                repo, base = self._repo()
                tid, validation, run = self._same_card_graph(repo)
                head = self._commit(repo, 1)
                judge, patcher = self._judge(("implementation", "done"))
                with patcher, patch("agent.auxiliary_client.get_text_auxiliary_client",
                                    lambda *a, **k: (object(), "judge-double")):
                    out = self._handoff(surface, tid, run, base, head)

                expected = ("Requested review" if "review" in surface else "Completed")
                if isinstance(out, dict):
                    self.assertTrue(out.get("ok"), out)
                else:
                    self.assertIn(expected, out)

                state = self._state(tid)
                self.assertEqual(state["task"].status, "review")
                self.assertEqual(state["task"].assignee, "reviewer")
                self.assertEqual(state["task"].candidate_run_id, run.current_run_id)
                self.assertEqual(state["lifecycle"]["acceptance"], "pending")
                self.assertEqual(state["lifecycle"]["head_sha"], head)
                asked = [event for event in state["events"] if event.kind == "review_requested"]
                self.assertEqual(asked[-1].payload["implementer"], "implementer")
                # Downstream validation stays gated until the review approves.
                with kbc.connect_closing() as conn:
                    self.assertFalse(kb.evaluate_dependencies(conn, validation)["satisfied"])
                self.assertFalse(self._claims(validation))

                # Judged in the implementation phase, with the card's whole-goal
                # text kept as context separate from the phase's own bar, and the
                # machine-readable handoff (revisions, changed files) included.
                self.assertEqual([call[0] for call in judge.calls], ["implementation"])
                prompt = judge.calls[0][2]
                self.assertIn("independently reviewed by the reviewer profile", prompt)
                self.assertIn(f"- head_sha: {head}", prompt)
                self.assertIn("- changed_files:", prompt)
                self.assertLess(
                    prompt.index("Goal (the card's whole objective"),
                    prompt.index("Definition of done for THIS phase:"),
                )

        for surface in ("tool_request_review", "cli_complete"):
            with self.subTest(surface=f"{surface}_refused"):
                repo, base = self._repo()
                tid, _, run = self._same_card_graph(repo)
                head = self._commit(repo, 1)
                judge, patcher = self._judge(("implementation", "continue"))
                fat_metadata = {
                    **self._implementation_metadata(base, head),
                    "changed_files": [f"src/module_{i}.py" for i in range(90)],
                    **{f"check_{i}": "passed" for i in range(30)},
                }
                with patcher, patch("agent.auxiliary_client.get_text_auxiliary_client",
                                    lambda *a, **k: (object(), "judge-double")):
                    # Long prose as well: the prose is clipped around the
                    # evidence block, never the other way round.
                    out = self._handoff(surface, tid, run, base, head,
                                        metadata=fat_metadata, summary="detail " * 700)

                text = json.dumps(out) if isinstance(out, dict) else out
                self.assertIn("rejected by judge", text)
                self.assertIn("implementation phase", text)
                state = self._state(tid)
                self.assertEqual(state["task"].status, "running")
                self.assertIsNone(state["task"].candidate_run_id)
                self.assertNotIn("review_requested", [e.kind for e in state["events"]])
                # Neither a fat value nor long prose can push the revision or
                # the executed checks out of the judged evidence.
                prompt = judge.calls[0][2]
                self.assertIn(f"- head_sha: {head}", prompt)
                self.assertIn("- changed_files:", prompt)
                self.assertIn("[truncated]", prompt)

        with self.subTest(surface="manual_promotion"):
            # A blocked review resumed by an operator promotion is implementation
            # work again: the implementer's handoff must not be read as a review.
            repo, base = self._repo()
            tid, _, run = self._same_card_graph(repo, validation_required=False)
            head = self._commit(repo, 1)
            judge, patcher = self._judge(("implementation", "done"),
                                         ("implementation", "done"))
            with patcher, patch("agent.auxiliary_client.get_text_auxiliary_client",
                                lambda *a, **k: (object(), "judge-double")):
                self._handoff("tool_request_review", tid, run, base, head)
                with kbc.connect_closing() as conn:
                    review_run = kb.claim_review_task(conn, tid, claimer="reviewer:1")
                    assert review_run is not None
                    self.assertTrue(kb.block_task(
                        conn, tid, reason="maintainer input required",
                        kind="needs_input", expected_run_id=review_run.current_run_id,
                    ))
                    self.assertEqual(kb._resume_status_from_events(conn, tid), "review")
                    self.assertEqual(
                        kb.promote_task(conn, tid, actor="operator",
                                        reason="manual promotion", force=True),
                        (True, None),
                    )
                    self.assertEqual(kb.get_task(conn, tid).status, "ready")
                    resumed = kb.claim_task(conn, tid, claimer="implementer:2")
                assert resumed is not None
                self.assertEqual(self._phase(tid), "implementation")
                with patch.dict(os.environ, {
                    "HERMES_KANBAN_TASK": tid,
                    "HERMES_KANBAN_RUN_ID": str(resumed.current_run_id),
                    "HERMES_PROFILE": "implementer",
                }):
                    out = self._handoff("tool_request_review", tid, resumed, base, head)

            self.assertTrue(out.get("ok"), out)
            self.assertEqual(self._state(tid)["task"].status, "review")
            self.assertEqual([call[0] for call in judge.calls],
                             ["implementation", "implementation"])

        with self.subTest(surface="secret_metadata"):
            # The auxiliary judge is a separate provider, so the evidence is
            # redacted at that boundary even when structured redaction upstream
            # had to keep the raw dict.
            repo, base = self._repo()
            tid, _, run = self._same_card_graph(repo)
            head = self._commit(repo, 1)
            secret = "ghp_" + "S" * 36
            judge, patcher = self._judge(("implementation", "continue"))
            with patcher, patch("agent.auxiliary_client.get_text_auxiliary_client",
                                lambda *a, **k: (object(), "judge-double")):
                # kanban_complete keeps the raw dict when structured redaction
                # cannot re-parse; request_review refuses that metadata outright.
                out = self._handoff(
                    "tool_complete", tid, run, base, head,
                    metadata={**self._implementation_metadata(base, head),
                              "secret": f'abc\"{secret}'},
                )
            self.assertIn("rejected by judge", json.dumps(out))
            self.assertNotIn(secret, judge.calls[0][2])

        with self.subTest(surface="review_run"):
            repo, base = self._repo()
            tid, _, run = self._same_card_graph(repo)
            head = self._commit(repo, 1)
            judge, patcher = self._judge(("implementation", "done"),
                                         ("review", "done"), ("review", "continue"))
            with patcher, patch("agent.auxiliary_client.get_text_auxiliary_client",
                                lambda *a, **k: (object(), "judge-double")):
                self._handoff("tool_request_review", tid, run, base, head)
                with kbc.connect_closing() as conn:
                    review_run = kb.claim_review_task(conn, tid, claimer="reviewer:1")
                self.assertIsNotNone(review_run)
                assert review_run is not None
                from tools import kanban_tools as tools

                with patch.dict(os.environ, {
                    "HERMES_KANBAN_TASK": tid,
                    "HERMES_KANBAN_RUN_ID": str(review_run.current_run_id),
                    "HERMES_PROFILE": "reviewer",
                }):
                    handoff = json.loads(tools._handle_request_review({
                        "summary": "Handing the candidate back as implementation.",
                        "metadata": self._implementation_metadata(base, head),
                    }))
                    approval = json.loads(tools._handle_complete({
                        "summary": "APPROVE",
                        "verdict": "APPROVE",
                    }))

            self.assertIn(self.GUARD_REASON, handoff["error"])
            self.assertIn("rejected by judge", approval["error"])
            state = self._state(tid)
            self.assertEqual(state["task"].status, "running")
            self.assertEqual(state["task"].assignee, "reviewer")
            self.assertEqual(state["task"].candidate_run_id, run.current_run_id)
            self.assertIsNone(state["lifecycle"]["review_verdict"])
            # No implementation evidence stamped on the review run. The run does
            # carry additive provenance (its host instantiation epoch), so assert
            # on the evidence keys rather than on the whole metadata column.
            self.assertEqual(
                set(state["run"].metadata or {})
                & set(self._implementation_metadata(base, head)),
                set(),
            )
            self.assertEqual([e.kind for e in state["events"]].count("review_requested"), 1)
            self.assertEqual([call[0] for call in judge.calls],
                             ["implementation", "review", "review"])
            # The review rubric needs the revision under review, so the reviewer's
            # metadata reaches the judge too.
            self.assertIn(f"- head_sha: {head}", judge.calls[1][2])

        # The goal loop judges its own phase only, and a caller without a phase
        # keeps whole-goal judging and the effective-revision refresh.
        from hermes_cli import goals

        loop_judge = _PhaseJudge(("implementation", "done"))
        statuses = iter(["running", "review"])
        prompts: list[str] = []
        with patch.multiple("hermes_cli.goals", _call_goal_judge_llm=loop_judge):
            looped = goals.run_kanban_goal_loop(
                task_id="t_phase",
                goal_text=self.GOAL,
                phase="implementation",
                run_turn=lambda prompt: prompts.append(prompt) or "handed off",
                task_status_fn=lambda: next(statuses),
                block_fn=lambda reason: self.fail(f"must not block: {reason}"),
                first_response="Implemented the candidate; 12 passed locally.",
            )
        self.assertEqual(looped["outcome"], "review_requested_by_worker")
        self.assertEqual(len(loop_judge.calls), 1)  # the next phase is never judged
        self.assertIn("implementation phase", prompts[0])
        self.assertIn("kanban_request_review", prompts[0])
        self.assertIn("not final acceptance", prompts[0])

        # A finalize nudge in the review phase must not send the reviewer to the
        # implementation handoff, which the runtime now rejects for review runs.
        review_judge = _PhaseJudge(("review", "done"))
        review_statuses = iter(["running", "changes_requested"])
        review_prompts: list[str] = []
        with patch.multiple("hermes_cli.goals", _call_goal_judge_llm=review_judge):
            reviewed = goals.run_kanban_goal_loop(
                task_id="t_review",
                goal_text=self.GOAL,
                phase="review",
                run_turn=lambda prompt: review_prompts.append(prompt) or "verdict submitted",
                task_status_fn=lambda: next(review_statuses),
                block_fn=lambda reason: self.fail(f"must not block: {reason}"),
                first_response="Reviewed the candidate at its head; findings listed.",
            )
        self.assertEqual(reviewed["outcome"], "changes_requested_by_reviewer")
        self.assertNotIn("kanban_request_review", review_prompts[0])
        self.assertIn("REQUEST_CHANGES", review_prompts[0])

        with self.subTest(surface="phase_less_loop"):
            generic_judge = _PhaseJudge((None, "continue"))
            generic_prompts: list[str] = []
            with patch.multiple("hermes_cli.goals", _call_goal_judge_llm=generic_judge):
                stopped = goals.run_kanban_goal_loop(
                    task_id="t_generic",
                    goal_text="original goal",
                    goal_text_fn=lambda: "revised goal",
                    run_turn=lambda prompt: generic_prompts.append(prompt) or "still working",
                    task_status_fn=lambda: "running",
                    block_fn=lambda reason: None,
                    max_turns=2,
                    first_response="first response",
                )
            self.assertEqual(stopped["outcome"], "blocked_budget")
            self.assertEqual(generic_judge.calls[0][0], "generic")
            self.assertIn("revised goal", generic_judge.calls[0][2])
            self.assertIn("not done yet", generic_prompts[0])
            self.assertNotIn("implementation phase", generic_prompts[0])

    # --- surface 2: typed rework and validation ---------------------------
    def _review_ready(self, route: str):
        """Implementation handed off and claimed for review on a fresh graph."""
        repo, base = self._repo()
        tid, validation, run = self._same_card_graph(repo)
        head = self._commit(repo, 1)
        judge, patcher = self._judge(("implementation", "done"), ("review", "done"))
        with patcher, patch("agent.auxiliary_client.get_text_auxiliary_client",
                            lambda *a, **k: (object(), "judge-double")):
            self._handoff("tool_request_review", tid, run, base, head)
            with kbc.connect_closing() as conn:
                review_run = kb.claim_review_task(conn, tid, claimer="reviewer:1")
            self.assertIsNotNone(review_run)
            assert review_run is not None
            from tools import kanban_tools as tools

            with patch.dict(os.environ, {
                "HERMES_KANBAN_TASK": tid,
                "HERMES_KANBAN_RUN_ID": str(review_run.current_run_id),
                "HERMES_PROFILE": "reviewer",
            }):
                if route == "request_changes":
                    out = json.loads(tools._handle_request_changes({
                        "reason": "Fix the boundary assertion in phase.py",
                        "metadata": {"reviewed_head_sha": head},
                    }))
                else:
                    out = json.loads(tools._handle_complete({
                        "summary": "Reviewed phase.py at this head: the boundary "
                                   "assertion is wrong. Requires rework.",
                        "verdict": "REQUEST_CHANGES",
                    }))
        return tid, validation, run.current_run_id, base, head, out

    def test_typed_rework_and_validation_decide_acceptance(self) -> None:
        # Invariant: a review rejection is rework returned to the implementer —
        # never a repair by the reviewer and never acceptance — and only a fresh
        # tester verdict at the approved head decides the card.
        for route in ("request_changes", "complete_request_changes"):
            with self.subTest(route=route):
                tid, validation, candidate_run_id, _base, head, out = self._review_ready(route)
                self.assertTrue(out.get("ok"), out)
                state = self._state(tid)
                self.assertIn(state["task"].status, {"ready", "todo"})
                self.assertEqual(state["task"].assignee, "implementer")
                self.assertIsNone(state["task"].candidate_run_id)
                self.assertEqual(state["lifecycle"]["review_verdict"], "REQUEST_CHANGES")
                self.assertIsNone(state["lifecycle"]["head_sha"])
                self.assertNotEqual(state["lifecycle"]["acceptance"], "accepted")
                self.assertEqual(state["run"].outcome, "changes_requested")
                # The negative verdict survives against the reviewed candidate/head.
                elapsed = state["run"].metadata["lifecycle"]
                self.assertEqual(elapsed["verdict"], "REQUEST_CHANGES")
                self.assertEqual(elapsed["head_sha"], head)
                self.assertEqual(elapsed["candidate_run_id"], candidate_run_id)
                with kbc.connect_closing() as conn:
                    self.assertFalse(kb.evaluate_dependencies(conn, validation)["satisfied"])
                self.assertFalse(self._claims(validation))

        for verdict, expected in (("PASS", "accepted"), ("FAIL", "rejected")):
            with self.subTest(validation=verdict):
                repo, base = self._repo()
                tid, validation, run = self._same_card_graph(repo)
                head1 = self._commit(repo, 1)
                judge, patcher = self._judge(("implementation", "done"),
                                             ("implementation", "done"), ("review", "done"))
                with patcher, patch("agent.auxiliary_client.get_text_auxiliary_client",
                                    lambda *a, **k: (object(), "judge-double")):
                    from tools import kanban_tools as tools

                    self._handoff("tool_request_review", tid, run, base, head1)
                    with kbc.connect_closing() as conn:
                        first_review = kb.claim_review_task(conn, tid, claimer="reviewer:1")
                    assert first_review is not None
                    with patch.dict(os.environ, {
                        "HERMES_KANBAN_TASK": tid,
                        "HERMES_KANBAN_RUN_ID": str(first_review.current_run_id),
                        "HERMES_PROFILE": "reviewer",
                    }):
                        rejected = json.loads(tools._handle_request_changes({
                            "reason": "Boundary assertion is wrong; fix it.",
                            "metadata": {"reviewed_head_sha": head1},
                        }))
                    self.assertTrue(rejected["ok"], rejected)

                    # The rejected candidate's verdict cannot release the tester.
                    self.assertEqual(self._phase(validation), "validation")
                    self.assertFalse(self._claims(validation))

                    head2 = self._commit(repo, 2)
                    self.assertNotEqual(head1, head2)
                    with kbc.connect_closing() as conn:
                        repair = kb.claim_task(conn, tid, claimer="implementer:2")
                    assert repair is not None
                    with patch.dict(os.environ, {
                        "HERMES_KANBAN_TASK": tid,
                        "HERMES_KANBAN_RUN_ID": str(repair.current_run_id),
                        "HERMES_PROFILE": "implementer",
                    }):
                        repaired = json.loads(tools._handle_complete({
                            "summary": "Fixed the boundary assertion; 13 passed.",
                            "metadata": self._implementation_metadata(base, head2),
                        }))
                    self.assertTrue(repaired["ok"], repaired)
                    repaired_state = self._state(tid)
                    self.assertEqual(repaired_state["task"].status, "review")
                    self.assertEqual(repaired_state["task"].candidate_run_id,
                                     repair.current_run_id)
                    self.assertEqual(repaired_state["lifecycle"]["head_sha"], head2)
                    self.assertNotEqual(repaired_state["lifecycle"]["acceptance"], "accepted")

                    with kbc.connect_closing() as conn:
                        second_review = kb.claim_review_task(conn, tid, claimer="reviewer:2")
                    assert second_review is not None
                    with patch.dict(os.environ, {
                        "HERMES_KANBAN_TASK": tid,
                        "HERMES_KANBAN_RUN_ID": str(second_review.current_run_id),
                        "HERMES_PROFILE": "reviewer",
                    }):
                        approved = json.loads(tools._handle_complete({
                            "summary": "Re-reviewed phase.py at the repaired head: "
                                       "13 passed, findings addressed.",
                            "verdict": "APPROVE",
                        }))
                    self.assertTrue(approved["ok"], approved)

                    # Approval releases the gated tester without accepting the card.
                    approved_state = self._state(tid)
                    self.assertEqual(approved_state["lifecycle"]["review_verdict"], "APPROVE")
                    self.assertEqual(approved_state["lifecycle"]["acceptance"], "pending")
                    self.assertEqual(approved_state["lifecycle"]["head_sha"], head2)
                    with kbc.connect_closing() as conn:
                        self.assertEqual(kb.get_task(conn, validation).status, "ready")
                        tester_run = kb.claim_task(conn, validation, claimer="tester:1")
                    assert tester_run is not None
                    with patch.dict(os.environ, {
                        "HERMES_KANBAN_TASK": validation,
                        "HERMES_KANBAN_RUN_ID": str(tester_run.current_run_id),
                        "HERMES_PROFILE": "tester",
                    }):
                        executed = ("13 passed" if verdict == "PASS" else "boundary case failed")
                        validated = json.loads(tools._handle_complete({
                            "summary": f"Executed the acceptance suite at the approved "
                                       f"head: {executed}.",
                            "verdict": verdict,
                        }))
                self.assertTrue(validated["ok"], validated)
                final = self._state(tid)
                self.assertEqual(final["lifecycle"]["validation_verdict"], verdict)
                self.assertEqual(final["lifecycle"]["acceptance"], expected)
                # An approval that carries no revision in its own metadata is
                # still judged against the persisted candidate head.
                self.assertIn(f"- reviewed_head_sha: {head2}", judge.calls[2][2])

        with self.subTest(route="approval_retry"):
            # A retried approval (validation not required) is judged against the
            # pinned candidate revision even though the finished review run is now
            # the newest handoff on the card.
            repo, base = self._repo()
            tid, _, run = self._same_card_graph(repo, validation_required=False)
            head = self._commit(repo, 1)
            judge, patcher = self._judge(("implementation", "done"),
                                         ("review", "done"), ("review", "done"))
            with patcher, patch("agent.auxiliary_client.get_text_auxiliary_client",
                                lambda *a, **k: (object(), "judge-double")):
                from tools import kanban_tools as tools

                self._handoff("tool_request_review", tid, run, base, head)
                with kbc.connect_closing() as conn:
                    review_run = kb.claim_review_task(conn, tid, claimer="reviewer:1")
                assert review_run is not None
                with patch.dict(os.environ, {
                    "HERMES_KANBAN_TASK": tid,
                    "HERMES_KANBAN_RUN_ID": str(review_run.current_run_id),
                    "HERMES_PROFILE": "reviewer",
                }):
                    approved = json.loads(tools._handle_complete({
                        "summary": "Re-reviewed the candidate; checks passed.",
                        "verdict": "APPROVE",
                    }))
                    self.assertTrue(approved["ok"], approved)
                    self.assertEqual(self._state(tid)["task"].status, "done")
                    # Same approval again, as a client that lost the first response.
                    retried = json.loads(tools._handle_complete({
                        "summary": "Re-reviewed the candidate; checks passed.",
                        "verdict": "APPROVE",
                    }))

            self.assertTrue(retried.get("ok"), retried)
            self.assertEqual(self._state(tid)["task"].status, "done")
            self.assertEqual([call[0] for call in judge.calls],
                             ["implementation", "review", "review"])
            self.assertIn(f"- reviewed_head_sha: {head}", judge.calls[2][2])

        with self.subTest(route="separate_review_card"):
            repo, base = self._repo()
            with kbc.connect_closing() as conn:
                implementation = kb.create_task(
                    conn,
                    title="Separate-card implementation",
                    body=self.GOAL,
                    assignee="implementer",
                    workspace_kind="dir",
                    workspace_path=str(repo),
                    goal_mode=True,
                    lifecycle_contract={
                        "kind": "code",
                        "review_mode": "separate_card",
                        "reviewer": "reviewer",
                        "validation_required": True,
                    },
                )
                review = kb.create_task(
                    conn,
                    title="Separate review",
                    assignee="reviewer",
                    goal_mode=True,
                    lifecycle_contract={"kind": "review", "candidate_task_id": implementation},
                )
                kb.link_tasks(conn, implementation, review)
                run = kb.claim_task(conn, implementation, claimer="implementer:1")
                assert run is not None
            head = self._commit(repo, 1)
            judge, patcher = self._judge(("implementation", "done"), ("review", "done"))
            with patcher, patch("agent.auxiliary_client.get_text_auxiliary_client",
                                lambda *a, **k: (object(), "judge-double")):
                from tools import kanban_tools as tools

                with patch.dict(os.environ, {
                    "HERMES_KANBAN_TASK": implementation,
                    "HERMES_KANBAN_RUN_ID": str(run.current_run_id),
                    "HERMES_PROFILE": "implementer",
                }):
                    handed_off = json.loads(tools._handle_complete({
                        "summary": "Implemented and verified locally: 12 passed.",
                        "metadata": self._implementation_metadata(base, head),
                    }))
                self.assertTrue(handed_off["ok"], handed_off)
                with kbc.connect_closing() as conn:
                    # The completed candidate promotes its separate review child.
                    self.assertEqual(kb.get_task(conn, review).status, "review")
                    self.assertEqual(self._phase(review), "review")
                    review_run = kb.claim_review_task(conn, review, claimer="reviewer:1")
                assert review_run is not None
                with patch.dict(os.environ, {
                    "HERMES_KANBAN_TASK": review,
                    "HERMES_KANBAN_RUN_ID": str(review_run.current_run_id),
                    "HERMES_PROFILE": "reviewer",
                }):
                    rejected = json.loads(tools._handle_complete({
                        "summary": "Inspected the diff: the retry path drops the error. "
                                   "Requires rework.",
                        "verdict": "REQUEST_CHANGES",
                    }))

            self.assertTrue(rejected["ok"], rejected)
            review_state = self._state(review)
            self.assertEqual(review_state["task"].status, "done")
            self.assertEqual(review_state["lifecycle"]["review_verdict"], "REQUEST_CHANGES")
            self.assertNotEqual(
                self._state(implementation)["lifecycle"]["acceptance"], "accepted")
            self.assertEqual([call[0] for call in judge.calls], ["implementation", "review"])

    # --- surface 3: typed code outranks plan-only prose --------------------
    PLAN_ONLY_PHRASING_GOALS = (
        # issue #41: the exclusion applies to sibling work (#486), not to this card.
        "Implement the bounded repair and regression tests. "
        "Review/comment scope only, no implementation of #486.",
        # The goal names the classifier defect it is NOT making.
        "IMPLEMENTATION task: implement the repair and regression tests. "
        "If this goal still produces plan-only rejection, report the tooling defect.",
    )

    def test_code_typed_handoff_ignores_plan_only_phrasing(self) -> None:
        # Invariant: the decoded kind=code contract decides that this handoff is
        # an implementation handoff. Goal prose that scopes out other work or
        # quotes the classifier defect must not turn a real patch into a
        # plan-only violation, and the card, its goal revision, its graph and
        # the reviewer/tester gates must survive the handoff unchanged.
        for index, goal in enumerate(self.PLAN_ONLY_PHRASING_GOALS):
            for surface in ("tool_request_review", "tool_complete"):
                with self.subTest(goal=index, surface=surface):
                    repo, base = self._repo()
                    self.GOAL = goal
                    tid, validation, run = self._same_card_graph(
                        repo, goal_mode=False, validation_required=True)
                    head = self._commit(repo, 1)
                    with kbc.connect_closing() as conn:
                        goal_before = kb.get_effective_goal(conn, tid)
                        task_before = kb.get_task(conn, tid)
                        validation_before = kb.get_task(conn, validation)
                    metadata = {"base_sha": base, "head_sha": head}
                    if index == 0:
                        # One case relies on the verified Git diff for
                        # changed_files, so both classifier entries are covered.
                        metadata["changed_files"] = ["phase.py"]
                    out = self._handoff(surface, tid, run, base, head, metadata=metadata)
                    self.assertTrue(isinstance(out, dict) and out.get("ok"), out)

                    state = self._state(tid)
                    self.assertEqual(state["task"].id, tid)
                    self.assertEqual(state["task"].status, "review")
                    self.assertEqual(state["task"].assignee, "reviewer")
                    self.assertEqual(state["task"].candidate_run_id, run.current_run_id)
                    self.assertEqual(
                        state["task"].goal_revision_id, task_before.goal_revision_id)
                    self.assertEqual(state["lifecycle"]["acceptance"], "pending")
                    self.assertIn("review_requested", [e.kind for e in state["events"]])
                    with kbc.connect_closing() as conn:
                        handoff = kb.latest_handoff(conn, tid)
                        goal_after = kb.get_effective_goal(conn, tid)
                        validation_after = kb.get_task(conn, validation)
                    self.assertEqual(
                        (goal_after["id"], goal_after["version"]),
                        (goal_before["id"], goal_before["version"]),
                    )
                    self.assertEqual(handoff["head_sha"], head)
                    self.assertIn("phase.py", handoff["changed_files"])
                    self.assertEqual(validation_after, validation_before)
                    self.assertFalse(self._claims(validation))


if __name__ == "__main__":
    unittest.main()

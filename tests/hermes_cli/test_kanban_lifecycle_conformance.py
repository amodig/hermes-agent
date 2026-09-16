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
from hermes_cli import kanban_db as kb
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


def _make_runtime_fixture(root: Path) -> None:
    """Small real install: production bootstrap, dynamic imports, and resources."""
    package = root / "hermes_cli"
    package.mkdir(parents=True)
    for name in ("kanban_runtime.py", "kanban_runtime_generation.py"):
        shutil.copy2(RUNTIME_ROOT / "hermes_cli" / name, package / name)
    (package / "__init__.py").write_text('__version__ = "fixture"\n', encoding="utf-8")
    (root / "fixture_early.py").write_text("value = 1\n", encoding="utf-8")
    (root / "fixture_lazy.py").write_text("value = 1\n", encoding="utf-8")
    dependency_root = root.parent / "site-packages"
    dependency_root.mkdir(exist_ok=True)
    for name in ("third_party_early", "third_party_dynamic"):
        dependency = dependency_root / name
        dependency.mkdir()
        (dependency / "__init__.py").write_text("value = 1\n", encoding="utf-8")
        (dependency / "data.bin").write_bytes(b"dependency-v1")
    for directory, _env_var in runtime._RUNTIME_RESOURCE_ROOTS:
        resource = root / directory
        resource.mkdir(exist_ok=True)
        (resource / "marker.txt").write_text(f"{directory}:v1\n", encoding="utf-8")
    executable = root / "skills" / "helper"
    executable.write_text("#!/bin/sh\nprintf sealed-helper\n", encoding="utf-8")
    executable.chmod(0o755)
    (package / "main.py").write_text(
        """
import importlib
import json
import os
import sys
import time
from pathlib import Path
from hermes_cli import kanban_runtime as runtime
import fixture_early
import third_party_early

runtime.worker_bootstrap_from_env()
runtime.worker_bootstrap_post_import(wait_for_grant=False)
assert os.environ.get("HERMES_KANBAN_RUNTIME_GRANTED") != "1"
runtime.worker_bootstrap_after_constructor()
release = os.environ.get("HERMES_TEST_RUNTIME_RELEASE")
while release and not Path(release).exists():
    time.sleep(0.02)
lazy = importlib.import_module("fixture_" + "lazy")
sdk = importlib.import_module(os.environ.get("HERMES_TEST_SDK", "third_party_dynamic"))
payload = {
    "identity": runtime.runtime_identity().as_dict(),
    "early": [fixture_early.value, third_party_early.value],
    "lazy": [lazy.value, sdk.value],
    "dependency_data": (Path(sdk.__file__).parent / "data.bin").read_text(),
    "locations": [fixture_early.__file__, third_party_early.__file__, lazy.__file__, sdk.__file__],
    "resources": {
        directory: (Path(os.environ[env_var]) / "marker.txt").read_text()
        for directory, env_var in runtime._RUNTIME_RESOURCE_ROOTS
    },
    "argv": sys.argv[1:],
    "cwd": os.getcwd(),
    "profile": os.environ.get("HERMES_PROFILE"),
    "home": os.environ.get("HERMES_HOME"),
    "task": os.environ.get("HERMES_KANBAN_TASK"),
    "board": os.environ.get("HERMES_KANBAN_BOARD"),
    "run": os.environ.get("HERMES_KANBAN_RUN_ID"),
    "claim": os.environ.get("HERMES_KANBAN_CLAIM_LOCK"),
    "granted": os.environ.get("HERMES_KANBAN_RUNTIME_GRANTED"),
    "secret": os.environ.get("ANTHROPIC_API_KEY"),
}
if Path("/proc/self/cgroup").exists():
    payload["cgroup"] = Path("/proc/self/cgroup").read_text()
Path(os.environ["HERMES_TEST_RUNTIME_RECEIPT"]).write_text(json.dumps(payload))
""",
        encoding="utf-8",
    )

def _prepare_fixture_generation(
    root: Path, prepare=generations.prepare_runtime_generation, *, workspace=None, profile_home=None,
    project_plugins_enabled: bool | None = None,
):
    resources = {env_var: str(root / directory) for directory, env_var in runtime._RUNTIME_RESOURCE_ROOTS}
    with patch.dict(os.environ, resources), patch.object(
        generations, "_runtime_import_roots", return_value=[root.parent / "site-packages"],
    ):
        return prepare(
            runtime_identity(root), workspace=workspace, profile_home=profile_home,
            project_plugins_enabled=project_plugins_enabled,
        )


def _wait_for_receipt(path: Path) -> dict:
    deadline = time.monotonic() + 10
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    return json.loads(path.read_text(encoding="utf-8"))


class KanbanLifecycleConformance(unittest.TestCase):
    def setUp(self) -> None:
        self.home = tempfile.TemporaryDirectory(prefix="kanban-conformance-home-")
        self.storage = patch.object(
            generations, "_runtime_storage_root",
            return_value=Path(self.home.name) / "runtime-storage",
        )
        self.storage.start()
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
        self.storage.stop()
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
            trace.append("validation:running->done")
            acceptance = get_lifecycle_state(self.conn, implementation)
            self.assertEqual(acceptance["acceptance"], "accepted")
            self.assertEqual(acceptance["review_verdict"], "APPROVE")
            self.assertEqual(acceptance["validation_verdict"], "PASS")
            trace.append("implementation:acceptance=pending->accepted")
            self.assertEqual(trace[-1], fixture["expected_trace"][-1])

    def test_same_card_review_waits_for_validation_before_terminal_completion(self) -> None:
        with tempfile.TemporaryDirectory(prefix="kanban-conformance-validation-") as raw_repo:
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
                initial_status="blocked",
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

        def terminate_after_commit(pid, claim_lock):
            self.assertFalse(self.conn.in_transaction)
            self.assertEqual((pid, claim_lock), (424242, claimed.claim_lock))
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
        self.assertTrue(kb.complete_task(self.conn, satisfied))
        for status in ("done", "review", "running"):
            with self.subTest(status=status):
                claimed = None
                if status == "done":
                    child, _ = self._completed_candidate()
                else:
                    child = kb.create_task(
                        self.conn, title=f"{status} child", assignee="worker",
                    )
                    if status == "review":
                        self.assertTrue(kb.request_review(self.conn, child, reviewer="reviewer"))
                    else:
                        claimed = kb.claim_task(self.conn, child)
                        self.assertIsNotNone(claimed)
                        kbd._set_worker_pid(self.conn, child, 424242)
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
                if claimed is not None:
                    with self.assertRaises(kb.TaskUpdateConflict):
                        kb.link_tasks(
                            self.conn, other_gate, child,
                            expected_parent_version=self._task(other_gate).version,
                            expected_child_version=before.version, reason="cannot rebind a claim",
                        )
                    self.assertEqual(self._task(child), before)
                    self.assertIsNone(self.conn.execute(
                        "SELECT requirement FROM task_links WHERE parent_id = ? AND child_id = ?",
                        (other_gate, child),
                    ).fetchone()["requirement"])
                with patch.object(kb, "_terminate_reclaimed_worker") as terminate:
                    kb.link_tasks(self.conn, gate, child, requirement="phase_finished")
                    if claimed is None:
                        terminate.assert_not_called()
                    else:
                        terminate.assert_called_once_with(424242, claimed.claim_lock)
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
                if claimed is not None:
                    self.assertEqual(kb.latest_run(self.conn, child).outcome, "reclaimed")

    def test_archived_children_reject_changed_dependencies(self) -> None:
        parent = kb.create_task(self.conn, title="completed dependency")
        self.assertTrue(kb.complete_task(self.conn, parent))
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
        self.assertTrue(kb.complete_task(self.conn, parent))
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
        _git(repo, "init", "-q")
        _git(repo, "config", "user.email", "conformance@example.invalid")
        _git(repo, "config", "user.name", "Kanban Conformance")
        (repo / "README").write_text("legacy\n", encoding="utf-8")
        _git(repo, "add", "README")
        _git(repo, "commit", "-qm", "legacy")
        branch = _git(repo, "branch", "--show-current")
        head = _git(repo, "rev-parse", "HEAD")

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

    def test_default_dispatch_grants_worker_before_claimed_hook(self) -> None:
        task_id = kb.create_task(
            self.conn,
            title="grant before claimed hook",
            assignee="implementer",
            initial_status="blocked",
        )
        self.assertTrue(kb.unblock_task(self.conn, task_id))
        identity = runtime_identity(RUNTIME_ROOT)
        order: list[str] = []

        def fake_default_spawn(task, workspace, *, board=None, defer_grant=False):
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
        _git(repo, "init", "-q")
        _git(repo, "config", "user.email", "conformance@example.invalid")
        _git(repo, "config", "user.name", "Kanban Conformance")
        (repo / "README").write_text("base\n", encoding="utf-8")
        _git(repo, "add", "README")
        _git(repo, "commit", "-qm", "base")
        base_sha = _git(repo, "rev-parse", "HEAD")
        (repo / "implementation.py").write_text("print('proof')\n", encoding="utf-8")
        _git(repo, "add", "implementation.py")
        _git(repo, "commit", "-qm", "implementation")
        head_sha = _git(repo, "rev-parse", "HEAD")
        for phase in ("review", "blocked", "running"):
            with self.subTest(phase=phase):
                parent = kb.create_task(self.conn, title=f"{phase} ancestor")
                self.assertTrue(kb.complete_task(self.conn, parent))
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
                self.assertTrue(kb.complete_task(self.conn, parent))
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
        _git(repo, "add", "lifecycle.py")
        _git(repo, "commit", "-qm", "revised implementation")
        revised_head = _git(repo, "rev-parse", "HEAD")
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
            _make_runtime_fixture(source)
            first = _prepare_fixture_generation(source)
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
                early = _wait_for_receipt(preparation)
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
                    ready = _wait_for_receipt(preparation)
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
                observed = _wait_for_receipt(receipt)
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
                second = _prepare_fixture_generation(source)
                try:
                    self.assertFalse(same_code_identity(first.identity, second.identity))
                    self.assertNotEqual(first.identity.dependency_fingerprint, second.identity.dependency_fingerprint)
                    next_receipt = base / "second.json"
                    next_env = {**os.environ, **second.env, "HERMES_TEST_RUNTIME_RECEIPT": str(next_receipt)}
                    completed = subprocess.run(second.command_prefix, env=next_env, check=False, timeout=10)
                    self.assertEqual(completed.returncode, 0)
                    current = _wait_for_receipt(next_receipt)
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
            _make_runtime_fixture(source)
            prepared = _prepare_fixture_generation(source)
            try:
                helper = Path(prepared.env["HERMES_BUNDLED_SKILLS"]) / "helper"
                completed = subprocess.run([str(helper)], capture_output=True, text=True, check=True)
                self.assertEqual(completed.stdout, "sealed-helper")
            finally:
                generations.cleanup_runtime_generation(prepared.root, force=True)

    def test_generation_sweep_preserves_live_owner_and_rejects_pid_reuse(self) -> None:
        with tempfile.TemporaryDirectory(prefix="kanban-conformance-owner-") as raw:
            source = Path(raw) / "install"
            _make_runtime_fixture(source)
            prepared = _prepare_fixture_generation(source)
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
            _make_runtime_fixture(source)
            for name in ("main.py", "_subprocess_compat.py", "_startup_fast.py"):
                shutil.copy2(RUNTIME_ROOT / "hermes_cli" / name, source / "hermes_cli" / name)
            shutil.copy2(RUNTIME_ROOT / "hermes_bootstrap.py", source / "hermes_bootstrap.py")
            recovery_marker = Path(raw) / "recovery-mutated-install"
            (source / "hermes_cli" / "_early_recovery.py").write_text(
                "from pathlib import Path\n"
                f"def recover_if_needed():\n    Path({str(recovery_marker)!r}).touch()\n",
                encoding="utf-8",
            )
            prepared = _prepare_fixture_generation(source)
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
import sys
from pathlib import Path
from hermes_cli import _early_recovery as recovery
from hermes_cli import _install_repair as repair
root = Path(sys.argv[1])
def install(*args):
    (root / "installer-ran").touch()
    return True
recovery._probe_broken_packages = lambda: ["PyYAML"]
recovery._run_repair_install = install
repair.run_core_install = install
recovery.recover_if_needed(project_root=root, argv=[])
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
                env = {**os.environ, "PYTHONPATH": str(RUNTIME_ROOT), "TMPDIR": raw}
                command = [sys.executable, "-c", script, raw]
                # Other test files share the interpreter installation, not this fixture's locks.
                with patch.object(tempfile, "tempdir", raw), generations.installation_mutation_lock(source):
                    deferred = subprocess.run(
                        command, env=env, capture_output=True, text=True, timeout=10,
                    )
                self.assertEqual(deferred.returncode, 0, deferred.stderr)
                self.assertFalse((source / "installer-ran").exists())
                self.assertTrue((source / marker).exists())
                recovered = subprocess.run(
                    command, env=env, capture_output=True, text=True, timeout=10,
                )
                self.assertEqual(recovered.returncode, 0, recovered.stderr)
                self.assertTrue((source / "installer-ran").exists())

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
            _make_runtime_fixture(source)
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
            _make_runtime_fixture(source)
            receipt = Path(raw) / "grant.json"
            with patch.object(kbd, "_profile_exists_fn", return_value=None), patch.object(
                kbd, "_restart_safe_worker_argv",
                side_effect=lambda task, command, preparation_id=None: command,
            ), patch.object(
                generations, "prepare_runtime_generation",
                side_effect=lambda expected, workspace=None, profile_home=None, project_plugins_enabled=None: _prepare_fixture_generation(
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
                granted = _wait_for_receipt(receipt)
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
        with tempfile.TemporaryDirectory(prefix="kanban-worker-mismatch-") as raw:
            source = Path(raw) / "install"
            _make_runtime_fixture(source)
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
                side_effect=lambda expected, workspace=None, profile_home=None, project_plugins_enabled=None: _prepare_fixture_generation(
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


if __name__ == "__main__":
    unittest.main()

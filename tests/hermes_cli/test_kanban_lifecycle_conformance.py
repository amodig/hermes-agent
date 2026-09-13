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
from hermes_cli.kanban_lifecycle import get_lifecycle_state
from hermes_cli.kanban_parser import build_parser
from hermes_cli.kanban_runtime import (
    RuntimeIdentityError,
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
                "from hermes_cli.kanban_runtime import worker_bootstrap_from_env; "
                "worker_bootstrap_from_env(); "
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

"""Hermetic conformance for Kanban lifecycle upgrade paths."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_lifecycle import evaluate_dependencies, get_lifecycle_state
from tests.hermes_cli import kanban_conformance_fixture as MODULE


class KanbanLifecycleUpgradeConformance(MODULE.KanbanConformanceFixture):
    def _repo_with_head(self, repo: Path) -> tuple[str, str]:
        MODULE._git(repo, "init", "-q")
        MODULE._git(repo, "config", "user.email", "conformance@example.invalid")
        MODULE._git(repo, "config", "user.name", "Kanban Conformance")
        (repo / "README").write_text("base\n", encoding="utf-8")
        MODULE._git(repo, "add", "README")
        MODULE._git(repo, "commit", "-qm", "base")
        base = MODULE._git(repo, "rev-parse", "HEAD")
        (repo / "lifecycle.py").write_text("version 1\n", encoding="utf-8")
        MODULE._git(repo, "add", "lifecycle.py")
        MODULE._git(repo, "commit", "-qm", "implementation H1")
        return base, MODULE._git(repo, "rev-parse", "HEAD")

    def _new_head(self, repo: Path, marker: str) -> str:
        (repo / "lifecycle.py").write_text(f"{marker}\n", encoding="utf-8")
        MODULE._git(repo, "add", "lifecycle.py")
        MODULE._git(repo, "commit", "-qm", marker)
        return MODULE._git(repo, "rev-parse", "HEAD")

    def _typed_graph(self, repo: Path, mode: str) -> tuple[str, str | None, str]:
        implementation = kb.create_task(
            self.conn,
            title=f"{mode} implementation",
            assignee="implementer",
            initial_status="blocked",
            workspace_kind="dir",
            workspace_path=str(repo),
            lifecycle_contract={
                "kind": "code",
                "review_mode": mode,
                "reviewer": "reviewer",
                "validation_required": True,
            },
        )
        reviewer = None
        if mode == "separate_card":
            reviewer = kb.create_task(
                self.conn,
                title="separate reviewer",
                assignee="reviewer",
                initial_status="blocked",
                lifecycle_contract={"kind": "review", "candidate_task_id": implementation},
            )
            kb.link_tasks(self.conn, implementation, reviewer, requirement="phase_finished")
        tester = kb.create_task(
            self.conn,
            title="acceptance tester",
            assignee="tester",
            initial_status="blocked",
            lifecycle_contract={"kind": "validation", "candidate_task_id": implementation},
        )
        if mode == "same_card":
            kb.link_tasks(self.conn, implementation, tester, requirement="review_approved")
        else:
            kb.link_tasks(self.conn, reviewer, tester, requirement="review_approved")
        return implementation, reviewer, tester

    def _run_implementation(self, implementation: str, base: str, head: str):
        if self._task(implementation).status == "blocked":
            self.assertTrue(kb.unblock_task(self.conn, implementation))
        run = kb.claim_task(self.conn, implementation, claimer="implementer:conformance")
        self.assertIsNotNone(run)
        self.assertTrue(
            kb.complete_task(
                self.conn,
                implementation,
                expected_run_id=run.current_run_id,
                summary="implementation fixture",
                metadata={
                    "base_sha": base,
                    "head_sha": head,
                    "changed_files": ["lifecycle.py"],
                },
            )
        )
        return run
    def _run_review(
        self,
        implementation: str,
        reviewer: str | None,
        head: str,
        verdict: str,
        finding: str | None = None,
    ):
        review_id = implementation if reviewer is None else reviewer
        run = kb.claim_review_task(self.conn, review_id, claimer="reviewer:conformance")
        self.assertIsNotNone(run)
        self.assertTrue(
            kb.complete_task(
                self.conn,
                review_id,
                expected_run_id=run.current_run_id,
                verdict=verdict,
                summary=finding or f"review {verdict.lower()}",
                metadata={"reviewed_head_sha": head},
            )
        )
        return run


    def _rework_typed_graph(
        self, implementation: str, reviewer: str | None, tester: str,
    ) -> dict:
        versions = {
            task_id: self._task(task_id).version
            for task_id in (implementation, reviewer, tester)
            if task_id is not None
        }
        return kb.rework_review_graph(
            self.conn,
            implementation,
            implementation if reviewer is None else reviewer,
            tester,
            expected_implementation_version=versions[implementation],
            expected_reviewer_version=None if reviewer is None else versions[reviewer],
            expected_tester_version=versions[tester],
            reason="repair the rejected lifecycle evidence",
            author="conformance",
        )

    def _assert_final_pass(
        self, implementation: str, reviewer: str | None, tester: str, head: str,
    ) -> None:
        self.assertEqual(self._task(tester).status, "ready")
        validation_run = kb.claim_task(self.conn, tester, claimer="tester:conformance")
        self.assertIsNotNone(validation_run)
        self.assertTrue(
            kb.complete_task(
                self.conn,
                tester,
                expected_run_id=validation_run.current_run_id,
                verdict="PASS",
                summary="validation passed on newest head",
                metadata={"head_sha": head},
            )
        )
        state = get_lifecycle_state(self.conn, implementation)
        self.assertEqual(state["acceptance"], "accepted")
        self.assertEqual(state["validation_verdict"], "PASS")
        self.assertEqual(state["head_sha"], head)
        if reviewer is not None:
            self.assertEqual(get_lifecycle_state(self.conn, reviewer)["review_verdict"], "APPROVE")

    def _exercise_two_rejections(self, mode: str) -> None:
        with tempfile.TemporaryDirectory(prefix=f"kanban-conformance-{mode}-rejections-") as raw:
            repo = Path(raw)
            base, head = self._repo_with_head(repo)
            implementation, reviewer, tester = self._typed_graph(repo, mode)
            ids = (implementation, reviewer, tester)
            implementation_heads = []
            for finding, marker in (("finding A", "version 2"), ("finding B", "version 3")):
                implementation_heads.append(head)
                self._run_implementation(implementation, base, head)
                self._run_review(implementation, reviewer, head, "REQUEST_CHANGES", finding)
                self.assertNotEqual(
                    get_lifecycle_state(self.conn, implementation)["acceptance"], "accepted",
                )
                self.assertIn(self._task(tester).status, {"todo", "blocked"})
                self.assertIsNone(kb.claim_task(self.conn, tester, claimer="tester:too-early"))
                outcome = self._rework_typed_graph(implementation, reviewer, tester)
                self.assertEqual(
                    (outcome["implementation_id"], outcome["reviewer_id"], outcome["tester_id"]),
                    (implementation, reviewer, tester),
                )
                head = self._new_head(repo, marker)
            implementation_heads.append(head)
            self._run_implementation(implementation, base, head)
            self._run_review(implementation, reviewer, head, "APPROVE")
            self._assert_final_pass(implementation, reviewer, tester, head)
            rejected_reviews = []
            for row in self.conn.execute(
                "SELECT summary, metadata FROM task_runs WHERE task_id = ? ORDER BY id",
                (implementation if reviewer is None else reviewer,),
            ):
                lifecycle = json.loads(row["metadata"] or "{}").get("lifecycle", {})
                if (
                    lifecycle.get("phase") == "review"
                    and lifecycle.get("verdict") == "REQUEST_CHANGES"
                ):
                    rejected_reviews.append((row["summary"], lifecycle["head_sha"]))
            self.assertEqual(
                rejected_reviews,
                [("finding A", implementation_heads[0]), ("finding B", implementation_heads[1])],
            )
            self.assertEqual((implementation, reviewer, tester), ids)
            runs = self.conn.execute(
                "SELECT metadata FROM task_runs WHERE task_id = ? ORDER BY id",
                (implementation,),
            ).fetchall()
            heads = {json.loads(row["metadata"] or "{}").get("head_sha") for row in runs}
            self.assertTrue(set(implementation_heads).issubset(heads))
            self.assertEqual(
                kb.child_ids(self.conn, implementation),
                [tester] if mode == "same_card" else [reviewer],
            )

    def test_same_card_two_distinct_rejections_then_final_pass(self) -> None:
        self._exercise_two_rejections("same_card")

    def test_separate_card_two_distinct_rejections_then_final_pass(self) -> None:
        self._exercise_two_rejections("separate_card")

    def _exercise_validator_failure_rework(self, mode: str) -> None:
        with tempfile.TemporaryDirectory(prefix=f"kanban-conformance-{mode}-validator-") as raw:
            repo = Path(raw)
            base, first_head = self._repo_with_head(repo)
            implementation, reviewer, tester = self._typed_graph(repo, mode)
            self._run_implementation(implementation, base, first_head)
            self._run_review(implementation, reviewer, first_head, "APPROVE")
            validation_run = kb.claim_task(self.conn, tester, claimer="tester:conformance")
            self.assertIsNotNone(validation_run)
            self.assertTrue(
                kb.complete_task(
                    self.conn,
                    tester,
                    expected_run_id=validation_run.current_run_id,
                    verdict="FAIL",
                    summary="validator found a regression",
                    metadata={"head_sha": first_head},
                )
            )
            failed = get_lifecycle_state(self.conn, implementation)
            self.assertNotEqual(failed["acceptance"], "accepted")
            self.assertEqual(failed["validation_verdict"], "FAIL")
            self.assertIsNone(kb.claim_task(self.conn, tester, claimer="tester:retry"))
            outcome = self._rework_typed_graph(implementation, reviewer, tester)
            self.assertEqual(outcome["rejected_head_sha"], first_head)
            second_head = self._new_head(repo, "validator repair")
            self._run_implementation(implementation, base, second_head)
            self.assertFalse(evaluate_dependencies(self.conn, tester)["satisfied"])
            self.assertEqual(self._task(tester).status, "todo")
            self.assertIsNone(kb.claim_task(self.conn, tester, claimer="tester:before-rereview"))
            self._run_review(implementation, reviewer, second_head, "APPROVE")
            self.assertTrue(evaluate_dependencies(self.conn, tester)["satisfied"])
            self.assertNotEqual(
                get_lifecycle_state(self.conn, implementation)["acceptance"], "accepted",
            )
            self._assert_final_pass(implementation, reviewer, tester, second_head)
            self.assertNotEqual(first_head, second_head)
            historical_heads = {
                json.loads(row["metadata"] or "{}").get("head_sha")
                for row in self.conn.execute(
                    "SELECT metadata FROM task_runs WHERE task_id = ?", (implementation,)
                )
            }
            self.assertIn(first_head, historical_heads)
            self.assertIn(second_head, historical_heads)
            validation_history = [
                json.loads(row["metadata"])["lifecycle"]
                for row in self.conn.execute(
                    "SELECT metadata FROM task_runs WHERE task_id = ? ORDER BY id", (tester,)
                )
            ]
            self.assertEqual(
                [(entry["verdict"], entry["head_sha"]) for entry in validation_history],
                [("FAIL", first_head), ("PASS", second_head)],
            )

    def test_same_card_validator_fail_repairs_and_retests(self) -> None:
        self._exercise_validator_failure_rework("same_card")

    def test_separate_card_validator_fail_repairs_and_retests(self) -> None:
        self._exercise_validator_failure_rework("separate_card")

    def test_effective_goal_survives_reopened_process(self) -> None:
        db_path = Path(self.home.name) / "persistent-goal.db"
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        conn.executescript(kb.SCHEMA_SQL)
        kb._ensure_lifecycle_schema(conn)
        kb._ensure_goal_revision_schema(conn)
        task_id = kb.create_task(
            conn,
            title="draft goal",
            body="draft body",
            assignee="implementer",
            initial_status="blocked",
        )
        self.assertTrue(kb.unblock_task(conn, task_id))
        task = kb.get_task(conn, task_id)
        self.assertIsNotNone(task)
        self.assertTrue(
            kb.update_task(
                conn,
                task_id,
                title="authorized goal",
                body="authorized body",
                goal_mode=True,
                expected_version=task.version,
                reason="persist authorized goal",
                author="cto",
            )
        )
        expected = kb.get_effective_goal(conn, task_id)
        conn.close()
        script = (
            "import json, sys\n"
            "from pathlib import Path\n"
            "from hermes_cli import kanban_db as kb\n"
            "from hermes_cli import kanban_db_connect as kbc\n"
            "with kbc.connect(Path(sys.argv[1])) as conn:\n"
            "    print(json.dumps(kb.get_effective_goal(conn, sys.argv[2])))\n"
        )
        env = {
            **os.environ,
            "HERMES_HOME": self.home.name,
            "HERMES_KANBAN_HOME": self.home.name,
            "PYTHONPATH": os.pathsep.join(
                [str(MODULE.RUNTIME_ROOT), os.environ.get("PYTHONPATH", "")]
            ),
        }
        child = subprocess.run(
            [sys.executable, "-c", script, str(db_path), task_id],
            cwd=str(MODULE.RUNTIME_ROOT),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(child.returncode, 0, child.stderr)
        self.assertEqual(json.loads(child.stdout), expected)

    def test_fresh_registry_discovers_describes_and_invokes_kanban_handler(self) -> None:
        (Path(self.home.name) / "config.yaml").write_text(
            "toolsets: [kanban]\n", encoding="utf-8",
        )
        script = (
            "import json\n"
            "from hermes_cli import kanban_db as kb\n"
            "from hermes_cli import kanban_db_connect as kbc\n"
            "from tools.registry import discover_builtin_tools, registry\n"
            "kb.init_db()\n"
            "with kbc.connect() as conn:\n"
            "    tid = kb.create_task(conn, title='fresh registry task', assignee='worker', "
            "initial_status='blocked')\n"
            "modules = discover_builtin_tools()\n"
            "definitions = registry.get_definitions({'kanban_show', 'kanban_unblock'}, quiet=True)\n"
            "unblocked = json.loads(registry.dispatch('kanban_unblock', {'task_id': tid}))\n"
            "shown = json.loads(registry.dispatch('kanban_show', {'task_id': tid}))\n"
            "print(json.dumps({'module': 'tools.kanban_tools' in modules, "
            "'schemas': [entry['function']['name'] for entry in definitions], "
            "'unblocked': unblocked, 'task_id': shown['task']['id'], "
            "'status': shown['task']['status']}))\n"
        )
        env = {
            **{key: value for key, value in os.environ.items() if not key.startswith("HERMES_KANBAN_")},
            "HERMES_HOME": self.home.name,
            "HERMES_KANBAN_HOME": self.home.name,
            "HERMES_PROFILE": "cto",
            "PYTHONPATH": os.pathsep.join(
                [str(MODULE.RUNTIME_ROOT), os.environ.get("PYTHONPATH", "")]
            ),
        }
        child = subprocess.run(
            [sys.executable, "-c", script],
            cwd=str(MODULE.RUNTIME_ROOT),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(child.returncode, 0, child.stderr)
        result = json.loads(child.stdout.strip().splitlines()[-1])
        self.assertTrue(result["module"])
        self.assertEqual(set(result["schemas"]), {"kanban_show", "kanban_unblock"})
        self.assertEqual(
            result["unblocked"],
            {"ok": True, "task_id": result["task_id"], "status": "ready"},
        )
        self.assertEqual(result["status"], "ready")

if __name__ == "__main__":
    unittest.main()

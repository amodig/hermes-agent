"""Hermetic conformance for Kanban lifecycle upgrade paths."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

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

    def _typed_graph(
        self, repo: Path, mode: str, *, validation_required: bool = True,
        branch: str | None = None, workspace_kind: str = "dir",
    ) -> tuple[str, str | None, str | None]:
        implementation = kb.create_task(
            self.conn,
            title=f"{mode} implementation",
            assignee="implementer",
            initial_status="blocked",
            workspace_kind=workspace_kind,
            workspace_path=str(repo),
            branch_name=branch,
            lifecycle_contract={
                "kind": "code",
                "review_mode": mode,
                "reviewer": "reviewer",
                "validation_required": validation_required,
            },
        )
        reviewer = None
        if mode == "separate_card":
            reviewer = kb.create_task(
                self.conn,
                title="separate reviewer",
                assignee="reviewer",
                lifecycle_contract={"kind": "review", "candidate_task_id": implementation},
            )
            kb.link_tasks(self.conn, implementation, reviewer, requirement="phase_finished")
        tester = None
        if validation_required:
            tester = kb.create_task(
                self.conn,
                title="acceptance tester",
                assignee="tester",
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

    # ------------------------------------------------------------------
    # Candidate supersession recovery (#485): one atomic typed operation
    # retires a moved-head quarantine whose review never started, without
    # rewriting any run, event, comment, or commit.
    # ------------------------------------------------------------------

    _SUPERSEDE_MUTABLE_FIELDS = frozenset({
        "status", "version", "completed_at", "result", "candidate_run_id",
        "current_run_id", "claim_lock", "claim_expires", "worker_pid", "worker_started_at",
        "block_kind", "block_recurrences",
    })

    def _candidate_repo(self, repo: Path, branch: str) -> tuple[str, str]:
        """Real repository whose checked-out branch carries the task branch."""
        MODULE._git(repo, "init", "-q", "-b", "main")
        MODULE._git(repo, "config", "user.email", "conformance@example.invalid")
        MODULE._git(repo, "config", "user.name", "Kanban Conformance")
        (repo / "README").write_text("base\n", encoding="utf-8")
        MODULE._git(repo, "add", "README")
        MODULE._git(repo, "commit", "-qm", "base")
        base = MODULE._git(repo, "rev-parse", "HEAD")
        MODULE._git(repo, "checkout", "-q", "-b", branch)
        (repo / "lifecycle.py").write_text("version 1\n", encoding="utf-8")
        MODULE._git(repo, "add", "lifecycle.py")
        MODULE._git(repo, "commit", "-qm", "implementation H1")
        return base, MODULE._git(repo, "rev-parse", "HEAD")

    def _divergent_head(self, repo: Path, marker: str, base: str) -> str:
        """A commit that does not contain the candidate head as an ancestor."""
        MODULE._git(repo, "reset", "-q", "--hard", base)
        return self._new_head(repo, marker)

    def _rows(self, table: str) -> list[tuple]:
        return [tuple(row) for row in self.conn.execute(f"SELECT * FROM {table} ORDER BY id")]

    def _dump_db(self) -> list[str]:
        return list(self.conn.iterdump())

    def _task_rows(self, ids) -> dict[str, dict]:
        ordered = [task_id for task_id in ids if task_id]
        placeholders = ",".join("?" for _ in ordered)
        return {
            row["id"]: dict(row)
            for row in self.conn.execute(
                f"SELECT * FROM tasks WHERE id IN ({placeholders}) ORDER BY id",
                tuple(ordered),
            )
        }

    def _events(self, task_id: str) -> list[dict]:
        return [
            {
                "id": row["id"],
                "kind": row["kind"],
                "run_id": row["run_id"],
                "payload": json.loads(row["payload"] or "{}"),
            }
            for row in self.conn.execute(
                "SELECT id, kind, payload, run_id FROM task_events WHERE task_id = ? ORDER BY id",
                (task_id,),
            )
        ]

    def _run_rows(self, task_id: str) -> list[tuple]:
        return [
            tuple(row) for row in self.conn.execute(
                "SELECT * FROM task_runs WHERE task_id = ? ORDER BY id", (task_id,),
            )
        ]

    def _versions(self, *ids: str | None) -> dict[str, int]:
        return {task_id: self._task(task_id).version for task_id in ids if task_id}

    def _raw_sql(self, statement: str, params: tuple = ()) -> None:
        """Committed fixture DML: a bare execute would strand an open txn."""
        with kb.write_txn(self.conn):
            self.conn.execute(statement, params)

    def _insert_run(self, task_id: str, *, status: str, worker_pid=None, metadata=None) -> int:
        """Raw run row for refusal fixtures (no lifecycle envelope by default)."""
        now = int(time.time())
        with kb.write_txn(self.conn):
            cursor = self.conn.execute(
                "INSERT INTO task_runs (task_id, profile, status, worker_pid, started_at, ended_at, "
                "outcome, summary, metadata) VALUES (?, 'fixture', ?, ?, ?, ?, ?, 'fixture', ?)",
                (
                    task_id, status, worker_pid, now,
                    None if status == "running" else now,
                    None if status == "running" else "completed",
                    json.dumps(metadata) if metadata is not None else None,
                ),
            )
            return int(cursor.lastrowid)

    def _detach_edge(self, parent_id: str, child_id: str) -> None:
        """Break a declared lifecycle edge the public unlink guard protects."""
        with kb.write_txn(self.conn):
            self.conn.execute(
                "DELETE FROM task_links WHERE parent_id = ? AND child_id = ?",
                (parent_id, child_id),
            )

    def _sticky_block(self, task_id: str) -> None:
        """Unrelated operator block: history says the card must stay blocked."""
        with kb.write_txn(self.conn):
            self.conn.execute(
                "UPDATE tasks SET status = 'blocked', block_kind = 'capability', "
                "block_recurrences = 1, version = version + 1 WHERE id = ?",
                (task_id,),
            )
            kb._append_event(self.conn, task_id, "blocked", {
                "kind": "capability",
                "cause": "capability",
                "reason": "unrelated operator block",
                "recurrences": 1,
                "source_status": "ready",
            })

    def _quarantined_candidate(
        self, repo: Path, mode: str, *, validation_required: bool = True,
        extra_child: bool = False, prior_rejection: bool = False, divergent: bool = False,
        branch: str = "task/candidate",
    ) -> dict:
        """Base -> H1 handoff, branch moved to H2, review quarantined unstarted."""
        base, head = self._candidate_repo(repo, branch)
        implementation, reviewer, tester = self._typed_graph(
            repo, mode, validation_required=validation_required,
            branch=branch, workspace_kind="worktree",
        )
        extra = None
        if extra_child:
            extra = kb.create_task(
                self.conn, title="unrelated descendant", assignee="worker",
                parents=[implementation],
            )
            self._sticky_block(extra)
        if prior_rejection:
            self._run_implementation(implementation, base, head)
            self._run_review(implementation, reviewer, head, "REQUEST_CHANGES", "prior finding")
            self._rework_typed_graph(implementation, reviewer, tester)
            head = self._new_head(repo, "version 1b")
        self._run_implementation(implementation, base, head)
        review_id = implementation if reviewer is None else reviewer
        self.assertEqual(self._task(review_id).status, "review")
        moved = (
            self._divergent_head(repo, "divergent head", base)
            if divergent else self._new_head(repo, "version 2")
        )
        self.assertTrue(
            kb.block_task(
                self.conn, review_id,
                reason="moved head quarantine pending operator recovery",
                kind="capability", quarantine_review=True,
                expected_version=self._task(review_id).version,
            )
        )
        candidate_run_id = get_lifecycle_state(self.conn, implementation)["candidate_run_id"]
        self.assertIsNotNone(candidate_run_id)
        return {
            "repo": repo, "base": base, "head": head, "moved_head": moved,
            "implementation": implementation, "reviewer": reviewer, "tester": tester,
            "review_id": review_id, "extra": extra, "candidate_run_id": int(candidate_run_id),
        }

    def _candidate_fixture(self, mode: str, *, prefix: str | None = None, **kwargs) -> dict:
        raw = tempfile.TemporaryDirectory(
            prefix=prefix or f"kanban-conformance-{mode}-supersede-",
        )
        self.addCleanup(raw.cleanup)
        return self._quarantined_candidate(Path(raw.name), mode, **kwargs)

    def _affected(self, case: dict) -> list[str]:
        return [
            task_id for task_id in (
                case["implementation"], case["reviewer"], case["tester"], case["extra"],
            ) if task_id
        ]

    def _supersede(self, case: dict, *, versions=None, **overrides) -> dict:
        kwargs = {
            "expected_versions": (
                versions if versions is not None else self._versions(*self._affected(case))
            ),
            "expected_candidate_run_id": case["candidate_run_id"],
            "expected_head_sha": case["head"],
            "replacement_head_sha": case["moved_head"],
            "reason": "supersede the moved, never-reviewed candidate before a fresh handoff",
            "author": "conformance",
        }
        kwargs.update(overrides)
        return kb.supersede_candidate(self.conn, case["implementation"], **kwargs)

    def _assert_supersede_refused(
        self, case: dict, *, versions=None, check_git: bool = True, **overrides,
    ) -> None:
        dump_before = self._dump_db()
        head_before = MODULE._git(case["repo"], "rev-parse", "HEAD") if check_git else None
        with self.assertRaises(ValueError):
            self._supersede(case, versions=versions, **overrides)
        self.assertEqual(self._dump_db(), dump_before)
        if check_git:
            self.assertEqual(MODULE._git(case["repo"], "rev-parse", "HEAD"), head_before)

    def _bound_role_runs(self, task_id: str, candidate_run_id: int) -> list[tuple]:
        found: list[tuple] = []
        for row in self.conn.execute(
            "SELECT id, metadata FROM task_runs WHERE task_id = ? ORDER BY id", (task_id,),
        ):
            metadata = json.loads(row["metadata"] or "{}")
            lifecycle = metadata.get("lifecycle")
            lifecycle = lifecycle if isinstance(lifecycle, dict) else {}
            if (
                lifecycle.get("candidate_run_id") == candidate_run_id
                or metadata.get("candidate_run_id") == candidate_run_id
            ):
                found.append(tuple(row))
        return found

    def _supersession_provenance(self, task_id: str) -> list[dict]:
        found: list[dict] = []

        def _entry(container) -> dict | None:
            if not isinstance(container, dict):
                return None
            value = container.get("candidate_supersession")
            if not isinstance(value, dict):
                lifecycle = container.get("lifecycle")
                value = (
                    lifecycle.get("candidate_supersession")
                    if isinstance(lifecycle, dict) else None
                )
            return value if isinstance(value, dict) else None

        for row in self.conn.execute(
            "SELECT metadata FROM task_runs WHERE task_id = ? ORDER BY id", (task_id,),
        ):
            entry = _entry(json.loads(row["metadata"] or "{}"))
            if entry:
                found.append(entry)
        for row in self.conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? ORDER BY id", (task_id,),
        ):
            entry = _entry(json.loads(row["payload"] or "{}"))
            if entry:
                found.append(entry)
        return found

    def test_same_card_supersede_recovers_unreviewed_moved_candidate(self) -> None:
        self._exercise_supersede_recovery("same_card")

    def test_separate_card_supersede_recovers_unreviewed_moved_candidate(self) -> None:
        self._exercise_supersede_recovery("separate_card")

    def _exercise_supersede_recovery(self, mode: str) -> None:
        case = self._candidate_fixture(mode, extra_child=True)
        implementation = case["implementation"]
        reviewer = case["reviewer"]
        tester = case["tester"]
        extra = case["extra"]
        affected = self._affected(case)
        # Reconciliation may clear the PID while leaving its old start fingerprint.
        with kb.write_txn(self.conn):
            self.conn.executemany(
                "UPDATE tasks SET worker_started_at = ? WHERE id = ?",
                [(123456, task_id) for task_id in affected],
            )
        versions = self._versions(*affected)
        before_rows = self._task_rows(affected)
        before_children = kb.child_ids(self.conn, implementation)
        before_runs = self._rows("task_runs")
        before_events = self._rows("task_events")

        outcome = self._supersede(case, versions=versions)

        # no role run is created and no historical row is rewritten
        self.assertEqual(self._rows("task_runs"), before_runs)
        self.assertEqual(self._rows("task_events")[: len(before_events)], before_events)
        # graph identity and both commits survive on the existing branch
        self.assertEqual(kb.child_ids(self.conn, implementation), before_children)
        self.assertEqual(MODULE._git(case["repo"], "rev-parse", "HEAD"), case["moved_head"])
        self.assertEqual(MODULE._git(case["repo"], "cat-file", "-t", case["head"]), "commit")
        self.assertEqual(MODULE._git(case["repo"], "cat-file", "-t", case["moved_head"]), "commit")

        self.assertEqual(outcome["implementation_id"], implementation)
        self.assertEqual(outcome["reviewer_id"], reviewer)
        self.assertEqual(outcome["tester_id"], tester)
        self.assertEqual(outcome["status"], "ready")
        self.assertEqual(outcome["superseded_candidate_run_id"], case["candidate_run_id"])
        self.assertEqual(outcome["superseded_head_sha"], case["head"])
        self.assertEqual(outcome["replacement_head_sha"], case["moved_head"])
        self.assertEqual(set(outcome["versions"]), set(affected))
        self.assertEqual(
            {task_id: before_rows[task_id]["version"] + 1 for task_id in affected},
            outcome["versions"],
        )
        self.assertEqual({entry["id"] for entry in outcome["invalidated"]}, set(affected))
        for entry in outcome["invalidated"]:
            self.assertEqual(entry["new_version"], outcome["versions"][entry["id"]])
            self.assertIn("prior_status", entry)
            self.assertIn("new_status", entry)

        # current scheduling state cleared; every other column untouched
        for task_id in affected:
            before = before_rows[task_id]
            after = self._task_rows([task_id])[task_id]
            # same-card recovery restores the original implementer on the
            # implementation card, which also serves as the review card.
            exempt = self._SUPERSEDE_MUTABLE_FIELDS
            if reviewer is None and task_id == implementation:
                exempt = exempt | {"assignee"}
            for field, value in before.items():
                if field in exempt:
                    continue
                self.assertEqual(after[field], value, f"{task_id}.{field}")
            for field in (
                "candidate_run_id", "current_run_id", "claim_lock", "claim_expires",
                "worker_pid", "worker_started_at", "completed_at", "result",
            ):
                self.assertIsNone(after[field], f"{task_id}.{field}")
            self.assertEqual(after["version"], before["version"] + 1)
            invalidations = [
                event for event in self._events(task_id)
                if event["kind"] == "acceptance_invalidated"
            ]
            self.assertEqual(len(invalidations), 1, task_id)
            self.assertEqual(invalidations[0]["payload"]["superseded_head_sha"], case["head"])
            self.assertNotIn("rejected_head_sha", invalidations[0]["payload"])
            self.assertEqual(invalidations[0]["payload"]["reviewer"], reviewer)
            self.assertEqual(invalidations[0]["payload"]["tester"], tester)

        # the unrelated sticky descendant keeps its cause; role gates drop to todo
        self.assertEqual(self._task(implementation).status, "ready")
        if reviewer is None:
            # same-card recovery restores the original implementation routing
            self.assertEqual(
                self._task(implementation).assignee, kb._canonical_assignee("implementer"),
            )
        self.assertEqual(self._task(tester).status, "todo")
        if reviewer is not None:
            self.assertEqual(self._task(reviewer).status, "todo")
        self.assertEqual(self._task(extra).status, "blocked")
        self.assertEqual(self._task(extra).block_kind, before_rows[extra]["block_kind"])
        self.assertEqual(
            self._task(extra).block_recurrences, before_rows[extra]["block_recurrences"],
        )

        # no candidate, no verdict, no acceptance
        state = get_lifecycle_state(self.conn, implementation)
        self.assertIsNone(state["candidate_run_id"])
        self.assertIsNone(state["head_sha"])
        self.assertIsNone(state["review_verdict"])
        self.assertIsNone(state["validation_verdict"])
        self.assertEqual(state["acceptance"], "pending")

        # exactly the matched quarantine is retired, with supersession provenance
        unblocked = [
            event for event in self._events(case["review_id"])
            if event["kind"] == "unblocked"
            and event["payload"].get("recovery") == "candidate_superseded"
        ]
        self.assertEqual(len(unblocked), 1)
        self.assertEqual(
            unblocked[0]["payload"]["status"], "ready" if reviewer is None else "todo",
        )
        self.assertEqual(
            unblocked[0]["payload"]["resume_status"], "ready" if reviewer is None else "review",
        )
        quarantine = [
            event for event in self._events(case["review_id"]) if event["kind"] == "blocked"
        ][-1]
        self.assertEqual(unblocked[0]["payload"]["prior_block_event_id"], quarantine["id"])

        supersession = [
            event for event in self._events(implementation)
            if event["kind"] == "candidate_superseded"
        ]
        self.assertEqual(len(supersession), 1)
        self.assertEqual(supersession[0]["id"], outcome["supersession_event_id"])
        self.assertEqual(supersession[0]["run_id"], case["candidate_run_id"])
        event = supersession[0]["payload"]
        self.assertEqual(event["actor"], "conformance")
        self.assertEqual(event["implementation_id"], implementation)
        self.assertEqual(event["reviewer_id"], reviewer)
        self.assertEqual(event["tester_id"], tester)
        self.assertEqual(event["superseded_candidate_run_id"], case["candidate_run_id"])
        self.assertEqual(event["superseded_head_sha"], case["head"])
        self.assertEqual(event["replacement_head_sha"], case["moved_head"])
        self.assertEqual(event["quarantine_event_id"], quarantine["id"])
        self.assertEqual(event["expected_versions"], versions)
        self.assertEqual(event["invalidated"], outcome["invalidated"])
        self.assertEqual(event["handoff_error"]["kind"], "handoff_head_moved")
        self.assertEqual(event["handoff_error"]["expected_head_sha"], case["head"])
        self.assertEqual(event["handoff_error"]["actual_head_sha"], case["moved_head"])
        completion = [
            item for item in self._events(implementation)
            if item["run_id"] == case["candidate_run_id"]
            and item["kind"] in {"completed", "review_requested"}
        ]
        self.assertTrue(completion)
        self.assertEqual(event["supersedes_completion_event_id"], completion[0]["id"])

        # readiness recompute must not release the retired role gates early
        self.assertEqual(kb.recompute_ready(self.conn), 0)
        self.assertEqual(self._task(implementation).status, "ready")
        self.assertEqual(self._task(tester).status, "todo")
        if reviewer is not None:
            self.assertEqual(self._task(reviewer).status, "todo")
        self.assertIsNone(kb.claim_review_task(self.conn, case["review_id"]))
        self.assertIsNone(kb.claim_task(self.conn, tester))
        if reviewer is None:
            self.assertEqual(
                kb.handoff_phase(self.conn, self._task(implementation)), "implementation",
            )

    def test_same_card_supersede_fresh_handoff_reuses_existing_newer_head(self) -> None:
        self._exercise_supersede_fresh_handoff("same_card")

    def test_separate_card_supersede_fresh_handoff_reuses_existing_newer_head(self) -> None:
        self._exercise_supersede_fresh_handoff("separate_card")

    def _exercise_supersede_fresh_handoff(self, mode: str) -> None:
        case = self._candidate_fixture(mode)
        implementation = case["implementation"]
        reviewer = case["reviewer"]
        tester = case["tester"]
        outcome = self._supersede(case)

        # the retired review is unclaimable and the validator stays gated
        self.assertIsNone(kb.claim_review_task(self.conn, case["review_id"]))
        self.assertFalse(evaluate_dependencies(self.conn, tester)["satisfied"])
        self.assertIsNone(kb.claim_task(self.conn, tester))
        self.assertEqual(self._task(tester).status, "todo")

        runs_before = self._rows("task_runs")
        events_before = self._rows("task_events")

        # fresh handoff at the already-existing newer commit; no extra commit
        run = self._run_implementation(implementation, case["base"], case["moved_head"])
        state = get_lifecycle_state(self.conn, implementation)
        self.assertEqual(state["head_sha"], case["moved_head"])
        self.assertIsNotNone(state["candidate_run_id"])
        self.assertNotEqual(state["candidate_run_id"], case["candidate_run_id"])
        self.assertEqual(state["candidate_run_id"], run.current_run_id)
        self.assertEqual(MODULE._git(case["repo"], "rev-parse", "HEAD"), case["moved_head"])
        self.assertEqual(self._rows("task_runs")[: len(runs_before)], runs_before)
        self.assertEqual(self._rows("task_events")[: len(events_before)], events_before)

        # the fresh handoff records supersession provenance
        provenance = [
            entry for entry in self._supersession_provenance(implementation)
            if entry.get("supersession_event_id") == outcome["supersession_event_id"]
        ]
        self.assertTrue(provenance)
        for entry in provenance:
            self.assertEqual(entry["superseded_candidate_run_id"], case["candidate_run_id"])
            self.assertEqual(entry["superseded_head_sha"], case["head"])
            self.assertEqual(entry["replacement_head_sha"], case["moved_head"])

        # only the original review opens; the tester waits for a fresh verdict
        self.assertEqual(self._task(case["review_id"]).status, "review")
        self.assertEqual(self._task(tester).status, "todo")
        self._run_review(implementation, reviewer, case["moved_head"], "APPROVE")
        self.assertEqual(self._task(tester).status, "ready")
        self.assertNotEqual(get_lifecycle_state(self.conn, implementation)["acceptance"], "accepted")
        self._assert_final_pass(implementation, reviewer, tester, case["moved_head"])

    def test_same_card_supersede_without_declared_validation(self) -> None:
        self._exercise_supersede_without_validation("same_card")

    def test_separate_card_supersede_without_declared_validation(self) -> None:
        self._exercise_supersede_without_validation("separate_card")

    def _exercise_supersede_without_validation(self, mode: str) -> None:
        case = self._candidate_fixture(
            mode, validation_required=False,
            prefix=f"kanban-conformance-{mode}-supersede-unvalidated-",
        )
        self.assertIsNone(case["tester"])
        affected = self._affected(case)
        outcome = self._supersede(case)
        self.assertIsNone(outcome["tester_id"])
        self.assertEqual(set(outcome["versions"]), set(affected))
        self.assertEqual(outcome["status"], "ready")

        self._run_implementation(case["implementation"], case["base"], case["moved_head"])
        self._run_review(
            case["implementation"], case["reviewer"], case["moved_head"], "APPROVE",
        )
        state = get_lifecycle_state(self.conn, case["implementation"])
        self.assertEqual(state["acceptance"], "accepted")
        self.assertEqual(state["head_sha"], case["moved_head"])
        self.assertEqual(state["review_verdict"], "APPROVE")
        self.assertIsNone(state["validation_verdict"])
        self.assertEqual(
            kb.child_ids(self.conn, case["implementation"]),
            [case["reviewer"]] if case["reviewer"] else [],
        )

    def test_same_card_supersede_after_older_accepted_candidate_stays_pending(self) -> None:
        self._exercise_supersede_after_older_accepted_candidate("same_card")

    def test_separate_card_supersede_after_older_accepted_candidate_stays_pending(self) -> None:
        self._exercise_supersede_after_older_accepted_candidate("separate_card")

    def _exercise_supersede_after_older_accepted_candidate(self, mode: str) -> None:
        raw = tempfile.TemporaryDirectory(
            prefix=f"kanban-conformance-{mode}-older-accepted-",
        )
        self.addCleanup(raw.cleanup)
        repo = Path(raw.name)
        branch = "task/candidate"
        base, first_head = self._candidate_repo(repo, branch)
        implementation, reviewer, tester = self._typed_graph(
            repo, mode, branch=branch, workspace_kind="worktree",
        )
        # candidate 1 is accepted, leaving successful APPROVE/PASS runs behind
        self._run_implementation(implementation, base, first_head)
        self._run_review(implementation, reviewer, first_head, "APPROVE")
        self._assert_final_pass(implementation, reviewer, tester, first_head)
        first_candidate_run_id = get_lifecycle_state(self.conn, implementation)["candidate_run_id"]
        self.assertIsNotNone(first_candidate_run_id)
        review_id = implementation if reviewer is None else reviewer
        self.assertEqual(get_lifecycle_state(self.conn, review_id)["review_verdict"], "APPROVE")
        self.assertEqual(get_lifecycle_state(self.conn, tester)["validation_verdict"], "PASS")
        role_runs = {
            task_id: self._run_rows(task_id) for task_id in (review_id, tester)
        }

        # a goal revision reopens the same cards for a new, never-reviewed candidate
        current = self._task(implementation)
        self.assertTrue(
            kb.update_task(
                self.conn, implementation, body="revised candidate goal",
                expected_version=current.version, reason="new candidate after acceptance",
                author="cto",
            )
        )
        head = self._new_head(repo, "version 1b")
        self._run_implementation(implementation, base, head)
        self.assertEqual(self._task(review_id).status, "review")
        moved = self._new_head(repo, "version 2")
        self.assertTrue(
            kb.block_task(
                self.conn, review_id,
                reason="moved head quarantine pending operator recovery",
                kind="capability", quarantine_review=True,
                expected_version=self._task(review_id).version,
            )
        )
        candidate_run_id = get_lifecycle_state(self.conn, implementation)["candidate_run_id"]
        self.assertIsNotNone(candidate_run_id)
        self.assertGreater(candidate_run_id, first_candidate_run_id)
        case = {
            "repo": repo, "base": base, "head": head, "moved_head": moved,
            "implementation": implementation, "reviewer": reviewer, "tester": tester,
            "review_id": review_id, "extra": None, "candidate_run_id": int(candidate_run_id),
        }

        outcome = self._supersede(case)
        self.assertEqual(outcome["superseded_candidate_run_id"], candidate_run_id)

        # older successful role history is preserved and still bound to candidate 1
        self.assertTrue(self._bound_role_runs(review_id, first_candidate_run_id))
        self.assertTrue(self._bound_role_runs(tester, first_candidate_run_id))
        for task_id, rows in role_runs.items():
            self.assertEqual(self._run_rows(task_id)[: len(rows)], rows)

        # the current candidate carries no verdict and stays pending, gates closed
        self.assertIsNone(get_lifecycle_state(self.conn, implementation)["candidate_run_id"])
        for task_id in (reviewer, tester):
            if task_id:
                self.assertEqual(self._bound_role_runs(task_id, candidate_run_id), [])
        for task_id in (implementation, reviewer, tester):
            if task_id:
                self.assertEqual(get_lifecycle_state(self.conn, task_id)["acceptance"], "pending")
        self.assertIn(
            "candidate_missing", get_lifecycle_state(self.conn, tester)["diagnostics"],
        )
        self.assertIsNone(kb.claim_review_task(self.conn, review_id))
        self.assertFalse(evaluate_dependencies(self.conn, tester)["satisfied"])
        self.assertIsNone(kb.claim_task(self.conn, tester))

        # the existing chain still completes on the newer head
        self._run_implementation(implementation, base, moved)
        self._run_review(implementation, reviewer, moved, "APPROVE")
        self.assertEqual(self._task(tester).status, "ready")
        self._assert_final_pass(implementation, reviewer, tester, moved)

    def test_same_card_supersede_refuses_version_head_and_workspace_faults(self) -> None:
        self._exercise_supersede_refusals("same_card")

    def test_separate_card_supersede_refuses_version_head_and_workspace_faults(self) -> None:
        self._exercise_supersede_refusals("separate_card")

    def _exercise_supersede_refusals(self, mode: str) -> None:
        # stale, missing, and extra version-map entries
        case = self._candidate_fixture(mode)
        implementation = case["implementation"]
        versions = self._versions(*self._affected(case))
        self._assert_supersede_refused(
            case, versions=dict(versions, **{implementation: versions[implementation] + 1}),
        )
        self._assert_supersede_refused(
            case, versions={k: v for k, v in versions.items() if k != implementation},
        )
        self._assert_supersede_refused(
            case, versions={k: v for k, v in versions.items() if k != case["tester"]},
        )

        unrelated = self._candidate_fixture(mode)
        extra_card = kb.create_task(self.conn, title="unrelated card", assignee="worker")
        self._assert_supersede_refused(
            unrelated,
            versions=dict(self._versions(*self._affected(unrelated)), **{extra_card: 1}),
        )

        grown = self._candidate_fixture(mode)
        kb.create_task(
            self.conn, title="late descendant", assignee="worker",
            parents=[grown["implementation"]],
        )
        self._assert_supersede_refused(grown, versions=self._versions(*self._affected(grown)))

        # wrong candidate run, wrong expected head, unchanged or bogus replacement
        case = self._candidate_fixture(mode)
        versions = self._versions(*self._affected(case))
        self._assert_supersede_refused(
            case, versions=versions, expected_candidate_run_id=case["candidate_run_id"] + 1,
        )
        self._assert_supersede_refused(case, versions=versions, expected_head_sha=case["base"])
        self._assert_supersede_refused(case, versions=versions, expected_head_sha=case["moved_head"])
        self._assert_supersede_refused(case, versions=versions, replacement_head_sha=case["head"])
        self._assert_supersede_refused(case, versions=versions, replacement_head_sha="0" * 40)
        MODULE._git(case["repo"], "checkout", "-q", "--detach", case["base"])
        self._assert_supersede_refused(case, versions=versions)

        # invalid arguments, and a worker context, never mutate
        case = self._candidate_fixture(mode)
        implementation = case["implementation"]
        versions = self._versions(*self._affected(case))
        for overrides in (
            {"reason": "   "},
            {"expected_candidate_run_id": True},
            {"expected_candidate_run_id": 0},
            {"expected_head_sha": "  "},
            {"replacement_head_sha": ""},
        ):
            self._assert_supersede_refused(case, versions=versions, **overrides)
        for bad_versions in (
            {},
            {implementation: 0},
            {implementation: True},
            {"   ": 1},
            {**versions, "   ": 1},
        ):
            self._assert_supersede_refused(case, versions=bad_versions)
        dump_before = self._dump_db()
        with self.assertRaises(ValueError):
            kb.supersede_candidate(
                self.conn, "   ", expected_versions=versions,
                expected_candidate_run_id=case["candidate_run_id"],
                expected_head_sha=case["head"], replacement_head_sha=case["moved_head"],
                reason="blank implementation id",
            )
        with patch.dict(os.environ, {"HERMES_KANBAN_TASK": implementation}):
            with self.assertRaises(ValueError):
                self._supersede(case, versions=versions)
        self.assertEqual(self._dump_db(), dump_before)

        # workspace faults: dirty tree, unavailable workspace, wrong branch, divergent head
        dirty = self._candidate_fixture(mode)
        (dirty["repo"] / "untracked.txt").write_text("dirty\n", encoding="utf-8")
        self._assert_supersede_refused(dirty)

        mismatched = self._candidate_fixture(mode)
        MODULE._git(mismatched["repo"], "checkout", "-q", "-b", "other-branch")
        self._assert_supersede_refused(mismatched)

        missing = self._candidate_fixture(mode)
        shutil.rmtree(missing["repo"])
        self._assert_supersede_refused(missing, check_git=False)

        divergent = self._candidate_fixture(
            mode, prefix=f"kanban-conformance-{mode}-divergent-", divergent=True,
        )
        self._assert_supersede_refused(divergent)

    def test_same_card_supersede_refuses_conflicting_graph_and_role_state(self) -> None:
        self._exercise_supersede_state_refusals("same_card")

    def test_separate_card_supersede_refuses_conflicting_graph_and_role_state(self) -> None:
        self._exercise_supersede_state_refusals("separate_card")

    def _exercise_supersede_state_refusals(self, mode: str) -> None:
        # the declared validation edge removed: the chain is no longer complete
        case = self._candidate_fixture(mode)
        validation_parent = case["reviewer"] or case["implementation"]
        self._detach_edge(validation_parent, case["tester"])
        self._assert_supersede_refused(
            case, versions=self._versions(case["implementation"], case["reviewer"]),
        )

        if case["reviewer"] is not None:
            orphaned = self._candidate_fixture(mode)
            self._detach_edge(orphaned["implementation"], orphaned["reviewer"])
            self._assert_supersede_refused(
                orphaned, versions=self._versions(orphaned["implementation"]),
            )

        # live claim fields, an unclosed run, and a closed run keeping its worker
        for mutate in (
            lambda case: self._raw_sql(
                "UPDATE tasks SET claim_lock = 'fixture', worker_pid = 4242 WHERE id = ?",
                (case["tester"],),
            ),
            lambda case: self._insert_run(case["tester"], status="running"),
            lambda case: self._insert_run(case["tester"], status="completed", worker_pid=4242),
        ):
            case = self._candidate_fixture(mode)
            mutate(case)
            self._assert_supersede_refused(case)

        # archived descendant
        case = self._candidate_fixture(mode)
        with kb.write_txn(self.conn):
            self.conn.execute(
                "UPDATE tasks SET status = 'archived' WHERE id = ?", (case["tester"],),
            )
        self._assert_supersede_refused(case)

        # a verdict and a start attempt already bound to the current candidate
        case = self._candidate_fixture(mode)
        self._insert_run(case["review_id"], status="completed", metadata={
            "lifecycle": {
                "phase": "review", "verdict": "APPROVE",
                "candidate_run_id": case["candidate_run_id"],
            },
        })
        self._assert_supersede_refused(case)

        case = self._candidate_fixture(mode)
        self._insert_run(case["review_id"], status="completed")
        self._assert_supersede_refused(case)

        # an unrelated later block is not a moved-head quarantine
        case = self._candidate_fixture(mode)
        with kb.write_txn(self.conn):
            kb._append_event(self.conn, case["review_id"], "blocked", {
                "kind": "capability", "cause": "capability",
                "reason": "unrelated operator block", "recurrences": 1,
                "source_status": "review",
            })
        self._assert_supersede_refused(case)

        # the quarantine is already retired
        case = self._candidate_fixture(mode)
        self.assertTrue(kb.unblock_task(self.conn, case["review_id"]))
        self._assert_supersede_refused(case)

        # the quarantine no longer matches the workspace: the branch moved again
        case = self._candidate_fixture(mode)
        third = self._new_head(case["repo"], "version 3")
        self._assert_supersede_refused(case, replacement_head_sha=third)

    def test_same_card_supersede_replay_after_success_is_stale(self) -> None:
        self._exercise_supersede_replay("same_card")

    def test_separate_card_supersede_replay_after_success_is_stale(self) -> None:
        self._exercise_supersede_replay("separate_card")

    def _exercise_supersede_replay(self, mode: str) -> None:
        case = self._candidate_fixture(mode)
        versions = self._versions(*self._affected(case))
        outcome = self._supersede(case, versions=versions)
        self.assertEqual(outcome["superseded_candidate_run_id"], case["candidate_run_id"])
        dump_before = self._dump_db()
        with self.assertRaises(ValueError):
            self._supersede(case, versions=versions)
        self.assertEqual(self._dump_db(), dump_before)

    def test_same_card_fresh_handoff_rejects_superseded_head_and_accepts_descendant(self) -> None:
        self._exercise_fresh_handoff_ancestry("same_card")

    def test_separate_card_fresh_handoff_rejects_superseded_head_and_accepts_descendant(self) -> None:
        self._exercise_fresh_handoff_ancestry("separate_card")

    def _exercise_fresh_handoff_ancestry(self, mode: str) -> None:
        case = self._candidate_fixture(mode)
        implementation = case["implementation"]
        self._supersede(case)

        # the branch was rewound: it no longer contains the recorded replacement
        MODULE._git(case["repo"], "reset", "-q", "--hard", case["head"])
        run = kb.claim_task(self.conn, implementation, claimer="implementer:conformance")
        self.assertIsNotNone(run)
        with self.assertRaises(ValueError):
            kb.complete_task(
                self.conn, implementation, expected_run_id=run.current_run_id,
                summary="rewound handoff",
                metadata={"base_sha": case["base"], "head_sha": case["head"]},
            )
        self.assertIsNone(get_lifecycle_state(self.conn, implementation)["candidate_run_id"])

        # a descendant of the recorded replacement is accepted
        MODULE._git(case["repo"], "reset", "-q", "--hard", case["moved_head"])
        descendant = self._new_head(case["repo"], "version 3")
        self.assertNotEqual(descendant, case["moved_head"])
        self.assertTrue(
            kb.complete_task(
                self.conn, implementation, expected_run_id=run.current_run_id,
                summary="fresh implementation handoff",
                metadata={"base_sha": case["base"], "head_sha": descendant},
            )
        )
        state = get_lifecycle_state(self.conn, implementation)
        self.assertEqual(state["head_sha"], descendant)
        self.assertIsNotNone(state["candidate_run_id"])
        self.assertNotEqual(state["candidate_run_id"], case["candidate_run_id"])

    def test_same_card_supersede_allows_older_rejected_candidate_role_history(self) -> None:
        self._exercise_supersede_allows_older_role_runs("same_card")

    def test_separate_card_supersede_allows_older_rejected_candidate_role_history(self) -> None:
        self._exercise_supersede_allows_older_role_runs("separate_card")

    def _exercise_supersede_allows_older_role_runs(self, mode: str) -> None:
        case = self._candidate_fixture(
            mode, prior_rejection=True,
            prefix=f"kanban-conformance-{mode}-older-role-runs-",
        )
        case["candidate_run_id"] = int(case["candidate_run_id"])
        candidate_run_id = case["candidate_run_id"]
        role_ids = [case["review_id"]]
        if case["tester"]:
            role_ids.append(case["tester"])
        older = [
            row["id"]
            for task_id in role_ids
            for row in self.conn.execute(
                "SELECT id FROM task_runs WHERE task_id = ? ORDER BY id", (task_id,),
            )
            if row["id"] < candidate_run_id
        ]
        self.assertTrue(older)
        outcome = self._supersede(case)
        self.assertEqual(outcome["superseded_candidate_run_id"], candidate_run_id)
        self.assertEqual(set(outcome["versions"]), set(self._affected(case)))

if __name__ == "__main__":
    unittest.main()

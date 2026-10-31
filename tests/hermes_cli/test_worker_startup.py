"""Authenticated worker startup metadata: grant authority and read-only board proof.

Every fixture is an isolated temporary board built from the real schema plus a
real workspace directory. No live profile, claim, bootstrap, application
database, subprocess or network endpoint is touched; only the runtime sealing
helpers are patched, because the gateway generation environment is not what
these tests are about.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_runtime as runtime
from hermes_cli import kanban_worker_runtime as worker_runtime
from hermes_cli import worker_startup

TASK_ID = "t_fixture_startup"
RUN_ID = 7
PROFILE = "reviewer"
BOARD = "fixture-board"
CLAIM_LOCK = "claim-fixture-lock"
PREPARATION_ID = "preparation-fixture"


def fixture_identity():
    """This process's real identity, sealed with an explicit generation marker."""
    return replace(
        runtime.runtime_identity(),
        generation="fixture-generation",
        dependency_fingerprint="fixture-dependency-fingerprint",
    )


class WorkerStartupFixture(unittest.TestCase):
    def setUp(self):
        self.identity = fixture_identity()
        for target, attribute, value in (
            (runtime, "runtime_identity", mock.Mock(return_value=self.identity)),
            (runtime, "assert_runtime_import_root", mock.Mock(return_value=self.identity)),
        ):
            patcher = mock.patch.object(target, attribute, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.new_fixture()

    def new_fixture(self):
        """Fresh board, workspace and retained-grant slot for one assertion."""
        worker_startup._GRANT = None
        directory = tempfile.TemporaryDirectory(prefix="worker-startup-")
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.workspace = self.directory / "workspace"
        self.workspace.mkdir()
        self.board = self.directory / "board.db"
        self.bootstrap_path = self.directory / "bootstrap.json"
        connection = sqlite3.connect(self.board)
        try:
            connection.executescript(kb.SCHEMA_SQL)
            connection.commit()
        finally:
            connection.close()

    # ---- fixture writers -------------------------------------------------
    def write_worker_state(
        self, *, status="running", ended_at=None, claim_lock=CLAIM_LOCK,
        run_claim_lock=CLAIM_LOCK, task_pid=None, metadata=None,
    ):
        """Create the exact task/run pair a granted worker owns."""
        connection = sqlite3.connect(self.board)
        connection.row_factory = sqlite3.Row
        try:
            task_id = kb.create_task(
                connection,
                title="worker startup fixture",
                assignee=PROFILE,
                workspace_kind="scratch",
                workspace_path=str(self.workspace),
                initial_status="running",
            )
            connection.execute("BEGIN")
            connection.execute(
                "UPDATE tasks SET status = ?, claim_lock = ?, claim_expires = 1, "
                "worker_pid = ?, worker_started_at = NULL, branch_name = 'wt/fixture' "
                "WHERE id = ?",
                (status, claim_lock, task_pid or os.getpid(), task_id),
            )
            cursor = connection.execute(
                "INSERT INTO task_runs (task_id, profile, status, metadata, claim_lock, "
                "claim_expires, worker_pid, started_at, ended_at) "
                "VALUES (?, ?, 'running', ?, ?, 1, ?, 1, ?)",
                (
                    task_id, PROFILE,
                    json.dumps(self.claim_metadata() if metadata is None else metadata),
                    run_claim_lock, task_pid or os.getpid(), ended_at,
                ),
            )
            run_id = cursor.lastrowid
            connection.execute(
                "UPDATE tasks SET current_run_id = ? WHERE id = ?", (run_id, task_id)
            )
            connection.commit()
        finally:
            connection.close()
        self.task_id = task_id
        self.run_id = run_id
        return task_id, run_id

    def claim_metadata(self):
        return {
            "runtime_identity": self.identity.as_dict(),
            "worker_pid": self.identity.pid,
            "worker_start_time": self.identity.start_time,
            "preparation_id": PREPARATION_ID,
        }

    def mutate(self, statement):
        connection = sqlite3.connect(self.board)
        try:
            connection.execute(statement)
            connection.commit()
        finally:
            connection.close()

    def append_event(self, payload):
        connection = sqlite3.connect(self.board)
        try:
            connection.execute(
                "INSERT INTO task_events (task_id, run_id, kind, payload, created_at) "
                "VALUES (?, ?, 'spawned', ?, 1)",
                (
                    self.task_id, self.run_id,
                    payload if isinstance(payload, str) else json.dumps(payload),
                ),
            )
            connection.commit()
        finally:
            connection.close()

    def worker_context(self, **overrides):
        context = {
            "task_id": getattr(self, "task_id", TASK_ID),
            "profile": PROFILE,
            "board": BOARD,
            "workspace": str(self.workspace),
            "db_path": str(self.board),
        }
        context.update(overrides)
        return context

    def grant(self, context=None, **overrides):
        value = {
            "grant": True,
            "preparation_id": PREPARATION_ID,
            "runtime_identity": self.identity.as_dict(),
            "run_id": getattr(self, "run_id", RUN_ID),
            "claim_lock": CLAIM_LOCK,
            "worker_context": self.worker_context() if context is None else context,
        }
        value.update(overrides)
        return value

    def environment(self, **overrides):
        values = {
            "HERMES_KANBAN_RUNTIME_GRANTED": "1",
            "HERMES_KANBAN_TASK": getattr(self, "task_id", TASK_ID),
            "HERMES_KANBAN_RUN_ID": str(getattr(self, "run_id", RUN_ID)),
            "HERMES_PROFILE": PROFILE,
            "HERMES_KANBAN_BOARD": BOARD,
            "HERMES_KANBAN_WORKSPACE": str(self.workspace),
            "HERMES_KANBAN_CLAIM_LOCK": CLAIM_LOCK,
            "HERMES_DELEGATED_CHILD_CONTEXT": "",
        }
        values.update(overrides)
        return mock.patch.dict(os.environ, values)

    def retain(self):
        """Write the matching board state and retain the granted capability."""
        self.write_worker_state()
        with self.environment():
            worker_startup._retain_worker_grant(self.grant(), self.identity)

    def refusal(self):
        return self.assertRaises(worker_startup.WorkerStartupUnavailable)


class GrantAuthorityTests(WorkerStartupFixture):
    def test_no_capability_before_a_grant(self):
        self.write_worker_state()
        with self.environment():
            self.assertFalse(worker_startup.worker_startup_available())
            with self.refusal():
                worker_startup.worker_startup_snapshot()

    def test_exact_snapshot_for_a_granted_worker(self):
        self.retain()
        with self.environment():
            self.assertTrue(worker_startup.worker_startup_available())
            snapshot = worker_startup.worker_startup_snapshot()
        self.assertEqual(
            snapshot,
            {
                "task_id": self.task_id,
                "run_id": self.run_id,
                "profile": PROFILE,
                "board": BOARD,
                "workspace_path": str(self.workspace),
                "workspace_kind": "scratch",
                "branch_name": "wt/fixture",
                "preparation_id": PREPARATION_ID,
                "runtime_identity": self.identity.as_dict(),
                "spawn_receipt": "pending",
                # A frame with no execution fields at all is diagnosable, not guessed at.
                "execution": {
                    "granted": False, "inspect": False, "recipes": [],
                    "refusal": "execution_unavailable",
                },
            },
        )
        with self.environment():
            self.assertIsNot(
                snapshot["runtime_identity"],
                worker_startup.worker_startup_snapshot()["runtime_identity"],
            )

    def test_identical_repeat_is_a_noop_and_a_different_grant_is_refused(self):
        self.write_worker_state()
        with self.environment():
            worker_startup._retain_worker_grant(self.grant(), self.identity)
            worker_startup._retain_worker_grant(self.grant(), self.identity)
            with self.refusal():
                worker_startup._retain_worker_grant(
                    self.grant(run_id=RUN_ID + 1), self.identity
                )
            self.assertEqual(
                worker_startup.worker_startup_snapshot()["run_id"], self.run_id
            )

    def test_grant_context_must_be_complete_and_exact(self):
        malformed = (
            {},
            {"task_id": TASK_ID, "profile": PROFILE, "board": BOARD,
             "workspace": str(self.workspace)},
            self.worker_context(extra="x"),
            self.worker_context(workspace="relative/workspace"),
            self.worker_context(workspace=str(self.directory / "missing")),
            self.worker_context(db_path="relative.db"),
            self.worker_context(db_path=str(self.directory / "missing.db")),
            self.worker_context(task_id=""),
            self.worker_context(board=None),
        )
        for context in malformed:
            with self.subTest(context=context), self.environment(), self.refusal():
                worker_startup._retain_worker_grant(self.grant(context=context), self.identity)
        for override in (
            {"run_id": 0}, {"run_id": "7"}, {"run_id": None},
            {"claim_lock": ""}, {"claim_lock": None},
            {"preparation_id": ""}, {"grant": False},
        ):
            with self.subTest(override=override), self.environment(), self.refusal():
                worker_startup._retain_worker_grant(self.grant(**override), self.identity)
        self.assertIsNone(worker_startup._GRANT)

    def test_grant_requires_a_sealed_identity_for_this_process(self):
        for actual in (
            replace(self.identity, generation="", dependency_fingerprint=""),
            replace(self.identity, pid=os.getpid() + 1),
            replace(self.identity, start_time=self.identity.start_time + 1),
            "not-an-identity",
        ):
            with self.subTest(actual=actual), self.environment(), self.refusal():
                worker_startup._retain_worker_grant(self.grant(), actual)

    def test_forged_environment_cannot_authorize_inspection(self):
        self.retain()
        forged = (
            {"HERMES_KANBAN_RUNTIME_GRANTED": ""},
            {"HERMES_KANBAN_RUNTIME_GRANTED": "0"},
            {"HERMES_KANBAN_TASK": "t_other"},
            {"HERMES_KANBAN_RUN_ID": str(RUN_ID + 1)},
            {"HERMES_PROFILE": "cto"},
            {"HERMES_KANBAN_BOARD": "other-board"},
            {"HERMES_KANBAN_WORKSPACE": str(self.directory)},
            {"HERMES_KANBAN_CLAIM_LOCK": "other-lock"},
            {"HERMES_DELEGATED_CHILD_CONTEXT": "1"},
        )
        for override in forged:
            with self.subTest(override=override), self.environment(**override), self.refusal():
                worker_startup.worker_startup_snapshot()

    def test_delegated_and_non_dispatcher_contexts_are_refused(self):
        from agent import delegation_context

        self.retain()
        with self.environment(), delegation_context.delegated_child_context("fixture"):
            self.assertFalse(worker_startup.worker_startup_available())
            with self.refusal():
                worker_startup.worker_startup_snapshot()
        with self.environment(), delegation_context.non_dispatcher_owned_context():
            self.assertFalse(worker_startup.worker_startup_available())
            with self.refusal():
                worker_startup.worker_startup_snapshot()

    def test_any_dispatcher_granted_profile_may_inspect(self):
        """The grant decides, not a fixed role list.

        ``kanban_db_dispatch._profile_exists_fn`` lets the dispatcher launch a task
        for any profile present on the home, so a hard-coded worker-role allowlist
        would hide this operation from supported workers. The granted profile must
        still match the board rows and the environment, which
        ``test_granted_profile_must_match_the_current_environment`` covers.
        """
        for profile in ("default", "researcher"):
            with self.subTest(profile=profile):
                self.new_fixture()
                with mock.patch.object(sys.modules[__name__], "PROFILE", profile):
                    self.write_worker_state()
                    with self.environment():
                        worker_startup._retain_worker_grant(self.grant(), self.identity)
                    with self.environment():
                        self.assertTrue(worker_startup.worker_startup_available())
                        snapshot = worker_startup.worker_startup_snapshot()
                self.assertEqual(snapshot["profile"], profile)

    def test_granted_profile_must_match_the_current_environment(self):
        self.retain()
        with self.environment():
            worker_startup._GRANT = replace(worker_startup._GRANT, profile="orchestrator")
            # Visibility is not authorization: the catalog check only proves a grant
            # exists, so the invocation re-checks it against this process.
            with self.refusal():
                worker_startup.worker_startup_snapshot()

    def test_unsupported_platform_refuses_before_any_io(self):
        self.retain()
        # Simulates a foreign platform to prove the refusal is host-independent
        # containment rather than macOS behaviour: the assertion is "an unsupported
        # platform refuses before any board I/O", which holds on every host.
        foreign = mock.patch.object(sys, "platform", "darwin")  # os-marker: ok — host-independent refusal, no macOS branch
        with self.environment(), foreign, \
                mock.patch.object(worker_startup.sqlite3, "connect") as connect:
            self.assertFalse(worker_startup.worker_startup_available())
            with self.refusal():
                worker_startup.worker_startup_snapshot()
        connect.assert_not_called()

    def test_changed_runtime_or_process_marker_refuses(self):
        self.retain()
        with self.environment(), \
                mock.patch.object(runtime, "runtime_identity",
                                  return_value=replace(self.identity, fingerprint="other")):
            with self.refusal():
                worker_startup.worker_startup_snapshot()
        with self.environment(), \
                mock.patch.object(runtime, "assert_runtime_import_root",
                                  side_effect=runtime.RuntimeIdentityError("mixed roots")):
            with self.refusal():
                worker_startup.worker_startup_snapshot()
        with self.environment(), \
                mock.patch.object(runtime, "process_start_time", return_value=1):
            with self.refusal():
                worker_startup.worker_startup_snapshot()


class BootstrapGrantPathTests(WorkerStartupFixture):
    def bootstrap_environment(self, **overrides):
        """Patch the bootstrap handshake environment for one call."""
        values = {
            "HERMES_KANBAN_BOOTSTRAP_PATH": str(self.bootstrap_path),
            "HERMES_KANBAN_PREPARATION_ID": PREPARATION_ID,
            "HERMES_KANBAN_EXPECTED_RUNTIME": runtime.encode_identity(self.identity),
            "HERMES_KANBAN_RUNTIME_GENERATION": str(self.directory / "generation"),
            "HERMES_KANBAN_BOOTSTRAP_WAIT": "1",
        }
        values.update(overrides)
        return mock.patch.dict(os.environ, values)

    def test_post_import_grant_path_retains_capability(self):
        self.write_worker_state()
        with self.environment(), self.bootstrap_environment(), \
                mock.patch.object(runtime, "_read_bootstrap_message",
                                  return_value=self.grant(run_id=self.run_id)):
            runtime._finish_worker_bootstrap_grant(
                self.identity, PREPARATION_ID, self.identity
            )
            self.assertTrue(worker_startup.worker_startup_available())
            self.assertEqual(
                worker_startup.worker_startup_snapshot()["task_id"], self.task_id
            )

    def test_direct_validated_grant_path_retains_capability(self):
        self.write_worker_state()
        with self.environment(), self.bootstrap_environment(), \
                mock.patch.object(runtime, "decode_identity", return_value=self.identity), \
                mock.patch.object(runtime, "_read_bootstrap_message",
                                  return_value=self.grant(run_id=self.run_id)):
            payload = runtime.worker_bootstrap_from_env()
            self.assertTrue(payload["ready"])
            self.assertTrue(worker_startup.worker_startup_available())
            self.assertEqual(
                worker_startup.worker_startup_snapshot()["run_id"], self.run_id
            )

    def test_continue_imports_and_no_wait_never_retain(self):
        self.write_worker_state()
        with self.environment(), self.bootstrap_environment(), \
                mock.patch.object(runtime, "decode_identity", return_value=self.identity), \
                mock.patch.object(runtime, "_read_bootstrap_message", return_value={
                    "continue_imports": True,
                    "preparation_id": PREPARATION_ID,
                    "runtime_identity": self.identity.as_dict(),
                }):
            runtime.worker_bootstrap_from_env()
        self.assertIsNone(worker_startup._GRANT)
        with self.environment(), \
                self.bootstrap_environment(HERMES_KANBAN_BOOTSTRAP_WAIT="0"), \
                mock.patch.object(runtime, "decode_identity", return_value=self.identity):
            runtime.worker_bootstrap_from_env()
        self.assertIsNone(worker_startup._GRANT)
        self.assertFalse(worker_startup.worker_startup_available())

    def test_absent_worker_context_authorizes_only_the_launch(self):
        self.write_worker_state()
        grant = self.grant(run_id=self.run_id)
        grant.pop("worker_context")
        with self.environment(), self.bootstrap_environment(), \
                mock.patch.object(runtime, "_read_bootstrap_message", return_value=grant):
            runtime._finish_worker_bootstrap_grant(
                self.identity, PREPARATION_ID, self.identity
            )
            self.assertFalse(worker_startup.worker_startup_available())
            with self.refusal():
                worker_startup.worker_startup_snapshot()


class SpawnContextTests(unittest.TestCase):
    def test_context_is_captured_from_resolved_values(self):
        with tempfile.TemporaryDirectory(prefix="worker-startup-spawn-") as raw:
            directory = Path(raw)
            workspace = directory / "workspace"
            workspace.mkdir()
            board = directory / "board.db"
            board.write_bytes(b"")
            self.assertEqual(
                worker_runtime._worker_startup_context(
                    "t_fixture", "reviewer", "board-fixture", str(workspace), str(board),
                ),
                {
                    "task_id": "t_fixture",
                    "profile": "reviewer",
                    "board": "board-fixture",
                    "workspace": str(workspace),
                    "db_path": str(board),
                },
            )
            for values in (
                ("t_fixture", "reviewer", "", str(workspace), str(board)),
                ("t_fixture", "reviewer", "board-fixture", "workspace", str(board)),
                ("t_fixture", "reviewer", "board-fixture", str(directory / "gone"),
                 str(board)),
                ("t_fixture", "reviewer", "board-fixture", str(workspace), "board.db"),
                ("t_fixture", "reviewer", "board-fixture", str(workspace),
                 str(directory / "gone")),
            ):
                with self.subTest(values=values):
                    self.assertIsNone(worker_runtime._worker_startup_context(*values))


class BoardProofTests(WorkerStartupFixture):
    def test_missing_and_corrupt_boards_refuse(self):
        self.retain()
        with self.environment(), self.refusal():
            worker_startup._GRANT = replace(
                worker_startup._GRANT, db_path=str(self.directory / "missing.db")
            )
            worker_startup.worker_startup_snapshot()
        self.new_fixture()
        self.retain()
        corrupt = self.directory / "corrupt.db"
        corrupt.write_bytes(b"not a database")
        with self.environment(), self.refusal():
            worker_startup._GRANT = replace(worker_startup._GRANT, db_path=str(corrupt))
            worker_startup.worker_startup_snapshot()

    def test_stale_or_replaced_claim_refuses(self):
        for statement in (
            "UPDATE task_runs SET ended_at = 5",
            "UPDATE task_runs SET status = 'done'",
            "UPDATE task_runs SET claim_lock = 'replacement'",
            "UPDATE tasks SET claim_lock = 'replacement'",
            "UPDATE tasks SET status = 'ready'",
            "UPDATE tasks SET worker_pid = 1",
            "UPDATE tasks SET current_run_id = 999",
            "UPDATE tasks SET workspace_path = 'relative/workspace'",
            "UPDATE tasks SET workspace_path = '/nonexistent/workspace'",
        ):
            with self.subTest(statement=statement):
                self.new_fixture()
                self.retain()
                self.mutate(statement)
                with self.environment(), self.refusal():
                    worker_startup.worker_startup_snapshot()

    def test_claim_time_provenance_must_match_the_grant(self):
        for metadata in (
            {},
            {**self.claim_metadata(), "preparation_id": "other"},
            {**self.claim_metadata(), "worker_pid": os.getpid() + 1},
            {**self.claim_metadata(), "worker_start_time": 1},
            {**self.claim_metadata(),
             "runtime_identity": {**self.identity.as_dict(), "fingerprint": "other"}},
            "{not json",
        ):
            with self.subTest(metadata=metadata):
                self.new_fixture()
                self.write_worker_state(metadata=metadata)
                with self.environment():
                    worker_startup._retain_worker_grant(self.grant(), self.identity)
                    with self.refusal():
                        worker_startup.worker_startup_snapshot()

    def test_spawn_receipt_pending_and_verified(self):
        self.retain()
        with self.environment():
            self.assertEqual(
                worker_startup.worker_startup_snapshot()["spawn_receipt"], "pending"
            )
        # A legacy PID-only receipt cannot authenticate a spawn by itself.
        self.append_event({"pid": self.identity.pid})
        with self.environment():
            self.assertEqual(
                worker_startup.worker_startup_snapshot()["spawn_receipt"], "pending"
            )
        self.append_event({
            "pid": self.identity.pid,
            "runtime_identity": self.identity.as_dict(),
            "preparation_id": PREPARATION_ID,
            "board": BOARD,
        })
        with self.environment():
            self.assertEqual(
                worker_startup.worker_startup_snapshot()["spawn_receipt"], "verified"
            )

    def test_conflicting_or_malformed_spawn_receipts_refuse(self):
        for payload in (
            {"pid": self.identity.pid + 1},
            {"pid": self.identity.pid, "runtime_identity": {"protocol": 1}},
            {"pid": self.identity.pid, "runtime_identity": self.identity.as_dict(),
             "preparation_id": "other"},
            {"pid": self.identity.pid, "runtime_identity": self.identity.as_dict(),
             "preparation_id": PREPARATION_ID, "board": "other-board"},
            {"pid": self.identity.pid,
             "runtime_identity": {**self.identity.as_dict(), "fingerprint": "other"},
             "preparation_id": PREPARATION_ID},
            "{not json",
            "[1, 2]",
        ):
            with self.subTest(payload=payload):
                self.new_fixture()
                self.retain()
                self.append_event(payload)
                with self.environment(), self.refusal():
                    worker_startup.worker_startup_snapshot()

    def test_inspection_is_read_only_and_launches_nothing(self):
        self.retain()
        before_bytes = self.board.read_bytes()

        def counts():
            connection = sqlite3.connect(self.board)
            try:
                return tuple(
                    connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
                    for table in ("tasks", "task_runs", "task_events")
                )
            finally:
                connection.close()

        before_counts = counts()
        with self.environment(), \
                mock.patch("subprocess.Popen", side_effect=AssertionError("subprocess")), \
                mock.patch("subprocess.run", side_effect=AssertionError("subprocess.run")), \
                mock.patch("os.system", side_effect=AssertionError("os.system")):
            worker_startup.worker_startup_snapshot()
        self.assertEqual(counts(), before_counts)
        self.assertEqual(self.board.read_bytes(), before_bytes)
        self.assertFalse((self.board.parent / f"{self.board.name}-journal").exists())

    def test_deleted_workspace_refuses(self):
        """A workspace removed mid-claim must not be reported as the granted one."""
        self.retain()
        self.workspace.rmdir()
        with self.environment(), self.refusal():
            worker_startup.worker_startup_snapshot()

    def test_snapshot_refresh_runs_only_for_a_granted_worker(self):
        calls = []
        with mock.patch(
            "tools.mcp_tool_agent.refresh_agent_mcp_tools",
            side_effect=lambda *args, **kwargs: calls.append("refresh"),
        ), mock.patch(
            "tools.mcp_tool_agent.reprobe_tool_availability",
            side_effect=lambda: calls.append("reprobe"),
        ):
            self.write_worker_state()
            runtime._publish_granted_worker_tools(object())
            self.assertEqual(calls, [])
            with self.environment():
                worker_startup._retain_worker_grant(self.grant(), self.identity)
                runtime._publish_granted_worker_tools(object())
        self.assertEqual(calls, ["reprobe", "refresh"])

    def test_oneshot_pipeline_prunes_before_publishing_and_pinning(self):
        """A rebuild must not restore tools a ``chat -q`` session pruned at build time."""
        from tools.registry import discover_builtin_tools
        from tools.mcp_tool_agent import refresh_agent_mcp_tools
        from types import SimpleNamespace
        from model_tools import get_tool_definitions

        discover_builtin_tools()
        pinned = {}

        class SessionDb:
            def update_session_tool_names(self, session_id, value):
                pinned.update(value)

        agent = SimpleNamespace(
            tools=[], valid_tool_names=set(), quiet_mode=True,
            enabled_toolsets=["kanban", "skills"], disabled_toolsets=None,
            _tool_snapshot_generation=0, _session_db=SessionDb(), session_id="fixture",
        )
        with self.environment(HERMES_SINGLE_QUERY_SESSION="1"):
            # Precondition: the rebuild WOULD offer the pruned tool without the filter.
            offered = {
                tool["function"]["name"]
                for tool in get_tool_definitions(
                    enabled_toolsets=["kanban", "skills"], quiet_mode=False,
                )
            }
            self.assertIn("skill_manage", offered)
            refresh_agent_mcp_tools(agent, quiet_mode=True, preserve_prefix=True)
            self.assertNotIn("skill_manage", agent.valid_tool_names)
            self.assertNotIn(
                "skill_manage",
                {tool["function"]["name"] for tool in pinned.get("tools", [])},
            )
            # A later refresh (compaction boundary calls the same helper) must not
            # resurrect it from the pinned/current surface.
            refresh_agent_mcp_tools(agent, quiet_mode=True, preserve_prefix=True)
            self.assertNotIn("skill_manage", agent.valid_tool_names)

    def test_refresh_failure_cannot_fail_a_granted_worker(self):
        self.retain()
        with self.environment(), mock.patch(
            "tools.mcp_tool_agent.refresh_agent_mcp_tools",
            side_effect=RuntimeError("refresh broke"),
        ):
            runtime._publish_granted_worker_tools(object())
        self.assertTrue(worker_startup.worker_startup_available())

    def test_board_errors_are_reported_as_refusals(self):
        self.retain()
        with self.environment(), \
                mock.patch.object(worker_startup.sqlite3, "connect",
                                  side_effect=sqlite3.OperationalError("locked")), \
                self.refusal():
            worker_startup.worker_startup_snapshot()


class ExecutionDiagnosticsTests(WorkerStartupFixture):
    """A withheld capability must be diagnosable without granting anything."""

    def execution_section(
        self, *, inspect=True, recipes=(), task_id=None, board=None,
        preparation_id=PREPARATION_ID, workspace=None,
    ):
        """A real bound authority section for this fixture's workspace."""
        from hermes_cli.worker_execution_common import ExecutionGrant

        directories = {}
        for name in ("execution-parent", "journal", "runtime-parent"):
            path = self.directory / name
            path.mkdir(exist_ok=True)
            info = os.lstat(path)
            directories[name] = {
                "path": str(path), "device": int(info.st_dev), "inode": int(info.st_ino),
            }
        workspace_path = Path(workspace or self.workspace)
        info = os.lstat(workspace_path)
        return ExecutionGrant(
            policy_digest="sha256:" + "0" * 64,
            task_id=task_id or self.task_id,
            profile=PROFILE,
            board=board or BOARD,
            preparation_id=preparation_id,
            workspace={
                "path": str(workspace_path),
                "device": int(info.st_dev), "inode": int(info.st_ino),
            },
            git=None,
            image_store=None,
            inspect=inspect,
            role="review",
            extra_source_paths=(),
            recipes=tuple(dict(recipe) for recipe in recipes),
            review=None,
            policy_plugin={},
            runtime={
                "generation": "fixture-generation",
                "dependency_fingerprint": "fixture-dependency-fingerprint",
                "code_sha": "fixture-code-sha",
            },
            execution_parent=directories["execution-parent"],
            journal_dir=directories["journal"],
            runtime_parent=directories["runtime-parent"],
        ).bind(self.run_id, CLAIM_LOCK)

    def retain_with(self, **frame):
        """Retain a frame for a live board; call after ``write_worker_state()``."""
        with self.environment():
            worker_startup._retain_worker_grant(self.grant(**frame), self.identity)
            return worker_startup.worker_startup_snapshot()["execution"]

    def test_inspect_only_authority_reports_granted_inspection(self):
        """Also the pre-existing frame shape: an authority section and no refusal key."""
        self.write_worker_state()
        execution = self.retain_with(worker_execution=self.execution_section(inspect=True))
        self.assertEqual(
            execution,
            {"granted": True, "inspect": True, "recipes": [], "refusal": None},
        )

    def test_withheld_inspection_is_reported_as_not_granted(self):
        self.write_worker_state()
        execution = self.retain_with(worker_execution=self.execution_section(inspect=False))
        self.assertEqual(
            execution,
            {"granted": True, "inspect": False, "recipes": [], "refusal": None},
        )

    def test_granted_recipe_is_named_without_its_argv_or_image(self):
        self.write_worker_state()
        recipe = {
            "id": "offline-verification",
            "image": "sha256:" + "1" * 64,
            "argv": ["python3", "-I", "/workspace/check.py"],
            "timeout_seconds": 30,
            "services": [],
        }
        execution = self.retain_with(
            worker_execution=self.execution_section(recipes=(recipe,)))
        self.assertEqual(
            execution,
            {
                "granted": True, "inspect": True,
                "recipes": ["offline-verification"], "refusal": None,
            },
        )
        encoded = json.dumps(execution)
        for detail in ("/workspace/check.py", "1" * 64, "timeout_seconds", "services"):
            self.assertNotIn(detail, encoded)

    def test_withheld_authority_reports_its_refusal_code(self):
        for code in (
            "policy_not_configured", "candidate_workspace_unavailable",
            "workspace_ambiguous", "inspection_disabled", "resolution_failed",
        ):
            with self.subTest(code=code):
                self.new_fixture()
                self.write_worker_state()
                execution = self.retain_with(worker_execution_refusal=code)
                self.assertEqual(
                    execution,
                    {
                        "granted": False, "inspect": False, "recipes": [],
                        "refusal": code,
                    },
                )

    def test_absent_and_null_refusals_do_not_invent_a_configuration_diagnosis(self):
        self.retain()
        with self.environment():
            self.assertEqual(
                worker_startup.worker_startup_snapshot()["execution"]["refusal"],
                "execution_unavailable",
            )
        self.new_fixture()
        self.write_worker_state()
        self.assertEqual(
            self.retain_with(worker_execution_refusal=None)["refusal"],
            "execution_unavailable",
        )

    def test_dispatcher_null_refusal_beside_a_grant_is_not_a_refusal(self):
        """The dispatcher sends the field on every frame; null beside authority means granted.

        Real dispatcher proof caught this: reading an explicit null as a refusal made
        the retained grant refuse itself and dropped the execution tools.
        """
        self.write_worker_state()
        execution = self.retain_with(
            worker_execution=self.execution_section(inspect=True),
            worker_execution_refusal=None,
        )
        self.assertEqual(
            execution,
            {"granted": True, "inspect": True, "recipes": [], "refusal": None},
        )

    def test_authority_and_refusal_together_refuse(self):
        self.write_worker_state()
        with self.environment(), self.refusal():
            worker_startup._retain_worker_grant(
                self.grant(
                    worker_execution=self.execution_section(),
                    worker_execution_refusal="policy_not_configured",
                ),
                self.identity,
            )
        self.assertIsNone(worker_startup._GRANT)

    def test_malformed_refusal_fields_refuse(self):
        for value in (
            "Policy_Not_Configured", "resolution_failed: /home/x/secret",
            "", "a" * 65, "with space", "../etc/passwd", "ünicode", 7,
            ["policy_not_configured"],
        ):
            with self.subTest(value=value):
                self.new_fixture()
                self.write_worker_state()
                with self.environment(), self.refusal():
                    worker_startup._retain_worker_grant(
                        self.grant(worker_execution_refusal=value), self.identity)
                self.assertIsNone(worker_startup._GRANT)

    def test_authority_bound_elsewhere_is_refused_even_with_a_matching_digest(self):
        """The digest is unkeyed, so retained-context checks must refuse a rebuilt section."""
        cases = (
            {"task_id": "t_someone_else"},
            {"board": "another-board"},
            {"preparation_id": "another-preparation"},
        )
        for overrides in cases:
            with self.subTest(overrides=overrides):
                self.new_fixture()
                self.write_worker_state()
                with self.environment(), self.refusal():
                    worker_startup._retain_worker_grant(
                        self.grant(worker_execution=self.execution_section(**overrides)),
                        self.identity,
                    )

    def test_bootstrap_loses_the_capability_instead_of_crashing(self):
        """The bootstrap catches only WorkerStartupUnavailable, so this must arrive as one."""
        self.write_worker_state()
        with self.environment():
            runtime._retain_worker_startup_grant(
                self.grant(
                    worker_execution=self.execution_section(task_id="t_someone_else")),
                self.identity,
            )
            self.assertIsNone(worker_startup._GRANT)
            self.assertFalse(worker_startup.worker_startup_available())

    def test_environment_cannot_forge_the_diagnostic(self):
        self.retain()
        with self.environment(
            HERMES_WORKER_EXECUTION="1",
            HERMES_WORKER_EXECUTION_RECIPE="offline-verification",
            HERMES_KANBAN_EXECUTION_INSPECT="1",
            HERMES_KANBAN_WORKER_EXECUTION='{"inspect": true}',
        ):
            execution = worker_startup.worker_startup_snapshot()["execution"]
        self.assertEqual(
            execution,
            {
                "granted": False, "inspect": False, "recipes": [],
                "refusal": "execution_unavailable",
            },
        )


class ExecutionRefusalCodeTests(unittest.TestCase):
    """Only a stable lowercase code may cross the dispatcher's grant pipe."""

    def test_only_the_code_crosses(self):
        code = worker_runtime._execution_refusal_code
        self.assertEqual(code(None), "policy_not_configured")
        self.assertEqual(code("workspace_ambiguous"), "workspace_ambiguous")
        self.assertEqual(
            code("candidate_workspace_unavailable: head deadbeef"), "candidate_workspace_unavailable",
        )
        self.assertEqual(
            code("execution_directories_unavailable: [Errno 13] /home/x/.cache"),
            "execution_directories_unavailable",
        )
        self.assertEqual(
            code("resolution_failed: RuntimeError: /home/x/secret"), "resolution_failed",
        )

    def test_anything_else_becomes_resolution_failed(self):
        code = worker_runtime._execution_refusal_code
        for detail in ("Policy_Not_Configured", "", "with space", "../etc/passwd", "a" * 65, "ünicode"):
            with self.subTest(detail=detail):
                self.assertEqual(code(detail), "resolution_failed")


if __name__ == "__main__":
    unittest.main()

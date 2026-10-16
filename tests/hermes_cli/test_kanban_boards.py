"""Tests for the multi-board kanban layer (``hermes kanban boards …``).

Covers the pieces added when boards became a first-class concept:

* Slug validation and normalisation.
* Path resolution for ``default`` (legacy ``<root>/kanban.db``) vs
  named boards (``<root>/kanban/boards/<slug>/kanban.db``).
* Current-board persistence via ``<root>/kanban/current`` and
  ``HERMES_KANBAN_BOARD`` env var.
* ``connect(board=)`` isolation — writes on one board don't leak.
* ``create_board`` / ``list_boards`` / ``remove_board`` round trip.
* CLI surface: ``hermes kanban boards list/create/switch/rm``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

# Ensure the worktree (not the stale global clone) is first on sys.path.
_WORKTREE = Path(__file__).resolve().parents[2]
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def fresh_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with no prior kanban state.

    The autouse hermetic conftest already nukes credentials + TZ; this
    fixture layers a per-test HERMES_HOME plus a path-init cache reset
    so each test sees a truly empty board set.
    """
    home = tmp_path / "hermes_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    for var in (
        "HERMES_KANBAN_DB",
        "HERMES_KANBAN_WORKSPACES_ROOT",
        "HERMES_KANBAN_HOME",
        "HERMES_KANBAN_BOARD",
    ):
        monkeypatch.delenv(var, raising=False)
    # Also reset hermes_constants cache so get_default_hermes_root() re-reads.
    try:
        import hermes_constants
        hermes_constants._cached_default_hermes_root = None  # type: ignore[attr-defined]
    except Exception:
        pass
    # Kanban module-level init cache must not leak between tests.
    kb._INITIALIZED_PATHS.clear()
    return home


# ---------------------------------------------------------------------------
# Slug validation
# ---------------------------------------------------------------------------

class TestSlugValidation:
    @pytest.mark.parametrize("good", [
        "default", "atm10-server", "hermes-agent", "proj_1", "a",
        "very-long-but-still-ok-slug-with-hyphens-and-numbers-1234",
    ])
    def test_accepts_valid(self, good):
        assert kb._normalize_board_slug(good) == good


    def test_empty_returns_none(self):
        assert kb._normalize_board_slug(None) is None
        assert kb._normalize_board_slug("") is None
        assert kb._normalize_board_slug("   ") is None


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

class TestPathResolution:
    def test_default_board_legacy_path(self, fresh_home):
        """The default board's DB lives at ``<root>/kanban.db`` for back-compat."""
        assert kb.kanban_db_path() == fresh_home / "kanban.db"
        assert kb.kanban_db_path(board="default") == fresh_home / "kanban.db"

    def test_named_board_under_boards_dir(self, fresh_home):
        p = kb.kanban_db_path(board="atm10-server")
        assert p == fresh_home / "kanban" / "boards" / "atm10-server" / "kanban.db"


    def test_env_var_db_override_still_wins(self, fresh_home, tmp_path, monkeypatch):
        """``HERMES_KANBAN_DB`` pins the file regardless of ``board=`` arg for every
        execution the dispatcher fences (the 5ec6baa multi-boards isolation: workers
        physically cannot see other boards): its dispatched workers (``HERMES_KANBAN_TASK``)
        and delegated children / descendants (``HERMES_DELEGATED_CHILD_CONTEXT``). An
        explicit board that outranked the pin would also escape
        ``kanban_path_is_fenced``, which checks the pinned path / fenced root. Outside
        those fences an explicit board is the caller's own intent and wins (see
        ``test_explicit_board_trumps_env_var_db_override`` below)."""
        forced = tmp_path / "custom.db"
        monkeypatch.setenv("HERMES_KANBAN_DB", str(forced))
        assert kb.kanban_db_path() == forced
        assert kb.kanban_db_path(board=None) == forced
        # Dispatched worker identity: the pin still fences explicit board intent.
        monkeypatch.setenv("HERMES_KANBAN_TASK", "t_fence_probe")
        assert kb.kanban_db_path(board="ignored") == forced
        # Delegated children / spawned descendants are fenced the same way.
        from agent.delegation_context import DELEGATED_CHILD_ENV_MARKER
        monkeypatch.setenv(DELEGATED_CHILD_ENV_MARKER, "1")
        assert kb.kanban_db_path(board="ignored") == forced

    def test_env_var_db_override_wins_when_board_not_passed(self, fresh_home, tmp_path, monkeypatch):
        """``HERMES_KANBAN_DB`` pins the file when no explicit ``board=`` is given
        (back-compat for dispatcher-spawned workers with no board override)."""
        forced = tmp_path / "custom.db"
        monkeypatch.setenv("HERMES_KANBAN_DB", str(forced))
        assert kb.kanban_db_path() == forced
        assert kb.kanban_db_path(board=None) == forced

    def test_explicit_board_trumps_env_var_db_override(self, fresh_home, tmp_path, monkeypatch):
        """Documented priority (module docstring, predates this test): explicit
        ``board=`` arg > ``HERMES_KANBAN_BOARD`` > ``HERMES_KANBAN_DB`` > current >
        default — for UNFENCED callers. An explicit ``board=`` must resolve to that
        board's own path even when ``HERMES_KANBAN_DB`` pins a different file: this is
        what makes cross-board ``kanban_create(board=...)`` / ``kanban_show(board=...)``
        work from a user-facing session instead of silently landing on the pinned board
        (t_3f1c63a5). Fenced callers (dispatched workers, delegated children) keep the
        pinned path — see ``test_env_var_db_override_still_wins`` above."""
        forced = tmp_path / "custom.db"
        monkeypatch.setenv("HERMES_KANBAN_DB", str(forced))
        p = kb.kanban_db_path(board="atm10-server")
        assert p == fresh_home / "kanban" / "boards" / "atm10-server" / "kanban.db"
        assert p != forced


# ---------------------------------------------------------------------------
# Current-board resolution
# ---------------------------------------------------------------------------

class TestCurrentBoard:



    def test_stale_file_pointer_falls_back_to_default(self, fresh_home):
        current = fresh_home / "kanban" / "current"
        current.parent.mkdir(parents=True, exist_ok=True)
        current.write_text("missing-board\n", encoding="utf-8")

        assert kb.get_current_board() == "default"
        assert not kb.board_exists("missing-board")
        assert [b["slug"] for b in kb.list_boards()] == ["default"]



    def test_kanban_db_path_reads_current(self, fresh_home):
        """kanban_db_path() with no args respects the on-disk pointer."""
        kb.create_board("my-proj")
        kb.set_current_board("my-proj")
        expected = fresh_home / "kanban" / "boards" / "my-proj" / "kanban.db"
        assert kb.kanban_db_path() == expected


# ---------------------------------------------------------------------------
# Board CRUD
# ---------------------------------------------------------------------------

class TestBoardCRUD:






    @pytest.mark.parametrize("archive", [True, False])
    def test_remove_clears_init_cache_for_recreated_db(self, fresh_home, archive):
        # Regression for #23833: poll loops that call connect(board=slug) right
        # after remove_board() recreate an empty kanban.db at the same path
        # (connect() does mkdir(exist_ok=True)). If _INITIALIZED_PATHS still
        # contains the resolved path, the CREATE TABLE pass is skipped and
        # downstream readers hit `no such table: task_events`.
        kb.create_board("recycle")
        # First connect populates _INITIALIZED_PATHS for this DB.  Use
        # connect_closing: `with connect() as conn` does NOT close the fd, and
        # on Windows an open connection locks kanban.db so remove_board's
        # rename below fails with WinError 5/32.
        with kbc.connect_closing(board="recycle") as conn:
            kb.create_task(conn, title="t1", assignee="dev")
        db_path = kb.board_dir("recycle") / "kanban.db"
        assert str(db_path.resolve()) in kb._INITIALIZED_PATHS

        kb.remove_board("recycle", archive=archive)
        # remove_board must drop the cache entry so a re-create through
        # connect() gets a fresh schema-init pass.
        assert str(db_path.resolve()) not in kb._INITIALIZED_PATHS

        # Simulate the event-stream poll: re-open the same slug. connect()
        # recreates the directory + empty .db; the schema must be re-applied.
        with kbc.connect_closing(board="recycle") as conn:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
        assert "task_events" in tables
        assert "tasks" in tables

    @pytest.mark.parametrize("archive", [True, False])
    def test_remove_waits_for_dispatch_and_active_claims(self, fresh_home, monkeypatch, archive):
        kb.create_board("busy")
        kb.set_current_board("busy")
        with kbc.connect_closing(board="busy") as conn:
            task_id = kb.create_task(conn, title="owned work", assignee="dev")
        with kbc._dispatch_admission_lock() as held:
            assert held
            with pytest.raises(ValueError, match="admission"):
                kb.remove_board("busy", archive=archive)
        with kbc.connect_closing(board="busy") as conn:
            claimed = kb.claim_task(conn, task_id)
            assert claimed is not None
        kb.create_board("unrelated")
        monkeypatch.setenv("HERMES_KANBAN_DB", str(kb.kanban_db_path(board="unrelated")))
        with kb.pin_first_board_resolution(), pytest.raises(ValueError, match="active claims"):
            kb.remove_board("busy", archive=archive)
        assert kb.board_exists("busy")
        assert kb.get_current_board() == "busy"
        with kbc.connect_closing(board="busy") as conn:
            assert kb.get_task(conn, task_id).current_run_id == claimed.current_run_id
            assert kb.complete_task(
                conn, task_id, expected_run_id=claimed.current_run_id,
                summary="owned work finished",
            )
        result = kb.remove_board("busy", archive=archive)
        assert result["action"] == ("archived" if archive else "deleted")
        assert not kb.board_exists("busy")
        assert kb.get_current_board() == "default"

    @pytest.mark.platforms("posix")
    @pytest.mark.parametrize("action,archive,recreate,backing", [
        ("ready", True, False, "file"),
        ("review", True, True, "file"),
        ("ready", False, True, "file"),
        ("dispatch", True, True, "file"),
        ("ready", True, False, "db-symlink"),
        ("dispatch", True, False, "dir-symlink"),
        ("ready", False, False, "dir-symlink"),
    ])
    def test_removed_connection_cannot_claim_old_board(
        self, fresh_home, monkeypatch, action, archive, recreate, backing,
    ):
        from hermes_cli import kanban_db_dispatch as dispatch

        kb.create_board("stale")
        kb.set_current_board("stale")
        external = fresh_home.parent / "external"
        if backing == "db-symlink":
            external.mkdir()
            database = kb.kanban_db_path(board="stale")
            database.rename(external / "kanban.db")
            database.symlink_to(external / "kanban.db")
        elif backing == "dir-symlink":
            directory = kb.board_dir("stale")
            directory.rename(external)
            directory.symlink_to(external, target_is_directory=True)
        with kbc.connect_closing(board="stale") as old:
            task_id = kb.create_task(old, title="old board", assignee="dev")
            lane = "review" if action == "review" else "ready"
            old.execute("UPDATE tasks SET status = ? WHERE id = ?", (lane, task_id))
            kb.remove_board("stale", archive=archive)
            assert kb.get_current_board() == "default"
            if recreate:
                kb.create_board("stale")
            calls = []
            if action == "dispatch":
                monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: True)
                monkeypatch.setattr(dispatch, "_memory_pressure_level", lambda: "ok")
                result = dispatch.dispatch_once(
                    old, board="stale", reconcile_orphans=False,
                    spawn_fn=lambda task, workspace, **kw: calls.append(task.id),
                )
                assert result.skipped_locked and not result.spawned
            else:
                claim = kb.claim_review_task if action == "review" else kb.claim_task
                assert claim(old, task_id) is None
            assert calls == []
            assert kb.get_task(old, task_id).status == lane
            assert old.execute("SELECT COUNT(*) FROM task_runs").fetchone()[0] == 0
        if backing != "file":
            assert (external / "kanban.db").is_file()
        if recreate:
            with kbc.connect_closing(board="stale") as new:
                assert kb.get_task(new, task_id) is None
                replacement = kb.create_task(new, title="new board", assignee="dev")
                assert kb.claim_task(new, replacement) is not None
        else:
            assert not kb.board_exists("stale")

    def test_failed_delete_preserves_selection_and_init_cache(self, fresh_home, monkeypatch):
        import shutil

        kb.create_board("keep")
        kb.set_current_board("keep")
        database = kb.kanban_db_path(board="keep")
        cache_key = str(database.resolve())
        assert cache_key in kb._INITIALIZED_PATHS

        def refuse_delete(*args, **kwargs):
            raise PermissionError("fixture deletion refused")

        monkeypatch.setattr(shutil, "rmtree", refuse_delete)
        with pytest.raises(PermissionError, match="fixture deletion refused"):
            kb.remove_board("keep", archive=False)
        assert kb.get_current_board() == "keep"
        assert cache_key in kb._INITIALIZED_PATHS
        assert database.is_file()

    def test_remove_does_not_initialize_a_missing_database(self, fresh_home):
        kb.create_board("unknown")
        kb.set_current_board("unknown")
        database = kb.kanban_db_path(board="unknown")
        database.unlink()
        with pytest.raises(ValueError, match="inspect"):
            kb.remove_board("unknown")
        assert kb.board_exists("unknown")
        assert kb.get_current_board() == "unknown"
        assert not database.exists()




# ---------------------------------------------------------------------------
# Connection isolation
# ---------------------------------------------------------------------------

class TestConnectionIsolation:
    def test_tasks_do_not_leak_across_boards(self, fresh_home):
        kb.create_board("alpha")
        kb.create_board("beta")

        with kbc.connect(board="alpha") as conn:
            kb.create_task(conn, title="alpha-task-1", assignee="dev")
            kb.create_task(conn, title="alpha-task-2", assignee="dev")

        with kbc.connect(board="beta") as conn:
            kb.create_task(conn, title="beta-only", assignee="dev")

        with kbc.connect(board="alpha") as conn:
            a = kb.list_tasks(conn)
        with kbc.connect(board="beta") as conn:
            b = kb.list_tasks(conn)
        with kbc.connect(board="default") as conn:
            d = kb.list_tasks(conn)

        assert {t.title for t in a} == {"alpha-task-1", "alpha-task-2"}
        assert {t.title for t in b} == {"beta-only"}
        assert d == []

    def test_connect_without_args_uses_current(self, fresh_home):
        kb.create_board("curr")
        kb.set_current_board("curr")
        with kbc.connect() as conn:
            kb.create_task(conn, title="implicit", assignee="x")
        with kbc.connect(board="curr") as conn:
            tasks = kb.list_tasks(conn)
        assert [t.title for t in tasks] == ["implicit"]

    def test_connect_env_var_overrides_current(self, fresh_home, monkeypatch):
        kb.create_board("persist")
        kb.create_board("envwin")
        kb.set_current_board("persist")
        monkeypatch.setenv("HERMES_KANBAN_BOARD", "envwin")
        with kbc.connect() as conn:
            kb.create_task(conn, title="via-env", assignee="x")
        with kbc.connect(board="envwin") as conn:
            assert [t.title for t in kb.list_tasks(conn)] == ["via-env"]
        with kbc.connect(board="persist") as conn:
            assert kb.list_tasks(conn) == []




# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------

def _cli(args: list[str], env_extra: dict | None = None) -> subprocess.CompletedProcess:
    """Run ``hermes kanban …`` with PYTHONPATH pinned to the worktree."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_WORKTREE)
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, "-m", "hermes_cli.main", "kanban"] + args,
        env=env,
        capture_output=True,
        text=True,
        cwd=str(_WORKTREE),
        timeout=30,
    )


class TestCLI:


    def test_per_board_task_isolation_via_cli(self, tmp_path):
        env = {"HERMES_HOME": str(tmp_path)}
        assert _cli(["boards", "create", "projA"], env_extra=env).returncode == 0
        assert _cli(["boards", "create", "projB"], env_extra=env).returncode == 0

        # Create one task on each via --board.
        r = _cli(["--board", "projA", "create", "Task A", "--assignee", "dev"], env_extra=env)
        assert r.returncode == 0, r.stderr
        r = _cli(["--board", "projB", "create", "Task B", "--assignee", "dev"], env_extra=env)
        assert r.returncode == 0, r.stderr

        # list on each board only shows its own.
        listA = _cli(["--board", "projA", "list", "--json"], env_extra=env)
        listB = _cli(["--board", "projB", "list", "--json"], env_extra=env)
        listD = _cli(["list", "--json"], env_extra=env)

        titlesA = [t["title"] for t in json.loads(listA.stdout)]
        titlesB = [t["title"] for t in json.loads(listB.stdout)]
        titlesD = [t["title"] for t in json.loads(listD.stdout)]

        assert titlesA == ["Task A"]
        assert titlesB == ["Task B"]
        assert titlesD == []



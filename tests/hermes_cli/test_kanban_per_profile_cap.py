"""Regression tests for #21582 — per-profile concurrency cap in dispatcher.

When ``kanban.max_in_progress_per_profile`` is set, no single profile
gets more than N workers running at once even if the global
``max_in_progress`` cap would allow it. Prevents one profile's local
model / API quota / browser pool from being overwhelmed by a fan-out.
"""
from __future__ import annotations

import multiprocessing
import os
import sqlite3
from pathlib import Path
from contextlib import closing

import pytest


@pytest.fixture()
def isolated_kanban_home_with_profiles(tmp_path, monkeypatch):
    """Fresh shared root; no native scopes or worker payloads."""
    test_home = tmp_path / ".hermes"
    for prof in ("alpha", "beta", "default"):
        profile = test_home / "profiles" / prof
        profile.mkdir(parents=True)
        (profile / "config.yaml").write_text("{}\n")
    monkeypatch.setenv("HERMES_HOME", str(test_home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(test_home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    from hermes_cli import kanban_db
    from hermes_cli import kanban_db_dispatch

    monkeypatch.setattr(kanban_db_dispatch, "_memory_pressure_level", lambda: "ok")
    monkeypatch.setattr(kanban_db_dispatch, "reconcile_orphan_worker_scopes", lambda *a, **k: [])
    return kanban_db


def _fake_spawn(*args, **kwargs):
    return 12345




def test_cap_2_balances_two_profiles(isolated_kanban_home_with_profiles):
    """With cap=2: 2 alpha + 2 beta dispatched; remaining 3 alpha + 1 beta
    deferred to skipped_per_profile_capped."""
    kb = isolated_kanban_home_with_profiles
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        for i in range(5):
            kb.create_task(conn, title=f"a{i}", assignee="alpha")
        for i in range(3):
            kb.create_task(conn, title=f"b{i}", assignee="beta")
    with kbc.connect_closing() as conn:
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn, dry_run=True,
            max_in_progress_per_profile=2,
        )
    spawn_assignees = [s[1] for s in res.spawned]
    capped_assignees = [c[1] for c in res.skipped_per_profile_capped]
    assert spawn_assignees.count("alpha") == 2
    assert spawn_assignees.count("beta") == 2
    assert capped_assignees.count("alpha") == 3
    assert capped_assignees.count("beta") == 1




def test_capped_tasks_dispatched_on_subsequent_tick(isolated_kanban_home_with_profiles):
    """A task deferred this tick because its profile was at cap should be
    eligible for dispatch on the next tick (after running tasks complete).
    This verifies the cap is per-tick state, not a permanent block."""
    kb = isolated_kanban_home_with_profiles
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        ids = [kb.create_task(conn, title=f"a{i}", assignee="alpha") for i in range(3)]

    # First tick: cap=1, only 1 alpha dispatched
    with kbc.connect_closing() as conn:
        res1 = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn, dry_run=False,
            max_in_progress_per_profile=1,
        )
    assert len(res1.spawned) == 1
    assert len(res1.skipped_per_profile_capped) == 2

    # Simulate the running task completing — set it back to done so the
    # 'running' count drops
    spawned_id = res1.spawned[0][0]
    with kbc.connect_closing() as conn:
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status = 'done', claim_lock = NULL WHERE id = ?",
                (spawned_id,),
            )

    # Second tick: 1 more alpha should now dispatch
    with kbc.connect_closing() as conn:
        res2 = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn, dry_run=False,
            max_in_progress_per_profile=1,
        )
    assert len(res2.spawned) == 1
    assert len(res2.skipped_per_profile_capped) == 1
    assert res2.spawned[0][0] != spawned_id  # different task this time



@pytest.mark.platforms("posix")
@pytest.mark.parametrize("failure", [
    None, "lock_missing", "lock_denied", "inventory_denied", "database_unreadable",
    "database_missing", "dangling_alias", "hardlink_alias",
])
def test_shared_root_profile_admission_and_release(
    isolated_kanban_home_with_profiles, monkeypatch, failure,
):
    """Pins/aliases/archiving cannot hide or multiply another board's worker."""
    kb = isolated_kanban_home_with_profiles
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    from hermes_cli import sqlite_safe_read

    kb.create_board("board-a")
    kb.create_board("board-b")
    with kbc.connect_closing(board="board-a") as first:
        running = kb.create_task(first, title="running alpha", assignee="alpha")
        assert kb.claim_task(first, running) is not None
    # Simulate a legacy archive created before active-board removal was guarded.
    archived = kb.boards_root() / "_archived" / "board-a-legacy"
    archived.parent.mkdir()
    kb.board_dir("board-a").rename(archived)
    first_path = archived / "kanban.db"
    alias_dir = kb.boards_root() / "alias"
    alias_dir.mkdir()
    (alias_dir / "kanban.db").symlink_to(first_path)
    # Gateway-style pin-first resolution maps every slug to board B. Accounting
    # must still inspect physical A, counting each DB once, including B itself.
    monkeypatch.setenv("HERMES_KANBAN_DB", str(kb.kanban_db_path(board="board-b")))
    (kb.kanban_home() / "config.yaml").write_text(
        "kanban:\n  max_in_progress: 2\n  max_in_progress_per_profile: 1\n",
    )
    with kb.pin_first_board_resolution(), kbc.connect_closing(board="board-b") as second:
        alpha = kb.create_task(second, title="waiting alpha", assignee="alpha", priority=10)
        beta = kb.create_task(second, title="waiting beta", assignee="beta")
        calls = []

        def dispatch(*, dry_run=True):
            return kbd.dispatch_once(
                second, board="board-b",
                dry_run=dry_run, reconcile_orphans=False,
                spawn_fn=lambda task, workspace, board=None: calls.append(task.id),
                # Omitted caps must resolve config for daemon/dashboard.
            )

        if failure:
            with monkeypatch.context() as patch:
                if failure.startswith("lock_"):
                    real_open = Path.open

                    def unavailable_lock(path, *args, **kwargs):
                        if path.name == ".dispatch-admission.lock":
                            error = FileNotFoundError if failure == "lock_missing" else PermissionError
                            raise error("admission lock unavailable")
                        return real_open(path, *args, **kwargs)

                    patch.setattr(Path, "open", unavailable_lock)
                elif failure == "inventory_denied":
                    real_iterdir = Path.iterdir

                    def unavailable_inventory(path):
                        if path == kb.boards_root():
                            raise PermissionError("board inventory unavailable")
                        return real_iterdir(path)

                    patch.setattr(Path, "iterdir", unavailable_inventory)
                elif failure == "database_unreadable":
                    real_connect = sqlite_safe_read.connect_tracked

                    def unavailable_database(path, **kwargs):
                        if kwargs.get("tracking_path") == first_path.resolve():
                            raise sqlite3.OperationalError("unable to open database file")
                        return real_connect(path, **kwargs)

                    patch.setattr(sqlite_safe_read, "connect_tracked", unavailable_database)
                elif failure == "database_missing":
                    real_lstat = Path.lstat

                    def missing_database(path, *args, **kwargs):
                        if path == first_path:
                            raise FileNotFoundError("known board database missing")
                        return real_lstat(path, *args, **kwargs)

                    patch.setattr(Path, "lstat", missing_database)
                else:
                    bad_alias = kb.boards_root() / "bad-alias"
                    bad_alias.mkdir()
                    if failure == "hardlink_alias":
                        os.link(first_path, bad_alias / "kanban.db")
                    else:
                        (bad_alias / "kanban.db").symlink_to(bad_alias / "missing.db")
                try:
                    refused = dispatch(dry_run=False)
                    assert refused.spawned == []
                    assert calls == []
                    assert kb.get_task(second, alpha).claim_lock is None
                    assert kb.get_task(second, beta).claim_lock is None
                    assert refused.skipped_locked == failure.startswith("lock_")
                finally:
                    if failure in {"dangling_alias", "hardlink_alias"}:
                        (bad_alias / "kanban.db").unlink()
                        bad_alias.rmdir()

        eligible = dispatch()
        assert eligible.skipped_per_profile_capped == [(alpha, "alpha", 1)]
        assert [task for task, *_ in eligible.spawned] == [beta]
        admitted = dispatch(dry_run=False)
        assert [task for task, *_ in admitted.spawned] == [beta]
        assert calls == [beta]
        assert dispatch().spawned == []  # Total 2, even across boards.

        # Finish A and close its connection before the next capacity snapshot.
        with closing(sqlite3.connect(first_path, isolation_level=None)) as first:
            first.execute(
                "UPDATE tasks SET status = 'done', claim_lock = NULL WHERE id = ?", (running,),
            )
        released = dispatch(dry_run=False)
        assert [task for task, *_ in released.spawned] == [alpha]
        assert released.skipped_per_profile_capped == []
        assert calls == [beta, alpha]
        assert kb.get_task(second, alpha).status == "running"


def _competing_dispatch(home, board, hold_preparation, prepared, release, output):
    """A real dispatcher process with only the native launch replaced."""
    os.environ["HERMES_HOME"] = home
    os.environ["HERMES_KANBAN_HOME"] = home
    os.environ.pop("HERMES_KANBAN_DB", None)
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    from hermes_cli.kanban_runtime import prospective_identity

    grants = []

    def prepare(task, workspace, **kwargs):
        if hold_preparation:
            prepared.set()
            assert release.wait(30), "parent did not release worker preparation"
        identity = prospective_identity()
        return kbd.WorkerLaunch(
            pid=identity.pid, runtime_identity=identity.as_dict(), preparation_id="test-admission",
            grant=lambda *args: grants.append(task.id), cancel=lambda: None,
        )

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(kbd, "_memory_pressure_level", lambda: "ok")
        patch.setattr(kbd, "_require_autonomous_worker_capacity", lambda: None)
        patch.setattr(kbd, "reconcile_orphan_worker_scopes", lambda *a, **k: [])
        patch.setattr(kbd, "_default_spawn", prepare)
        with kbc.connect_closing(board=board) as conn:
            result = kbd.dispatch_once(
                conn, board=board, max_in_progress=2, max_in_progress_per_profile=1,
                reconcile_orphans=False,
            )
        output.put((board, result.skipped_locked, result.spawned, grants))


def test_cross_process_gate_covers_preparation_until_claim_and_grant(
    isolated_kanban_home_with_profiles,
):
    """A different board cannot grant alpha while A prepares its unclaimed task."""
    kb = isolated_kanban_home_with_profiles
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd

    tasks = {}
    for board in ("board-a", "board-b"):
        kb.create_board(board)
        with kbc.connect_closing(board=board) as conn:
            tasks[board] = kb.create_task(conn, title=board, assignee="alpha")
    ctx = multiprocessing.get_context("spawn")
    prepared, release, output = ctx.Event(), ctx.Event(), ctx.Queue()
    processes = [
        ctx.Process(
            target=_competing_dispatch,
            args=(str(kb.kanban_home()), board, board == "board-a", prepared, release, output),
        )
        for board in tasks
    ]
    try:
        processes[0].start()
        assert prepared.wait(30), "first dispatcher never reached preparation"
        with kbc.connect_closing(board="board-a") as conn:
            assert kb.get_task(conn, tasks["board-a"]).claim_lock is None
        processes[1].start()
        losing_board, locked, spawned, grants = output.get(timeout=30)
        assert losing_board == "board-b"
        assert locked and not spawned and not grants
        release.set()
        winning_board, locked, spawned, grants = output.get(timeout=30)
        assert winning_board == "board-a" and not locked
        assert [task for task, *_ in spawned] == grants == [tasks["board-a"]]
        for process in processes:
            process.join(timeout=30)
            assert process.exitcode == 0
        with kbc.connect_closing(board="board-b") as conn:
            held = kbd.dispatch_once(
                conn, board="board-b", dry_run=True,
                max_in_progress=2, max_in_progress_per_profile=1,
            )
            assert held.skipped_per_profile_capped == [(tasks["board-b"], "alpha", 1)]
        with kbc.connect_closing(board="board-a") as conn:
            conn.execute(
                "UPDATE tasks SET status = 'done', claim_lock = NULL WHERE id = ?",
                (tasks["board-a"],),
            )
        with kbc.connect_closing(board="board-b") as conn:
            eligible = kbd.dispatch_once(
                conn, board="board-b", dry_run=True,
                max_in_progress=2, max_in_progress_per_profile=1,
            )
            assert [task for task, *_ in eligible.spawned] == [tasks["board-b"]]
    finally:
        release.set()
        for process in processes:
            if process.pid is not None:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=30)
        output.close()
        output.join_thread()


@pytest.mark.parametrize("limits,expected", [
    ({}, {"alpha": 0, "beta": 1}),
    ({"max_in_progress": None, "max_in_progress_per_profile": None}, {"alpha": 2, "beta": 2}),
    ({"max_in_progress": 3, "max_in_progress_per_profile": 2}, {"alpha": 1, "beta": 1}),
])
def test_omitted_limits_inherit_config_but_explicit_cached_limits_do_not(
    isolated_kanban_home_with_profiles, limits, expected,
):
    kb = isolated_kanban_home_with_profiles
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd

    kb.create_board("busy")
    kb.create_board("ready")
    with kbc.connect_closing(board="busy") as conn:
        task = kb.create_task(conn, title="running alpha", assignee="alpha")
        assert kb.claim_task(conn, task) is not None
    (kb.kanban_home() / "config.yaml").write_text(
        "kanban:\n  max_in_progress: 2\n  max_in_progress_per_profile: 1\n",
    )
    with kbc.connect_closing(board="ready") as conn:
        for profile in ("alpha", "alpha", "beta", "beta"):
            kb.create_task(conn, title=profile, assignee=profile)
        result = kbd.dispatch_once(conn, board="ready", dry_run=True, **limits)
    assignees = [profile for _, profile, _ in result.spawned]
    assert {profile: assignees.count(profile) for profile in expected} == expected

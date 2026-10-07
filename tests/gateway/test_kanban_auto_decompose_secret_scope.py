"""Dispatcher board enumeration and auto-decompose scope.

Regression for #107955 / #57837: the tick runs off-turn in a fresh Context, so
``get_secret`` fails closed unless the tick installs the launch profile's scope.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent import secret_scope as ss
from gateway import kanban_watchers_dispatcher as kwd
from gateway.kanban_watchers_common import _to_thread_process_service
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli.kanban_runtime import prospective_identity


def _dispatcher():
    settings = kwd._DispatcherSettings(60.0, None, None, 2, 0, True, None, None)
    return kwd._KanbanDispatcher(SimpleNamespace(DEFAULT_BOARD="default"), settings)


def test_auto_decompose_tick_reads_launch_profile_secrets_under_multiplex(monkeypatch, tmp_path):
    import hermes_cli

    (tmp_path / ".env").write_text("ANTHROPIC_API_KEY=launch-profile-key\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(kwd, "_board_slugs", lambda kb: ["default"])

    seen = {}

    def fake_decompose(task_id, author=None):
        seen["value"] = ss.get_secret("ANTHROPIC_API_KEY")
        return SimpleNamespace(ok=True, fanout=False, child_ids=None, reason=None)

    fake = SimpleNamespace(list_triage_ids=lambda: ["t1"], decompose_task=fake_decompose)
    monkeypatch.setitem(sys.modules, "hermes_cli.kanban_decompose", fake)
    monkeypatch.setattr(hermes_cli, "kanban_decompose", fake, raising=False)

    ss.set_multiplex_active(True)
    try:
        # Same hop the gateway uses: fresh Context, no inherited per-turn scope.
        decomposed = asyncio.run(_to_thread_process_service(_dispatcher().auto_decompose_tick, 5))
    finally:
        ss.set_multiplex_active(False)

    assert decomposed == 1
    assert seen["value"] == "launch-profile-key"
    assert ss.current_secret_scope() is None


@pytest.fixture
def dispatcher_boards(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    for name in (
        "HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD", "HERMES_KANBAN_TASK",
        "HERMES_KANBAN_WORKSPACES_ROOT", "HERMES_DELEGATED_CHILD_CONTEXT",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(kbd, "_memory_pressure_level", lambda: "normal")
    ready, triage = {}, {}
    for slug in ("default", "alpha", "beta"):
        kb.create_board(slug)
        with kbc.connect_closing(board=slug) as conn:
            ready[slug] = [
                kb.create_task(conn, title=f"{slug} ready {i}", assignee="default")
                for i in range(2)
            ]
            triage[slug] = kb.create_task(
                conn, title=f"{slug} triage", assignee="default", triage=True,
            )
    kb.set_current_board("beta")
    settings = kwd._DispatcherSettings(60.0, 1, 10, 2, 0, True, None, 1)
    return kwd._KanbanDispatcher(kb, settings), ready, triage


@pytest.mark.parametrize("pinned", [False, True])
@pytest.mark.parametrize("operation", ["dispatch", "probe", "decompose"])
def test_dispatcher_visits_each_resolved_board_once(
    dispatcher_boards, monkeypatch, pinned, operation,
):
    """Real board resolution and SQLite work, with only native/LLM I/O replaced."""
    dispatcher, ready, triage = dispatcher_boards
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "beta")
    if pinned:
        monkeypatch.setenv("HERMES_KANBAN_DB", str(kb.kanban_db_path("beta")))
    expected_boards = ["beta"] if pinned else list(ready)

    if operation == "probe":
        assert sorted(dispatcher.spawnable_ids()) == sorted(
            f"{slug}/{tid}" for slug in expected_boards for tid in ready[slug]
        )
    elif operation == "decompose":
        from hermes_cli import kanban_decompose as decomp

        attempts = []

        def unavailable_aux(_action, task_id, **_kwargs):
            attempts.append((kb.get_current_board(), task_id))
            # Keep the task in triage: duplicate board visits would spend
            # another attempt on the same failed decomposition.
            return None, "auxiliary provider unavailable"

        monkeypatch.setattr(decomp, "_call_aux", unavailable_aux)
        assert dispatcher.auto_decompose_tick(2) == 0
        assert attempts == [(slug, triage[slug]) for slug in expected_boards[:2]]
        assert os.environ["HERMES_KANBAN_BOARD"] == "beta"
    else:
        identity = prospective_identity()
        grants = []

        def spawn(task, workspace, *, board=None, **_kwargs):
            return kbd.WorkerLaunch(
                pid=identity.pid, runtime_identity=identity.as_dict(),
                preparation_id=f"prep-{task.id}",
                grant=lambda run_id, claim_lock: grants.append((board, task.id)),
            )

        monkeypatch.setattr(kbd, "_default_spawn", spawn)
        results = dispatcher.tick_once()
        assert [slug for slug, _ in results] == expected_boards
        assert [slug for slug, _ in grants] == expected_boards
        for slug, result in results:
            assert len(result.spawned) == dispatcher.settings.max_spawn
            assert result.spawned[0][0] in ready[slug]
        # The per-board concurrency cap still holds on subsequent ticks.
        assert all(not result.spawned for _, result in dispatcher.tick_once())


def test_auto_decompose_does_not_redirect_concurrent_board_writes(dispatcher_boards, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    from hermes_cli import kanban_decompose as decomp

    dispatcher, _, triage = dispatcher_boards
    entered, release = Event(), Event()
    attempts = []

    def waiting_aux(_action, task_id, **_kwargs):
        attempts.append((kb.get_current_board(), task_id))
        entered.set()
        assert release.wait(10), "concurrent board write never completed"
        return None, "auxiliary provider unavailable"

    monkeypatch.setattr(decomp, "_call_aux", waiting_aux)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(dispatcher.auto_decompose_tick, 1)
        try:
            assert entered.wait(10), "decomposition never reached the auxiliary call"
            with kbc.connect_closing() as conn:
                created = kb.create_task(conn, title="concurrent beta task")
            with kbc.connect_closing(board="beta") as conn:
                assert kb.get_task(conn, created).title == "concurrent beta task"
            with kbc.connect_closing(board="default") as conn:
                assert kb.get_task(conn, created) is None
        finally:
            release.set()
        assert pending.result(timeout=10) == 0
    assert attempts == [("default", triage["default"])]
    assert kb.get_current_board() == "beta"


@pytest.mark.parametrize("stop_at", ["grant_boundary", "after_grant"])
def test_dispatcher_stops_before_later_boards(dispatcher_boards, monkeypatch, stop_at):
    dispatcher, ready, _ = dispatcher_boards
    identity = prospective_identity()
    stopped = False
    inside_guard = False
    grants, cancelled = [], []

    @contextlib.contextmanager
    def grant_guard():
        nonlocal inside_guard, stopped
        inside_guard = True
        if stop_at == "grant_boundary":
            stopped = True
        try:
            yield
        finally:
            inside_guard = False

    def spawn(task, workspace, **_kwargs):
        def grant(run_id, claim_lock):
            nonlocal stopped
            assert inside_guard
            grants.append(task.id)
            stopped = True

        return kbd.WorkerLaunch(
            pid=identity.pid, runtime_identity=identity.as_dict(),
            preparation_id=f"prep-{task.id}", grant=grant,
            cancel=lambda: cancelled.append(task.id),
        )

    monkeypatch.setattr(kbd, "_default_spawn", spawn)
    dispatcher.should_stop = lambda: stopped
    dispatcher.grant_guard = grant_guard
    results = dispatcher.tick_once()
    assert [slug for slug, _ in results] == ["default"]
    result = results[0][1]
    if stop_at == "grant_boundary":
        assert result.spawned == grants == []
        assert len(cancelled) == 1
        assert result.interrupted == cancelled
    else:
        assert cancelled == []
        assert len(grants) == 1
        assert [row[0] for row in result.spawned] == grants
    with kbc.connect_closing(board="default") as conn:
        for tid in ready["default"]:
            assert kb.get_task(conn, tid).consecutive_failures == 0
    for slug in ("alpha", "beta"):
        with kbc.connect_closing(board=slug) as conn:
            assert all(kb.get_task(conn, tid).status == "ready" for tid in ready[slug])
    assert dispatcher.tick_once() == []

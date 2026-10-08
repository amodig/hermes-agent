"""Regression tests for #33488 (CLI max_in_progress / max_spawn / per-profile
config passthrough) and #29415 (kanban_swarm humanizer skill ref).

These two fixes are bundled because they're both small, both touch the
kanban dispatcher's CLI surface, and they each guard against a silent
operator footgun that only manifests in long-running setups.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import shlex
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def isolated_kanban_home(monkeypatch):
    """Spin up a fresh HERMES_HOME with a clean kanban DB."""
    test_home = tempfile.mkdtemp(prefix="kanban_cli_passthrough_")
    os.makedirs(os.path.join(test_home, "profiles", "default"), exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", test_home)
    monkeypatch.setenv("HERMES_KANBAN_HOME", test_home)
    for mod in list(sys.modules.keys()):
        if mod.startswith("hermes_cli") or mod.startswith("hermes_state") or mod == "hermes_constants":
            del sys.modules[mod]
    yield test_home


def test_cli_dispatch_passes_max_in_progress_from_config(isolated_kanban_home, monkeypatch):
    """#33488: hermes kanban dispatch must pass kanban.max_in_progress from
    config to dispatch_once. Without this, the global concurrency cap is
    unreachable from the CLI even though it works from the gateway."""
    from hermes_cli import kanban as kb_cli
    from hermes_cli import kanban_db
    from hermes_cli import kanban_db_dispatch as kbd

    # Configure max_in_progress in the loaded config.
    fake_config = {
        "kanban": {
            "max_in_progress": 3,
            "max_spawn": 5,
            "default_assignee": "default",
            "max_in_progress_per_profile": 2,
        }
    }
    monkeypatch.setattr(
        "hermes_cli.config.load_config", lambda: fake_config
    )

    captured = {}

    def fake_dispatch_once(conn, **kwargs):
        captured.update(kwargs)
        return kanban_db.DispatchResult()

    monkeypatch.setattr(kbd, "dispatch_once", fake_dispatch_once)

    args = argparse.Namespace(dry_run=True, max=None, failure_limit=2, json=False)
    kb_cli._cmd_dispatch(args)

    # Every config value must have reached dispatch_once.
    assert captured.get("max_in_progress") == 3, (
        f"CLI must pass kanban.max_in_progress from config; got {captured.get('max_in_progress')!r}"
    )
    assert captured.get("max_spawn") == 5, (
        f"CLI must pass kanban.max_spawn from config when --max is not provided; got {captured.get('max_spawn')!r}"
    )
    assert captured.get("default_assignee") == "default"
    assert captured.get("max_in_progress_per_profile") == 2


def test_cli_max_flag_overrides_config_max_spawn(isolated_kanban_home, monkeypatch):
    """--max on the CLI takes precedence over kanban.max_spawn in config.
    The CLI flag is the explicit operator signal; config is the default."""
    from hermes_cli import kanban as kb_cli
    from hermes_cli import kanban_db
    from hermes_cli import kanban_db_dispatch as kbd

    fake_config = {"kanban": {"max_spawn": 10}}
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: fake_config)

    captured = {}
    monkeypatch.setattr(
        kbd, "dispatch_once",
        lambda conn, **kw: (captured.update(kw), kanban_db.DispatchResult())[1],
    )

    args = argparse.Namespace(dry_run=True, max=2, failure_limit=2, json=False)
    kb_cli._cmd_dispatch(args)

    assert captured.get("max_spawn") == 2, (
        f"CLI --max=2 must override config kanban.max_spawn=10; got {captured.get('max_spawn')!r}"
    )




@pytest.fixture()
def moved_review_dispatch(isolated_kanban_home, monkeypatch, tmp_path):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    from tests.hermes_cli.test_kanban_handoff import _repo, _commit

    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"kanban": {"max_in_progress": 1, "review_dispatch": True}},
    )
    monkeypatch.setattr(kbd, "_memory_pressure_level", lambda: "ok")
    monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: lambda name: True)
    kb.init_db()
    kb.create_board("recovery")
    kb.set_current_board("recovery")
    repo, base, branch = _repo(tmp_path)
    with kbc.connect_closing(board="recovery") as conn:
        parent = kb.create_task(
            conn, title="implementation", assignee="implementer",
            workspace_kind="worktree", workspace_path=str(repo), branch_name=branch,
            lifecycle_contract={
                "kind": "code", "review_mode": "separate_card", "reviewer": "reviewer",
                "validation_required": False,
            },
        )
        review = kb.create_task(
            conn, title="review", assignee="reviewer", parents=[parent],
            workspace_kind="worktree", workspace_path=str(repo), branch_name=branch,
            lifecycle_contract={"kind": "review", "candidate_task_id": parent},
        )
        implementation = kb.claim_task(conn, parent)
        assert implementation is not None
        approved = _commit(repo)
        assert kb.complete_task(
            conn, parent, expected_run_id=implementation.current_run_id,
            summary="implemented src/changed.py", metadata={"base_sha": base, "head_sha": approved},
        )
        moved = _commit(repo, "src/moved.py")
        ready = kb.create_task(conn, title="ready work", assignee="implementer")
        version = kb.get_task(conn, review).version
        event_count = len(kb.list_events(conn, review))
    return SimpleNamespace(
        parent=parent, review=review, ready=ready, approved=approved, moved=moved,
        version=version, event_count=event_count,
        database=kb.kanban_db_path(board="recovery").resolve(),
    )


@pytest.mark.parametrize("pinned_db", [False, True])
@pytest.mark.parametrize("json_output", [False, True])
def test_dispatch_reports_real_handoff_heads_and_fenced_recovery(
    moved_review_dispatch, monkeypatch, capsys, json_output, pinned_db,
):
    from hermes_cli import kanban as kb_cli, kanban_db as kb, kanban_db_connect as kbc
    from hermes_platform.host import facts as host_facts

    case = moved_review_dispatch
    parent, review, ready = case.parent, case.review, case.ready
    approved, moved, version = case.approved, case.moved, case.version
    event_count = case.event_count
    database = case.database
    if pinned_db:
        monkeypatch.setenv("HERMES_KANBAN_DB", str(database))
    args = argparse.Namespace(
        dry_run=True, max=1, failure_limit=2, json=json_output,
        board="default" if pinned_db else None,
    )
    with kb.pin_first_board_resolution():
        assert kb_cli._cmd_dispatch(args) == 0
    output = capsys.readouterr().out
    if json_output:
        payload = json.loads(output)
        assert [item["task_id"] for item in payload["spawned"]] == [ready]
        assert len(payload["handoff_refused"]) == 1
        refusal = payload["handoff_refused"][0]
        assert refusal["task_id"] == review and refusal["parent_id"] == parent
        assert refusal["expected_head_sha"] == approved
        assert refusal["actual_head_sha"] == moved
        if host_facts.os_family() != "win32":
            command = shlex.split(refusal["command"])
            prefix = (
                ["env", f"HERMES_KANBAN_DB={database}", "hermes", "kanban", "block"]
                if pinned_db else ["hermes", "kanban", "--board", "recovery", "block"]
            )
            assert command[:5] == prefix
            assert command[5] == review
            assert command[command.index("--expected-version") + 1] == str(version)
            assert command[command.index("--kind") + 1] == "capability"
        assert "does not authorize a new candidate or release validation" in refusal["recovery"]
    else:
        for value in (review, parent, ready, approved, moved, "handoff_head_moved"):
            assert value in output
        if host_facts.os_family() != "win32":
            route = f"HERMES_KANBAN_DB={database}" if pinned_db else "--board recovery block"
            assert route in output
        assert "--quarantine-review" in output
        assert "does not authorize a new candidate or release validation" in output
    with kbc.connect_closing(board="recovery") as conn:
        assert kb.get_task(conn, review).status == "review"
        assert kb.get_task(conn, review).current_run_id is None
        assert kb.get_task(conn, ready).status == "ready"
        assert len(kb.list_events(conn, review)) == event_count


def test_pinned_recovery_inspection_and_readback_keep_original_database(
    moved_review_dispatch, monkeypatch, tmp_path, capsys,
):
    import sqlite3
    from hermes_cli import kanban as kb_cli, kanban_db as kb, kanban_db_connect as kbc
    from tools import kanban_tools

    case = moved_review_dispatch
    pinned = tmp_path / "pinned.db"
    with kbc.connect_closing(board="recovery") as source:
        with sqlite3.connect(pinned) as target:
            source.backup(target)
    monkeypatch.setenv("HERMES_KANBAN_DB", str(pinned))
    with kbc.connect_closing() as conn:
        assert kb.update_task(
            conn, case.review, expected_version=case.version,
            reason="Operator changes only the pinned review's model",
            model="unit-test-model",
        )
    args = argparse.Namespace(task_id=case.review, board=None, json=True)
    assert kb_cli._cmd_show(args) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["task"]["version"] == case.version + 1
    assert json.loads(kanban_tools._handle_show({"task_id": case.review}))["task"]["version"] == case.version + 1
    result = json.loads(kanban_tools._handle_block({
        "task_id": case.review, "quarantine_review": True, "kind": "capability",
        "expected_version": shown["task"]["version"],
        "reason": "Immutable parent handoff is not startable; quarantine pending operator recovery",
    }))
    assert result["ok"] and result["status"] == "blocked"
    assert kb_cli._cmd_show(args) == 0
    readback = json.loads(capsys.readouterr().out)["task"]
    assert (readback["status"], readback["version"]) == ("blocked", case.version + 2)
    assert json.loads(kanban_tools._handle_show({"task_id": case.review}))["task"]["status"] == "blocked"
    original = json.loads(kanban_tools._handle_show({"task_id": case.review, "board": "recovery"}))["task"]
    assert (original["status"], original["version"]) == ("review", case.version)


@pytest.mark.platforms("windows")
@pytest.mark.parametrize("pinned_db", [False, True])
def test_powershell_recovery_replays_original_database(
    moved_review_dispatch, monkeypatch, tmp_path, capsys, pinned_db,
):
    import sqlite3
    import subprocess
    from hermes_cli import kanban as kb_cli, kanban_db as kb, kanban_db_connect as kbc

    case = moved_review_dispatch
    database = tmp_path / "operator's pinned database.db"
    if pinned_db:
        with kbc.connect_closing(board="recovery") as source:
            target = sqlite3.connect(database)
            try:
                source.backup(target)
            finally:
                target.close()
        monkeypatch.setenv("HERMES_KANBAN_DB", str(database))
    args = argparse.Namespace(
        dry_run=True, max=1, failure_limit=2, json=True,
        board="default" if pinned_db else None,
    )
    with kb.pin_first_board_resolution():
        assert kb_cli._cmd_dispatch(args) == 0
    refusal, = json.loads(capsys.readouterr().out)["handoff_refused"]
    kb.set_current_board("default")
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    executable = "'" + sys.executable.replace("'", "''") + "'"
    script = (
        f"function hermes {{ & {executable} -m hermes_cli.main @args }}; "
        "$previousPin = $env:HERMES_KANBAN_DB; "
        + refusal["command"]
        + "; if ($env:HERMES_KANBAN_DB -ne $previousPin) { throw 'database pin leaked' }; "
        "exit $LASTEXITCODE"
    )
    subprocess.run(["powershell.exe", "-NoProfile", "-Command", script], check=True)
    with kbc.connect_closing(db_path=database if pinned_db else case.database) as conn:
        assert kb.get_task(conn, case.review).status == "blocked"
        assert kb.get_task(conn, case.review).version == case.version + 1
    if pinned_db:
        with kbc.connect_closing(board="recovery") as conn:
            assert kb.get_task(conn, case.review).status == "review"



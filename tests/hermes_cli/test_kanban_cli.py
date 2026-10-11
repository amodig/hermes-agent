"""Tests for the kanban CLI surface (hermes_cli.kanban)."""

from __future__ import annotations

import argparse
import json
import threading
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


# ---------------------------------------------------------------------------
# Workspace flag parsing
# ---------------------------------------------------------------------------







# ---------------------------------------------------------------------------
# run_slash smoke tests (end-to-end via the same entry both CLI and gateway use)
# ---------------------------------------------------------------------------



def test_kanban_list_json_includes_session_id(kanban_home):
    """JSON output exposes `session_id` so external clients (Scarf, web
    dashboards) don't need a side query to filter by chat session."""
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    with kbc.connect() as conn:
        kb.create_task(
            conn, title="acp task", assignee="alice", session_id="acp-x"
        )
    raw = kc.run_slash("list --json")
    payload = json.loads(raw)
    assert any(
        row.get("title") == "acp task"
        and row.get("session_id") == "acp-x"
        for row in payload
    )


def test_kanban_show_json_includes_runtime_limit(kanban_home):
    with kbc.connect() as conn:
        bounded_id = kb.create_task(
            conn, title="bounded task", max_runtime_seconds=2700
        )
        uncapped_id = kb.create_task(conn, title="uncapped task")

    bounded = json.loads(kc.run_slash(f"show {bounded_id} --json"))
    uncapped = json.loads(kc.run_slash(f"show {uncapped_id} --json"))

    assert bounded["task"]["max_runtime_seconds"] == 2700
    assert uncapped["task"]["max_runtime_seconds"] is None


def test_kanban_show_text_renders_graph_with_open_connection(kanban_home):
    with kbc.connect_closing() as conn:
        parent_id = kb.create_task(conn, title="parent task")
        child_id = kb.create_task(conn, title="child task")
        kb.link_tasks(conn, parent_id=parent_id, child_id=child_id)

    output = kc.run_slash(f"show {child_id}")

    assert "child task" in output
    assert parent_id in output
    assert "Cannot operate on a closed database" not in output


def test_kanban_edit_updates_documented_task_fields(kanban_home):
    with kbc.connect_closing() as conn:
        task_id = kb.create_task(conn, title="old title", body="old body", priority=2)

    kc.run_slash(
        f"edit {task_id} --title 'new title' --body 'new body' --priority 70"
    )

    with kbc.connect_closing() as conn:
        task = kb.get_task(conn, task_id)
        events = kb.list_events(conn, task_id)
    assert (task.title, task.body, task.priority) == ("new title", "new body", 70)
    assert any(event.kind == "reprioritized" for event in events)


def test_worker_link_preserves_foreign_child_rules(kanban_home, monkeypatch):
    with kbc.connect_closing() as conn:
        worker = kb.create_task(conn, title="worker")
        assert kb.claim_task(conn, worker, claimer="worker") is not None
        worker_run_id = kb.get_task(conn, worker).current_run_id
        parent = kb.create_task(conn, title="unfinished parent")
        ready_child = kb.create_task(conn, title="foreign ready child")
        running_child = kb.create_task(conn, title="foreign running child")
        assert kb.claim_task(conn, running_child, claimer="other") is not None

    monkeypatch.setenv("HERMES_KANBAN_TASK", worker)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(worker_run_id))

    assert kc._cmd_link(argparse.Namespace(
        parent_id=parent, child_id=ready_child,
    )) == 0
    with pytest.raises(ValueError, match="child is already running"):
        kc._cmd_link(argparse.Namespace(
            parent_id=parent, child_id=running_child,
        ))
    # Owner handoff: the worker links its own running card, proving ownership
    # with HERMES_KANBAN_RUN_ID — the one path _cmd_link forwards a run id for.
    assert kc._cmd_link(argparse.Namespace(
        parent_id=parent, child_id=worker,
    )) == 0

    with kbc.connect_closing() as conn:
        assert kb.parent_ids(conn, ready_child) == [parent]
        assert kb.parent_ids(conn, running_child) == []
        assert kb.parent_ids(conn, worker) == [parent]


def test_board_override_is_isolated_per_concurrent_call(kanban_home, monkeypatch):
    kb.create_board("alpha")
    kb.create_board("beta")

    parser = argparse.ArgumentParser(prog="hermes", add_help=False)
    sub = parser.add_subparsers(dest="command")
    kc.build_parser(sub)

    barrier = threading.Barrier(2)
    original_init_db = kb.init_db

    def slow_init_db(*args, **kwargs):
        try:
            barrier.wait(timeout=5)
        except threading.BrokenBarrierError:
            pass
        return original_init_db(*args, **kwargs)

    monkeypatch.setattr(kb, "init_db", slow_init_db)

    failures: list[str] = []

    def worker(board: str, title: str) -> None:
        args = parser.parse_args(["kanban", "--board", board, "create", title])
        rc = kc.kanban_command(args)
        if rc != 0:
            failures.append(f"{board}:{rc}")

    t1 = threading.Thread(target=worker, args=("alpha", "alpha-task"))
    t2 = threading.Thread(target=worker, args=("beta", "beta-task"))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert failures == []

    with kbc.connect_closing(board="alpha") as conn:
        alpha_titles = [row.title for row in kb.list_tasks(conn, limit=100)]
    with kbc.connect_closing(board="beta") as conn:
        beta_titles = [row.title for row in kb.list_tasks(conn, limit=100)]

    assert alpha_titles == ["alpha-task"]
    assert beta_titles == ["beta-task"]



def test_lifecycle_contract_cli_errors_are_concise(kanban_home, capsys):
    parser = argparse.ArgumentParser(prog="hermes", add_help=False)
    kc.build_parser(parser.add_subparsers(dest="command"))
    args = parser.parse_args(
        [
            "kanban",
            "create",
            "invalid lifecycle task",
            "--assignee",
            "worker",
            "--lifecycle-contract",
            json.dumps(
                {
                    "kind": "code",
                    "review_mode": "separate_card",
                    "reviewer": "worker",
                    "validation_required": False,
                }
            ),
        ]
    )

    assert kc.kanban_command(args) == 1
    error = capsys.readouterr().err
    assert "reviewer must differ from the implementation assignee" in error
    assert "Traceback" not in error

def test_lifecycle_contract_help_describes_legacy_binding_only(capsys):
    parser = argparse.ArgumentParser(prog="hermes", add_help=False)
    kc.build_parser(parser.add_subparsers(dest="command"))

    with pytest.raises(SystemExit):
        parser.parse_args(["kanban", "update", "--help"])
    help_text = " ".join(capsys.readouterr().out.split())

    assert "bind a historical NULL contract" in help_text
    assert "classification must be supplied at task creation" in help_text
    assert "bind or replace" not in help_text


def _quarantine_review_case(conn, tmp_path):
    from tests.hermes_cli.test_kanban_handoff import _git, _moved_head_review, _repo

    repo, base, branch = _repo(tmp_path)
    parent, review, head = _moved_head_review(conn, repo, base, branch)
    return repo, parent, review, head, _git(repo, "rev-parse", "HEAD")


def test_block_quarantine_cli_preserves_candidate_and_refuses_stale_retry(
    kanban_home, tmp_path, capsys,
):
    from tests.hermes_cli.test_kanban_handoff import _git

    kb.create_board("quarantine")
    with kb.scoped_current_board("quarantine"), kbc.connect_closing() as conn:
        repo, parent, review, head, moved_head = _quarantine_review_case(
            conn, tmp_path,
        )
        before = dict(conn.execute("SELECT * FROM tasks WHERE id = ?", (review,)).fetchone())
        runs = list(conn.execute("SELECT * FROM task_runs"))
        links = list(conn.execute("SELECT * FROM task_links"))
        handoff = kb.latest_handoff(conn, parent)

    parser = argparse.ArgumentParser(prog="hermes", add_help=False)
    kc.build_parser(parser.add_subparsers(dest="command"))
    args = parser.parse_args([
        "kanban", "--board", "quarantine", "block", review,
        "--quarantine-review", "--expected-version", str(before["version"]),
        "--kind", "capability", "--reason",
        "Immutable parent handoff is not startable; quarantine pending operator recovery",
    ])
    assert kc.kanban_command(args) == 0
    assert f"Blocked {review}" in capsys.readouterr().out
    with kb.scoped_current_board("quarantine"), kbc.connect_closing() as conn:
        after = dict(conn.execute("SELECT * FROM tasks WHERE id = ?", (review,)).fetchone())
        assert after["status"] == "blocked"
        assert after["version"] == before["version"] + 1
        assert after["block_kind"] == "capability"
        for field in before.keys() - {"status", "version", "block_kind", "block_recurrences"}:
            assert after[field] == before[field], field
        assert list(conn.execute("SELECT * FROM task_runs")) == runs
        assert list(conn.execute("SELECT * FROM task_links")) == links
        assert kb.latest_handoff(conn, parent) == handoff
        event, = [event for event in kb.list_events(conn, review) if event.kind == "blocked"]
        assert event.run_id is None
        assert event.payload["source_status"] == "review"
        assert event.payload["handoff_error"]["expected_head_sha"] == head
        assert event.payload["handoff_error"]["actual_head_sha"] == moved_head
        assert kb.recompute_ready(conn) == 0
        assert kb.get_task(conn, review).status == "blocked"
        quarantined = list(conn.iterdump())
    assert _git(repo, "cat-file", "-t", head) == "commit"
    assert _git(repo, "cat-file", "-t", moved_head) == "commit"

    assert kc.kanban_command(args) == 1
    assert "review quarantine refused" in capsys.readouterr().err
    with kb.scoped_current_board("quarantine"), kbc.connect_closing() as conn:
        assert list(conn.iterdump()) == quarantined
    assert kc.kanban_command(parser.parse_args([
        "kanban", "--board", "quarantine", "unblock", review,
    ])) == 0
    with kb.scoped_current_board("quarantine"), kbc.connect_closing() as conn:
        assert kb.get_task(conn, review).status == "review"
        assert kb.claim_review_task(conn, review) is None



# ---------------------------------------------------------------------------

def test_archive_rm_purges_archived_lifecycle_graph(kanban_home):
    with kbc.connect() as conn:
        implementation = kb.create_task(
            conn,
            title="cli purge candidate",
            assignee="builder",
            initial_status="blocked",
            lifecycle_contract={
                "kind": "code",
                "review_mode": "separate_card",
                "reviewer": "reviewer",
                "validation_required": True,
            },
        )
        review = kb.create_task(
            conn,
            title="cli purge review",
            assignee="reviewer",
            initial_status="blocked",
            lifecycle_contract={"kind": "review", "candidate_task_id": implementation},
        )
        validation = kb.create_task(
            conn,
            title="cli purge validation",
            assignee="tester",
            initial_status="blocked",
            lifecycle_contract={
                "kind": "validation",
                "candidate_task_id": implementation,
            },
        )
        kb.link_tasks(conn, implementation, review, requirement="phase_finished")
        kb.link_tasks(conn, review, validation, requirement="review_approved")
        for task_id in (implementation, review, validation):
            assert kb.archive_task(conn, task_id)

    output = kc.run_slash(f"archive --rm {implementation} {review} {validation}")
    assert output.count("Deleted ") == 3
    with kbc.connect() as conn:
        assert all(kb.get_task(conn, task_id) is None for task_id in (
            implementation, review, validation,
        ))


# Integration with the COMMAND_REGISTRY
# ---------------------------------------------------------------------------






# ---------------------------------------------------------------------------
# reclaim + reassign CLI smoke tests
# ---------------------------------------------------------------------------

def test_run_slash_reclaim_running_task(kanban_home):
    import re
    import time
    import secrets
    from hermes_cli import kanban_db_connect as kbc

    out1 = kc.run_slash("create 'stuck worker task' --assignee broken-model")
    m = re.search(r"(t_[a-f0-9]+)", out1)
    assert m
    tid = m.group(1)

    # Simulate a running claim outside TTL.
    conn = kbc.connect()
    try:
        lock = secrets.token_hex(4)
        conn.execute(
            "UPDATE tasks SET status='running', claim_lock=?, claim_expires=?, "
            "worker_pid=? WHERE id=?",
            (lock, int(time.time()) + 3600, 4242, tid),
        )
        conn.execute(
            "INSERT INTO task_runs (task_id, status, claim_lock, claim_expires, "
            "worker_pid, started_at) VALUES (?, 'running', ?, ?, ?, ?)",
            (tid, lock, int(time.time()) + 3600, 4242, int(time.time())),
        )
        rid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute("UPDATE tasks SET current_run_id=? WHERE id=?", (rid, tid))
        conn.commit()
    finally:
        conn.close()

    out = kc.run_slash(f"reclaim {tid} --reason 'test'")
    assert "Reclaimed" in out, out
    # Status back to ready.
    out2 = kc.run_slash(f"show {tid}")
    assert "ready" in out2.lower()




# ---------------------------------------------------------------------------
# /kanban specify — slash surface (same entry point CLI + gateway use)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# /kanban help / no-args / unknown-action UX (issue #21794)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# supersede-candidate — unreviewed moved-head candidate recovery
# ---------------------------------------------------------------------------

def _supersede_candidate_case(conn, tmp_path):
    """Base -> H1 handoff, branch moved to H2, unstarted review quarantined.

    The H2 commit stays on the branch, so the recovery lane has no new commit
    to manufacture: the implementer hands off the existing newer head.
    """
    from tests.hermes_cli.test_kanban_handoff import _commit, _lane, _repo

    repo, base, branch = _repo(tmp_path)
    implementation, reviewer = _lane(conn, repo, branch, validation_required=True)
    tester = kb.create_task(
        conn,
        title="recovery tester",
        assignee="tester",
        lifecycle_contract={"kind": "validation", "candidate_task_id": implementation},
    )
    kb.link_tasks(conn, reviewer, tester, requirement="review_approved")
    run = kb.claim_task(conn, implementation)
    head = _commit(repo)
    assert kb.complete_task(
        conn,
        implementation,
        expected_run_id=run.current_run_id,
        summary="implemented src/changed.py",
        metadata={"base_sha": base, "head_sha": head},
    )
    assert kb.get_task(conn, reviewer).status == "review"
    moved_head = _commit(repo, "src/moved.py")
    assert kb.block_task(
        conn,
        reviewer,
        reason="immutable parent handoff moved; quarantine pending supported recovery",
        kind="capability",
        quarantine_review=True,
        expected_version=kb.get_task(conn, reviewer).version,
    )
    candidate_run_id = conn.execute(
        "SELECT candidate_run_id FROM tasks WHERE id = ?", (implementation,)
    ).fetchone()[0]
    return {
        "repo": repo,
        "head": head,
        "moved_head": moved_head,
        "implementation": implementation,
        "reviewer": reviewer,
        "tester": tester,
        "candidate_run_id": candidate_run_id,
        "versions": {
            task_id: kb.get_task(conn, task_id).version
            for task_id in (implementation, reviewer, tester)
        },
    }


def _supersede_candidate_flags(case, **overrides):
    values = {
        "--implementation-id": case["implementation"],
        "--expected-versions": json.dumps(case["versions"]),
        "--expected-candidate-run-id": str(case["candidate_run_id"]),
        "--expected-head-sha": case["head"],
        "--replacement-head-sha": case["moved_head"],
        "--reason": "Supersede moved candidate before any reviewer run",
    }
    values.update(overrides)
    flags: list[str] = []
    for flag, value in values.items():
        flags += [flag, value]
    return flags


def test_supersede_candidate_cli_recovers_unreviewed_moved_candidate(
    kanban_home, tmp_path, monkeypatch, capsys,
):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)
    kb.create_board("recovery")
    with kb.scoped_current_board("recovery"), kbc.connect_closing() as conn:
        case = _supersede_candidate_case(conn, tmp_path)
        runs_before = list(conn.execute("SELECT * FROM task_runs"))
        links_before = list(conn.execute("SELECT * FROM task_links"))

    parser = argparse.ArgumentParser(prog="hermes", add_help=False)
    kc.build_parser(parser.add_subparsers(dest="command"))
    args = parser.parse_args([
        "kanban", "--board", "recovery", "supersede-candidate",
        *_supersede_candidate_flags(case), "--json",
    ])
    assert kc.kanban_command(args) == 0

    outcome = json.loads(capsys.readouterr().out)
    assert outcome["implementation_id"] == case["implementation"]
    assert outcome["reviewer_id"] == case["reviewer"]
    assert outcome["tester_id"] == case["tester"]
    assert outcome["status"] == "ready"
    assert outcome["superseded_candidate_run_id"] == case["candidate_run_id"]
    assert (outcome["superseded_head_sha"], outcome["replacement_head_sha"]) == (
        case["head"], case["moved_head"],
    )
    assert outcome["versions"] == {
        task_id: version + 1 for task_id, version in case["versions"].items()
    }

    with kb.scoped_current_board("recovery"), kbc.connect_closing() as conn:
        assert kb.get_task(conn, case["implementation"]).status == "ready"
        assert kb.get_task(conn, case["reviewer"]).status == "todo"
        assert kb.get_task(conn, case["tester"]).status == "todo"
        # History and edges are preserved; no review/test run was started.
        assert list(conn.execute("SELECT * FROM task_runs")) == runs_before
        assert list(conn.execute("SELECT * FROM task_links")) == links_before
        assert kb.claim_review_task(conn, case["reviewer"]) is None
        assert kb.claim_task(conn, case["tester"]) is None
        # Readiness recompute must not release the reviewer/tester early.
        assert kb.recompute_ready(conn) == 0
        assert kb.get_task(conn, case["reviewer"]).status == "todo"
        assert kb.get_task(conn, case["tester"]).status == "todo"
        superseded = [
            event for event in kb.list_events(conn, case["implementation"])
            if event.kind == "candidate_superseded"
        ]
        assert len(superseded) == 1
        assert superseded[0].run_id == case["candidate_run_id"]


def test_supersede_candidate_cli_requires_explicit_board_and_valid_json(
    kanban_home, tmp_path, monkeypatch, capsys,
):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)
    kb.create_board("recovery")
    with kb.scoped_current_board("recovery"), kbc.connect_closing() as conn:
        case = _supersede_candidate_case(conn, tmp_path)
        quarantined = list(conn.iterdump())

    parser = argparse.ArgumentParser(prog="hermes", add_help=False)
    kc.build_parser(parser.add_subparsers(dest="command"))

    missing_board = parser.parse_args([
        "kanban", "supersede-candidate", *_supersede_candidate_flags(case),
    ])
    assert kc.kanban_command(missing_board) == 2
    assert "requires an explicit --board" in capsys.readouterr().err

    bad_json = parser.parse_args([
        "kanban", "--board", "recovery", "supersede-candidate",
        *_supersede_candidate_flags(case, **{"--expected-versions": "not-json"}),
    ])
    assert kc.kanban_command(bad_json) == 2
    assert "JSON object" in capsys.readouterr().err

    with kb.scoped_current_board("recovery"), kbc.connect_closing() as conn:
        assert list(conn.iterdump()) == quarantined


def test_supersede_candidate_cli_rejects_worker_context(
    kanban_home, tmp_path, monkeypatch, capsys,
):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)
    kb.create_board("recovery")
    with kb.scoped_current_board("recovery"), kbc.connect_closing() as conn:
        case = _supersede_candidate_case(conn, tmp_path)
        quarantined = list(conn.iterdump())

    parser = argparse.ArgumentParser(prog="hermes", add_help=False)
    kc.build_parser(parser.add_subparsers(dest="command"))
    args = parser.parse_args([
        "kanban", "--board", "recovery", "supersede-candidate",
        *_supersede_candidate_flags(case), "--json",
    ])
    monkeypatch.setenv("HERMES_KANBAN_TASK", case["implementation"])
    assert kc.kanban_command(args) == 1
    assert "orchestrator-only" in capsys.readouterr().err
    with kb.scoped_current_board("recovery"), kbc.connect_closing() as conn:
        assert list(conn.iterdump()) == quarantined



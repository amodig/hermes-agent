"""Tests for kanban lifecycle plugin hooks.

Verifies that claim/complete/block transitions fire the
kanban_task_claimed / kanban_task_completed / kanban_task_blocked plugin
hooks AFTER the board DB change is committed, with the documented kwargs,
and that a misbehaving hook callback never breaks the transition.
"""

from __future__ import annotations

import json
from pathlib import Path
import threading
import subprocess

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_lifecycle_evidence as evidence
from hermes_cli.kanban_lifecycle import get_lifecycle_state
from hermes_cli.plugins import get_plugin_manager


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def captured_hooks(monkeypatch):
    """Register capturing callbacks for the three kanban lifecycle hooks.

    Patches the plugin manager's _hooks dict directly (the same registry
    invoke_hook reads) and restores it afterward.
    """
    mgr = get_plugin_manager()
    events: list[tuple[str, dict]] = []
    saved = {k: list(v) for k, v in mgr._hooks.items()}
    for hook in ("kanban_task_claimed", "kanban_task_completed", "kanban_task_blocked"):
        mgr._hooks.setdefault(hook, []).append(
            lambda _h=hook, **kw: events.append((_h, kw))
        )
    try:
        yield events
    finally:
        mgr._hooks = saved




def test_claim_fires_hook(kanban_home, captured_hooks):
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="t", assignee="worker")
        claimed = kb.claim_task(conn, tid)
        assert claimed is not None
    finally:
        conn.close()
    fired = [e for e in captured_hooks if e[0] == "kanban_task_claimed"]
    assert len(fired) == 1
    kw = fired[0][1]
    assert kw["task_id"] == tid
    assert kw["assignee"] == "worker"
    assert "profile_name" in kw
    assert kw["run_id"] is not None




def test_review_claim_fires_hook(kanban_home, captured_hooks):
    conn = kbc.connect()
    try:
        task_id = kb.create_task(conn, title="review", assignee="builder")
        implementation = kb.claim_task(conn, task_id, claimer="builder")
        assert implementation is not None
        assert kb.request_review(
            conn, task_id, reviewer="reviewer",
            expected_run_id=implementation.current_run_id,
        )
        claimed = kb.claim_review_task(conn, task_id, claimer="reviewer")
        assert claimed is not None
    finally:
        conn.close()
    fired = [
        event for event in captured_hooks
        if event[0] == "kanban_task_claimed"
        and event[1]["run_id"] == claimed.current_run_id
    ]
    assert len(fired) == 1
    assert fired[0][1]["task_id"] == task_id
    assert fired[0][1]["assignee"] == "reviewer"

def test_general_completion_fires_hook(kanban_home, captured_hooks):
    conn = kbc.connect()
    try:
        task_id = kb.create_task(conn, title="general", assignee="worker")
        claimed = kb.claim_task(conn, task_id)
        assert claimed is not None
        assert kb.complete_task(conn, task_id, summary="done") is True
    finally:
        conn.close()

    completed = [
        event
        for event in captured_hooks
        if event[0] == "kanban_task_completed"
    ]
    assert len(completed) == 1
    assert completed[0][1]["task_id"] == task_id
    assert completed[0][1]["run_id"] == claimed.current_run_id



@pytest.mark.parametrize(
    ("review_mode", "validation_required"),
    [("same_card", False), ("same_card", True), ("separate_card", False), ("separate_card", True)],
)
def test_typed_completion_hooks_follow_acceptance_once(
    kanban_home, captured_hooks, tmp_path, review_mode, validation_required,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=Lifecycle Test",
         "-c", "user.email=lifecycle@example.invalid", "commit", "--allow-empty", "-qm", "base"],
        check=True, capture_output=True,
    )
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    conn = kbc.connect()
    try:
        candidate = kb.create_task(
            conn, title="implementation", assignee="builder", workspace_kind="dir",
            workspace_path=str(repo),
            lifecycle_contract={
                "kind": "code", "review_mode": review_mode,
                "reviewer": "reviewer", "validation_required": validation_required,
            },
        )
        review = candidate
        if review_mode == "separate_card":
            review = kb.create_task(
                conn, title="review", assignee="reviewer", parents=[candidate],
                lifecycle_contract={"kind": "review", "candidate_task_id": candidate},
            )
        if validation_required:
            validation = kb.create_task(
                conn, title="validation", assignee="tester", parents=[review],
                lifecycle_contract={"kind": "validation", "candidate_task_id": candidate},
            )
        implementation = kb.claim_task(conn, candidate, claimer="builder")
        assert implementation is not None
        assert kb.complete_task(
            conn, candidate, expected_run_id=implementation.current_run_id,
            metadata={"base_sha": head, "head_sha": head}, summary="implementation ready",
        )
        assert [name for name, _kwargs in captured_hooks if name == "kanban_task_completed"] == []
        assert any(event.kind == "review_requested" for event in kb.list_events(conn, candidate))
        reviewer = kb.claim_review_task(conn, review, claimer="reviewer")
        assert reviewer is not None
        assert kb.complete_task(
            conn, review, expected_run_id=reviewer.current_run_id,
            verdict="APPROVE", metadata={"reviewed_head_sha": head}, summary="approved",
        )
        if validation_required:
            expected_review_hooks = (
                [(review, reviewer.current_run_id)] if review_mode == "separate_card" else []
            )
            assert [
                (kwargs["task_id"], kwargs["run_id"])
                for name, kwargs in captured_hooks if name == "kanban_task_completed"
            ] == expected_review_hooks
            if review_mode == "same_card":
                assert any(event.kind == "validation_requested" for event in kb.list_events(conn, review))
            validator = kb.claim_task(conn, validation, claimer="tester")
            assert validator is not None
            assert kb.complete_task(
                conn, validation, expected_run_id=validator.current_run_id,
                verdict="PASS", metadata={"head_sha": head}, summary="validated",
            )
            terminal, verdict, metadata = validation, "PASS", {"head_sha": head}
        else:
            terminal, verdict, metadata = review, "APPROVE", {"reviewed_head_sha": head}
        assert get_lifecycle_state(conn, candidate)["acceptance"] == "accepted"
        completed = [
            (kwargs["task_id"], kwargs["run_id"])
            for name, kwargs in captured_hooks if name == "kanban_task_completed"
        ]
        expected = [(candidate, implementation.current_run_id)]
        if validation_required:
            expected.append((validation, validator.current_run_id))
        if review_mode == "separate_card":
            expected.append((review, reviewer.current_run_id))
        assert sorted(completed) == sorted(expected)
        assert [
            event.payload["new"] for event in kb.list_events(conn, candidate)
            if event.kind == "acceptance_changed"
        ] == ["accepted"]
        assert kb.complete_task(conn, terminal, verdict=verdict, metadata=metadata)
        assert [
            (kwargs["task_id"], kwargs["run_id"])
            for name, kwargs in captured_hooks if name == "kanban_task_completed"
        ] == completed
        if terminal == candidate:
            assert kb.repair_archive_task(
                conn, terminal, expected_version=kb.get_task(conn, terminal).version,
                reason="archive completed candidate",
            )
        else:
            assert kb.archive_task(conn, terminal)
        archived_acceptance = "stale" if terminal == candidate else "pending"
        assert get_lifecycle_state(conn, candidate)["acceptance"] == archived_acceptance
        transitions = [
            event.payload for event in kb.list_events(conn, candidate)
            if event.kind == "acceptance_changed"
        ]
        assert [(event["old"], event["new"]) for event in transitions] == [
            ("pending", "accepted"), ("accepted", archived_acceptance),
        ]
        assert transitions[-1]["source_task_id"] == terminal
        assert not kb.archive_task(conn, terminal)
        assert [
            event.payload for event in kb.list_events(conn, candidate)
            if event.kind == "acceptance_changed"
        ] == transitions
    finally:
        conn.close()


def test_completed_result_edit_rejects_handoff_metadata(kanban_home):
    conn = kbc.connect()
    try:
        task_id = kb.create_task(conn, title="immutable", assignee="worker")
        claimed = kb.claim_task(conn, task_id)
        assert claimed is not None
        assert kb.complete_task(
            conn,
            task_id,
            result="done",
            metadata={"head_sha": "original"},
        ) is True

        with pytest.raises(kb.LifecycleEvidenceError, match="handoff"):
            kb.edit_completed_task_result(
                conn,
                task_id,
                result="edited",
                metadata={"head_sha": "spoofed"},
            )

        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.result == "done"
        run = conn.execute(
            "SELECT metadata FROM task_runs "
            "WHERE task_id = ? AND outcome = 'completed' "
            "ORDER BY id DESC LIMIT 1",
            (task_id,),
        ).fetchone()
        assert run is not None
        assert json.loads(run["metadata"])["head_sha"] == "original"
    finally:
        conn.close()

def test_acceptance_event_projection_holds_write_lock(kanban_home):
    conn = kbc.connect()
    writer_start = threading.Event()
    writer_acquired = threading.Event()
    writer_done = threading.Event()
    writer_errors = []
    try:
        task_id = kb.create_task(conn, title="race", assignee="worker")
        with kbc.write_txn(conn):
            conn.execute("UPDATE tasks SET lifecycle_contract = NULL WHERE id = ?", (task_id,))
        before = {task_id: get_lifecycle_state(conn, task_id)["acceptance"]}

        def mutate_task():
            competing = kbc.connect()
            try:
                writer_start.wait(timeout=5)
                with kbc.write_txn(competing):
                    writer_acquired.set()
                    competing.execute(
                        "UPDATE tasks SET lifecycle_contract = NULL, version = version + 1 WHERE id = ?",
                        (task_id,),
                    )
            except BaseException as error:
                writer_errors.append(error)
            finally:
                competing.close()
                writer_done.set()

        writer = threading.Thread(target=mutate_task)
        writer.start()

        with kbc.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET lifecycle_contract = ? WHERE id = ?",
                (json.dumps({"kind": "general"}), task_id),
            )
            writer_start.set()
            accepted = evidence._emit_acceptance_changes(
                conn, before, source_task_id=task_id,
            )
            assert accepted == []
            assert not writer_acquired.is_set()
            event = conn.execute(
                "SELECT payload FROM task_events "
                "WHERE task_id = ? AND kind = 'acceptance_changed' "
                "ORDER BY id DESC LIMIT 1",
                (task_id,),
            ).fetchone()
            assert event is not None
            payload = json.loads(event["payload"])
            assert (payload["old"], payload["new"]) == ("unclassified", "not_applicable")

        assert writer_done.wait(timeout=5)
        writer.join(timeout=5)
        assert not writer.is_alive()
        assert writer_errors == []
        assert get_lifecycle_state(conn, task_id)["acceptance"] == "unclassified"
    finally:
        conn.close()





def test_misbehaving_hook_does_not_break_transition(kanban_home, monkeypatch):
    """A hook callback that raises must not break the board transition."""
    mgr = get_plugin_manager()
    saved = {k: list(v) for k, v in mgr._hooks.items()}

    def _boom(**kw):
        raise RuntimeError("plugin exploded")

    mgr._hooks.setdefault("kanban_task_completed", []).append(_boom)
    try:
        conn = kbc.connect()
        try:
            tid = kb.create_task(conn, title="t", assignee="worker")
            kb.claim_task(conn, tid)
            # Despite the raising hook, completion succeeds and persists.
            assert kb.complete_task(conn, tid, summary="ok") is True
            assert kb.get_task(conn, tid).status == "done"
        finally:
            conn.close()
    finally:
        mgr._hooks = saved

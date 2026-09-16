"""Regressions for domain-layer descendant invalidation on ancestor reopen.

``kanban_db.invalidate_descendants_for_parent_reopen`` is the single
implementation of "a done ancestor was reopened, retract everything that
assumed its result" (M3). These tests pin:

* done descendants are demoted to ``todo`` with a ``descendant_invalidated``
  event AND a comment naming the ancestor (non-silent),
* running descendants have their audit trail committed BEFORE their worker
  is terminated, and the kill routes through ``_terminate_reclaimed_worker``
  (the same helper the reclaim paths use),
* ``consecutive_failures`` resets to 0 (deliberate operator action —
  opposite of the review-loop rule pinned in M2), and
* the dashboard ``_set_status_direct`` reopen path and the DB function
  produce identical descendant outcomes.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli.kanban_lifecycle import get_lifecycle_state, lifecycle_metadata


@pytest.fixture
def conn(tmp_path: Path):
    db = kbc.connect(tmp_path / "kanban.db")
    try:
        yield db
    finally:
        db.close()


@pytest.fixture
def dashboard_api():
    pytest.importorskip("fastapi")
    plugin_file = (
        Path(__file__).resolve().parents[2]
        / "plugins" / "kanban" / "dashboard" / "plugin_api.py"
    )
    spec = importlib.util.spec_from_file_location(
        "hermes_dashboard_plugin_kanban_reopen_test", plugin_file,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(spec.name, None)


def _accepted_candidate(conn, *, parents=(), review_mode="same_card"):
    """Persist historical, fresh evidence; exercise the real read projection."""
    candidate = kb.create_task(
        conn, title="accepted candidate", assignee="builder", parents=parents,
        lifecycle_contract={
            "kind": "code", "review_mode": review_mode,
            "reviewer": "reviewer", "validation_required": False,
        },
    )
    review = candidate
    if review_mode == "separate_card":
        review = kb.create_task(
            conn, title="accepted review", assignee="reviewer", parents=[candidate],
            lifecycle_contract={"kind": "review", "candidate_task_id": candidate},
        )
    with kb.write_txn(conn):
        run_id = kb._synthesize_ended_run(conn, candidate, outcome="completed")
        implementation = lifecycle_metadata(
            conn, candidate, phase="implementation", run_id=run_id,
            verdict=None, head_sha="a" * 40,
        )
        conn.execute(
            "UPDATE task_runs SET metadata = ? WHERE id = ?",
            (json.dumps({"lifecycle": implementation}), run_id),
        )
        conn.execute(
            "UPDATE tasks SET candidate_run_id = ? WHERE id = ?", (run_id, candidate),
        )
        kb._synthesize_ended_run(
            conn, review, outcome="completed",
            metadata={"lifecycle": lifecycle_metadata(
                conn, review, phase="review", run_id=run_id,
                verdict="APPROVE", head_sha="a" * 40,
            )},
        )
        conn.execute(
            "UPDATE tasks SET status = 'done', completed_at = 123456, result = 'obsolete' "
            "WHERE id IN (?, ?)", (candidate, review),
        )
    assert get_lifecycle_state(conn, candidate)["acceptance"] == "accepted"
    return candidate, review


def _done_parent_with_done_child(conn):
    parent_id = kb.create_task(conn, title="ancestor", assignee="planner")
    assert kb.complete_task(conn, parent_id)
    child_id = kb.create_task(
        conn, title="child", assignee="builder", parents=[parent_id],
    )
    assert kb.complete_task(conn, child_id, summary="child result")
    return parent_id, child_id


def _reopen_parent_directly(conn, parent_id: str) -> None:
    """Minimal stand-in for a reopen surface: flip done -> todo."""
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET status = 'todo', completed_at = NULL WHERE id = ?",
            (parent_id,),
        )


def test_reopen_demotes_done_descendants_with_events_and_comments(conn):
    parent_id, child_id = _done_parent_with_done_child(conn)
    grandchild_id = kb.create_task(
        conn, title="grandchild", assignee="writer", parents=[child_id],
    )
    assert kb.complete_task(conn, grandchild_id, summary="grandchild result")
    versions = {
        tid: kb.get_task(conn, tid).version for tid in (child_id, grandchild_id)
    }

    _reopen_parent_directly(conn, parent_id)
    result = kb.invalidate_descendants_for_parent_reopen(
        conn, parent_id, author="operator",
    )

    demoted = {entry["id"]: entry for entry in result["invalidated"]}
    assert set(demoted) == {child_id, grandchild_id}
    for tid in (child_id, grandchild_id):
        assert demoted[tid]["prior_status"] == "done"
        assert demoted[tid]["new_status"] == "todo"
        task = kb.get_task(conn, tid)
        assert task is not None and task.status == "todo"
        assert task.completed_at is None
        assert task.result is None
        assert task.version == versions[tid] + 1
        with pytest.raises(kb.TaskUpdateConflict):
            kb.update_task(
                conn, tid, expected_version=versions[tid],
                reason="stale editor", body="must not overwrite invalidated output",
            )

        events = kb.list_events(conn, tid)
        inval = [e for e in events if e.kind == "descendant_invalidated"]
        assert len(inval) == 1
        payload = inval[0].payload
        assert payload["ancestor"] == parent_id
        assert payload["prior_status"] == "done"
        assert payload["new_status"] == "todo"

        comments = kb.list_comments(conn, tid)
        assert any(
            parent_id in c.body and c.author == "operator" for c in comments
        ), f"no invalidation comment naming {parent_id} on {tid}"

    assert result["terminations"] == []


def test_running_descendant_event_precedes_termination_via_reclaim_helper(
    conn, tmp_path, monkeypatch,
):
    parent_id = kb.create_task(conn, title="ancestor", assignee="planner")
    assert kb.complete_task(conn, parent_id)
    child_id = kb.create_task(
        conn, title="running child", assignee="builder", parents=[parent_id],
    )
    claimed = kb.claim_task(conn, child_id)
    assert claimed is not None and claimed.status == "running"
    kbd._set_worker_pid(conn, child_id, 424242)
    prior_version = kb.get_task(conn, child_id).version

    kills: list[tuple] = []

    def fake_terminate(pid, claim_lock, **kwargs):
        # The audit trail must already be durable when the kill fires:
        # standalone calls commit before terminating.
        side = kbc.connect(tmp_path / "kanban.db")
        try:
            kinds = [e.kind for e in kb.list_events(side, child_id)]
        finally:
            side.close()
        assert "descendant_invalidated" in kinds
        kills.append((pid, claim_lock))
        return {"terminated": True}

    monkeypatch.setattr(kb, "_terminate_reclaimed_worker", fake_terminate)

    _reopen_parent_directly(conn, parent_id)
    result = kb.invalidate_descendants_for_parent_reopen(
        conn, parent_id, author="operator",
    )

    assert kills and kills[0][0] == 424242
    assert result["terminations"] == kills
    child = kb.get_task(conn, child_id)
    assert child is not None
    assert child.status == "todo"
    assert child.current_run_id is None
    assert child.version == prior_version + 1
    assert child.claim_lock is None
    assert child.claim_expires is None
    assert child.worker_pid is None
    with pytest.raises(kb.TaskUpdateConflict):
        kb.update_task(
            conn, child_id, expected_version=prior_version, reason="stale worker editor",
            body="must not revise after reclaim",
        )
    run = kb.latest_run(conn, child_id)
    assert run is not None and run.outcome == "reclaimed"


def test_counter_reset_on_invalidated_descendants(conn):
    parent_id, child_id = _done_parent_with_done_child(conn)
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET consecutive_failures = 4 WHERE id = ?",
            (child_id,),
        )

    _reopen_parent_directly(conn, parent_id)
    kb.invalidate_descendants_for_parent_reopen(conn, parent_id, author="op")

    child = kb.get_task(conn, child_id)
    assert child is not None
    # Deliberate operator action = fresh start with the breaker; contrast
    # with reopen_review_task, which PRESERVES the counter (M2 rule).
    assert child.consecutive_failures == 0


def test_dashboard_and_db_paths_produce_identical_outcomes(
    tmp_path, monkeypatch, dashboard_api,
):
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()

    app = fastapi.FastAPI()
    app.include_router(dashboard_api.router, prefix="/api/plugins/kanban")
    client = TestClient(app)

    def build_graph(tag: str):
        with kbc.connect() as c:
            parent = kb.create_task(c, title=f"{tag}-parent", assignee="planner")
            assert kb.complete_task(c, parent)
            child = kb.create_task(
                c, title=f"{tag}-child", assignee="builder", parents=[parent],
            )
            assert kb.complete_task(c, child)
        return parent, child

    dash_parent, dash_child = build_graph("dash")
    db_parent, db_child = build_graph("db")

    # Surface 1: dashboard drag (done -> todo) via _set_status_direct.
    r = client.patch(
        f"/api/plugins/kanban/tasks/{dash_parent}", json={"status": "todo"},
    )
    assert r.status_code == 200, r.text

    # Surface 2: DB function directly (the single domain implementation).
    with kbc.connect() as c:
        with kb.write_txn(c):
            c.execute(
                "UPDATE tasks SET status = 'todo', completed_at = NULL "
                "WHERE id = ?",
                (db_parent,),
            )
        kb.invalidate_descendants_for_parent_reopen(
            c, db_parent, author="dashboard",
        )

    with kbc.connect() as c:
        def snapshot(tid: str):
            t = kb.get_task(c, tid)
            assert t is not None
            kinds = sorted(e.kind for e in kb.list_events(c, tid))
            n_comments = len(kb.list_comments(c, tid))
            return (
                t.status,
                t.completed_at,
                t.current_run_id,
                t.consecutive_failures,
                kinds,
                n_comments,
            )

        assert snapshot(dash_child) == snapshot(db_child)
        status, completed_at, _run, failures, kinds, n_comments = snapshot(db_child)
        assert status == "todo"
        assert completed_at is None
        assert failures == 0
        assert "descendant_invalidated" in kinds
        assert n_comments >= 1


@pytest.mark.parametrize(
    "reopen",
    ["typed_revision", "legacy_binding", "standalone", "general_dashboard", "legacy_dashboard"],
)
def test_recursive_reopen_retracts_each_candidate_once(conn, request, reopen):
    if reopen == "typed_revision":
        parent_id, _ = _accepted_candidate(conn)
    else:
        parent_id = kb.create_task(conn, title="ancestor", assignee="planner")
        assert kb.complete_task(conn, parent_id)
    bridge = kb.create_task(conn, title="bridge", parents=[parent_id])
    assert kb.complete_task(conn, bridge)
    candidate, review = _accepted_candidate(
        conn, parents=[bridge], review_mode="separate_card",
    )
    nested_bridge = kb.create_task(conn, title="nested bridge", parents=[review])
    assert kb.complete_task(conn, nested_bridge)
    nested, _ = _accepted_candidate(conn, parents=[nested_bridge])
    ready = kb.create_task(conn, title="ready descendant", parents=[nested])
    in_review = kb.create_task(conn, title="review descendant", parents=[nested], assignee="builder")
    assert kb.request_review(conn, in_review, reviewer="reviewer")
    descendants = (bridge, candidate, review, nested_bridge, nested, ready, in_review)
    if reopen.startswith("legacy"):
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET lifecycle_contract = NULL WHERE id = ?", (parent_id,),
            )
            conn.execute(
                "UPDATE task_links SET requirement = NULL WHERE parent_id = ?", (parent_id,),
            )
    before = {tid: kb.get_task(conn, tid) for tid in (parent_id, *descendants)}
    historical_runs = [tuple(row) for row in conn.execute("SELECT * FROM task_runs ORDER BY id")]
    if reopen == "typed_revision":
        assert kb.update_task(
            conn, parent_id, expected_version=before[parent_id].version,
            reason="revise accepted ancestor", body="new implementation scope",
        )
    elif reopen == "legacy_binding":
        assert kb.bind_lifecycle_contract(
            conn, parent_id,
            {"kind": "code", "review_mode": "separate_card", "reviewer": "reviewer",
             "validation_required": False},
            expected_version=before[parent_id].version, reason="classify historical ancestor",
        )
    elif reopen == "standalone":
        _reopen_parent_directly(conn, parent_id)
        kb.invalidate_descendants_for_parent_reopen(conn, parent_id, author="operator")
    else:
        dashboard_api = request.getfixturevalue("dashboard_api")
        assert dashboard_api._set_status_direct(conn, parent_id, "todo")
    assert [tuple(row) for row in conn.execute("SELECT * FROM task_runs ORDER BY id")] == historical_runs

    for tid in descendants:
        task = kb.get_task(conn, tid)
        assert task.status == "todo"
        # Binding also revises the immediate edge endpoint's CAS version.
        bumps = 2 if reopen == "legacy_binding" and tid == bridge else 1
        assert task.version == before[tid].version + bumps
        assert (
            task.completed_at, task.result, task.candidate_run_id, task.current_run_id,
            task.claim_lock, task.claim_expires, task.worker_pid,
        ) == (None,) * 7
        with pytest.raises(kb.TaskUpdateConflict):
            kb.update_task(
                conn, tid, expected_version=before[tid].version,
                reason="stale descendant edit", title="must not land",
            )
    candidates = (parent_id, candidate, nested) if reopen == "typed_revision" else (candidate, nested)
    for tid in candidates:
        state = get_lifecycle_state(conn, tid)["acceptance"]
        assert state in {"pending", "stale"}
        changes = [
            event.payload for event in kb.list_events(conn, tid)
            if event.kind == "acceptance_changed"
        ]
        assert [(change["old"], change["new"], change["source_task_id"]) for change in changes] == [
            ("accepted", state, parent_id),
        ]


def test_outer_rollback_preserves_descendants_events_and_worker(conn, monkeypatch):
    parent_id, _ = _accepted_candidate(conn)
    bridge = kb.create_task(conn, title="bridge", parents=[parent_id])
    assert kb.complete_task(conn, bridge)
    candidate, review = _accepted_candidate(
        conn, parents=[bridge], review_mode="separate_card",
    )
    running = kb.create_task(conn, title="running descendant", parents=[review], assignee="worker")
    claimed = kb.claim_task(conn, running)
    assert claimed is not None
    kbd._set_worker_pid(conn, running, 424242)
    tables = ("tasks", "task_runs", "task_events", "task_comments", "task_goal_revisions")
    before = {
        table: [tuple(row) for row in conn.execute(f"SELECT * FROM {table} ORDER BY id")]
        for table in tables
    }
    signals = []

    def signal_worker(pid, sig):
        assert not conn.in_transaction
        assert kb.get_task(conn, running).status == "todo"
        signals.append((pid, sig))
        raise ProcessLookupError

    monkeypatch.setattr(kbd, "_kill_fn", lambda _signal_fn: signal_worker)
    with pytest.raises(RuntimeError, match="abort outer request"):
        with kbc.composite_write_txn(conn):
            assert kb.update_task(
                conn, parent_id, expected_version=kb.get_task(conn, parent_id).version,
                reason="revise accepted ancestor", body="uncommitted replacement",
            )
            assert kb.get_task(conn, running).status == "todo"
            assert get_lifecycle_state(conn, candidate)["acceptance"] == "stale"
            assert [
                event.payload["old"] for event in kb.list_events(conn, candidate)
                if event.kind == "acceptance_changed"
            ] == ["accepted"]
            assert signals == []
            raise RuntimeError("abort outer request")
    assert signals == []
    for table in tables:
        assert [tuple(row) for row in conn.execute(f"SELECT * FROM {table} ORDER BY id")] == before[table]
    with kbc.composite_write_txn(conn):
        assert kb.update_task(
            conn, parent_id, expected_version=kb.get_task(conn, parent_id).version,
            reason="commit ancestor revision", body="committed replacement",
        )
        assert signals == []
    assert len(signals) == 1 and signals[0][0] == 424242
    assert kb.latest_run(conn, running).outcome == "reclaimed"

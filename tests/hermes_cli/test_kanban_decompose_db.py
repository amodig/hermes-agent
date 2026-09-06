"""Tests for kb.decompose_triage_task — the DB-layer atomic fan-out
from the triage column. LLM-free by design.
"""

from __future__ import annotations

from pathlib import Path

import pytest

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


def _create_triage(conn, title="rough idea", body=None, assignee=None, tenant=None):
    return kb.create_task(
        conn,
        title=title,
        body=body,
        assignee=assignee,
        tenant=tenant,
        triage=True,
    )


def test_decompose_creates_children_and_promotes_root(kanban_home):
    with kbc.connect() as conn:
        tid = _create_triage(conn, title="ship a feature")
        assert kb.get_task(conn, tid).status == "triage"

    children = [
        {"title": "research", "body": "look at prior art", "assignee": "researcher", "parents": []},
        {"title": "build it", "body": "write code", "assignee": "engineer", "parents": [0]},
    ]
    with kbc.connect() as conn:
        child_ids = kb.decompose_triage_task(
            conn,
            tid,
            root_assignee="orchestrator",
            children=children,
            author="decomposer",
        )
    assert child_ids is not None
    assert len(child_ids) == 2

    with kbc.connect() as conn:
        root = kb.get_task(conn, tid)
        c0 = kb.get_task(conn, child_ids[0])
        c1 = kb.get_task(conn, child_ids[1])

    # Root flipped to todo with orchestrator assignee, gated by children.
    assert root.status == "todo"
    assert root.assignee == "orchestrator"
    # First child has no internal parents → ready on recompute_ready.
    assert c0.status == "ready"
    assert c0.assignee == "researcher"
    # Second child has parents=[0] → stays in todo until c0 completes.
    assert c1.status == "todo"
    assert c1.assignee == "engineer"


def test_decompose_records_audit_comment_and_event(kanban_home):
    with kbc.connect() as conn:
        tid = _create_triage(conn)
        comment_id = kb.add_comment(
            conn, tid, "reviewer", "Correct the command before decomposition."
        )
        child_ids = kb.decompose_triage_task(
            conn,
            tid,
            root_assignee="orch",
            children=[{"title": "task A", "assignee": "researcher"}],
            author="alice",
        )
    assert child_ids is not None

    with kbc.connect() as conn:
        comments = kb.list_comments(conn, tid)
        events = kb.list_events(conn, tid)
    decomposed = [event for event in events if event.kind == "decomposed"]
    assert decomposed
    assert decomposed[-1].payload["comment_ids_seen"] == [comment_id]

    assert any("Decomposed into" in (c.body or "") for c in comments)
    assert any(ev.kind == "decomposed" for ev in events)


def test_decompose_db_proposes_for_existing_graph_without_mutation(kanban_home):
    with kbc.connect() as conn:
        root = _create_triage(
            conn,
            title="existing graph",
            body="Do not recreate the implementation graph.",
            assignee="implementer",
        )
        reviewer = kb.create_task(
            conn, title="review implementation", assignee="reviewer"
        )
        kb.link_tasks(conn, root, reviewer)
        comment_id = kb.add_comment(
            conn, root, "reviewer", "Keep the corrective review context."
        )
        before_root = kb.get_task(conn, root)
        before_reviewer = kb.get_task(conn, reviewer)
        before_edges = (kb.parent_ids(conn, root), kb.child_ids(conn, root))

        result = kb.decompose_triage_task(
            conn,
            root,
            root_assignee="orchestrator",
            children=[{"title": "must not be inserted", "assignee": "engineer"}],
            author="decomposer",
        )
        assert result is None

        after_root = kb.get_task(conn, root)
        after_reviewer = kb.get_task(conn, reviewer)
        assert after_root is not None and before_root is not None
        assert after_reviewer is not None and before_reviewer is not None
        assert (
            after_root.status,
            after_root.assignee,
            after_root.version,
        ) == (
            before_root.status,
            before_root.assignee,
            before_root.version,
        )
        assert (
            after_reviewer.status,
            after_reviewer.assignee,
            after_reviewer.version,
        ) == (
            before_reviewer.status,
            before_reviewer.assignee,
            before_reviewer.version,
        )
        assert (kb.parent_ids(conn, root), kb.child_ids(conn, root)) == before_edges
        assert len(kb.list_tasks(conn, include_archived=True)) == 2
        proposals = [
            event
            for event in kb.list_events(conn, root)
            if event.kind == "decompose_proposed"
        ]

    assert len(proposals) == 1
    payload = proposals[0].payload
    assert payload["dry_run"] is True
    assert payload["mutation"] is False
    assert payload["comment_ids"] == [comment_id]
    assert [child["id"] for child in payload["children"]] == [reviewer]





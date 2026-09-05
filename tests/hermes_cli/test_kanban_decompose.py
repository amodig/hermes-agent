"""Tests for the decomposer module + `hermes kanban decompose` CLI surface.

The auxiliary LLM client is mocked — no network calls. Tests exercise the
prompt plumbing, response parsing, DB writes (via the real DB helper),
and the assignee-fallback logic.
"""

from __future__ import annotations

import json as jsonlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_decompose as decomp


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _fake_aux_response(content: str):
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    return resp


def _mock_client_returning(content: str):
    client = MagicMock()
    client.chat.completions.create = MagicMock(return_value=_fake_aux_response(content))
    return client


def _patch_aux_client(content: str, *, model: str = "test-model"):
    # decompose_task now routes through call_llm (see #35566) — mock it at
    # the source module so task config, extra_body, and retries stay out of
    # unit-test scope.
    return patch(
        "agent.auxiliary_client.call_llm",
        return_value=_fake_aux_response(content),
    )


def _patch_extra_body():
    # No-op shim retained for call-site compatibility: extra_body plumbing
    # now lives inside call_llm, which _patch_aux_client already mocks.
    return patch("agent.auxiliary_client.get_auxiliary_extra_body", return_value={})


def _patch_list_profiles(names: list[str]):
    """Pretend the named profiles exist. The decomposer uses
    profiles_mod.list_profiles() to build the roster + valid-set, and
    profiles_mod.profile_exists() to resolve orchestrator/default."""
    from types import SimpleNamespace
    fake_profiles = [
        SimpleNamespace(
            name=n, is_default=(i == 0), description=f"desc for {n}",
            description_auto=False, model="m", provider="p", skill_count=1,
        )
        for i, n in enumerate(names)
    ]
    return [
        patch("hermes_cli.profiles.list_profiles", return_value=fake_profiles),
        patch("hermes_cli.profiles.profile_exists", side_effect=lambda x: x in names),
        patch("hermes_cli.profiles.get_active_profile_name", return_value=names[0] if names else "default"),
    ]


def test_decompose_with_fanout_creates_children(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn,
            title="ship a feature",
            body="Use the supported hermes-github command.",
            triage=True,
            idempotency_key="feature-42",
        )
        correction_id = kb.add_comment(
            conn, tid, "cto", "Use the supported hermes-github command."
        )

    llm_payload = jsonlib.dumps({
        "fanout": True,
        "rationale": "test split",
        "tasks": [
            {"title": "research", "body": "look it up", "assignee": "researcher", "parents": []},
            {"title": "build", "body": "code it", "assignee": "engineer", "parents": [0]},
        ],
    })

    patches = _patch_list_profiles(["orchestrator", "researcher", "engineer"])
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(llm_payload) as aux, _patch_extra_body():
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason
    assert outcome.fanout is True
    assert outcome.child_ids and len(outcome.child_ids) == 2

    with kbc.connect() as conn:
        root = kb.get_task(conn, tid)
        c0 = kb.get_task(conn, outcome.child_ids[0])
        c1 = kb.get_task(conn, outcome.child_ids[1])
    assert root.status == "todo"
    assert c0.status == "ready"
    assert c1.status == "todo"
    assert c0.assignee == "researcher"
    assert c1.assignee == "engineer"
    prompt = aux.call_args.kwargs["messages"][1]["content"]
    context_marker = (
        "Prior task purpose and graph/history (authoritative; do not recreate work\n"
        "already represented here):\n"
    )
    context_text = (
        prompt.split(context_marker, 1)[1].split("\n\nAvailable profiles", 1)[0].strip()
    )
    context = jsonlib.loads(context_text)
    assert context["comments"][0]["id"] == correction_id
    assert context["comments"][0]["body"] == "Use the supported hermes-github command."
    assert context["parents"] == []
    assert context["children"] == []
    assert context["descendants"] == []
    assert context["nontrivial_graph"] is False
    assert context["prior_purpose"]["title"] == "ship a feature"
    assert (
        context["prior_purpose"]["body"] == "Use the supported hermes-github command."
    )
    assert context["prior_purpose"]["idempotency_key"] == "feature-42"
    with kbc.connect() as conn:
        events = [
            event for event in kb.list_events(conn, tid) if event.kind == "decomposed"
        ]
    assert events[-1].payload["comment_ids_seen"] == [correction_id]
    outcome_again = decomp.decompose_task(tid, author="me")
    assert outcome_again.ok is False
    assert aux.call_count == 1
    with kbc.connect() as conn:
        assert len(kb.list_tasks(conn, include_archived=True)) == 3


def test_decompose_fanout_false_invalid_llm_assignee_uses_default(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="route me safely", triage=True)

    llm_payload = jsonlib.dumps({
        "fanout": False,
        "rationale": "single unit",
        "title": "Tightened title",
        "body": "Route to fallback.",
        "assignee": "made_up",
    })

    patches = _patch_list_profiles(["orchestrator", "fallback"])
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(llm_payload), _patch_extra_body(), patch(
            "hermes_cli.kanban_decompose._load_config",
            return_value={"kanban": {"default_assignee": "fallback"}},
        ):
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok, outcome.reason
    with kbc.connect() as conn:
        task = kb.get_task(conn, tid)
    assert task is not None
    assert task.assignee == "fallback"


def test_decompose_returns_false_when_task_not_triage(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="x")  # ready, not triage

    patches = _patch_list_profiles(["orchestrator"])
    for p in patches:
        p.start()
    try:
        outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()
    assert outcome.ok is False
    assert "not in triage" in outcome.reason



def test_decompose_skips_capability_block_with_existing_review_graph(kanban_home):
    with kbc.connect() as conn:
        root = kb.create_task(
            conn, title="implement the change", assignee="implementer"
        )
        reviewer = kb.create_task(
            conn, title="review the implementation", assignee="reviewer", parents=[root]
        )
        tester = kb.create_task(
            conn, title="test the implementation", assignee="tester", parents=[reviewer]
        )
        assert kb.claim_task(conn, root, claimer="implementer") is not None
        assert kb.block_task(
            conn,
            root,
            reason="Authorization is required from a human approver",
        )
        assert kb.unblock_task(conn, root)
        assert kb.claim_task(conn, root, claimer="implementer") is not None
        assert kb.block_task(
            conn,
            root,
            reason="GitHub access is unavailable",
        )
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status = 'triage' WHERE id = ?", (root,))
        correction_id = kb.add_comment(
            conn, root, "cto", "Use the supported hermes-github command."
        )
        before = {
            task_id: (
                kb.get_task(conn, task_id).status,
                kb.get_task(conn, task_id).assignee,
                kb.get_task(conn, task_id).version,
            )
            for task_id in (root, reviewer, tester)
        }
        assert kb.get_task(conn, root).block_kind == "capability"
        assert kb.get_task(conn, root).block_recurrences == 1
        before_edges = (
            kb.parent_ids(conn, root),
            kb.child_ids(conn, root),
            kb.parent_ids(conn, reviewer),
            kb.child_ids(conn, reviewer),
            kb.parent_ids(conn, tester),
            kb.child_ids(conn, tester),
        )

    with _patch_aux_client("{}") as aux:
        outcome = decomp.decompose_task(root, author="auto-decomposer")

    assert outcome.ok is False
    assert "capability" in outcome.reason
    aux.assert_not_called()
    with kbc.connect() as conn:
        after = {
            task_id: (
                kb.get_task(conn, task_id).status,
                kb.get_task(conn, task_id).assignee,
                kb.get_task(conn, task_id).version,
            )
            for task_id in (root, reviewer, tester)
        }
        assert after == before
        assert (
            kb.parent_ids(conn, root),
            kb.child_ids(conn, root),
            kb.parent_ids(conn, reviewer),
            kb.child_ids(conn, reviewer),
            kb.parent_ids(conn, tester),
            kb.child_ids(conn, tester),
        ) == before_edges
        assert kb.list_comments(conn, root)[-1].id == correction_id
        assert len(kb.list_tasks(conn, include_archived=True)) == 3


def test_decompose_proposes_dry_run_for_existing_graph_without_mutation(kanban_home):
    with kbc.connect() as conn:
        root = kb.create_task(
            conn,
            title="rough implementation",
            body="Preserve the existing implementation graph.",
            assignee="implementer",
            triage=True,
            idempotency_key="implementation-17",
        )
        reviewer = kb.create_task(
            conn, title="review implementation", assignee="reviewer"
        )
        kb.link_tasks(conn, root, reviewer)
        comment_id = kb.add_comment(
            conn, root, "reviewer", "Correct the command before another attempt."
        )
        before_root = kb.get_task(conn, root)
        before_reviewer = kb.get_task(conn, reviewer)
        before_edges = (kb.parent_ids(conn, root), kb.child_ids(conn, root))

    with _patch_aux_client("{}") as aux:
        outcome = decomp.decompose_task(root, author="auto-decomposer")

    assert outcome.ok is False
    assert "existing task graph" in outcome.reason
    aux.assert_not_called()
    with kbc.connect() as conn:
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
    assert payload["prior_purpose"]["title"] == "rough implementation"
    assert payload["prior_purpose"]["idempotency_key"] == "implementation-17"


def test_decompose_rechecks_graph_and_comments_before_mutating(kanban_home):
    with kbc.connect() as conn:
        root = kb.create_task(conn, title="race-safe decomposition", triage=True)
        existing_child = kb.create_task(
            conn, title="review already queued", assignee="reviewer"
        )

    llm_payload = jsonlib.dumps({
        "fanout": True,
        "rationale": "split the work",
        "tasks": [{"title": "new child", "assignee": "orchestrator", "parents": []}],
    })
    race_comment_id = None

    def add_graph_during_aux_call(*_args, **_kwargs):
        nonlocal race_comment_id
        with kbc.connect() as conn:
            kb.link_tasks(conn, root, existing_child)
            race_comment_id = kb.add_comment(
                conn, root, "reviewer", "The existing reviewer graph is authoritative."
            )
        return _fake_aux_response(llm_payload)

    patches = _patch_list_profiles(["orchestrator"])
    for p in patches:
        p.start()
    try:
        with _patch_aux_client(llm_payload) as aux:
            aux.side_effect = add_graph_during_aux_call
            outcome = decomp.decompose_task(root, author="auto-decomposer")
    finally:
        for p in patches:
            p.stop()

    assert outcome.ok is False
    assert aux.call_count == 1
    with kbc.connect() as conn:
        assert kb.get_task(conn, root).status == "triage"
        assert kb.child_ids(conn, root) == [existing_child]
        assert len(kb.list_tasks(conn, include_archived=True)) == 2
        proposals = [
            event
            for event in kb.list_events(conn, root)
            if event.kind == "decompose_proposed"
        ]
    assert len(proposals) == 1
    assert race_comment_id is not None
    assert race_comment_id in proposals[0].payload["comment_ids"]


def test_decompose_rejects_goal_revision_race_without_mutation(kanban_home):
    with kbc.connect() as conn:
        root = kb.create_task(
            conn,
            title="rough decomposition",
            body="Initial decomposition goal.",
            assignee="owner",
            triage=True,
        )

    llm_payload = jsonlib.dumps({
        "fanout": True,
        "rationale": "split the work",
        "tasks": [
            {
                "title": "obsolete child",
                "body": "Must not be inserted after the goal changes.",
                "assignee": "engineer",
                "parents": [],
            }
        ],
    })

    def revise_goal_during_aux_call(*_args, **_kwargs):
        with kbc.connect_closing() as conn:
            assert kb.update_task(
                conn,
                root,
                title="authoritative revised goal",
                body="The goal changed while decomposition was in flight.",
                expected_version=1,
                reason="concurrent goal revision",
                author="cto",
            )
        return _fake_aux_response(llm_payload)

    patches = _patch_list_profiles(["orchestrator", "engineer"])
    for patcher in patches:
        patcher.start()
    try:
        with _patch_aux_client(llm_payload) as aux:
            aux.side_effect = revise_goal_during_aux_call
            outcome = decomp.decompose_task(root, author="auto-decomposer")
    finally:
        for patcher in patches:
            patcher.stop()

    assert outcome.ok is False
    assert outcome.child_ids is None
    assert "goal revision" in outcome.reason
    assert aux.call_count == 1
    with kbc.connect_closing() as conn:
        task = kb.get_task(conn, root)
        proposals = [
            event
            for event in kb.list_events(conn, root)
            if event.kind == "decompose_proposed"
        ]
        children = kb.child_ids(conn, root)
        task_count = len(kb.list_tasks(conn, include_archived=True))

    assert task is not None
    assert task.status == "triage"
    assert task.assignee == "owner"
    assert task.version == 2
    assert task.title == "authoritative revised goal"
    assert children == []
    assert task_count == 1
    assert proposals
    assert proposals[-1].payload["dry_run"] is True
    assert proposals[-1].payload["mutation"] is False
    assert "goal revision" in proposals[-1].payload["reason"]

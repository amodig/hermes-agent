"""Regression coverage for intentional waiting/approval gates."""

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _task(conn, title="waiting"):
    return kb.create_task(conn, title=title, assignee="worker")


def test_expected_gate_stays_blocked_and_deadlines_emit(kanban_home: Path) -> None:
    with kb.connect_closing() as conn:
        tid = _task(conn)
        assert kb.block_task(
            conn,
            tid,
            reason="approval",
            expected=True,
            notify=False,
            waiting_for="reviewer",
            wake_condition="approval recorded",
            reminder_at=10,
            escalation_at=20,
        )
        task = kb.get_task(conn, tid)
        assert task.status == "blocked"
        assert task.block_metadata == {
            "expected": True,
            "notify": False,
            "waiting_for": "reviewer",
            "wake_condition": "approval recorded",
            "reminder_at": 10,
            "escalation_at": 20,
        }
        assert kb.emit_due_gate_events(conn, now=20) == 2
        assert [
            event.kind for event in kb.list_events(conn, tid)
            if event.kind.startswith("gate_")
        ] == ["gate_reminder", "gate_escalation"]
        assert kb.unblock_task(conn, tid)
        release = [event for event in kb.list_events(conn, tid) if event.kind == "unblocked"][-1]
        assert release.payload["gate_released"] is True


def test_dependency_waiting_gate_does_not_auto_route_to_todo(kanban_home: Path) -> None:
    with kb.connect_closing() as conn:
        parent = _task(conn, "parent")
        child = _task(conn, "child")
        assert kb.block_task(
            conn, child, reason="approval", kind="dependency", expected=True,
        )
        kb.link_tasks(conn, parent_id=parent, child_id=child)
        assert kb.get_task(conn, child).status == "blocked"


def test_expected_gate_claim_rejection_is_audited(kanban_home: Path) -> None:
    with kb.connect_closing() as conn:
        tid = _task(conn)
        assert kb.block_task(conn, tid, reason="approval", expected=True)
        assert kb.claim_task(conn, tid) is None
        rejection = [
            event for event in kb.list_events(conn, tid)
            if event.kind == "claim_rejected"
        ][-1]
        assert rejection.payload["expected"] is True
        assert rejection.payload["gate"]["expected"] is True


def test_expected_to_unexpected_reblock_is_marked(kanban_home: Path) -> None:
    with kb.connect_closing() as conn:
        tid = _task(conn)
        assert kb.block_task(conn, tid, reason="approval", expected=True)
        assert kb.unblock_task(conn, tid)
        assert kb.block_task(conn, tid, reason="failure", expected=False)
        event = [event for event in kb.list_events(conn, tid) if event.kind == "blocked"][-1]
        assert event.payload["gate_transition"] == "expected_to_unexpected"


def test_ordinary_failure_block_remains_ordinary(kanban_home: Path) -> None:
    with kb.connect_closing() as conn:
        tid = _task(conn)
        assert kb.block_task(conn, tid, reason="failure")
        task = kb.get_task(conn, tid)
        event = [event for event in kb.list_events(conn, tid) if event.kind == "blocked"][-1]
        assert task.status == "blocked"
        assert task.block_metadata is None
        assert event.payload["expected"] is False

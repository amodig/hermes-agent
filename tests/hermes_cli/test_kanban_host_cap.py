"""Host-level concurrency accounting + review-lane fairness (OOF-30 review).

Three gaps found in review of the original memory-guard PR:

1. The standalone daemon path (``hermes kanban daemon --force`` /
   :func:`hermes_cli.kanban_db_dispatch.run_daemon`) never resolved
   ``kanban.max_in_progress`` at all — the one shipped entry point that
   could still fan out an entire backlog in a single tick.
2. ``max_in_progress`` was enforced per-board while the gateway dispatcher
   ticks every active board — N boards multiplied the host budget by N.
3. The ready loop consumed the entire shared spawn budget before the
   review loop ran, so a sustained ready backlog starved autonomous
   reviews indefinitely.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setattr(kbd, "_memory_pressure_level", lambda: "ok")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _set_task_status(conn: sqlite3.Connection, task_id: str, status: str) -> None:
    conn.execute("UPDATE tasks SET status = ? WHERE id = ?", (status, task_id))


def _fake_spawn_factory(spawns: list):
    def fake_spawn(task, workspace, board=None):
        spawns.append(task.id)
        return 42
    return fake_spawn


# ---------------------------------------------------------------------------
# 1. Standalone daemon resolves max_in_progress (P1a)
# ---------------------------------------------------------------------------


def test_run_daemon_resolves_and_passes_max_in_progress(
    kanban_home, monkeypatch,
):
    """The daemon tick must pass a resolved cap into dispatch_once.

    Regression guard for the OOF-30 review finding: ``run_daemon`` only
    forwarded ``max_spawn`` — with no explicit ``--max`` (the shipped
    systemd shape) nothing capped the tick even though the gateway and
    ``hermes kanban dispatch`` paths both resolved the memory-derived
    default.
    """
    captured: dict = {}
    stop = threading.Event()

    def fake_dispatch_once(conn, **kwargs):
        captured.update(kwargs)
        return kb.DispatchResult()

    monkeypatch.setattr(kbd, "dispatch_once", fake_dispatch_once)
    # No explicit config → the derived default must flow through.
    monkeypatch.setattr(kbd, "configured_max_in_progress", lambda: None)
    monkeypatch.setattr(kbd, "derive_default_max_in_progress", lambda sample=None: 3)

    def on_tick(res):
        stop.set()

    kbd.run_daemon(interval=0.01, stop_event=stop, on_tick=on_tick)

    assert captured.get("max_in_progress") == 3




def test_configured_max_in_progress_parsing(monkeypatch):
    import hermes_cli.config as cfgmod

    cases = [
        ({"kanban": {"max_in_progress": 4}}, 4),
        ({"kanban": {"max_in_progress": "5"}}, 5),
        ({"kanban": {"max_in_progress": 0}}, None),
        ({"kanban": {"max_in_progress": -2}}, None),
        ({"kanban": {"max_in_progress": "lots"}}, None),
        ({"kanban": {}}, None),
        ({}, None),
    ]
    for config, expected in cases:
        monkeypatch.setattr(
            cfgmod, "load_config_readonly", lambda c=config: c
        )
        assert kbd.configured_max_in_progress() == expected, config


# ---------------------------------------------------------------------------
# 2. max_in_progress counts running work on ALL boards (P1b)
# ---------------------------------------------------------------------------


def test_max_in_progress_counts_other_boards(
    kanban_home, all_assignees_spawnable,
):
    """Workers running on another board consume the same host budget."""
    kb.create_board("second")

    # Two workers already running on the second board.
    with kbc.connect(board="second") as conn:
        for title in ("busy-1", "busy-2"):
            tid = kb.create_task(conn, title=title, assignee="alice")
            assert kb.claim_task(conn, tid) is not None

    spawns: list = []
    with kbc.connect() as conn:
        kb.create_task(conn, title="wants-to-run", assignee="alice")
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn_factory(spawns), max_in_progress=2,
        )

    # Host budget (2) already consumed by the second board → nothing spawns.
    assert not spawns
    assert not res.spawned


def test_max_in_progress_partial_budget_across_boards(
    kanban_home, all_assignees_spawnable,
):
    kb.create_board("second")

    with kbc.connect(board="second") as conn:
        tid = kb.create_task(conn, title="busy", assignee="alice")
        assert kb.claim_task(conn, tid) is not None

    spawns: list = []
    with kbc.connect() as conn:
        for title in ("a", "b", "c"):
            kb.create_task(conn, title=title, assignee="alice")
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn_factory(spawns), max_in_progress=2,
        )

    # 1 running elsewhere + budget 2 → exactly one new spawn here.
    assert len(spawns) == 1
    assert len(res.spawned) == 1


def test_count_running_tasks_other_boards_fails_open(
    kanban_home, monkeypatch,
):
    """A broken board enumeration must not brick dispatch (returns 0)."""
    monkeypatch.setattr(
        kb, "list_boards",
        lambda **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert kbd.count_running_tasks_other_boards() == 0


def test_max_spawn_stays_per_board(kanban_home, all_assignees_spawnable):
    """``max_spawn`` keeps its historical per-board semantics."""
    kb.create_board("second")
    with kbc.connect(board="second") as conn:
        tid = kb.create_task(conn, title="busy", assignee="alice")
        assert kb.claim_task(conn, tid) is not None

    spawns: list = []
    with kbc.connect() as conn:
        kb.create_task(conn, title="a", assignee="alice")
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn_factory(spawns), max_spawn=1,
        )

    # The other board's worker does NOT count against max_spawn.
    assert len(spawns) == 1
    assert len(res.spawned) == 1


# ---------------------------------------------------------------------------
# 3. Review lane cannot be starved by a sustained ready backlog (P2)
# ---------------------------------------------------------------------------


def _park_in_review(conn: sqlite3.Connection, title: str, assignee: str) -> str:
    tid = kb.create_task(conn, title=title, assignee=assignee)
    _set_task_status(conn, tid, "review")
    return tid


def test_review_lane_gets_first_service_under_ready_backlog(
    kanban_home, all_assignees_spawnable, monkeypatch,
):
    import hermes_cli.config as cfgmod
    monkeypatch.setattr(
        cfgmod, "load_config",
        lambda *a, **k: {"kanban": {"review_dispatch": True}},
    )

    spawns: list = []
    with kbc.connect() as conn:
        for title in ("ready-1", "ready-2", "ready-3"):
            kb.create_task(conn, title=title, assignee="alice")
        review_id = _park_in_review(conn, "review-me", "reviewer")
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn_factory(spawns), max_in_progress=2,
        )

    spawned_ids = [s[0] for s in res.spawned]
    # Budget 2: one review followed by READY — never 2×READY.
    assert len(spawned_ids) == 2
    assert review_id in spawned_ids


def _guard_review_row(conn: sqlite3.Connection, review_id: str) -> dict:
    """Latest run ``rate_limited`` → ``check_respawn_guard`` returns a cooldown."""
    now = int(time.time())
    with kb.write_txn(conn):
        conn.execute(
            "INSERT INTO task_runs (task_id, profile, status, outcome, "
            "started_at, ended_at) VALUES (?, 'reviewer', 'rate_limited', "
            "'rate_limited', ?, ?)",
            (review_id, now, now),
        )
    assert kbd.check_respawn_guard(conn, review_id, lane="review") == "rate_limit_cooldown"
    return {"max_in_progress": 1}


def _cap_review_row(conn: sqlite3.Connection, review_id: str) -> dict:
    """``reviewer`` already has one running worker → the review row is per-profile capped."""
    busy_id = kb.create_task(conn, title="busy", assignee="reviewer")
    assert kb.claim_task(conn, busy_id) is not None
    return {"max_in_progress": 2, "max_in_progress_per_profile": 1}


@pytest.mark.parametrize("dry_run", [False, True])
@pytest.mark.parametrize("make_unspawnable", [_guard_review_row, _cap_review_row])
def test_unspawnable_review_does_not_consume_the_only_ready_slot(
    kanban_home, all_assignees_spawnable, monkeypatch, make_unspawnable, dry_run,
):
    """A guarded or profile-capped review must leave capacity for READY."""
    import hermes_cli.config as cfgmod

    monkeypatch.setattr(
        cfgmod, "load_config",
        lambda *a, **k: {"kanban": {"review_dispatch": True}},
    )

    spawns: list = []
    with kbc.connect() as conn:
        ready_id = kb.create_task(conn, title="ready-now", assignee="alice")
        review_id = _park_in_review(conn, "review-unspawnable", "reviewer")
        caps = make_unspawnable(conn, review_id)
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn_factory(spawns), dry_run=dry_run, **caps,
        )

    assert [task_id for task_id, *_ in res.spawned] == [ready_id]


def test_unguarded_review_gets_the_only_ready_slot(
    kanban_home, all_assignees_spawnable, monkeypatch,
):
    """A dispatchable review card still receives the single shared slot."""
    import hermes_cli.config as cfgmod

    monkeypatch.setattr(
        cfgmod, "load_config",
        lambda *a, **k: {"kanban": {"review_dispatch": True}},
    )

    spawns: list = []
    with kbc.connect() as conn:
        kb.create_task(conn, title="ready-now", assignee="alice")
        review_id = _park_in_review(conn, "review-now", "reviewer")
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn_factory(spawns), max_in_progress=1,
        )

    assert [task_id for task_id, *_ in res.spawned] == [review_id]


def test_ready_gets_full_budget_when_no_review_work(
    kanban_home, all_assignees_spawnable, monkeypatch,
):
    import hermes_cli.config as cfgmod
    monkeypatch.setattr(
        cfgmod, "load_config",
        lambda *a, **k: {"kanban": {"review_dispatch": True}},
    )

    spawns: list = []
    with kbc.connect() as conn:
        for title in ("ready-1", "ready-2", "ready-3"):
            kb.create_task(conn, title=title, assignee="alice")
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn_factory(spawns), max_in_progress=2,
        )

    # No review work → ready lane keeps the full budget.
    assert len(res.spawned) == 2


def test_nonspawnable_review_does_not_tax_ready_budget(
    kanban_home, monkeypatch,
):
    """Review tasks parked for humans (no real profile) release the slot."""
    import hermes_cli.config as cfgmod
    import hermes_cli.profiles as profmod

    monkeypatch.setattr(
        cfgmod, "load_config",
        lambda *a, **k: {"kanban": {"review_dispatch": True}},
    )
    # Only 'alice' is a real profile; the review assignee is a human lane.
    monkeypatch.setattr(
        profmod, "profile_exists", lambda name: name == "alice"
    )

    spawns: list = []
    with kbc.connect() as conn:
        for title in ("ready-1", "ready-2"):
            kb.create_task(conn, title=title, assignee="alice")
        _park_in_review(conn, "human-review", "some-human")
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn_factory(spawns), max_in_progress=2,
        )

    # Human-lane review is not spawnable; READY gets both slots.
    assert len(res.spawned) == 2


def test_review_budget_still_bounded_by_shared_cap(
    kanban_home, all_assignees_spawnable, monkeypatch,
):
    """The review opportunity grants no extra slots beyond the shared cap."""
    import hermes_cli.config as cfgmod
    monkeypatch.setattr(
        cfgmod, "load_config",
        lambda *a, **k: {"kanban": {"review_dispatch": True}},
    )

    spawns: list = []
    with kbc.connect() as conn:
        kb.create_task(conn, title="ready-1", assignee="alice")
        for i in range(3):
            _park_in_review(conn, f"review-{i}", "reviewer")
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn_factory(spawns), max_in_progress=2,
        )

    # Budget 2 total across both lanes.
    assert len(res.spawned) == 2


def _unverifiable_review(conn, title, *, priority, validation_required=False):
    parent = kb.create_task(
        conn, title=f"{title} parent", assignee="implementer",
        lifecycle_contract={
            "kind": "code", "review_mode": "separate_card", "reviewer": "reviewer",
            "validation_required": validation_required,
        },
    )
    _set_task_status(conn, parent, "done")
    review = kb.create_task(
        conn, title=title, assignee="reviewer", parents=[parent], priority=priority,
        lifecycle_contract={"kind": "review", "candidate_task_id": parent},
    )
    _set_task_status(conn, review, "review")
    return review


@pytest.mark.parametrize("dry_run", [False, True])
@pytest.mark.parametrize("cap", [1, 2, 3, None])
@pytest.mark.parametrize("ready_count,valid_count", [(4, 3), (1, 3), (4, 0)])
def test_review_service_order_uses_only_successful_attempts(
    kanban_home, all_assignees_spawnable, monkeypatch,
    dry_run, cap, ready_count, valid_count,
):
    """Invalid reviews cost no slots, and no review is retried in the final pass."""
    monkeypatch.setattr(kbd, "review_dispatch_enabled", lambda: True)
    attempted = []
    dispatch = kbd._dispatch_lane_task

    def track(conn, row, *args, **kwargs):
        attempted.append(row["id"])
        return dispatch(conn, row, *args, **kwargs)

    monkeypatch.setattr(kbd, "_dispatch_lane_task", track)
    spawns = []
    with kbc.connect() as conn:
        ready = [
            kb.create_task(conn, title=f"ready-{i}", assignee="alice", priority=10-i)
            for i in range(ready_count)
        ]
        invalid = [
            _unverifiable_review(conn, f"invalid-{i}", priority=100-i) for i in range(2)
        ]
        valid = [_park_in_review(conn, f"valid-{i}", "reviewer") for i in range(valid_count)]
        for i, tid in enumerate(valid):
            conn.execute("UPDATE tasks SET priority = ? WHERE id = ?", (10-i, tid))
        before_runs = conn.execute("SELECT COUNT(*) FROM task_runs").fetchone()[0]
        result = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn_factory(spawns), max_in_progress=cap, dry_run=dry_run,
        )
        expected = ready + valid if cap is None else (valid[:1] + ready + valid[1:])[:cap]
        assert [tid for tid, *_ in result.spawned] == expected
        assert spawns == ([] if dry_run else expected)
        assert len(attempted) == len(set(attempted))
        assert [item["task_id"] for item in result.handoff_refused] == invalid
        assert all(item["kind"] == "handoff_unverifiable" for item in result.handoff_refused)
        for tid in invalid:
            task = kb.get_task(conn, tid)
            assert task.status == "review" and task.current_run_id is None
            events = [e for e in kb.list_events(conn, tid) if e.kind == "handoff_unverifiable"]
            assert len(events) == (0 if dry_run else 1)
        if dry_run:
            assert conn.execute("SELECT COUNT(*) FROM task_runs").fetchone()[0] == before_runs


@pytest.mark.parametrize("dry_run", [False, True])
@pytest.mark.parametrize("cap", [1, 2, 3])
def test_review_service_respects_cross_board_capacity(
    kanban_home, all_assignees_spawnable, monkeypatch, dry_run, cap,
):
    monkeypatch.setattr(kbd, "review_dispatch_enabled", lambda: True)
    kb.create_board("second")
    with kbc.connect(board="second") as other:
        busy = kb.create_task(other, title="other-board worker", assignee="alice")
        assert kb.claim_task(other, busy) is not None
    spawns = []
    with kbc.connect() as conn:
        review = _park_in_review(conn, "review", "reviewer")
        ready = kb.create_task(conn, title="ready", assignee="alice")
        result = kbd.dispatch_once(
            conn, board="default", spawn_fn=_fake_spawn_factory(spawns),
            max_in_progress=cap, max_spawn=3, dry_run=dry_run,
        )
        assert [tid for tid, *_ in result.spawned] == [review, ready][:cap-1]
        assert kbd.count_running_tasks(conn) == (0 if dry_run else cap-1)


@pytest.mark.parametrize("dry_run", [False, True])
def test_disabled_review_dispatch_preserves_ready_default_assignment(
    kanban_home, all_assignees_spawnable, monkeypatch, dry_run,
):
    monkeypatch.setattr(kbd, "review_dispatch_enabled", lambda: False)
    with kbc.connect() as conn:
        review = _unverifiable_review(conn, "disabled review", priority=100)
        ready = kb.create_task(conn, title="unassigned")
        result = kbd.dispatch_once(
            conn, spawn_fn=lambda *a, **kw: None, max_in_progress=1,
            default_assignee="alice", dry_run=dry_run,
        )
        assert [tid for tid, *_ in result.spawned] == [ready]
        assert result.auto_assigned_default == [ready]
        assert result.handoff_refused == []
        assert kb.get_task(conn, review).status == "review"
        assert kb.get_task(conn, ready).assignee == (None if dry_run else "alice")


@pytest.mark.parametrize("failure", ["workspace", "preparation", "claim", "snapshot", "stop"])
def test_failed_initial_review_releases_capacity_without_retry(
    kanban_home, all_assignees_spawnable, monkeypatch, failure,
):
    from hermes_cli import kanban_db_workspace as kbw
    from hermes_cli.kanban_runtime import prospective_identity

    monkeypatch.setattr(kbd, "review_dispatch_enabled", lambda: True)
    attempts, grants, cancelled = [], [], []
    stopped = False
    with kbc.connect() as conn:
        review = _park_in_review(conn, "review", "reviewer")
        ready = kb.create_task(conn, title="ready", assignee="alice")
        resolve = kbw.resolve_workspace

        def workspace(task, **kwargs):
            if task.id == review and failure == "workspace":
                attempts.append(task.id)
                raise RuntimeError("fixture workspace unavailable")
            return resolve(task, **kwargs)

        def prepare(task, workspace, **kwargs):
            nonlocal stopped
            attempts.append(task.id)
            if task.id == review:
                if failure == "preparation":
                    raise RuntimeError("fixture preparation refused")
                if failure == "snapshot":
                    conn.execute("UPDATE tasks SET version = version + 1 WHERE id = ?", (review,))
                if failure == "stop":
                    stopped = True
            identity = prospective_identity()
            return kbd.WorkerLaunch(
                pid=identity.pid, runtime_identity=identity.as_dict(), preparation_id="fixture",
                grant=lambda *args: grants.append(task.id),
                cancel=lambda: cancelled.append(task.id),
            )

        if failure == "claim":
            monkeypatch.setattr(kb, "claim_review_task", lambda *a, **kw: None)
        monkeypatch.setattr(kbw, "resolve_workspace", workspace)
        monkeypatch.setattr(kbd, "_default_spawn", prepare)
        result = kbd.dispatch_once(
            conn, max_in_progress=1, failure_limit=3, should_stop=lambda: stopped,
        )
        expected = [] if failure == "stop" else [ready]
        assert [tid for tid, *_ in result.spawned] == expected
        assert grants == expected
        assert attempts.count(review) == 1
        assert result.handoff_refused == []
        assert review not in grants
        if failure in {"claim", "snapshot", "stop"}:
            assert cancelled == [review]
            assert kb.get_task(conn, review).current_run_id is None


@pytest.mark.parametrize("cap_kind", ["host", "board", "profile"])
def test_concurrent_review_claim_refreshes_caps_before_another_attempt(
    kanban_home, all_assignees_spawnable, monkeypatch, cap_kind,
):
    from hermes_cli.kanban_runtime import prospective_identity

    monkeypatch.setattr(kbd, "review_dispatch_enabled", lambda: True)
    prepared, granted, cancelled = [], [], []
    with kbc.connect() as conn:
        review = _park_in_review(conn, "claimed elsewhere during preparation", "reviewer")
        conn.execute("UPDATE tasks SET priority = 100 WHERE id = ?", (review,))
        second_review = _park_in_review(conn, "another review", "reviewer")
        same_profile = kb.create_task(conn, title="same profile ready", assignee="reviewer", priority=10)
        ready = kb.create_task(conn, title="other profile ready", assignee="alice")

        def prepare(task, workspace, **kwargs):
            prepared.append(task.id)
            if task.id == review:
                with kbc.connect() as other:
                    assert kb.claim_review_task(other, review, claimer="other:reviewer") is not None
            identity = prospective_identity()
            return kbd.WorkerLaunch(
                pid=identity.pid, runtime_identity=identity.as_dict(), preparation_id="fixture",
                grant=lambda *args: granted.append(task.id),
                cancel=lambda: cancelled.append(task.id),
            )

        monkeypatch.setattr(kbd, "_default_spawn", prepare)
        caps = {
            "host": {"max_in_progress": 1},
            "board": {"max_spawn": 1},
            "profile": {"max_in_progress": 3, "max_in_progress_per_profile": 1},
        }[cap_kind]
        result = kbd.dispatch_once(conn, **caps)
        expected = [ready] if cap_kind == "profile" else []
        assert [tid for tid, *_ in result.spawned] == expected
        assert granted == expected
        assert prepared == [review] + expected
        assert cancelled == [review]
        assert result.handoff_refused == []
        assert kbd.count_running_tasks(conn) == 1 + len(expected)
        if cap_kind == "profile":
            assert {tid for tid, *_ in result.skipped_per_profile_capped} == {
                second_review, same_profile,
            }


@pytest.mark.parametrize("change", ["moved", "unavailable", "claimed_elsewhere"])
def test_handoff_change_during_preparation_cancels_without_grant(
    kanban_home, all_assignees_spawnable, monkeypatch, tmp_path, change,
):
    from hermes_cli.kanban_runtime import prospective_identity
    from tests.hermes_cli.test_kanban_handoff import _repo, _lane, _commit

    monkeypatch.setattr(kbd, "review_dispatch_enabled", lambda: True)
    repo, base, branch = _repo(tmp_path)
    grants, cancelled = [], []
    changed_head = None
    with kbc.connect() as conn:
        parent, review = _lane(conn, repo, branch)
        implementation = kb.claim_task(conn, parent)
        assert implementation is not None
        approved_head = _commit(repo)
        assert kb.complete_task(
            conn, parent, expected_run_id=implementation.current_run_id,
            summary="implemented src/changed.py",
            metadata={"base_sha": base, "head_sha": approved_head},
        )
        # Preparation is a local stand-in; the guard still resolves the real
        # parent's Git checkout, independent of the review worker's workspace.
        conn.execute(
            "UPDATE tasks SET workspace_kind = 'scratch', workspace_path = NULL, "
            "branch_name = NULL WHERE id = ?", (review,),
        )
        ready = kb.create_task(conn, title="ready", assignee="alice")

        def prepare(task, workspace, **kwargs):
            nonlocal changed_head
            if task.id == review:
                if change == "claimed_elsewhere":
                    with kbc.connect() as other:
                        assert kb.claim_review_task(other, review, claimer="other:reviewer") is not None
                if change == "unavailable":
                    repo.rename(tmp_path / "temporarily-unavailable")
                else:
                    changed_head = _commit(repo, "src/moved.py")
            identity = prospective_identity()
            return kbd.WorkerLaunch(
                pid=identity.pid, runtime_identity=identity.as_dict(), preparation_id="fixture",
                grant=lambda *args: grants.append(task.id),
                cancel=lambda: cancelled.append(task.id),
            )

        monkeypatch.setattr(kbd, "_default_spawn", prepare)
        result = kbd.dispatch_once(conn, max_in_progress=1, board="default")
        expected = [] if change == "claimed_elsewhere" else [ready]
        assert [tid for tid, *_ in result.spawned] == expected
        assert grants == expected
        assert cancelled == [review]
        if change == "claimed_elsewhere":
            assert result.handoff_refused == [], "a competing claim is not our handoff refusal"
            assert kb.get_task(conn, review).claim_lock == "other:reviewer"
        else:
            assert kb.get_task(conn, review).current_run_id is None
            assert conn.execute(
                "SELECT COUNT(*) FROM task_runs WHERE task_id = ?", (review,),
            ).fetchone()[0] == 0
            assert len(result.handoff_refused) == 1
            refusal = result.handoff_refused[0]
            assert refusal["task_id"] == review and refusal["parent_id"] == parent
            assert refusal["expected_head_sha"] == approved_head
            assert refusal["kind"] == (
                "handoff_unverifiable" if change == "unavailable" else "handoff_head_moved"
            )
            if change == "moved":
                assert refusal["actual_head_sha"] == changed_head
            assert "--quarantine-review" in refusal["command"]
            assert "--board default" in refusal["command"]


def test_ready_validation_handoff_refusal_never_offers_review_quarantine(
    kanban_home, all_assignees_spawnable, monkeypatch,
):
    monkeypatch.setattr(kbd, "review_dispatch_enabled", lambda: True)
    with kbc.connect() as conn:
        review = _unverifiable_review(
            conn, "reviewed parent without handoff", priority=10, validation_required=True,
        )
        candidate = kb.parent_ids(conn, review)[0]
        _set_task_status(conn, review, "done")
        validation = kb.create_task(
            conn, title="validation", assignee="tester", parents=[review],
            lifecycle_contract={"kind": "validation", "candidate_task_id": candidate},
        )
        _set_task_status(conn, validation, "ready")
        row = next(row for row in kbd._lane_rows(conn, "ready") if row["id"] == validation)
        result = kbd.DispatchResult()
        assert not kbd._dispatch_lane_task(
            conn, row, "tester", result, lane="ready", dry_run=True,
            ttl_seconds=None, board=None, failure_limit=2, spawn_fn=None,
            per_profile_cap=None, per_profile_running={},
        )
        refusal = result.handoff_refused[0]
        assert refusal["task_id"] == validation and refusal["parent_id"] == review
        assert refusal["command"] == ""
        assert review in refusal["recovery"]
        assert "does not authorize a new candidate or release validation" in refusal["recovery"]
        assert "handoff_unverifiable=1" in kbd.describe_suppression([None, result])


def test_review_service_cannot_expand_the_original_tick_budget(
    kanban_home, all_assignees_spawnable, monkeypatch,
):
    monkeypatch.setattr(kbd, "review_dispatch_enabled", lambda: True)
    kb.create_board("second")
    with kbc.connect(board="second") as other:
        busy = kb.create_task(other, title="finishes during review launch", assignee="alice")
        claimed = kb.claim_task(other, busy)
        assert claimed is not None
    with kbc.connect() as conn:
        review = _park_in_review(conn, "review", "reviewer")
        ready = [
            kb.create_task(conn, title=f"ready-{i}", assignee="alice", priority=10-i)
            for i in range(3)
        ]

        def spawn(task, workspace, **kwargs):
            if task.id == review:
                with kbc.connect(board="second") as other:
                    assert kb.complete_task(
                        other, busy, expected_run_id=claimed.current_run_id, summary="finished",
                    )
            return None

        result = kbd.dispatch_once(conn, spawn_fn=spawn, max_in_progress=3, board="default")
        assert [tid for tid, *_ in result.spawned] == [review, ready[0]]
        assert kbd.count_running_tasks_other_boards("default") == 0
        assert kbd.count_running_tasks(conn) == 2

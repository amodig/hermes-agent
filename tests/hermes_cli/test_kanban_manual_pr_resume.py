"""Existing-PR continuation: the explicit, audited replacement for the old
timestamp-based ``promoted_manual`` PR exception (#39).

A comment recording a GitHub PR keeps ``active_pr`` holding the card until the
operator runs the CAS ``continue_existing_pr`` transition. Only that transition
acknowledges the work, and only while the effective goal is unchanged and no
newer run has completed.
"""

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as dispatch

PR1 = "https://github.com/example/repo/pull/1"
PR2 = "https://github.com/example/repo/pull/2"


@pytest.fixture
def clock(monkeypatch):
    """Freeze wall clock (``dispatch.time`` is the ``time`` module itself, so
    comment/run timestamps are frozen with it) and expose an advance hook."""

    def _set(value: int) -> None:
        monkeypatch.setattr(dispatch.time, "time", lambda: value)

    _set(2000000000)
    return _set


def _continue(conn, tid, reason="Explicitly authorized: continue the existing PR"):
    return kb.update_task(
        conn,
        tid,
        expected_version=kb.get_task(conn, tid).version,
        reason=reason,
        transition="continue_existing_pr",
    )


def _events(conn, tid, kind):
    return [e for e in kb.list_events(conn, tid) if e.kind == kind]


def test_pr_comment_holds_until_the_explicit_transition(clock, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    kb.init_db()
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="Finish published PR gates", assignee="cto", triage=True)
        kb.add_comment(conn, tid, author="worker", body=f"Published {PR1}. Continue, do not recreate.")
        assert dispatch.check_respawn_guard(conn, tid) == "active_pr"

        # The reported native sequence: triage requeue, then a goal revision.
        assert kb.update_task(
            conn, tid, expected_version=kb.get_task(conn, tid).version,
            reason="promote triage", transition="triage_to_ready",
        )
        assert kb.get_task(conn, tid).status == "ready"
        assert dispatch.check_respawn_guard(conn, tid) == "active_pr"
        assert kb.update_task(
            conn, tid, expected_version=kb.get_task(conn, tid).version,
            reason="Polish the PR with review loops", body="Polish the PR with review loops.",
        )
        assert dispatch.check_respawn_guard(conn, tid) == "active_pr"
        assert kb._pr_continuation(conn, tid) is None

        # Same second: only the explicit transition acknowledges the work.
        assert _continue(conn, tid)
        assert dispatch.check_respawn_guard(conn, tid) is None
        continuation = kb._pr_continuation(conn, tid)
        assert continuation is not None
        assert continuation["pr_urls"] == [PR1]
        event = _events(conn, tid, "pr_continuation_authorized")[-1]
        assert event.payload["pr_urls"] == [PR1]
        assert event.payload["goal_revision_id"] == kb.get_task(conn, tid).goal_revision_id
        assert event.payload["through_comment_id"] >= 1
        assert event.payload["after_run_id"] == 0

        # A later second behaves identically.
        clock(2000000005)
        assert dispatch.check_respawn_guard(conn, tid) is None


def test_repeat_of_authorized_url_is_allowed_but_a_new_url_holds(clock, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    kb.init_db()
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="PR work", assignee="cto")
        kb.add_comment(conn, tid, author="worker", body=PR1)
        assert _continue(conn, tid)
        assert dispatch.check_respawn_guard(conn, tid) is None

        kb.add_comment(conn, tid, author="reviewer", body=f"Still following up on {PR1}")
        assert dispatch.check_respawn_guard(conn, tid) is None

        # A brand-new URL guards even in the same second as the acknowledgment.
        kb.add_comment(conn, tid, author="worker", body=f"Also opened {PR2}")
        assert dispatch.check_respawn_guard(conn, tid) == "active_pr"
        clock(2000000001)
        assert dispatch.check_respawn_guard(conn, tid) == "active_pr"


def test_later_goal_revision_and_completed_run_expire_the_authorization(clock, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    kb.init_db()
    with kbc.connect() as conn:
        revised = kb.create_task(conn, title="revised goal", assignee="cto")
        kb.add_comment(conn, revised, author="worker", body=PR1)
        assert _continue(conn, revised)
        assert dispatch.check_respawn_guard(conn, revised) is None
        assert kb.update_task(
            conn, revised, expected_version=kb.get_task(conn, revised).version,
            reason="re-scope", title="revised goal v2",
        )
        assert kb._pr_continuation(conn, revised) is None
        assert dispatch.check_respawn_guard(conn, revised) == "active_pr"

        completed = kb.create_task(conn, title="completed", assignee="cto")
        kb.add_comment(conn, completed, author="worker", body=PR1)
        assert _continue(conn, completed)
        claimed = kb.claim_task(conn, completed)
        assert claimed is not None
        assert kb.complete_task(conn, completed, summary="published")
        assert kb._pr_continuation(conn, completed) is None
        # The completed run is a recent success, which takes priority.
        assert dispatch.check_respawn_guard(conn, completed) == "recent_success"


def test_failed_run_keeps_the_authorization_for_the_same_work(clock, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    kb.init_db()
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="retryable", assignee="cto")
        kb.add_comment(conn, tid, author="worker", body=PR1)
        assert _continue(conn, tid)
        assert kb.claim_task(conn, tid) is not None
        assert kb.block_task(conn, tid, reason="worker bailed after a failed run")
        assert kb._pr_continuation(conn, tid) is not None
        assert kb.unblock_task(conn, tid)
        assert dispatch.check_respawn_guard(conn, tid) is None


def test_manual_promotion_and_ordinary_comments_do_not_authorize(clock, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    kb.init_db()
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="operator resume", assignee="cto")
        kb.add_comment(conn, tid, author="worker", body=PR1)
        assert kb.block_task(conn, tid, reason="Explicit operator resume")
        clock(2000000001)
        assert kb.promote_task(conn, tid, actor="operator", reason="Continue existing PR") == (True, None)
        assert dispatch.check_respawn_guard(conn, tid) == "active_pr"
        assert kb._pr_continuation(conn, tid) is None


def test_auth_and_cooldown_take_priority_over_pr_authorization(clock, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    kb.init_db()
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="auth wall", assignee="cto")
        kb.add_comment(conn, tid, author="worker", body=PR1)
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET last_failure_error = ? WHERE id = ?", ("401 Unauthorized", tid))
        assert dispatch.check_respawn_guard(conn, tid) == "blocker_auth"
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET last_failure_error = NULL WHERE id = ?", (tid,))

        with kb.write_txn(conn):
            conn.execute(
                "INSERT INTO task_runs (task_id, profile, status, outcome, started_at, ended_at) "
                "VALUES (?, 'cto', 'rate_limited', 'rate_limited', ?, ?)",
                (tid, 2000000000, 2000000000),
            )
        assert dispatch.check_respawn_guard(conn, tid) == "rate_limit_cooldown"
        # Authorization never bypasses an active cooldown.
        assert _continue(conn, tid)
        assert dispatch.check_respawn_guard(conn, tid) == "rate_limit_cooldown"

        # Cooldown elapsed: the quota wall no longer traps the card, but the PR
        # fence still applies until the recorded PR is acknowledged.
        clock(2000000000 + dispatch.DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS + 1)
        assert dispatch.check_respawn_guard(conn, tid) is None


def test_rejected_updates_leave_no_authorization_or_partial_goal_edit(clock, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    kb.init_db()
    with kbc.connect() as conn:
        # No recorded PR URL at all.
        bare = kb.create_task(conn, title="nothing recorded", assignee="cto")
        before = kb.get_task(conn, bare).version
        with pytest.raises(ValueError, match="requires an existing GitHub PR URL"):
            _continue(conn, bare, reason="authorize")
        assert kb.get_task(conn, bare).version == before
        assert _events(conn, bare, "pr_continuation_authorized") == []
        assert kb._pr_continuation(conn, bare) is None

        # Forbidden status: a blocked card keeps its own recovery path.
        blocked = kb.create_task(conn, title="blocked card", assignee="cto")
        kb.add_comment(conn, blocked, author="worker", body=PR1)
        assert kb.block_task(conn, blocked, reason="hold")
        before = kb.get_task(conn, blocked).version
        with pytest.raises(ValueError, match="requires an unclaimed task"):
            _continue(conn, blocked)
        assert kb.get_task(conn, blocked).version == before
        assert kb._pr_continuation(conn, blocked) is None

        # A stale expected_version rolls back the whole transaction.
        stale = kb.create_task(conn, title="stale cas", assignee="cto")
        kb.add_comment(conn, stale, author="worker", body=PR1)
        with pytest.raises(kb.TaskUpdateConflict):
            kb.update_task(
                conn, stale, expected_version=99, reason="authorize",
                transition="continue_existing_pr", title="never applied",
            )
        assert kb.get_task(conn, stale).title == "stale cas"
        assert _events(conn, stale, "pr_continuation_authorized") == []
        assert dispatch.check_respawn_guard(conn, stale) == "active_pr"

        # A claimed card refuses the update entirely.
        claimed_task = kb.create_task(conn, title="claimed", assignee="cto")
        assert kb.claim_task(conn, claimed_task) is not None
        kb.add_comment(conn, claimed_task, author="worker", body=PR1)
        with pytest.raises(kb.TaskUpdateConflict):
            _continue(conn, claimed_task)
        assert _events(conn, claimed_task, "pr_continuation_authorized") == []


def test_repeated_ticks_record_one_guard_event_but_report_every_tick(clock, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    # Assume the assignee maps to a real profile so the lane reaches the guard
    # rather than stopping at the non-spawnable-assignee check.
    monkeypatch.setattr(dispatch, "_profile_exists_fn", lambda: None)
    kb.init_db()

    def _spawn(*_args, **_kwargs):  # pragma: no cover - never reached while held
        raise AssertionError("guarded task must not spawn")

    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="held", assignee="cto")
        kb.add_comment(conn, tid, author="worker", body=PR1)

        guarded = 0
        for _ in range(100):
            res = dispatch.dispatch_once(conn, spawn_fn=_spawn, reconcile_orphans=False)
            assert res.spawned == []
            if dict(res.respawn_guarded).get(tid) == "active_pr":
                guarded += 1
        assert guarded == 100
        assert len(_events(conn, tid, "respawn_guarded")) == 1

        # Intervening activity opens a new guard episode; a repeat of the same
        # reason in between does not.
        assert kb.update_task(
            conn, tid, expected_version=kb.get_task(conn, tid).version,
            reason="intervening revision", body="intervening body",
        )
        dispatch.dispatch_once(conn, spawn_fn=_spawn, reconcile_orphans=False)
        assert len(_events(conn, tid, "respawn_guarded")) == 2
        dispatch.dispatch_once(conn, spawn_fn=_spawn, reconcile_orphans=False)
        assert len(_events(conn, tid, "respawn_guarded")) == 2

        # Dry runs report the hold but never write.
        before = len(_events(conn, tid, "respawn_guarded"))
        for _ in range(5):
            res = dispatch.dispatch_once(conn, dry_run=True, reconcile_orphans=False)
            assert dict(res.respawn_guarded).get(tid) == "active_pr"
        assert len(_events(conn, tid, "respawn_guarded")) == before


def test_recovery_hint_is_a_runnable_cli_command(clock, tmp_path, monkeypatch):
    """The live hold's recovery text must be copy-paste runnable.

    Regression: ``--board`` is a ``kanban``-level flag, so a hint that appends
    it after the task id produces an argparse error instead of a recovery.
    """
    import argparse
    import shlex

    from hermes_cli.kanban_parser import build_parser

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    kb.init_db()
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="hint", assignee="cto")
        kb.add_comment(conn, tid, author="worker", body=PR1)
        guard = kb.get_dispatch_guard(conn, tid, board="default")
        assert guard is not None and guard["reason"] == "active_pr"
        command = guard["command"]
        version = kb.get_task(conn, tid).version
        assert guard["recovery"] and "hermes kanban" not in guard["recovery"]

    # The copyable field is the command alone - never prose with a command
    # buried inside it, which is what the dashboard pastes to the clipboard.
    assert command.startswith("hermes kanban ")
    argv = shlex.split(command)
    assert argv[0] == "hermes" and argv[1] == "kanban"
    root = argparse.ArgumentParser()
    build_parser(root.add_subparsers(dest="cmd"))
    parsed = root.parse_args(argv[1:])
    assert parsed.board == "default"
    assert parsed.kanban_action == "update"
    assert parsed.task_id == tid
    assert parsed.expected_version == version
    assert parsed.transition == "continue_existing_pr"
    assert parsed.reason


def test_guard_window_lapses_but_the_grant_survives(clock, tmp_path, monkeypatch):
    """The documented boundary: the hold only inspects recent comments, while an
    authorization stays usable and still reaches the next run."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    kb.init_db()
    window = dispatch._RESPAWN_GUARD_PR_WINDOW
    with kbc.connect() as conn:
        authorized = kb.create_task(conn, title="authorized", assignee="cto")
        kb.add_comment(conn, authorized, author="worker", body=PR1)
        assert _continue(conn, authorized)
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE task_comments SET created_at = ? WHERE task_id = ?",
                (2000000000 - window - 60, authorized),
            )
        assert dispatch.check_respawn_guard(conn, authorized) is None
        assert kb._pr_continuation(conn, authorized) is not None
        claimed = kb.claim_task(conn, authorized)
        assert claimed is not None
        run_row = conn.execute(
            "SELECT metadata FROM task_runs WHERE id = ?", (claimed.current_run_id,),
        ).fetchone()
        assert PR1 in run_row["metadata"]

        # An unauthorized card is equally unheld once its only comment ages out,
        # and a fresh comment re-arms the hold.
        stale = kb.create_task(conn, title="stale", assignee="cto")
        kb.add_comment(conn, stale, author="worker", body=PR2)
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE task_comments SET created_at = ? WHERE task_id = ?",
                (2000000000 - window - 60, stale),
            )
        assert dispatch.check_respawn_guard(conn, stale) is None
        kb.add_comment(conn, stale, author="worker", body=f"still open: {PR2}")
        assert dispatch.check_respawn_guard(conn, stale) == "active_pr"


def test_review_claim_keeps_the_live_continuation(clock, tmp_path, monkeypatch):
    """A same-card reviewer still gets the published-PR context.

    The review lane is exempt from the guard fence, but losing the claim-bound
    authorization there would let the review phase act as if no PR existed.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    kb.init_db()
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="published PR review", assignee="cto")
        kb.add_comment(conn, tid, author="worker", body=f"Published {PR1}")
        assert _continue(conn, tid)
        implementation = kb.claim_task(conn, tid)
        assert implementation is not None
        assert kb.request_review(
            conn, tid, summary="published PR ready for review",
            expected_run_id=implementation.current_run_id,
        )
        reviewer = kb.claim_review_task(conn, tid)
        assert reviewer is not None
        run = kb.get_run(conn, reviewer.current_run_id)
        assert run.metadata["pr_continuation"]["pr_urls"] == [PR1]
        claimed_event = [
            e for e in kb.list_events(conn, tid) if e.kind == "claimed"
        ][-1]
        assert claimed_event.payload["pr_continuation_event_id"] == (
            kb._pr_continuation(conn, tid)["event_id"]
        )
        assert "Existing PR — continue, do not replace" in kb.build_worker_context(conn, tid)

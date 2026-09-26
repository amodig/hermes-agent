"""Manual promotion resumes existing PR work without disabling respawn guards."""

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as dispatch


def test_manual_promotion_resumes_only_older_pr_comments(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    monkeypatch.setattr(dispatch.time, "time", lambda: 2000000000)
    kb.init_db()
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="Finish published PR gates", assignee="cto")
        kb.add_comment(conn, tid, author="worker", body="https://github.com/example/repo/pull/1")
        assert dispatch.check_respawn_guard(conn, tid) == "active_pr"
        kb.block_task(conn, tid, reason="Explicit operator resume")
        monkeypatch.setattr(dispatch.time, "time", lambda: 2000000001)
        assert kb.promote_task(conn, tid, actor="operator", reason="Continue existing PR") == (True, None)
        assert dispatch.check_respawn_guard(conn, tid) is None
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET last_failure_error = ? WHERE id = ?", ("401 Unauthorized", tid))
        assert dispatch.check_respawn_guard(conn, tid) == "blocker_auth"
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET last_failure_error = NULL WHERE id = ?", (tid,))

        # A subsequent PR comment, including one in the same second, still guards.
        kb.add_comment(conn, tid, author="worker", body="https://github.com/example/repo/pull/2")
        assert dispatch.check_respawn_guard(conn, tid) == "active_pr"

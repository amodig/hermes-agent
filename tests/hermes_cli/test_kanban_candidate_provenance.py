"""A candidate's identity is reconciled against the launch receipt, not a cache file.

The sealed payload's ``generation.json`` is a cache the sweeper reclaims; the run's
``spawned`` event is written once and never rewritten, so it is the evidence a
reviewer needs when the author's own copy is malformed or absent. Retention keeps
that receipt for the same reason it keeps the run row.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli.kanban_db_receipts import candidate_identity

RECEIPT = {"code_sha": "a" * 40, "generation": "b" * 64}


@pytest.fixture
def conn(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    with kbc.connect() as c:
        yield c


def _spawned_run(conn, *, author=None, receipt=RECEIPT):
    tid = kb.create_task(conn, title="candidate", assignee="coder")
    assert kb.claim_task(conn, tid, claimer=kb._claimer_id()) is not None
    kbd._set_worker_pid(conn, tid, os.getpid())
    run_id = kb._current_run_id(conn, tid)
    if receipt is not None:
        kb._append_event(conn, tid, "spawned", {"runtime_identity": receipt}, run_id=run_id)
    if author is not None:
        conn.execute(
            "UPDATE task_runs SET metadata = ? WHERE id = ?",
            (json.dumps({"runtime_identity": author}), run_id),
        )
    conn.commit()
    return tid


def _verdict(conn, tid):
    return candidate_identity(conn, tid)["verdict"]


def test_candidate_identity_reconciles_against_the_launch_receipt(conn):
    # No author copy, and an agreeing one: the receipt alone answers.
    assert _verdict(conn, _spawned_run(conn)) == "verified"
    assert _verdict(conn, _spawned_run(conn, author=dict(RECEIPT))) == "verified"

    # The truncated digest the incident carried is malformed, and the receipt still
    # reports the authoritative 64-character value.
    verdict = candidate_identity(conn, _spawned_run(conn, author={**RECEIPT, "generation": "b" * 62}))
    assert verdict["verdict"] == "author_malformed"
    assert verdict["author_generation"] == "b" * 62
    assert verdict["receipt_generation"] == "b" * 64

    # Corrupt shapes this path exists to diagnose are not silently "verified".
    assert _verdict(conn, _spawned_run(conn, author="bogus")) == "author_malformed"
    assert _verdict(conn, _spawned_run(conn, author={**RECEIPT, "code_sha": ""})) == "author_malformed"
    # A partially conflicting copy is a conflict, not a pass.
    assert _verdict(conn, _spawned_run(conn, author={**RECEIPT, "code_sha": "d" * 40})) == "conflict"
    assert _verdict(conn, _spawned_run(conn, author={**RECEIPT, "generation": "c" * 64})) == "conflict"

    # A record that will not decode is present and untrustworthy, not absent.
    tid = _spawned_run(conn)
    conn.execute("UPDATE task_runs SET metadata = ? WHERE task_id = ?", ("not json", tid))
    conn.commit()
    assert _verdict(conn, tid) == "author_malformed"
    conn.execute("UPDATE task_runs SET metadata = ? WHERE task_id = ?", (json.dumps([1, 2]), tid))
    conn.commit()
    assert _verdict(conn, tid) == "author_malformed"

    # Runs without a receipt say so instead of guessing.
    assert _verdict(conn, _spawned_run(conn, receipt=None)) == "receipt_missing"


def test_event_retention_keeps_the_launch_receipt(conn):
    tid = _spawned_run(conn, author={**RECEIPT, "generation": "b" * 62})
    conn.execute("UPDATE tasks SET status = 'done' WHERE id = ?", (tid,))
    conn.execute("UPDATE task_events SET created_at = 0 WHERE task_id = ?", (tid,))
    conn.commit()

    assert kb.gc_events(conn, older_than_seconds=60) > 0  # ordinary events go
    kinds = {
        row[0] for row in conn.execute(
            "SELECT kind FROM task_events WHERE task_id = ?", (tid,),
        )
    }
    assert "spawned" in kinds, "the launch receipt must outlive event retention"
    assert "claimed" not in kinds
    # And it still reconciles after the sweep.
    assert _verdict(conn, tid) == "author_malformed"

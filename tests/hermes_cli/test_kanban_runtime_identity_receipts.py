"""The launch receipt owns ``runtime_identity``; a completion may only agree.

The dispatcher writes the identity when it spawns the worker and mirrors it in the
run's ``spawned`` event. Every worker and CLI completion closes the run through
``_end_run``, so that merge is where an author-supplied copy must be checked: a
malformed or differing value is a typed refusal naming the field, never something
that reads as a provenance or head verdict, and never a replacement for the receipt.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd

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


def _run_with_receipt(conn) -> tuple[str, int]:
    tid = kb.create_task(conn, title="receipt", assignee="coder")
    assert kb.claim_task(conn, tid, claimer=kb._claimer_id()) is not None
    kbd._set_worker_pid(conn, tid, os.getpid())
    run_id = kb._current_run_id(conn, tid)
    conn.execute(
        "UPDATE task_runs SET metadata = ? WHERE id = ?",
        (json.dumps({"runtime_identity": RECEIPT}), run_id),
    )
    conn.commit()
    return tid, run_id


def _stored_metadata(conn, run_id: int) -> dict:
    row = conn.execute("SELECT metadata FROM task_runs WHERE id = ?", (run_id,)).fetchone()
    return json.loads(row[0] or "{}")


def test_run_close_keeps_the_receipt_and_refuses_a_disagreeing_copy(conn):
    tid, run_id = _run_with_receipt(conn)

    # A truncated digest is refused before the run is closed or the row rewritten.
    with pytest.raises(kb.LifecycleEvidenceError, match="64-character digest"):
        kb._end_run(
            conn,
            tid,
            outcome="completed",
            metadata={"runtime_identity": {**RECEIPT, "generation": "b" * 62}},
        )
    assert _stored_metadata(conn, run_id)["runtime_identity"] == RECEIPT
    assert kb._current_run_id(conn, tid) == run_id

    # A value that is not an object is refused rather than replaced silently.
    with pytest.raises(kb.LifecycleEvidenceError, match="must be an object"):
        kb._end_run(conn, tid, outcome="completed", metadata={"runtime_identity": "bogus"})
    assert _stored_metadata(conn, run_id)["runtime_identity"] == RECEIPT

    # A different digest conflicts with the receipt instead of replacing it.
    with pytest.raises(kb.LifecycleEvidenceError, match="does not match the launch receipt"):
        kb._end_run(
            conn,
            tid,
            outcome="completed",
            metadata={"runtime_identity": {**RECEIPT, "generation": "c" * 64}},
        )
    assert _stored_metadata(conn, run_id)["runtime_identity"] == RECEIPT

    # An agreeing copy closes the run with the receipt intact.
    kb._end_run(
        conn, tid, outcome="completed", metadata={"runtime_identity": dict(RECEIPT)},
    )
    assert _stored_metadata(conn, run_id)["runtime_identity"] == RECEIPT
    assert kb._current_run_id(conn, tid) is None

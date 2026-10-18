"""A candidate's identity is reconciled against the launch receipt, not a cache file.

The sealed payload's ``generation.json`` is a cache the sweeper reclaims; the run's
``spawned`` event is written once and never rewritten, so it is the evidence a
reviewer needs when the author's own copy is malformed or absent.
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


def _spawned_run(conn, *, author: dict | None = None, with_receipt: bool = True) -> str:
    tid = kb.create_task(conn, title="candidate", assignee="coder")
    assert kb.claim_task(conn, tid, claimer=kb._claimer_id()) is not None
    kbd._set_worker_pid(conn, tid, os.getpid())
    run_id = kb._current_run_id(conn, tid)
    if with_receipt:
        kb._append_event(conn, tid, "spawned", {"runtime_identity": RECEIPT}, run_id=run_id)
    if author is not None:
        conn.execute(
            "UPDATE task_runs SET metadata = ? WHERE id = ?",
            (json.dumps({"runtime_identity": author}), run_id),
        )
    conn.commit()
    return tid


def test_candidate_identity_reconciles_against_the_launch_receipt(conn):
    # No author copy: the receipt alone answers.
    tid = _spawned_run(conn)
    verdict = candidate_identity(conn, tid)
    assert verdict["verdict"] == "verified"
    assert verdict["receipt_generation"] == "b" * 64

    # The truncated digest the incident carried is named as malformed, and the
    # receipt still reports the authoritative 64-character value.
    tid = _spawned_run(conn, author={**RECEIPT, "generation": "b" * 62})
    verdict = candidate_identity(conn, tid)
    assert verdict["verdict"] == "author_malformed"
    assert verdict["author_generation"] == "b" * 62
    assert verdict["receipt_generation"] == "b" * 64

    # A different 64-character digest is a conflict, not a silent pass.
    tid = _spawned_run(conn, author={**RECEIPT, "generation": "c" * 64})
    assert candidate_identity(conn, tid)["verdict"] == "conflict"

    # Agreeing copies verify, and a run with no receipt says so instead of guessing.
    tid = _spawned_run(conn, author=dict(RECEIPT))
    assert candidate_identity(conn, tid)["verdict"] == "verified"
    tid = _spawned_run(conn, with_receipt=False)
    assert candidate_identity(conn, tid)["verdict"] == "receipt_missing"

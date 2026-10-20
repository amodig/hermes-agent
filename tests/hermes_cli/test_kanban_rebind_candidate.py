"""A cleared candidate binding is recoverable, on proof, without redoing the work.

An out-of-phase claim used to clear ``candidate_run_id`` and leave nothing to review.
Recovery re-binds the existing run, and a claim that is then refused restores the
binding it displaced.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli.kanban_db_receipts import rebind_candidate, restored_candidate_run
from hermes_cli.kanban_lifecycle import LifecycleEvidenceError

RECEIPT = {"code_sha": "a" * 40, "generation": "b" * 64}
HEAD = "e" * 40


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


def _candidate(conn, head: str = HEAD):
    tid = kb.create_task(conn, title="candidate", assignee="coder")
    assert kb.claim_task(conn, tid, claimer=kb._claimer_id()) is not None
    # The typed contract is bound after the first claim, so the later out-of-phase
    # claim is the one that exercises the ready-claim clearing path.
    conn.execute(
        "UPDATE tasks SET lifecycle_contract = ? WHERE id = ?",
        (json.dumps({
            "kind": "code", "review_mode": "same_card", "reviewer": "reviewer",
            "validation_required": True,
        }), tid),
    )
    kbd._set_worker_pid(conn, tid, os.getpid())
    run_id = kb._current_run_id(conn, tid)
    kb._append_event(conn, tid, "spawned", {"runtime_identity": RECEIPT}, run_id=run_id)
    conn.execute(
        "UPDATE task_runs SET metadata = ? WHERE id = ?",
        (json.dumps({"head_sha": head, "lifecycle": {"candidate_run_id": run_id}}), run_id),
    )
    # A candidate is a *finished* implementation handoff, not an active attempt.
    conn.execute(
        "UPDATE task_runs SET status = 'review', outcome = 'review_requested' WHERE id = ?",
        (run_id,),
    )
    conn.execute("UPDATE tasks SET candidate_run_id = ? WHERE id = ?", (run_id, tid))
    conn.commit()
    return tid, run_id


def _binding(conn, tid):
    return conn.execute("SELECT candidate_run_id FROM tasks WHERE id = ?", (tid,)).fetchone()[0]


def test_rebind_restores_the_candidate_on_proof(conn):
    tid, run_id = _candidate(conn)
    conn.execute("UPDATE tasks SET candidate_run_id = NULL WHERE id = ?", (tid,))
    conn.commit()

    payload = rebind_candidate(conn, tid, run_id=run_id, head_sha=HEAD, reason="recovery")
    assert payload["candidate_run_id"] == run_id
    assert payload["receipt_generation"] == RECEIPT["generation"]
    assert _binding(conn, tid) == run_id
    kinds = [r[0] for r in conn.execute("SELECT kind FROM task_events WHERE task_id = ?", (tid,))]
    assert "candidate_rebound" in kinds

    # An existing binding is never overwritten unless the operator says so, and the
    # substitution is recorded.
    conn.execute("UPDATE tasks SET candidate_run_id = ? WHERE id = ?", (run_id + 1, tid))
    conn.commit()
    with pytest.raises(LifecycleEvidenceError, match="already bound"):
        rebind_candidate(conn, tid, run_id=run_id, head_sha=HEAD, reason="recovery")
    # A substitution must name the run it displaces, or the premise may be stale.
    with pytest.raises(LifecycleEvidenceError, match="not the run named for replacement"):
        rebind_candidate(
            conn, tid, run_id=run_id, head_sha=HEAD, reason="recovery",
            replace=True, replaced_run=run_id,
        )
    payload = rebind_candidate(
        conn, tid, run_id=run_id, head_sha=HEAD, reason="recovery",
        replace=True, replaced_run=run_id + 1,
    )
    assert payload["replaced_candidate_run_id"] == run_id + 1
    assert _binding(conn, tid) == run_id

    # An attempt that has not finished is not a candidate.
    conn.execute("UPDATE task_runs SET status = 'running' WHERE id = ?", (run_id,))
    conn.commit()
    with pytest.raises(LifecycleEvidenceError, match="finished implementation handoff"):
        rebind_candidate(conn, tid, run_id=run_id, head_sha=HEAD, reason="recovery")
    conn.execute(
        "UPDATE task_runs SET status = 'review', outcome = 'review_requested' WHERE id = ?",
        (run_id,),
    )
    conn.commit()

    # A head the run never recorded is refused: a rebind restores, never redefines.
    with pytest.raises(LifecycleEvidenceError, match="recorded head"):
        rebind_candidate(conn, tid, run_id=run_id, head_sha="f" * 40, reason="recovery")

    # So is a run with no launch receipt to attest it.
    conn.execute("DELETE FROM task_events WHERE task_id = ? AND kind = 'spawned'", (tid,))
    conn.commit()
    with pytest.raises(LifecycleEvidenceError, match="no launch receipt"):
        rebind_candidate(conn, tid, run_id=run_id, head_sha=HEAD, reason="recovery")


def test_a_refused_attempt_restores_the_binding_it_displaced(conn):
    tid, run_id = _candidate(conn)
    # An out-of-phase claim records what it displaces, then clears the binding.
    metadata = json.loads(
        conn.execute("SELECT metadata FROM task_runs WHERE id = ?", (run_id,)).fetchone()[0]
    )
    metadata["superseded_candidate"] = {"run_id": run_id}
    conn.execute(
        "UPDATE task_runs SET metadata = ? WHERE id = ?", (json.dumps(metadata), run_id),
    )
    conn.execute("UPDATE tasks SET candidate_run_id = NULL WHERE id = ?", (tid,))
    conn.commit()
    assert _binding(conn, tid) is None

    # The attempt is then refused, and the binding comes back.
    assert kb._end_run(conn, tid, outcome="blocked", summary="wrong phase") is not None
    assert _binding(conn, tid) == run_id
    kinds = [r[0] for r in conn.execute("SELECT kind FROM task_events WHERE task_id = ?", (tid,))]
    assert "candidate_binding_restored" in kinds


def test_request_review_keeps_a_restored_candidate(conn):
    """The handoff path must not synthesize a run over a restored candidate."""
    tid, run_id = _candidate(conn)

    assert restored_candidate_run(conn, tid, {"head_sha": HEAD}) == run_id
    # A different head cannot inherit the binding, and neither can an unfinished run.
    assert restored_candidate_run(conn, tid, {"head_sha": "f" * 40}) is None
    conn.execute("UPDATE task_runs SET status = 'running' WHERE id = ?", (run_id,))
    conn.commit()
    assert restored_candidate_run(conn, tid, {"head_sha": HEAD}) is None
    # A run with no launch receipt cannot stand as the candidate either.
    conn.execute(
        "UPDATE task_runs SET status = 'review', outcome = 'review_requested' WHERE id = ?",
        (run_id,),
    )
    conn.execute("DELETE FROM task_events WHERE task_id = ? AND kind = 'spawned'", (tid,))
    conn.commit()
    assert restored_candidate_run(conn, tid, {"head_sha": HEAD}) is None


def test_request_review_keeps_the_restored_candidate(conn, tmp_path):
    """The caller keeps the binding through the transition, not just the selector."""
    import subprocess

    # A hermetic worktree: the handoff snapshot must not depend on this checkout's state.
    ws = tmp_path / "ws"
    ws.mkdir()
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com",
    }
    def _git(*args):
        return subprocess.run(
            ["git", *args], cwd=ws, capture_output=True, text=True, check=True, env=env,
        ).stdout.strip()

    _git("init", "-q")
    (ws / "f.py").write_text("x = 1\n", encoding="utf-8")
    _git("add", ".")
    _git("commit", "-qm", "init")
    real_head = _git("rev-parse", "HEAD")
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")

    tid, run_id = _candidate(conn, head=real_head)
    ok, reason = kb.request_review(
        conn, tid, summary="repair resubmitted for review", reviewer="reviewer",
        with_reason=True, expected_run_id=run_id,
        metadata={
            "base_sha": real_head, "head_sha": real_head, "changed_files": ["plugins/x.py"],
            "branch_name": branch, "workspace_path": str(ws), "dirty_state": "clean",
        },
    )
    assert ok, f"request_review refused a restored candidate: {reason}"
    assert _binding(conn, tid) == run_id, "the restored candidate must survive the transition"
    assert conn.execute("SELECT status FROM tasks WHERE id = ?", (tid,)).fetchone()[0] == "review"

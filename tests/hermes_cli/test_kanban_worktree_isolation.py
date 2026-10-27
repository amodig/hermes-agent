"""Per-task worktree isolation for decompose siblings.

Decompose children used to inherit the root's literal ``workspace_path``,
so every sibling of a worktree-kind root pointed at the SAME checkout —
and ``_resolve_worktree_workspace``'s existing-checkout shortcut reused it
on whatever branch was there, letting sibling workers run concurrently in
one directory on one branch (cross-task provenance corruption, no lock).

Two-part fix under test:
- ``decompose_triage_task`` leaves worktree children's ``workspace_path``
  unset so each child materializes its own ``<repo>/.worktrees/<child-id>``.
- ``_resolve_worktree_workspace`` falls back to a fresh per-task worktree
  when the requested path is occupied by another task's branch (heals
  pre-existing rows that still carry a shared path).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_db_graph import decompose_triage_task
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_workspace as kbw


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(
        [
            "git", "-C", str(cwd),
            "-c", "user.name=Test User",
            "-c", "user.email=test@example.com",
            "-c", "commit.gpgsign=false",
            *args,
        ],
        check=True, capture_output=True, text=True,
    )


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(
        ["git", "init", "-b", "main", str(repo)],
        check=True, capture_output=True, text=True,
    )
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "init")
    return repo


def _add_worktree(repo: Path, target: Path, branch: str) -> Path:
    _git(repo, "worktree", "add", str(target), "-b", branch, "HEAD")
    return target


def test_decompose_worktree_children_get_own_workspace(kanban_home):
    with kbc.connect() as conn:
        root = kb.create_task(conn, title="build the feature", triage=True)
        conn.execute(
            "UPDATE tasks SET workspace_kind='worktree', "
            "workspace_path='/repo/.worktrees/root' WHERE id = ?",
            (root,),
        )
        conn.commit()

        child_ids = decompose_triage_task(
            conn,
            root,
            root_assignee="orchestrator",
            children=[
                {"title": "spec it", "assignee": "alice", "parents": []},
                {"title": "implement it", "assignee": "bob", "parents": [0]},
            ],
            author="decomposer",
        )
        assert child_ids is not None and len(child_ids) == 2

        for cid in child_ids:
            row = conn.execute(
                "SELECT workspace_kind, workspace_path FROM tasks WHERE id = ?",
                (cid,),
            ).fetchone()
            assert row["workspace_kind"] == "worktree"
            # Each child resolves its own <repo>/.worktrees/<child-id> at
            # dispatch; the root's literal path must never be shared.
            assert row["workspace_path"] is None




def test_resolve_worktree_falls_back_when_path_occupied(kanban_home, tmp_path):
    repo = _make_repo(tmp_path)
    occupied = _add_worktree(repo, repo / ".worktrees" / "sibling", "wt/sibling")

    with kbc.connect() as conn:
        tid = kb.create_task(
            conn,
            title="second sibling",
            workspace_kind="worktree",
            workspace_path=str(occupied),  # inherited shared/stale path
        )
        task = kb.get_task(conn, tid)

    workspace, branch = kbw._resolve_worktree_workspace(task)
    assert workspace == (repo / ".worktrees" / tid).resolve()
    assert branch == f"wt/{tid}"
    # The sibling's checkout is untouched, still on its own branch.
    assert (occupied / "README.md").exists()
    head = subprocess.run(
        ["git", "-C", str(occupied), "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert head == "wt/sibling"






# ---------------------------------------------------------------------------
# Typed role cards: their own worktree at the candidate commit
#
# A reviewer or tester that runs at the anchor repository's current HEAD reviews
# whatever landed there later, which is the wrong work. The candidate commit is
# derived from the card's own lifecycle projection and its candidate's handoff,
# and every refusal below leaves the existing checkout exactly as it was.
# ---------------------------------------------------------------------------


def _claim_args(task_id: str):
    import argparse

    return argparse.Namespace(task_id=task_id, ttl=None)


def _head(repo: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def _commit(repo: Path, name: str, text: str) -> str:
    (repo / name).write_text(text, encoding="utf-8")
    _git(repo, "add", name)
    _git(repo, "commit", "-m", f"add {name}")
    return _head(repo)


def _candidate_card(conn, repo: Path, *, base: str, head: str) -> str:
    """A finished implementation handoff: the run metadata the projection reads."""
    tid = kb.create_task(
        conn, title="implementation", assignee="coder", workspace_kind="worktree",
        workspace_path=str(repo),
        lifecycle_contract={
            "kind": "code", "review_mode": "separate_card", "reviewer": "reviewer",
            "validation_required": True,
        },
    )
    assert kb.claim_task(conn, tid, claimer=kb._claimer_id()) is not None
    run_id = kb._current_run_id(conn, tid)
    conn.execute(
        "UPDATE task_runs SET metadata = ?, status = 'review', outcome = 'review_requested' "
        "WHERE id = ?",
        (json.dumps({
            "base_sha": base, "head_sha": head,
            "lifecycle": {"head_sha": head, "base_sha": base, "candidate_run_id": run_id},
        }), run_id))
    conn.execute("UPDATE tasks SET candidate_run_id = ? WHERE id = ?", (run_id, tid))
    conn.commit()
    return tid


def _role_card(conn, kind: str, candidate: str, repo: Path, assignee: str) -> str:
    return kb.create_task(
        conn, title=f"{kind} the candidate", assignee=assignee, workspace_kind="worktree",
        workspace_path=str(repo),
        lifecycle_contract={"kind": kind, "candidate_task_id": candidate},
    )


def test_role_cards_get_their_own_worktree_at_the_candidate(kanban_home, tmp_path):
    repo = _make_repo(tmp_path)
    base = _head(repo)
    candidate = _commit(repo, "candidate.txt", "candidate\n")

    with kbc.connect() as conn:
        implementation = _candidate_card(conn, repo, base=base, head=candidate)
        review = _role_card(conn, "review", implementation, repo, "reviewer")
        validation = _role_card(conn, "validation", implementation, repo, "tester")
        review_task = kb.get_task(conn, review)
        validation_task = kb.get_task(conn, validation)
        review_head = kbw.candidate_head_for_task(conn, review_task)
        validation_head = kbw.candidate_head_for_task(conn, validation_task)

    assert review_head == candidate
    assert validation_head == candidate
    # The anchor repository is still where it was; nothing was moved to find this.
    assert _head(repo) == candidate

    reviewer_path, reviewer_branch = kbw._resolve_worktree_workspace(
        review_task, candidate_head=review_head)
    tester_path, tester_branch = kbw._resolve_worktree_workspace(
        validation_task, candidate_head=validation_head)
    assert reviewer_path == (repo / ".worktrees" / review).resolve()
    assert tester_path == (repo / ".worktrees" / validation).resolve()
    assert reviewer_path != tester_path
    assert reviewer_branch == "" and tester_branch == ""
    assert _head(reviewer_path) == candidate
    assert _head(tester_path) == candidate

    # The anchor repository moving on must not move a role worktree prepared for
    # this candidate: the commit id, not the branch tip, is the contract.
    later = _commit(repo, "later.txt", "later\n")
    assert later != candidate
    again, _ = kbw._resolve_worktree_workspace(review_task, candidate_head=review_head)
    assert again == reviewer_path
    assert _head(reviewer_path) == candidate


def test_role_worktree_refuses_a_dirty_checkout_and_preserves_it(kanban_home, tmp_path):
    repo = _make_repo(tmp_path)
    base = _head(repo)
    candidate = _commit(repo, "candidate.txt", "candidate\n")

    with kbc.connect() as conn:
        implementation = _candidate_card(conn, repo, base=base, head=candidate)
        review = _role_card(conn, "review", implementation, repo, "reviewer")
        task = kb.get_task(conn, review)
        head = kbw.candidate_head_for_task(conn, task)

    reviewer_path, _ = kbw._resolve_worktree_workspace(task, candidate_head=head)
    note = reviewer_path / "reviewer-notes.md"
    note.write_text("do not lose this\n", encoding="utf-8")
    later = _commit(repo, "later.txt", "later\n")

    with pytest.raises(ValueError, match="candidate_workspace_unavailable"):
        kbw._resolve_worktree_workspace(task, candidate_head=later)
    assert note.read_text(encoding="utf-8") == "do not lose this\n"
    assert _head(reviewer_path) == candidate


def test_role_worktree_refuses_a_siblings_checkout(kanban_home, tmp_path):
    repo = _make_repo(tmp_path)
    base = _head(repo)
    candidate = _commit(repo, "candidate.txt", "candidate\n")
    sibling = _add_worktree(repo, repo / ".worktrees" / "sibling", "wt/sibling")
    sibling_head = _head(sibling)

    with kbc.connect() as conn:
        implementation = _candidate_card(conn, repo, base=base, head=candidate)
        review = kb.create_task(
            conn, title="review", assignee="reviewer", workspace_kind="worktree",
            workspace_path=str(sibling),
            lifecycle_contract={"kind": "review", "candidate_task_id": implementation})
        task = kb.get_task(conn, review)
        head = kbw.candidate_head_for_task(conn, task)

    assert head == candidate
    with pytest.raises(ValueError, match="candidate_workspace_unavailable"):
        kbw._resolve_worktree_workspace(task, candidate_head=head)
    assert _head(sibling) == sibling_head
    assert subprocess.run(
        ["git", "-C", str(sibling), "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip() == "wt/sibling"


def test_role_cards_refuse_a_missing_or_conflicting_candidate(kanban_home, tmp_path):
    repo = _make_repo(tmp_path)
    base = _head(repo)
    candidate = _commit(repo, "candidate.txt", "candidate\n")

    with kbc.connect() as conn:
        # A candidate that never finished a handoff has no head to work from.
        unfinished = kb.create_task(
            conn, title="unfinished", assignee="coder",
            lifecycle_contract={
                "kind": "code", "review_mode": "separate_card", "reviewer": "reviewer",
                "validation_required": True,
            })
        review = _role_card(conn, "review", unfinished, repo, "reviewer")
        task = kb.get_task(conn, review)
        with pytest.raises(ValueError, match="candidate_workspace_unavailable"):
            kbw.candidate_head_for_task(conn, task)
        with pytest.raises(ValueError, match="candidate_workspace_unavailable"):
            kbw._resolve_worktree_workspace(task)

        # A handoff whose recorded head is not the projected head is not evidence.
        implementation = _candidate_card(conn, repo, base=base, head=candidate)
        run_id = conn.execute(
            "SELECT current_run_id FROM tasks WHERE id = ?", (implementation,)).fetchone()[0]
        conn.execute(
            "UPDATE task_runs SET metadata = ? WHERE id = ?",
            (json.dumps({
                "base_sha": base, "head_sha": candidate,
                "lifecycle": {"head_sha": "f" * 40, "base_sha": base},
            }), run_id))
        conn.commit()
        conflict = _role_card(conn, "validation", implementation, repo, "tester")
        conflicted = kb.get_task(conn, conflict)
        with pytest.raises(ValueError, match="candidate_workspace_unavailable"):
            kbw.candidate_head_for_task(conn, conflicted)

    assert not (repo / ".worktrees" / review).exists()


def test_role_directory_workspace_is_checked_in_place(kanban_home, tmp_path):
    repo = _make_repo(tmp_path)
    base = _head(repo)
    candidate = _commit(repo, "candidate.txt", "candidate\n")

    with kbc.connect() as conn:
        implementation = _candidate_card(conn, repo, base=base, head=candidate)
        review = kb.create_task(
            conn, title="review", assignee="reviewer", workspace_kind="dir",
            workspace_path=str(repo),
            lifecycle_contract={"kind": "review", "candidate_task_id": implementation})
        task = kb.get_task(conn, review)
        head = kbw.candidate_head_for_task(conn, task)

    # At the candidate and clean: usable, and never moved.
    assert kbw.resolve_workspace(task, candidate_head=head) == repo
    assert _head(repo) == candidate
    # Dirty: refused, with the file preserved.
    scratch = repo / "local.txt"
    scratch.write_text("keep me\n", encoding="utf-8")
    with pytest.raises(ValueError, match="candidate_workspace_unavailable"):
        kbw.resolve_workspace(task, candidate_head=head)
    assert scratch.read_text(encoding="utf-8") == "keep me\n"
    scratch.unlink()
    # Clean but at another commit: refused, never checked out for the worker.
    later = _commit(repo, "later.txt", "later\n")
    with pytest.raises(ValueError, match="candidate_workspace_unavailable"):
        kbw.resolve_workspace(task, candidate_head=head)
    assert _head(repo) == later


def test_role_worktree_resolves_from_its_own_persisted_path(kanban_home, tmp_path):
    """The dispatcher persists the canonical worktree, so a retry names that path."""
    repo = _make_repo(tmp_path)
    base = _head(repo)
    candidate = _commit(repo, "candidate.txt", "candidate\n")

    with kbc.connect() as conn:
        implementation = _candidate_card(conn, repo, base=base, head=candidate)
        review = _role_card(conn, "review", implementation, repo, "reviewer")
        task = kb.get_task(conn, review)
        head = kbw.candidate_head_for_task(conn, task)
        first, branch = kbw._resolve_worktree_workspace(task, candidate_head=head)
        assert branch == ""
        kbw.set_workspace_path(conn, review, str(first))
        persisted = kb.get_task(conn, review)
        again, branch = kbw._resolve_worktree_workspace(persisted, candidate_head=head)

    assert again == first
    assert branch == ""
    assert _head(first) == candidate


def test_role_worktree_refuses_to_detach_a_branch_holding_unique_work(kanban_home, tmp_path):
    repo = _make_repo(tmp_path)
    base = _head(repo)
    candidate = _commit(repo, "candidate.txt", "candidate\n")

    with kbc.connect() as conn:
        implementation = _candidate_card(conn, repo, base=base, head=candidate)
        review = _role_card(conn, "review", implementation, repo, "reviewer")
        task = kb.get_task(conn, review)
        head = kbw.candidate_head_for_task(conn, task)
        target = repo / ".worktrees" / review
        _git(repo, "worktree", "add", str(target), "-b", f"wt/{review}", base)
        (target / "unique.txt").write_text("only here\n", encoding="utf-8")
        _git(target, "add", "unique.txt")
        _git(target, "commit", "-m", "work the candidate does not contain")
        unique = _head(target)

        with pytest.raises(ValueError, match="candidate_workspace_unavailable"):
            kbw._resolve_worktree_workspace(task, candidate_head=head)

        # Nothing moved: the commit, the branch and the files are all still there.
        assert _head(target) == unique
        branch = subprocess.run(
            ["git", "-C", str(target), "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        assert branch == f"wt/{review}"
        assert (target / "unique.txt").read_text(encoding="utf-8") == "only here\n"

        # A branch the candidate already contains is safe to leave dangling.
        _git(repo, "worktree", "add", str(repo / ".worktrees" / f"{review}-contained"),
             "-b", "wt/contained", base)
        with kbc.connect() as conn:
            other_review = _role_card(conn, "review", implementation, repo, "reviewer")
            other = kb.get_task(conn, other_review)
            other_head = kbw.candidate_head_for_task(conn, other)
        contained = kbw._resolve_worktree_workspace(other, candidate_head=other_head)
    assert contained[1] == ""
    assert _head(contained[0]) == candidate


def test_claim_cli_leaves_a_role_card_unclaimed_when_its_candidate_is_missing(
    kanban_home, tmp_path, capsys,
):
    """The candidate is resolved before the claim, so a refusal strands no card."""
    from hermes_cli import kanban as kc

    repo = _make_repo(tmp_path)
    base = _head(repo)
    candidate = _commit(repo, "candidate.txt", "candidate\n")

    with kbc.connect() as conn:
        implementation = kb.create_task(
            conn, title="implementation", assignee="coder", workspace_kind="worktree",
            workspace_path=str(repo),
            lifecycle_contract={
                "kind": "code", "review_mode": "separate_card", "reviewer": "reviewer",
                "validation_required": True,
            })
        review = _role_card(conn, "review", implementation, repo, "reviewer")
        kb.link_tasks(conn, implementation, review, requirement="phase_finished")
        run = kb.claim_task(conn, implementation, claimer=kb._claimer_id())
        assert run is not None
        assert kb.complete_task(
            conn, implementation, expected_run_id=run.current_run_id,
            summary="implementation handoff",
            metadata={"base_sha": base, "head_sha": candidate, "changed_files": []})
        kb.recompute_ready(conn)
        assert kb.get_task(conn, review).status == "review"
        # The candidate's own handoff now records a base that is not a commit, so the
        # board cannot name one immutable commit to work from.
        conn.execute(
            "UPDATE task_runs SET metadata = ? WHERE task_id = ?",
            (json.dumps({
                "base_sha": "not-a-commit", "head_sha": candidate,
                "lifecycle": {"head_sha": candidate, "base_sha": "not-a-commit"},
            }), implementation))
        conn.commit()

    assert kc._cmd_claim(_claim_args(review)) != 0
    captured = capsys.readouterr()
    assert "candidate_workspace_unavailable" in captured.out + captured.err
    with kbc.connect() as conn:
        after = kb.get_task(conn, review)
    assert after.status == "review"
    assert after.claim_lock is None
    assert after.current_run_id is None

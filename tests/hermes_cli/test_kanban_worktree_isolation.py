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

from concurrent.futures import ThreadPoolExecutor
import json
import subprocess
import os
import shutil
import sys
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


def test_role_worktree_refuses_a_copied_git_pointer(kanban_home, tmp_path):
    repo = _make_repo(tmp_path)
    base = _head(repo)
    sibling = _add_worktree(repo, repo / ".worktrees" / "sibling", "wt/sibling")
    candidate = _commit(repo, "candidate.txt", "candidate\n")
    with kbc.connect_closing() as conn:
        implementation = _candidate_card(conn, repo, base=base, head=candidate)
        review = _role_card(conn, "review", implementation, repo, "reviewer")
        task = kb.get_task(conn, review)
    target = repo / ".worktrees" / review
    target.mkdir()
    pointer = (sibling / ".git").read_bytes()
    (target / ".git").write_bytes(pointer)
    (target / "README.md").write_bytes((sibling / "README.md").read_bytes())

    with pytest.raises(ValueError, match="candidate_workspace_unavailable.*unregistered checkout") as error:
        kbw.resolve_workspace(task, candidate_head=candidate)
    assert str(target) in str(error.value)
    assert _head(sibling) == _head(target) == base
    assert (target / ".git").read_bytes() == (sibling / ".git").read_bytes() == pointer
    assert (target / "README.md").read_bytes() == (sibling / "README.md").read_bytes() == b"base\n"
    assert not (target / "candidate.txt").exists()
    assert not (sibling / "candidate.txt").exists()
    assert _head(repo) == candidate


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


def test_claim_cli_leaves_a_role_card_unclaimed_when_its_workspace_is_refused(
    kanban_home, tmp_path, capsys,
):
    """Workspace resolution can refuse too, and it must refuse before the claim.

    A role card pinned to an operator ``dir`` sitting on another commit is a real
    refusal (that directory is never moved for you). Resolving it after
    ``claim_task`` committed ready->running would strand the card running with no
    worker holding it, so the fence has to come first.
    """
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
        # The handoff is recorded while the checkout still sits on the candidate…
        assert kb.complete_task(
            conn, implementation, expected_run_id=run.current_run_id,
            summary="implementation handoff",
            metadata={"base_sha": base, "head_sha": candidate, "changed_files": []})
        # …and the operator's repository checkout advances afterwards, past it.
        advanced = _commit(repo, "later.txt", "later\n")
        assert advanced != candidate
        # The operator pinned the review card to that checkout.
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET workspace_kind = 'dir', status = 'ready' WHERE id = ?",
                (review,))
        pinned = kb.get_task(conn, review)
        assert pinned.status == "ready"
        assert _head(repo) == advanced
        # A dirty operator checkout refuses too, with the handoff still valid: the
        # same card, refused by workspace resolution rather than by the candidate.
        (repo / "uncommitted.txt").write_text("dirty\n", encoding="utf-8")

    assert kc._cmd_claim(_claim_args(review)) != 0
    captured = capsys.readouterr()
    assert ("candidate_workspace_unavailable" in captured.out + captured.err
            or "parent handoff cannot be started" in captured.out + captured.err)
    with kbc.connect() as conn:
        after = kb.get_task(conn, review)
    assert after.status == "ready"
    assert after.claim_lock is None
    assert after.current_run_id is None
    assert after.worker_pid is None


def test_typed_role_card_with_the_default_scratch_kind_gets_the_candidate(
    kanban_home, tmp_path,
):
    """``scratch`` is create_task's default, so a role card can arrive without one.

    Before this fence the resolver returned the card's scratch path: empty, or — for a
    legacy explicit path — the repository's current tree instead of the candidate the
    card was created to review. Either way a verdict could be issued without ever
    seeing the implementation.
    """
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
        # Deliberately the default kind: no workspace_kind argument at all.
        review = kb.create_task(
            conn, title="review the candidate", assignee="reviewer",
            lifecycle_contract={"kind": "review", "candidate_task_id": implementation})
        run = kb.claim_task(conn, implementation, claimer=kb._claimer_id())
        assert run is not None
        assert kb.complete_task(
            conn, implementation, expected_run_id=run.current_run_id,
            summary="implementation handoff",
            metadata={"base_sha": base, "head_sha": candidate, "changed_files": []})
        task = kb.get_task(conn, review)
        assert task.workspace_kind == "scratch"
        head = kbw.candidate_head_for_task(conn, task)
        # An unanchored board refuses the anonymous scratch card instead of handing a
        # reviewer an empty directory to "review".
        with pytest.raises(ValueError, match="candidate_workspace_unavailable"):
            kbw.resolve_workspace(task, candidate_head=head)
        # Anchored, the same card materializes its own worktree at the candidate.
        kb.write_board_metadata(None, default_workdir=str(repo))
        workspace = kbw.resolve_workspace(task, candidate_head=head)

        # …and the same card with a legacy explicit scratch path.
        pinned = kb.create_task(
            conn, title="review from a pinned path", assignee="reviewer",
            workspace_kind="scratch", workspace_path=str(repo),
            lifecycle_contract={"kind": "review", "candidate_task_id": implementation})
        pinned_task = kb.get_task(conn, pinned)
        pinned_head = kbw.candidate_head_for_task(conn, pinned_task)
        pinned_workspace = kbw.resolve_workspace(pinned_task, candidate_head=pinned_head)
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET workspace_path = 'ignored' WHERE id = ?", (review,))
        advanced = _commit(repo, "later.txt", "later\n")

    assert advanced != candidate
    assert _head(repo) == advanced
    for path in (workspace, pinned_workspace):
        assert path.is_dir()
        assert _head(path) == candidate  # the candidate checkout, never the moving tree
        assert path != repo

def test_a_converted_role_card_is_recorded_as_a_worktree(kanban_home, tmp_path):
    """The kind that cleanup branches on must match the workspace that was resolved.

    ``scratch`` is create_task's default, so a role card resolves a linked worktree while
    the column still says scratch. Completion cleanup branches on the column: scratch is
    a plain directory delete, worktree is ``git worktree remove`` — so both call sites
    (``hermes kanban claim`` and the dispatcher) persist the effective kind.
    """
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
        review = kb.create_task(
            conn, title="review the candidate", assignee="reviewer",
            lifecycle_contract={"kind": "review", "candidate_task_id": implementation})
        plain = kb.create_task(conn, title="plain scratch work", assignee="worker")
        # A role card with no explicit path anchors on the board's repository.
        kb.write_board_metadata(None, default_workdir=str(repo))
        run = kb.claim_task(conn, implementation, claimer=kb._claimer_id())
        assert run is not None
        assert kb.complete_task(
            conn, implementation, expected_run_id=run.current_run_id,
            summary="implementation handoff",
            metadata={"base_sha": base, "head_sha": candidate, "changed_files": []})
        task = kb.get_task(conn, review)
        head = kbw.candidate_head_for_task(conn, task)
        workspace = kbw.resolve_workspace(task, candidate_head=head)

        # A role card always resolves as a worktree; ordinary scratch work does not.
        assert kbw.effective_workspace_kind(task) == "worktree"
        assert kbw.effective_workspace_kind(kb.get_task(conn, plain)) == "scratch"

        # Persisting exactly what the claim paths persist: path plus the effective kind.
        effective = kbw.effective_workspace_kind(task)
        kbw.set_workspace_path(
            conn, review, str(workspace),
            kind=effective if effective != (task.workspace_kind or "scratch") else None,
        )
        recorded = kb.get_task(conn, review)

    assert recorded.workspace_kind == "worktree"
    assert recorded.workspace_path == str(workspace)
    assert _head(Path(recorded.workspace_path)) == candidate


def test_cleanup_of_a_converted_role_card_uses_the_worktree_path(kanban_home, tmp_path):
    """The recorded kind is what decides cleanup, so a converted card removes cleanly."""
    from hermes_cli import kanban_db_workspace as workspace_module

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
        review = kb.create_task(
            conn, title="review the candidate", assignee="reviewer",
            workspace_path=str(repo),
            lifecycle_contract={"kind": "review", "candidate_task_id": implementation})
        run = kb.claim_task(conn, implementation, claimer=kb._claimer_id())
        assert run is not None
        assert kb.complete_task(
            conn, implementation, expected_run_id=run.current_run_id,
            summary="implementation handoff",
            metadata={"base_sha": base, "head_sha": candidate, "changed_files": []})
        task = kb.get_task(conn, review)
        head = kbw.candidate_head_for_task(conn, task)
        path = kbw.resolve_workspace(task, candidate_head=head)
        kbw.set_workspace_path(conn, review, str(path), kind="worktree")
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status = 'done' WHERE id = ?", (review,))
        registered = subprocess.run(
            ["git", "-C", str(repo), "worktree", "list", "--porcelain"],
            capture_output=True, text=True, check=True,
        ).stdout
        assert str(path) in registered

        workspace_module._cleanup_workspace(conn, review)

    leftover = subprocess.run(
        ["git", "-C", str(repo), "worktree", "list", "--porcelain"],
        capture_output=True, text=True, check=True,
    ).stdout
    assert str(path) not in leftover   # removed with git worktree remove, not rmtree
    assert not path.exists()

def test_claim_refuses_an_unclaimable_card_before_materializing_anything(
    kanban_home, tmp_path, capsys,
):
    """A refused claim must not leave a checkout (or a branch) behind.

    Materializing first meant a card that was already running, claimed, or still
    dependency-blocked still created ``.worktrees/<id>``; for a role card the leftover
    detached checkout could then refuse a later retry against a newer candidate.
    """
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
        review = kb.create_task(
            conn, title="review the candidate", assignee="reviewer",
            workspace_path=str(repo),
            lifecycle_contract={"kind": "review", "candidate_task_id": implementation})
        run = kb.claim_task(conn, implementation, claimer=kb._claimer_id())
        assert run is not None
        assert kb.complete_task(
            conn, implementation, expected_run_id=run.current_run_id,
            summary="implementation handoff",
            metadata={"base_sha": base, "head_sha": candidate, "changed_files": []})
        # The card is claim-eligible by dependency, but a live claim already owns it.
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status = 'ready', claim_lock = 'someone-else', "
                "current_run_id = 999 WHERE id = ?", (review,))
        target = repo / ".worktrees" / review
        assert not target.exists()

    assert kc._cmd_claim(_claim_args(review)) != 0
    assert "already claimed" in capsys.readouterr().err
    assert not target.exists()
    with kbc.connect() as conn:
        after = kb.get_task(conn, review)
    assert after.claim_lock == "someone-else"
    assert after.workspace_path == str(repo)

def test_clean_role_checkout_follows_a_new_candidate_after_rework(kanban_home, tmp_path):
    """REQUEST_CHANGES + rework leaves the review worktree on the rejected candidate.

    Refusing to move a clean, detached checkout there would make the second review
    undispatchable — every spawn failing on the relic of the first. The old candidate
    is contained in the new one, so following it strands nothing.
    """
    repo = _make_repo(tmp_path)
    _commit(repo, ".gitignore", ".worktrees/\n")
    base = _head(repo)
    first = _commit(repo, "candidate.txt", "candidate\n")

    with kbc.connect() as conn:
        implementation = kb.create_task(
            conn, title="implementation", assignee="coder", workspace_kind="worktree",
            workspace_path=str(repo),
            lifecycle_contract={
                "kind": "code", "review_mode": "separate_card", "reviewer": "reviewer",
                "validation_required": True,
            })
        review = _role_card(conn, "review", implementation, repo, "reviewer")
        run = kb.claim_task(conn, implementation, claimer=kb._claimer_id())
        assert run is not None
        assert kb.complete_task(
            conn, implementation, expected_run_id=run.current_run_id,
            summary="first attempt",
            metadata={"base_sha": base, "head_sha": first, "changed_files": []})
        task = kb.get_task(conn, review)
        workspace, _branch = kbw._resolve_worktree_workspace(
            task, candidate_head=kbw.candidate_head_for_task(conn, task))
    assert _head(workspace) == first

    # The implementer answers REQUEST_CHANGES with a second candidate on top.
    second = _commit(repo, "fix.txt", "fixed\n")
    with kbc.connect() as conn:
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (implementation,))
        run = kb.claim_task(conn, implementation, claimer=kb._claimer_id())
        assert run is not None
        assert kb.complete_task(
            conn, implementation, expected_run_id=run.current_run_id,
            summary="reworked candidate",
            metadata={"base_sha": base, "head_sha": second, "changed_files": []})
        resolved = kbw._resolve_worktree_workspace(
            kb.get_task(conn, review), candidate_head=second)
    assert resolved[0] == workspace
    assert _head(workspace) == second


@pytest.mark.parametrize("ref,survives", [
    (None, False),
    ("refs/worktree/saved", False),
    ("refs/bisect/saved", False),
    ("refs/rewritten/saved", False),
    ("refs/heads/saved", True),
    ("refs/tags/saved", True),
    ("refs/remotes/origin/saved", True),
    ("refs/stash", True),
    ("refs/other/saved", True),
])
def test_role_checkout_requires_a_reference_surviving_removal(kanban_home, tmp_path, ref, survives):
    """A detached unique commit needs a common ref, not one removed with its checkout."""
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
        run = kb.claim_task(conn, implementation, claimer=kb._claimer_id())
        assert run is not None
        assert kb.complete_task(
            conn, implementation, expected_run_id=run.current_run_id,
            summary="attempt",
            metadata={"base_sha": base, "head_sha": candidate, "changed_files": []})
        task = kb.get_task(conn, review)
        workspace, _branch = kbw._resolve_worktree_workspace(
            task, candidate_head=kbw.candidate_head_for_task(conn, task))

    stranded = _commit(workspace, "unique.txt", "preserve this work\n")
    if ref:
        _git(workspace, "update-ref", ref, stranded)
    with kbc.connect() as conn:
        task = kb.get_task(conn, review)
        if survives:
            kbw._resolve_worktree_workspace(task, candidate_head=candidate)
        else:
            with pytest.raises(ValueError, match="would strand work"):
                kbw._resolve_worktree_workspace(task, candidate_head=candidate)
    assert _head(repo) == candidate
    assert (repo / "candidate.txt").read_text() == "candidate\n"
    assert not (repo / "unique.txt").exists()
    if survives:
        assert _head(workspace) == candidate
        _git(repo, "worktree", "remove", str(workspace))
        _git(repo, "merge-base", "--is-ancestor", stranded, ref)
    else:
        assert _head(workspace) == stranded
        assert (workspace / "unique.txt").read_text() == "preserve this work\n"
        assert (workspace / "candidate.txt").read_text() == "candidate\n"
        if ref:
            _git(workspace, "merge-base", "--is-ancestor", stranded, ref)


def test_role_worktree_refuses_to_overwrite_an_ignored_local_file(kanban_home, tmp_path):
    """An ignored file is invisible to `git status`, so only this guard protects it.

    Verified git behaviour: `git checkout --detach` onto a commit that starts tracking
    an ignored path overwrites the local file and reports success, while
    `status --porcelain --untracked-files=all` calls the checkout clean.
    """
    repo = _make_repo(tmp_path)
    (repo / ".gitignore").write_text("notes.txt\n", encoding="utf-8")
    _git(repo, "add", ".gitignore")
    _git(repo, "commit", "-m", "ignore notes")
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
        run = kb.claim_task(conn, implementation, claimer=kb._claimer_id())
        assert run is not None
        assert kb.complete_task(
            conn, implementation, expected_run_id=run.current_run_id,
            summary="attempt",
            metadata={"base_sha": base, "head_sha": candidate, "changed_files": []})
        task = kb.get_task(conn, review)
        workspace, _branch = kbw._resolve_worktree_workspace(
            task, candidate_head=kbw.candidate_head_for_task(conn, task))
        # Only now does a newer commit start tracking the path the .gitignore covers.
        (repo / "notes.txt").write_text("tracked by the newer commit\n", encoding="utf-8")
        _git(repo, "add", "-f", "notes.txt")
        _git(repo, "commit", "-m", "track notes.txt")
        later = _head(repo)

    notes = workspace / "notes.txt"
    notes.write_text("operator notes\n", encoding="utf-8")
    # Git calls this clean: the file is ignored.
    status = subprocess.run(
        ["git", "-C", str(workspace), "status", "--porcelain", "--untracked-files=all"],
        capture_output=True, text=True, check=True,
    ).stdout
    assert status.strip() == ""

    with kbc.connect() as conn:
        with pytest.raises(ValueError, match="ignored file"):
            kbw._resolve_worktree_workspace(kb.get_task(conn, review), candidate_head=later)
    assert notes.read_text(encoding="utf-8") == "operator notes\n"
    assert _head(workspace) == candidate

def test_ignored_directory_that_the_candidate_turns_into_a_file_is_refused(
    kanban_home, tmp_path,
):
    """A file/directory conflict is overwritten too, and exact-path equality misses it."""
    repo = _make_repo(tmp_path)
    (repo / ".gitignore").write_text("cache\n", encoding="utf-8")
    _git(repo, "add", ".gitignore")
    _git(repo, "commit", "-m", "ignore cache")
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
        run = kb.claim_task(conn, implementation, claimer=kb._claimer_id())
        assert run is not None
        assert kb.complete_task(
            conn, implementation, expected_run_id=run.current_run_id,
            summary="attempt",
            metadata={"base_sha": base, "head_sha": candidate, "changed_files": []})
        task = kb.get_task(conn, review)
        workspace, _branch = kbw._resolve_worktree_workspace(
            task, candidate_head=kbw.candidate_head_for_task(conn, task))
        # The candidate turns the ignored directory's path into a tracked file.
        (repo / "cache").write_text("now a file\n", encoding="utf-8")
        _git(repo, "add", "-f", "cache")
        _git(repo, "commit", "-m", "cache is a file now")
        later = _head(repo)

    (workspace / "cache" / "local").mkdir(parents=True)
    (workspace / "cache" / "local" / "payload.bin").write_bytes(b"operator data\n")

    with kbc.connect() as conn:
        with pytest.raises(ValueError, match="ignored file"):
            kbw._resolve_worktree_workspace(kb.get_task(conn, review), candidate_head=later)
    assert (workspace / "cache" / "local" / "payload.bin").read_bytes() == b"operator data\n"
    assert _head(workspace) == candidate


@pytest.mark.parametrize("layout", ["exact", "nested", "addition"])
@pytest.mark.parametrize("ignored", [pytest.param(False, id="unignored"), pytest.param(True, id="ignored")])
@pytest.mark.parametrize("detached", [pytest.param(True, id="detached"), pytest.param(False, id="branch")])
def test_role_worktree_preserves_untracked_empty_directories(
    kanban_home, tmp_path, layout, ignored, detached,
):
    repo = _make_repo(tmp_path)
    base = _commit(repo, ".gitignore", "top\n" if ignored else "")
    candidate_path = "top" if layout == "exact" else "top/nested"
    (repo / candidate_path).parent.mkdir(parents=True, exist_ok=True)
    (repo / candidate_path).write_text("candidate file\n")
    _git(repo, "add", "-f", candidate_path)
    _git(repo, "commit", "-m", "track nested file")
    candidate = _head(repo)
    with kbc.connect() as conn:
        implementation = _candidate_card(conn, repo, base=base, head=candidate)
        review = _role_card(conn, "review", implementation, repo, "reviewer")
        task = kb.get_task(conn, review)
        workspace = repo / ".worktrees" / review
        checkout = ("--detach",) if detached else ("-b", f"wt/{review}")
        _git(repo, "worktree", "add", *checkout, str(workspace), base)
        (workspace / "top").mkdir()
        local = workspace / candidate_path
        if layout == "nested":
            local.mkdir()
        assert not subprocess.run(
            ["git", "-C", str(workspace), "status", "--porcelain", "--untracked-files=all"],
            capture_output=True, text=True, check=True,
        ).stdout
        if layout != "addition":
            with pytest.raises(ValueError, match=candidate_path):
                kbw._resolve_worktree_workspace(task, candidate_head=candidate)
            assert local.is_dir() and not list(local.iterdir())
            assert _head(workspace) == base
        else:
            kbw._resolve_worktree_workspace(task, candidate_head=candidate)
            assert (workspace / "top").is_dir()
            assert local.read_text() == "candidate file\n"
            assert _head(workspace) == candidate
        if not detached:
            _git(repo, "merge-base", "--is-ancestor", base, f"wt/{review}")


def test_cleanup_never_deletes_a_branch_the_role_never_recorded(kanban_home, tmp_path):
    """An empty recorded branch is the truth for a detached checkout, not ``wt/<id>``."""
    from hermes_cli import kanban_db_workspace as workspace_module

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
        run = kb.claim_task(conn, implementation, claimer=kb._claimer_id())
        assert run is not None
        assert kb.complete_task(
            conn, implementation, expected_run_id=run.current_run_id,
            summary="attempt",
            metadata={"base_sha": base, "head_sha": candidate, "changed_files": []})
        task = kb.get_task(conn, review)
        path, branch = kbw._resolve_worktree_workspace(
            task, candidate_head=kbw.candidate_head_for_task(conn, task))
        assert branch == ""

    # A same-named branch with unique work exists, and this card never used it.
    _git(repo, "branch", f"wt/{review}", base)
    with kbc.connect() as conn:
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status = 'done' WHERE id = ?", (review,))
        workspace_module._cleanup_worktree_workspace(review, str(path), branch)

    survivors = subprocess.run(
        ["git", "-C", str(repo), "branch", "--list", f"wt/{review}"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert survivors, "a branch this card never recorded must survive cleanup"


def test_discard_prepared_worktree_only_removes_a_clean_checkout(
    kanban_home, tmp_path,
):
    """A refused claim discards what it created, never operator work."""
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
        run = kb.claim_task(conn, implementation, claimer=kb._claimer_id())
        assert run is not None
        assert kb.complete_task(
            conn, implementation, expected_run_id=run.current_run_id,
            summary="attempt",
            metadata={"base_sha": base, "head_sha": candidate, "changed_files": []})
        task = kb.get_task(conn, review)
        path, _branch = kbw._resolve_worktree_workspace(
            task, candidate_head=kbw.candidate_head_for_task(conn, task))

        assert kbw.discard_prepared_worktree(
            task, path, candidate, prepared_new=False) is False
        assert path.is_dir()

        # A clean checkout at the prepared head is discarded…
        assert kbw.discard_prepared_worktree(task, path, candidate) is True
        assert not path.exists()

        # …and anything else is preserved: here, a checkout with local edits.
        path, _branch = kbw._resolve_worktree_workspace(
            task, candidate_head=kbw.candidate_head_for_task(conn, task))
        (path / "app.py").write_text("dirty\n", encoding="utf-8")
        assert kbw.discard_prepared_worktree(task, path, candidate) is False
        assert path.exists()


@pytest.mark.parametrize("referenced", [False, True])
def test_discard_preserves_the_last_candidate_reference(kanban_home, tmp_path, referenced):
    repo = _make_repo(tmp_path)
    base = _head(repo)
    _git(repo, "checkout", "-b", "candidate")
    candidate = _commit(repo, "candidate.txt", "candidate\n")
    _git(repo, "checkout", "main")
    with kbc.connect() as conn:
        implementation = _candidate_card(conn, repo, base=base, head=candidate)
        review = _role_card(conn, "review", implementation, repo, "reviewer")
        task = kb.get_task(conn, review)
        path, _branch = kbw._resolve_worktree_workspace(task, candidate_head=candidate)
        if not referenced:
            _git(repo, "update-ref", "-d", "refs/heads/candidate")

        assert kbw.discard_prepared_worktree(task, path, candidate)
        assert not path.exists()
        if not referenced:
            path, = path.parent.glob(f"{review}.recovery-*")
            assert _head(path) == candidate
            assert (path / "candidate.txt").read_text() == "candidate\n"
    assert _head(repo) == base


def test_discard_recovers_unreferenced_candidate_and_allows_rebound_retry(
    kanban_home, tmp_path,
):
    from hermes_cli.kanban_db_receipts import rebind_candidate
    from hermes_cli.kanban_runtime import runtime_identity

    repo = _make_repo(tmp_path)
    base = _head(repo)
    _git(repo, "checkout", "--detach", base)
    first = _commit(repo, "candidate.txt", "first\n")
    receipt = runtime_identity(Path(__file__).resolve().parents[2]).as_dict()
    with kbc.connect_closing() as conn:
        review = _ready_review_card(conn, repo, base, first)
        task = kb.get_task(conn, review)
        implementation = task.lifecycle_contract["candidate_task_id"]
        first_run = kb.get_task(conn, implementation).candidate_run_id
        _git(repo, "checkout", "main")
        second = _commit(repo, "candidate.txt", "second\n")
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (implementation,))
        second_run = kb.claim_task(conn, implementation, claimer=kb._claimer_id())
        assert second_run is not None
        assert kb.complete_task(
            conn, implementation, expected_run_id=second_run.current_run_id,
            summary="replacement candidate",
            metadata={"base_sha": base, "head_sha": second, "changed_files": []},
        )
        with kb.write_txn(conn):
            for run_id in (first_run, second_run.current_run_id):
                kb._append_event(conn, implementation, "spawned",
                                 {"runtime_identity": receipt}, run_id=run_id)
        rebind_candidate(
            conn, implementation, run_id=first_run, head_sha=first,
            reason="restore first", replace=True, replaced_run=second_run.current_run_id)
        path, _, created = kbw.resolve_workspace_created(task, candidate_head=first)
        assert created
        private = _commit(path, "private.txt", "private recovery history\n")
        _git(path, "update-ref", "refs/worktree/private", private)
        _git(path, "reset", "--hard", first)
        rebind_candidate(
            conn, implementation, run_id=second_run.current_run_id, head_sha=second,
            reason="use replacement", replace=True, replaced_run=first_run)
        assert not kbw._commit_referenced(repo, first)
        assert not kbw._is_ancestor(repo, first, second)
        assert kbw.discard_prepared_worktree(task, path, first, prepared_new=False) is False
        assert path.is_dir()
        _git(repo, "worktree", "lock", str(path))
        assert kbw.discard_prepared_worktree(task, path, first, prepared_new=created) is False
        assert _head(path) == first
        assert kb._git_out(path, "rev-parse", "refs/worktree/private") == private
        _git(repo, "worktree", "unlock", str(path))
        kbw.discard_prepared_worktree(task, path, first, prepared_new=created)
        assert not path.exists(), "failed preparation must free the canonical path for retry"
        recovery, = path.parent.glob(f"{review}.recovery-*")
        retry, _, retry_created = kbw.resolve_workspace_created(
            task, candidate_head=kbw.candidate_head_for_task(conn, task))
        assert retry == path and retry_created
        assert _head(retry) == second
        assert _head(recovery) == first
        assert kb._git_out(recovery, "rev-parse", "ORIG_HEAD") == private
        assert kb._git_out(recovery, "rev-parse", "refs/worktree/private") == private
        assert private in kb._git_out(recovery, "reflog", "show", "--format=%H", "HEAD")
        unreachable = kb._git_out(recovery, "fsck", "--unreachable") or ""
        assert first not in unreachable and private not in unreachable


@pytest.mark.parametrize("namespace", ["worktree", "bisect", "rewritten"])
@pytest.mark.parametrize("surviving", [False, True])
def test_discard_protects_every_worktree_ref_commit(
    kanban_home, tmp_path, caplog, namespace, surviving,
):
    repo = _make_repo(tmp_path)
    base = _head(repo)
    candidate = _commit(repo, "candidate.txt", "candidate\n")
    with kbc.connect() as conn:
        implementation = _candidate_card(conn, repo, base=base, head=candidate)
        review = _role_card(conn, "review", implementation, repo, "reviewer")
        task = kb.get_task(conn, review)
        path, _branch = kbw._resolve_worktree_workspace(task, candidate_head=candidate)
        unique = _commit(path, "private.txt", "worktree-only commit\n")
        ref = f"refs/{namespace}/private"
        _git(path, "update-ref", ref, unique)
        _git(path, "checkout", "--detach", candidate)
        if surviving:
            _git(repo, "update-ref", "refs/keep/rescued", unique)

        with caplog.at_level("INFO", logger=kb._log.name):
            assert kbw.discard_prepared_worktree(task, path, candidate)
        assert not path.exists()
        if not surviving:
            path, = path.parent.glob(f"{review}.recovery-*")
            assert _head(path) == candidate
            assert unique in caplog.text and ref in caplog.text
            assert kb._git_out(path, "rev-parse", "--verify", ref) == unique
        else:
            assert kb._git_out(repo, "rev-parse", "--verify", "refs/keep/rescued") == unique
    assert _head(repo) == candidate


@pytest.mark.parametrize("anchor", ["both", "ORIG_HEAD", "reflog"])
@pytest.mark.parametrize("surviving", [False, True])
def test_discard_protects_private_recovery_anchors(
    kanban_home, tmp_path, caplog, anchor, surviving,
):
    repo = _make_repo(tmp_path)
    base = _head(repo)
    candidate = _commit(repo, "candidate.txt", "candidate\n")
    with kbc.connect() as conn:
        implementation = _candidate_card(conn, repo, base=base, head=candidate)
        review = _role_card(conn, "review", implementation, repo, "reviewer")
        task = kb.get_task(conn, review)
        path, _branch = kbw._resolve_worktree_workspace(task, candidate_head=candidate)
        unique = _commit(path, "operator.txt", "operator recovery commit\n")
        _git(path, "reset", "--hard", candidate)
        assert kb._git_out(path, "rev-parse", "ORIG_HEAD") == unique
        if anchor == "ORIG_HEAD":
            _git(path, "reflog", "expire", "--expire=now", "HEAD")
        elif anchor == "reflog":
            _git(path, "update-ref", "-d", "ORIG_HEAD")
        if surviving:
            _git(repo, "update-ref", "refs/keep/operator", unique)
        with caplog.at_level("INFO", logger=kb._log.name):
            assert kbw.discard_prepared_worktree(task, path, candidate)
        assert not path.exists()
        if surviving:
            assert kb._git_out(repo, "rev-parse", "refs/keep/operator") == unique
        else:
            path, = path.parent.glob(f"{review}.recovery-*")
            assert unique in caplog.text
            if anchor != "reflog":
                assert kb._git_out(path, "rev-parse", "ORIG_HEAD") == unique
                assert unique in kb._git_out(path, "rev-list", "ORIG_HEAD")
            if anchor != "ORIG_HEAD":
                assert unique in kb._git_out(path, "reflog", "show", "--format=%H", "HEAD")
                # Recovery history must keep U reachable, not merely unpruned.
                assert unique not in (kb._git_out(path, "fsck", "--unreachable") or "")


@pytest.mark.parametrize("root_ref", ["ORIG_HEAD", "NOTES_MERGE_PARTIAL", "NOTES_MERGE_REF",
                                    "FETCH_HEAD", "MERGE_HEAD"])
@pytest.mark.parametrize("operation", ["discard", "cleanup"])
@pytest.mark.parametrize("anchor", ["unreferenced", "referenced", "enumeration_failure"])
def test_worktree_removal_with_git_without_root_ref_option(
    kanban_home, tmp_path, monkeypatch, caplog, operation, anchor, root_ref,
):
    """Old Git must still protect private history without blocking safe removal."""
    repo = _make_repo(tmp_path)
    base = _head(repo)
    candidate = _commit(repo, "candidate.txt", "candidate\n")
    _git(repo, "update-ref", "refs/remotes/origin/main", candidate)
    with kbc.connect() as conn:
        implementation = _candidate_card(conn, repo, base=base, head=candidate)
        review = _role_card(conn, "review", implementation, repo, "reviewer")
        task = kb.get_task(conn, review)
        path, branch = kbw._resolve_worktree_workspace(task, candidate_head=candidate)
    unique = _commit(path, "operator.txt", f"{root_ref}-only recovery\n")
    _git(path, "reset", "--hard", candidate)
    _git(path, "reflog", "expire", "--expire=now", "HEAD")
    if root_ref != "ORIG_HEAD":
        _git(path, "update-ref", "-d", "ORIG_HEAD")
        git_dir = Path(kb._git_out(path, "rev-parse", "--absolute-git-dir"))
        # FETCH_HEAD and MERGE_HEAD can hold more than one commit. The second
        # object must be checked too, not merely the first peelable revision.
        contents = f"{candidate}\n{unique}\n" if root_ref in {"FETCH_HEAD", "MERGE_HEAD"} else f"{unique}\n"
        (git_dir / root_ref).write_text(contents)
    if anchor != "unreferenced":
        _git(repo, "update-ref", "refs/keep/operator", unique)

    real_git = shutil.which("git")
    binary_dir = tmp_path / "bin"
    binary_dir.mkdir()
    stub = binary_dir / "git"
    stub.write_text(
        f"#!{sys.executable}\n"
        "import os, sys\n"
        "args = sys.argv[1:]\n"
        "if '--include-root-refs' in args:\n"
        "    sys.stderr.write(\"error: unknown option `include-root-refs'\\n\")\n"
        "    sys.exit(129)\n"
        "if 'for-each-ref' in args and '-h' in args:\n"
        "    sys.stdout.write('usage: git for-each-ref [options]\\n')\n"
        "    sys.exit(129)\n"
        f"if {anchor == 'enumeration_failure'!r} and args[-2:] == ['for-each-ref', '--format=%(refname)']:\n"
        "    sys.stderr.write('fatal: unable to enumerate refs\\n')\n"
        "    sys.exit(128)\n"
        f"os.execv({real_git!r}, [{real_git!r}, *args])\n"
    )
    stub.chmod(0o700)
    monkeypatch.setenv("PATH", str(binary_dir) + os.pathsep + os.environ["PATH"])
    with caplog.at_level("INFO", logger=kb._log.name):
        if operation == "discard":
            assert kbw.discard_prepared_worktree(task, path, candidate)
        else:
            kbw._cleanup_worktree_workspace(review, str(path), branch)
    if operation == "discard" and anchor != "referenced":
        assert not path.exists()
        path, = path.parent.glob(f"{review}.recovery-*")
    assert path.exists() is (anchor != "referenced")
    if anchor == "unreferenced":
        assert unique in caplog.text and root_ref in caplog.text
        assert unique in (Path(kb._git_out(path, "rev-parse", "--absolute-git-dir")) / root_ref).read_text()
    elif anchor == "enumeration_failure":
        assert "cannot enumerate checkout refs" in caplog.text


@pytest.mark.parametrize("root_ref", ["FETCH_HEAD", "MERGE_HEAD"])
@pytest.mark.parametrize("operation", ["discard", "cleanup"])
@pytest.mark.parametrize("surviving", [False, True])
def test_modern_git_removal_protects_multivalued_pseudorefs(
    kanban_home, tmp_path, monkeypatch, caplog, root_ref, operation, surviving,
):
    repo = _make_repo(tmp_path)
    candidate = _head(repo)
    _git(repo, "update-ref", "refs/remotes/origin/main", candidate)
    with kbc.connect() as conn:
        implementation = _candidate_card(conn, repo, base=candidate, head=candidate)
        review = _role_card(conn, "review", implementation, repo, "reviewer")
        task = kb.get_task(conn, review)
        path, branch = kbw._resolve_worktree_workspace(task, candidate_head=candidate)
    # Fetch without a destination ref: no local branch, ORIG_HEAD or reflog
    # ever points at this commit in the repository being cleaned up.
    source = tmp_path / "source"
    _git(tmp_path, "clone", str(repo), str(source))
    unique = _commit(source, "operator.txt", "fetched recovery\n")
    _git(path, "fetch", str(source), "main")
    git_dir = Path(kb._git_out(path, "rev-parse", "--absolute-git-dir"))
    fetched = (git_dir / "FETCH_HEAD").read_text()
    assert fetched.split()[0] == unique
    if root_ref == "FETCH_HEAD":
        (git_dir / root_ref).write_text(f"{candidate}\t\talready saved\n" + fetched)
    else:
        (git_dir / "FETCH_HEAD").unlink()
        (git_dir / root_ref).write_text(f"{candidate}\n{unique}\n")
    if surviving:
        _git(repo, "update-ref", "refs/keep/operator", unique)

    real_git = kbw._git
    enumerated = []

    def modern_git(repo_root, *args, timeout):
        if args == ("for-each-ref", "-h"):
            return subprocess.CompletedProcess(args, 129, "usage: --include-root-refs\n", "")
        if "--include-root-refs" in args:
            enumerated.append(True)
            # Modern Git includes root refs but omits these multivalued files.
            # HEAD is the only other root ref in this fixture.
            result = real_git(repo_root, *(arg for arg in args if arg != "--include-root-refs"),
                              timeout=timeout)
            result.stdout += "HEAD\n"
            return result
        return real_git(repo_root, *args, timeout=timeout)

    monkeypatch.setattr(kbw, "_git", modern_git)
    with caplog.at_level("INFO", logger=kb._log.name):
        if operation == "discard":
            assert kbw.discard_prepared_worktree(task, path, candidate)
        else:
            kbw._cleanup_worktree_workspace(review, str(path), branch)
    assert enumerated
    if operation == "discard" and not surviving:
        assert not path.exists()
        path, = path.parent.glob(f"{review}.recovery-*")
    assert path.is_dir() is not surviving
    if not surviving:
        assert unique in caplog.text and root_ref in caplog.text
        assert unique in (git_dir / root_ref).read_text()
        assert unique in kb._git_out(path, "rev-list", unique)
    else:
        assert kb._git_out(repo, "rev-parse", "refs/keep/operator") == unique


@pytest.mark.parametrize("kind", ["blob", "tree", "commit"])
@pytest.mark.parametrize("operation", ["discard", "cleanup"])
@pytest.mark.parametrize("surviving", [False, True])
def test_worktree_removal_protects_noncommit_anchors(
    kanban_home, tmp_path, caplog, kind, operation, surviving,
):
    repo = _make_repo(tmp_path)
    candidate = _head(repo)
    _git(repo, "update-ref", "refs/remotes/origin/main", candidate)
    with kbc.connect() as conn:
        implementation = _candidate_card(conn, repo, base=candidate, head=candidate)
        review = _role_card(conn, "review", implementation, repo, "reviewer")
        task = kb.get_task(conn, review)
        path, branch = kbw._resolve_worktree_workspace(task, candidate_head=candidate)
    # Build objects without moving HEAD or leaving recovery reflogs.
    payload = tmp_path / "saved.txt"
    payload.write_text("private saved object\n")
    blob = kb._git_out(path, "hash-object", "-w", str(payload))
    tree = subprocess.run(
        ["git", "-C", str(path), "mktree"], input=f"100644 blob {blob}\tsaved.txt\n",
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    commit = kb._git_out(
        path, "-c", "user.name=Test User", "-c", "user.email=test@example.com",
        "commit-tree", tree, "-p", candidate, "-m", "saved objects")
    unique = {"blob": blob, "tree": tree, "commit": commit}[kind]
    ref = "refs/worktree/saved"
    _git(path, "update-ref", ref, unique)
    assert kb._git_out(path, "cat-file", "-t", unique) == kind
    if surviving:
        # A common commit ref transitively retains its tree and blob too.
        _git(repo, "update-ref", "refs/keep/operator", commit)
    else:
        # The main checkout's private ref is not evidence of a common anchor.
        _git(repo, "update-ref", "refs/worktree/not-common", commit)
    with caplog.at_level("INFO", logger=kb._log.name):
        if operation == "discard":
            assert kbw.discard_prepared_worktree(task, path, candidate)
        else:
            kbw._cleanup_worktree_workspace(review, str(path), branch)
    if operation == "discard" and not surviving:
        assert not path.exists()
        path, = path.parent.glob(f"{review}.recovery-*")
    assert path.is_dir() is not surviving
    if not surviving:
        assert unique in caplog.text and ref in caplog.text
        assert kb._git_out(path, "rev-parse", ref) == unique
    else:
        assert unique in kb._git_out(repo, "rev-list", "--objects", "refs/keep/operator")



@pytest.mark.parametrize("surviving", [False, True])
def test_role_completion_protects_worktree_rescue_ref(
    kanban_home, tmp_path, caplog, surviving,
):
    from hermes_cli import kanban as kc

    repo = _make_repo(tmp_path)
    base = _head(repo)
    candidate = _commit(repo, "candidate.txt", "candidate\n")
    _git(repo, "update-ref", "refs/remotes/origin/main", candidate)
    kb.write_board_metadata(None, default_workdir=str(repo))
    with kbc.connect_closing() as conn:
        review = _ready_review_card(conn, repo, base, candidate)
        kbw.set_workspace_path(conn, review, "", kind="scratch")
    assert kc._cmd_claim(_claim_args(review)) == 0
    path = repo / ".worktrees" / review
    unique = _commit(path, "rescue.txt", "worktree-only commit\n")
    ref = "refs/worktree/rescue"
    _git(path, "update-ref", ref, unique)
    _git(path, "checkout", "--detach", candidate)
    if surviving:
        _git(repo, "update-ref", "refs/keep/rescued", unique)
    with kbc.connect_closing() as conn:
        task = kb.get_task(conn, review)
        assert task.workspace_kind == "worktree"
        with caplog.at_level("INFO", logger=kb._log.name):
            assert kb.complete_task(
                conn, review, expected_run_id=task.current_run_id,
                summary="approved", verdict="APPROVE", metadata={"head_sha": candidate})
        assert kb.get_task(conn, review).status == "done"
    assert path.exists() is not surviving
    if not surviving:
        assert _head(path) == candidate
        assert kb._git_out(path, "rev-parse", "--verify", ref) == unique
        assert unique in caplog.text and ref in caplog.text
    else:
        assert kb._git_out(repo, "rev-parse", "--verify", "refs/keep/rescued") == unique


def test_discard_preserves_a_worktree_holding_ignored_content(kanban_home, tmp_path):
    """`git worktree remove` deletes ignored files too, so any ignored entry preserves it."""
    repo = _make_repo(tmp_path)
    (repo / ".gitignore").write_text("private.tmp\n", encoding="utf-8")
    _git(repo, "add", ".gitignore")
    _git(repo, "commit", "-m", "ignore scratch state")
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
        run = kb.claim_task(conn, implementation, claimer=kb._claimer_id())
        assert run is not None
        assert kb.complete_task(
            conn, implementation, expected_run_id=run.current_run_id,
            summary="attempt",
            metadata={"base_sha": base, "head_sha": candidate, "changed_files": []})
        task = kb.get_task(conn, review)
        path, _branch = kbw._resolve_worktree_workspace(
            task, candidate_head=kbw.candidate_head_for_task(conn, task))
        (path / "private.tmp").write_text("operator state\n", encoding="utf-8")
        # Nothing is dirty to git: the file is ignored.
        assert kbw.discard_prepared_worktree(task, path, candidate) is False
    assert (path / "private.tmp").read_text(encoding="utf-8") == "operator state\n"


def test_cleanup_preserves_an_adopted_branch_holding_unique_work(kanban_home, tmp_path):
    """A recorded ``wt/*`` branch is deleted only when a remote ref already contains it."""
    from hermes_cli import kanban_db_workspace as workspace_module

    repo = _make_repo(tmp_path)
    base = _head(repo)
    candidate = _commit(repo, "candidate.txt", "candidate\n")
    # The adopted branch holds a commit no remote-tracking ref contains.
    _git(repo, "branch", "wt/adopted", base)
    _git(repo, "checkout", "-q", "wt/adopted")
    unique = _commit(repo, "unique.txt", "only here\n")
    _git(repo, "checkout", "-q", base)

    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="worktree task", assignee="worker", workspace_kind="worktree",
            workspace_path=str(repo))
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status = 'done', workspace_path = ? WHERE id = ?",
                (str(repo / ".worktrees" / tid), tid))
    worktree = repo / ".worktrees" / tid
    worktree.parent.mkdir(parents=True, exist_ok=True)
    _git(repo, "worktree", "add", "-q", "--detach", str(worktree), candidate)
    assert _head(worktree) == candidate

    workspace_module._cleanup_worktree_workspace(tid, str(worktree), "wt/adopted")

    survives = subprocess.run(
        ["git", "-C", str(repo), "branch", "--list", "wt/adopted"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert survives, "a branch with commits no remote contains must survive cleanup"
    assert subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "wt/adopted"],
        capture_output=True, text=True, check=True,
    ).stdout.strip() == unique

def test_discard_preserves_an_unignored_empty_directory(kanban_home, tmp_path):
    repo = _make_repo(tmp_path)
    base = _head(repo)
    candidate = _commit(repo, "candidate.txt", "candidate\n")
    with kbc.connect_closing() as conn:
        review = _ready_review_card(conn, repo, base, candidate)
        task = kb.get_task(conn, review)
        path, _, created = kbw.resolve_workspace_created(task, candidate_head=candidate)
        (path / "operator-empty").mkdir()
        assert not kb._git_out(path, "status", "--porcelain", "--untracked-files=all")
        assert kbw.discard_prepared_worktree(
            task, path, candidate, prepared_new=created) is False
        assert (path / "operator-empty").is_dir()
        (path / "operator-empty").rmdir()
        assert kbw.discard_prepared_worktree(task, path, candidate, prepared_new=created)
        assert not path.exists()


def test_discard_preserves_a_worktree_holding_an_ignored_empty_directory(
    kanban_home, tmp_path,
):
    """A plain `ls-files` listing omits ignored directories, and removal deletes them."""
    repo = _make_repo(tmp_path)
    (repo / ".gitignore").write_text("empty/\n", encoding="utf-8")
    _git(repo, "add", ".gitignore")
    _git(repo, "commit", "-m", "ignore empty dir")
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
        run = kb.claim_task(conn, implementation, claimer=kb._claimer_id())
        assert run is not None
        assert kb.complete_task(
            conn, implementation, expected_run_id=run.current_run_id,
            summary="attempt",
            metadata={"base_sha": base, "head_sha": candidate, "changed_files": []})
        task = kb.get_task(conn, review)
        path, _branch = kbw._resolve_worktree_workspace(
            task, candidate_head=kbw.candidate_head_for_task(conn, task))
        (path / "empty").mkdir()
        assert kbw.discard_prepared_worktree(task, path, candidate) is False
    assert (path / "empty").is_dir()


def test_cleanup_preserves_a_branch_shadowed_by_a_same_named_tag(kanban_home, tmp_path):
    """`rev-parse --verify <name>` picks the tag and warns; the branch tip is what matters."""
    from hermes_cli import kanban_db_workspace as workspace_module

    repo = _make_repo(tmp_path)
    base = _head(repo)
    candidate = _commit(repo, "candidate.txt", "candidate\n")
    _git(repo, "branch", "wt/shadowed", base)
    _git(repo, "tag", "wt/shadowed", candidate)     # same short name, newer tip

    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="worktree task", assignee="worker", workspace_kind="worktree",
            workspace_path=str(repo))
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status = 'done' WHERE id = ?", (tid,))
    worktree = repo / ".worktrees" / tid
    worktree.parent.mkdir(parents=True, exist_ok=True)
    _git(repo, "worktree", "add", "-q", "--detach", str(worktree), candidate)

    workspace_module._cleanup_worktree_workspace(tid, str(worktree), "wt/shadowed")

    survives = subprocess.run(
        ["git", "-C", str(repo), "branch", "--list", "wt/shadowed"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert survives, "the branch tip is not contained by any remote ref, so it must survive"

def test_discard_refuses_after_a_real_claim_release_cycle(kanban_home, tmp_path):
    """A claim/release rewinds every task column, so only the run ledger remembers it.

    Produced here by a real cycle (claim, then the dispatcher's release path) rather than
    an edited row: `claim_lock`, `current_run_id` and `worker_pid` all return to NULL and
    `started_at` keeps the first claim's value, so the ownership tuple compares equal.
    """
    from hermes_cli import kanban as kc
    from hermes_cli.kanban_db_dispatch import _record_task_failure

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
        run = kb.claim_task(conn, implementation, claimer=kb._claimer_id())
        assert run is not None
        assert kb.complete_task(
            conn, implementation, expected_run_id=run.current_run_id,
            summary="attempt",
            metadata={"base_sha": base, "head_sha": candidate, "changed_files": []})
        pending = kb.get_task(conn, review)
        path, _branch = kbw._resolve_worktree_workspace(
            pending, candidate_head=kbw.candidate_head_for_task(conn, pending))
        prepared_head = kbw.candidate_head_for_task(conn, pending)
        prepared_run_head = kb._run_ledger_head(conn, review)

        # A real cycle: the row is taken exactly as a claim takes it (a running row plus a
        # new run), then the dispatcher's release path ends it.
        with kb.write_txn(conn):
            run_id = conn.execute(
                "INSERT INTO task_runs (task_id, profile, status, started_at) "
                "VALUES (?, 'rival', 'running', 1)", (review,),
            ).lastrowid
            conn.execute(
                "UPDATE tasks SET status = 'running', claim_lock = 'rival-lock', "
                "claim_expires = 1, current_run_id = ?, worker_pid = 999999, "
                "worker_started_at = 1 WHERE id = ?", (run_id, review))
        _record_task_failure(
            conn, review, "spawn failed", outcome="spawn_failed",
            release_claim=True, end_run=True,
        )
        # Restore only the status the retry path chose; the release itself nulled the
        # lock/run/pid, which is the state a loser would read.
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status = ?, claim_lock = NULL, current_run_id = NULL, "
                "worker_pid = NULL, consecutive_failures = 0, last_failure_error = NULL "
                "WHERE id = ?", (pending.status, review))
        after = kb.get_task(conn, review)

        assert kc._claim_ownership(after) == kc._claim_ownership(pending), (
            "a released cycle must be invisible to the task columns")
        assert kb._run_ledger_head(conn, review) > prepared_run_head

        assert kc._discard_if_still_ours(
            conn, pending, path, prepared_head, prepared_run_head, prepared_new=True) is False
    assert path.exists()


def test_discard_reclaims_an_ordinary_prepared_worktree(kanban_home, tmp_path):
    """A non-role card has no candidate head, so the prepared checkout supplies it.

    Without that, the guard compared against "" and could never reclaim an ordinary
    prepared worktree, leaving it behind for a cleanup pass that follows the recorded
    repository root instead.
    """
    repo = _make_repo(tmp_path)
    base = _head(repo)

    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="ordinary worktree work", assignee="worker",
            workspace_kind="worktree", workspace_path=str(repo))
        task = kb.get_task(conn, tid)
        assert kbw.candidate_head_for_task(conn, task) is None
        path = kbw.resolve_workspace(task, candidate_head=None)
        prepared_head = kbw._git_head(path)
        assert prepared_head
        assert kbw.discard_prepared_worktree(task, path, prepared_head) is True
    assert not path.exists()
    assert _head(repo) == base


def test_claim_cli_claims_an_ordinary_card(kanban_home, tmp_path, capsys):
    """Every non-role card takes this path: no candidate, so nothing to short-circuit.

    The prepared-head computation once read the resolved workspace before it existed,
    raising UnboundLocalError for every ordinary claim.
    """
    from hermes_cli import kanban as kc

    repo = _make_repo(tmp_path)
    with kbc.connect() as conn:
        tid = kb.create_task(
            conn, title="ordinary worktree card", assignee="worker",
            workspace_kind="worktree", workspace_path=str(repo))
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (tid,))

    assert kc._cmd_claim(_claim_args(tid)) == 0, capsys.readouterr().err
    with kbc.connect() as conn:
        claimed = kb.get_task(conn, tid)
    assert claimed.status == "running"
    assert claimed.workspace_path and claimed.workspace_path != str(repo)
    assert _head(Path(claimed.workspace_path)) == _head(repo)


@pytest.mark.parametrize("nested_metadata", [True, False])
def test_role_candidate_head_follows_the_bound_run_not_the_newest(
    kanban_home, tmp_path, nested_metadata,
):
    """After rebinding to an earlier attested run, the candidate is that run's head.

    `latest_handoff` answers "newest", so a rebind to run A while run B is newer used to
    raise candidate_workspace_unavailable for every review attempt.
    """
    from hermes_cli.kanban_db_receipts import rebind_candidate
    from hermes_cli.kanban_runtime import runtime_identity

    receipt = runtime_identity(Path(__file__).resolve().parents[2]).as_dict()
    repo = _make_repo(tmp_path)
    base = _head(repo)
    first = _commit(repo, "candidate.txt", "candidate\n")

    with kbc.connect() as conn:
        implementation = kb.create_task(
            conn, title="implementation", assignee="coder", workspace_kind="worktree",
            workspace_path=str(repo),
            lifecycle_contract={
                "kind": "code", "review_mode": "separate_card", "reviewer": "reviewer",
                "validation_required": True,
            })
        run_a = kb.claim_task(conn, implementation, claimer=kb._claimer_id())
        assert run_a is not None
        assert kb.complete_task(
            conn, implementation, expected_run_id=run_a.current_run_id,
            summary="first attempt",
            metadata={"base_sha": base, "head_sha": first, "changed_files": []})
        run_a_id = run_a.current_run_id
        # A second, newer attested run on a different head.
        second = _commit(repo, "second.txt", "second\n")
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (implementation,))
        run_b = kb.claim_task(conn, implementation, claimer=kb._claimer_id())
        assert run_b is not None
        assert kb.complete_task(
            conn, implementation, expected_run_id=run_b.current_run_id,
            summary="second attempt",
            metadata={"base_sha": base, "head_sha": second, "changed_files": []})
        # Both handoffs have launch attestations. Older records may lack the nested
        # lifecycle envelope, but their top-level head is still authoritative.
        with kb.write_txn(conn):
            for run in (run_a, run_b):
                kb._append_event(
                    conn, implementation, "spawned", {"runtime_identity": receipt},
                    run_id=run.current_run_id)
            if not nested_metadata:
                metadata = json.loads(conn.execute(
                    "SELECT metadata FROM task_runs WHERE id = ?", (run_a_id,)).fetchone()[0])
                metadata.pop("lifecycle", None)
                conn.execute(
                    "UPDATE task_runs SET metadata = ? WHERE id = ?",
                    (json.dumps(metadata), run_a_id))
        rebind_candidate(
            conn, implementation, run_id=run_a_id, head_sha=first,
            reason="restore earlier candidate", replace=True, replaced_run=run_b.current_run_id)
        review = _role_card(conn, "review", implementation, repo, "reviewer")
        kb.link_tasks(conn, implementation, review, requirement="phase_finished")
        task = kb.get_task(conn, review)
        state = kb.get_lifecycle_state(conn, review)
        assert state["candidate_run_id"] == run_a_id
        assert state["head_sha"] == first
        bound = kbw.candidate_head_for_task(conn, task)
        newest = kb.latest_handoff(conn, implementation)["head_sha"]
        context = kb.build_worker_context(conn, review)
        assert f"- Head: `{first}`" in context
        assert "first attempt" in context
        assert second not in context
        assert "second attempt" not in context
        # Restore the implementation checkout too; claim admission correctly
        # refuses a bound handoff whose recorded workspace still points at B.
        _git(repo, "checkout", "--detach", first)
        workspace = kbw.resolve_workspace(task, candidate_head=bound)
        kbw.set_workspace_path(conn, review, str(workspace))
        kb.recompute_ready(conn)
        review_run = kb.claim_review_task(conn, review, claimer=kb._claimer_id())
        assert review_run is not None
        assert kb.complete_task(
            conn, review, expected_run_id=review_run.current_run_id,
            summary="approve restored candidate", verdict="APPROVE",
            metadata={"head_sha": first},
        )
        evidence = json.loads(conn.execute(
            "SELECT metadata FROM task_runs WHERE id = ?",
            (review_run.current_run_id,)).fetchone()[0])["lifecycle"]
        from hermes_cli.kanban_lifecycle import _freshness

        assert _freshness(conn, evidence, implementation, evidence_task_id=review) == (
            True, None,
        ), f"candidate run {run_a_id}, review run {review_run.current_run_id}: {evidence}"
        assert evidence["candidate_run_id"] == run_a_id != review_run.current_run_id
        assert evidence["head_sha"] == first
        assert kb.get_lifecycle_state(conn, implementation)["review_verdict"] == "APPROVE"
        validation = _role_card(conn, "validation", implementation, repo, "tester")
        kb.link_tasks(conn, review, validation, requirement="review_approved")
        kb.recompute_ready(conn)
        assert kb.evaluate_dependencies(conn, validation)["satisfied"]
        validation_run = kb.claim_task(conn, validation, claimer=kb._claimer_id())
        assert validation_run is not None
        assert kb.complete_task(
            conn, validation, expected_run_id=validation_run.current_run_id,
            summary="validated restored candidate", verdict="PASS",
            metadata={"head_sha": first},
        )
        validation_evidence = json.loads(conn.execute(
            "SELECT metadata FROM task_runs WHERE id = ?",
            (validation_run.current_run_id,)).fetchone()[0])["lifecycle"]
        assert validation_evidence["candidate_run_id"] == run_a_id
        assert _freshness(conn, validation_evidence, implementation,
                          evidence_task_id=validation) == (True, None)

    assert bound == first
    assert newest == second          # the newest handoff is the rejected candidate


def test_same_card_review_completes_with_the_rebound_implementation_head(
    kanban_home, tmp_path,
):
    from hermes_cli.kanban_db_receipts import rebind_candidate
    from hermes_cli.kanban_runtime import runtime_identity

    receipt = runtime_identity(Path(__file__).resolve().parents[2]).as_dict()
    repo = _make_repo(tmp_path)
    base = _head(repo)
    first = _commit(repo, "app.py", "first\n")
    with kbc.connect_closing() as conn:
        task_id = kb.create_task(
            conn, title="same-card candidate", assignee="coder",
            workspace_kind="dir", workspace_path=str(repo),
            lifecycle_contract={
                "kind": "code", "review_mode": "same_card", "reviewer": "reviewer",
                "validation_required": False,
            },
        )
        first_run = kb.claim_task(conn, task_id, claimer=kb._claimer_id())
        assert first_run is not None
        assert kb.complete_task(
            conn, task_id, expected_run_id=first_run.current_run_id, summary="first",
            metadata={"base_sha": base, "head_sha": first, "changed_files": ["app.py"]},
        )
        review = kb.claim_review_task(conn, task_id, claimer=kb._claimer_id())
        assert review is not None
        # Start a second implementation without rejecting H1: opposite verdicts
        # on the restored run would intentionally leave acceptance conflicted.
        assert kb.block_task(
            conn, task_id, expected_run_id=review.current_run_id,
            reason="operator considering another candidate", kind="needs_input",
        )
        assert kb.promote_task(
            conn, task_id, actor="operator", reason="try another implementation", force=True,
        ) == (True, None)
        second_run = kb.claim_task(conn, task_id, claimer=kb._claimer_id())
        assert second_run is not None
        second = _commit(repo, "app.py", "second\n")
        assert kb.complete_task(
            conn, task_id, expected_run_id=second_run.current_run_id, summary="second",
            metadata={"base_sha": base, "head_sha": second, "changed_files": ["app.py"]},
        )
        with kb.write_txn(conn):
            for run in (first_run, second_run):
                kb._append_event(
                    conn, task_id, "spawned", {"runtime_identity": receipt},
                    run_id=run.current_run_id,
                )
        rebind_candidate(
            conn, task_id, run_id=first_run.current_run_id, head_sha=first,
            reason="restore earlier implementation", replace=True,
            replaced_run=second_run.current_run_id,
        )
        _git(repo, "checkout", "--detach", first)
        assert kb.latest_handoff(conn, task_id)["head_sha"] == second
        review = kb.claim_review_task(conn, task_id, claimer=kb._claimer_id())
        assert review is not None
        assert kb.complete_task(
            conn, task_id, expected_run_id=review.current_run_id, verdict="APPROVE",
            summary="approve restored implementation", metadata={"head_sha": first},
        )
        evidence = json.loads(conn.execute(
            "SELECT metadata FROM task_runs WHERE id = ?", (review.current_run_id,),
        ).fetchone()[0])["lifecycle"]
        assert evidence["candidate_run_id"] == first_run.current_run_id
        assert evidence["head_sha"] == first
        assert kb.get_task(conn, task_id).status == "done"
        assert kb.get_lifecycle_state(conn, task_id)["acceptance"] == "accepted"


@pytest.mark.parametrize("flag", ["skip-worktree", "assume-unchanged"])
def test_role_checkout_refuses_index_hidden_changes(kanban_home, tmp_path, flag):
    repo = _make_repo(tmp_path)
    base = _head(repo)
    candidate = _commit(repo, "app.py", "candidate\n")
    with kbc.connect() as conn:
        implementation = _candidate_card(conn, repo, base=base, head=candidate)
        review = _role_card(conn, "review", implementation, repo, "reviewer")
        task = kb.get_task(conn, review)
    path = kbw.resolve_workspace(task, candidate_head=candidate)
    _git(path, "update-index", f"--{flag}", "app.py")
    (path / "app.py").write_text("hidden local changes\n", encoding="utf-8")
    with pytest.raises(ValueError, match="candidate_workspace_unavailable.*index flags.*app.py"):
        kbw.resolve_workspace(task, candidate_head=candidate)
    assert (path / "app.py").read_text() == "hidden local changes\n"
    _git(path, "update-index", f"--no-{flag}", "app.py")
    (path / "app.py").write_text("candidate\n", encoding="utf-8")
    assert kbw.resolve_workspace(task, candidate_head=candidate) == path
    assert _head(path) == candidate


def _ready_review_card(conn, repo: Path, base: str, candidate: str) -> str:
    implementation = kb.create_task(
        conn, title="implementation", assignee="coder", workspace_kind="worktree",
        workspace_path=str(repo),
        lifecycle_contract={
            "kind": "code", "review_mode": "separate_card", "reviewer": "reviewer",
            "validation_required": True,
        },
    )
    review = _role_card(conn, "review", implementation, repo, "reviewer")
    kb.link_tasks(conn, implementation, review, requirement="phase_finished")
    run = kb.claim_task(conn, implementation, claimer=kb._claimer_id())
    assert run is not None
    assert kb.complete_task(
        conn, implementation, expected_run_id=run.current_run_id, summary="candidate",
        metadata={"base_sha": base, "head_sha": candidate, "changed_files": []},
    )
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (review,))
    assert kb.evaluate_dependencies(conn, review)["satisfied"]
    return review


@pytest.mark.parametrize("surface", ["cli", "resolver"])
def test_stale_role_preparation_refuses_rebound_successor_checkout(
    kanban_home, tmp_path, monkeypatch, capsys, surface,
):
    from hermes_cli import kanban as kc
    from hermes_cli.kanban_db_receipts import rebind_candidate
    from hermes_cli.kanban_runtime import runtime_identity

    repo = _make_repo(tmp_path)
    base = _head(repo)
    first = _commit(repo, "app.py", "first\n")
    receipt = runtime_identity(Path(__file__).resolve().parents[2]).as_dict()
    with kbc.connect_closing() as conn:
        review = _ready_review_card(conn, repo, base, first)
        pending = kb.get_task(conn, review)
        implementation = pending.lifecycle_contract["candidate_task_id"]
        first_run = kb.get_task(conn, implementation).candidate_run_id
        second = _commit(repo, "app.py", "second\n")
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (implementation,))
        run = kb.claim_task(conn, implementation, claimer=kb._claimer_id())
        assert run is not None
        assert kb.complete_task(
            conn, implementation, expected_run_id=run.current_run_id,
            summary="second candidate",
            metadata={"base_sha": base, "head_sha": second, "changed_files": []})
        with kb.write_txn(conn):
            for run_id in (first_run, run.current_run_id):
                kb._append_event(conn, implementation, "spawned",
                                 {"runtime_identity": receipt}, run_id=run_id)
        rebind_candidate(
            conn, implementation, run_id=first_run, head_sha=first,
            reason="restore first", replace=True, replaced_run=run.current_run_id)
        assert kbw.candidate_head_for_task(conn, pending) == first
        target = kbw.resolve_workspace(pending, candidate_head=first)

    def rebind_and_claim():
        with kbc.connect_closing() as other:
            rebind_candidate(
                other, implementation, run_id=run.current_run_id, head_sha=second,
                reason="use second", replace=True, replaced_run=first_run)
        return kc._cmd_claim(_claim_args(review))

    if surface == "resolver":
        # A cached H1 before it acquired admission. B now owns this same task at H2.
        assert rebind_and_claim() == 0
        with pytest.raises(ValueError, match="candidate moved"):
            kbw.resolve_workspace_created(pending, candidate_head=first)
    else:
        resolve_candidate = kbw.candidate_head_for_task
        paused = False

        def read_then_rebind(conn, task):
            nonlocal paused
            head = resolve_candidate(conn, task)
            if not paused:
                paused = True
                with ThreadPoolExecutor(max_workers=1) as pool:
                    assert pool.submit(rebind_and_claim).result(timeout=30) != 0
            return head

        with monkeypatch.context() as patch:
            patch.setattr(kbw, "candidate_head_for_task", read_then_rebind)
            assert kc._cmd_claim(_claim_args(review)) != 0
        assert "candidate moved" in capsys.readouterr().err
        assert _head(target) == first  # No checkout mutation on the stale attempt.
        assert kc._cmd_claim(_claim_args(review)) == 0
    with kbc.connect_closing() as conn:
        successor = kb.get_task(conn, review)
        assert successor.status == "running" and successor.current_run_id
        assert successor.workspace_path == str(target)
    assert _head(target) == second
    assert (target / "app.py").read_text() == "second\n"


def test_cli_persistence_fences_reclaim_and_scratch_to_role_successor(
    kanban_home, tmp_path, monkeypatch,
):
    from hermes_cli import kanban as kc

    repo = _make_repo(tmp_path)
    base = _head(repo)
    candidate = _commit(repo, "app.py", "candidate\n")
    kb.write_board_metadata(None, default_workdir=str(repo))
    with kbc.connect_closing() as conn:
        review = _ready_review_card(conn, repo, base, candidate)
        contract = kb.get_task(conn, review).lifecycle_contract
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET lifecycle_contract = NULL, workspace_kind = 'scratch', "
                "workspace_path = NULL, branch_name = 'obsolete' WHERE id = ?", (review,))
    claim = kb.claim_task
    attempts = []
    old_runs = []

    def reclaim_and_take_over():
        with kbc.connect_closing() as other:
            assert kb.reclaim_task(other, review, reason="operator recovery", signal_fn=lambda *a: None)
            with kb.write_txn(other):
                other.execute("UPDATE tasks SET lifecycle_contract = ? WHERE id = ?",
                              (json.dumps(contract), review))
        return kc._cmd_claim(_claim_args(review))

    def claim_then_reclaim(conn, task_id, **kwargs):
        task = claim(conn, task_id, **kwargs)
        if task is not None and not old_runs:
            old_runs.append(task.current_run_id)
            with ThreadPoolExecutor(max_workers=1) as pool:
                attempts.append(pool.submit(reclaim_and_take_over).result(timeout=30))
        return task

    with monkeypatch.context() as patch:
        patch.setattr(kb, "claim_task", claim_then_reclaim)
        assert kc._cmd_claim(_claim_args(review)) == 0
    assert len(attempts) == 1 and attempts[0] != 0
    # The successor can enter only after A's path AND branch persistence have ended.
    assert kc._cmd_claim(_claim_args(review)) == 0
    target = repo / ".worktrees" / review
    with kbc.connect_closing() as conn:
        successor = kb.get_task(conn, review)
        assert successor.current_run_id != old_runs[0] and successor.status == "running"
        assert successor.workspace_kind == "worktree" and successor.workspace_path == str(target)
        assert successor.branch_name == ""
    assert _head(target) == candidate



@pytest.mark.parametrize("managed", [True, False])
def test_retained_scratch_role_dispatch_uses_only_managed_path_as_no_anchor(
    kanban_home, tmp_path, monkeypatch, managed,
):
    from hermes_cli import kanban_db_dispatch as kbd
    from hermes_cli.kanban_runtime import runtime_identity

    repo = _make_repo(tmp_path)
    base = _head(repo)
    candidate = _commit(repo, "candidate.txt", "candidate\n")
    kb.write_board_metadata(None, default_workdir=str(repo))
    identity = runtime_identity(Path(__file__).resolve().parents[2])
    granted = []
    monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: None)
    monkeypatch.setattr(kbd, "_require_autonomous_worker_capacity", lambda: None)
    monkeypatch.setattr(
        kbd, "_default_spawn",
        lambda *args, **kwargs: kbd.WorkerLaunch(
            identity.pid, identity.as_dict(), "retained-scratch",
            grant=lambda *args: granted.append(args)),
    )
    with kbc.connect() as conn:
        review = _ready_review_card(conn, repo, base, candidate)
        retained = (kb.workspaces_root() if managed else tmp_path / "explicit") / review
        retained.mkdir(parents=True)
        marker = retained / "retained.txt"
        marker.write_text("keep previous attempt\n")
        kbw.set_workspace_path(conn, review, retained, kind="scratch")
        task = kb.get_task(conn, review)
        target = repo / ".worktrees" / review
        if managed:
            assert kbw.resolve_workspace(task, candidate_head=candidate) == target
            assert _head(target) == candidate
        else:
            with pytest.raises(ValueError, match="candidate_workspace_unavailable"):
                kbw.resolve_workspace(task, candidate_head=candidate)
        # Resolution is not a migration: only the successful claim persists the kind.
        assert kb.get_task(conn, review) == task
        assert task.workspace_kind == "scratch" and task.workspace_path == str(retained)
        result = kbd.DispatchResult()
        accepted = kbd._dispatch_lane_task(
            conn, {"id": review}, "reviewer", result, lane="ready", dry_run=False,
            ttl_seconds=None, board=None, failure_limit=3, spawn_fn=None,
            per_profile_cap=None, per_profile_running={},
        )
        current = kb.get_task(conn, review)
        assert accepted is managed
        assert bool(granted) is managed
        if managed:
            assert result.spawned == [(review, "reviewer", str(target))]
            assert current.workspace_kind == "worktree"
            assert current.workspace_path == str(target)
            assert current.status == "running" and current.current_run_id
        else:
            assert not result.spawned and not target.exists()
            assert current.current_run_id is None and current.claim_lock is None
            assert current.workspace_path == str(retained)
        assert marker.read_text() == "keep previous attempt\n"


@pytest.mark.parametrize("refusal", ["claim", "candidate"])
@pytest.mark.parametrize("reused", [False, True])
def test_cli_refusal_discards_only_new_worktrees(
    kanban_home, tmp_path, monkeypatch, capsys, refusal, reused,
):
    from hermes_cli import kanban as kc

    repo = _make_repo(tmp_path)
    base = _head(repo)
    candidate = _commit(repo, "app.py", "candidate\n")
    with kbc.connect() as conn:
        review = _ready_review_card(conn, repo, base, candidate)
        if reused:
            kbw.resolve_workspace(kb.get_task(conn, review), candidate_head=candidate)
    if refusal == "claim":
        monkeypatch.setattr(kb, "claim_task", lambda *a, **kw: None)
    else:
        heads = iter((candidate, candidate, base))
        monkeypatch.setattr(kbw, "candidate_head_for_task", lambda *a: next(heads))
    assert kc._cmd_claim(_claim_args(review)) != 0
    assert ("the task changed" if refusal == "claim" else "candidate moved") in capsys.readouterr().err
    assert (repo / ".worktrees" / review).exists() is reused
    with kbc.connect() as conn:
        assert kb.get_task(conn, review).current_run_id is None



def test_cli_claim_releases_a_checkout_moved_by_another_board(
    kanban_home, tmp_path, monkeypatch, capsys,
):
    from hermes_cli import kanban as kc

    repo = _make_repo(tmp_path)
    base = _head(repo)
    candidate = _commit(repo, "app.py", "first\n")
    with kbc.connect_closing() as conn:
        review = _ready_review_card(conn, repo, base, candidate)
    other_source = _add_worktree(repo, tmp_path / "other-source", "other-source")
    other_head = _commit(other_source, "app.py", "second\n")
    kb.create_board("other")
    with kbc.connect_closing(board="other") as other, monkeypatch.context() as patch:
        ids = iter(("t_11111111", review))
        patch.setattr(kb, "_new_task_id", lambda: next(ids))
        assert _ready_review_card(other, other_source, base, other_head) == review
        kbw.set_workspace_path(other, review, str(repo))

    resolve = kbw.resolve_workspace_created

    def prepare_then_move(task, **kwargs):
        prepared = resolve(task, **kwargs)
        assert _head(prepared[0]) == candidate and prepared[2]
        with kbc.connect_closing(board="other") as other:
            rival = kb.get_task(other, review)
            moved = resolve(rival, board="other", candidate_head=other_head)
            assert moved[0] == prepared[0] and not moved[2]
        return prepared

    monkeypatch.setattr(kbw, "resolve_workspace_created", prepare_then_move)
    assert kc._cmd_claim(_claim_args(review)) != 0
    error = capsys.readouterr().err
    assert "candidate_workspace_unavailable" in error and "the claim was released" in error
    assert candidate in error and other_head in error
    with kbc.connect_closing() as conn:
        after = kb.get_task(conn, review)
        assert after.current_run_id is None and after.claim_lock is None
        assert after.status != "running" and after.workspace_path == str(repo)
        assert kb.latest_run(conn, review).outcome == "spawn_failed"
    target = repo / ".worktrees" / review
    assert target.is_dir() and _head(target) == other_head
    assert (target / "app.py").read_text() == "second\n"

@pytest.mark.parametrize("contend_at", ["entry", "rollback"])
def test_cli_candidate_rollback_reuses_preparation_admission(
    kanban_home, tmp_path, monkeypatch, capsys, contend_at,
):
    from hermes_cli import kanban as kc

    repo = _make_repo(tmp_path)
    base = _head(repo)
    candidate = _commit(repo, "app.py", "candidate\n")
    with kbc.connect_closing() as conn:
        review = _ready_review_card(conn, repo, base, candidate)
    claim = kb.claim_task
    runs = []
    lock_path = kb.kanban_home() / "kanban" / ".dispatch-admission.lock"
    with lock_path.open("a+b") as holder:
        def claim_then_contend(conn, task_id, **kwargs):
            task = claim(conn, task_id, **kwargs)
            assert task is not None
            runs.append(task.current_run_id)
            # A competing file-lock owner can no longer open a rollback gap.
            assert not kbc._try_lock_nb(holder)
            return task

        monkeypatch.setattr(kb, "claim_task", claim_then_contend)
        heads = iter((candidate, candidate, base))
        monkeypatch.setattr(kbw, "candidate_head_for_task", lambda *args: next(heads))
        if contend_at == "entry":
            assert kbc._try_lock_nb(holder)
        try:
            assert kc._cmd_claim(_claim_args(review)) != 0
        finally:
            kbc._unlock(holder)
        assert kbc._try_lock_nb(holder)  # Every refusal has left the fence.
        kbc._unlock(holder)
    error = capsys.readouterr().err
    path = repo / ".worktrees" / review
    with kbc.connect_closing() as conn:
        task = kb.get_task(conn, review)
        assert task.current_run_id is None and task.claim_lock is None
        assert not path.exists()
        if contend_at == "entry":
            assert not runs and "dispatch admission unavailable" in error
        else:
            assert "the claim was released" in error
            assert kb.latest_run(conn, review).outcome == "spawn_failed"


@pytest.mark.parametrize("takeover", ["claim", "row"])
def test_cli_candidate_rollback_preserves_a_successor_claim(
    kanban_home, tmp_path, monkeypatch, takeover,
):
    from hermes_cli import kanban as kc
    from hermes_cli.kanban_db_dispatch import _record_task_failure

    repo = _make_repo(tmp_path)
    base = _head(repo)
    candidate = _commit(repo, "app.py", "candidate\n")
    with kbc.connect() as conn:
        review = _ready_review_card(conn, repo, base, candidate)
    claim = kb.claim_task
    successors = []

    def replace_claim(conn, task_id, **kwargs):
        original = claim(conn, task_id, **kwargs)
        assert original is not None
        if takeover == "row":
            with kb.write_txn(conn):
                conn.execute(
                    "UPDATE tasks SET current_run_id = ?, claim_lock = 'successor' WHERE id = ?",
                    (original.current_run_id + 1000, task_id))
            successor = kb.get_task(conn, task_id)
        else:
            _record_task_failure(
                conn, task_id, "first attempt ended", outcome="spawn_failed",
                release_claim=True, end_run=True,
            )
            successor = claim(conn, task_id, claimer="successor")
            assert successor is not None
        successors.append(successor)
        return original

    monkeypatch.setattr(kb, "claim_task", replace_claim)
    heads = iter((candidate, candidate, base))
    monkeypatch.setattr(kbw, "candidate_head_for_task", lambda *a: next(heads))
    assert kc._cmd_claim(_claim_args(review)) != 0
    path = repo / ".worktrees" / review
    assert path.is_dir()
    assert _head(path) == candidate
    with kbc.connect() as conn:
        current = kb.get_task(conn, review)
        assert current.status == "running"
        assert current.current_run_id == successors[0].current_run_id
        assert current.claim_lock == successors[0].claim_lock
        assert current.consecutive_failures == successors[0].consecutive_failures


@pytest.mark.parametrize("refusal", ["claim", "candidate", "error", "interrupted"])
@pytest.mark.parametrize("reused", [False, True])
def test_deferred_dispatch_discards_only_new_worktrees(
    kanban_home, tmp_path, monkeypatch, refusal, reused,
):
    from hermes_cli import kanban_db_dispatch as kbd
    from hermes_cli.kanban_runtime import runtime_identity

    repo = _make_repo(tmp_path)
    base = _head(repo)
    candidate = _commit(repo, "app.py", "candidate\n")
    identity = runtime_identity(Path(__file__).resolve().parents[2])
    cancelled = []
    monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: None)
    monkeypatch.setattr(kbd, "_require_autonomous_worker_capacity", lambda: None)
    with kbc.connect() as conn:
        review = _ready_review_card(conn, repo, base, candidate)
        if reused:
            kbw.resolve_workspace(kb.get_task(conn, review), candidate_head=candidate)
        path = repo / ".worktrees" / review

        def prepare(task, workspace, **kwargs):
            assert kwargs["defer_grant"]
            assert Path(workspace) == path
            assert _head(path) == candidate
            if refusal == "error":
                raise RuntimeError("preparation failed")
            if refusal == "interrupted":
                raise kbd.WorkerLaunchInterrupted("drain during preparation")
            if refusal == "claim":
                kb.add_comment(
                    conn, review, author="worker",
                    body="Published https://github.com/example/repo/pull/7",
                )
            return kbd.WorkerLaunch(
                identity.pid, identity.as_dict(), "worktree-refusal",
                cancel=lambda: cancelled.append(True),
            )

        monkeypatch.setattr(kbd, "_default_spawn", prepare)
        if refusal == "candidate":
            heads = iter((candidate, candidate, base))
            monkeypatch.setattr(kbw, "candidate_head_for_task", lambda *a: next(heads))
        result = kbd.DispatchResult()
        assert not kbd._dispatch_lane_task(
            conn, {"id": review}, "reviewer", result, lane="ready", dry_run=False,
            ttl_seconds=None, board=None, failure_limit=3, spawn_fn=None,
            per_profile_cap=None, per_profile_running={},
        )
        assert path.exists() is reused
        assert not result.spawned
        assert kb.get_task(conn, review).current_run_id is None
        if refusal in {"claim", "candidate"}:
            assert cancelled == [True]
        if refusal == "claim":
            assert result.respawn_guarded == [(review, "active_pr")]
        elif refusal == "candidate":
            assert kb.get_task(conn, review).last_failure_error.startswith("candidate moved")
        elif refusal == "interrupted":
            assert result.cancelled == [review]
        else:
            assert kb.get_task(conn, review).last_failure_error == "preparation failed"


@pytest.mark.parametrize("reused", [False, True])
@pytest.mark.parametrize("entered", [False, True])
@pytest.mark.parametrize("interrupted", [False, True])
def test_custom_spawn_discard_stops_at_callback_entry(
    kanban_home, tmp_path, monkeypatch, reused, entered, interrupted,
):
    from hermes_cli import kanban_db_dispatch as kbd

    monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: lambda name: name == "reviewer")

    repo = _make_repo(tmp_path)
    base = _head(repo)
    candidate = _commit(repo, "app.py", "candidate\n")
    calls = []
    commits = []
    failure = kbd.WorkerLaunchInterrupted if interrupted else RuntimeError
    with kbc.connect_closing() as conn:
        review = _ready_review_card(conn, repo, base, candidate)
        if reused:
            kbw.resolve_workspace(kb.get_task(conn, review), candidate_head=candidate)
        path = repo / ".worktrees" / review

        def exploding_spawn(task, workspace, **kwargs):
            calls.append(task.id)
            assert Path(workspace) == path and _head(path) == candidate
            # U is no longer HEAD, but ORIG_HEAD and the private reflog retain it.
            commits.append(_commit(path, "callback.txt", "unpublished\n"))
            _git(path, "reset", "--hard", candidate)
            raise failure("custom launcher refused")

        if not entered:
            def fail_before_callback(*args, **kwargs):
                assert path.is_dir()
                raise failure("custom launcher refused")

            monkeypatch.setattr(kbw, "set_branch_name", fail_before_callback)

        result = kbd.DispatchResult()
        assert not kbd._dispatch_lane_task(
            conn, {"id": review}, "reviewer", result, lane="ready", dry_run=False,
            ttl_seconds=None, board=None, failure_limit=3, spawn_fn=exploding_spawn,
            per_profile_cap=None, per_profile_running={},
        )
        assert calls == ([review] if entered else []) and not result.spawned
        task = kb.get_task(conn, review)
        assert task.current_run_id is None and task.claim_lock is None
        if interrupted:
            assert result.interrupted == [review]
        else:
            assert task.last_failure_error == "custom launcher refused"
    assert path.exists() is (entered or reused)
    if entered or reused:
        assert _head(path) == candidate
    if entered:
        assert subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "ORIG_HEAD"], text=True).strip() == commits[0]
        reachable = subprocess.check_output(
            ["git", "-C", str(path), "rev-list", "--all", "--reflog"], text=True).splitlines()
        assert commits[0] in reachable


@pytest.mark.parametrize("lane", ["ready", "review"])
def test_deferred_candidate_rollback_preserves_operator_reclaim(
    kanban_home, tmp_path, monkeypatch, lane,
):
    from hermes_cli import kanban_db_dispatch as kbd
    from hermes_cli.kanban_runtime import runtime_identity

    repo = _make_repo(tmp_path)
    base = _head(repo)
    candidate = _commit(repo, "app.py", "candidate\n")
    identity = runtime_identity(Path(__file__).resolve().parents[2])
    monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: None)
    monkeypatch.setattr(kbd, "_require_autonomous_worker_capacity", lambda: None)
    with kbc.connect() as conn:
        review = _ready_review_card(conn, repo, base, candidate)
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status = ? WHERE id = ?", (lane, review))
        path = repo / ".worktrees" / review
        reclaimed = {}

        def cancel_and_reclaim():
            # Reclaim does not need dispatch admission, and closes exactly this run.
            with kbc.connect_closing() as other:
                assert kb.reclaim_task(other, review, reason="operator recovery", signal_fn=lambda *a: None)
                reclaimed["task"] = kb.get_task(other, review)
                reclaimed["runs"] = kb.list_runs(other, review)
                reclaimed["events"] = kb.list_events(other, review)

        def prepare(task, workspace, **kwargs):
            assert Path(workspace) == path and path.is_dir()
            return kbd.WorkerLaunch(
                identity.pid, identity.as_dict(), "reclaimed-candidate",
                cancel=cancel_and_reclaim,
            )

        monkeypatch.setattr(kbd, "_default_spawn", prepare)
        heads = iter((candidate, candidate, base))
        monkeypatch.setattr(kbw, "candidate_head_for_task", lambda *a: next(heads))
        result = kbd.DispatchResult()
        assert not kbd._dispatch_lane_task(
            conn, {"id": review}, "reviewer", result, lane=lane, dry_run=False,
            ttl_seconds=None, board=None, failure_limit=1, spawn_fn=None,
            per_profile_cap=None, per_profile_running={},
        )
        assert not path.exists()
        assert not result.spawned and not result.auto_blocked
        current = kb.get_task(conn, review)
        assert current == reclaimed["task"]
        assert current.status == lane and current.current_run_id is None
        assert current.consecutive_failures == 0 and current.last_failure_error is None
        assert kb.list_runs(conn, review) == reclaimed["runs"]
        assert kb.list_events(conn, review) == reclaimed["events"]
        assert not any(event.kind in {"gave_up", "blocked"} for event in reclaimed["events"])


@pytest.mark.parametrize(
    "interleaving", ["after-claim", "grant-mismatch", "grant-missing", "implementation-grant",
                     "grant-match", "no-grant"],
)
def test_deferred_dispatch_fences_a_rebound_base_with_the_same_head(
    kanban_home, tmp_path, monkeypatch, interleaving,
):
    from types import SimpleNamespace
    from hermes_cli import kanban_db_dispatch as kbd
    from hermes_cli.kanban_runtime import runtime_identity
    from hermes_cli.worker_execution_policy import resolve_review_context

    repo = _make_repo(tmp_path)
    base = _head(repo)
    other_base = _commit(repo, "middle.txt", "middle\n")
    candidate = _commit(repo, "candidate.txt", "candidate\n")
    identity = runtime_identity(Path(__file__).resolve().parents[2])
    cancelled, granted = [], []
    monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: None)
    monkeypatch.setattr(kbd, "_require_autonomous_worker_capacity", lambda: None)
    with kbc.connect() as conn:
        review = _ready_review_card(conn, repo, base, candidate)
        implementation = kb.get_task(conn, review).lifecycle_contract["candidate_task_id"]
        first_run = kb.get_task(conn, implementation).candidate_run_id
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (implementation,))
        second_run = kb.claim_task(conn, implementation)
        assert second_run is not None
        assert kb.complete_task(
            conn, implementation, expected_run_id=second_run.current_run_id,
            summary="same head, different review base",
            metadata={"base_sha": other_base, "head_sha": candidate, "changed_files": []})
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET candidate_run_id = ? WHERE id = ?", (first_run, implementation))

        def prepare(task, workspace, **kwargs):
            assert kwargs["defer_grant"]
            if interleaving == "grant-mismatch":
                with kb.write_txn(conn):
                    conn.execute(
                        "UPDATE tasks SET candidate_run_id = ? WHERE id = ?",
                        (second_run.current_run_id, implementation))
            role, pair = resolve_review_context(str(kb.kanban_db_path()), review)
            captured_base = other_base if interleaving == "grant-mismatch" else base
            assert role == "review" and pair == {"base": captured_base, "head": candidate}
            # Restore A before the claim: both board reads see A, but the grant
            # retained B. Only the grant comparison can detect this interleaving.
            with kb.write_txn(conn):
                conn.execute(
                    "UPDATE tasks SET candidate_run_id = ? WHERE id = ?", (first_run, implementation))
            execution = None
            if interleaving not in {"no-grant", "after-claim"}:
                execution = SimpleNamespace(
                    role="implementation" if interleaving == "implementation-grant" else role,
                    review=None if interleaving in {"grant-missing", "implementation-grant"} else pair,
                )
            return kbd.WorkerLaunch(
                identity.pid, identity.as_dict(), "base-rebind",
                cancel=lambda: cancelled.append(True),
                grant=lambda *args: granted.append(args),
                execution_grant=execution,
            )

        claim = kb.claim_task

        def rebind_after_claim(connection, task_id, **kwargs):
            claimed = claim(connection, task_id, **kwargs)
            assert claimed is not None
            with kb.write_txn(connection):
                connection.execute(
                    "UPDATE tasks SET candidate_run_id = ? WHERE id = ?",
                    (second_run.current_run_id, implementation))
            return claimed

        monkeypatch.setattr(kbd, "_default_spawn", prepare)
        if interleaving == "after-claim":
            monkeypatch.setattr(kb, "claim_task", rebind_after_claim)
        result = kbd.DispatchResult()
        accepted = kbd._dispatch_lane_task(
            conn, {"id": review}, "reviewer", result, lane="ready", dry_run=False,
            ttl_seconds=None, board=None, failure_limit=3, spawn_fn=None,
            per_profile_cap=None, per_profile_running={},
        )
        current = kb.get_task(conn, review)
        if interleaving in {"grant-match", "no-grant"}:
            assert accepted and result.spawned and granted and not cancelled
            assert current.current_run_id == granted[0][0] and current.claim_lock
            return
        assert not accepted
        assert current.current_run_id is None and current.claim_lock is None
        if interleaving == "after-claim":
            assert current.last_failure_error.startswith("candidate moved")
            assert base in current.last_failure_error and other_base in current.last_failure_error
        else:
            run = conn.execute(
                "SELECT id FROM task_runs WHERE task_id = ? ORDER BY id DESC LIMIT 1", (review,),
            ).fetchone()[0]
            captured = f"{other_base}..{candidate}" if interleaving == "grant-mismatch" else "<missing>..<missing>"
            assert current.last_failure_error == (
                f"run {run} execution grant range {captured} "
                f"differs from accepted candidate {base}..{candidate}")
            assert kb.get_task(conn, implementation).candidate_run_id == first_run
        assert cancelled == [True] and granted == [] and not result.spawned
        assert not (repo / ".worktrees" / review).exists()


@pytest.mark.parametrize("rebind_at", ["after-grant", "during-completion", "same-pair", "unchanged"])
def test_review_completion_is_fenced_to_granted_candidate(
    kanban_home, tmp_path, monkeypatch, rebind_at,
):
    from types import SimpleNamespace
    from hermes_cli import kanban_db_dispatch as kbd
    from hermes_cli import kanban_pr_acceptance_store
    from hermes_cli.kanban_db_receipts import rebind_candidate
    from hermes_cli.kanban_lifecycle import LifecycleEvidenceError
    from hermes_cli.kanban_runtime import runtime_identity

    repo = _make_repo(tmp_path)
    base = _head(repo)
    middle = _commit(repo, "middle.txt", "middle\n")
    head = _commit(repo, "candidate.txt", "candidate\n")
    other_base = base if rebind_at == "same-pair" else middle
    identity = runtime_identity(Path(__file__).resolve().parents[2])
    granted = []
    monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: None)
    monkeypatch.setattr(kbd, "_require_autonomous_worker_capacity", lambda: None)
    with kbc.connect() as conn:
        review = _ready_review_card(conn, repo, base, head)
        implementation = kb.get_task(conn, review).lifecycle_contract["candidate_task_id"]
        first_run = kb.get_task(conn, implementation).candidate_run_id
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (implementation,))
        second = kb.claim_task(conn, implementation)
        assert kb.complete_task(
            conn, implementation, expected_run_id=second.current_run_id,
            summary="legacy candidate with the same head",
            metadata={"base_sha": other_base, "head_sha": head, "changed_files": []})
        with kb.write_txn(conn):
            for run_id in (first_run, second.current_run_id):
                kb._append_event(conn, implementation, "spawned",
                                 {"runtime_identity": identity.as_dict()}, run_id=run_id)
            legacy = json.loads(conn.execute(
                "SELECT metadata FROM task_runs WHERE id = ?", (second.current_run_id,),
            ).fetchone()[0])
            legacy.pop("lifecycle")
            conn.execute("UPDATE task_runs SET metadata = ? WHERE id = ?",
                         (json.dumps(legacy), second.current_run_id))
        rebind_candidate(
            conn, implementation, run_id=first_run, head_sha=head, reason="review A",
            replace=True, replaced_run=second.current_run_id)

        monkeypatch.setattr(kbd, "_default_spawn", lambda *args, **kwargs: kbd.WorkerLaunch(
            identity.pid, identity.as_dict(), "completion-pair",
            grant=lambda *args: granted.append(args),
            execution_grant=SimpleNamespace(role="review", review={"base": base, "head": head}),
        ))
        result = kbd.DispatchResult()
        assert kbd._dispatch_lane_task(
            conn, {"id": review}, "reviewer", result, lane="ready", dry_run=False,
            ttl_seconds=None, board=None, failure_limit=3, spawn_fn=None,
            per_profile_cap=None, per_profile_running={})
        review_run = granted[0][0]
        retained = json.loads(conn.execute(
            "SELECT metadata FROM task_runs WHERE id = ?", (review_run,),
        ).fetchone()[0])["review_candidate"]
        assert retained == {"task_id": implementation, "run_id": first_run, "base": base, "head": head}

        def rebind():
            rebind_candidate(
                conn, implementation, run_id=second.current_run_id, head_sha=head,
                reason="operator replacement after grant", replace=True, replaced_run=first_run)

        if rebind_at in {"after-grant", "same-pair"}:
            rebind()
        elif rebind_at == "during-completion":
            def prepare_acceptance(*args):
                rebind()
                return None
            monkeypatch.setattr(kanban_pr_acceptance_store, "prepare_acceptance", prepare_acceptance)

        def complete():
            return kb.complete_task(
                conn, review, expected_run_id=review_run,
                summary="reviewed granted range", verdict="APPROVE", metadata={"head_sha": head})

        if rebind_at != "unchanged":
            with pytest.raises(LifecycleEvidenceError) as error:
                complete()
            assert f"{base}..{head}" in str(error.value)
            assert f"{other_base}..{head}" in str(error.value)
            current = kb.get_task(conn, review)
            assert current.status == "running" and current.current_run_id == review_run
            assert kb.get_lifecycle_state(conn, implementation)["acceptance"] != "accepted"
            assert not any(event.kind == "review_approved" for event in kb.list_events(conn, review))
            run = next(run for run in kb.list_runs(conn, review) if run.id == review_run)
            assert run.ended_at is None and run.outcome is None
        else:
            assert complete()
            assert kb.get_task(conn, review).status == "done"
            evidence = json.loads(conn.execute(
                "SELECT metadata FROM task_runs WHERE id = ?", (review_run,),
            ).fetchone()[0])["lifecycle"]
            assert evidence["candidate_run_id"] == first_run
            assert evidence["head_sha"] == head and evidence["verdict"] == "APPROVE"


@pytest.mark.parametrize("surface", ["cli", "deferred"])
@pytest.mark.parametrize("target_kind", ["custom", "inherited"])
@pytest.mark.parametrize("reused", [False, True])
def test_ordinary_claim_tracks_creation_at_the_resolved_target(
    kanban_home, tmp_path, monkeypatch, capsys, surface, target_kind, reused,
):
    from hermes_cli import kanban as kc
    from hermes_cli import kanban_db_dispatch as kbd
    from hermes_cli.kanban_runtime import runtime_identity

    repo = _make_repo(tmp_path)
    requested = repo / "custom-checkout"
    if target_kind == "inherited":
        requested = _add_worktree(repo, repo / ".worktrees" / "sibling", "wt/sibling")
    with kbc.connect() as conn:
        task_id = kb.create_task(
            conn, title="ordinary custom worktree", assignee="worker",
            workspace_kind="worktree", workspace_path=str(requested))
        task = kb.get_task(conn, task_id)
        target = requested if target_kind == "custom" else repo / ".worktrees" / task_id
        if reused:
            _add_worktree(repo, target, f"wt/{task_id}")
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (task_id,))
    resolved = []
    resolve = kbw.resolve_workspace_created

    def track_creation(task, **kwargs):
        result = resolve(task, **kwargs)
        resolved.append(result)
        return result

    monkeypatch.setattr(kbw, "resolve_workspace_created", track_creation)
    if surface == "cli":
        assert kc._cmd_claim(_claim_args(task_id)) == 0, capsys.readouterr().err
    else:
        identity = runtime_identity(Path(__file__).resolve().parents[2])
        monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: None)
        monkeypatch.setattr(kbd, "_require_autonomous_worker_capacity", lambda: None)
        monkeypatch.setattr(
            kbd, "_default_spawn",
            lambda *args, **kwargs: kbd.WorkerLaunch(
                identity.pid, identity.as_dict(), "ordinary-target"))
        with kbc.connect() as conn:
            result = kbd.DispatchResult()
            assert kbd._dispatch_lane_task(
                conn, {"id": task_id}, "worker", result, lane="ready", dry_run=False,
                ttl_seconds=None, board=None, failure_limit=3, spawn_fn=None,
                per_profile_cap=None, per_profile_running={},
            )
            assert result.spawned == [(task_id, "worker", str(target))]
    assert resolved == [(target, f"wt/{task_id}", not reused)]
    with kbc.connect() as conn:
        claimed = kb.get_task(conn, task_id)
        assert claimed.status == "running"
        assert claimed.workspace_path == str(target)
        assert claimed.branch_name == f"wt/{task_id}"
    assert _head(target) == _head(repo)
    assert resolve(task) == (target, f"wt/{task_id}", False)
    if target_kind == "inherited":
        assert (requested / "README.md").read_text() == "base\n"
        assert kbw._git_current_branch(requested) == "wt/sibling"


def test_candidate_rollback_fences_failure_after_takeover_inside_admission(
    kanban_home, tmp_path, monkeypatch,
):
    from hermes_cli import kanban as kc
    from hermes_cli import kanban_db_dispatch as kbd

    repo = _make_repo(tmp_path)
    base = _head(repo)
    candidate = _commit(repo, "app.py", "candidate\n")
    with kbc.connect() as conn:
        review = _ready_review_card(conn, repo, base, candidate)
    discard = kbw.discard_prepared_worktree
    record_failure = kbd._record_task_failure
    successor = {}
    results = []

    def discard_then_take_over(*args, **kwargs):
        with kbc._dispatch_admission_lock() as admitted:
            assert not admitted  # The rollback holds admission across the discard.
        discard(*args, **kwargs)
        with kbc.connect() as other:
            original = kb.get_task(other, review)
            with kb.write_txn(other):
                run_id = other.execute(
                    "INSERT INTO task_runs (task_id, status, claim_lock, started_at) "
                    "VALUES (?, 'running', 'successor', 1)", (review,)).lastrowid
                other.execute(
                    "UPDATE tasks SET current_run_id = ?, claim_lock = 'successor' WHERE id = ?",
                    (run_id, review))
            successor["original_run"] = original.current_run_id
            successor["task"] = kb.get_task(other, review)
            successor["runs"] = [tuple(row) for row in other.execute(
                "SELECT * FROM task_runs WHERE task_id = ? ORDER BY id", (review,))]
            successor["events"] = [tuple(row) for row in other.execute(
                "SELECT * FROM task_events WHERE task_id = ? ORDER BY id", (review,))]

    def record_fenced_failure(*args, **kwargs):
        assert kwargs["expected_run_id"] == successor["original_run"]
        with kbc._dispatch_admission_lock() as admitted:
            assert not admitted
        result = record_failure(*args, **kwargs)
        results.append(result)
        return result

    monkeypatch.setattr(kbw, "discard_prepared_worktree", discard_then_take_over)
    monkeypatch.setattr(kbd, "_record_task_failure", record_fenced_failure)
    heads = iter((candidate, candidate, base))
    monkeypatch.setattr(kbw, "candidate_head_for_task", lambda *args: next(heads))
    assert kc._cmd_claim(_claim_args(review)) != 0
    assert results == [False]
    with kbc.connect() as conn:
        assert kb.get_task(conn, review) == successor["task"]
        assert [tuple(row) for row in conn.execute(
            "SELECT * FROM task_runs WHERE task_id = ? ORDER BY id", (review,))] == successor["runs"]
        assert [tuple(row) for row in conn.execute(
            "SELECT * FROM task_events WHERE task_id = ? ORDER BY id", (review,))] == successor["events"]


@pytest.mark.parametrize("target_kind", ["role", "anchor", "custom", "inherited"])
def test_worktree_creation_is_reported_at_the_reuse_decision(
    kanban_home, tmp_path, monkeypatch, target_kind,
):
    repo = _make_repo(tmp_path)
    base = _head(repo)
    candidate = _commit(repo, "app.py", "candidate\n")
    with kbc.connect() as conn:
        if target_kind == "role":
            task_id = _ready_review_card(conn, repo, base, candidate)
        else:
            requested = repo
            if target_kind == "custom":
                requested = repo / "custom-checkout"
            elif target_kind == "inherited":
                requested = _add_worktree(repo, repo / ".worktrees" / "sibling", "wt/sibling")
            task_id = kb.create_task(
                conn, title="ordinary worktree", assignee="worker",
                workspace_kind="worktree", workspace_path=str(requested))
        task = kb.get_task(conn, task_id)
    target = repo / "custom-checkout" if target_kind == "custom" else repo / ".worktrees" / task_id
    assert not target.exists()
    if target_kind == "role":
        resolve = kbw._role_worktree

        def create_before_role_reuse(*args):
            _git(repo, "worktree", "add", "--detach", str(target), candidate)
            return resolve(*args)

        monkeypatch.setattr(kbw, "_role_worktree", create_before_role_reuse)
    else:
        resolve = kbw._ensure_git_worktree

        def create_before_reuse(repo_root, path, branch):
            _add_worktree(repo_root, path, branch)
            return resolve(repo_root, path, branch)

        monkeypatch.setattr(kbw, "_ensure_git_worktree", create_before_reuse)
    path, branch, created = kbw.resolve_workspace_created(task, candidate_head=candidate)
    assert path == target
    assert branch == ("" if target_kind == "role" else f"wt/{task_id}")
    assert created is False
    assert _head(target) == candidate


@pytest.mark.parametrize("owner", ["other-live", "other-terminal", "own"])
def test_role_preparation_preserves_another_boards_registered_checkout(
    kanban_home, tmp_path, owner,
):
    repo = _make_repo(tmp_path)
    base = _head(repo)
    candidate = _commit(repo, "app.py", "candidate\n")
    with kbc.connect_closing() as conn:
        review = _ready_review_card(conn, repo, base, candidate)
        task = kb.get_task(conn, review)
    target = repo / ".worktrees" / review
    _git(repo, "worktree", "add", "--detach", str(target), base)
    board = "default"
    if owner != "own":
        kb.create_board("other")
        board = "other"
        with kbc.connect_closing(board=board) as conn:
            other = kb.create_task(
                conn, title="other board's checkout", assignee="reviewer",
                workspace_kind="worktree", workspace_path=str(target),
            )
            with kb.write_txn(conn):
                conn.execute("UPDATE tasks SET id = ? WHERE id = ?", (review, other))
            assert kb.claim_task(conn, review, claimer="reviewer:other") is not None
            if owner == "other-terminal":
                assert kb.complete_task(conn, review, summary="finished")
    else:
        with kbc.connect_closing() as conn:
            kbw.set_workspace_path(conn, review, str(target))

    if owner == "other-live":
        with pytest.raises(ValueError, match="candidate_workspace_unavailable.*checkout collision") as error:
            kbw.resolve_workspace(task, board="default", candidate_head=candidate)
        assert str(target) in str(error.value)
        assert "another non-terminal task" in str(error.value)
        assert _head(target) == base
        with kbc.connect_closing(board="other") as conn:
            assert kb.get_task(conn, review).status == "running"
    else:
        assert kbw.resolve_workspace(task, board="default", candidate_head=candidate) == target
        assert _head(target) == candidate

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

    assert kc._cmd_claim(_claim_args(review)) != 0
    captured = capsys.readouterr()
    assert "candidate_workspace_unavailable" in captured.out + captured.err
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

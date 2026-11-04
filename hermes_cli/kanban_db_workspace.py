"""Task workspace lifecycle: scratch/dir/worktree resolution (incl. git worktree creation), post-completion cleanup with containment guards, worker tmux teardown and the first-use scratch-workspace tip.

Split out of ``hermes_cli.kanban_db``; origin-resident helpers are reached
late-bound via ``_kb`` (import-cycle breaking) so monkeypatching
``kanban_db.<name>`` keeps working.
"""

from __future__ import annotations

import os
import re
import shutil
import sqlite3
import subprocess
import time
import stat
import unicodedata
import bisect
from pathlib import Path
from typing import Optional
from typing import TYPE_CHECKING
import contextlib

from hermes_cli.worktree_ops import release_lsp_clients

if TYPE_CHECKING:
    from hermes_cli.kanban_db import Task

_REMOVABLE_KINDS = ("scratch", "worktree")

# Typed role cards (review / validation) work from a candidate commit, never from the
# repository's current HEAD, and land in their own task-owned worktree.
_ROLE_KINDS = frozenset({"review", "validation"})
_COMMIT_ID = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_CANDIDATE_WORKSPACE_UNAVAILABLE = "candidate_workspace_unavailable"


def _path_key(path: Path | str | None) -> str:
    """Unicode-form-insensitive identity for a filesystem path.

    macOS hands back DECOMPOSED path strings (NFD: ``o`` + U+0308) for names the
    user typed in composed form (NFC: ``ö``) — a OneDrive/FileProvider path like
    ``OneDrive-Persönlich`` round-trips through ``git rev-parse --show-toplevel``
    as NFD while the DB row holds NFC. Raw ``Path`` equality then reports a real
    repo root as "not a repo" purely on Unicode form, so every path identity
    check here goes through this key.
    """
    return unicodedata.normalize("NFC", str(path)) if path is not None else ""

# Statuses after which a task no longer needs a workspace: a child stops
# needing its parent's handoff artifacts, and a sharer stops holding a dir.
_TERMINAL_STATUSES_SQL = "('done', 'archived', 'failed', 'cancelled')"

_ACTIVE_CHILDREN_SQL = (
    "SELECT 1 FROM task_links l "
    "JOIN tasks t ON t.id = l.child_id "
    f"WHERE l.parent_id = ? AND t.status NOT IN {_TERMINAL_STATUSES_SQL} "
    "LIMIT 1"
)

_WORKSPACE_ROW_SQL = "SELECT status, workspace_kind, workspace_path, branch_name FROM tasks WHERE id = ?"


def _git(repo_root: Path, *args: str, timeout: int) -> subprocess.CompletedProcess:
    """``git -C repo_root args``; never raises on a non-zero exit.

    :func:`noninteractive_repo_git_env` (GHSA-7x36-8jrh-v4pw): the dispatcher runs ``worktree add``
    unattended, which executes the repo's hooks, ``core.fsmonitor`` and smudge filters.
    """
    from hermes_cli._subprocess_compat import FILTER_DISCOVERY_FAILED, noninteractive_repo_git_env
    env = noninteractive_repo_git_env(repo_root)
    if env is None:
        return subprocess.CompletedProcess(["git", "-C", str(repo_root), *args], 1, "", FILTER_DISCOVERY_FAILED)
    return subprocess.run(
        ["git", "-C", str(repo_root), *args],
        capture_output=True,
        text=True, encoding='utf-8', errors='replace',
        timeout=timeout,
        check=False,
        stdin=subprocess.DEVNULL,
        env=env,
    )


def _has_active_children(conn: sqlite3.Connection, task_id: str) -> bool:
    return conn.execute(_ACTIVE_CHILDREN_SQL, (task_id,)).fetchone() is not None


_OTHER_LIVE_PATHS_SQL = (
    "SELECT workspace_path FROM tasks "
    "WHERE id != ? AND workspace_path IS NOT NULL "
    f"AND status NOT IN {_TERMINAL_STATUSES_SQL}"
)
# Sibling boards mint their own ids, so the id being cleaned up here can
# name a different live task there. Do not exclude it.
_ANY_LIVE_PATHS_SQL = (
    "SELECT workspace_path FROM tasks "
    "WHERE workspace_path IS NOT NULL "
    f"AND status NOT IN {_TERMINAL_STATUSES_SQL}"
)


def _conn_uses_path(
    conn: sqlite3.Connection, task_id: str, key: str, *, exclude_task_id: bool = True
) -> bool:
    if exclude_task_id:
        rows = conn.execute(_OTHER_LIVE_PATHS_SQL, (task_id,)).fetchall()
    else:
        rows = conn.execute(_ANY_LIVE_PATHS_SQL).fetchall()
    for row in rows:
        other = row["workspace_path"]
        if not other:
            continue
        try:
            other_key = _path_key(Path(other).expanduser().resolve(strict=False))
        except OSError:
            continue
        if other_key == key:
            return True
    return False


def _sibling_board_db_files(conn: sqlite3.Connection) -> list[Path]:
    """Every other board's ``kanban.db``. Raises ``OSError`` when the set is unknown.

    ``kanban_db_path`` follows ``HERMES_KANBAN_DB`` and would collapse every
    slug onto the pinned file, so the scan uses the on-disk layout: the
    default board at ``<home>/kanban.db`` and named boards at
    ``<home>/kanban/boards/<slug>/kanban.db``.
    """
    from hermes_cli.kanban_db_connect import _main_db_file  # late: import cycle

    current_file = _main_db_file(conn)
    current = Path(current_file).resolve() if current_file else None
    candidates = [_kb.kanban_home() / "kanban.db"]
    root = _kb.boards_root()
    if root.is_dir():
        for child in root.iterdir():
            if child.is_dir():
                candidates.append(child / "kanban.db")
    found: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        if not path.is_file():
            continue
        resolved = path.resolve()
        if current is not None and resolved == current:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        found.append(resolved)
    return found


def _other_board_uses_path(db_file: Path, task_id: str, key: str) -> bool:
    uri = db_file.as_uri() + "?mode=ro"
    other = sqlite3.connect(uri, uri=True, timeout=1.0)
    try:
        other.row_factory = sqlite3.Row
        return _conn_uses_path(other, task_id, key, exclude_task_id=False)
    finally:
        other.close()


def _workspace_in_use_by_other(
    conn: sqlite3.Connection, task_id: str, path: Path | str
) -> Optional[str]:
    """Why *path* must be kept: ``"shared"``, ``"unknown"``, or None when free.

    ``gc``, completion and deferred parent cleanup used to ``rmtree`` a shared
    ``workspace_path`` as soon as one of its tasks went terminal. Compare the
    resolved path: a row may store ``~`` or a symlinked spelling.

    The connection covers one board. A ready task on a named board can point
    at the same directory. If that set of databases cannot be read, refuse
    the delete (``"unknown"``) and log the board DB that failed, so one broken
    sibling DB is not reported as a live sharer.
    """
    try:
        key = _path_key(Path(path).expanduser().resolve(strict=False))
    except OSError as exc:
        _kb._log.warning("Cannot resolve workspace %s for task %s: %s", path, task_id, exc)
        return "unknown"
    if not key:
        return None
    try:
        if _conn_uses_path(conn, task_id, key):
            return "shared"
    except sqlite3.Error as exc:
        _kb._log.warning("Cannot read live workspaces for task %s: %s", task_id, exc)
        return "unknown"
    try:
        siblings = _sibling_board_db_files(conn)
    except OSError as exc:
        _kb._log.warning("Cannot list kanban boards for task %s: %s", task_id, exc)
        return "unknown"
    for db_file in siblings:
        try:
            if _other_board_uses_path(db_file, task_id, key):
                return "shared"
        except (OSError, sqlite3.Error) as exc:
            _kb._log.warning(
                "Cannot read board db %s for task %s: %s", db_file, task_id, exc,
            )
            return "unknown"
    return None


def _defer_shared_workspace_cleanup(
    conn: sqlite3.Connection, task_id: str, path: Path | str
) -> bool:
    """Skip removal and record why, when *path* may still be used by a live task."""
    reason = _workspace_in_use_by_other(conn, task_id, path)
    if reason is None:
        return False
    if reason == "shared":
        _kb._log.warning(
            "Deferring workspace cleanup for task %s: %s is still used by "
            "another non-terminal task",
            task_id, path,
        )
    else:
        _kb._log.warning(
            "Deferring workspace cleanup for task %s: cannot tell whether "
            "another task still uses %s",
            task_id, path,
        )
    try:
        _kb._append_event(
            conn, task_id, "workspace_cleanup_deferred_shared",
            {"path": str(path), "reason": reason},
        )
    except sqlite3.Error:
        pass
    return True


def _defer_shared_worktree_cleanup(
    conn: sqlite3.Connection, task_id: str, path: Path | str
) -> bool:
    """Defer removing a linked worktree another live task may still use.

    Only a real linked worktree is ever removed, so the sharing scan runs just
    for those.
    """
    wt = Path(path).expanduser()
    return (
        wt.is_dir()
        and _is_linked_worktree_checkout(wt)
        and _defer_shared_workspace_cleanup(conn, task_id, path)
    )


def _lexical_path(path: Path | str) -> Path:
    """Absolute, ``..``-collapsed, NFC form of *path* WITHOUT following symlinks."""
    return Path(_path_key(os.path.abspath(path)))


def _managed_scratch_path_info(p: Path) -> tuple[bool, Optional[str]]:
    """Return whether *p* is managed scratch storage and the matching board.

    *p* must be strictly below a managed root both after resolving symlinks
    AND lexically (as spelled, without resolving). Resolved containment alone
    is not enough: when a root is itself a symlink to a broad directory
    (relocated storage, or a planted link), every path inside the link target
    would resolve "under" the root, so a scratch task naming such a path
    directly would get it rmtree'd. Tasks created through the root are spelled
    through it, so the lexical check keeps them managed. A root's lexical form
    is accepted both as configured and with its anchor (kanban home, or the
    override's parent) resolved, so a process spelling a symlinked home by its
    real path still matches; the managed ``kanban/.../workspaces`` components
    themselves are never resolved for the lexical check.
    """
    try:
        p_abs = p.resolve(strict=False)
    except OSError:
        return False, None
    p_lex = _lexical_path(p)
    # (resolved root, lexical spellings of the root, board)
    roots: list[tuple[Path, tuple[Path, ...], Optional[str]]] = []

    def _add_root(
        anchor: Path, anchor_real: Path, parts: tuple[str, ...], board: Optional[str]
    ) -> None:
        root = anchor.joinpath(*parts)
        with contextlib.suppress(OSError):
            roots.append((
                root.resolve(strict=False),
                (_lexical_path(root), _lexical_path(anchor_real.joinpath(*parts))),
                board,
            ))

    override = os.environ.get("HERMES_KANBAN_WORKSPACES_ROOT", "").strip()
    if override:
        override_root = Path(override).expanduser()
        with contextlib.suppress(OSError):
            override_parent = override_root.parent
            _add_root(
                override_parent,
                override_parent.resolve(strict=False),
                (override_root.name,),
                None,
            )
    try:
        home = _kb.kanban_home()
        # Resolve the shared anchor once, not once per board root.
        home_real = home.resolve(strict=False)
    except OSError:
        home = None
    if home is not None:
        _add_root(home, home_real, ("kanban", "workspaces"), _kb.DEFAULT_BOARD)
        entries: list[Path] = []
        with contextlib.suppress(OSError):
            entries = list((home / "kanban" / "boards").resolve(strict=False).iterdir())
        for entry in entries:
            with contextlib.suppress(OSError):
                if entry.is_dir():
                    _add_root(
                        home, home_real, ("kanban", "boards", entry.name, "workspaces"), entry.name
                    )
    for root, lexical_roots, board in roots:
        if p_abs == root:
            continue
        try:
            if p_abs.is_relative_to(root) and any(
                p_lex != lex and p_lex.is_relative_to(lex) for lex in lexical_roots
            ):
                return True, board
        except ValueError:
            continue
    return False, None


def _scratch_workspace(conn: sqlite3.Connection, task_id: str) -> Optional[Path]:
    """Expanded ``workspace_path`` when the task uses a scratch workspace, else ``None``."""
    row = conn.execute(
        "SELECT workspace_kind, workspace_path FROM tasks WHERE id = ?",
        (task_id,),
    ).fetchone()
    if not row or row["workspace_kind"] != "scratch" or not row["workspace_path"]:
        return None
    return Path(row["workspace_path"]).expanduser()


def _is_managed_scratch_path(p: Path) -> bool:
    """True iff *p* is a STRICT descendant of a kanban-managed ``workspaces/``
    root (``HERMES_KANBAN_WORKSPACES_ROOT``, ``<kanban_home>/kanban/workspaces``,
    or ``<kanban_home>/kanban/boards/<slug>/workspaces``). A path equal to a
    root is not managed (deleting it would wipe every task's scratch dir);
    ``<kanban_home>/kanban``, ``.../logs`` and ``.../boards/<slug>`` hold
    Hermes' own DB and metadata. :func:`_cleanup_workspace` refuses
    ``rmtree`` outside managed storage — a board ``default_workdir`` on a real
    source tree paired with ``workspace_kind='scratch'`` would otherwise make
    task completion delete user data.

    See #28818.
    """
    return _managed_scratch_path_info(p)[0]


def _cleanup_workspace(conn: sqlite3.Connection, task_id: str) -> None:
    """Remove a task's scratch workspace dir and kill its stale tmux session.
    Called from :func:`complete_task` after the transaction commits; best-effort
    so cleanup never blocks completion. ``scratch`` is removed; ``worktree``
    only when provably free of work (clean tree, every commit reachable from a
    remote-tracking ref); ``dir`` is intentionally preserved."""
    if _kb._defer_post_commit(
        lambda: _cleanup_workspace(conn, task_id),
        conn=conn,
    ):
        return
    try:
        row = conn.execute(_WORKSPACE_ROW_SQL, (task_id,)).fetchone()
        if not row or row["status"] not in {"done", "archived", "failed", "cancelled"}:
            return
        kind: Optional[str] = row["workspace_kind"]
        path: Optional[str] = row["workspace_path"]
        if kind not in _REMOVABLE_KINDS or not path:
            # Not removable itself, but completing may still unblock a deferred
            # parent scratch cleanup (e.g. a 'dir' child of a scratch parent).
            # See #33774.
            _try_cleanup_parent_workspaces(conn, task_id)
            return
        # Defer while any child is not yet terminal so it can still read
        # handoff artifacts from this workspace.
        if _has_active_children(conn, task_id):
            _kb._log.debug(
                "Deferring %s workspace cleanup for task %s: "
                "active children still need workspace at %s",
                kind, task_id, path,
            )
            return
        # Kill the (dead) tmux worker session BEFORE removing a worktree so a
        # lingering worker never has its cwd deleted from under it.
        if kind == "worktree":
            _cleanup_worker_tmux(conn, task_id)
            if not _defer_shared_worktree_cleanup(conn, task_id, path):
                _cleanup_worktree_workspace(task_id, path, row["branch_name"])
            _try_cleanup_parent_workspaces(conn, task_id)
            return
        wp = Path(path)
        if wp.is_dir():
            # Containment guard: a board's ``default_workdir`` can pair
            # ``workspace_kind='scratch'`` with a user path pointing at a real
            # source tree; without this, completion would rmtree the user's data.
            # See #28818. Containment runs first so an unmanaged dir that is
            # never removed does not pay for (or log) the shared-use scan.
            if _is_managed_scratch_path(wp):
                if not _defer_shared_workspace_cleanup(conn, task_id, path):
                    release_lsp_clients(str(wp))
                    shutil.rmtree(wp, ignore_errors=True)
                    _kb._log.debug("Removed scratch workspace: %s", wp)
            else:
                _kb._log.warning(
                    "Refusing to remove out-of-scratch workspace for task %s: %s "
                    "(workspace_kind='scratch' but path is outside any "
                    "kanban-managed workspaces root)",
                    task_id, wp,
                )
        # Kill the owning worker's tmux session if it is now dead, then let any
        # parent whose children are all done run its deferred cleanup.
        _cleanup_worker_tmux(conn, task_id)
        # After cleaning up this task's workspace, check if any parent tasks now have all children done —
        # their deferred cleanup can proceed (#33774).
        _try_cleanup_parent_workspaces(conn, task_id)
    except Exception:
        pass  # best-effort — never block completion


def _cleanup_worktree_workspace(
    task_id: str, path: str, branch_name: Optional[str] = None
) -> None:
    """Remove a finished task's linked git worktree when it holds no work.
    Mirrors the CLI startup pruner (``cli._prune_stale_worktrees``): removal
    requires a clean tree AND every commit reachable from a remote-tracking
    ref; any doubt (dirty, unpushed, unresolvable repo, failing git) preserves
    it. The auto-generated ``wt/<task-id>`` branch is deleted with it; custom
    branches are kept. Best-effort."""
    try:
        from hermes_cli.worktree_ops import _worktree_has_unpushed_commits, _worktree_is_dirty
    except Exception:
        return  # CLI safety predicates unavailable — preserve
    try:
        wp = Path(path).expanduser()
        if not wp.is_dir():
            return
        common = _git_common_dir(wp)
        if common is None or common.name != ".git":
            return  # not a linked worktree of a normal repo — never guess
        repo_root = common.parent
        if _path_key(wp.resolve(strict=False)) == _path_key(repo_root.resolve(strict=False)):
            return  # never remove the main checkout
        if _worktree_is_dirty(str(wp), str(repo_root)) or _worktree_has_unpushed_commits(str(wp)):
            _kb._log.info(
                "Preserving worktree for task %s: dirty or unpushed work at %s",
                task_id, wp,
            )
            return
        blockers = _worktree_removal_blockers(wp)
        if blockers:
            # ``worktree remove`` deletes these with the directory even though `status`
            # called the tree clean; operator content outranks a tidy workspace.
            _kb._log.info(
                "Preserving worktree for task %s: %s at %s",
                task_id, "; ".join(blockers), wp,
            )
            return
        if not _worktree_refs_survive_removal(repo_root, wp):
            return
        # Windows cannot delete a directory while this process has its current
        # directory inside it. Completed workers normally run from their own
        # linked worktree, so move this process back to the main checkout
        # before asking Git to remove the worktree.
        worktree_path = wp.resolve(strict=False)
        try:
            cwd = Path.cwd().resolve(strict=False)
        except OSError:
            # cwd was already deleted (a scratch-kind child's own workspace is
            # rmtree'd before this deferred parent cleanup runs, #33774). A
            # dead cwd cannot hold the worktree open, so leaving it is safe.
            cwd = None
        if cwd is None or cwd == worktree_path or cwd.is_relative_to(worktree_path):
            try:
                os.chdir(repo_root)
            except OSError as exc:
                _kb._log.warning(
                    "Preserving worktree for task %s: cannot leave %s for %s: %s",
                    task_id, cwd or "<deleted cwd>", repo_root, exc,
                )
                return
        # No --force: git's own dirty guard re-verifies at removal time, so if
        # the tree became dirty since our check (TOCTOU) removal fails safe.
        # Which branch the checkout is really on, before it disappears: an empty recorded
        # branch is only the legacy auto-generated ``wt/<id>`` when the worktree was
        # actually sitting on it. A detached role checkout has no branch to delete.
        current_branch = _git_current_branch(wp) or ""
        release_lsp_clients(str(worktree_path))
        result = _git(repo_root, "worktree", "remove", str(wp), timeout=60)
        if result.returncode != 0:
            # Windows can retain a directory handle briefly after cwd changes.
            # Retry once without --force; Git still enforces its dirty guard.
            time.sleep(0.1)
            result = _git(repo_root, "worktree", "remove", str(wp), timeout=60)
        if result.returncode != 0:
            _kb._log.warning(
                "git worktree remove failed for task %s at %s: %s",
                task_id, wp, (result.stderr or result.stdout or "").strip(),
            )
            return
        _kb._log.debug("Removed worktree workspace: %s", wp)
        # Only a branch this task recorded — or the legacy auto-generated name its
        # worktree was demonstrably checked out on — may be deleted. Inventing
        # ``wt/<id>`` for a detached role checkout would force-delete a same-named
        # branch the card never used.
        branch = (branch_name or "").strip()
        if not branch and current_branch == f"wt/{task_id}":
            branch = current_branch
        if branch.startswith("wt/"):
            # The HEAD check above covers the checkout; the branch ref can still hold
            # unique work (an adopted ``wt/*`` checkout that was later detached). Delete
            # only a branch whose tip some remote-tracking ref already contains.
            # ``refs/heads/<name>`` explicitly: a tag of the same short name shadows the
            # branch under ``rev-parse --verify`` (warning only), which would verify the
            # tag's tip and then delete a branch whose own tip has unique work.
            tip = _git(
                repo_root, "rev-parse", "--verify", f"refs/heads/{branch}^{{commit}}", timeout=30)
            reachable = False
            if tip.returncode == 0 and (tip.stdout or "").strip():
                contains = _git(
                    repo_root, "branch", "-r", "--contains", (tip.stdout or "").strip(), timeout=60)
                reachable = contains.returncode == 0 and bool((contains.stdout or "").strip())
            if reachable:
                _git(repo_root, "branch", "-D", branch, timeout=30)
            else:
                _kb._log.info(
                    "Preserving branch %s for task %s: no remote-tracking ref contains it",
                    branch, task_id)
    except Exception:
        pass  # best-effort — never block completion


def _try_cleanup_parent_workspaces(conn: sqlite3.Connection, task_id: str) -> None:
    """Run the deferred cleanup of any parent scratch/worktree workspace whose
    children are now all done/archived/failed/cancelled (called after each
    child completes).

    See #33774.
    """
    try:
        parents = conn.execute(
            "SELECT parent_id FROM task_links WHERE child_id = ?",
            (task_id,),
        ).fetchall()
        for (parent_id,) in parents:
            row = conn.execute(_WORKSPACE_ROW_SQL, (parent_id,)).fetchone()
            if (
                not row
                or row["workspace_kind"] not in _REMOVABLE_KINDS
                or not row["workspace_path"]
                or _has_active_children(conn, parent_id)
            ):
                continue
            ws_path = row["workspace_path"]
            if row["workspace_kind"] == "worktree":
                if _defer_shared_worktree_cleanup(conn, parent_id, ws_path):
                    continue
                _cleanup_worktree_workspace(parent_id, ws_path, row["branch_name"])
                continue
            wp = Path(ws_path)
            if not wp.is_dir():
                continue
            if _is_managed_scratch_path(wp):
                if _defer_shared_workspace_cleanup(conn, parent_id, ws_path):
                    continue
                release_lsp_clients(str(wp))
                shutil.rmtree(wp, ignore_errors=True)
                _kb._log.debug("Deferred cleanup: removed parent %s scratch workspace: %s", parent_id, wp)
    except Exception:
        pass  # best-effort


def _cleanup_worker_tmux(conn: sqlite3.Connection, task_id: str) -> None:
    """Kill the tmux session associated with a task's assignee, if dead."""
    try:
        row = conn.execute(
            "SELECT assignee FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()
        if not row or not row["assignee"]:
            return
        # Workers named swarm1-12 use tmux sessions named swarm-swarm1 etc.
        session = f"swarm-{row['assignee']}"
        out = subprocess.run(
            ["tmux", "list-panes", "-t", session, "-F", "#{pane_dead}"],
            capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=5,
        )
        if out.stdout.strip() == "1":
            subprocess.run(["tmux", "kill-session", "-t", session], capture_output=True, timeout=5)
            _kb._log.debug("Killed stale tmux session: %s", session)
    except Exception:
        pass  # best-effort — never block completion


_SCRATCH_TIP_SENTINEL_NAME = ".scratch_tip_shown"


_SCRATCH_TIP_MESSAGE = (
    "scratch workspaces are ephemeral — they're deleted when the task "
    "completes. Use --workspace worktree: (git worktree) or "
    "--workspace dir:/abs/path (existing dir) to preserve worker output."
)


def _scratch_tip_sentinel_path() -> Path:
    """Path to the per-install scratch-workspace-tip sentinel file."""
    return _kb.kanban_home() / _SCRATCH_TIP_SENTINEL_NAME


def _scratch_tip_shown() -> bool:
    """True iff the scratch-workspace tip was already emitted on this install.
    Best-effort — any error re-emits, the safer failure mode for a help message."""
    try:
        return _scratch_tip_sentinel_path().exists()
    except OSError:
        return False


def _mark_scratch_tip_shown() -> None:
    """Touch the sentinel so future scratch workspaces stay silent. Best-effort:
    a failure means the tip may appear once more, preferable to crashing dispatch."""
    try:
        path = _scratch_tip_sentinel_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)
    except OSError:
        pass


def _maybe_emit_scratch_tip(
    conn: sqlite3.Connection,
    task_id: str,
    workspace_kind: Optional[str],
) -> None:
    """Emit the first-use scratch-workspace tip once per install, right after a
    scratch workspace is materialized. No-op for ``worktree``/``dir`` (preserved
    by design) and once the sentinel exists."""
    if (workspace_kind or "scratch") != "scratch" or _scratch_tip_shown():
        return
    try:
        _kb._log.warning("kanban: %s (task %s)", _SCRATCH_TIP_MESSAGE, task_id)
        with _kb.write_txn(conn):
            _kb._append_event(
                conn, task_id, "tip_scratch_workspace",
                {"message": _SCRATCH_TIP_MESSAGE},
            )
    except Exception:
        # Best-effort — never block the spawn loop over a help message.
        pass
    finally:
        _mark_scratch_tip_shown()


# ---------------------------------------------------------------------------
# Workspace resolution
# ---------------------------------------------------------------------------

def _git_toplevel(path: Path) -> Optional[Path]:
    """Return the git toplevel containing ``path``, or ``None`` if not in a repo."""
    out = _kb._git_out(path, "rev-parse", "--show-toplevel")
    if out is None:
        return None
    try:
        return Path(out).expanduser().resolve()
    except Exception:
        return Path(out).expanduser()


def _git_branch_exists(repo_root: Path, branch_name: str) -> bool:
    try:
        result = _git(repo_root, "show-ref", "--verify", f"refs/heads/{branch_name}", timeout=30)
    except Exception:
        return False
    return result.returncode == 0


def _git_abs_path(path: Path, flag: str) -> Optional[Path]:
    out = _kb._git_out(path, "rev-parse", "--path-format=absolute", flag)
    return Path(out).expanduser().resolve(strict=False) if out else None


def _git_common_dir(path: Path) -> Optional[Path]:
    return _git_abs_path(path, "--git-common-dir")


def _git_current_branch(path: Path) -> Optional[str]:
    return _kb._git_out(path, "branch", "--show-current")


def _is_linked_worktree_checkout(path: Path) -> bool:
    """True when *path* is a linked worktree (git-dir differs from common-dir).

    One ``rev-parse`` call answers both directories.
    """
    out = _kb._git_out(
        path, "rev-parse", "--path-format=absolute", "--git-dir", "--git-common-dir"
    )
    lines = out.splitlines() if out else []
    return len(lines) == 2 and (
        Path(lines[0]).resolve(strict=False) != Path(lines[1]).resolve(strict=False)
    )


def _validate_worktree_registration(repo_root: Path, target: Path) -> None:
    """A copied .git pointer must never authorize moving a sibling's HEAD/index."""
    from hermes_cli.worker_execution_policy import _worktree_registration_backlinks

    try:
        pointer = target / ".git"
        info = pointer.lstat()
        if target.is_symlink() or not stat.S_ISREG(info.st_mode) or info.st_size > 4096:
            raise ValueError
        contents = pointer.read_text(encoding="utf-8").strip()
        if not contents.startswith("gitdir: ") or not contents[8:].strip():
            raise ValueError
        git_dir = (target / contents[8:].strip()).resolve()
        common = _git_common_dir(repo_root)
        if common is None or _path_key(_git_common_dir(target)) != _path_key(common):
            raise ValueError
        if not _worktree_registration_backlinks(
            str(common), str(git_dir), str(target.resolve()),
        ):
            raise ValueError
        registered = _git(repo_root, "worktree", "list", "--porcelain", "-z", timeout=30)
        fields = registered.stdout.split("\0")
        if registered.returncode == 129:  # Git before 2.36 has no worktree-list -z.
            import ast

            registered = _git(
                repo_root, "-c", "core.quotePath=true", "worktree", "list", "--porcelain",
                timeout=30)
            # Porcelain without -z C-quotes unusual paths, including UTF-8 octets.
            fields = [
                "worktree " + ast.literal_eval("b" + field[9:]).decode("utf-8")
                if field.startswith('worktree "') else field
                for field in registered.stdout.split("\n")
            ]
        if registered.returncode != 0 or not any(
            _path_key(Path(field[9:]).resolve()) == _path_key(target.resolve())
            for field in fields if field.startswith("worktree ")
        ):
            raise ValueError
    except (OSError, UnicodeError, ValueError, SyntaxError, subprocess.SubprocessError):
        raise ValueError(
            f"{_CANDIDATE_WORKSPACE_UNAVAILABLE}: unregistered checkout {target}; "
            f"its .git pointer, registration backlink and worktree list must agree in {repo_root}"
        ) from None


def _nearest_existing_path(path: Path) -> Path:
    current = path
    while not current.exists() and current != current.parent:
        current = current.parent
    return current


def _repo_root_for_worktree_target(path: Path) -> Optional[Path]:
    current = _nearest_existing_path(path).resolve(strict=False)
    while True:
        repo_root = _git_toplevel(current)
        if repo_root is not None:
            return repo_root
        if current == current.parent:
            return None
        current = current.parent


def _ensure_git_worktree(repo_root: Path, target: Path, branch_name: str) -> bool:
    """Materialize ``target``, returning False when reusing a linked worktree."""
    target = target.expanduser()
    repo_common = _git_common_dir(repo_root)
    if target.exists() and repo_common is not None and _path_key(_git_common_dir(target)) == _path_key(repo_common):
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    if _git_branch_exists(repo_root, branch_name):
        args = ["worktree", "add", str(target), branch_name]
    else:
        args = ["worktree", "add", "-b", branch_name, str(target), "HEAD"]
    result = _git(repo_root, *args, timeout=60)
    if result.returncode != 0:
        stderr = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(
            f"git worktree add failed for {target} on branch {branch_name}: {stderr}"
        )
    return True


def _anchored_worktree(repo_root: Path, task_id: str, branch_name: str) -> tuple[Path, str, bool]:
    """Materialize the canonical worktree, reporting whether this call created it."""
    target = repo_root / ".worktrees" / task_id
    created = _ensure_git_worktree(repo_root, target, branch_name)
    return target, branch_name, created


def _role_contract(task: Task) -> Optional[dict]:
    """The typed ``review``/``validation`` contract, or ``None`` for every other card."""
    contract = getattr(task, "lifecycle_contract", None)
    if isinstance(contract, dict) and contract.get("kind") in _ROLE_KINDS:
        return contract
    return None


def candidate_head_for_task(conn: sqlite3.Connection, task: Task) -> Optional[str]:
    """The immutable commit a typed review/validation card must work from.

    ``None`` for a card that is not a typed role card. A role card returns the full
    commit id its candidate's own handoff recorded — never HEAD, a branch name, or a
    commit read out of prose — and raises ``ValueError`` naming
    ``candidate_workspace_unavailable`` when the board cannot supply one
    unambiguously. The candidate is the card its contract names, so a completed
    sibling, a later parent, or a moving branch is never mistaken for the work being
    reviewed.
    """
    contract = _role_contract(task)
    if contract is None:
        return None
    candidate_id = str(contract.get("candidate_task_id") or "").strip()
    if not candidate_id:
        raise ValueError(
            f"{_CANDIDATE_WORKSPACE_UNAVAILABLE}: {task.id} names no candidate card")
    state = _kb.get_lifecycle_state(conn, task.id)
    # The lifecycle projection already names the BOUND candidate run, so the documented
    # head must come from that same run.
    handoff = _kb.bound_handoff(conn, candidate_id)
    head = str(state.get("head_sha") or "").strip()
    documented = str(handoff.get("head_sha") or "").strip()
    base = str(handoff.get("base_sha") or "").strip()
    if not _COMMIT_ID.match(head):
        raise ValueError(
            f"{_CANDIDATE_WORKSPACE_UNAVAILABLE}: candidate {candidate_id} of {task.id} "
            "has no recorded head commit")
    if documented.lower() != head.lower():
        raise ValueError(
            f"{_CANDIDATE_WORKSPACE_UNAVAILABLE}: candidate {candidate_id} of {task.id} "
            f"reports head {head} but its handoff records {documented or 'none'}")
    if not _COMMIT_ID.match(base):
        raise ValueError(
            f"{_CANDIDATE_WORKSPACE_UNAVAILABLE}: candidate {candidate_id} of {task.id} "
            "records no base commit")
    return head


def _git_head(path: Path) -> Optional[str]:
    """The full commit id at ``path``'s HEAD, or ``None`` when it cannot be read."""
    out = _kb._git_out(path, "rev-parse", "HEAD")
    if not out or not _COMMIT_ID.match(out):
        return None
    return out


def _untracked_paths_target_tracks(repo: Path, worktree: Path, head: str) -> list[str]:
    """Ignored files and untracked directories the target would overwrite.

    ``git status --porcelain --untracked-files=all`` misses ignored files and empty
    directories. Even ``checkout --no-overwrite-ignore`` can replace an unignored
    empty directory with a tracked file.
    """
    # Two listings on purpose. Per-file first: ``--directory`` collapses a nonempty
    # ignored directory to ``dir/``, losing which children exist, and a tracked sibling
    # (``cache/new`` beside an ignored ``cache/local``) is not a conflict. The second
    # listing adds all untracked directory roots, regardless of ignore status. Git
    # collapses descendants, so check candidate paths beneath those roots on disk.
    files = _git(
        worktree, "ls-files", "--others", "--ignored", "--exclude-standard", "-z", timeout=60)
    if files.returncode != 0:
        return []
    directories = _git(
        worktree, "ls-files", "--others", "--directory", "-z", timeout=60,
    )
    if directories.returncode != 0:
        return []
    present = [path for path in (files.stdout or "").split("\0") if path]
    empty_dirs = []
    known = set(present)
    for entry in (directories.stdout or "").split("\0"):
        if not entry or not entry.endswith("/"):
            continue
        bare = entry.rstrip("/")
        if bare not in known:
            empty_dirs.append(bare)
    if not present and not empty_dirs:
        return []
    tracked = _git(repo, "ls-tree", "-r", "--name-only", "-z", head, timeout=120)
    if tracked.returncode != 0:
        return []
    # Exact matching only: case identity belongs to the filesystem, and git's own
    # ``--no-overwrite-ignore`` guard already decides case-only conflicts correctly.
    # Casefolding here would refuse a move that git would perform safely.
    paths = sorted({path for path in (tracked.stdout or "").split("\0") if path})
    if not paths:
        return []
    def ancestor_tracked(key):
        parent = key
        while "/" in parent:
            parent = parent.rsplit("/", 1)[0]
            index = bisect.bisect_left(paths, parent)
            if index < len(paths) and paths[index] == parent:
                return True
        return False

    def exact_or_tracked_child(key):
        index = bisect.bisect_left(paths, key)
        return (index < len(paths) and paths[index] == key) or (
            index < len(paths) and paths[index].startswith(key + "/"))

    hits = []
    for entry in present:
        # A file can be replaced by a tracked file, a tracked child, or a tracked
        # ancestor (its directory path became a file).
        if exact_or_tracked_child(entry) or ancestor_tracked(entry):
            hits.append(entry)
    for entry in empty_dirs:
        # Git collapses untracked directories, hiding empty descendants. Inspect each
        # tracked descendant too: adding a file is safe, replacing a directory is not.
        index = bisect.bisect_left(paths, entry)
        if index < len(paths) and paths[index] == entry:
            hits.append(entry)
        index = bisect.bisect_left(paths, entry + "/")
        while index < len(paths) and paths[index].startswith(entry + "/"):
            if (worktree / paths[index]).is_dir():
                hits.append(paths[index])
            index += 1
    return hits


def _index_flagged_paths(worktree: Path) -> list[str]:
    flags = _git(worktree, "ls-files", "-v", "-z", timeout=60)
    if flags.returncode != 0:
        return ["<unreadable index>"]
    return [
        item.split(" ", 1)[1] for item in (flags.stdout or "").split("\0")
        if item and not item.startswith("H ")
    ]


def _clean_tree(path: Path, label: str) -> None:
    """Refuse local tracked changes or untracked files before reusing a checkout.

    A reviewer's scratch notes are still work, and only the operator can decide
    they are disposable. A failing ``git status`` never reads as clean. Ignored
    files are invisible here; checkout must use ``--no-overwrite-ignore`` so a
    candidate's tracked file cannot silently replace them.
    """
    result = _git(path, "status", "--porcelain", "--untracked-files=all", timeout=60)
    if result.returncode != 0:
        raise ValueError(
            f"{_CANDIDATE_WORKSPACE_UNAVAILABLE}: cannot verify {label} is clean: "
            f"{(result.stderr or result.stdout or '').strip()[:200]}")
    if (result.stdout or "").strip():
        raise ValueError(
            f"{_CANDIDATE_WORKSPACE_UNAVAILABLE}: {label} has local changes or untracked "
            "files; nothing was moved and no file was removed")
    flagged = _index_flagged_paths(path)
    if flagged:
        raise ValueError(
            f"{_CANDIDATE_WORKSPACE_UNAVAILABLE}: {label} has index flags hiding "
            f"local changes ({', '.join(flagged[:3])}); clear the index flags first")


def _role_repo_root(task: Task, board: Optional[str]) -> Path:
    """The main repository whose ``.worktrees/<task-id>`` this role card owns.

    A card that names a repository root, or already names its own canonical
    worktree, anchors that same repository. Every other explicit path is refused
    instead of retargeted: a reviewer pointed at a sibling's checkout would review
    work that is not its candidate.
    """
    # A retained managed scratch directory is storage, not a repository anchor.
    # Ignore it only during scratch-to-role conversion; reads never rewrite the row.
    managed_scratch = (
        (task.workspace_kind or "scratch") == "scratch"
        and task.workspace_path
        and _managed_scratch_path_info(Path(task.workspace_path).expanduser())[0]
    )
    if not task.workspace_path or managed_scratch:
        board_slug = board if board else _kb.get_current_board()
        board_default = (_kb.read_board_metadata(board_slug).get("default_workdir") or "").strip()
        if not board_default:
            raise ValueError(
                f"{_CANDIDATE_WORKSPACE_UNAVAILABLE}: role worktree for {task.id} needs a "
                f"repository, and board {board_slug!r} has no default_workdir set")
        anchor = Path(board_default).expanduser()
        if not anchor.is_absolute():
            raise ValueError(
                f"{_CANDIDATE_WORKSPACE_UNAVAILABLE}: board {board_slug!r} default_workdir "
                f"{board_default!r} is not absolute")
        repo_root = _git_toplevel(anchor)
        if repo_root is None:
            raise ValueError(
                f"{_CANDIDATE_WORKSPACE_UNAVAILABLE}: board {board_slug!r} default_workdir "
                f"{board_default!r} is not inside a git repository")
    else:
        requested = Path(task.workspace_path).expanduser()
        if not requested.is_absolute():
            raise ValueError(
                f"{_CANDIDATE_WORKSPACE_UNAVAILABLE}: role worktree path "
                f"{task.workspace_path!r} is not absolute")
        requested_resolved = requested.resolve(strict=False)
        repo_root = None
        if requested.exists() and _is_linked_worktree_checkout(requested):
            # A dispatch persists this card's own canonical worktree, so a retry
            # names that checkout instead of the repository root. Resolve it back to
            # the main checkout it belongs to rather than pretending it is one.
            common = _git_common_dir(requested)
            if common is None or common.name != ".git":
                raise ValueError(
                    f"{_CANDIDATE_WORKSPACE_UNAVAILABLE}: role worktree path "
                    f"{task.workspace_path!r} is a linked checkout whose main repository "
                    "cannot be identified")
            repo_root = common.parent.resolve(strict=False)
        elif requested.exists():
            toplevel = _git_toplevel(requested)
            if toplevel is not None and _path_key(toplevel) == _path_key(requested_resolved):
                repo_root = toplevel
        if repo_root is None:
            # Not a repository root: only this card's own canonical worktree anchors.
            repo_root = _repo_root_for_worktree_target(requested.parent)
            if repo_root is None:
                raise ValueError(
                    f"{_CANDIDATE_WORKSPACE_UNAVAILABLE}: role worktree path "
                    f"{task.workspace_path!r} is not inside a git repository")
        canonical = repo_root / ".worktrees" / task.id
        if _path_key(requested_resolved) != _path_key(repo_root) and (
            _path_key(requested_resolved) != _path_key(canonical.resolve(strict=False))
        ):
            raise ValueError(
                f"{_CANDIDATE_WORKSPACE_UNAVAILABLE}: role worktree path "
                f"{task.workspace_path!r} is neither the repository root nor this card's "
                f"own {canonical}; a sibling's checkout is never reused")
    if _is_linked_worktree_checkout(repo_root):
        raise ValueError(
            f"{_CANDIDATE_WORKSPACE_UNAVAILABLE}: role worktrees anchor a main checkout, "
            f"but {repo_root} is itself a linked worktree")
    return repo_root


def _is_ancestor(repo: Path, commit: str, head: str) -> bool:
    if not commit:
        return False
    return _git(repo, "merge-base", "--is-ancestor", commit, head, timeout=60).returncode == 0


def _commit_referenced(repo: Path, commit: str) -> bool:
    """True when a ref that survives worktree removal still reaches ``commit``."""
    if not commit:
        return False
    result = _git(repo, "for-each-ref", "--format=%(refname)", "--contains", commit, timeout=60)
    # Git stores these three namespaces in the worktree's administrative directory,
    # not the common ref store; removing that worktree deletes them along with HEAD.
    return result.returncode == 0 and any(
        not ref.startswith(("refs/worktree/", "refs/bisect/", "refs/rewritten/"))
        for ref in (result.stdout or "").splitlines()
    )


def _role_worktree(
    repo_root: Path, task: Task, candidate_head: str, board: Optional[str] = None,
) -> tuple[Path, str, bool]:
    """Materialize this role card's own worktree at ``candidate_head``.

    Returns ``(path, branch, created)`` with an empty branch for the detached
    checkout that a role worktree normally is. An existing canonical worktree
    must have its own exact registration backlink before any status or checkout
    can touch its index. It is reused only when already at ``candidate_head``;
    a clean named-branch checkout of another commit is moved onto the candidate
    with a non-force detach (its branch ref is untouched). Local files, including
    ignored collisions, are preserved on refusal.
    """
    if not _COMMIT_ID.match(candidate_head or ""):
        raise ValueError(
            f"{_CANDIDATE_WORKSPACE_UNAVAILABLE}: {task.id} has no full candidate commit")
    resolved = _kb._git_out(repo_root, "rev-parse", "--verify", f"{candidate_head}^{{commit}}")
    if not resolved or not _COMMIT_ID.match(resolved):
        raise ValueError(
            f"{_CANDIDATE_WORKSPACE_UNAVAILABLE}: candidate commit {candidate_head} is not "
            f"in {repo_root}")
    head = resolved
    target = repo_root / ".worktrees" / task.id

    def confirm_candidate() -> None:
        from hermes_cli.sqlite_safe_read import connect_tracked

        db_path = _kb.kanban_db_path(board=board).resolve()
        try:
            with contextlib.closing(connect_tracked(
                db_path.as_uri() + "?mode=ro", tracking_path=db_path,
                uri=True, timeout=1.0,
            )) as conn:
                conn.row_factory = sqlite3.Row
                shared = _workspace_in_use_by_other(conn, task.id, target) if target.exists() else None
                # Operator rebinds do not take admission. Re-read immediately before
                # checkout, not before the potentially slow Git safety checks.
                confirmed = candidate_head_for_task(conn, task)
                if confirmed != candidate_head:
                    raise ValueError(
                        f"{_CANDIDATE_WORKSPACE_UNAVAILABLE}: candidate moved from "
                        f"{candidate_head} to {confirmed} before workspace preparation")
        except (OSError, sqlite3.Error):
            shared = "unknown"
        if shared is not None:
            detail = ("is registered to another non-terminal task" if shared == "shared"
                      else "cannot be checked for another task's registration")
            raise ValueError(
                f"{_CANDIDATE_WORKSPACE_UNAVAILABLE}: checkout collision at {target}: {detail}")

    if target.is_symlink() or (target.exists() and not target.is_dir()):
        raise ValueError(
            f"{_CANDIDATE_WORKSPACE_UNAVAILABLE}: {target} is not a real directory")
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        confirm_candidate()
        result = _git(repo_root, "worktree", "add", "--detach", str(target), head, timeout=120)
        if result.returncode != 0:
            raise ValueError(
                f"{_CANDIDATE_WORKSPACE_UNAVAILABLE}: git worktree add failed for {target} at "
                f"{head}: {(result.stderr or result.stdout or '').strip()[:300]}")
        _validate_worktree_registration(repo_root, target)
        if _git_head(target) != head:
            raise ValueError(
                f"{_CANDIDATE_WORKSPACE_UNAVAILABLE}: {target} did not land on {head}")
        return target.resolve(strict=False), "", True
    # An existing canonical checkout: prove it belongs to this repository, then never
    # discard what is in it.
    _validate_worktree_registration(repo_root, target)
    _clean_tree(target, str(target))
    # Ignored files and empty directories are invisible to the clean-tree check.
    clobbered = _untracked_paths_target_tracks(repo_root, target, head)
    if clobbered:
        shown = ", ".join(clobbered[:3])
        raise ValueError(
            f"{_CANDIDATE_WORKSPACE_UNAVAILABLE}: {target} holds ignored file(s) or "
            f"untracked directories the candidate {head} tracks ({shown}); "
            "moving it would overwrite them")
    current = _git_head(target)
    branch = _git_current_branch(target) or ""
    if current == head:
        confirm_candidate()
        return target.resolve(strict=False), branch, False
    if not branch:
        # A detached checkout is the normal shape for a role worktree, and after a
        # rework its old candidate is simply not the new one. Follow the new candidate
        # when moving cannot strand anything — the old commit is contained in the new
        # candidate, or a common ref still reaches it — and refuse otherwise.
        if not (_is_ancestor(target, current, head) or _commit_referenced(target, current)):
            raise ValueError(
                f"{_CANDIDATE_WORKSPACE_UNAVAILABLE}: {target} is detached at "
                f"{current or 'an unknown commit'}, not the candidate {head}, and nothing "
                "reaches that commit; moving it would strand work")
        confirm_candidate()
        result = _git(
            target, "checkout", "--no-overwrite-ignore", "--detach", head, timeout=120)
        if result.returncode != 0:
            raise ValueError(
                f"{_CANDIDATE_WORKSPACE_UNAVAILABLE}: could not move {target} onto {head}: "
                f"{(result.stderr or result.stdout or '').strip()[:300]}")
        if _git_head(target) != head:
            raise ValueError(
                f"{_CANDIDATE_WORKSPACE_UNAVAILABLE}: {target} did not land on {head}")
        return target.resolve(strict=False), "", False
    # Detaching leaves the branch ref behind, and the completion cleanup deletes a
    # ``wt/<id>`` branch once the *detached* HEAD is clean and pushed. So only move a
    # branch whose tip the candidate already contains: anything else is unique work.
    if _git(target, "merge-base", "--is-ancestor", current, head, timeout=60).returncode != 0:
        raise ValueError(
            f"{_CANDIDATE_WORKSPACE_UNAVAILABLE}: {target} is checked out on {branch} at "
            f"{current}, which the candidate {head} does not contain; detaching would "
            "strand that work")
    confirm_candidate()
    result = _git(
            target, "checkout", "--no-overwrite-ignore", "--detach", head, timeout=120)
    if result.returncode != 0:
        raise ValueError(
            f"{_CANDIDATE_WORKSPACE_UNAVAILABLE}: could not move {target} onto {head}: "
            f"{(result.stderr or result.stdout or '').strip()[:300]}")
    if _git_head(target) != head:
        raise ValueError(
            f"{_CANDIDATE_WORKSPACE_UNAVAILABLE}: {target} did not land on {head}")
    return target.resolve(strict=False), "", False


def _validate_role_directory(task: Task, path: Path, candidate_head: Optional[str]) -> None:
    """Check an operator-supplied role workspace in place; never checkout or reset it."""
    if not candidate_head:
        raise ValueError(
            f"{_CANDIDATE_WORKSPACE_UNAVAILABLE}: role workspace for {task.id} needs the "
            "candidate commit, never HEAD")
    if not path.is_dir() or path.is_symlink():
        raise ValueError(
            f"{_CANDIDATE_WORKSPACE_UNAVAILABLE}: role workspace {path} does not exist")
    head = _git_head(path)
    if head is None:
        raise ValueError(
            f"{_CANDIDATE_WORKSPACE_UNAVAILABLE}: role workspace {path} is not a git checkout")
    if head != candidate_head:
        raise ValueError(
            f"{_CANDIDATE_WORKSPACE_UNAVAILABLE}: role workspace {path} is at {head}, not the "
            f"candidate {candidate_head}; this directory is never moved for you")
    _clean_tree(path, str(path))


def _resolve_worktree_workspace(task: Task, *, board: Optional[str] = None,
                                candidate_head: Optional[str] = None) -> tuple[Path, str]:
    """Resolve a worktree, retaining the historical (path, branch) API."""
    path, branch, _created = _resolve_worktree_workspace_created(
        task, board=board, candidate_head=candidate_head)
    return path, branch


def _resolve_worktree_workspace_created(
    task: Task, *, board: Optional[str] = None, candidate_head: Optional[str] = None,
) -> tuple[Path, str, bool]:
    """Resolve + materialize a linked git worktree for ``task``. With no
    ``task.workspace_path`` the anchor is the board's ``default_workdir`` so
    every worktree lands under a board-owned repo (``<repo>/.worktrees/<id>``)
    instead of the dispatcher's incidental CWD (whatever dir the gateway was
    launched from); with no anchor configured we fail loudly rather than guess."""
    if _role_contract(task) is not None:
        if candidate_head is None:
            raise ValueError(
                f"{_CANDIDATE_WORKSPACE_UNAVAILABLE}: {task.id} is a typed role card; its "
                "worktree is materialized at the candidate commit, never at HEAD")
        repo_root = _role_repo_root(task, board)
        return _role_worktree(repo_root, task, candidate_head, board)
    branch_name = (task.branch_name or "").strip() or f"wt/{task.id}"
    if not task.workspace_path:
        board_slug = board if board else _kb.get_current_board()
        board_default = (_kb.read_board_metadata(board_slug).get("default_workdir") or "").strip()
        if not board_default:
            raise ValueError(
                f"task {task.id} has workspace_kind=worktree but no workspace_path, "
                f"and board {board_slug!r} has no default_workdir set. Set a board "
                "default workdir (a git repo) or create the task with "
                "--workspace worktree:<absolute-repo-path>."
            )
        anchor = Path(board_default).expanduser()
        if not anchor.is_absolute():
            raise ValueError(
                f"board {board_slug!r} default_workdir {board_default!r} is not "
                "absolute; use an absolute path to a git repo"
            )
        repo_root = _git_toplevel(anchor)
        if repo_root is None:
            raise ValueError(
                f"task {task.id} has workspace_kind=worktree but board "
                f"{board_slug!r} default_workdir {board_default!r} is not inside a git repo"
            )
        return _anchored_worktree(repo_root, task.id, branch_name)

    requested = Path(task.workspace_path).expanduser()
    if not requested.is_absolute():
        raise ValueError(
            f"task {task.id} has non-absolute worktree path "
            f"{task.workspace_path!r}; use an absolute path"
        )
    requested_resolved = requested.resolve(strict=False)

    if requested.exists() and _is_linked_worktree_checkout(requested):
        actual_branch = _git_current_branch(requested)
        if actual_branch == branch_name:
            return requested_resolved, actual_branch, False
        # The requested path is an existing checkout of a DIFFERENT task's
        # branch (decompose children inherit the root's workspace_path
        # verbatim, so siblings all point here). Reusing it would run this task
        # on the other task's branch — silent cross-task provenance corruption,
        # unsafe under concurrency — so fall back to our own worktree.
        fallback_root = _repo_root_for_worktree_target(requested.parent)
        if fallback_root is not None:
            fallback = fallback_root / ".worktrees" / task.id
            if _path_key(fallback.resolve(strict=False)) != _path_key(requested_resolved):
                created = _ensure_git_worktree(fallback_root, fallback, branch_name)
                return fallback.resolve(strict=False), branch_name, created
        # No repo to anchor a fallback on (or the occupied path IS this task's
        # own canonical worktree): keep the legacy reuse rather than fail dispatch.
        return requested_resolved, actual_branch or branch_name, False

    repo_root = _git_toplevel(requested)
    if repo_root is not None and _path_key(requested_resolved) == _path_key(repo_root):
        return _anchored_worktree(repo_root, task.id, branch_name)

    repo_root = _repo_root_for_worktree_target(requested.parent)
    if repo_root is None:
        raise ValueError(
            f"task {task.id} worktree path {task.workspace_path!r} is not inside a git repo "
            "and does not point at a git repo root"
        )
    created = _ensure_git_worktree(repo_root, requested, branch_name)
    return requested, branch_name, created


def resolve_workspace(task: Task, *, board: Optional[str] = None,
                     candidate_head: Optional[str] = None) -> Path:
    """Resolve (and create if needed) the workspace for a task.

    ``scratch``: ``<board-root>/workspaces/<id>/`` — path-stable across the
    dispatcher and every profile worker. ``dir``: ``workspace_path``, created
    if missing; MUST be absolute (relative paths would resolve against the
    dispatcher's CWD — confused-deputy traversal). ``worktree``: a linked git
    worktree; a repo-root ``workspace_path`` anchors ``<repo>/.worktrees/<id>``,
    a concrete path is created/reused, none -> the board's ``default_workdir``
    (raises if unset rather than guessing). A typed role card additionally needs
    ``candidate_head``: its worktree is its own checkout of the candidate commit,
    and an explicit ``dir`` workspace is verified in place against that commit.
    Persist via ``set_workspace_path``.
    """
    return resolve_workspace_created(
        task, board=board, candidate_head=candidate_head)[0]


def resolve_workspace_created(
    task: Task, *, board: Optional[str] = None, candidate_head: Optional[str] = None,
) -> tuple[Path, Optional[str], bool]:
    """(path, branch, created) — created is True only when this call made the checkout.

    Ordinary cards retain custom targets and inherited-sibling repair; role cards
    retain their candidate-pinned canonical checkout. Non-worktree directories
    have no branch and never authorize worktree discard.
    """
    kind = effective_workspace_kind(task)
    if kind == "worktree":
        # ``scratch`` is the create_task default, so a typed review/validation card can
        # arrive here without a checkout. An empty scratch directory is not evidence:
        # the role would either inspect nothing or — with a legacy explicit path — the
        # repository's current tree instead of the candidate. A role card therefore
        # always resolves a worktree at the candidate commit, or a named refusal.
        return _resolve_worktree_workspace_created(
            task, board=board, candidate_head=candidate_head)
    if kind == "scratch" and not task.workspace_path:
        p = _kb.workspaces_root(board=board) / task.id
    elif kind == "scratch":
        # Legacy explicit-path scratch tasks get the same absolute-path guard
        # as dir: — same threat model.
        p = Path(task.workspace_path).expanduser()
        if not p.is_absolute():
            raise ValueError(
                f"task {task.id} has non-absolute workspace_path "
                f"{task.workspace_path!r}; workspace paths must be absolute"
            )
    elif kind == "dir":
        if not task.workspace_path:
            raise ValueError(f"task {task.id} has workspace_kind=dir but no workspace_path")
        p = Path(task.workspace_path).expanduser()
        if not p.is_absolute():
            raise ValueError(
                f"task {task.id} has non-absolute workspace_path "
                f"{task.workspace_path!r}; use an absolute path "
                f"(relative paths are ambiguous against the dispatcher's CWD)"
            )
        if _role_contract(task) is not None:
            _validate_role_directory(task, p, candidate_head)
    else:
        raise ValueError(f"unknown workspace_kind: {kind}")
    p.mkdir(parents=True, exist_ok=True)
    return p, None, False


def _set_task_column(conn: sqlite3.Connection, task_id: str, column: str, value: str) -> None:
    with _kb.write_txn(conn):
        conn.execute(f"UPDATE tasks SET {column} = ? WHERE id = ?", (value, task_id))


def _worktree_removal_blockers(worktree: Path) -> list[str]:
    """Content ``git worktree remove`` would delete although ``status`` calls it clean.

    Index flags hide modified files; status also omits ignored entries and
    untracked empty directories. Non-force ``worktree remove`` deletes them all.
    """
    reasons: list[str] = []
    flagged = _index_flagged_paths(worktree)
    if flagged:
        reasons.append(f"index flags prevent verification: {', '.join(flagged[:3])}")
    untracked = _git(
        worktree, "ls-files", "--others", "--directory", "-z", timeout=60,
    )
    if untracked.returncode != 0:
        reasons.append("untracked entries could not be read")
    elif (untracked.stdout or "").strip("\0"):
        reasons.append("untracked entries (including ignored files or empty directories) are present")
    return reasons


def _worktree_refs_survive_removal(repo_root: Path, worktree: Path) -> bool:
    """Preserve checkout refs and private recovery history until common refs reach them."""
    help_result = _git(worktree, "for-each-ref", "-h", timeout=60)
    if help_result.returncode not in (0, 129):
        _kb._log.warning("Preserving worktree at %s: cannot probe checkout refs", worktree)
        return False
    root_refs_supported = "--include-root-refs" in (
        (help_result.stdout or "") + (help_result.stderr or ""))
    options = ["--include-root-refs"] if root_refs_supported else []
    refs = _git(worktree, "for-each-ref", *options, "--format=%(refname)", timeout=60)
    if refs.returncode != 0:
        _kb._log.warning("Preserving worktree at %s: cannot enumerate checkout refs", worktree)
        return False
    revisions = [(ref, ref) for ref in ["HEAD", *(refs.stdout or "").splitlines()]]
    # Even --include-root-refs omits multivalued FETCH_HEAD/MERGE_HEAD.
    # Scan their files on every Git version; older Git needs other root refs too.
    git_dir = _git_abs_path(worktree, "--git-dir")
    if git_dir is None:
        return False
    try:
        for path in git_dir.iterdir():
            ref = path.name
            if ref in {"FETCH_HEAD", "MERGE_HEAD"}:
                for line in path.read_text(encoding="ascii").splitlines():
                    object_id = line.split()[0]
                    if not _COMMIT_ID.fullmatch(object_id):
                        raise ValueError("invalid pseudoref")
                    revisions.append((ref, object_id))
            elif not root_refs_supported and (ref.endswith("_HEAD") or ref in {
                "AUTO_MERGE", "BISECT_EXPECTED_REV", "MERGE_AUTOSTASH",
                "NOTES_MERGE_PARTIAL", "NOTES_MERGE_REF",
            }):
                revisions.append((ref, ref))
    except (OSError, ValueError, IndexError):
        _kb._log.warning("Preserving worktree at %s: cannot enumerate checkout refs", worktree)
        return False
    common_objects = None

    def anchor_survives(ref: str, revision: str) -> bool:
        nonlocal common_objects
        # A tag's message/signature belongs to its raw object, not its peeled commit.
        resolved = _git(worktree, "rev-parse", "--verify", revision, timeout=60)
        object_id = (resolved.stdout or "").strip()
        kind = _git(worktree, "cat-file", "-t", object_id, timeout=60)
        if resolved.returncode != 0 or kind.returncode != 0:
            _kb._log.warning(
                "Preserving worktree at %s: cannot resolve checkout ref %s", worktree, ref)
            return False
        if kind.stdout.strip() == "commit":
            survives = _commit_referenced(repo_root, object_id)
        else:
            if common_objects is None:
                # Default objectname is unpeeled; --contains would peel tag objects.
                objects = _git(
                    repo_root, "for-each-ref", "--format=%(objectname) %(refname)", timeout=60)
                if objects.returncode != 0:
                    _kb._log.warning(
                        "Preserving worktree at %s: cannot enumerate surviving objects", worktree)
                    return False
                common_objects = {
                    oid for oid, name in (
                        line.split(" ", 1) for line in (objects.stdout or "").splitlines()
                    )
                    if not name.startswith(("refs/worktree/", "refs/bisect/", "refs/rewritten/"))
                }
            survives = object_id in common_objects
        if not survives:
            _kb._log.info(
                "Preserving worktree at %s: no surviving ref reaches %s %s from %s",
                worktree, kind.stdout.strip(), object_id, ref)
        return survives

    for ref, revision in revisions:
        if not anchor_survives(ref, revision):
            return False
    # --single-worktree excludes sibling HEAD logs; shared branch logs survive
    # removal and must not hold an otherwise disposable checkout hostage.
    reflog = _git(
        worktree, "reflog", "show", "--single-worktree", "--all",
        "--format=%gD", timeout=60)
    if reflog.returncode != 0:
        _kb._log.warning("Preserving worktree at %s: cannot enumerate reflogs", worktree)
        return False
    checked = set()
    for ref in dict.fromkeys(
        entry.split("@{", 1)[0] for entry in (reflog.stdout or "").splitlines()
    ):
        if ref.startswith("refs/") and not ref.startswith(
            ("refs/worktree/", "refs/bisect/", "refs/rewritten/")
        ):
            continue
        # %H exposes only new IDs. The oldest retained record's old ID may be
        # an object's sole anchor, so inspect both columns of the raw log.
        log_path = _git(
            worktree, "rev-parse", "--path-format=absolute", "--git-path", f"logs/{ref}",
            timeout=60)
        try:
            if log_path.returncode != 0 or not (log_path.stdout or "").strip():
                raise ValueError("cannot locate reflog")
            with Path(log_path.stdout.strip()).open("rb") as entries:
                for entry in entries:
                    old, new, _ = entry.split(b" ", 2)
                    for raw in (old, new):
                        object_id = raw.decode("ascii")
                        if not _COMMIT_ID.fullmatch(object_id):
                            raise ValueError("invalid reflog object ID")
                        if not object_id.strip("0") or object_id in checked:
                            continue
                        if not anchor_survives(f"{ref} reflog", object_id):
                            return False
                        checked.add(object_id)
        except (OSError, ValueError):
            _kb._log.warning(
                "Preserving worktree at %s: cannot read reflog %s", worktree, ref)
            return False
    return True


def discard_prepared_worktree(
    task: Task, path: Path | str, prepared_head: str, *, board: Optional[str] = None,
    prepared_new: bool = True,
) -> bool:
    """Reclaim a worktree prepared for a claim that was then refused.

    Only this attempt's new, clean canonical checkout at its prepared head can be
    reclaimed. If removing it would lose private refs or recovery history, move the
    whole registered worktree aside instead, leaving the canonical path free for a
    retry at a different candidate. Return True when that path has been reclaimed.
    """
    # A reused detached checkout's commit can be its last anchor, so an attempt
    # that did not create the directory must not remove it.
    if not prepared_new:
        return False
    try:
        repo_root = _role_repo_root(task, board)
        target = repo_root / ".worktrees" / task.id
        candidate = Path(path).expanduser()
        if _path_key(candidate.resolve(strict=False)) != _path_key(target.resolve(strict=False)):
            return False
        if not target.is_dir() or not _is_linked_worktree_checkout(target):
            return False
        _validate_worktree_registration(repo_root, target)
        if _git_head(target) != prepared_head:
            return False
        _clean_tree(target, str(target))
        if _worktree_removal_blockers(target):
            return False
        if not _worktree_refs_survive_removal(repo_root, target):
            recovery = target.with_name(f"{task.id}.recovery-{time.time_ns()}")
            result = _git(repo_root, "worktree", "move", str(target), str(recovery), timeout=60)
            if result.returncode != 0:
                _kb._log.warning(
                    "prepared workspace at %s could not move to %s: %s",
                    target, recovery, (result.stderr or result.stdout or "").strip()[:300])
                return False
            _kb._log.info("Preserved prepared workspace for task %s at %s", task.id, recovery)
            return True
        result = _git(repo_root, "worktree", "remove", str(target), timeout=60)
        if result.returncode != 0:
            # Not atomic: a partial removal leaves a checkout that resolves as dirty, so
            # make the failure visible instead of silently poisoning every retry.
            _kb._log.warning(
                "prepared workspace at %s was not fully removed: %s",
                target, (result.stderr or result.stdout or "").strip()[:300])
            return False
        return True
    except Exception as exc:
        _kb._log.warning("prepared workspace at %s was not removed: %s", path, exc)
        return False


def effective_workspace_kind(task: Task) -> str:
    """The kind a workspace resolves as, which is what must be persisted.

    A typed role card is never ``scratch``: resolution gives it a linked worktree at the
    candidate commit, so recording it as scratch would send completion through scratch
    cleanup — an ``rmtree`` of a registered worktree instead of ``git worktree remove``.
    """
    kind = task.workspace_kind or "scratch"
    if kind == "scratch" and _role_contract(task) is not None:
        return "worktree"
    return kind


def set_workspace_path(
    conn: sqlite3.Connection, task_id: str, path: Path | str, *, kind: Optional[str] = None,
) -> None:
    """Persist the resolved workspace. Path and kind go in one transaction: a crash
    between two writes would leave a worktree path recorded as ``scratch``, which is
    exactly the pair completion cleanup mis-branches on."""
    if kind is None:
        _set_task_column(conn, task_id, "workspace_path", str(path))
        return
    with _kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET workspace_path = ?, workspace_kind = ? WHERE id = ?",
            (str(path), kind, task_id),
        )


def set_branch_name(conn: sqlite3.Connection, task_id: str, branch_name: str) -> None:
    _set_task_column(conn, task_id, "branch_name", str(branch_name))


# Late-bound origin namespace (see module docstring); imported LAST so this
# module is fully populated before ``kanban_db`` imports from it.
from hermes_cli import kanban_db as _kb  # noqa: E402

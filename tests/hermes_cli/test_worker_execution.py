"""Behavioral authority and inspection regressions for the granted worker capabilities.

The real dispatcher/container proof lives in ``scripts/verify_worker_execution.py``; these
tests pin the policy schema, the frozen grant, source selection/digests, Git inspection
refusals and the final-argument boundary without needing a board or an engine.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import worker_execution as we
from hermes_cli import worker_execution_common as wec
from hermes_cli import worker_execution_policy as wep
from hermes_cli import worker_execution_source as wes

pytestmark = pytest.mark.platforms("linux")  # procfs, namespaces and rootless Podman are Linux-only

PYTHON = __import__("sys").executable
PINNED_PY = "docker.io/library/python@sha256:" + "a" * 64
PINNED_PG = "docker.io/library/postgres@sha256:" + "b" * 64


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True,
    )
    return result.stdout.strip()


def _policy_section(workspace: Path, image_store, *, recipes=None, inspect=True, extras=None):
    return {
        "protocol": 1,
        "image_store": str(image_store) if image_store else None,
        "workspaces": [{
            "path": str(workspace),
            "inspect": inspect,
            "extra_source_paths": list(extras or []),
            "recipes": recipes or {},
        }],
    }


def _write_config(home: Path, section) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        "kanban:\n  worker_execution:\n" + "".join(
            f"    {line}\n" for line in json.dumps(section, indent=2).splitlines()
        ),
        encoding="utf-8",
    )


def _stage_plugin(home: Path, *, module: str = "# guard\n", manifest: str = 'name: cto-consequence-guard\n'):
    plugin = home / "plugins" / "cto-consequence-guard"
    plugin.mkdir(parents=True, exist_ok=True)
    (plugin / "__init__.py").write_text(module, encoding="utf-8")
    (plugin / "plugin.yaml").write_text(manifest, encoding="utf-8")


@pytest.fixture()
def fixture(tmp_path, monkeypatch):
    home = tmp_path / "home"
    for name in ("home", "config", "cache", "data", "hermes", "kanban"):
        (tmp_path / name).mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path / "kanban"))
    monkeypatch.setenv("TMPDIR", str(tmp_path / "tmp"))
    (tmp_path / "tmp").mkdir()
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "fixture@example.invalid")
    _git(repo, "config", "user.name", "Fixture")
    (repo / "app.py").write_text("print('base')\n", encoding="utf-8")
    (repo / "docs").mkdir()
    (repo / "docs" / "readme.md").write_text("hello\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")
    base = _git(repo, "rev-parse", "HEAD")
    (repo / "app.py").write_text("print('changed')\n", encoding="utf-8")
    _git(repo, "add", "app.py")
    _git(repo, "commit", "-qm", "head")
    head = _git(repo, "rev-parse", "HEAD")
    (repo / "app.py").write_text("print('dirty')\n", encoding="utf-8")
    (repo / "notes.txt").write_text("staged\n", encoding="utf-8")
    _git(repo, "add", "notes.txt")
    image_store = tmp_path / "images"
    image_store.mkdir(mode=0o700)
    (image_store / "inventory.json").write_text('{"images": {}}', encoding="utf-8")
    return SimpleNamespace(
        root=tmp_path, home=home, repo=repo, base=base, head=head,
        hermes=tmp_path / "hermes", images=image_store,
    )


def _board(fixture, status: str = "running") -> tuple[str, str]:
    """A real board with one claimed-looking task, so the grant role is resolved."""
    from hermes_cli import kanban_db as kb

    db_path = fixture.hermes / "kanban.db"
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        connection.executescript(kb.SCHEMA_SQL)
        kb._ensure_lifecycle_schema(connection)
        task_id = kb.create_task(
            connection, title="fixture task", assignee="reviewer",
            workspace_kind="dir", workspace_path=str(fixture.repo),
        )
        connection.execute("UPDATE tasks SET status = ? WHERE id = ?", (status, task_id))
        connection.commit()
    finally:
        connection.close()
    return task_id, str(db_path)


def _authority(
    fixture, *, recipes=None, inspect=True, extras=None, review=None, image_store=True,
    status: str = "running",
):
    task_id, db_path = _board(fixture, status)
    section = _policy_section(
        fixture.repo, fixture.images if image_store else None,
        recipes=recipes, inspect=inspect, extras=extras,
    )
    _write_config(fixture.hermes, section)
    _stage_plugin(fixture.hermes)
    identity = SimpleNamespace(
        generation="gen", dependency_fingerprint="dep", code_sha="sha",
        pid=os.getpid(), start_time=0, as_dict=lambda: {"generation": "gen"},
    )
    grant, refusal = wep.resolve_execution_grant(
        profile_home=str(fixture.hermes), workspace=str(fixture.repo), task_id=task_id,
        profile="reviewer", board="fixture", db_path=db_path,
        preparation_id="prep-fixture", runtime_identity=identity,
    )
    assert refusal is None, refusal
    assert grant is not None
    section = grant.bind(7, "claim-fixture")
    if review is not None:
        body = dict(section)
        body.pop("digest")
        body["review"] = review
        body["digest"] = wec.digest_of(body)
        section = body
    stub = SimpleNamespace(
        task_id=task_id, profile="reviewer", board="fixture", preparation_id="prep-fixture",
        run_id=7, claim_lock="claim-fixture", workspace=str(fixture.repo),
        db_path=db_path, runtime_identity=identity,
    )
    return wep.parse_grant_section(section, stub)


# ---- policy schema -------------------------------------------------------


def test_policy_rejects_unknown_keys_and_bad_images(fixture):
    base = _policy_section(fixture.repo, fixture.images)
    bad = json.loads(json.dumps(base))
    bad["extra"] = True
    with pytest.raises(wec.WorkerExecutionRefused) as excinfo:
        wep.parse_policy(bad)
    assert excinfo.value.code == "policy_malformed"

    bad = json.loads(json.dumps(base))
    bad["workspaces"][0]["recipes"] = {"r": {"image": "python:3.14", "argv": ["/bin/x"], "timeout_seconds": 1, "services": []}}
    with pytest.raises(wec.WorkerExecutionRefused):
        wep.parse_policy(bad)

    policy = wep.parse_policy(base)
    assert policy.digest.startswith("sha256:")


def test_policy_service_and_timeout_bounds(fixture):
    recipe = {
        "image": PINNED_PY, "argv": ["/bin/true"], "timeout_seconds": 300,
        "services": [{"id": "db", "image": PINNED_PG, "argv": ["/bin/true"],
                      "ready_argv": ["/bin/true"], "port": 55432}],
    }
    section = _policy_section(fixture.repo, fixture.images, recipes={"check": recipe})
    assert wep.parse_policy(section).recipes()["check"].services[0].port == 55432

    for mutate in (
        lambda r: r.update(timeout_seconds=0),
        lambda r: r.update(timeout_seconds=True),
        lambda r: r.update(timeout_seconds=3601),
        lambda r: r["services"][0].update(port=80),
        lambda r: r["services"][0].update(id="DB"),
        lambda r: r["services"][0].update(argv=[]),
        lambda r: r["services"].append(dict(r["services"][0], id="db2")),
        lambda r: r.update(extra=True),
        lambda r: r["services"][0].update(mount="/data"),
    ):
        bad = json.loads(json.dumps(section))
        mutate(bad["workspaces"][0]["recipes"]["check"])
        with pytest.raises(wec.WorkerExecutionRefused):
            wep.parse_policy(bad)

    no_store = _policy_section(fixture.repo, None, recipes={"check": recipe})
    with pytest.raises(wec.WorkerExecutionRefused):
        wep.parse_policy(no_store)


def test_policy_extra_source_paths_are_strict(fixture):
    for extra in ("/abs/file", "../escape", "a//b", "a/./b", "a\\b"):
        section = _policy_section(fixture.repo, fixture.images, extras=[extra])
        with pytest.raises(wec.WorkerExecutionRefused):
            wep.parse_policy(section)


# ---- frozen grant --------------------------------------------------------


def test_grant_digest_and_identities_are_frozen(fixture):
    authority = _authority(fixture)
    assert authority.inspect is True
    assert authority.workspace_path == str(fixture.repo)

    section = _policy_section(fixture.repo, fixture.images)
    _write_config(fixture.hermes, section)
    task_id, db_path = _board(fixture)
    identity = SimpleNamespace(generation="gen", dependency_fingerprint="dep", code_sha="sha", pid=os.getpid())
    grant, refusal = wep.resolve_execution_grant(
        profile_home=str(fixture.hermes), workspace=str(fixture.repo), task_id=task_id,
        profile="reviewer", board="fixture", db_path=db_path,
        preparation_id="prep-fixture", runtime_identity=identity,
    )
    assert refusal is None
    bound = grant.bind(7, "claim-fixture")
    body = dict(bound)
    body["inspect"] = False
    stub = SimpleNamespace(
        task_id=task_id, profile="reviewer", board="fixture", preparation_id="prep-fixture",
        run_id=7, claim_lock="claim-fixture", workspace=str(fixture.repo),
        db_path=db_path, runtime_identity=identity,
    )
    with pytest.raises(wec.WorkerExecutionUnavailable):
        wep.parse_grant_section(body, stub)


def test_grant_withheld_for_unmatched_workspace_and_missing_plugin(fixture):
    section = _policy_section(fixture.root / "elsewhere", fixture.images)
    _write_config(fixture.hermes, section)
    _stage_plugin(fixture.hermes)
    identity = SimpleNamespace(generation="gen", dependency_fingerprint="dep", code_sha="sha", pid=os.getpid())
    grant, refusal = wep.resolve_execution_grant(
        profile_home=str(fixture.hermes), workspace=str(fixture.repo), task_id="t_fixture",
        profile="reviewer", board="fixture", db_path=str(fixture.hermes / "kanban.db"),
        preparation_id="prep-fixture", runtime_identity=identity,
    )
    assert grant is None and refusal == "workspace_not_granted"

    _write_config(fixture.hermes, _policy_section(fixture.repo, fixture.images))
    for path in (fixture.hermes / "plugins").rglob("*"):
        if path.is_file():
            path.unlink()
    grant, refusal = wep.resolve_execution_grant(
        profile_home=str(fixture.hermes), workspace=str(fixture.repo), task_id="t_fixture",
        profile="reviewer", board="fixture", db_path=str(fixture.hermes / "kanban.db"),
        preparation_id="prep-fixture", runtime_identity=identity,
    )
    assert grant is None and refusal == "policy_plugin_unavailable"


# ---- source inventory and inspection -------------------------------------


def test_scan_digest_is_stable_and_content_addressed(fixture):
    authority = _authority(fixture, extras=["notes.txt"])
    first = wes._scan_source(authority)
    second = wes._scan_source(authority)
    assert first.source_digest == second.source_digest
    assert first.head == fixture.head
    assert first.branch is not None and not first.detached
    assert {entry.path for entry in first.entries} == {"app.py", "docs/readme.md", "notes.txt"}

    (fixture.repo / "app.py").write_text("print('changed again')\n", encoding="utf-8")
    third = wes._scan_source(authority)
    assert third.source_digest != first.source_digest


def test_reserved_tracked_path_refuses_the_whole_inventory(fixture):
    (fixture.repo / ".env").write_text("SECRET=1\n", encoding="utf-8")
    _git(fixture.repo, "add", ".env")
    _git(fixture.repo, "commit", "-qm", "secret")
    authority = _authority(fixture)
    with pytest.raises(wec.WorkerExecutionRefused) as excinfo:
        wes._scan_source(authority)
    assert excinfo.value.code == "reserved_source_paths"


def test_hardlinked_and_escaping_symlink_sources_refuse(fixture):
    os.link(fixture.repo / "app.py", fixture.repo / "copy.py")
    _git(fixture.repo, "add", "copy.py")
    _git(fixture.repo, "commit", "-qm", "hardlink")
    authority = _authority(fixture)
    with pytest.raises(wec.WorkerExecutionRefused) as excinfo:
        wes._scan_source(authority)
    assert excinfo.value.code == "hardlinked_source"

    (fixture.repo / "copy.py").unlink()
    os.symlink("/etc/passwd", fixture.repo / "escape")
    _git(fixture.repo, "add", "-A")
    _git(fixture.repo, "commit", "-qm", "symlink")
    authority = _authority(fixture)
    with pytest.raises(wec.WorkerExecutionRefused) as excinfo:
        wes._scan_source(authority)
    assert excinfo.value.code == "unsafe_symlink"


def test_deleted_tracked_paths_are_recorded(fixture):
    (fixture.repo / "gone.py").write_text("x = 1\n", encoding="utf-8")
    _git(fixture.repo, "add", "gone.py")
    _git(fixture.repo, "commit", "-qm", "gone")
    (fixture.repo / "gone.py").unlink()
    authority = _authority(fixture)
    snapshot = wes._scan_source(authority)
    entry = snapshot.entry("gone.py")
    assert entry is not None and entry.kind == "deleted"


def test_non_repository_is_explicitly_unavailable(fixture):
    plain = fixture.root / "plain"
    plain.mkdir()
    section = _policy_section(plain, fixture.images)
    _write_config(fixture.hermes, section)
    _stage_plugin(fixture.hermes)
    task_id, db_path = _board(fixture)
    identity = SimpleNamespace(generation="gen", dependency_fingerprint="dep", code_sha="sha", pid=os.getpid())
    grant, refusal = wep.resolve_execution_grant(
        profile_home=str(fixture.hermes), workspace=str(plain), task_id=task_id,
        profile="reviewer", board="fixture", db_path=db_path,
        preparation_id="prep-fixture", runtime_identity=identity,
    )
    assert refusal is None and grant is not None
    stub = SimpleNamespace(
        task_id=task_id, profile="reviewer", board="fixture", preparation_id="prep-fixture",
        run_id=7, claim_lock="claim-fixture", workspace=str(plain),
        db_path=db_path, runtime_identity=identity,
    )
    authority = wep.parse_grant_section(grant.bind(7, "claim-fixture"), stub)
    assert authority.git is None
    with pytest.raises(wec.WorkerExecutionRefused) as excinfo:
        wes._scan_source(authority)
    assert excinfo.value.code == "workspace_not_a_repository"


def test_nested_relative_symlink_resolves_against_its_parent(fixture):
    # docs/link -> ../app.py is a valid in-tree target; docs/link2 -> ../outside is not.
    os.symlink("../app.py", fixture.repo / "docs" / "link")
    _git(fixture.repo, "add", "docs/link")
    _git(fixture.repo, "commit", "-qm", "link")
    authority = _authority(fixture)
    snapshot = wes._scan_source(authority)
    entry = snapshot.entry("docs/link")
    assert entry is not None and entry.kind == "link"

    os.symlink("../outside", fixture.repo / "docs" / "link2")
    _git(fixture.repo, "add", "docs/link2")
    _git(fixture.repo, "commit", "-qm", "escape-link")
    authority = _authority(fixture)
    with pytest.raises(wec.WorkerExecutionRefused) as excinfo:
        wes._scan_source(authority)
    assert excinfo.value.code == "unsafe_symlink"


def test_validation_card_resolves_review_role(fixture):
    from hermes_cli import kanban_db as kb

    db_path = fixture.hermes / "kanban.db"
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        connection.executescript(kb.SCHEMA_SQL)
        kb._ensure_lifecycle_schema(connection)
        implementation = kb.create_task(
            connection, title="candidate", assignee="implementer",
            workspace_kind="dir", workspace_path=str(fixture.repo),
            lifecycle_contract={
                "kind": "code", "review_mode": "same_card", "reviewer": "reviewer",
                "validation_required": True,
            },
        )
        validation = kb.create_task(
            connection, title="validate", assignee="validator",
            lifecycle_contract={"kind": "validation", "candidate_task_id": implementation},
        )
        connection.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (validation,))
        connection.execute(
            "INSERT INTO task_runs (task_id, profile, status, outcome, metadata, started_at) "
            "VALUES (?, 'implementer', 'done', 'completed', ?, 1)",
            (implementation, json.dumps({"base_sha": fixture.base, "head_sha": fixture.head})),
        )
        connection.commit()
    finally:
        connection.close()
    role, pair = wep.resolve_review_context(str(db_path), validation)
    assert role == "review"
    assert pair == {"base": fixture.base, "head": fixture.head}


def test_symlink_cycle_is_refused(fixture):
    os.symlink("b.py", fixture.repo / "a.py")
    os.symlink("a.py", fixture.repo / "b.py")
    _git(fixture.repo, "add", "a.py", "b.py")
    _git(fixture.repo, "commit", "-qm", "cycle")
    authority = _authority(fixture)
    with pytest.raises(wec.WorkerExecutionRefused) as excinfo:
        wes._scan_source(authority)
    assert excinfo.value.code == "unsafe_symlink"


def test_symlink_digest_binds_the_exact_link_text(fixture):
    os.symlink("app.py", fixture.repo / "alias")
    _git(fixture.repo, "add", "alias")
    _git(fixture.repo, "commit", "-qm", "alias")
    authority = _authority(fixture)
    first = wes._scan_source(authority).entry("alias")
    assert first is not None and first.digest.startswith("sha256:")
    # Same destination, different text: the copied workspace differs, so the digest must.
    (fixture.repo / "alias").unlink()
    os.symlink("./app.py", fixture.repo / "alias")
    second = wes._scan_source(authority).entry("alias")
    assert second is not None and second.digest != first.digest


def test_tracked_symlink_replaced_by_file_is_hashed_not_omitted(fixture):
    os.symlink("app.py", fixture.repo / "alias")
    _git(fixture.repo, "add", "alias")
    _git(fixture.repo, "commit", "-qm", "alias")
    (fixture.repo / "alias").unlink()
    (fixture.repo / "alias").write_text("print('real file')\n", encoding="utf-8")
    authority = _authority(fixture)
    snapshot = wes._scan_source(authority)
    entry = snapshot.entry("alias")
    assert entry is not None and entry.kind == "file"
    first = snapshot.source_digest
    (fixture.repo / "alias").write_text("print('changed file')\n", encoding="utf-8")
    second = wes._scan_source(authority).source_digest
    assert second != first


def test_read_reports_next_line_and_matches_the_digest(fixture):
    (fixture.repo / "multi.py").write_text("one\ntwo\nthree\n", encoding="utf-8")
    _git(fixture.repo, "add", "multi.py")
    _git(fixture.repo, "commit", "-qm", "multi")
    authority = _authority(fixture)
    snapshot = wes._scan_source(authority)
    entry = snapshot.entry("multi.py")
    with wes.SafeRoot(authority.workspace_path) as root:
        text, next_line = wes._read_file_page(root, ["multi.py"], entry, 1, 1)
        assert text == "one\n"
        assert next_line == 2
        whole, end = wes._read_file_page(root, ["multi.py"], entry, 1, 10)
        assert end is None and whole == "one\ntwo\nthree\n"
        # A digest from another state refuses instead of returning bytes.
        stale = wes.SourceEntry(entry.path, entry.kind, entry.executable, entry.size, "sha256:" + "0" * 64)
        with pytest.raises(wec.WorkerExecutionRefused) as excinfo:
            wes._read_file_page(root, ["multi.py"], stale, 1, 1)
    assert excinfo.value.code == "source_changed"


def test_repository_paths_must_stay_relative_and_component_clean():
    """A crafted index or diff cannot name an escaping path."""
    for name in ("../outside.txt", "/etc/passwd", "a//b", "a/./b", "a/../b", "a\\b", ""):
        with pytest.raises(wec.WorkerExecutionRefused) as excinfo:
            wes._validate_source_name(name)
        assert excinfo.value.code == "git_unsupported"
    wes._validate_source_name("src/app.py")


def test_image_store_rejects_symlinked_leaf_and_ancestor(tmp_path):
    real = tmp_path / "images"
    real.mkdir(mode=0o700)
    policy = wep.WorkerExecutionPolicy

    leaf = tmp_path / "leaf"
    leaf.symlink_to(real, target_is_directory=True)
    with pytest.raises(wec.WorkerExecutionRefused) as excinfo:
        wep._resolve_image_store(policy(image_store=str(leaf), workspaces=(), digest="sha256:" + "0" * 64))
    assert excinfo.value.code == "policy_malformed"

    workspace = tmp_path / "workspace"
    (workspace / "images").mkdir(parents=True, mode=0o700)
    ancestor = tmp_path / "link"
    ancestor.symlink_to(workspace, target_is_directory=True)
    with pytest.raises(wec.WorkerExecutionRefused) as excinfo:
        wep._resolve_image_store(
            policy(image_store=str(ancestor / "images"), workspaces=(), digest="sha256:" + "0" * 64),
        )
    assert excinfo.value.code == "policy_malformed"

    resolved = wep._resolve_image_store(
        policy(image_store=str(real), workspaces=(), digest="sha256:" + "0" * 64),
    )
    assert resolved["path"] == str(real.resolve())


def test_required_approval_resolves_async_callbacks(monkeypatch):
    from hermes_cli import plugins, plugins_dispatch

    async def async_callback(**kwargs):
        return {
            "action": "allow",
            "policy_digest": kwargs["policy_digest"],
            "args_digest": kwargs["args_digest"],
        }

    class Manager:
        _hooks = {"pre_worker_execution": [async_callback]}

        def required_hook_callback(self, hook_name, owner):
            return async_callback, "ok"

        @staticmethod
        def _hook_callback_kwargs(cb, payload):
            return dict(payload)

        def _run_hook_callback_bounded(self, *args):
            raise AssertionError("the threaded path is disabled")

    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: Manager())
    monkeypatch.setattr(plugins, "_resolve_hook_callback_timeout", lambda: 0.0)
    ok, reason = plugins_dispatch.dispatch_required_worker_execution(
        owner=object(), tool_name="kanban_worker_inspect", args={},
        policy_digest="policy", args_digest="args", operation="summary", recipe=None,
    )
    assert ok, reason


def test_required_approval_verifies_hosted_sources_without_the_module(tmp_path):
    """A proxy names a module the parent never imports; recorded bytes must verify."""
    import sys

    from hermes_cli import plugins
    from hermes_cli.plugins_dispatch import PluginDispatchMixin

    class Manager(PluginDispatchMixin):
        def __init__(self, hooks):
            self._hooks = hooks

    plugin_dir = tmp_path / "guard"
    plugin_dir.mkdir()
    module = plugin_dir / "__init__.py"
    manifest = plugin_dir / "plugin.yaml"
    module.write_text("def callback():\n    return {}\n", encoding="utf-8")
    manifest.write_text("name: cto-consequence-guard\n", encoding="utf-8")

    def callback(**kwargs):
        return {}

    callback.__module__ = "guard_hosted_only"
    assert callback.__module__ not in sys.modules
    owner = {
        "id": "cto-consequence-guard",
        "module_sha256": wec.sha256_hex(module.read_bytes()),
        "manifest_sha256": wec.sha256_hex(manifest.read_bytes()),
    }
    plugins.REQUIRED_HOOK_CALLBACK_OWNERS[callback] = "cto-consequence-guard"
    plugins.REQUIRED_HOOK_CALLBACK_SOURCES[callback] = (str(module), str(manifest))
    try:
        manager = Manager({"pre_worker_execution": [callback]})
        found, reason = manager.required_hook_callback("pre_worker_execution", owner)
        assert found is callback, reason

        module.write_bytes(module.read_bytes() + b"\n# tampered\n")
        found, reason = manager.required_hook_callback("pre_worker_execution", owner)
        assert found is None and "do not match" in reason

        module.write_text("def callback():\n    return {}\n", encoding="utf-8")
        manifest.unlink()
        found, reason = manager.required_hook_callback("pre_worker_execution", owner)
        assert found is None and "do not match" in reason
    finally:
        plugins.REQUIRED_HOOK_CALLBACK_OWNERS.pop(callback, None)
        plugins.REQUIRED_HOOK_CALLBACK_SOURCES.pop(callback, None)


def test_required_approval_survives_disabled_timeouts_and_concurrent_calls(monkeypatch):
    from hermes_cli import plugins, plugins_dispatch

    calls: list[dict] = []

    def callback(**kwargs):
        calls.append(kwargs)
        return {
            "action": "allow",
            "policy_digest": kwargs["policy_digest"],
            "args_digest": kwargs["args_digest"],
        }

    class Manager:
        _hooks = {"pre_worker_execution": [callback]}

        def required_hook_callback(self, hook_name, owner):
            return callback, "ok"

        @staticmethod
        def _hook_callback_kwargs(cb, payload):
            return dict(payload)

        def _run_hook_callback_bounded(self, *args):
            raise AssertionError("the threaded path is disabled")

    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: Manager())
    monkeypatch.setattr(plugins, "_resolve_hook_callback_timeout", lambda: 0.0)
    for _ in range(2):
        ok, reason = plugins_dispatch.dispatch_required_worker_execution(
            owner=object(), tool_name="kanban_worker_inspect", args={},
            policy_digest="policy", args_digest="args", operation="summary", recipe=None,
        )
        assert ok, reason
    assert len(calls) == 2
    assert calls[0]["tool_call_id"] != calls[1]["tool_call_id"]


def test_read_pages_and_binary_refusal(fixture):
    authority = _authority(fixture)
    snapshot = wes._scan_source(authority)
    entry = snapshot.entry("app.py")
    assert entry is not None
    with wes.SafeRoot(authority.workspace_path) as root:
        text, next_line = wes._read_file_page(root, ["app.py"], entry, 1, 1)
    assert text == "print('dirty')\n"
    assert next_line is None

    (fixture.repo / "blob.bin").write_bytes(b"\x00\x01\x02")
    _git(fixture.repo, "add", "blob.bin")
    _git(fixture.repo, "commit", "-qm", "blob")
    authority = _authority(fixture)
    snapshot = wes._scan_source(authority)
    blob = snapshot.entry("blob.bin")
    with wes.SafeRoot(authority.workspace_path) as root:
        with pytest.raises(wec.WorkerExecutionRefused) as excinfo:
            wes._read_file_page(root, ["blob.bin"], blob, 1, 10)
    assert excinfo.value.code == "binary_content"


def test_diff_only_accepts_the_granted_review_pair(fixture):
    authority = _authority(
        fixture, status="review", review={"base": fixture.base, "head": fixture.head},
    )
    args = {"operation": "diff", "base": fixture.base, "head": fixture.head}
    result = wes._inspect_diff(args, authority)
    assert result["ok"] is True and result["mode"] == "commit"
    assert "changed" in result["patch"]
    assert result["diff_digest"].startswith("sha256:")

    with pytest.raises(wec.WorkerExecutionRefused) as excinfo:
        wes._inspect_diff({"operation": "diff", "base": fixture.head, "head": fixture.base}, authority)
    assert excinfo.value.code == "diff_not_granted"

    with pytest.raises(wec.WorkerExecutionRefused) as excinfo:
        wes._inspect_diff({"operation": "diff", "mode": "working"}, authority)
    assert excinfo.value.code == "diff_not_granted"

    with pytest.raises(wec.WorkerExecutionRefused) as excinfo:
        wes._inspect_diff(dict(args, expected_diff_digest="sha256:" + "c" * 64), authority)
    assert excinfo.value.code == "source_changed"


def test_working_diff_requires_implementation_claim(fixture):
    authority = _authority(fixture)
    result = wes._inspect_diff({"operation": "diff", "mode": "working"}, authority)
    assert "dirty" in result["patch"]
    staged = wes._inspect_diff({"operation": "diff", "mode": "staged"}, authority)
    assert "notes.txt" in staged["patch"]
    with pytest.raises(wec.WorkerExecutionRefused) as excinfo:
        wes._inspect_diff({"operation": "diff", "base": fixture.base, "head": fixture.head}, authority)
    assert excinfo.value.code == "diff_not_granted"


def test_diff_refuses_reserved_paths_and_option_injection(fixture):
    authority = _authority(fixture)
    with pytest.raises(wec.WorkerExecutionRefused):
        wes._validate_inspect_args({"operation": "diff", "base": "--output=/tmp/x", "head": fixture.head})
    with pytest.raises(wec.WorkerExecutionRefused) as excinfo:
        wes._inspect_diff({"operation": "diff", "mode": "working", "path": ".env"}, authority)
    assert excinfo.value.code == "reserved_source_paths"


def test_inspect_argument_origin_is_enforced(fixture):
    result = we.worker_inspect({"operation": "summary"}, original_args={"operation": "read"})
    assert json.loads(result)["error"] == "arguments_changed"
    result = we.worker_inspect({"operation": "summary"}, original_args=None)
    assert json.loads(result)["error"] in {"worker_execution_unavailable", "worker_execution_failed"}


# ---- registry reservation ------------------------------------------------


def test_reserved_native_tools_cannot_be_replaced_or_removed():
    from tools import registry as registry_module

    registry_module.discover_builtin_tools()
    registry = registry_module.registry
    assert registry.get_entry("kanban_worker_inspect") is not None
    assert registry.get_entry("kanban_worker_verify") is not None

    def _lookalike(args, **kwargs):
        return json.dumps({"ok": True})

    with pytest.raises(PermissionError):
        registry.register(
            name="kanban_worker_inspect", toolset="kanban",
            schema={"name": "kanban_worker_inspect", "parameters": {"type": "object"}},
            handler=_lookalike, override=True,
        )
    with pytest.raises(PermissionError):
        registry.deregister("kanban_worker_inspect")
    with pytest.raises(PermissionError):
        registry.register(
            name="kanban_worker_verify", toolset="kanban",
            schema={"name": "kanban_worker_verify", "parameters": {"type": "object"}},
            handler=_lookalike, scope="/tmp/scope",
        )


# ---- hook ownership ------------------------------------------------------


def test_required_hook_requires_the_owned_callback_bytes(tmp_path, monkeypatch):
    from hermes_cli import plugins as plugins_module
    from hermes_cli import plugins_dispatch

    module = tmp_path / "guard.py"
    module.write_text("def _on_pre_worker_execution(**kwargs):\n    return None\n", encoding="utf-8")
    manifest = tmp_path / "plugin.yaml"
    manifest.write_text("name: cto-consequence-guard\n", encoding="utf-8")
    spec = __import__("importlib.util", fromlist=["spec_from_file_location"]).spec_from_file_location(
        "guard_fixture_module", module,
    )
    loaded = __import__("importlib.util", fromlist=["module_from_spec"]).module_from_spec(spec)
    __import__("sys").modules["guard_fixture_module"] = loaded
    spec.loader.exec_module(loaded)

    class _Manager(plugins_dispatch.PluginDispatchMixin):
        def __init__(self):
            self._hooks = {"pre_worker_execution": [loaded._on_pre_worker_execution]}
            self._hook_failures_reported = set()
            self._hook_timeout_lock = __import__("threading").RLock()
            self._hook_timeout_suppressed_until = {}
            self._hook_running_callbacks = set()
            self._hook_abandoned = {}

    manager = _Manager()
    plugins_module.REQUIRED_HOOK_CALLBACK_OWNERS[loaded._on_pre_worker_execution] = "cto-consequence-guard"
    try:
        import hashlib

        owner = {
            "id": "cto-consequence-guard",
            "module_sha256": hashlib.sha256(module.read_bytes()).hexdigest(),
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        }
        callback, reason = manager.required_hook_callback("pre_worker_execution", owner)
        assert callback is loaded._on_pre_worker_execution, reason
        tampered = dict(owner, module_sha256="0" * 64)
        callback, reason = manager.required_hook_callback("pre_worker_execution", tampered)
        assert callback is None and reason
        renamed = dict(owner, id="other-plugin")
        callback, reason = manager.required_hook_callback("pre_worker_execution", renamed)
        assert callback is None and reason
    finally:
        plugins_module.REQUIRED_HOOK_CALLBACK_OWNERS.pop(loaded._on_pre_worker_execution, None)

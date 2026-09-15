"""Managed-gateway isolation for dispatcher-owned immutable Kanban workers."""

from __future__ import annotations

import json
import os
import signal
import shutil
import sqlite3
import subprocess
import sys
import sysconfig
import tempfile
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_runtime_generation as generations
from hermes_cli.kanban_runtime import process_start_time
from tests.hermes_cli.test_kanban_lifecycle_conformance import (
    _make_runtime_fixture,
    _prepare_fixture_generation,
    _wait_for_receipt,
)
from tests.hermes_cli.test_kanban_runtime_generation import (
    _install_memory_loaders,
    _write_memory_provider,
)


@pytest.fixture
def worker_setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / ".hermes"
    profile = root / "profiles" / "coder"
    profile.mkdir(parents=True)
    root.joinpath("config.yaml").write_text("{}\n", encoding="utf-8")
    profile.joinpath("config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(root))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_BIN", raising=False)
    monkeypatch.setenv("INVOCATION_ID", "managed-gateway-test")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr("tools.process_registry._is_supervised_gateway_process", lambda: False)
    source = tmp_path / "install"
    _make_runtime_fixture(source)
    prepared = []

    def prepare(_expected, *, workspace=None, profile_home=None, project_plugins_enabled=None):
        generation = _prepare_fixture_generation(
            source, workspace=workspace, profile_home=profile_home,
            project_plugins_enabled=project_plugins_enabled,
        )
        prepared.append(generation)
        return generation

    monkeypatch.setattr(generations, "prepare_runtime_generation", prepare)
    monkeypatch.setenv("HERMES_TEST_RUNTIME_RECEIPT", str(tmp_path / "receipt.json"))
    workspace = tmp_path / "candidate-worktree"
    workspace.mkdir()
    task = kb.Task(
        id="t_candidate_restart", title="activate candidate", body=None,
        assignee="coder", status="running", priority=0, created_by="test",
        created_at=1, started_at=1, completed_at=None,
        workspace_kind="worktree", workspace_path=str(workspace),
        claim_lock="host:dispatcher", claim_expires=999, tenant=None,
        branch_name="wt/t_candidate_restart", current_run_id=23,
    )
    with tempfile.TemporaryDirectory(prefix="runtime-storage-", dir=tmp_path) as storage:
        monkeypatch.setattr(generations, "_runtime_storage_root", lambda: Path(storage))
        try:
            yield workspace, task
        finally:
            for generation in prepared:
                generations.cleanup_runtime_generation(generation.root)


@pytest.fixture
def forbid_worker_spawn(monkeypatch: pytest.MonkeyPatch):
    real_popen = subprocess.Popen

    def guarded_popen(cmd, *args, **kwargs):
        # Runtime identity reads Git provenance before checking the worker scope.
        if cmd[:2] == ["git", "-C"] and cmd[3:] == ["rev-parse", "HEAD"]:
            return real_popen(cmd, *args, **kwargs)
        pytest.fail(f"unsafe worker spawn: {cmd!r}")

    monkeypatch.setattr(subprocess, "Popen", guarded_popen)


def _finish_launch(launch) -> None:
    if launch.cancel:
        launch.cancel()
    deadline = time.monotonic() + 5
    while kbd._pid_alive(launch.pid) and time.monotonic() < deadline:
        time.sleep(0.02)
    kbd.reap_worker_zombies()
    generations.sweep_runtime_generations()


def test_worker_generation_scopes_profile_board_and_secrets_before_grant(
    worker_setup, monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, task = worker_setup
    monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-cross-profile")
    monkeypatch.setattr("agent.secret_scope.is_multiplex_active", lambda: True)
    launch = kbd._default_spawn(task, str(workspace), board="review", defer_grant=True)
    receipt = workspace.parent / "receipt.json"
    generation = kbd._worker_runtime_snapshots[launch.launcher_pid]
    try:
        assert not receipt.exists()
        launch.grant(task.current_run_id, task.claim_lock)
        observed = _wait_for_receipt(receipt)
        assert observed["argv"][:2] == ["-p", "coder"]
        assert observed["argv"][-3:] == ["chat", "-q", f"work kanban task {task.id}"]
        assert observed["cwd"] == str(workspace)
        assert observed["profile"] == "coder"
        assert observed["home"] == str(workspace.parent / ".hermes" / "profiles" / "coder")
        assert observed["task"] == task.id
        assert observed["board"] == "review"
        assert observed["run"] == "23"
        assert observed["claim"] == task.claim_lock
        assert observed["secret"] is None
        assert all(Path(location).is_relative_to(generation) for location in observed["locations"])
    finally:
        _finish_launch(launch)
    assert not generation.exists()


@pytest.mark.parametrize("profile_change", ["mutate", "remove"])
def test_worker_captures_assigned_profile_plugins_before_child_imports(
    worker_setup, monkeypatch, profile_change,
):
    workspace, task = worker_setup
    source = workspace.parent / "install"
    dispatcher_home = workspace.parent / ".hermes" / "profiles" / "cto"
    worker_home = dispatcher_home.with_name(task.assignee)
    repository = Path(__file__).resolve().parents[2]
    (source / "providers").mkdir()
    for name in ("__init__.py", "base.py"):
        shutil.copy2(repository / "providers" / name, source / "providers" / name)
    (source / "hermes_constants.py").write_text(
        "import os\nfrom pathlib import Path\n"
        "def get_hermes_home(): return Path(os.environ['HERMES_HOME'])\n",
        encoding="utf-8",
    )
    for home, label in ((dispatcher_home, "dispatcher"), (worker_home, "assigned")):
        plugin = home / "plugins" / "model-providers" / "profile-fixture"
        plugin.mkdir(parents=True)
        (plugin / "__init__.py").write_text(
            "from providers import register_provider\n"
            "from providers.base import ProviderProfile\n"
            "class Fixture(ProviderProfile):\n"
            f"    def resolve_aux_model(self, *, vision=False): return {label!r}\n"
            "register_provider(Fixture(name='profile-fixture'))\n",
            encoding="utf-8",
        )
        (plugin / "resource.txt").write_text(f"{label} resource", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(dispatcher_home))
    # This module loads before the first bootstrap handshake, not just after the grant.
    (source / "fixture_early.py").write_text(
        """
import importlib
import importlib.resources
from providers import get_provider_profile
profile = get_provider_profile('profile-fixture')
plugin = importlib.import_module(type(profile).__module__)
resource = importlib.resources.files(plugin).joinpath('resource.txt')
value = {
    'provider': profile.resolve_aux_model(),
    'resource': resource.read_text(),
    'locations': [plugin.__file__, str(resource)],
}
""",
        encoding="utf-8",
    )
    restart_safe_argv = kbd._restart_safe_worker_argv

    def change_profile_before_spawn(task, command, *, preparation_id=None):
        command = restart_safe_argv(task, command, preparation_id=preparation_id)
        if profile_change == "remove":
            shutil.rmtree(worker_home)
        else:
            plugin = worker_home / "plugins" / "model-providers" / "profile-fixture"
            (plugin / "__init__.py").write_text("raise RuntimeError('mutable plugin')\n", encoding="utf-8")
            (plugin / "resource.txt").write_text("mutable resource", encoding="utf-8")
        return command

    monkeypatch.setattr(kbd, "_restart_safe_worker_argv", change_profile_before_spawn)
    launch = kbd._default_spawn(task, str(workspace), defer_grant=True)
    snapshot = kbd._worker_runtime_snapshots[launch.launcher_pid]
    try:
        launch.grant(task.current_run_id, task.claim_lock)
        observed = _wait_for_receipt(workspace.parent / "receipt.json")
        plugin = observed["early"][0]
        assert plugin["provider"] == "assigned"
        assert plugin["resource"] == "assigned resource"
        assert all(Path(path).is_relative_to(snapshot) for path in plugin["locations"])
        assert observed["home"] == str(worker_home)
        assert os.environ["HERMES_HOME"] == str(dispatcher_home)
    finally:
        _finish_launch(launch)


@pytest.mark.parametrize("source_change", ["mutate", "remove"])
def test_worker_captures_assigned_profile_memory_loaders_and_resources(
    worker_setup, monkeypatch, source_change,
):
    workspace, task = worker_setup
    source = workspace.parent / "install"
    _install_memory_loaders(source)
    dispatcher_home = workspace.parent / ".hermes" / "profiles" / "cto"
    worker_home = dispatcher_home.with_name(task.assignee)
    for home, label in ((dispatcher_home, "dispatcher"), (worker_home, "assigned")):
        _write_memory_provider(home / "plugins" / "profilememory", label)
    worker_home.joinpath("config.yaml").write_text("memory:\n  provider: inactive\n", encoding="utf-8")
    worker_home.joinpath("memory-state.txt").write_text("before capture", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(dispatcher_home))
    (source / "fixture_early.py").write_text("""
import argparse
import importlib
import importlib.resources
from plugins.memory import discover_plugin_cli_commands, find_provider_dir, load_memory_provider
from plugins.memory.config_schema import get_provider_config_schema
schema = get_provider_config_schema('profilememory')
command, = discover_plugin_cli_commands()
parser = argparse.ArgumentParser()
command['setup_fn'](parser)
provider = load_memory_provider('profilememory')
module = importlib.import_module(type(provider).__module__)
cli = importlib.import_module(module.__name__ + '.cli')
helper = importlib.import_module(module.__name__ + '.helper')
resource = importlib.resources.files(module).joinpath('resource.txt')
value = {
    'provider': provider.system_prompt_block(),
    'state': provider.prefetch('current state'),
    'schema': schema.label,
    'cli': parser.parse_args([]).memory_resource,
    'description': command['description'],
    'resource': resource.read_text(),
    'locations': [
        module.__file__, cli.__file__, helper.__file__, str(resource),
        str(find_provider_dir('profilememory') / 'config_schema.py'),
    ],
}
""", encoding="utf-8")
    restart_safe_argv = kbd._restart_safe_worker_argv

    def change_profile_before_spawn(task, command, *, preparation_id=None):
        command = restart_safe_argv(task, command, preparation_id=preparation_id)
        if source_change == "remove":
            shutil.rmtree(worker_home / "plugins")
        else:
            plugin = worker_home / "plugins" / "profilememory"
            for name in ("__init__.py", "helper.py", "cli.py", "config_schema.py"):
                (plugin / name).write_text("raise RuntimeError('live memory code')\n", encoding="utf-8")
            (plugin / "resource.txt").write_text("live memory resource", encoding="utf-8")
            (plugin / "plugin.yaml").write_text("description: live manifest\n", encoding="utf-8")
        # Config and provider data remain live even though executable resources do not.
        worker_home.joinpath("config.yaml").write_text("memory:\n  provider: profilememory\n", encoding="utf-8")
        worker_home.joinpath("memory-state.txt").write_text("after capture", encoding="utf-8")
        return command

    monkeypatch.setattr(kbd, "_restart_safe_worker_argv", change_profile_before_spawn)
    launch = kbd._default_spawn(task, str(workspace), defer_grant=True)
    snapshot = kbd._worker_runtime_snapshots[launch.launcher_pid]
    try:
        launch.grant(task.current_run_id, task.claim_lock)
        observed = _wait_for_receipt(workspace.parent / "receipt.json")
        memory = observed["early"][0]
        assert memory["provider"] == "assigned:assigned resource"
        assert memory["resource"] == memory["schema"] == memory["cli"] == "assigned resource"
        assert memory["description"] == "assigned memory"
        assert memory["state"] == "after capture"
        assert all(Path(path).is_relative_to(snapshot) for path in memory["locations"])
        assert observed["home"] == str(worker_home)
        assert os.environ["HERMES_HOME"] == str(dispatcher_home)
    finally:
        _finish_launch(launch)


@pytest.mark.parametrize(("dispatcher_gate", "profile_env", "enabled"), [
    ("0", "HERMES_ENABLE_PROJECT_PLUGINS=1\n", True),
    ("1", "HERMES_ENABLE_PROJECT_PLUGINS=0\n", False),
    ("0", "WORKER_PROJECT_GATE=on\nHERMES_ENABLE_PROJECT_PLUGINS=${WORKER_PROJECT_GATE}\n", True),
    ("1", "WORKER_PROJECT_GATE\nHERMES_ENABLE_PROJECT_PLUGINS=${WORKER_PROJECT_GATE:-1}\n", False),
])
def test_worker_project_plugin_capture_matches_profile_dotenv(
    worker_setup, monkeypatch, dispatcher_gate, profile_env, enabled,
):
    workspace, task = worker_setup
    source = workspace.parent / "install"
    _install_memory_loaders(source)
    worker_home = workspace.parent / ".hermes" / "profiles" / task.assignee
    worker_home.joinpath(".env").write_text(profile_env, encoding="utf-8")
    monkeypatch.setenv("HERMES_ENABLE_PROJECT_PLUGINS", dispatcher_gate)
    monkeypatch.delenv("WORKER_PROJECT_GATE", raising=False)
    plugin = workspace / ".hermes" / "plugins" / "projectmemory"
    _write_memory_provider(plugin, "project")
    (source / "fixture_early.py").write_text("""
import importlib
from pathlib import Path
from hermes_cli.env_loader import load_hermes_dotenv
from hermes_cli.kanban_runtime import RuntimeIdentityError
from hermes_cli.kanban_runtime_generation import generation_runtime_path
from plugins.memory import find_provider_dir, list_memory_provider_names, load_memory_provider
from utils import env_var_enabled
gate_before = env_var_enabled('HERMES_ENABLE_PROJECT_PLUGINS')
discovered_before = 'projectmemory' in list_memory_provider_names()
load_hermes_dotenv(load_external_secrets=False)
try:
    generation_runtime_path(Path.cwd() / '.hermes' / 'plugins')
    captured = True
except RuntimeIdentityError:
    captured = False
directory = find_provider_dir('projectmemory')
provider = load_memory_provider('projectmemory', register_skills=False) if directory else None
module = importlib.import_module(type(provider).__module__) if provider else None
value = {
    'captured': captured,
    'gate_before': gate_before,
    'discovered_before': discovered_before,
    'discovered': 'projectmemory' in list_memory_provider_names(),
    'provider': provider.system_prompt_block() if provider else None,
    'origin': module.__file__ if module else None,
    'gate_after': env_var_enabled('HERMES_ENABLE_PROJECT_PLUGINS'),
}
""", encoding="utf-8")
    restart_safe_argv = kbd._restart_safe_worker_argv

    def change_project_before_spawn(task, command, *, preparation_id=None):
        command = restart_safe_argv(task, command, preparation_id=preparation_id)
        if enabled:
            shutil.rmtree(plugin.parent)
        else:
            # A disabled worker must not discover even a still-present plugin.
            (plugin / "__init__.py").write_text("raise RuntimeError('disabled plugin imported')\n", encoding="utf-8")
        return command

    monkeypatch.setattr(kbd, "_restart_safe_worker_argv", change_project_before_spawn)
    launch = kbd._default_spawn(task, str(workspace), defer_grant=True)
    snapshot = kbd._worker_runtime_snapshots[launch.launcher_pid]
    try:
        launch.grant(task.current_run_id, task.claim_lock)
        plugin = _wait_for_receipt(workspace.parent / "receipt.json")["early"][0]
        assert plugin["discovered"] is enabled
        assert plugin["discovered_before"] is enabled
        assert plugin["captured"] is enabled
        assert plugin["gate_before"] is plugin["gate_after"] is enabled
        if enabled:
            assert plugin["provider"] == "project:project resource"
            assert Path(plugin["origin"]).is_relative_to(snapshot)
        else:
            assert plugin["provider"] is None
            assert plugin["origin"] is None
        assert os.environ["HERMES_ENABLE_PROJECT_PLUGINS"] == dispatcher_gate
        assert "WORKER_PROJECT_GATE" not in os.environ
    finally:
        _finish_launch(launch)


@pytest.mark.linux_only
def test_managed_gateway_worker_spawn_fails_closed_without_scope(
    worker_setup, monkeypatch, forbid_worker_spawn,
):
    workspace, task = worker_setup
    monkeypatch.setattr("tools.process_registry._is_supervised_gateway_process", lambda: True)
    monkeypatch.setattr("tools.process_registry._systemd_run_user_scope_available", lambda: False)
    with pytest.raises(RuntimeError, match="restart-safe systemd scope"):
        kbd._default_spawn(task, str(workspace))


@pytest.mark.linux_only
def test_managed_gateway_scope_builder_fails_closed_if_binary_disappears(
    worker_setup, monkeypatch, forbid_worker_spawn,
):
    workspace, task = worker_setup
    monkeypatch.setattr("tools.process_registry._is_supervised_gateway_process", lambda: True)
    monkeypatch.setattr("tools.process_registry._systemd_run_user_scope_available", lambda: True)
    monkeypatch.setattr("shutil.which", lambda _name: None)
    with pytest.raises(RuntimeError, match="restart-safe systemd scope"):
        kbd._default_spawn(task, str(workspace))


def test_generation_ignores_workspace_and_pythonpath_before_any_import(worker_setup, monkeypatch):
    workspace, task = worker_setup
    marker = workspace / "untrusted-import"
    for name in ("fixture_early", "fixture_lazy", "third_party_early", "third_party_dynamic"):
        (workspace / f"{name}.py").write_text(
            f"from pathlib import Path\nPath({str(marker)!r}).touch()\nraise SystemExit(42)\n",
            encoding="utf-8",
        )
    monkeypatch.setenv("PYTHONPATH", str(workspace))
    launch = kbd._default_spawn(task, str(workspace), defer_grant=True)
    try:
        launch.grant(task.current_run_id, task.claim_lock)
        observed = _wait_for_receipt(workspace.parent / "receipt.json")
        assert observed["early"] == [1, 1]
        assert observed["lazy"] == [1, 1]
        assert not marker.exists()
    finally:
        _finish_launch(launch)


def test_launcher_exit_does_not_delete_live_worker_generation(worker_setup, monkeypatch):
    workspace, task = worker_setup
    launch = kbd._default_spawn(task, str(workspace), defer_grant=True)
    generation = kbd._worker_runtime_snapshots[launch.launcher_pid]
    launcher = subprocess.Popen([sys.executable, "-c", "pass"])
    launcher.wait(timeout=10)
    kbd._worker_pid_aliases[launcher.pid] = launch.pid
    kbd._worker_runtime_snapshots[launcher.pid] = generation
    try:
        kbd._record_worker_exit(launcher.pid, 0)
        assert kbd._pid_alive(launch.pid)
        assert generation.exists()
        generations.sweep_runtime_generations()
        assert generation.exists()
    finally:
        _finish_launch(launch)
        kbd._recent_worker_exits.pop(launch.pid, None)
    assert not generation.exists()


def test_explicit_current_install_entrypoint_is_sealed_and_custom_wrapper_is_refused(worker_setup, monkeypatch):
    workspace, task = worker_setup
    entrypoint = Path(sysconfig.get_path("scripts")) / ("hermes.exe" if os.name == "nt" else "hermes")
    monkeypatch.setenv("HERMES_BIN", str(entrypoint))
    launch = kbd._default_spawn(task, str(workspace), defer_grant=True)
    try:
        launch.grant(task.current_run_id, task.claim_lock)
        observed = _wait_for_receipt(workspace.parent / "receipt.json")
        assert observed["early"] == [1, 1]
        generation = kbd._worker_runtime_snapshots[launch.launcher_pid]
        assert all(Path(location).is_relative_to(generation) for location in observed["locations"])
    finally:
        _finish_launch(launch)
    wrapper = workspace / "hermes-custom"
    wrapper.write_text("#!/bin/sh\nexit 42\n", encoding="utf-8")
    wrapper.chmod(0o755)
    monkeypatch.setenv("HERMES_BIN", str(wrapper))
    with pytest.raises(RuntimeError, match="HERMES_BIN.*this installation"):
        kbd._default_spawn(task, str(workspace), defer_grant=True)


@pytest.mark.linux_only
def test_real_user_systemd_scope_preserves_worker_context(worker_setup, monkeypatch):
    from tools import process_registry

    if not process_registry._systemd_run_user_scope_available():
        pytest.skip("systemd-run --user --scope is unavailable on this host")
    workspace, task = worker_setup
    monkeypatch.setattr(process_registry, "_is_supervised_gateway_process", lambda: True)
    launch = kbd._default_spawn(task, str(workspace), defer_grant=True)
    try:
        launch.grant(task.current_run_id, task.claim_lock)
        observed = _wait_for_receipt(workspace.parent / "receipt.json")
        assert observed["identity"]["pid"] == launch.pid
        assert observed["cwd"] == str(workspace)
        assert observed["task"] == task.id
        assert observed["run"] == "23"
        assert ".scope" in observed["cgroup"]
        assert "hermes-gateway.service" not in observed["cgroup"]
    finally:
        _finish_launch(launch)


@pytest.mark.linux_only
@pytest.mark.live_system_guard_bypass  # cleanup signals our start-time-verified, reparented worker
def test_worker_and_claim_survive_dispatcher_exit_and_shared_install_replacement(worker_setup, monkeypatch):
    workspace, _task = worker_setup
    base = workspace.parent
    database = base / "restart.db"
    conn = sqlite3.connect(database)
    conn.row_factory = sqlite3.Row
    conn.executescript(kb.SCHEMA_SQL)
    kb._ensure_lifecycle_schema(conn)
    kb._ensure_goal_revision_schema(conn)
    task_id = kb.create_task(
        conn, title="restart survivor", assignee="coder", initial_status="blocked",
        workspace_kind="scratch", workspace_path=str(workspace),
    )
    promoted, reason = kb.promote_task(conn, task_id, actor="test")
    assert promoted, reason
    assert kb.get_task(conn, task_id).status == "ready"
    release = base / "release"
    monkeypatch.setenv("HERMES_TEST_RUNTIME_RELEASE", str(release))
    script = """
import json, os, sqlite3, sys
from pathlib import Path
from unittest.mock import patch
from hermes_cli import kanban_runtime_generation as generations
generations._runtime_storage_root = lambda: Path(sys.argv[2])
from hermes_cli import kanban_db_dispatch as kbd
from tests.hermes_cli.test_kanban_lifecycle_conformance import _prepare_fixture_generation
base = Path(sys.argv[1])
conn = sqlite3.connect(base / 'restart.db')
conn.row_factory = sqlite3.Row
with patch.object(kbd, '_profile_exists_fn', return_value=None), patch.object(
    kbd, '_restart_safe_worker_argv', side_effect=lambda task, command, preparation_id=None: command,
), patch.object(generations, 'prepare_runtime_generation', side_effect=lambda expected, workspace=None, profile_home=None, project_plugins_enabled=None: _prepare_fixture_generation(base / 'install', workspace=workspace, profile_home=profile_home, project_plugins_enabled=project_plugins_enabled)):
    result = kbd.dispatch_once(conn, max_spawn=1, reconcile_orphans=False)
assert len(result.spawned) == 1, result
conn.close()
os._exit(0)
"""
    launch_env = {**os.environ, "PYTHONPATH": str(Path(kbd.__file__).resolve().parents[1])}
    completed = subprocess.run(
        [sys.executable, "-c", script, str(base), str(generations._runtime_storage_root())],
        env=launch_env, timeout=30,
    )
    assert completed.returncode == 0
    claimed = kb.get_task(conn, task_id)
    assert claimed.status == "running"
    identity = json.loads(conn.execute(
        "SELECT metadata FROM task_runs WHERE id = ?", (claimed.current_run_id,),
    ).fetchone()["metadata"])["runtime_identity"]
    try:
        (base / "install" / "fixture_lazy.py").write_text("value = 2\n", encoding="utf-8")
        (base / "site-packages" / "third_party_dynamic" / "__init__.py").write_text("value = 2\n", encoding="utf-8")
        generations.sweep_runtime_generations()
        assert kbd.detect_crashed_workers(conn) == []
        assert kbd.reconcile_orphaned_running(conn) == []
        recovered = kb.get_task(conn, task_id)
        assert recovered.current_run_id == claimed.current_run_id
        assert recovered.claim_lock == claimed.claim_lock
        release.touch()
        observed = _wait_for_receipt(base / "receipt.json")
        assert observed["lazy"] == [1, 1]
        assert observed["run"] == str(claimed.current_run_id)
        assert observed["claim"] == claimed.claim_lock
        assert observed["identity"] == identity
    finally:
        if kbd._pid_alive(claimed.worker_pid) and process_start_time(claimed.worker_pid) == identity["start_time"]:
            try:
                os.kill(claimed.worker_pid, signal.SIGTERM)
            except ProcessLookupError:
                pass  # The released worker can exit between the identity check and signal.
        deadline = time.monotonic() + 5
        while kbd._pid_alive(claimed.worker_pid) and time.monotonic() < deadline:
            time.sleep(0.02)
        conn.close()
        generations.sweep_runtime_generations()

"""Managed-gateway isolation for dispatcher-owned immutable Kanban workers."""

from __future__ import annotations

import codecs
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
import uuid
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_runtime_generation as generations
from hermes_cli.kanban_runtime import process_start_time
from tests.hermes_cli import kanban_conformance_fixture as MODULE
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
    monkeypatch.delenv("INVOCATION_ID", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr("tools.process_registry._is_supervised_gateway_process", lambda: False)
    source = tmp_path / "install"
    MODULE._make_runtime_fixture(source)
    prepared = []

    def prepare(_expected, *, workspace=None, profile_home=None, project_plugins_enabled=None):
        generation = MODULE._prepare_fixture_generation(
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
    # The one process a failing launch is allowed to run: a bounded, read-only
    # user-manager state query. It launches nothing and is the evidence the
    # shutdown-race rule requires; every other subprocess means the launch path
    # did not fail closed.
    manager_state_probe = ("systemctl", "--user", "is-system-running")

    def guarded_popen(cmd, *args, **kwargs):
        # Runtime identity reads Git provenance before checking the worker scope.
        if cmd[:2] == ["git", "-C"] and cmd[3:] == ["rev-parse", "HEAD"]:
            return real_popen(cmd, *args, **kwargs)
        if tuple(cmd) == manager_state_probe:
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
        observed = MODULE._wait_for_receipt(receipt)
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
    shutil.copy2(repository / "hermes_constants.py", source / "hermes_constants.py")
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
        observed = MODULE._wait_for_receipt(workspace.parent / "receipt.json")
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
        observed = MODULE._wait_for_receipt(workspace.parent / "receipt.json")
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


@pytest.mark.parametrize(("dispatcher_gate", "scope", "encoding", "profile_env", "enabled"), [
    ("0", "profile", "utf-8", "HERMES_ENABLE_PROJECT_PLUGINS=1\n", True),
    ("1", "profile", "utf-8", "HERMES_ENABLE_PROJECT_PLUGINS=0\n", False),
    ("0", "profile", "utf-8", "WORKER_PROJECT_GATE=on\nHERMES_ENABLE_PROJECT_PLUGINS=${WORKER_PROJECT_GATE}\n", True),
    ("1", "profile", "utf-8", "WORKER_PROJECT_GATE\nHERMES_ENABLE_PROJECT_PLUGINS=${WORKER_PROJECT_GATE:-1}\n", False),
    ("1", "profile", "utf-8", "HERMES_ENABLE_PROJECT_PLUGINS\n", True),
    (None, "profile", "utf-8", "HERMES_ENABLE_PROJECT_PLUGINS\n", False),
    ("0", "profile", "utf-16", "HERMES_ENABLE_PROJECT_PLUGINS=1\n", True),
    ("1", "profile", "utf-16-be", "HERMES_ENABLE_PROJECT_PLUGINS=0\n", False),
    ("0", "profile", "utf-16-le", "WORKER_PROJECT_GATE=on\nHERMES_ENABLE_PROJECT_PLUGINS=${WORKER_PROJECT_GATE}\n", True),
    ("1", "profile", "utf-16-le", "HERMES_ENABLE_PROJECT_PLUGINS=0\n", False),
    ("0", "managed", "utf-16-be", "HERMES_ENABLE_PROJECT_PLUGINS=1\n", True),
    ("1", "managed", "utf-16", "HERMES_ENABLE_PROJECT_PLUGINS=0\n", False),
    ("0", "managed", "utf-16-le", "HERMES_ENABLE_PROJECT_PLUGINS=1\n", True),
    ("1", "managed", "utf-16-le", "HERMES_ENABLE_PROJECT_PLUGINS=0\n", False),
    ("0", "managed", "utf-8-sig", "HERMES_ENABLE_PROJECT_PLUGINS=1\n", True),
])
def test_worker_project_plugin_capture_matches_profile_dotenv(
    worker_setup, monkeypatch, dispatcher_gate, scope, encoding, profile_env, enabled,
):
    workspace, task = worker_setup
    source = workspace.parent / "install"
    _install_memory_loaders(source)
    worker_home = workspace.parent / ".hermes" / "profiles" / task.assignee
    managed = workspace.parent / "managed"
    managed.mkdir()
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    if scope == "managed":
        worker_home.joinpath(".env").write_text(
            f"HERMES_ENABLE_PROJECT_PLUGINS={dispatcher_gate}\n", encoding="utf-8",
        )
    env_file = (worker_home if scope == "profile" else managed) / ".env"
    secret = "dotenv-parity-secret-must-not-be-captured"
    content = profile_env + f"DOTENV_PARITY_SECRET={secret}\n"
    raw = content.encode(encoding)  # utf-16-le without a BOM exercises NUL-padded ASCII.
    if encoding == "utf-16-be":
        raw = codecs.BOM_UTF16_BE + raw
    env_file.write_bytes(raw)
    originals = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in (worker_home / ".env", managed / ".env") if path.exists()
    }
    if dispatcher_gate is None:
        monkeypatch.delenv("HERMES_ENABLE_PROJECT_PLUGINS", raising=False)
    else:
        monkeypatch.setenv("HERMES_ENABLE_PROJECT_PLUGINS", dispatcher_gate)
    monkeypatch.delenv("DOTENV_PARITY_SECRET", raising=False)
    monkeypatch.delenv("WORKER_PROJECT_GATE", raising=False)
    plugin = workspace / ".hermes" / "plugins" / "projectmemory"
    _write_memory_provider(plugin, "project")
    (source / "fixture_early.py").write_text("""
import importlib
import os
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
    'secret_loaded': bool(os.getenv('DOTENV_PARITY_SECRET')),
}
""", encoding="utf-8")
    restart_safe_argv = kbd._restart_safe_worker_argv

    def change_project_before_spawn(task, command, *, preparation_id=None):
        command = restart_safe_argv(task, command, preparation_id=preparation_id)
        # This hook runs after capture but before any child can normalize the live files.
        assert dict(os.environ) == ambient
        for path, original in originals.items():
            assert (path.read_bytes(), path.stat().st_mtime_ns) == original
        if enabled:
            shutil.rmtree(plugin.parent)
        else:
            # A disabled worker must not discover even a still-present plugin.
            (plugin / "__init__.py").write_text("raise RuntimeError('disabled plugin imported')\n", encoding="utf-8")
        return command

    monkeypatch.setattr(kbd, "_restart_safe_worker_argv", change_project_before_spawn)
    ambient = dict(os.environ)
    launch = kbd._default_spawn(task, str(workspace), defer_grant=True)
    snapshot = kbd._worker_runtime_snapshots[launch.launcher_pid]
    try:
        launch.grant(task.current_run_id, task.claim_lock)
        plugin = MODULE._wait_for_receipt(workspace.parent / "receipt.json")["early"][0]
        assert plugin["discovered"] is enabled
        assert plugin["discovered_before"] is enabled
        assert plugin["captured"] is enabled
        assert plugin["gate_before"] is plugin["gate_after"] is enabled
        assert plugin["secret_loaded"] is True
        assert not any(path.name in {".env", ".op.env"} for path in snapshot.rglob("*"))
        assert all(
            secret.encode() not in path.read_bytes()
            for path in snapshot.rglob("*") if path.is_file()
        )
        if enabled:
            assert plugin["provider"] == "project:project resource"
            assert Path(plugin["origin"]).is_relative_to(snapshot)
        else:
            assert plugin["provider"] is None
            assert plugin["origin"] is None
        assert dict(os.environ) == ambient
    finally:
        _finish_launch(launch)


@pytest.mark.platforms("linux")
def test_managed_gateway_worker_spawn_fails_closed_without_scope(
    worker_setup, monkeypatch, forbid_worker_spawn,
):
    workspace, task = worker_setup
    monkeypatch.setenv("INVOCATION_ID", "managed-gateway-test")
    monkeypatch.setattr("tools.process_registry._is_supervised_gateway_process", lambda: True)
    monkeypatch.setattr("tools.process_registry._systemd_run_user_scope_available", lambda: False)
    with pytest.raises(RuntimeError, match="restart-safe systemd scope"):
        kbd._default_spawn(task, str(workspace))


@pytest.mark.platforms("linux")
def test_managed_gateway_scope_builder_fails_closed_if_binary_disappears(
    worker_setup, monkeypatch, forbid_worker_spawn,
):
    workspace, task = worker_setup
    monkeypatch.setenv("INVOCATION_ID", "managed-gateway-test")
    monkeypatch.setattr("tools.process_registry._is_supervised_gateway_process", lambda: True)
    monkeypatch.setattr("tools.process_registry._systemd_run_user_scope_available", lambda: True)
    monkeypatch.setattr("tools.process_registry._slice_inherit_supported", lambda: True)
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
        observed = MODULE._wait_for_receipt(workspace.parent / "receipt.json")
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
        observed = MODULE._wait_for_receipt(workspace.parent / "receipt.json")
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


@pytest.mark.platforms("linux")
def test_oneshot_unit_dispatcher_scope_wraps_or_warns_never_dooms_silently(
    worker_setup, monkeypatch, caplog,
):
    from tools import process_registry

    _, task = worker_setup
    command = [sys.executable, "-m", "hermes_cli.main"]
    monkeypatch.setenv("INVOCATION_ID", "oneshot-dispatch-timer")
    monkeypatch.setattr(process_registry, "_is_supervised_gateway_process", lambda: False)
    monkeypatch.setattr(process_registry, "_systemd_run_user_scope_available", lambda: True)
    monkeypatch.setattr(process_registry, "_slice_inherit_supported", lambda: True)
    monkeypatch.setattr(
        process_registry, "_build_systemd_scope_argv",
        lambda cmd, unit_suffix: ["systemd-run", "--user", "--scope", "--unit", f"hermes-worker-{unit_suffix}", *cmd],
    )
    wrapped = kbd._restart_safe_worker_argv(task, command)
    assert wrapped[:3] == ["systemd-run", "--user", "--scope"]
    assert f"hermes-worker-kanban-{task.id}-run-{task.current_run_id}" in wrapped
    monkeypatch.setattr(process_registry, "_systemd_run_user_scope_available", lambda: False)
    monkeypatch.setattr(process_registry, "_scope_degraded_warned", False)
    with caplog.at_level("WARNING", logger=process_registry.logger.name):
        assert kbd._restart_safe_worker_argv(task, command) == command
    warned = [r.getMessage() for r in caplog.records if "KILLED when the unit exits" in r.getMessage()]
    assert len(warned) == 1 and "KillMode=process" in warned[0]
    assert process_registry.restart_safe_gateway_child_argv(
        ["hermes", "cron"], unit_suffix="cron-job-1", require_restart_safe_scope=True,
    ).mode == "in_process"


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("slice_inherit", [False, True])
def test_real_user_systemd_scope_preserves_worker_context(worker_setup, monkeypatch, slice_inherit):
    from tools import process_registry

    if not process_registry._systemd_run_user_scope_available():
        pytest.skip("systemd-run --user --scope is unavailable on this host")
    if slice_inherit and not process_registry._slice_inherit_supported():
        pytest.skip("systemd-run --slice-inherit is unavailable on this host")
    monkeypatch.setattr(process_registry, "_slice_inherit_supported", lambda: slice_inherit)
    workspace, task = worker_setup
    monkeypatch.setattr(process_registry, "_is_supervised_gateway_process", lambda: True)
    monkeypatch.setenv("INVOCATION_ID", "managed-gateway-test")
    launch = kbd._default_spawn(task, str(workspace), defer_grant=True)
    try:
        launch.grant(task.current_run_id, task.claim_lock)
        observed = MODULE._wait_for_receipt(workspace.parent / "receipt.json")
        assert observed["identity"]["pid"] == launch.pid
        assert observed["cwd"] == str(workspace)
        assert observed["task"] == task.id
        assert observed["run"] == "23"
        assert ".scope" in observed["cgroup"]
        assert "hermes-gateway.service" not in observed["cgroup"]
    finally:
        _finish_launch(launch)


@pytest.mark.platforms("linux")
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
from tests.hermes_cli.kanban_conformance_fixture import _prepare_fixture_generation
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
    # This child also materializes the sealed interpreter; a cold copy can
    # consume the old 30-second deadline before the dispatch being exercised.
    completed = subprocess.run(
        [sys.executable, "-c", script, str(base), str(generations._runtime_storage_root())],
        env=launch_env, timeout=60,
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
        observed = MODULE._wait_for_receipt(base / "receipt.json")
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


def _unit_property(unit: str, name: str) -> str:
    completed = subprocess.run(
        ["systemctl", "--user", "show", unit, f"--property={name}"],
        check=True, capture_output=True, text=True, timeout=15,
    )
    return completed.stdout.strip().partition("=")[2]


def _proc_cgroup(pid: int) -> str:
    for line in Path(f"/proc/{pid}/cgroup").read_text(encoding="utf-8").splitlines():
        if line.startswith("0::"):
            return line[3:]
    raise AssertionError(f"PID {pid} has no unified cgroup")


@pytest.mark.platforms("linux")
@pytest.mark.live_system_guard_bypass  # cleanup signals our start-time-verified worker
def test_worker_survives_a_real_user_service_restart(tmp_path, monkeypatch):
    """A REAL ``systemctl --user restart`` must not take the granted worker down.

    ``tests/cron/test_restart_safe_worker.py::
    test_managed_gateway_restart_preserves_active_worker_and_single_side_effect``
    SIGTERMs a harness subprocess; it never restarts a service, so it cannot
    establish restart survival. This test starts a uniquely named transient user
    service in the same slice the gateway occupies, has that service launch its
    child through the real restart-safe scope, restarts the service for real, and
    then observes the worker itself: same PID, same start time, same cgroup, and
    a cgroup that is a SIBLING of the service inside the shared slice. Finally it
    releases the worker and requires exactly one side effect.
    """
    from tools import process_registry

    if not process_registry._systemd_run_user_scope_available():
        pytest.skip("systemd-run --user --scope is unavailable on this host")
    bus_env = process_registry.systemd_user_bus_env(os.environ)
    for key in ("XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS"):
        if key in bus_env:
            monkeypatch.setenv(key, bus_env[key])
    if not process_registry._slice_inherit_supported():
        pytest.skip("systemd-run --slice-inherit is required for shared-slice placement")

    base = tmp_path / "service-smoke"
    base.mkdir()
    started = base / "started"
    release = base / "release"
    side_effect = base / "side-effect"
    info = base / "worker.json"
    worker_py = base / "worker.py"
    worker_py.write_text(
        "import pathlib, time\n"
        f"started = pathlib.Path({str(started)!r})\n"
        f"release = pathlib.Path({str(release)!r})\n"
        f"side_effect = pathlib.Path({str(side_effect)!r})\n"
        "started.write_text('started')\n"
        "deadline = time.monotonic() + 60\n"
        "while not release.exists() and time.monotonic() < deadline:\n"
        "    time.sleep(0.05)\n"
        "if not release.exists():\n"
        "    raise SystemExit('release timeout')\n"
        "with side_effect.open('a') as handle:\n"
        "    handle.write('once\\n')\n",
        encoding="utf-8",
    )
    suffix = f"issue43-smoke-{uuid.uuid4().hex[:8]}"
    unit = f"{suffix}.service"
    harness_py = base / "harness.py"
    harness_py.write_text(
        "import json, os, pathlib, subprocess, sys, time\n"
        f"sys.path.insert(0, {str(Path(kbd.__file__).resolve().parents[1])!r})\n"
        f"os.environ['HERMES_HOME'] = {str(base / 'home')!r}\n"
        f"os.environ['HOME'] = {str(base)!r}\n"
        "os.environ.setdefault('INVOCATION_ID', 'issue43-smoke')\n"
        "from tools import process_registry\n"
        "process_registry._is_supervised_gateway_process = lambda: True\n"
        "dispatch = process_registry.restart_safe_gateway_child_argv(\n"
        f"    [sys.executable, '-B', {str(worker_py)!r}], unit_suffix={suffix!r},\n"
        "    require_restart_safe_scope=True, outlives_parent=True)\n"
        "proc = subprocess.Popen(dispatch.argv, start_new_session=True,\n"
        "    env=process_registry.systemd_user_bus_env(os.environ))\n"
        "# ``systemd-run --scope`` moves the command into its scope asynchronously,\n"
        "# so read the cgroup only once it has settled, or we record the service's.\n"
        "cgroup = ''\n"
        "deadline = time.monotonic() + 10\n"
        "while time.monotonic() < deadline:\n"
        "    cgroup = pathlib.Path('/proc/%d/cgroup' % proc.pid).read_text().strip()\n"
        "    cgroup = cgroup.splitlines()[-1][3:]\n"
        "    if cgroup.endswith('.scope'):\n"
        "        break\n"
        "    time.sleep(0.02)\n"
        f"pathlib.Path({str(info)!r}).write_text(\n"
        "    json.dumps({'pid': proc.pid, 'cgroup': cgroup, 'launcher_pid': os.getpid()}))\n"
        "while True:\n"
        "    time.sleep(0.2)\n",
        encoding="utf-8",
    )
    worker_pid = None
    worker_start_time = None
    try:
        launch = subprocess.run(
            # Same slice the CTO gateway occupies: the worker must stay inside
            # that shared budget, not escape to app.slice.
            ["systemd-run", "--user", "--unit", unit,
             "--slice=agents-controls.slice", "--collect",
             sys.executable, "-B", str(harness_py)],
            capture_output=True, text=True, timeout=30,
        )
        assert launch.returncode == 0, launch.stderr
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and not (info.is_file() and started.is_file()):
            time.sleep(0.05)
        assert info.is_file(), "the launcher service never recorded its worker"
        record = json.loads(info.read_text(encoding="utf-8"))
        worker_pid = int(record["pid"])
        worker_cgroup = str(record["cgroup"])
        worker_start_time = process_start_time(worker_pid)
        assert worker_cgroup.endswith(f"hermes-worker-{suffix}.scope"), worker_cgroup

        service_cgroup = _unit_property(unit, "ControlGroup")
        launcher_before = _unit_property(unit, "MainPID")
        assert service_cgroup, "the fixture service has no cgroup"
        budget = service_cgroup.rsplit("/", 1)[0]
        assert worker_cgroup.startswith(budget + "/"), (
            f"worker {worker_cgroup} left the launcher's shared slice {budget}"
        )
        assert not worker_cgroup.startswith(service_cgroup + "/"), (
            "the worker must be a sibling of the launcher service, not inside it"
        )
        assert record["launcher_pid"] != worker_pid

        # Claim the card BEFORE the restart so the assertion below is about a
        # claim that genuinely spans it, not a claim created afterwards.
        monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
        conn = sqlite3.connect(base / "restart.db")
        conn.row_factory = sqlite3.Row
        conn.executescript(kb.SCHEMA_SQL)
        kb._ensure_lifecycle_schema(conn)
        kb._ensure_goal_revision_schema(conn)
        task_id = kb.create_task(
            conn, title="granted worker", assignee="coder", initial_status="blocked",
            workspace_kind="scratch", workspace_path=str(base),
        )
        promoted, reason = kb.promote_task(conn, task_id, actor="test")
        assert promoted, reason
        claimed = kb.claim_task(
            conn, task_id, claimer=f"{kb._claimer_id().split(':', 1)[0]}:smoke",
        )
        assert claimed is not None
        kbd._set_worker_pid(conn, task_id, worker_pid)
        run_id_before = claimed.current_run_id
        claim_lock_before = claimed.claim_lock

        subprocess.run(
            ["systemctl", "--user", "restart", unit],
            check=True, capture_output=True, text=True, timeout=30,
        )
        assert _unit_property(unit, "MainPID") != launcher_before, "service did not restart"
        assert kbd._pid_alive(worker_pid), "the service restart killed the worker"
        assert process_start_time(worker_pid) == worker_start_time
        assert _proc_cgroup(worker_pid) == worker_cgroup

        # The claim that existed across the restart is intact, and a replacement
        # dispatcher leaves a worker the restart did not kill alone.
        survived = kb.get_task(conn, task_id)
        assert survived.status == "running"
        assert survived.current_run_id == run_id_before
        assert survived.claim_lock == claim_lock_before
        assert survived.worker_pid == worker_pid
        result = kbd.DispatchResult()
        kbd._run_reclaim_phase(
            conn, result, stale_timeout_seconds=0, failure_limit=3,
            reconcile_orphans=True,
        )
        assert (result.interrupted, result.crashed, result.reclaimed) == ([], [], 0)
        assert kb.get_task(conn, task_id).status == "running"
        conn.close()

        release.write_text("go", encoding="utf-8")
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and not side_effect.is_file():
            time.sleep(0.05)
        assert side_effect.read_text(encoding="utf-8").splitlines() == ["once"]
    finally:
        subprocess.run(
            ["systemctl", "--user", "stop", unit],
            check=False, capture_output=True, text=True, timeout=30,
        )
        subprocess.run(
            ["systemctl", "--user", "stop", f"hermes-worker-{suffix}.scope"],
            check=False, capture_output=True, text=True, timeout=30,
        )
        if worker_pid is not None:
            deadline = time.monotonic() + 5
            while kbd._pid_alive(worker_pid) and time.monotonic() < deadline:
                time.sleep(0.05)
            if kbd._pid_alive(worker_pid) and process_start_time(worker_pid) == worker_start_time:
                try:
                    os.kill(worker_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

"""Regression coverage for immutable runtime roots and profile-state churn."""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from hermes_cli import kanban_runtime as runtime
from hermes_cli import kanban_runtime_generation as generation
from scripts.verify_kanban_lifecycle import _dispatcher_materialization_check
from tests.hermes_cli.test_kanban_runtime_generation import installation, runtime_storage


REPOSITORY = Path(__file__).resolve().parents[2]


def _run_generation_worker(
    prepared, *, cwd: Path, env: dict[str, str] | None = None, args: tuple[str, ...] = (),
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [*prepared.command_prefix, *args],
        cwd=cwd,
        env={**os.environ, **prepared.env, **(env or {})},
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )


def _write_plugin(root: Path, module: str, label: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{module}.py").write_text(
        f"VALUE = {label!r}\n"
        "from pathlib import Path\n"
        "RESOURCE = Path(__file__).with_name('resource.txt').read_text()\n",
        encoding="utf-8",
    )
    (root / "resource.txt").write_text(f"{module}-sealed\n", encoding="utf-8")

@pytest.mark.live_system_guard_bypass
def test_dispatcher_materialization_ignores_churning_profile_state(tmp_path: Path) -> None:
    identity = runtime.runtime_identity(REPOSITORY).as_dict()
    check = _dispatcher_materialization_check(
        "source", REPOSITORY, sys.executable, identity, temp_root=tmp_path,
    )
    assert check["status"] == "pass", check
    assert check["evidence"]["profile_churn"] is True
    assert check["evidence"]["worker_bootstrap"] is True
    assert check["evidence"]["runtime_identity"]["module_root"] == str(REPOSITORY)


def test_profile_home_module_parent_is_filtered_and_plugins_still_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = tmp_path / "profile"
    plugin_root = profile / "plugins"
    plugin_root.mkdir(parents=True)
    (profile / "issue22_direct.py").write_text("value = 'profile'\n", encoding="utf-8")
    (plugin_root / "issue22_plugin.py").write_text("value = 'plugin'\n", encoding="utf-8")
    monkeypatch.chdir(profile)
    monkeypatch.setenv("HERMES_HOME", str(profile))
    monkeypatch.setattr(sys, "path", ["", str(tmp_path), str(plugin_root), *sys.path])
    direct = importlib.import_module("issue22_direct")
    plugin = importlib.import_module("issue22_plugin")
    roots = generation._runtime_import_roots(
        REPOSITORY, profile_home=profile, project_plugins_enabled=False,
    )

    assert direct.value == "profile"
    assert plugin.value == "plugin"
    assert profile not in roots
    assert plugin_root in roots


def test_managed_venv_dependency_root_survives_profile_filter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = tmp_path / ".hermes"
    venv = profile / "hermes-agent" / "venv"
    site_packages = venv / "lib" / "python3.11" / "site-packages"
    site_packages.mkdir(parents=True)
    state_roots = [profile / name for name in ("sessions", "logs", "state", "cache")]
    for root in state_roots:
        root.mkdir()
    monkeypatch.chdir(profile)
    monkeypatch.setattr(sys, "path", ["", str(profile), *(str(root) for root in state_roots), str(site_packages)])
    monkeypatch.setattr(sys, "prefix", str(venv))
    monkeypatch.setattr(sys, "base_prefix", str(venv))
    original_get_path = generation.sysconfig.get_path
    monkeypatch.setattr(
        generation.sysconfig,
        "get_path",
        lambda key: str(site_packages) if key in {"purelib", "platlib"} else original_get_path(key),
    )

    roots = generation._runtime_import_roots(
        REPOSITORY, profile_home=profile, project_plugins_enabled=False,
    )

    assert site_packages in roots
    assert profile not in roots
    assert not any(root in roots for root in state_roots)


def test_profile_and_project_plugins_and_dependencies_are_sealed(
    installation, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, dependencies, plugins, _external, _native_probe, prepare = installation
    workspace = source.parent / "workspace"
    workspace.mkdir()
    project_plugins = workspace / ".hermes" / "plugins"
    dependency = dependencies / "issue22_dependency.py"
    dependency.write_text("VALUE = 'dependency-sealed'\n", encoding="utf-8")
    _write_plugin(plugins, "issue22_profile_plugin", "profile-sealed")
    _write_plugin(project_plugins, "issue22_project_plugin", "project-sealed")
    (source / "hermes_cli" / "main.py").write_text(
        "import json\n"
        "from pathlib import Path\n"
        "import issue22_dependency\n"
        "import issue22_profile_plugin\n"
        "import issue22_project_plugin\n"
        "print(json.dumps({\n"
        "    'dependency': issue22_dependency.VALUE,\n"
        "    'profile': {\n"
        "        'module': issue22_profile_plugin.__file__,\n"
        "        'resource': issue22_profile_plugin.RESOURCE,\n"
        "    },\n"
        "    'project': {\n"
        "        'module': issue22_project_plugin.__file__,\n"
        "        'resource': issue22_project_plugin.RESOURCE,\n"
        "    },\n"
        "}))\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(workspace)
    prepared = prepare(profile_home=plugins.parent, project_plugins_enabled=True)
    try:
        dependency.write_text("VALUE = 'dependency-mutable'\n", encoding="utf-8")
        shutil.rmtree(plugins)
        shutil.rmtree(project_plugins)
        worker = _run_generation_worker(prepared, cwd=workspace)
        assert worker.returncode == 0, worker.stderr
        observed = json.loads(worker.stdout.splitlines()[-1])
        assert observed["dependency"] == "dependency-sealed"
        assert observed["profile"]["resource"] == "issue22_profile_plugin-sealed\n"
        assert observed["project"]["resource"] == "issue22_project_plugin-sealed\n"
        assert Path(observed["profile"]["module"]).is_relative_to(prepared.root)
        assert Path(observed["project"]["module"]).is_relative_to(prepared.root)

        manifest = generation.generation_manifest(prepared.root)
        sealed_dependency = generation._mapped_path(dependency, prepared.root, manifest["paths"])
        sealed_dependency.unlink()
        sealed_dependency.write_text("VALUE = 'tampered'\n", encoding="utf-8")
        tampered = _run_generation_worker(prepared, cwd=workspace)
        assert tampered.returncode != 0
        assert tampered.stdout == ""
    finally:
        generation.cleanup_runtime_generation(prepared.root, force=True)

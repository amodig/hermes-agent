"""All source selection routes hand off once, without old-process maintenance."""
from unittest.mock import Mock

import pytest

from hermes_cli import main, main_web_build, update_cmd, update_cmd_maint
from tests.compat.old_updater_support import fresh_child, no_external_work  # noqa: F401


@pytest.mark.parametrize("hook,args,kwargs", [
    (update_cmd._prepare_updated_checkout, ("unused",), {"desktop": False}),
    (update_cmd._reload_config_modules, (), {}),
    (update_cmd._reload_process_scan_modules, (), {}),
    (update_cmd._run_pending_fleet_restart, (), {}),
    (main_web_build._run_with_idle_timeout, (["unused"], "unused"), {"idle_timeout_seconds": 10}),
    (main_web_build._run_npm_install_deterministic, ("unused", "unused"), {"extra_args": ("arg",)}),
    (main_web_build._nixos_build_env, (), {}),
    (main._reexec_dependency_sync_off_windows_shim, (), {}),
    (update_cmd_maint._print_update_summary, (), {
        "node_failures": [], "desktop_build_ok": True, "pre_update_version": None}),
    (update_cmd_maint._print_update_summary, (), {
        "node_failures": ["dashboard"], "desktop_build_ok": False, "pre_update_version": "0.20.1"}),
    (update_cmd_maint._finish_dashboard_update_cleanup, ([],), {}),
    (update_cmd_maint._finish_dashboard_update_cleanup, (["dashboard"],), {
        "already_restarted_units": {"hermes-serve"}}),
])
def test_historical_completion_hook_never_reports_success(hook, args, kwargs, fresh_child, capsys):
    if hook in {update_cmd._prepare_updated_checkout, update_cmd._reload_config_modules,
                update_cmd._reload_process_scan_modules, update_cmd._run_pending_fleet_restart}:
        with pytest.raises(SystemExit) as error:
            hook(*args, **kwargs)
        assert error.value.code == 1
        assert fresh_child.requests == []
        assert 'run `hermes update` again' in capsys.readouterr().err
        return
    with fresh_child.exits():
        hook(*args, **kwargs)


def test_incomplete_handoff_requires_explicit_update_retry(tmp_path, monkeypatch, capsys):
    from hermes_cli import main

    monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)
    completion = Mock()
    monkeypatch.setattr(update_cmd, "run_completion", completion)
    with pytest.raises(SystemExit) as error:
        update_cmd._complete_source_update(None)
    assert error.value.code == 1
    completion.assert_not_called()
    output = capsys.readouterr()
    assert "run `hermes update` again" in output.err
    assert "Update complete" not in output.out
    assert not (tmp_path / ".update-incomplete").exists()
    assert not (tmp_path / ".lazy-refresh-incomplete").exists()


def test_completion_child_can_lock_installation_after_parent_handoff(tmp_path, monkeypatch):
    import os
    from pathlib import Path
    import subprocess
    import sys
    from hermes_cli.kanban_runtime_generation import installation_mutation_lock
    from hermes_cli.venv_sync import completion_pending_path

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "cache"))
    monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(update_cmd, "_unrestored_autostash_notice", lambda: None)
    monkeypatch.setattr(update_cmd, "_write_fleet_restart_pending_marker", lambda **kw: None)
    request = {
        "source": str(tmp_path), "home": os.environ["HERMES_HOME"],
        "expected_sha": "a" * 40, "windows_resume": None,
        "receipt": {"update_id": "c" * 32},
    }
    root = Path(__file__).resolve().parents[2]

    def complete(_):
        # A runtime builder must see the incomplete marker throughout the handoff,
        # while the new interpreter can acquire the lock for its own mutations.
        assert completion_pending_path(tmp_path).is_file()
        result = subprocess.run(
            [sys.executable, "-S", "-c",
             "import sys; sys.path.insert(0, sys.argv[1]); "
             "from hermes_cli.kanban_runtime_generation import installation_mutation_lock; "
             "lock = installation_mutation_lock(sys.argv[2], blocking=False); "
             "lock.__enter__(); lock.__exit__(None, None, None)",
             str(root), str(tmp_path)],
            cwd=tmp_path, capture_output=True, text=True, timeout=15,
        )
        assert result.returncode == 0, result.stderr
        return {"exit_code": 0, "receipt": None, "windows_resume": None}

    monkeypatch.setattr(update_cmd, "run_completion", complete)
    with installation_mutation_lock(tmp_path) as release:
        update_cmd._complete_source_update(request, release_installation=release)


def test_completion_allows_pm_workers_but_excludes_concurrent_build_mutations(tmp_path, monkeypatch):
    """A configured-feature repair must finish, without unlocking frontend writes."""
    import os
    from pathlib import Path
    import subprocess
    import sys
    import pm
    from hermes_cli import main_install_repair, memory_provider_migration, source_build
    from hermes_cli import source_stamp, update_completion, venv_sync

    for name in ("HOME", "USERPROFILE", "XDG_CACHE_HOME", "LOCALAPPDATA"):
        monkeypatch.setenv(name, str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "completion-lock-test"\nversion = "0"\n'
        '[project.optional-dependencies]\nmcp = []\n', encoding="utf-8")
    (tmp_path / "ui-tui").mkdir()
    (tmp_path / "ui-tui/package.json").write_text("{}", encoding="utf-8")
    pending = venv_sync.arm_completion(tmp_path)
    repository = Path(__file__).resolve().parents[2]

    def worker(name, *, locked):
        assert pending.exists(), "runtime capture must remain fenced until completion"
        result = subprocess.run(
            [sys.executable, "-I", "-S", "-c",
             "import sys\nfrom pathlib import Path\n"
             "sys.path.insert(0, sys.argv[1])\n"
             "from hermes_cli.kanban_runtime_generation import installation_mutation_lock\n"
             "try:\n"
             "    with installation_mutation_lock(sys.argv[2], blocking=False):\n"
             "        assert sys.argv[3] == 'free', 'installation mutation was not excluded'\n"
             "        (Path(sys.argv[2]) / sys.argv[4]).write_text('published')\n"
             "except BlockingIOError:\n"
             "    assert sys.argv[3] == 'locked', 'waiting parent holds the PM worker lock'\n",
             str(repository), str(tmp_path), "locked" if locked else "free", name],
            cwd=tmp_path, env=dict(os.environ), capture_output=True, text=True, timeout=15,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    monkeypatch.setattr(main_install_repair, "_configured_features_missing_deps",
                        lambda: [("MCP servers", "install MCP", "mcp")])
    monkeypatch.setattr(pm, "sync_venv", lambda *a, **kw: worker("repaired", locked=False))
    monkeypatch.setattr(source_build, "source_build_env", lambda **kw: {})
    monkeypatch.setattr(source_build, "prepare_source_dependencies",
                        lambda *a, **kw: worker("concurrent-node-writer", locked=True))

    def build(*args, **kwargs):
        assert (tmp_path / "repaired").read_text() == "published"
        worker("concurrent-build-writer", locked=True)

    monkeypatch.setattr(source_build, "build_source_tui", build)
    monkeypatch.setattr(memory_provider_migration, "migrate_all_homes",
                        lambda: worker("migrated", locked=False))
    monkeypatch.setattr(update_cmd, "_sweep_bytecode_after_update",
                        lambda branch: worker("concurrent-bytecode-writer", locked=True))
    monkeypatch.setattr(venv_sync, "publish_launchers",
                        lambda root: worker("concurrent-launcher-writer", locked=True))
    monkeypatch.setattr(update_cmd_maint, "_run_post_update_maintenance",
                        lambda **kw: worker("maintained", locked=False) is None)
    monkeypatch.setattr(source_stamp, "write_source_stamp",
                        lambda root: worker("concurrent-stamp-writer", locked=True))

    update_completion._complete_selected({
        "source": str(tmp_path), "branch": "main", "sibling_snapshots": {}, "plan": None,
        "desktop": False, "assume_yes": True, "gateway_mode": False,
        "snapshot_id": None, "pre_update_version": None, "no_gateway_restart": True,
    })
    assert not pending.exists()
    assert (tmp_path / "migrated").read_text() == "published"
    assert (tmp_path / "maintained").read_text() == "published"
    assert not list(tmp_path.glob("concurrent-*-writer"))

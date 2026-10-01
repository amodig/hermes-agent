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

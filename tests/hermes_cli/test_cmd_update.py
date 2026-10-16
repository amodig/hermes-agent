"""Git trampoline recovery; branch updates use the real target-identity suite."""

import subprocess
from unittest.mock import patch

import pytest

from hermes_cli import update_cmd


@pytest.fixture(autouse=True)
def _isolate_venv_holders(monkeypatch):
    """The update flow's venv-holder guard sees the live gateway processes on
    a dev machine and aborts with SystemExit 2 before reaching the branch
    logic under test.  Isolate it so the test exercises the intended path."""
    monkeypatch.setattr("hermes_cli.update_cmd_windows._detect_venv_python_processes", lambda: [])


class TestGitTrampolineSelfHeal:
    """Proactive Git-for-Windows trampoline self-heal (#87876).

    A broken bin\\git.exe / cmd\\git.exe shim (~46KB) refuses every git call
    with a "BUG (fork bomb)" guard instead of re-execing the real git-core
    binary. _ensure_non_trampoline_git detects this up front and swaps in a
    real git binary when one can be located, so the normal git update path
    survives instead of degrading to the ZIP fallback.
    """

    @staticmethod
    def _fake_run_healthy(command, **_kwargs):
        return subprocess.CompletedProcess(
            command, 0, stdout="git version 2.50.0.windows.1\n", stderr=""
        )

    @staticmethod
    def _fake_run_trampoline(command, **_kwargs):
        return subprocess.CompletedProcess(
            command,
            1,
            stdout="",
            stderr="BUG (fork bomb): tried to spawn itself, check your PATH\n",
        )

    @pytest.mark.platforms("windows")
    def test_healthy_git_command_unchanged(self):
        from hermes_cli import update_cmd

        git_cmd = ["git", "-c", "windows.appendAtomically=false"]
        with (
            patch(
                "hermes_cli.update_cmd.subprocess.run",
                side_effect=self._fake_run_healthy,
            ),
            patch("hermes_cli.update_cmd._locate_real_git") as locate,
        ):
            result = update_cmd._ensure_non_trampoline_git(git_cmd)
        assert result == git_cmd
        locate.assert_not_called()

    @pytest.mark.platforms("windows")
    def test_trampoline_swaps_to_real_git(self, capsys):
        from pathlib import Path

        from hermes_cli import update_cmd

        git_cmd = ["git", "-c", "windows.appendAtomically=false"]
        real = Path(r"C:\Program Files\Git\mingw64\libexec\git-core\git.exe")
        with (
            patch(
                "hermes_cli.update_cmd.subprocess.run",
                side_effect=self._fake_run_trampoline,
            ),
            patch(
                "hermes_cli.update_cmd._locate_real_git", return_value=real
            ),
        ):
            result = update_cmd._ensure_non_trampoline_git(git_cmd)
        assert result == [str(real), "-c", "windows.appendAtomically=false"]
        out = capsys.readouterr().out
        assert "switching to real git" in out

    @pytest.mark.platforms("windows")
    def test_trampoline_no_real_git_keeps_command(self, capsys):
        from hermes_cli import update_cmd

        git_cmd = ["git", "-c", "windows.appendAtomically=false"]
        with (
            patch(
                "hermes_cli.update_cmd.subprocess.run",
                side_effect=self._fake_run_trampoline,
            ),
            patch("hermes_cli.update_cmd._locate_real_git", return_value=None),
        ):
            result = update_cmd._ensure_non_trampoline_git(git_cmd)
        assert result == git_cmd
        out = capsys.readouterr().out
        assert "ZIP path" in out

    @pytest.mark.platforms("not windows")
    def test_off_windows_noop(self):
        from hermes_cli import update_cmd

        git_cmd = ["git"]
        with patch("hermes_cli.update_cmd.subprocess.run") as run:
            result = update_cmd._ensure_non_trampoline_git(git_cmd)
        assert result == git_cmd
        run.assert_not_called()

    @pytest.mark.parametrize("arch", ["ucrt64", "clangarm64", "mingw64", "mingw32"])
    @pytest.mark.parametrize("location", ["standard", "user", "shared", "profile"])
    def test_locator_finds_supported_layout(self, tmp_path, monkeypatch, arch, location):
        from pathlib import Path
        import ntpath

        import hermes_constants
        from hermes_cli import update_cmd_git

        root = tmp_path / "root"
        profile = root / "profiles" / "foo"
        monkeypatch.setattr(hermes_constants, "get_default_hermes_root", lambda: root)
        monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: profile)
        local_app_data = tmp_path / "LocalAppData"
        monkeypatch.setenv("LOCALAPPDATA", str(local_app_data))
        base = {
            "standard": Path(ntpath.join(r"C:\Program Files", "Git")),
            "user": Path(ntpath.join(str(local_app_data), "Programs", "Git")),
            "shared": root / "git",
            "profile": profile / "git",
        }[location]
        real = base / arch / "libexec" / "git-core" / "git.exe"
        monkeypatch.setattr(Path, "exists", lambda path: path == real)
        monkeypatch.setattr(update_cmd_git.subprocess, "run", self._fake_run_healthy)

        assert update_cmd_git._locate_real_git() == real

    def test_portable_git_prefers_shared_root(self, tmp_path, monkeypatch):
        import hermes_constants
        from hermes_cli import update_cmd_git

        root = tmp_path / "root"
        profile = root / "profiles" / "foo"
        monkeypatch.setattr(hermes_constants, "get_default_hermes_root", lambda: root)
        monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: profile)
        shared = root / "git" / "mingw64" / "libexec" / "git-core" / "git.exe"
        scoped = profile / "git" / "ucrt64" / "libexec" / "git-core" / "git.exe"
        for binary in (shared, scoped):
            binary.parent.mkdir(parents=True)
            binary.touch()
        monkeypatch.setattr(update_cmd_git.subprocess, "run", self._fake_run_healthy)

        assert update_cmd_git._locate_real_git() == shared

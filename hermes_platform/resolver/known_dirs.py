"""Directories where tools install outside PATH, grouped by the ecosystem that owns them.

Every table in Hermes lives here. A directory literal outside this module fails the ratchet in
`tests/test_managed_runtime_resolution.py`. Host directory tables are empty on an OS
where the ecosystem does not install there. Windows-targeted Git candidate APIs
accept target environment data independently of the querying host.
"""

from __future__ import annotations

import ntpath
import os
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path

_POSIX = sys.platform != "win32"
_WIN = sys.platform == "win32"


def homebrew_dirs() -> tuple[str, ...]:
    return ("/opt/homebrew/bin", "/usr/local/bin") if sys.platform == "darwin" else ()


def user_local_bin() -> tuple[str, ...]:
    return ("~/.local/bin",) if _POSIX else ("%USERPROFILE%/.local/bin",)


def rust_tool_dirs() -> tuple[str, ...]:
    return ("~/.cargo/bin",) if _POSIX else ("%USERPROFILE%/.cargo/bin",)


def node_tool_dirs() -> tuple[str, ...]:
    return ("~/.npm-global/bin", "~/.bun/bin", "~/.volta/bin") if _POSIX else ("%APPDATA%/npm", "%USERPROFILE%/.bun/bin", "%LOCALAPPDATA%/Volta/bin")


def hermes_vendored_dirs() -> tuple[str, ...]:
    return ("~/.hermes/bin",) if _POSIX else ("%USERPROFILE%/.hermes/bin",)


def windows_user_program_dirs() -> tuple[str, ...]:
    if not _WIN:
        return ()
    return (
        "%LOCALAPPDATA%/Programs",
        "%USERPROFILE%/scoop/shims",
        "%LOCALAPPDATA%/Microsoft/WinGet/Links",
    )


# Git for Windows 2.56 moved x64 from mingw64 to ucrt64.
_GIT_WINDOWS_TOOLCHAINS = ("ucrt64", "clangarm64", "mingw64", "mingw32")


def windows_git_roots(env: Mapping[str, str] | None = None) -> tuple[str, ...]:
    """Windows-targeted install roots, independent of the querying host OS."""
    env = os.environ if env is None else env
    roots = [ntpath.join(env.get("ProgramFiles", r"C:\Program Files"), "Git")]
    if env.get("ProgramFiles(x86)"):
        roots.append(ntpath.join(env["ProgramFiles(x86)"], "Git"))
    if env.get("LOCALAPPDATA"):
        roots.extend((
            ntpath.join(env["LOCALAPPDATA"], "hermes", "git"),
            ntpath.join(env["LOCALAPPDATA"], "Programs", "Git"),
        ))
    return tuple(roots)


def windows_git_core_candidates(managed_roots: Iterable[Path] = ()) -> list[Path]:
    """Real Git binaries: installed trees, then managed trees in caller order.

    Discovery only: callers check existence and health. Managed roots are Hermes
    homes, shared home before profile home; resolving profile scope stays with
    the caller rather than importing application state into the platform layer.
    """
    roots = [Path(root) for root in windows_git_roots()]
    roots.extend(root / "git" for root in managed_roots)
    return [
        root / arch / "libexec" / "git-core" / "git.exe"
        for root in roots
        for arch in _GIT_WINDOWS_TOOLCHAINS
    ]

"""Worker exit classification must not depend on POSIX-only ``os.WIF*`` helpers.

``_classify_worker_exit`` feeds the dead-worker reclaim: a ``rate_limited``
verdict requeues the card without counting a failure, ``unknown`` counts as a
crash. Windows has neither ``os.WIFEXITED`` nor ``waitpid(-1)``, so both the
decode and the reaper's exit capture have a Windows-independent path.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_runtime_generation as generation
from hermes_cli import kanban_worker_runtime as runtime


def _spawn_exit(code: int) -> subprocess.Popen:
    proc = subprocess.Popen([sys.executable, "-c", f"raise SystemExit({code})"])  # noqa: S603
    proc.wait()
    return proc




@pytest.mark.platforms("windows")
def test_native_windows_reaper_and_decode(monkeypatch, tmp_path):
    """Native Windows, reaper/decode path unpatched: ``_IS_WINDOWS`` selects the
    Popen-poll reaper and the decode runs where ``os.WIFEXITED`` does not exist,
    so the rate-limit sentinel exit is a requeue, not a crash. The reaper also
    sweeps the profile-independent runtime cache (``%LOCALAPPDATA%\\hermes\\
    kanban-runtime`` on Windows, i.e. the real hermes home), so that root is
    redirected; nothing about the exit handling is patched."""
    storage = tmp_path / "runtime-storage"
    (storage / "workers").mkdir(parents=True)
    monkeypatch.setattr(generation, "_runtime_storage_root", lambda: storage)
    monkeypatch.setattr(runtime, "_worker_processes", {})
    monkeypatch.setattr(runtime, "_recent_worker_exits", {})
    assert not hasattr(os, "WIFEXITED")
    proc = _spawn_exit(kb.KANBAN_RATE_LIMIT_EXIT_CODE)
    runtime._worker_processes[proc.pid] = proc
    assert runtime.reap_worker_zombies() == [proc.pid]
    assert runtime._classify_worker_exit(proc.pid) == ("rate_limited", kb.KANBAN_RATE_LIMIT_EXIT_CODE)

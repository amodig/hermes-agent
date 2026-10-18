"""The sealed-bootstrap phase budget must fit the payload each phase re-hashes.

One bootstrap phase performs a full generation verification before the worker may
publish its identity, and a second one after imports. A 3.7 GiB / 171,589-file
sealed payload measured 30 s on an idle host and 74 s under load, so a fixed 60 s
phase produced ``worker bootstrap timed out`` on live dispatches.
"""

from __future__ import annotations

import pytest

from hermes_cli import kanban_worker_runtime as runtime

GIB = 1024 ** 3
# Worst measured single-phase verification of the 3.7 GiB payload under load.
MEASURED_WORST_PHASE_SECONDS = 74.0


def test_phase_budget_covers_the_measured_payload_verification(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime, "_payload_bytes", lambda root: int(3.7 * GIB))
    timeout = runtime._bootstrap_phase_timeout(tmp_path)
    assert timeout >= MEASURED_WORST_PHASE_SECONDS, (
        f"budget {timeout:.0f}s does not contain the "
        f"{MEASURED_WORST_PHASE_SECONDS:.0f}s verification it must wait for"
    )


def test_small_payloads_keep_the_floor(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime, "_payload_bytes", lambda root: 8 * 1024 * 1024)
    assert runtime._bootstrap_phase_timeout(tmp_path) == runtime._BOOTSTRAP_PHASE_FLOOR_SECONDS


def test_both_phases_still_fit_one_claim_lease(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime, "_payload_bytes", lambda root: 400 * GIB)
    timeout = runtime._bootstrap_phase_timeout(tmp_path)
    claim_ttl = runtime._kb()._resolve_claim_ttl_seconds()
    assert 2 * timeout < claim_ttl, "the two phases must fit inside the claim lease"
    assert timeout < 400 * GIB / runtime._BOOTSTRAP_PAYLOAD_BYTES_PER_SECOND


def test_payload_bytes_sums_real_files(tmp_path):
    (tmp_path / "a").write_bytes(b"x" * 100)
    nested = tmp_path / "sub"
    nested.mkdir()
    (nested / "b").write_bytes(b"y" * 250)
    assert runtime._payload_bytes(tmp_path) == 350

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


def test_pathological_payloads_are_bounded(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime, "_payload_bytes", lambda root: 400 * GIB)
    timeout = runtime._bootstrap_phase_timeout(tmp_path)
    assert timeout == runtime._BOOTSTRAP_PHASE_CEILING_SECONDS
    assert timeout < 400 * GIB / runtime._BOOTSTRAP_PAYLOAD_BYTES_PER_SECOND


def test_short_claim_leases_do_not_shrink_the_budget(monkeypatch, tmp_path):
    # The pre-import phase runs before any claim exists, so a short
    # HERMES_KANBAN_CLAIM_TTL_SECONDS must not push the budget below the work.
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_TTL_SECONDS", "60")
    monkeypatch.setattr(runtime, "_payload_bytes", lambda root: int(3.7 * GIB))
    assert runtime._bootstrap_phase_timeout(tmp_path) >= MEASURED_WORST_PHASE_SECONDS


def test_reference_payload_fits_the_default_claim_lease(monkeypatch, tmp_path):
    # The surrounding window is asserted, not enforced by shrinking the budget.
    monkeypatch.delenv("HERMES_KANBAN_CLAIM_TTL_SECONDS", raising=False)
    monkeypatch.setattr(runtime, "_payload_bytes", lambda root: int(3.18 * GIB))
    timeout = runtime._bootstrap_phase_timeout(tmp_path)
    assert 2 * timeout < runtime._kb()._resolve_claim_ttl_seconds()


def test_payload_bytes_sums_real_files(tmp_path):
    (tmp_path / "a").write_bytes(b"x" * 100)
    nested = tmp_path / "sub"
    nested.mkdir()
    (nested / "b").write_bytes(b"y" * 250)
    assert runtime._payload_bytes(tmp_path) == 350

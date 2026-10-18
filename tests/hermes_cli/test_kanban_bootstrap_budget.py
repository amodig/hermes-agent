"""The sealed-bootstrap phase budget must fit the payload each phase re-hashes.

One bootstrap phase performs a full generation verification before the worker may
publish its identity, and a second one after imports. A sealed payload carries
interpreters and dependencies and is gigabytes, so a fixed wall clock sized for a
warm idle host times out under load.
"""

from __future__ import annotations

from hermes_cli import kanban_worker_runtime as runtime

GIB = 1024 ** 3
# A sealed payload's order of magnitude: interpreters plus dependencies.
PAYLOAD_SCALE = 4 * GIB


def _budget(monkeypatch, tmp_path, payload_bytes):
    monkeypatch.setattr(runtime, "_payload_bytes", lambda root: payload_bytes)
    return runtime._bootstrap_phase_timeout(tmp_path)


def test_gigabyte_scale_payload_raises_the_budget_above_the_floor(monkeypatch, tmp_path):
    timeout = _budget(monkeypatch, tmp_path, PAYLOAD_SCALE)
    assert timeout > runtime._BOOTSTRAP_PHASE_FLOOR_SECONDS, (
        "a payload of this size must not be budgeted at the small-payload floor"
    )
    assert timeout <= runtime._BOOTSTRAP_PHASE_CEILING_SECONDS


def test_budget_grows_with_the_payload(monkeypatch, tmp_path):
    small = _budget(monkeypatch, tmp_path, PAYLOAD_SCALE)
    large = _budget(monkeypatch, tmp_path, 2 * PAYLOAD_SCALE)
    assert large > small


def test_small_payloads_keep_the_floor(monkeypatch, tmp_path):
    assert _budget(monkeypatch, tmp_path, 8 * 1024 * 1024) == runtime._BOOTSTRAP_PHASE_FLOOR_SECONDS


def test_pathological_payloads_are_bounded(monkeypatch, tmp_path):
    assert _budget(monkeypatch, tmp_path, 400 * GIB) == runtime._BOOTSTRAP_PHASE_CEILING_SECONDS


def test_short_claim_leases_do_not_shrink_the_budget(monkeypatch, tmp_path):
    # The pre-import phase runs before any claim exists, so a short
    # HERMES_KANBAN_CLAIM_TTL_SECONDS must not push the budget below the work.
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_TTL_SECONDS", "60")
    assert _budget(monkeypatch, tmp_path, PAYLOAD_SCALE) > runtime._BOOTSTRAP_PHASE_FLOOR_SECONDS


def test_gigabyte_payload_fits_the_default_claim_lease(monkeypatch, tmp_path):
    # The surrounding window is asserted, not enforced by shrinking the budget.
    monkeypatch.delenv("HERMES_KANBAN_CLAIM_TTL_SECONDS", raising=False)
    timeout = _budget(monkeypatch, tmp_path, PAYLOAD_SCALE)
    assert 2 * timeout < runtime._kb()._resolve_claim_ttl_seconds()


def test_payload_bytes_sums_real_files(tmp_path):
    (tmp_path / "a").write_bytes(b"x" * 100)
    nested = tmp_path / "sub"
    nested.mkdir()
    (nested / "b").write_bytes(b"y" * 250)
    assert runtime._payload_bytes(tmp_path) == 350

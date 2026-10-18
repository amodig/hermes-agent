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


def test_phase_budget_derives_from_the_sealed_payload(monkeypatch, tmp_path):
    # Small payloads keep the historical floor.
    assert _budget(monkeypatch, tmp_path, 8 * 1024 * 1024) == runtime._BOOTSTRAP_PHASE_FLOOR_SECONDS

    # A gigabyte-scale payload raises the budget, and both phases still fit the
    # default lease.
    monkeypatch.delenv("HERMES_KANBAN_CLAIM_TTL_SECONDS", raising=False)
    budget = _budget(monkeypatch, tmp_path, PAYLOAD_SCALE)
    assert budget > runtime._BOOTSTRAP_PHASE_FLOOR_SECONDS
    assert 2 * budget < runtime._kb()._resolve_claim_ttl_seconds()

    # Each phase walks the payload twice, so the derivation covers both passes.
    assert _budget(monkeypatch, tmp_path, GIB) >= (
        2 * GIB / runtime._BOOTSTRAP_PAYLOAD_BYTES_PER_SECOND
    )

    # A pathological payload is bounded instead of wedging a dispatcher tick, and
    # both phases plus the scope gates still fit the attribution window.
    assert _budget(monkeypatch, tmp_path, 400 * GIB) == runtime._BOOTSTRAP_PHASE_CEILING_SECONDS
    assert (
        2 * runtime._BOOTSTRAP_PHASE_CEILING_SECONDS + runtime._SCOPE_GATE_ALLOWANCE_SECONDS
        <= runtime._SCOPE_ATTEMPT_WINDOW_SECONDS
    )

    # Both phases run before any claim exists (the dispatcher spawns with
    # defer_grant and claims afterwards), so a short lease must not shrink them.
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_TTL_SECONDS", "60")
    assert _budget(monkeypatch, tmp_path, PAYLOAD_SCALE) > runtime._BOOTSTRAP_PHASE_FLOOR_SECONDS


def test_payload_bytes_mirrors_the_digest_walk(tmp_path):
    (tmp_path / "a").write_bytes(b"x" * 100)
    nested = tmp_path / "sub"
    nested.mkdir()
    (nested / "b").write_bytes(b"y" * 250)
    # The manifest is not part of the payload the worker re-hashes.
    (tmp_path / "generation.json").write_bytes(b"z" * 999)
    assert runtime._payload_bytes(tmp_path) == 350

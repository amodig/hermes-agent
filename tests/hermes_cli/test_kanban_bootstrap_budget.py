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


def _budget(monkeypatch, tmp_path, *, payload_bytes, members=0):
    monkeypatch.setattr(runtime, "_payload_scan", lambda root: (payload_bytes, members))
    return runtime._bootstrap_phase_timeout(tmp_path)


def test_phase_budget_derives_from_the_sealed_payload(monkeypatch, tmp_path):
    # Small payloads keep the historical floor.
    assert (
        _budget(monkeypatch, tmp_path, payload_bytes=8 * 1024 * 1024)
        == runtime._BOOTSTRAP_PHASE_FLOOR_SECONDS
    )

    # A gigabyte-scale payload raises the budget, and both phases still fit the
    # default lease.
    monkeypatch.delenv("HERMES_KANBAN_CLAIM_TTL_SECONDS", raising=False)
    budget = _budget(monkeypatch, tmp_path, payload_bytes=PAYLOAD_SCALE)
    assert budget > runtime._BOOTSTRAP_PHASE_FLOOR_SECONDS
    assert 2 * budget < runtime._kb()._resolve_claim_ttl_seconds()

    # Each phase walks the payload twice, so the derivation covers both passes.
    assert _budget(monkeypatch, tmp_path, payload_bytes=GIB) >= (
        2 * GIB / runtime._BOOTSTRAP_PAYLOAD_BYTES_PER_SECOND
    )

    # Per-member cost is not described by bytes: many tiny files must raise the
    # budget on their own.
    assert (
        _budget(monkeypatch, tmp_path, payload_bytes=8 * 1024 * 1024, members=500_000)
        > runtime._BOOTSTRAP_PHASE_FLOOR_SECONDS
    )

    # A pathological payload is bounded instead of wedging a dispatcher tick, and
    # the phases alone stay inside the strict attribution window. The scope gates
    # between them are not statically bounded, so the window is not derived from
    # them or widened for it: a bootstrap slow enough to outrun the window leaves
    # its scope to reconciliation.
    assert (
        _budget(monkeypatch, tmp_path, payload_bytes=400 * GIB)
        == runtime._BOOTSTRAP_PHASE_CEILING_SECONDS
    )
    assert 2 * runtime._BOOTSTRAP_PHASE_CEILING_SECONDS < runtime._SCOPE_ATTEMPT_WINDOW_SECONDS

    # Both phases run before any claim exists (the dispatcher spawns with
    # defer_grant and claims afterwards), so a short lease must not shrink them.
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_TTL_SECONDS", "60")
    assert (
        _budget(monkeypatch, tmp_path, payload_bytes=PAYLOAD_SCALE)
        > runtime._BOOTSTRAP_PHASE_FLOOR_SECONDS
    )


def test_payload_scan_mirrors_the_digest_walk(tmp_path):
    payload = tmp_path / "payload"
    payload.mkdir()
    (payload / "a").write_bytes(b"x" * 100)
    nested = payload / "sub"
    nested.mkdir()
    (nested / "b").write_bytes(b"y" * 250)
    # The manifest is not part of the payload the worker re-hashes, and directories
    # are members: the digest hashes a name and separators for each one.
    (payload / "generation.json").write_bytes(b"z" * 999)
    assert runtime._payload_scan(payload) == (350, 4)

"""Autonomous-boundary verdict for tests that are not a deployed host.

The runtime refuses every autonomous launch unless the host really presents the
reviewed boundary: the caller inside ``autonomous.slice``, the single
``ConcurrencyHardMax=1`` slot free, and no misplaced worker scope. That refusal is
the point of the change, and no test runner -- least of all a CI host -- satisfies
it.

Tests that drive the dispatcher, the runtime-generation bootstrap or the scope
routing are exercising that PLUMBING, not host policy, so they get an explicit
verdict here instead of a refusal they cannot satisfy. This is a test-side patch:
nothing in production reads it, there is no environment variable that disables the
boundary, and the boundary's own decisions are exercised directly by
``tests/tools/test_autonomous_resources.py`` and against real transient slices by
``tests/_fixtures/autonomous_slice.py``.

A test that needs the REAL verdict replaces this patch with its own, which
``monkeypatch`` applies after this fixture has run (see
``tests/hermes_cli/test_kanban_gateway_restart_handoff.py``).
"""
from __future__ import annotations

import pytest

#: The verdict the stub reports. Deliberately descriptive rather than empty, so a
#: diagnostic printed from a stubbed run says plainly that no host was consulted.
STUBBED_BOUNDARY_REPORT = {
    "aggregate": {"unit": "autonomous.slice", "cgroup": "/stub/autonomous.slice"},
    "workers": {
        "unit": "autonomous-workers.slice",
        "cgroup": "/stub/autonomous.slice/autonomous-workers.slice",
    },
    "worker_scopes": [],
    "worker_slot_occupied": False,
    "launcher_cgroup": "/stub/autonomous.slice/launcher.scope",
}


@pytest.fixture(autouse=True)
def _stub_autonomous_boundary(monkeypatch):
    """Replace the host verdict for every test that does not opt out. See module docstring."""
    from tools import autonomous_resources as resources
    from tools import process_registry as registry

    monkeypatch.setattr(
        registry, "require_autonomous_boundary",
        lambda **_kwargs: dict(STUBBED_BOUNDARY_REPORT),
    )
    monkeypatch.setattr(
        registry, "autonomous_boundary_inventory",
        lambda: dict(STUBBED_BOUNDARY_REPORT),
    )
    monkeypatch.setattr(
        resources, "check_autonomous_worker",
        lambda pid, **_kwargs: {"worker_cgroup": f"/stub/hermes-worker-{pid}.scope"},
    )
    return STUBBED_BOUNDARY_REPORT


@pytest.fixture()
def stub_scope_capability(monkeypatch):
    """Assume this host can create a transient scope, for tests that assert argv.

    A runner with no user bus cannot create one, and the runtime now REFUSES an
    autonomous child it cannot place instead of silently running it in the
    gateway's cgroup (the old ``in_process`` route). Tests that inspect the argv or
    the child environment a launch would use never create a real scope, so they ask
    for this. It is deliberately NOT autouse: `tests/tools/test_process_registry.py`
    counts the capability PROBE, and a host-wide assumption would erase it.
    """
    from tools import process_registry as registry

    monkeypatch.setattr(registry, "_systemd_run_user_scope_available", lambda: True)

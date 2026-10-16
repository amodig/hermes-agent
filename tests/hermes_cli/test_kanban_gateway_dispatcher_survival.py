"""Dispatcher death and immutable worker generation survival."""

import pytest

from tests.hermes_cli.test_kanban_gateway_restart_handoff import (
    _assert_worker_and_claim_survive_dispatcher_exit_and_shared_install_replacement,
    worker_setup,
)


@pytest.mark.platforms("linux")
@pytest.mark.live_system_guard_bypass  # cleanup signals our start-time-verified, reparented worker
def test_worker_and_claim_survive_dispatcher_exit_and_shared_install_replacement(worker_setup, monkeypatch):
    _assert_worker_and_claim_survive_dispatcher_exit_and_shared_install_replacement(
        worker_setup, monkeypatch,
    )

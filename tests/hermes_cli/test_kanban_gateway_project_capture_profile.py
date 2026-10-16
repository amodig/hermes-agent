"""Worker project-plugin capture against profile-local dotenv encodings."""

import pytest

from tests.hermes_cli.test_kanban_gateway_restart_handoff import (
    _PROJECT_PLUGIN_CAPTURE_CASES,
    _assert_worker_project_plugin_capture_matches_profile_dotenv,
    worker_boot_boundary,
    worker_setup,
)


@pytest.mark.live_system_guard_bypass  # private transient worker slice on Linux
@pytest.mark.parametrize(
    ("dispatcher_gate", "scope", "encoding", "profile_env", "enabled"),
    _PROJECT_PLUGIN_CAPTURE_CASES[:8],
)
def test_worker_project_plugin_capture_matches_profile_dotenv(
    worker_setup, worker_boot_boundary, monkeypatch,
    dispatcher_gate, scope, encoding, profile_env, enabled,
):
    _assert_worker_project_plugin_capture_matches_profile_dotenv(
        worker_setup, worker_boot_boundary, monkeypatch,
        dispatcher_gate, scope, encoding, profile_env, enabled,
    )

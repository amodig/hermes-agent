"""Shallow-checkout guard on the `hermes update` apply path (#53479).

`rev-list --count HEAD..origin/<branch>` on a shallow install can enumerate
the entire remote ancestry ("Found 9980 new commit(s)" on a depth-1 clone).
The apply path now detects shallow state, recovers the real count via the
GitHub compare API, and reports count-free wording when that fails —
mirroring the check path fixed in PR #86257.

These tests exercise the real checkout-preparation path by faking only the
subprocess layer (git) and the compare API — commit-count decisions run
for real.
"""

from unittest.mock import MagicMock, patch

import hermes_cli.main as cli_main

import hermes_cli.update_cmd as update_cmd

SHA_A = "a" * 40
SHA_B = "b" * 40


def _git_responder(*, shallow: bool, count: str):
    """Answer the git subprocess calls the count block makes."""

    def fake_run(cmd, **kwargs):
        joined = " ".join(cmd)
        if "rev-list" in joined and "--count" in joined:
            return MagicMock(returncode=0, stdout=f"{count}\n", stderr="")
        if "--is-shallow-repository" in joined:
            return MagicMock(returncode=0, stdout=("true\n" if shallow else "false\n"), stderr="")
        if "rev-parse HEAD" in joined:
            return MagicMock(returncode=0, stdout=f"{SHA_A}\n", stderr="")
        if "rev-parse origin/main" in joined:
            return MagicMock(returncode=0, stdout=f"{SHA_B}\n", stderr="")
        return MagicMock(returncode=0, stdout="", stderr="")

    return fake_run


def _run_count_block(tmp_path, *, shallow: bool, raw_count: str, api_count):
    """Prepare a clean checkout using the real shallow-count decision."""
    fake = _git_responder(shallow=shallow, count=raw_count)
    with patch.object(cli_main, "PROJECT_ROOT", tmp_path), patch.object(
        update_cmd.subprocess, "run", side_effect=fake
    ), patch("hermes_cli.banner._github_compare_behind", return_value=api_count):
        plan = update_cmd._prepare_checkout_for_update(
            ["git"], "main", "main",
            is_fork=False,
            assume_yes=True,
            gateway_mode=False,
            gw_input_fn=None,
            switch_branch=False,
            _windows_gateway_resume=None,
        )
    return plan.commit_count


def test_full_clone_keeps_exact_count(tmp_path):
    assert _run_count_block(tmp_path, shallow=False, raw_count="7", api_count=None) == 7


def test_shallow_bogus_count_recovers_via_compare_api(tmp_path):
    """FAIL-BEFORE: reported the bogus 9980 as 'Found 9980 new commit(s)'."""
    assert _run_count_block(tmp_path, shallow=True, raw_count="9980", api_count=12) == 12


def test_shallow_bogus_count_offline_reports_unknown(tmp_path):
    assert _run_count_block(tmp_path, shallow=True, raw_count="9980", api_count=None) == -1


def test_shallow_local_ahead_treated_as_up_to_date(tmp_path):
    assert _run_count_block(tmp_path, shallow=True, raw_count="3", api_count=0) == 0


def test_shallow_zero_count_short_circuits_without_api(tmp_path):
    # A zero local count is authoritative even when the API would report updates.
    assert _run_count_block(tmp_path, shallow=True, raw_count="0", api_count=12) == 0

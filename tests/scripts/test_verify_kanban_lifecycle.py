"""Regression tests for layered lifecycle runtime selection."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "verify_kanban_lifecycle.py"
_SPEC = importlib.util.spec_from_file_location("verify_kanban_lifecycle", SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
verify_kanban_lifecycle = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(verify_kanban_lifecycle)


def test_installed_verifier_rejects_source_runtime_fallback(tmp_path: Path) -> None:
    receipt_path = tmp_path / "receipt.json"
    code = verify_kanban_lifecycle.main(
        [
            "--layer",
            "installed",
            "--runtime-root",
            str(tmp_path),
            "--receipt",
            str(receipt_path),
        ]
    )

    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert code == 2
    assert payload["result"]["status"] == "incomplete"
    assert payload["result"]["scenarios"] == []
    assert any("installed runtime identity root mismatch" in item for item in payload["result"]["diagnostics"])


@pytest.mark.parametrize("profiles_present", [False, True])
def test_policy_failure_writes_incomplete_receipt(tmp_path: Path, profiles_present: bool) -> None:
    if profiles_present:
        (tmp_path / "profiles").mkdir()
    receipt_path = tmp_path / "receipt.json"
    code = verify_kanban_lifecycle.main([
        "--layer", "source", "--policy-root", str(tmp_path), "--receipt", str(receipt_path),
    ])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert code == 2
    assert receipt["result"]["status"] == "incomplete"
    assert receipt["result"]["scenarios"] == []
    assert receipt["reference"]["policy_digest"] is None
    failed_path = "scripts/apply.sh" if profiles_present else "profiles"
    assert any(failed_path in message for message in receipt["result"]["diagnostics"])


def test_installed_suite_does_not_expose_source_runtime_fallback(tmp_path: Path, monkeypatch) -> None:
    source_root = tmp_path / "source"
    runtime_root = tmp_path / "runtime"
    source_root.mkdir()
    runtime_root.mkdir()
    (source_root / "toolsets.py").write_text("VALUE = 'source'\n", encoding="utf-8")
    test_file = source_root / "test_installed.py"
    test_file.write_text(
        "import unittest\n"
        "import toolsets\n"
        "\n"
        "class InstalledRuntimeTest(unittest.TestCase):\n"
        "    def test_source_only_module_is_not_available(self):\n"
        "        self.assertEqual(toolsets.VALUE, 'source')\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(verify_kanban_lifecycle, "REPO_ROOT", source_root)
    monkeypatch.setattr(verify_kanban_lifecycle, "TEST_FILE", test_file)
    monkeypatch.setattr(verify_kanban_lifecycle, "TEST_MODULE", "test_installed")

    code, output, runner, pythonpath = verify_kanban_lifecycle._run_suite(
        "installed", runtime_root, sys.executable, None,
    )

    assert code != 0
    assert pythonpath == str(runtime_root)
    assert "FAILED (errors=1)" in output
    assert runner == "python -c <installed conformance runner>"


def test_policy_digest_excludes_untracked_profile_state(tmp_path: Path) -> None:
    for relative in (
        "scripts/apply.sh",
        "scripts/verify.sh",
        "scripts/verify_lifecycle.py",
        "scripts/remnic_plugin.py",
        "scripts/remnic-project-context",
        "scripts/hermes-github",
        "omp/remnic-project-context.ts",
        "admin/loota-dnf",
        "admin/install-package-helper",
        "plugins/consequence_guard/plugin.yaml",
        "plugins/consequence_guard/__init__.py",
        "profiles/cto/SOUL.md",
        "profiles/cto/config.yaml",
    ):
        artifact = tmp_path / relative
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(relative, encoding="utf-8")
    digest = verify_kanban_lifecycle._policy_digest(tmp_path)
    for relative in (
        ".env", "profiles/cto/.env", "profiles/cto/sessions.json",
        "profiles/cto/sessions/session.json", "profiles/cto/untracked.txt",
    ):
        state = tmp_path / relative
        state.parent.mkdir(parents=True, exist_ok=True)
        state.write_text("untracked fixture state", encoding="utf-8")
    assert verify_kanban_lifecycle._policy_digest(tmp_path) == digest
    (tmp_path / "profiles/cto/config.yaml").write_text("changed policy", encoding="utf-8")
    assert verify_kanban_lifecycle._policy_digest(tmp_path) != digest

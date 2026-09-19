"""Regression tests for layered lifecycle runtime selection."""

from __future__ import annotations

import importlib.util
import json
import runpy
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


def test_policy_failure_writes_incomplete_receipt(tmp_path: Path) -> None:
    receipt_path = tmp_path / "receipt.json"
    code = verify_kanban_lifecycle.main([
        "--layer", "source", "--policy-root", str(tmp_path), "--receipt", str(receipt_path),
    ])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert code == 2
    assert receipt["result"]["status"] == "incomplete"
    assert receipt["result"]["scenarios"] == []
    assert receipt["reference"]["policy_digest"] is None
    assert any("scripts/verify_lifecycle.py" in message
               for message in receipt["result"]["diagnostics"])
    assert verify_kanban_lifecycle._policy_digest(None) is None


def test_installed_suite_runs_both_modules_without_source_runtime_fallback(
    tmp_path: Path, monkeypatch,
) -> None:
    source_root = tmp_path / "source"
    runtime_root = tmp_path / "runtime"
    test_package = source_root / "tests" / "hermes_cli"
    test_package.mkdir(parents=True)
    runtime_root.mkdir()
    (source_root / "tests" / "__init__.py").write_text("", encoding="utf-8")
    (test_package / "__init__.py").write_text("", encoding="utf-8")
    (source_root / "toolsets.py").write_text("VALUE = 'source'\n", encoding="utf-8")
    (runtime_root / "toolsets.py").write_text("VALUE = 'runtime'\n", encoding="utf-8")
    first_file = test_package / "test_installed.py"
    first_file.write_text(
        "import sys\n"
        "import unittest\n"
        "from pathlib import Path\n"
        "import toolsets\n"
        "\n"
        "class InstalledRuntimeTest(unittest.TestCase):\n"
        "    def test_runtime_module_wins_over_source_checkout(self):\n"
        "        source_root = Path(__file__).resolve().parents[2]\n"
        "        self.assertNotIn(str(source_root), sys.path)\n"
        "        self.assertEqual(toolsets.VALUE, 'runtime')\n",
        encoding="utf-8",
    )
    helper_file = test_package / "kanban_conformance_fixture.py"
    helper_file.write_text(
        "import unittest\n"
        "\n"
        "class KanbanConformanceFixture(unittest.TestCase):\n"
        "    pass\n",
        encoding="utf-8",
    )
    second_file = test_package / "test_installed_upgrade.py"
    second_file.write_text(
        "from tests.hermes_cli import kanban_conformance_fixture as MODULE\n"
        "\n"
        "class InstalledUpgradeTest(MODULE.KanbanConformanceFixture):\n"
        "    def test_second_module_does_not_expose_source_runtime(self):\n"
        "        import source_only_runtime\n"
        "        self.assertIsNotNone(source_only_runtime)\n",
        encoding="utf-8",
    )
    (source_root / "source_only_runtime.py").write_text(
        "VALUE = 'source'\n", encoding="utf-8",
    )
    monkeypatch.setattr(verify_kanban_lifecycle, "REPO_ROOT", source_root)
    monkeypatch.setattr(
        verify_kanban_lifecycle, "TEST_FILES", (first_file, second_file),
    )

    code, output, runner, pythonpath = verify_kanban_lifecycle._run_suite(
        "installed", runtime_root, sys.executable, None,
    )

    assert code != 0
    assert pythonpath == str(runtime_root)
    assert "test_runtime_module_wins_over_source_checkout" in output
    assert "test_second_module_does_not_expose_source_runtime" in output
    assert "FAILED (errors=1)" in output
    assert runner == "python -c <installed conformance runner>"


def test_policy_digest_uses_trusted_policy_owner_coverage(tmp_path: Path) -> None:
    owner_path = tmp_path / "scripts" / "verify_lifecycle.py"
    helper_path = tmp_path / "scripts" / "new_policy_helper.py"
    owner_path.parent.mkdir(parents=True)
    owner_path.write_text(
        "from hashlib import sha256\n"
        "from pathlib import Path\n"
        "\n"
        "def policy_digest(policy_root):\n"
        "    helper = Path(policy_root) / 'scripts' / 'new_policy_helper.py'\n"
        "    return sha256(helper.read_bytes()).hexdigest()\n",
        encoding="utf-8",
    )
    helper_path.write_text("VALUE = 1\n", encoding="utf-8")
    owner = runpy.run_path(str(owner_path))
    owner_digest = owner["policy_digest"](tmp_path)
    assert verify_kanban_lifecycle._policy_digest(tmp_path) == owner_digest

    helper_path.write_text("VALUE = 2\n", encoding="utf-8")
    changed_owner_digest = owner["policy_digest"](tmp_path)
    assert changed_owner_digest != owner_digest
    assert verify_kanban_lifecycle._policy_digest(tmp_path) == changed_owner_digest


def test_policy_digest_rejects_malformed_owner_digest(tmp_path: Path) -> None:
    owner_path = tmp_path / "scripts" / "verify_lifecycle.py"
    owner_path.parent.mkdir(parents=True)
    owner_path.write_text(
        "def policy_digest(policy_root):\n"
        "    return 'A' * 64\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="scripts/verify_lifecycle.py"):
        verify_kanban_lifecycle._policy_digest(tmp_path)

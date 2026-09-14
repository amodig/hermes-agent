"""Regression tests for layered lifecycle runtime selection."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


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

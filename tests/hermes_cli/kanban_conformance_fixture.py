"""Shared fixtures for the Kanban lifecycle conformance suites."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_runtime as runtime
from hermes_cli import kanban_runtime_generation as generations
from hermes_cli.kanban_runtime import runtime_identity


ROOT = Path(__file__).resolve().parents[2]
RUNTIME_ROOT = Path(os.environ.get("HERMES_CONFORMANCE_RUNTIME_ROOT", ROOT)).resolve()
FIXTURE = ROOT / "tests" / "fixtures" / "kanban_lifecycle_v0.json"


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _make_runtime_fixture(root: Path) -> None:
    """Small real install: production bootstrap, dynamic imports, and resources."""
    package = root / "hermes_cli"
    package.mkdir(parents=True)
    for name in ("kanban_runtime.py", "kanban_runtime_generation.py"):
        shutil.copy2(RUNTIME_ROOT / "hermes_cli" / name, package / name)
    (package / "__init__.py").write_text('__version__ = "fixture"\n', encoding="utf-8")
    (root / "fixture_early.py").write_text("value = 1\n", encoding="utf-8")
    (root / "fixture_lazy.py").write_text("value = 1\n", encoding="utf-8")
    dependency_root = root.parent / "site-packages"
    dependency_root.mkdir(exist_ok=True)
    for name in ("third_party_early", "third_party_dynamic"):
        dependency = dependency_root / name
        dependency.mkdir()
        (dependency / "__init__.py").write_text("value = 1\n", encoding="utf-8")
        (dependency / "data.bin").write_bytes(b"dependency-v1")
    for directory, _env_var in runtime._RUNTIME_RESOURCE_ROOTS:
        resource = root / directory
        resource.mkdir(exist_ok=True)
        (resource / "marker.txt").write_text(f"{directory}:v1\n", encoding="utf-8")
    executable = root / "skills" / "helper"
    executable.write_text("#!/bin/sh\nprintf sealed-helper\n", encoding="utf-8")
    executable.chmod(0o755)
    (package / "main.py").write_text(
        """
import importlib
import json
import os
import sys
import time
from pathlib import Path
from hermes_cli import kanban_runtime as runtime
import fixture_early
import third_party_early

runtime.worker_bootstrap_from_env()
runtime.worker_bootstrap_post_import(wait_for_grant=False)
assert os.environ.get("HERMES_KANBAN_RUNTIME_GRANTED") != "1"
runtime.worker_bootstrap_after_constructor()
release = os.environ.get("HERMES_TEST_RUNTIME_RELEASE")
while release and not Path(release).exists():
    time.sleep(0.02)
lazy = importlib.import_module("fixture_" + "lazy")
sdk = importlib.import_module(os.environ.get("HERMES_TEST_SDK", "third_party_dynamic"))
payload = {
    "identity": runtime.runtime_identity().as_dict(),
    "early": [fixture_early.value, third_party_early.value],
    "lazy": [lazy.value, sdk.value],
    "dependency_data": (Path(sdk.__file__).parent / "data.bin").read_text(),
    "locations": [fixture_early.__file__, third_party_early.__file__, lazy.__file__, sdk.__file__],
    "resources": {
        directory: (Path(os.environ[env_var]) / "marker.txt").read_text()
        for directory, env_var in runtime._RUNTIME_RESOURCE_ROOTS
    },
    "argv": sys.argv[1:],
    "cwd": os.getcwd(),
    "profile": os.environ.get("HERMES_PROFILE"),
    "home": os.environ.get("HERMES_HOME"),
    "task": os.environ.get("HERMES_KANBAN_TASK"),
    "board": os.environ.get("HERMES_KANBAN_BOARD"),
    "run": os.environ.get("HERMES_KANBAN_RUN_ID"),
    "claim": os.environ.get("HERMES_KANBAN_CLAIM_LOCK"),
    "granted": os.environ.get("HERMES_KANBAN_RUNTIME_GRANTED"),
    "secret": os.environ.get("ANTHROPIC_API_KEY"),
}
if Path("/proc/self/cgroup").exists():
    payload["cgroup"] = Path("/proc/self/cgroup").read_text()
Path(os.environ["HERMES_TEST_RUNTIME_RECEIPT"]).write_text(json.dumps(payload))
""",
        encoding="utf-8",
    )


def _prepare_fixture_generation(
    root: Path, prepare=generations.prepare_runtime_generation, *, workspace=None, profile_home=None,
    project_plugins_enabled: bool | None = None,
):
    resources = {env_var: str(root / directory) for directory, env_var in runtime._RUNTIME_RESOURCE_ROOTS}
    with patch.dict(os.environ, resources), patch.object(
        generations, "_runtime_import_roots", return_value=[root.parent / "site-packages"],
    ):
        return prepare(
            runtime_identity(root), workspace=workspace, profile_home=profile_home,
            project_plugins_enabled=project_plugins_enabled,
        )


def _wait_for_receipt(path: Path) -> dict:
    deadline = time.monotonic() + 10
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    return json.loads(path.read_text(encoding="utf-8"))


class KanbanConformanceFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.home = tempfile.TemporaryDirectory(prefix="kanban-conformance-home-")
        self.storage = patch.object(
            generations, "_runtime_storage_root",
            return_value=Path(self.home.name) / "runtime-storage",
        )
        self.storage.start()
        self.env = patch.dict(
            "os.environ",
            {
                "HERMES_HOME": self.home.name,
                "HERMES_KANBAN_HOME": self.home.name,
                "HERMES_PROFILE": "cto",
            },
            clear=False,
        )
        self.env.start()
        self.conn = sqlite3.connect(":memory:")
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(kb.SCHEMA_SQL)
        kb._ensure_lifecycle_schema(self.conn)
        kb._ensure_goal_revision_schema(self.conn)

    def tearDown(self) -> None:
        self.conn.close()
        self.env.stop()
        self.storage.stop()
        self.home.cleanup()

    def _task(self, task_id: str):
        task = kb.get_task(self.conn, task_id)
        self.assertIsNotNone(task)
        return task


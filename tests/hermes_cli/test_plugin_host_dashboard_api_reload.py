"""API-only dashboard (``dashboard/plugin_api.py``) freshness in the real plugin host.

Such a plugin has no ``plugin.yaml``/``__init__.py``, so it is never loaded as a general
plugin and never enters the host's ``plugin_paths``: manager unload cannot evict its cached
FastAPI app. The cache must therefore be invalidated by the *source content* — an update that
preserves byte size and mtime included — while leaving sibling dashboards running and exiting
each entered lifespan exactly once (on replacement and again on final host shutdown).
"""

import json
import os

import pytest

from hermes_cli import plugins as plugins_mod
from tests.hermes_cli.test_plugin_host import _home_with_plugins


# The api file is deliberately the ONLY thing present: no manifest, no plugin.yaml, no __init__.py.
API_SOURCE = '''
import itertools, os, uuid
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import APIRouter

ready = False
token = uuid.uuid4().hex  # fresh per module execution: a re-import mints a new one

@asynccontextmanager
async def lifespan(app):
    global ready
    marker = Path(os.environ["HERMES_HOME"]) / (Path(__file__).parent.parent.name + "-lifecycle")
    with marker.open("a") as handle:
        handle.write("start\\n")
    ready = True
    try:
        yield
    finally:
        ready = False
        with marker.open("a") as handle:
            handle.write("stop\\n")

router = APIRouter(lifespan=lifespan)
requests = itertools.count(1)

@router.get("/")
def status():
    assert ready, "dashboard startup has not run"
    return {"value": "before", "pid": os.getpid(), "requests": next(requests), "token": token}
'''


@pytest.mark.platforms("any")
def test_api_only_dashboard_reloads_on_content_change_without_resetting_siblings(tmp_path, monkeypatch):
    home = _home_with_plugins(tmp_path, monkeypatch, {})
    for name in ("apireload", "apisibling"):
        dashboard_dir = home / "plugins" / name / "dashboard"
        dashboard_dir.mkdir(parents=True)
        (dashboard_dir / "api.py").write_text(API_SOURCE, encoding="utf-8")
    manager = plugins_mod.get_plugin_manager()
    host = manager._plugin_host()

    def request(name):
        result = host.asgi_request(name, str(home / "plugins" / name / "dashboard"), "api.py",
                                   "GET", "/", "", [], b"")
        assert result["status"] == 200, result
        return json.loads(result["body"])

    try:
        first = request("apireload")
        assert first["pid"] != os.getpid()  # served by the child, not this process
        assert first["value"] == "before" and first["requests"] == 1
        sibling = request("apisibling")
        assert sibling["value"] == "before" and sibling["requests"] == 1
        assert sibling["token"] != first["token"]  # each dashboard imports its own module

        # Rewrite the source, then restore the exact mtime/size: only content identity can tell.
        api_file = home / "plugins" / "apireload" / "dashboard" / "api.py"
        before_stat = api_file.stat()
        api_file.write_text(API_SOURCE.replace("before", "after!"), encoding="utf-8")
        os.utime(api_file, ns=(before_stat.st_atime_ns, before_stat.st_mtime_ns))
        after_stat = api_file.stat()
        assert after_stat.st_size == before_stat.st_size
        assert after_stat.st_mtime_ns == before_stat.st_mtime_ns

        # Fresh module + router + lifespan without any unload/restart of the host: the value
        # changed and the module re-executed (new token, request counter back to 1).
        reloaded = request("apireload")
        assert reloaded["value"] == "after!" and reloaded["requests"] == 1
        assert reloaded["token"] != first["token"]
        assert reloaded["pid"] == first["pid"]
        # Old lifespan exited exactly once, fresh one entered once: start/stop/start.
        assert (home / "apireload-lifecycle").read_text().splitlines() == ["start", "stop", "start"]
        # Sibling untouched: same module (token unchanged), its counter keeps counting, no churn.
        assert request("apisibling") == {**sibling, "requests": 2}
        assert (home / "apisibling-lifecycle").read_text().splitlines() == ["start"]

        assert host.alive
    finally:
        host.shutdown()

    # Final shutdown closes the live (post-reload) app and the sibling exactly once each.
    assert (home / "apireload-lifecycle").read_text().splitlines() == ["start", "stop", "start", "stop"]
    assert (home / "apisibling-lifecycle").read_text().splitlines() == ["start", "stop"]

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
                                   "GET", f"http://testserver/api/plugins/{name}/",
                                   f"/api/plugins/{name}", [], b"")
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


@pytest.mark.platforms("any")
def test_dashboard_reload_drains_requests_without_blocking_siblings(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    import time

    home = _home_with_plugins(tmp_path, monkeypatch, {})
    held_source = API_SOURCE + '''
import asyncio
@router.get("/hold")
async def hold():
    home = Path(os.environ["HERMES_HOME"])
    (home / "holding-request").touch()
    while not (home / "release-request").exists():
        await asyncio.sleep(0.01)
    assert ready, "lifespan resources closed during request"
    return {"value": "before"}
'''
    for name in ("drain", "sibling"):
        dashboard_dir = home / "plugins" / name / "dashboard"
        dashboard_dir.mkdir(parents=True)
        (dashboard_dir / "api.py").write_text(held_source, encoding="utf-8")
    host = plugins_mod.get_plugin_manager()._plugin_host()
    pool = ThreadPoolExecutor(max_workers=2)
    replacement_started = threading.Event()

    def request(name, path="/"):
        if name == "drain" and path == "/":
            replacement_started.set()
        response = host.asgi_request(name, str(home / "plugins" / name / "dashboard"),
                                     "api.py", "GET", f"http://testserver/api/plugins/{name}{path}",
                                     f"/api/plugins/{name}", [], b"")
        assert response["status"] == 200
        return json.loads(response["body"])

    try:
        held = pool.submit(request, "drain", "/hold")
        deadline = time.monotonic() + 15
        while not (home / "holding-request").exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert (home / "holding-request").exists()
        (home / "plugins" / "drain" / "dashboard" / "api.py").write_text(
            held_source.replace("before", "after!"), encoding="utf-8")
        replacement = pool.submit(request, "drain")
        assert replacement_started.wait(10)
        with pytest.raises(TimeoutError):
            replacement.result(timeout=0.5)
        assert request("sibling")["value"] == "before"
        assert (home / "drain-lifecycle").read_text().splitlines() == ["start"]
        (home / "release-request").touch()
        assert held.result(timeout=10) == {"value": "before"}
        assert replacement.result(timeout=10)["value"] == "after!"
        assert (home / "drain-lifecycle").read_text().splitlines() == ["start", "stop", "start"]
    finally:
        (home / "release-request").touch()
        pool.shutdown(wait=True)
        host.shutdown()


@pytest.mark.platforms("any")
@pytest.mark.parametrize("operation", ["url", "gzip"])
def test_parent_dashboard_bridge_preserves_external_urls_and_encoded_bytes(tmp_path, monkeypatch, operation):
    import asyncio
    import gzip

    import httpx
    from fastapi import FastAPI
    from hermes_cli.web_server_dashboard import _mount_hosted_plugin_apis

    home = _home_with_plugins(tmp_path, monkeypatch, {"bridge": ""})
    dashboard_dir = home / "plugins" / "bridge" / "dashboard"
    dashboard_dir.mkdir()
    (dashboard_dir / "manifest.json").write_text(json.dumps({"name": "bridge", "api": "api.py"}))
    (dashboard_dir / "api.py").write_text('''
import gzip, json
from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse, Response
router = APIRouter()
@router.get("/redirect")
def redirect(request: Request):
    return RedirectResponse(str(request.url_for("target", item="x")) + "?q=one%2Ftwo&q=3")
@router.get("/target/{item}", name="target")
def target(request: Request, item: str):
    return {"url": str(request.url), "base": str(request.base_url)}
@router.get("/gzip")
def compressed(request: Request):
    assert "gzip" in request.headers["accept-encoding"]
    return Response(gzip.compress(json.dumps({"value": "compress me" * 100}).encode()),
                    media_type="application/json", headers={"content-encoding": "gzip"})
''')
    app = FastAPI()
    _mount_hosted_plugin_apis(app)
    host = plugins_mod.get_plugin_manager()._plugin_host()
    root = "https://example.test:9443/external/api/plugins/bridge"

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app, root_path="/external"),
                                     base_url="https://example.test:9443") as client:
            if operation == "url":
                response = await client.get(root + "/redirect")
                target = root + "/target/x?q=one%2Ftwo&q=3"
                assert response.status_code == 307
                assert response.headers["location"] == target
                followed = await client.get(response.headers["location"])
                assert followed.status_code == 200
                assert followed.json() == {"url": target, "base": root + "/"}
            else:
                async with client.stream("GET", root + "/gzip", headers={"Accept-Encoding": "gzip"}) as response:
                    raw = b"".join([chunk async for chunk in response.aiter_raw()])
                    assert response.status_code == 200
                    assert response.headers["content-encoding"] == "gzip"
                    assert int(response.headers["content-length"]) == len(raw)
                    assert json.loads(gzip.decompress(raw)) == {"value": "compress me" * 100}

    try:
        asyncio.run(exercise())
    finally:
        host.shutdown()


@pytest.mark.platforms("any")
@pytest.mark.parametrize("name", ["shared", "secondary-only", "launch-only"])
def test_hosted_dashboard_routes_use_only_the_requesting_profile(tmp_path, monkeypatch, name):
    import asyncio

    import httpx
    from fastapi import FastAPI
    from hermes_cli import web_server, web_server_dashboard
    from hermes_cli.web_server_profiles import _config_profile_scope

    home = _home_with_plugins(tmp_path, monkeypatch, {"shared": "", "launch-only": ""})
    secondary = home / "profiles" / "secondary"
    secondary.mkdir(parents=True)
    (secondary / "config.yaml").write_text(
        "plugins:\n  isolation: host\n  enabled: [shared, secondary-only, launch-only]\n")
    for owner, owner_home, names in (("launch", home, ["shared", "launch-only"]),
                                      ("secondary", secondary, ["shared", "secondary-only"])):
        for plugin_name in names:
            dashboard_dir = owner_home / "plugins" / plugin_name / "dashboard"
            dashboard_dir.mkdir(parents=True)
            (dashboard_dir / "manifest.json").write_text(json.dumps({"name": plugin_name, "api": "api.py"}))
            (dashboard_dir / "api.py").write_text(
                "from fastapi import APIRouter\nrouter = APIRouter()\n"
                f"@router.get('/')\ndef owner(): return {{'owner': {owner!r}}}\n")
    app = FastAPI()
    monkeypatch.setattr(web_server, "app", app)
    monkeypatch.setattr(web_server, "_dashboard_plugins_cache", None)
    web_server_dashboard._mount_plugin_api_routes()
    with _config_profile_scope("secondary"):
        secondary_host = plugins_mod.get_plugin_manager()._plugin_host()
    launch_host = plugins_mod.get_plugin_manager()._plugin_host()

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
            response = await client.get(f"/api/plugins/{name}/?profile=secondary")
            assert response.status_code == (404 if name == "launch-only" else 200)
            if response.status_code == 200:
                assert response.json() == {"owner": "secondary"}
            launch = await client.get(f"/api/plugins/{name}/")
            assert launch.status_code == (404 if name == "secondary-only" else 200)
            if launch.status_code == 200:
                assert launch.json() == {"owner": "launch"}

    try:
        asyncio.run(exercise())
    finally:
        secondary_host.shutdown()
        launch_host.shutdown()

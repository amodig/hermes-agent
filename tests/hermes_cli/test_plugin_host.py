"""plugins.isolation: host — third-party plugins run in a per-profile host process, reached only via ctx."""

import json
import os
import sys
import time

import hermes_yaml as yaml
import pytest

from hermes_cli import plugins as plugins_mod
from hermes_cli.plugins import PluginManager

PROBE_PLUGIN = '''
import json, os
from agent.image_gen_provider import ImageGenProvider

class Painter(ImageGenProvider):
    @property
    def name(self):
        return "hostpainter"
    def generate(self, prompt, aspect_ratio="landscape", **kw):
        return {"success": True, "image": f"{prompt}@{os.getpid()}"}

SEEN = []

def register(ctx):
    ctx.register_tool(name="hostprobe_pid", toolset="hostprobe", schema={"name": "hostprobe_pid",
        "description": "d", "parameters": {"type": "object", "properties": {}}},
        handler=lambda args, **kw: json.dumps({"pid": os.getpid(), "seen": SEEN, "x": args.get("x")}))
    ctx.register_tool(name="hostprobe_nested", toolset="hostprobe", schema={"name": "hostprobe_nested",
        "description": "d", "parameters": {"type": "object", "properties": {}}},
        handler=lambda args, **kw: ctx.dispatch_tool("hostprobe_pid", {"x": "nested"}))
    ctx.register_tool(name="hostprobe_crash", toolset="hostprobe", schema={"name": "hostprobe_crash",
        "description": "d", "parameters": {"type": "object", "properties": {}}},
        handler=lambda args, **kw: os._exit(3))
    ctx.register_hook("post_tool_call", lambda tool_name=None, **kw: SEEN.append(tool_name))
    ctx.register_image_gen_provider(Painter())
'''

PLATFORM_PLUGIN = '''
def register(ctx):
    ctx.register_platform("x", "X", adapter_factory=lambda cfg: None, check_fn=lambda: True)
'''


def _home_with_plugins(tmp_path, monkeypatch, plugins: dict, isolation="host"):
    home = tmp_path / ".hermes"
    home.mkdir(parents=True, exist_ok=True)
    for name, body in plugins.items():
        plugin_dir = home / "plugins" / name
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "plugin.yaml").write_text(yaml.safe_dump({"name": name, "version": "1.0"}), encoding="utf-8")
        (plugin_dir / "__init__.py").write_text(body, encoding="utf-8")
    (home / "config.yaml").write_text(yaml.safe_dump(
        {"plugins": {"enabled": list(plugins), "isolation": isolation}}), encoding="utf-8")
    empty_bundled = tmp_path / "bundled"
    empty_bundled.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "os-home"))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(plugins_mod, "get_bundled_plugins_dir", lambda: empty_bundled)
    return home


@pytest.mark.platforms("any")
@pytest.mark.parametrize("operation", ["spawn_task", "on_unload"])
def test_hosted_context_task_and_cleanup_contracts(tmp_path, monkeypatch, operation):
    bodies = {
        "spawn_task": '''
async def register(ctx):
    async def work():
        return 42
    task = ctx.spawn_task(work(), name="contract-task")
    assert await task == 42
    assert task.done() and task.get_name() == "contract-task"
''',
        "on_unload": '''
from pathlib import Path
import os
def register(ctx):
    marker = Path(os.environ["HERMES_HOME"]) / "cleanup"
    def cleanup():
        with marker.open("a") as handle:
            handle.write("once\\n")
    handle = ctx.on_unload(cleanup)
    assert handle.active
    handle.dispose()
    handle.dispose()
    assert not handle.active
    ctx.on_unload(cleanup)
''',
    }
    home = _home_with_plugins(tmp_path, monkeypatch, {"contract": bodies[operation]})
    manager = PluginManager()
    try:
        manager.discover_and_load()
        assert manager._plugins["contract"].error is None
        if operation == "on_unload":
            assert (home / "cleanup").read_text().splitlines() == ["once"]
        manager.unload()
        if operation == "on_unload":
            assert (home / "cleanup").read_text().splitlines() == ["once", "once"]
    finally:
        manager._plugin_host().shutdown()


@pytest.mark.platforms("posix")  # os._exit / SIGKILL process semantics
def test_plugin_runs_out_of_process_with_ctx_round_trips_and_survives_host_crash(tmp_path, monkeypatch):
    from agent import image_gen_registry
    from agent.image_gen_provider import ImageGenProvider
    from tools.registry import registry

    _home_with_plugins(tmp_path, monkeypatch, {"hostprobe": PROBE_PLUGIN, "platformprobe": PLATFORM_PLUGIN})
    manager = PluginManager()
    manager.discover_and_load()
    try:
        # The plugin's code never entered this interpreter, yet its registrations are ordinary.
        assert not any("hostprobe" in name for name in sys.modules)
        assert manager._plugins["hostprobe"].error is None
        first = json.loads(registry.dispatch("hostprobe_pid", {"x": 1}, scope=manager.scope_key))
        host_pid = first["pid"]
        assert host_pid != os.getpid()
        # A handler calling back into Hermes while Hermes waits on it (nested ctx.dispatch_tool).
        nested = json.loads(registry.dispatch("hostprobe_nested", {}, scope=manager.scope_key))
        assert nested["pid"] == host_pid and nested["x"] == "nested"
        # Hooks fire in the host; provider objects arrive as instances of the ABC Hermes checks.
        manager.invoke_hook("post_tool_call", tool_name="read_file", args={}, result="ok")
        assert "read_file" in json.loads(registry.dispatch("hostprobe_pid", {}, scope=manager.scope_key))["seen"]
        painter = image_gen_registry.get_provider("hostpainter")
        assert isinstance(painter, ImageGenProvider)
        assert painter.generate("cat") == {"success": True, "image": f"cat@{host_pid}"}
        # Live-object ctx surfaces fail the plugin with the boundary's reason, not a crash.
        assert "register_platform" in str(manager._plugins["platformprobe"].error)

        # A plugin killing its host costs one tool error; Hermes keeps running and the host restarts.
        crashed = registry.dispatch("hostprobe_crash", {}, scope=manager.scope_key)
        assert "plugin host" in crashed
        deadline, result = time.monotonic() + 15, ""
        while time.monotonic() < deadline:
            result = registry.dispatch("hostprobe_pid", {}, scope=manager.scope_key)
            if '"pid"' in result:
                break
            time.sleep(0.2)
        assert json.loads(result)["pid"] not in {host_pid, os.getpid()}
    finally:
        manager.unload()
        manager._plugin_host().shutdown()


@pytest.mark.platforms("any")
def test_failed_registration_cancels_background_work_and_runs_cleanup(tmp_path, monkeypatch):
    home = _home_with_plugins(tmp_path, monkeypatch, {"failedprobe": '''
import asyncio, os
from pathlib import Path

async def register(ctx):
    home = Path(os.environ["HERMES_HOME"])
    async def background():
        while not (home / "release").exists():
            await asyncio.sleep(0.01)
        (home / "unexpected-side-effect").touch()
    ctx.spawn_task(background())
    ctx.on_unload(lambda: (home / "cleaned").touch())
    await asyncio.sleep(0.05)
    raise ValueError("registration failed")
'''})
    manager = PluginManager()
    try:
        manager.discover_and_load()
        assert "registration failed" in str(manager._plugins["failedprobe"].error)
        assert (home / "cleaned").exists()
        (home / "release").touch()
        time.sleep(0.1)
        assert not (home / "unexpected-side-effect").exists()
        assert manager._plugin_host().alive
    finally:
        manager.unload()
        manager._plugin_host().shutdown()


@pytest.mark.platforms("any")
@pytest.mark.parametrize("stall", ["register", "async_register", "import"])
def test_load_deadline_retires_child_work_without_stopping_siblings(tmp_path, monkeypatch, stall):
    import threading

    from hermes_cli import plugins_loader
    from tools.registry import registry

    home = _home_with_plugins(tmp_path, monkeypatch, {"deadlineobserver": '''
import asyncio, json, os, sys

def register(ctx):
    async def inspect(args, **kw):
        await asyncio.sleep(0)  # let any wrongly accepted late task reach its side effect
        runtime = ctx._runtime
        key = "deadlineprobe"
        with runtime._lock:
            retained = [name for name in (
                "contexts", "loading", "modules", "plugin_paths", "background_tasks", "unload_callbacks"
            ) if key in getattr(runtime, name)]
            if key in runtime.owners.values():
                retained.append("refs")
        return json.dumps({"pid": os.getpid(), "retained": retained,
            "imported": any(name == args["module"] or name.startswith(args["module"] + ".")
                            for name in sys.modules.copy())})
    ctx.register_tool(name="deadline_inspect", toolset="deadlineobserver",
        schema={"name": "deadline_inspect", "description": "Inspect the child",
                "parameters": {"type": "object", "properties": {}}},
        handler=inspect, is_async=True)
'''})
    manager = plugins_mod.get_plugin_manager()
    host = manager._plugin_host()
    plugin_dir = home / "plugins" / "deadlineprobe"
    plugin_dir.mkdir()
    (plugin_dir / "__init__.py").write_text(f"STALL = {stall!r}\n" + '''
import asyncio, os, time
from pathlib import Path

home = Path(os.environ["HERMES_HOME"])
if STALL == "import":
    (home / "stall-started").touch()
    while not (home / "finish-register").exists():
        time.sleep(0.01)

async def background():
    (home / "background-started").touch()
    try:
        while not (home / "release-background").exists():
            await asyncio.sleep(0.01)
        (home / "unexpected-background").touch()
    finally:
        (home / "background-stopped").touch()

async def late_background():
    (home / "unexpected-late-task").touch()

def late(ctx):
    for call in (
        lambda: ctx.spawn_task(late_background()),
        lambda: ctx.on_unload(lambda: (home / "unexpected-late-cleanup").touch()),
        lambda: ctx.register_tool(name="deadline_late", toolset="deadlineprobe",
            schema={"name": "deadline_late", "description": "Late registration",
                    "parameters": {"type": "object", "properties": {}}},
            handler=lambda args, **kw: "late"),
    ):
        try:
            call()
        except RuntimeError:
            pass
    (home / "late-attempted").touch()

async def async_register(ctx):
    try:
        while not (home / "finish-register").exists():
            await asyncio.sleep(0.01)
    finally:
        late(ctx)  # also exercise code running as cancellation unwinds an async register()

def register(ctx):
    if STALL == "import":
        (home / "unexpected-register").touch()
        return
    ctx.on_unload(lambda: ctx.dispatch_tool("deadline_cleanup", {}))
    ctx.spawn_task(background())
    (home / "stall-started").touch()
    if STALL == "async_register":
        return async_register(ctx)
    while not (home / "finish-register").exists():
        time.sleep(0.01)
    late(ctx)
''', encoding="utf-8")
    manifest = plugins_mod.PluginManifest(name="deadlineprobe", source="user", path=str(plugin_dir))
    cleaned = []
    registry.register(
        name="deadline_cleanup", toolset="deadlineobserver",
        schema={"name": "deadline_cleanup", "description": "Cleanup",
                "parameters": {"type": "object", "properties": {}}},
        scope=manager.scope_key, handler=lambda args, **kw: cleaned.append(True) or "cleaned")
    load_done = threading.Event()

    def load():
        try:
            manager._load_plugin(manifest)
        finally:
            load_done.set()

    loader = threading.Thread(target=load, daemon=True)

    def await_marker(name):
        deadline = time.monotonic() + 15
        while not (home / name).exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert (home / name).exists(), name

    try:
        # Warm canonical discovery so later callbacks cannot load a second observer host.
        manager.discover_and_load()
        assert manager._plugins["deadlineobserver"].error is None
        pid = host.info["pid"]
        (home / "config.yaml").write_text(yaml.safe_dump({"plugins": {
            "enabled": ["deadlineobserver", "deadlineprobe"], "isolation": "host",
            "load_timeout_seconds": 5,
        }}), encoding="utf-8")
        loader.start()
        await_marker("stall-started")
        assert load_done.wait(15), "deadline cleanup waited on stalled import/register()"
        assert "timed out" in str(manager._plugins["deadlineprobe"].error)
        assert not manager._plugins["deadlineprobe"].enabled
        if stall != "import":
            assert cleaned == [True]  # cleanup could still call back through the parent context
            await_marker("background-stopped")
        if stall != "async_register":
            manager._load_plugin(manifest)
            assert "previous load" in str(manager._plugins["deadlineprobe"].error)
        assert host.alive and host.info["pid"] == pid
        module_name = manager._policy_module_name(manifest)
        assert json.loads(registry.dispatch(
            "deadline_inspect", {"module": module_name}, scope=manager.scope_key))["pid"] == pid
        (home / "finish-register").touch()
        (home / "release-background").touch()
        for worker in plugins_loader._ABANDONED_LOADERS:
            if worker.name == "plugin-load:deadlineprobe":
                worker.join(5)
                assert not worker.is_alive()
        if stall != "import":
            await_marker("late-attempted")
        state = json.loads(registry.dispatch(
            "deadline_inspect", {"module": module_name}, scope=manager.scope_key))
        assert state == {"pid": pid, "retained": [], "imported": False}
        assert "deadlineprobe" not in host._contexts
        assert not any(handle.plugin_key == "deadlineprobe" for handle in host._handles.values())
        assert "deadlineprobe" not in manager._ownership_ledger
        assert "deadline_late" not in manager._plugin_tool_names
        assert not list(home.glob("unexpected-*"))
        # Once the retired worker is gone, a fresh attempt at that same key is usable.
        (plugin_dir / "__init__.py").write_text('''
def register(ctx):
    ctx.register_tool(name="deadline_late", toolset="deadlineprobe",
        schema={"name": "deadline_late", "description": "Fresh registration",
                "parameters": {"type": "object", "properties": {}}},
        handler=lambda args, **kw: "reloaded")
''', encoding="utf-8")
        manager._load_plugin(manifest)
        assert manager._plugins["deadlineprobe"].error is None
        assert registry.dispatch("deadline_late", {}, scope=manager.scope_key) == "reloaded"
        assert host.alive and host.info["pid"] == pid
    finally:
        (home / "finish-register").touch()
        (home / "release-background").touch()
        if loader.ident is not None:
            loader.join(5)
        manager.unload()
        registry.deregister("deadline_cleanup", scope=manager.scope_key)
        host.shutdown()


@pytest.mark.platforms("any")
def test_load_deadline_retires_host_when_cleanup_callback_stalls(tmp_path, monkeypatch):
    from hermes_cli import plugin_host

    home = _home_with_plugins(tmp_path, monkeypatch, {"stalledcleanup": '''
import time
def register(ctx):
    ctx.on_unload(lambda: time.sleep(4))
    time.sleep(4)
'''})
    manager = PluginManager()
    host = manager._plugin_host()
    try:
        host.ensure_started()
        monkeypatch.setattr(plugin_host, "_SHUTDOWN_GRACE_SECS", 0.1)
        (home / "config.yaml").write_text(yaml.safe_dump({"plugins": {
            "enabled": ["stalledcleanup"], "isolation": "host", "load_timeout_seconds": 0.5,
        }}), encoding="utf-8")
        started = time.monotonic()
        manager.discover_and_load()
        assert time.monotonic() - started < 2
        assert "timed out" in str(manager._plugins["stalledcleanup"].error)
        assert not host.alive
        assert "stalledcleanup" not in host._contexts
    finally:
        host.shutdown()


@pytest.mark.platforms("any")
def test_unload_callback_can_dispatch_back_to_parent(tmp_path, monkeypatch):
    from tools.registry import registry
    _home_with_plugins(tmp_path, monkeypatch, {"cleanupprobe": '''
def register(ctx):
    ctx.on_unload(lambda: ctx.dispatch_tool("cleanup_target", {}))
'''})
    manager = PluginManager()
    cleaned = []
    def cleanup(args, **kwargs):
        cleaned.append(True)
        return "cleaned"
    registry.register(
        name="cleanup_target", toolset="cleanup",
        schema={"name": "cleanup_target", "description": "Cleanup",
                "parameters": {"type": "object", "properties": {}}},
        scope=manager.scope_key, handler=cleanup)
    try:
        manager.discover_and_load()
        assert manager._plugins["cleanupprobe"].error is None
        manager.unload()
        assert cleaned == [True]
        assert manager._plugin_host().alive
    finally:
        registry.deregister("cleanup_target", scope=manager.scope_key)
        manager._plugin_host().shutdown()


@pytest.mark.platforms("any")  # the host is a child process: its env/home resolution is per-OS
def test_isolation_host_keeps_every_user_import_path_out_of_process(tmp_path, monkeypatch):
    from hermes_cli.plugin_isolation_audit import audit_plugin_dir
    from plugins import plugin_loader

    home = _home_with_plugins(tmp_path, monkeypatch, {"platformprobe": PLATFORM_PLUGIN, "fine": PROBE_PLUGIN})
    # Category loaders' in-process import refuses user code under host isolation (the backstop
    # behind the memory / context-engine / cron host routes).
    import logging
    assert plugin_loader.load_plugin_module(
        "_hermes_user_x.fine", home / "plugins" / "fine", parents=("_hermes_user_x",),
        logger=logging.getLogger("t"), synthetic_namespace="_hermes_user_x") is None
    assert "_hermes_user_x.fine" not in sys.modules
    # The static audit reads the same boundary table the host enforces.
    assert audit_plugin_dir(home / "plugins" / "fine").verdict == "host"
    blocked = audit_plugin_dir(home / "plugins" / "platformprobe")
    assert blocked.verdict == "in_process" and "register_platform" in blocked.reasons[0]



def test_managed_scope_pins_host_isolation_over_the_profiles_own_config(tmp_path, monkeypatch):
    """The isolated party must not be able to opt out: an operator pin in the managed scope
    (/etc/hermes/config.yaml) wins over a profile config that says in_process."""
    from hermes_cli import managed_scope
    from tools.registry import registry

    _home_with_plugins(tmp_path, monkeypatch, {"hostprobe": PROBE_PLUGIN}, isolation="in_process")
    managed = tmp_path / "managed"
    managed.mkdir()
    (managed / "config.yaml").write_text(yaml.safe_dump({"plugins": {"isolation": "host"}}), encoding="utf-8")
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    managed_scope.invalidate_managed_cache()
    manager = PluginManager()
    manager.discover_and_load()
    try:
        assert not any("hostprobe" in name for name in sys.modules)
        result = json.loads(registry.dispatch("hostprobe_pid", {}, scope=manager.scope_key))
        assert result["pid"] != os.getpid()
    finally:
        host = getattr(manager, "_plugin_host_instance", None)
        if host is not None:
            host.shutdown()
        managed_scope.invalidate_managed_cache()

MODEL_PROVIDER_PLUGIN = '''
import os, time
from pathlib import Path
from providers import register_provider
from providers.base import ProviderProfile

with (Path(os.environ["HERMES_HOME"]) / "hostmodel-imports.txt").open("a", encoding="utf-8") as marker:
    marker.write(str(os.getpid()) + "\\n")
time.sleep(0.2)  # Keep concurrent first-use requests inside the import window.

class HostModel(ProviderProfile):
    def build_extra_body(self, *, session_id=None, **context):
        return {"pid": os.getpid(), "session_id": session_id}

register_provider(HostModel(name="hostmodel", base_url="https://hostmodel.example/v1", env_vars=("HOSTMODEL_KEY",)))
'''


@pytest.mark.platforms("any")  # the host is a child process: its env/home resolution is per-OS
def test_model_provider_profile_data_is_local_and_overrides_run_in_the_host(tmp_path, monkeypatch):
    import providers
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    home = _home_with_plugins(tmp_path, monkeypatch, {})
    plugin_dir = home / "plugins" / "model-providers" / "hostmodel"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text("name: hostmodel\nkind: model-provider\n", encoding="utf-8")
    (plugin_dir / "__init__.py").write_text(MODEL_PROVIDER_PLUGIN, encoding="utf-8")
    providers._discover_providers(bundled_only=True)
    barrier = Barrier(4)
    def first_lookup(index):
        barrier.wait(timeout=10)
        return providers.get_provider_profile("hostmodel")
    with ThreadPoolExecutor(max_workers=4) as pool:
        profiles = list(pool.map(first_lookup, range(4)))
    assert all(isinstance(profile, providers.ProviderProfile) for profile in profiles)
    profile = profiles[0]
    try:
        assert isinstance(profile, providers.ProviderProfile)
        # Discovery never started a host: data came from the cached, credential-free extraction.
        assert profile.base_url == "https://hostmodel.example/v1" and tuple(profile.env_vars) == ("HOSTMODEL_KEY",)
        assert list((home / "cache" / "plugin_host" / "model-providers").glob("hostmodel-*.json"))
        assert not any("hostmodel" in name for name in sys.modules)
        marker = home / "hostmodel-imports.txt"
        extraction_pids = marker.read_text(encoding="utf-8").splitlines()
        assert len(extraction_pids) == 1
        assert int(extraction_pids[0]) != os.getpid()
        from hermes_cli.plugin_isolation import user_plugin_host
        user_plugin_host().ensure_started()
        barrier = Barrier(4)
        def first_call(index):
            barrier.wait(timeout=10)
            return profile.build_extra_body(session_id=f"s{index}")
        with ThreadPoolExecutor(max_workers=4) as pool:
            bodies = list(pool.map(first_call, range(1, 5)))
        body = bodies[0]
        assert bodies == [{"pid": body["pid"], "session_id": f"s{index}"} for index in range(1, 5)]
        assert body["session_id"] == "s1" and body["pid"] != os.getpid()
        assert marker.read_text(encoding="utf-8").splitlines() == [*extraction_pids, str(body["pid"])]
        assert profile.build_extra_body(session_id="s2") == {"session_id": "s2", "pid": body["pid"]}
        assert marker.read_text(encoding="utf-8").splitlines() == [*extraction_pids, str(body["pid"])]
    finally:
        host = getattr(plugins_mod.get_plugin_manager(), "_plugin_host_instance", None)
        if host is not None:
            host.shutdown()


@pytest.mark.platforms("any")  # exercises the real credential-free extraction process and disk cache
def test_hosted_profile_temperature_survives_extraction(tmp_path, monkeypatch):
    from agent.transports.chat_completions import ChatCompletionsTransport
    from hermes_cli.plugin_host_profiles import load_hosted_profiles
    from providers.base import OMIT_TEMPERATURE

    home = _home_with_plugins(tmp_path, monkeypatch, {})
    plugin_dir = home / "plugins" / "model-providers" / "hosttemperature"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "__init__.py").write_text(
        "from providers import register_provider\n"
        "from providers.base import OMIT_TEMPERATURE, ProviderProfile\n"
        "for name, temperature in [('omit', OMIT_TEMPERATURE), ('default', None), "
        "('zero', 0.0), ('fixed', 0.65)]:\n"
        "    register_provider(ProviderProfile(name=name, fixed_temperature=temperature))\n",
        encoding="utf-8",
    )
    transport = ChatCompletionsTransport()
    for _ in range(2):  # first load extracts; the second decodes the persisted payload
        profiles = {p.name: p for p in load_hosted_profiles(plugin_dir, "_hermes_hosttemperature")}
        assert profiles["omit"].fixed_temperature is OMIT_TEMPERATURE
        for name, expected in (("omit", None), ("default", 0.4), ("zero", 0.0), ("fixed", 0.65)):
            kwargs = transport.build_kwargs(
                model="test-model", messages=[{"role": "user", "content": "Hi"}],
                provider_profile=profiles[name], temperature=0.4,
            )
            if name == "omit":
                assert "temperature" not in kwargs
            else:
                assert kwargs["temperature"] == expected
            json.dumps(kwargs)  # no opaque sentinel may leak into the API request


ASYNC_PLUGIN = '''
import asyncio, json
S = {"type": "object", "properties": {}}

def register(ctx):
    async def whoami(args, **kw):  # async plugin code calling back into Hermes
        return json.dumps({"inner": json.loads(ctx.dispatch_tool("session_probe", {})),
                           "acomplete_awaitable": asyncio.iscoroutinefunction(ctx.llm.acomplete)})
    ctx.register_tool(name="async_whoami", toolset="asyncprobe", schema={"name": "async_whoami",
        "description": "d", "parameters": S}, handler=whoami, is_async=True)
'''


@pytest.mark.platforms("any")  # the host is a child process: its env/home resolution is per-OS
def test_async_plugin_code_calls_back_in_the_callers_session(tmp_path, monkeypatch):
    import contextvars
    from tools.registry import registry

    session = contextvars.ContextVar("session", default="<unset>")
    session.set("host-creator")  # whoever first touches the host must not leak into later calls
    registry.register(name="session_probe", toolset="asyncprobe", schema={"name": "session_probe",
                      "description": "d", "parameters": {"type": "object", "properties": {}}},
                      handler=lambda args, **kw: json.dumps({"session": session.get()}))
    _home_with_plugins(tmp_path, monkeypatch, {"asyncprobe": ASYNC_PLUGIN})
    manager = PluginManager()
    manager.discover_and_load()
    try:
        def call_as(name):
            def run():
                session.set(name)
                return json.loads(registry.dispatch("async_whoami", {}, scope=manager.scope_key))
            return contextvars.copy_context().run(run)

        assert call_as("session-B") == {"inner": {"session": "session-B"}, "acomplete_awaitable": True}
        assert call_as("session-C")["inner"] == {"session": "session-C"}
    finally:
        registry.deregister("session_probe")
        manager.unload()
        manager._plugin_host().shutdown()


@pytest.mark.platforms("any")
@pytest.mark.parametrize("operation", ["unload", "reload"])
def test_hosted_background_task_is_cancelled_without_stopping_host(tmp_path, monkeypatch, operation):
    home = _home_with_plugins(tmp_path, monkeypatch, {"taskprobe": '''
import asyncio, os
from pathlib import Path

def record(event):
    with (Path(os.environ["HERMES_HOME"]) / "task-events").open("a") as handle:
        handle.write(event + "\\n")

async def background():
    record("started")
    try:
        await asyncio.Event().wait()
    finally:
        record("stopped")

def register(ctx):
    ctx.spawn_task(background(), name="taskprobe-background")
'''})
    manager = PluginManager()
    host = manager._plugin_host()
    marker = home / "task-events"
    def await_events(started, stopped):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            events = marker.read_text().splitlines() if marker.exists() else []
            if events.count("started") == started and events.count("stopped") == stopped:
                return
            time.sleep(0.01)
        pytest.fail(f"background lifecycle did not settle: {events}")
    try:
        manager.discover_and_load()
        await_events(1, 0)
        pid = host.pid
        if operation == "reload":
            manager.discover_and_load(force=True)
            await_events(2, 1)
        else:
            manager.unload("taskprobe")
            await_events(1, 1)
        assert host.alive and host.pid == pid
        manager.unload()
        await_events(2, 2) if operation == "reload" else await_events(1, 1)
    finally:
        manager.unload()
        host.shutdown()


@pytest.mark.platforms("any")
def test_hosted_dashboard_reload_refreshes_api_without_resetting_sibling(tmp_path, monkeypatch):
    home = _home_with_plugins(tmp_path, monkeypatch, {
        name: "def register(ctx): pass\n" for name in ("reloadprobe", "siblingprobe")
    })
    api_source = '''
import itertools, os, sys
from fastapi import APIRouter

router = APIRouter()
requests = itertools.count(1)

@router.get("/")
def status():
    return {"value": "before", "pid": os.getpid(), "requests": next(requests),
            "reload_loaded": "hermes_dashboard_plugin_dashboard-reloadprobe" in sys.modules}
'''
    for name in ("reloadprobe", "siblingprobe"):
        dashboard_dir = home / "plugins" / name / "dashboard"
        dashboard_dir.mkdir()
        (dashboard_dir / "api.py").write_text(api_source, encoding="utf-8")
    manager = PluginManager()
    host = manager._plugin_host()

    def request(name):
        # Dashboard manifests may use a name different from the path-derived plugin key.
        dashboard_name = "dashboard-reloadprobe" if name == "reloadprobe" else name
        result = host.asgi_request(
            dashboard_name, str(home / "plugins" / name / "dashboard"), "api.py", "GET", "/", "", [], b"")
        assert result["status"] == 200
        return json.loads(result["body"])

    try:
        manager.discover_and_load()
        assert manager._plugins["reloadprobe"].error is None
        assert manager._plugins["siblingprobe"].error is None
        manifest = manager._plugins["reloadprobe"].manifest
        first = request("reloadprobe")
        assert first["pid"] != os.getpid()
        assert first == {"value": "before", "pid": first["pid"], "requests": 1, "reload_loaded": True}
        sibling = request("siblingprobe")
        assert sibling == first

        api_file = home / "plugins" / "reloadprobe" / "dashboard" / "api.py"
        original_stat = api_file.stat()
        api_file.write_text(api_source.replace("before", "after!"), encoding="utf-8")
        # An update with unchanged size/mtime must not resurrect the old .pyc.
        os.utime(api_file, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
        assert manager.unload("reloadprobe")
        assert request("siblingprobe") == {**sibling, "requests": 2, "reload_loaded": False}

        manager._load_plugin(manifest)
        assert manager._plugins["reloadprobe"].error is None
        assert request("reloadprobe") == {**first, "value": "after!"}
        assert request("siblingprobe") == {**sibling, "requests": 3}
        assert host.alive
    finally:
        manager.unload()
        host.shutdown()


MEMORY_PLUGIN = '''
import os
from agent.memory_provider import MemoryProvider

class Probe(MemoryProvider):
    level = 0  # a class-level default the provider changes later
    @property
    def name(self): return "memprobe"
    def is_available(self): return True
    def initialize(self, session_id, **kw): pass
    def get_tool_schemas(self): return []
    def whoami(self):
        self.level += 1
        return os.getpid()

def register(ctx):
    ctx.register_memory_provider(Probe())
'''


@pytest.mark.platforms("any")
@pytest.mark.parametrize("category_first", [False, True])
def test_general_and_category_loads_share_import_and_refresh_on_reload(tmp_path, monkeypatch, category_first):
    source = (
        "from pathlib import Path\nimport os\n"
        "with (Path(os.environ['HERMES_HOME']) / 'shared-imports').open('a') as handle:\n"
        "    handle.write(str(os.getpid()) + '\\n')\n" + MEMORY_PLUGIN)
    home = _home_with_plugins(tmp_path, monkeypatch, {"memprobe": source})
    (home / "plugins" / "memprobe" / "plugin.yaml").write_text(
        "name: memprobe\nversion: '1.0'\nkind: standalone\n", encoding="utf-8")
    manager = PluginManager()
    host = manager._plugin_host()
    plugin_dir = home / "plugins" / "memprobe"
    def category():
        return host.load_instance(
            plugin_dir, module_name="_hermes_memory_memprobe",
            base_ref="agent.memory_provider:MemoryProvider", capture="register_memory_provider")
    try:
        if category_first:
            category().whoami()
        manager.discover_and_load()
        assert manager._plugins["memprobe"].error is None
        assert category().whoami() == host.info["pid"] != os.getpid()
        marker = home / "shared-imports"
        assert marker.read_text().splitlines() == [str(host.info["pid"])]
        manifest = manager._plugins["memprobe"].manifest
        manager.unload("memprobe")
        manager._load_plugin(manifest)
        assert manager._plugins["memprobe"].error is None
        assert category().whoami() == host.info["pid"]
        assert marker.read_text().splitlines() == [str(host.info["pid"])] * 2
    finally:
        manager.unload()
        host.shutdown()


@pytest.mark.platforms("any")
def test_concurrent_category_loads_import_once_and_create_distinct_instances(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    home = _home_with_plugins(tmp_path, monkeypatch, {})
    provider_dir = home / "plugins" / "memprobe"
    provider_dir.mkdir(parents=True)
    marker = home / "category-imports"
    (provider_dir / "__init__.py").write_text(
        "import os, time\nfrom pathlib import Path\n"
        f"with Path({str(marker)!r}).open('a') as handle:\n"
        "    handle.write(str(os.getpid()) + '\\n')\n"
        "time.sleep(0.2)\n" + MEMORY_PLUGIN, encoding="utf-8")
    host = plugins_mod.get_plugin_manager()._plugin_host()
    try:
        host.ensure_started()
        barrier = Barrier(4)
        def load(index):
            barrier.wait(timeout=10)
            return host.load_instance(
                provider_dir, module_name="_hermes_memory_memprobe",
                base_ref="agent.memory_provider:MemoryProvider", capture="register_memory_provider")
        with ThreadPoolExecutor(max_workers=4) as pool:
            instances = list(pool.map(load, range(4)))
        pids = [instance.whoami() for instance in instances]
        assert len(set(pids)) == 1
        assert pids[0] != os.getpid()
        assert [instance.level for instance in instances] == [1] * 4
        instances[0].whoami()
        assert [instance.level for instance in instances] == [2, 1, 1, 1]
        assert marker.read_text().splitlines() == [str(pids[0])]
    finally:
        host.shutdown()


@pytest.mark.platforms("any")
def test_concurrent_retained_provider_refresh_reuses_one_instance(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    home = _home_with_plugins(tmp_path, monkeypatch, {})
    provider_dir = home / "plugins" / "memprobe"
    provider_dir.mkdir(parents=True)
    (provider_dir / "__init__.py").write_text(MEMORY_PLUGIN, encoding="utf-8")
    host = plugins_mod.get_plugin_manager()._plugin_host()
    reloads = []
    def before_reload():
        reloads.append(True)
        time.sleep(0.1)  # let all first callers encounter the stale slot
    try:
        provider = host.load_instance(
            provider_dir, module_name="_hermes_memory_memprobe",
            base_ref="agent.memory_provider:MemoryProvider",
            capture="register_memory_provider", before_reload=before_reload)
        previous = host._proc
        previous.kill()
        previous.wait(timeout=10)
        barrier = Barrier(4)
        def use(index):
            barrier.wait(timeout=10)
            return provider.is_available()
        with ThreadPoolExecutor(max_workers=4) as pool:
            assert list(pool.map(use, range(4))) == [True] * 4
        assert reloads == [True]
        assert [provider.whoami() for _ in range(4)] == [host.info["pid"]] * 4
        assert provider.level == 4
    finally:
        host.shutdown()


@pytest.mark.platforms("posix")  # SIGKILL
def test_hosted_memory_provider_stays_live_across_a_host_crash(tmp_path, monkeypatch):
    import signal
    from plugins.memory import load_memory_provider
    from plugins.memory.config_schema import get_provider_config_schema

    home = _home_with_plugins(tmp_path, monkeypatch, {})
    provider_dir = home / "plugins" / "memprobe"
    provider_dir.mkdir(parents=True)
    (provider_dir / "__init__.py").write_text(MEMORY_PLUGIN, encoding="utf-8")
    marker = tmp_path / "schema_pid"
    (provider_dir / "config_schema.py").write_text(
        f"import os, pathlib\npathlib.Path({str(marker)!r}).write_text(str(os.getpid()))\nCONFIG_SCHEMA = None\n",
        encoding="utf-8")
    host = plugins_mod.get_plugin_manager()._plugin_host()
    try:
        provider = load_memory_provider("memprobe")
        first_pid = provider.whoami()
        assert first_pid == host.pid != os.getpid()
        assert provider.level == 1  # read live from the plugin's object, not the class default
        # The provider's schema file is user code too: it runs in the host, never here.
        get_provider_config_schema("memprobe")
        assert int(marker.read_text(encoding="utf-8-sig")) == host.pid

        os.kill(first_pid, signal.SIGKILL)  # windows-footgun: ok — posix-only test (platforms marker)
        deadline = time.monotonic() + 10
        while host.alive and time.monotonic() < deadline:
            time.sleep(0.1)
        # The proxy Hermes already holds reloads the provider into the new host on next use.
        assert provider.whoami() not in {first_pid, os.getpid()}
    finally:
        host.shutdown()

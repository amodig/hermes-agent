import threading

import pytest

from hermes_cli import plugins_loader


def test_nested_plugin_load_runs_inline_on_deadline_worker(monkeypatch):
    monkeypatch.setattr(plugins_loader, "_resolve_plugin_load_timeout", lambda: 0.3)
    abandoned = []

    class Context:
        def __init__(self, name):
            self.name = name

        def _abandon_load(self):
            abandoned.append(self.name)

    observed_threads = []
    release = threading.Event()

    def outer_load():
        observed_threads.append(threading.current_thread())
        plugins_loader.run_with_load_deadline(
            "done", Context("done"), lambda: observed_threads.append(threading.current_thread()),
        )
        plugins_loader.run_with_load_deadline("hung", Context("hung"), release.wait)

    with pytest.raises(plugins_loader.PluginLoadTimeout):
        plugins_loader.run_with_load_deadline("outer", Context("outer"), outer_load)
    release.set()

    assert len(observed_threads) == 2
    assert observed_threads[0] is observed_threads[1]
    # The outer timeout abandons only contexts still loading, not a nested load that already finished.
    assert abandoned == ["outer", "hung"]


@pytest.mark.platforms("any")
def test_concurrent_first_category_loads_share_the_managers_host(tmp_path, monkeypatch):
    import contextvars
    import os
    from concurrent.futures import ThreadPoolExecutor

    from hermes_cli import plugin_host, plugins
    from hermes_cli.plugin_isolation import user_plugin_host

    home = tmp_path / ".hermes"
    provider_dir = home / "plugins" / "singleflight"
    provider_dir.mkdir(parents=True)
    (home / "config.yaml").write_text("plugins:\n  isolation: host\n", encoding="utf-8")
    marker = home / "category-imports"
    (provider_dir / "__init__.py").write_text(
        "import os\nfrom pathlib import Path\n"
        "from agent.memory_provider import MemoryProvider\n"
        f"with Path({str(marker)!r}).open('a', encoding='utf-8') as handle:\n"
        "    handle.write(str(os.getpid()) + '\\n')\n"
        "class Probe(MemoryProvider):\n"
        "    @property\n"
        "    def name(self): return 'singleflight'\n"
        "    def is_available(self): return True\n"
        "    def initialize(self, session_id, **kw): pass\n"
        "    def get_tool_schemas(self): return []\n"
        "    def whoami(self): return {'pid': os.getpid(), 'caller': self.caller}\n"
        "def register(ctx):\n"
        "    provider = Probe()\n"
        "    provider.caller = ctx.get_config('caller')\n"
        "    ctx.register_memory_provider(provider)\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(tmp_path / "os-home"))
    monkeypatch.setenv("HERMES_HOME", str(home))
    manager = plugins.get_plugin_manager()
    caller = contextvars.ContextVar("host_creation_caller", default="<unset>")
    ctx = plugins.PluginContext(
        plugins.PluginManifest(name="singleflight", path=str(provider_dir), source="user"), manager,
    )
    monkeypatch.setattr(ctx, "get_config", lambda key, default=None: caller.get())
    first_constructor = threading.Event()
    second_arrived = threading.Event()
    init_lock = threading.Lock()
    created = []
    original_init = plugin_host.PluginHost.__init__

    class ObservedInitLock:
        def __enter__(self):
            if first_constructor.is_set():
                second_arrived.set()
            init_lock.acquire()

        def __exit__(self, *exc):
            init_lock.release()

    def paused_init(host, owner):
        original_init(host, owner)
        created.append((host, caller.get()))
        if len(created) == 1:
            first_constructor.set()
            assert second_arrived.wait(10), "the second category load never reached host creation"
        else:
            # The unfixed accessor skips the lock and enters a second constructor instead.
            second_arrived.set()

    monkeypatch.setattr(manager, "_plugin_host_lock", ObservedInitLock(), raising=False)
    monkeypatch.setattr(plugin_host.PluginHost, "__init__", paused_init)

    def load(name):
        caller.set(name)
        host = user_plugin_host()
        instance = host.load_instance(
            provider_dir, module_name="_hermes_memory_singleflight",
            base_ref="agent.memory_provider:MemoryProvider", capture="register_memory_provider", ctx=ctx,
        )
        return host, instance.whoami()

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            # Discovery owns this lock while its deadline worker first asks for a host.
            # Host creation and category callbacks must not try to acquire it.
            with manager._discovery_lock:
                first = pool.submit(load, "first")
                assert first_constructor.wait(10), "the first host constructor never started"
                second = pool.submit(load, "second")
                results = [first.result(timeout=30), second.result(timeout=30)]
        owned = manager._plugin_host()
        assert [host for host, _ in results] == [owned, owned]
        assert created == [(owned, "first")]
        assert [result for _, result in results] == [
            {"pid": owned.info["pid"], "caller": "first"},
            {"pid": owned.info["pid"], "caller": "second"},
        ]
        assert owned.info["pid"] != os.getpid()
        assert marker.read_text(encoding="utf-8").splitlines() == [str(owned.info["pid"])]
        child = owned._proc
        owned.shutdown()
        assert child.poll() is not None
        assert not owned.alive
    finally:
        second_arrived.set()
        for host, _ in created:
            host.shutdown()

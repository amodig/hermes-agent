"""Exercise dual-kind plugins through the real general and memory loaders."""

import os
import textwrap

import pytest

from hermes_cli.plugins import get_plugin_manager
from plugins.memory import load_memory_provider


def _install(home, monkeypatch, *, label="first", enabled=True, isolation="in_process"):
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_BUNDLED_PLUGINS", str(home / "empty"))
    monkeypatch.delenv("HERMES_ENABLE_PROJECT_PLUGINS", raising=False)
    monkeypatch.chdir(home)
    (home / "config.yaml").write_text(
        f"plugins:\n  enabled: {'[dual]' if enabled else '[]'}\n  isolation: {isolation}\n"
        "memory:\n  provider: dual\n"
    )
    plugin = home / "plugins" / "dual"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text("name: dual\nversion: 1.0.0\nkind: standalone\n")
    (plugin / "values.py").write_text(f"LABEL = {label!r}\n")
    (plugin / "__init__.py").write_text(textwrap.dedent('''\
        import os
        from pathlib import Path
        from agent.memory_provider import MemoryProvider
        from .values import LABEL

        with Path(__file__).with_name("imports").open("a") as marker:
            marker.write(str(os.getpid()) + "\\n")

        class Provider(MemoryProvider):
            name = "dual"
            def is_available(self): return True
            def initialize(self, session_id, **kwargs): pass
            def get_tool_schemas(self): return []

        def make_hook(label):
            def callback(**kwargs):
                return {"context": label}
            return callback

        def register(ctx):
            ctx.register_memory_provider(Provider())
            ctx.register_hook("pre_llm_call", make_hook(LABEL))
            ctx.register_hook("pre_llm_call", make_hook("second"))
    '''))
    return get_plugin_manager()


def _contexts(manager):
    return manager.invoke_hook("pre_llm_call", session_id="")


@pytest.mark.platforms("any")
@pytest.mark.parametrize("isolation", ["in_process", "host"])
@pytest.mark.parametrize("order", ["plugin-first", "memory-first", "memory-only"])
def test_dual_kind_plugin_hooks_run_once(tmp_path, monkeypatch, order, isolation):
    manager = _install(tmp_path, monkeypatch, enabled=order != "memory-only", isolation=isolation)
    host = manager._plugin_host() if isolation == "host" else None
    imports = tmp_path / "plugins" / "dual" / "imports"
    try:
        if order == "plugin-first":
            manager.discover_and_load()
        provider = load_memory_provider("dual")
        assert provider is not None and provider.name == "dual"
        if order == "memory-first":
            manager.discover_and_load()
        expected = [{"context": "first"}, {"context": "second"}]
        assert _contexts(manager) == expected
        # New provider instances must not append another fallback hook group.
        assert load_memory_provider("dual") is not provider
        assert _contexts(manager) == expected
        if host is not None:
            assert host.info["pid"] != os.getpid()
            assert imports.read_text().splitlines() == [str(host.info["pid"])]
        manager.unload()
        assert _contexts(manager) == []
        assert not manager._memory_hook_registrations
        if order != "memory-only":
            manager.discover_and_load(force=True)
        assert load_memory_provider("dual") is not None
        assert _contexts(manager) == expected
        if host is not None:
            epochs = 1 if order == "memory-only" else 2
            assert imports.read_text().splitlines() == [str(host.info["pid"])] * epochs
    finally:
        manager.unload()
        if host is not None:
            host.shutdown()


@pytest.mark.platforms("any")
@pytest.mark.parametrize("isolation", ["in_process", "host"])
def test_same_name_different_sources_are_not_suppressed(tmp_path, monkeypatch, isolation):
    import shutil

    home = tmp_path / "home"
    manager = _install(home, monkeypatch, isolation=isolation)
    host = manager._plugin_host() if isolation == "host" else None
    project = tmp_path / "project"
    source = project / ".hermes" / "plugins" / "dual"
    shutil.copytree(home / "plugins" / "dual", source)
    (source / "values.py").write_text('LABEL = "project"\n')
    monkeypatch.chdir(project)
    monkeypatch.setenv("HERMES_ENABLE_PROJECT_PLUGINS", "1")
    try:
        assert load_memory_provider("dual") is not None
        manager.discover_and_load()
        assert _contexts(manager) == [
            {"context": "first"}, {"context": "second"},
            {"context": "project"}, {"context": "second"},
        ]
        manifest = manager._plugins["dual"].manifest
        assert manager.unload("dual")
        assert _contexts(manager) == []
        (source / "values.py").write_text('LABEL = "project reloaded"\n')
        assert load_memory_provider("dual") is not None
        manager._load_plugin(manifest)
        assert _contexts(manager) == [
            {"context": "first"}, {"context": "second"},
            {"context": "project reloaded"}, {"context": "second"},
        ]
        if host is not None:
            pid = str(host.info["pid"])
            assert int(pid) != os.getpid()
            assert (home / "plugins" / "dual" / "imports").read_text().splitlines() == [pid]
            assert (source / "imports").read_text().splitlines() == [pid, pid]
    finally:
        manager.unload()
        if host is not None:
            host.shutdown()


@pytest.mark.platforms("any")
@pytest.mark.parametrize("isolation", ["in_process", "host"])
@pytest.mark.parametrize("order", ["plugin-first", "memory-first"])
def test_reexported_register_uses_plugin_source(tmp_path, monkeypatch, order, isolation):
    manager = _install(tmp_path, monkeypatch, isolation=isolation)
    host = manager._plugin_host() if isolation == "host" else None
    plugin = tmp_path / "plugins" / "dual"
    original = plugin / "__init__.py"
    (plugin / "implementation.py").write_text(original.read_text())
    original.write_text("from .implementation import register, Provider  # MemoryProvider\n")
    try:
        if order == "plugin-first":
            manager.discover_and_load()
        assert load_memory_provider("dual") is not None
        if order == "memory-first":
            manager.discover_and_load()
        assert _contexts(manager) == [{"context": "first"}, {"context": "second"}]
    finally:
        manager.unload()
        if host is not None:
            host.shutdown()

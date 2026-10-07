"""``context.engine: <name>`` names the active engine, so an engine dropped into
``$HERMES_HOME/plugins/<name>/`` loads without a ``plugins.enabled`` entry and never trips the
"Context engine 'X' not found — falling back to built-in compressor" warning (#61839)."""

import inspect
import json
import logging
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from textwrap import dedent

import pytest

_ENGINE_SRC = dedent('''
    from agent.context_engine import ContextEngine

    class Demo(ContextEngine):
        @property
        def name(self):
            return "ctx_demo"
        def update_from_response(self, usage):
            pass
        def should_compress(self, prompt_tokens=None):
            return False
        def compress(self, messages, current_tokens=None):
            return messages

    def register(ctx):
        ctx.register_context_engine(Demo())
''')


def test_user_installed_engine_is_selected_by_name(tmp_path: Path, monkeypatch, caplog):
    engine_dir = tmp_path / "plugins" / "ctx_demo"
    engine_dir.mkdir(parents=True)
    (engine_dir / "__init__.py").write_text(_ENGINE_SRC)
    (tmp_path / "plugins" / "notes").mkdir()  # unrelated user plugin: never treated as an engine
    (tmp_path / "plugins" / "notes" / "__init__.py").write_text("def register(ctx): pass\n")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from agent.agent_init import _select_context_engine
    from plugins.context_engine import discover_context_engines, load_context_engine

    assert load_context_engine("notes") is None
    assert "ctx_demo" in {name for name, _, _ in discover_context_engines()}
    with caplog.at_level(logging.WARNING, logger="run_agent"):
        engine = _select_context_engine({"context": {"engine": "ctx_demo"}})
    assert engine is not None and engine.name == "ctx_demo"
    assert "not found" not in caplog.text


_COMMAND_ENGINE_SRC = dedent('''
    import json
    import os
    from agent.context_engine import ContextEngine
    from hermes_constants import get_hermes_home

    class Commands(ContextEngine):
        @property
        def name(self):
            return "ctx_commands"
        def update_from_response(self, usage):
            pass
        def should_compress(self, prompt_tokens=None):
            return False
        def compress(self, messages, current_tokens=None):
            return messages

    def echo(raw_args):
        return json.dumps({"args": raw_args, "pid": os.getpid(), "home": str(get_hermes_home())})

    async def echo_async(raw_args):
        return echo(raw_args)

    def register(ctx):
        ctx.register_context_engine(Commands())
        ctx.register_command(" /Engine Echo ", echo, "Engine echo", " <text> ")
        ctx.register_command("engine-async", echo_async, args_hint="<text>")
        ctx.register_command("engine-claimed", echo)
        ctx.register_command("help", echo)
        if not (get_hermes_home() / "retire-command").exists():
            ctx.register_command("engine-retired", echo)
''')


def _command_engine_home(home, isolation):
    engine_dir = home / "plugins" / "ctx_commands"
    engine_dir.mkdir(parents=True)
    (engine_dir / "__init__.py").write_text(_COMMAND_ENGINE_SRC, encoding="utf-8")
    (home / "config.yaml").write_text(
        f"plugins:\n  enabled: []\n  isolation: {isolation}\n", encoding="utf-8")
    return home


@pytest.mark.platforms("any")
@pytest.mark.parametrize("isolation", ["in_process", "host"])
def test_engine_commands_preserve_dispatch_schema_and_ownership(tmp_path, monkeypatch, isolation):
    from hermes_cli import plugins as plugins_mod
    from hermes_cli.plugins import PluginContext, PluginManifest, resolve_plugin_command_result
    from plugins.context_engine import load_context_engine

    home = _command_engine_home(tmp_path / "profile", isolation)
    empty_bundled = tmp_path / "bundled"
    empty_bundled.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "os-home"))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(plugins_mod, "get_bundled_plugins_dir", lambda: empty_bundled)
    manager = plugins_mod.get_plugin_manager()
    try:
        manager.discover_and_load()
        claimant = PluginContext(PluginManifest(name="claimant"), manager)
        claimant.register_command("engine-claimed", lambda raw_args: "already owned")
        claimed = manager._plugin_commands["engine-claimed"]
        engine = load_context_engine("ctx_commands")
        assert engine is not None
        expected_pid = manager._plugin_host().info["pid"] if isolation == "host" else os.getpid()
        if isolation == "host":
            assert expected_pid != os.getpid()
            assert "_hermes_user_context_engine.ctx_commands" not in sys.modules

        for name, is_async, description in (
            ("engine-echo", False, "Engine echo"),
            ("engine-async", True, "Context engine command"),
        ):
            handler = plugins_mod.get_plugin_command_handler(name)
            assert handler is not None
            assert inspect.iscoroutinefunction(handler) is is_async
            assert manager._plugin_commands[name] == {
                "handler": handler, "description": description,
                "plugin": "context-engine:ctx_commands", "args_hint": "<text>",
            }
            assert json.loads(resolve_plugin_command_result(handler("hello world"))) == {
                "args": "hello world", "pid": expected_pid, "home": manager.scope_key,
            }
        assert manager._plugin_commands["engine-claimed"] is claimed
        assert "help" not in manager._plugin_commands

        original = plugins_mod.get_plugin_command_handler("engine-echo")
        assert load_context_engine("ctx_commands") is not None
        assert plugins_mod.get_plugin_command_handler("engine-echo") is original
        manager.discover_and_load(force=True)
        assert plugins_mod.get_plugin_command_handler("engine-echo") is None
        assert load_context_engine("ctx_commands") is not None
        reloaded = plugins_mod.get_plugin_command_handler("engine-echo")
        assert json.loads(reloaded("reloaded")) == {
            "args": "reloaded", "pid": expected_pid, "home": manager.scope_key,
        }
    finally:
        manager.unload()
        if isolation == "host":
            manager._plugin_host().shutdown()
        else:
            sys.modules.pop("_hermes_user_context_engine.ctx_commands", None)


@contextmanager
def _engine_profile_scope(home):
    from agent.secret_scope import build_profile_secret_scope, reset_secret_scope, set_secret_scope
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    home_token = set_hermes_home_override(home)
    secret_token = set_secret_scope(build_profile_secret_scope(home), profile_home=str(home))
    try:
        yield
    finally:
        reset_secret_scope(secret_token)
        reset_hermes_home_override(home_token)


@pytest.mark.platforms("any")
def test_hosted_engine_reload_refreshes_only_its_profiles_owned_commands(tmp_path, monkeypatch):
    from agent import secret_scope
    from hermes_cli import plugins as plugins_mod
    from hermes_cli.plugins import PluginContext, PluginManifest, resolve_plugin_command_result
    from plugins.context_engine import load_context_engine

    home_a, home_b = (_command_engine_home(tmp_path / name, "host") for name in ("a", "b"))
    empty_bundled = tmp_path / "bundled"
    empty_bundled.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "os-home"))
    monkeypatch.setenv("HERMES_HOME", str(home_a))
    monkeypatch.setattr(plugins_mod, "get_bundled_plugins_dir", lambda: empty_bundled)
    was_multiplex = secret_scope.is_multiplex_active()
    secret_scope.set_multiplex_active(True)
    managers, engines, hosts, handlers, pids = {}, {}, {}, {}, {}
    try:
        for home in (home_a, home_b):
            with _engine_profile_scope(home):
                manager = managers[home] = plugins_mod.get_plugin_manager()
                manager.discover_and_load()
                assert load_context_engine("ctx_commands") is not None
                # Keep a different instance than the one that first registered the commands.
                engines[home] = load_context_engine("ctx_commands")
                assert engines[home] is not None
                hosts[home] = manager._plugin_host()
                pids[home] = hosts[home].info["pid"]
                handlers[home] = plugins_mod.get_plugin_command_handler("engine-echo")
                assert handlers[home] is not None
                assert pids[home] != os.getpid()
                assert json.loads(handlers[home]("initial")) == {
                    "args": "initial", "pid": pids[home], "home": manager.scope_key,
                }
        assert pids[home_a] != pids[home_b]
        assert "_hermes_user_context_engine.ctx_commands" not in sys.modules

        with _engine_profile_scope(home_b):
            claimant = PluginContext(PluginManifest(name="claimant"), managers[home_a])
            claimant.register_command("engine-claimed", lambda raw_args: "new owner")
            claimed = managers[home_a]._plugin_commands["engine-claimed"]
            (home_a / "retire-command").touch()
            hosts[home_a].shutdown()
            # Retained engine reload must target A even when the ambient caller belongs to B.
            assert engines[home_a].should_compress() is False
            assert plugins_mod.get_plugin_command_handler("engine-echo") is handlers[home_b]
            assert managers[home_a]._plugin_commands["engine-claimed"] is claimed

        pids[home_a] = hosts[home_a].info["pid"]
        assert pids[home_a] not in {os.getpid(), pids[home_b]}
        for home in (home_a, home_b, home_a):
            with _engine_profile_scope(home):
                handler = plugins_mod.get_plugin_command_handler("engine-echo")
                assert handler is not None
                assert (handler is handlers[home]) is (home == home_b)
                assert json.loads(handler("after restart")) == {
                    "args": "after restart", "pid": pids[home], "home": managers[home].scope_key,
                }
                async_handler = plugins_mod.get_plugin_command_handler("engine-async")
                assert async_handler is not None
                assert json.loads(resolve_plugin_command_result(async_handler("async"))) == {
                    "args": "async", "pid": pids[home], "home": managers[home].scope_key,
                }
                assert ("engine-retired" in managers[home]._plugin_commands) is (home == home_b)
    finally:
        for manager in managers.values():
            manager.unload()
            manager._plugin_host().shutdown()
        secret_scope.set_multiplex_active(was_multiplex)

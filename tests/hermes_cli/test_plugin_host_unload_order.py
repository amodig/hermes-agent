"""Hosted callbacks share the ordinary reverse-acquisition registration ledger."""

import contextvars
import os

import pytest

from hermes_cli import plugins as plugins_mod
from hermes_cli.plugins_state import PluginState
from tests.hermes_cli.test_plugin_host import _home_with_plugins
from tools.registry import registry


_PLUGIN = '''
import asyncio, os


def register(ctx):
    def observe(phase):
        ctx.dispatch_tool("unload_order_observer", {
            "phase": phase,
            "early": ctx.dispatch_tool("unload_order_early", {}) == "early",
            "late": ctx.dispatch_tool("unload_order_late", {}) == "late",
            "pid": os.getpid(),
        })

    manual = ctx.on_unload(lambda: observe("manual"))
    assert manual.active
    manual.dispose()
    manual.dispose()
    assert not manual.active
    ctx.on_unload(lambda: observe("first"))
    def early(args, **kwargs):
        # First facade lookup is nested through an earlier tool during cleanup.
        ctx.state.set("early_calls", ctx.state.get("early_calls", 0) + 1)
        return "early"

    ctx.register_tool(name="unload_order_early", toolset="unloadorder",
        schema={"name": "unload_order_early", "description": "Earlier acquisition",
                "parameters": {"type": "object", "properties": {}}},
        handler=early)

    async def middle():
        await asyncio.sleep(0)
        # Async cleanup also keeps its deferred facade access after the child ctx is inactive.
        ctx.state.set("middle_ran", True)
        observe("middle")

    ctx.on_unload(middle if ASYNC_CLEANUP else lambda: asyncio.run(middle()))
    ctx.register_tool(name="unload_order_late", toolset="unloadorder",
        schema={"name": "unload_order_late", "description": "Later acquisition",
                "parameters": {"type": "object", "properties": {}}},
        handler=lambda args, **kw: "late")
    ctx.on_unload(lambda: observe("last"))
    if FAIL_LOAD:
        raise ValueError("rollback interleaved registrations")
'''


@pytest.mark.platforms("any")
@pytest.mark.parametrize("isolation", ["in_process", "host"])
@pytest.mark.parametrize("fail_load", [False, True], ids=["unload", "rollback"])
def test_callbacks_observe_reverse_acquisition_order(tmp_path, monkeypatch, isolation, fail_load):
    source = f"ASYNC_CLEANUP = {isolation == 'host'!r}\nFAIL_LOAD = {fail_load!r}\n" + _PLUGIN
    _home_with_plugins(tmp_path, monkeypatch, {"unloadorder": source}, isolation=isolation)
    manager = plugins_mod.get_plugin_manager()
    observations = []
    unrelated_calls = []

    def observe(args, **kwargs):
        observations.append(args)
        if isolation == "host" and args["phase"] == "last":
            unrelated_calls.append(contextvars.Context().run(
                registry.dispatch, "unload_order_early", {}, scope=manager.scope_key))
        return "observed"

    registry.register(
        name="unload_order_observer", toolset="unloadobserver", scope=manager.scope_key,
        schema={"name": "unload_order_observer", "description": "Observe real callback visibility",
                "parameters": {"type": "object", "properties": {}}}, handler=observe)
    try:
        manager.discover_and_load()
        loaded = manager._plugins["unloadorder"]
        if fail_load:
            assert not loaded.enabled and "rollback interleaved registrations" in str(loaded.error)
        else:
            assert loaded.enabled and loaded.error is None
            assert [row["phase"] for row in observations] == ["manual"]
        manager.unload("unloadorder")
        manager.unload("unloadorder")
        assert [(row["phase"], row["early"], row["late"]) for row in observations] == [
            ("manual", False, False),
            ("last", True, True),
            ("middle", True, False),
            ("first", False, False),
        ]
        assert all((row["pid"] == os.getpid()) == (isolation == "in_process") for row in observations)
        assert PluginState("unloadorder").get("middle_ran") is True
        assert PluginState("unloadorder").get("early_calls") == 2
        assert "unloadorder" not in manager._ownership_ledger
        if isolation == "host":
            assert len(unrelated_calls) == 1 and "unloaded" in unrelated_calls[0]
            assert manager._plugin_host().alive
            assert "unloadorder" not in manager._plugin_host()._contexts
    finally:
        manager.unload()
        registry.deregister("unload_order_observer", scope=manager.scope_key)
        if isolation == "host":
            manager._plugin_host().shutdown()

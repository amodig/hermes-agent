"""Hosted background callbacks keep their owner across later multiplex activation."""

import contextvars
import json
import os
from pathlib import Path
from queue import Queue

import pytest


_PLUGIN = '''
import asyncio, os
from pathlib import Path


def register(ctx):
    ctx.register_tool(
        name="scope_roundtrip", toolset="scopeprobe",
        schema={"name": "scope_roundtrip", "description": "Probe caller scope",
                "parameters": {"type": "object", "properties": {}}},
        handler=lambda args, **kw: ctx.dispatch_tool("scope_probe", {"pid": os.getpid()}))

    async def background():
        for index in range(2):
            release = Path(os.environ["HERMES_HOME"]) / f"scope-release-{index}"
            while not release.exists():
                await asyncio.sleep(0.01)
            ctx.dispatch_tool("scope_probe", {"pid": os.getpid(), "background": True})

    ctx.spawn_task(background(), name="scopeprobe-background")
'''


@pytest.mark.platforms("any")  # real hosted process and cross-process callback context
def test_deferred_callbacks_rebind_owner_secrets_after_activation(tmp_path, monkeypatch):
    from agent.secret_scope import (
        UnscopedSecretError, build_profile_secret_scope, current_secret_scope_home,
        get_secret, is_multiplex_active, set_multiplex_active, set_secret_scope,
    )
    from gateway.session_context import get_session_env, scoped_current_session_id
    from hermes_cli import plugins as plugins_mod
    from hermes_constants import get_hermes_home, hermes_home_key, set_hermes_home_override
    from tools.approval_context import get_current_session_key, set_current_session_key
    from tools.registry import registry
    from tools.terminal_scope import install_profile_terminal_scope, terminal_env
    from tui_gateway.launch_profile_policy import activate_multi_profile_hosting

    home = tmp_path / ".hermes" / "profiles" / "owner"
    plugin_dir = home / "plugins" / "scopeprobe"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text("name: scopeprobe\nversion: '1.0'\n", encoding="utf-8")
    (plugin_dir / "__init__.py").write_text(_PLUGIN, encoding="utf-8")
    (home / "config.yaml").write_text(
        "plugins:\n  isolation: host\n  enabled: [scopeprobe]\n"
        "terminal:\n  backend: docker\n  docker_image: owner-image\n", encoding="utf-8")
    (home / ".env").write_text("SCOPE_PROBE_KEY=owner\nSCOPE_REVOKED_KEY=revoked\n", encoding="utf-8")
    secondary = tmp_path / ".hermes" / "profiles" / "secondary"
    secondary.mkdir(parents=True)
    (secondary / ".env").write_text("SCOPE_PROBE_KEY=secondary\n", encoding="utf-8")
    (secondary / "config.yaml").write_text("terminal:\n  backend: local\n", encoding="utf-8")
    bundled = tmp_path / "bundled"
    bundled.mkdir()
    monkeypatch.setattr(plugins_mod, "get_bundled_plugins_dir", lambda: bundled)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setenv("SCOPE_PROBE_KEY", "launch-env")
    monkeypatch.setenv("SCOPE_ENV_ONLY_KEY", "must-not-leak")
    monkeypatch.setenv("TERMINAL_ENV", "ssh")
    monkeypatch.setenv("TERMINAL_SSH_HOST", "foreign-host")
    monkeypatch.setenv("TERMINAL_DOCKER_IMAGE", "foreign-image")
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    monkeypatch.delenv("HERMES_SESSION_KEY", raising=False)
    set_multiplex_active(False)
    def make_manager():
        set_hermes_home_override(home)
        return plugins_mod.PluginManager()
    manager = contextvars.Context().run(make_manager)
    observations = Queue()

    def probe(args, **kwargs):
        try:
            result = {
                "pid": args["pid"], "home": str(get_hermes_home()),
                "scope_home": current_secret_scope_home(),
                "secret": get_secret("SCOPE_PROBE_KEY"),
                "revoked": get_secret("SCOPE_REVOKED_KEY"),
                "env_only": get_secret("SCOPE_ENV_ONLY_KEY"),
                "session": get_session_env("HERMES_SESSION_ID", ""),
                "approval": get_current_session_key(default=""),
                "terminal": terminal_env("TERMINAL_ENV"),
                "image": terminal_env("TERMINAL_DOCKER_IMAGE"),
                "ssh_host": terminal_env("TERMINAL_SSH_HOST"),
            }
        except Exception as exc:
            result = {"error": f"{type(exc).__name__}: {exc}"}
        if args.get("background"):
            observations.put(result)
        return json.dumps(result)

    registry.register(
        name="scope_probe", toolset="scopeprobe", scope=manager.scope_key,
        schema={"name": "scope_probe", "description": "Read callback scope",
                "parameters": {"type": "object", "properties": {}}}, handler=probe)

    def exercise():
        set_current_session_key("creator-approval")
        set_hermes_home_override(home)
        with scoped_current_session_id("creator-session"):
            host = manager._plugin_host()
            try:
                manager.discover_and_load()
                assert manager._plugins["scopeprobe"].error is None
                assert host.info["pid"] != os.getpid() and host.alive
                assert not is_multiplex_active()
                # Loading has returned: the background task's original request is gone.
                activate_multi_profile_hosting()
                with pytest.raises(UnscopedSecretError):
                    contextvars.Context().run(get_secret, "SCOPE_PROBE_KEY")
                expected = {
                    "pid": host.info["pid"], "home": manager.scope_key, "scope_home": manager.scope_key,
                    "secret": "owner", "revoked": "revoked", "env_only": None,
                    "session": "", "approval": "",
                    "terminal": "docker", "image": "owner-image", "ssh_host": "",
                }
                (home / "scope-release-0").touch()
                assert observations.get(timeout=10) == expected

                def serve_secondary():
                    secondary_home = hermes_home_key(secondary)
                    set_hermes_home_override(secondary_home)
                    set_secret_scope(build_profile_secret_scope(secondary), profile_home=secondary_home)
                    install_profile_terminal_scope(secondary)
                    set_current_session_key("secondary-approval")
                    with scoped_current_session_id("secondary-session"):
                        # An in-flight call still uses its caller, not the host's fallback.
                        result = registry.dispatch("scope_roundtrip", {}, scope=manager.scope_key)
                        assert json.loads(result) == {
                            **expected, "home": secondary_home, "scope_home": secondary_home,
                            "secret": "secondary", "revoked": None,
                            "terminal": "local", "image": terminal_env("TERMINAL_DOCKER_IMAGE"),
                            "session": "secondary-session", "approval": "secondary-approval",
                        }
                        (home / ".env").write_text("SCOPE_PROBE_KEY=owner-rotated\n", encoding="utf-8")
                        (home / "scope-release-1").touch()
                        assert observations.get(timeout=10) == {
                            **expected, "secret": "owner-rotated", "revoked": None,
                        }

                contextvars.Context().run(serve_secondary)
            finally:
                try:
                    manager.unload()
                finally:
                    host.shutdown()

    try:
        contextvars.Context().run(exercise)
    finally:
        registry.deregister("scope_probe", scope=manager.scope_key)
        set_multiplex_active(False)

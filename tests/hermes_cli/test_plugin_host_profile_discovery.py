"""Discovery and RPC capture share one hosted provider instance per source generation."""

import json
import os
import py_compile
from pathlib import Path

import pytest

from hermes_cli import plugins as plugins_mod
from hermes_cli.plugin_host_profiles import load_hosted_profiles
from tests.hermes_cli.test_plugin_host import _home_with_plugins


_PROVIDER = '''import os
from pathlib import Path
from providers import register_provider
from providers.base import ProviderProfile

VERSION = "GENERATION"
with (Path(os.environ["HERMES_HOME"]) / "profile-imports.txt").open("a", encoding="utf-8") as marker:
    marker.write(f"{os.getpid()}:{VERSION}\\n")

class Discovered(ProviderProfile):
    def ping(self):
        return {"version": VERSION, "pid": os.getpid(), "instance": id(self)}

    def late(self):
        from .late import MARK
        return MARK

register_provider(Discovered(name="discovered", base_url="https://discovered.example/v1"))
'''

_OBSERVER = '''import json
import providers

# This also exercises discovery nested inside the host's general-plugin import lock.
providers.list_providers()

def register(ctx):
    def inspect(args, **kwargs):
        if args.get("rescan"):
            layer, home, key = providers._bound_home_layer()
            providers._scan_home_layer(layer, key)
        profile = providers.get_provider_profile("discovered")
        return json.dumps(profile.ping())
    ctx.register_tool(name="inspect_discovered", toolset="registry_probe",
        schema={"name": "inspect_discovered", "description": "Inspect discovered provider",
                "parameters": {"type": "object", "properties": {}}}, handler=inspect)
'''




@pytest.mark.platforms("any")
def test_discovery_then_rpc_reuses_instance_and_refreshes_source_generation(tmp_path, monkeypatch):
    from tools.registry import registry

    home = _home_with_plugins(tmp_path, monkeypatch, {"registry-probe": _OBSERVER})
    plugin_dir = home / "plugins" / "model-providers" / "discovered"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text("name: discovered\nkind: model-provider\n", encoding="utf-8")
    source = plugin_dir / "__init__.py"
    source.write_text(_PROVIDER.replace("GENERATION", "before"), encoding="utf-8")
    late = plugin_dir / "late.py"
    late.write_bytes(b"MARK = 'alpha'\n")
    stamp = 1_700_000_000_100_000_000
    os.utime(source, ns=(stamp, stamp))
    os.utime(late, ns=(stamp, stamp))
    bytecode = Path(py_compile.compile(str(late), doraise=True,
                                     invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP))
    original_bytecode = bytecode.read_bytes()

    manager = plugins_mod.get_plugin_manager()
    host = manager._plugin_host()
    try:
        manager.discover_and_load()
        assert manager._plugins["registry-probe"].error is None

        def inspect(**args):
            return json.loads(registry.dispatch("inspect_discovered", args, scope=manager.scope_key))

        discovered = inspect()
        assert discovered["version"] == "before" and discovered["pid"] != os.getpid()
        # Deliberately use a different import name from providers' per-home discovery name.
        proxy = load_hosted_profiles(plugin_dir, "_hermes_discovered_rpc")[0]
        assert proxy.ping() == discovered

        def host_imports():
            return [line.split(":", 1)[1] for line in
                    (home / "profile-imports.txt").read_text(encoding="utf-8").splitlines()
                    if line.startswith(f"{discovered['pid']}:")]

        assert host_imports() == ["before"]
        source.write_text(_PROVIDER.replace("GENERATION", "after!"), encoding="utf-8")
        late.write_bytes(b"MARK = 'omega'\n")
        os.utime(source, ns=(stamp, stamp))
        os.utime(late, ns=(stamp, stamp))
        fresh = load_hosted_profiles(plugin_dir, "_hermes_discovered_rpc")[0]
        refreshed = fresh.ping()
        assert refreshed["version"] == "after!" and refreshed["pid"] == discovered["pid"]
        assert refreshed["instance"] != discovered["instance"]
        # A deferred relative import must ignore the valid-looking old pyc, without deleting it.
        assert fresh.late() == "omega"
        assert bytecode.read_bytes() == original_bytecode
        # Capture is private; only an explicit discovery rescan publishes into the live layer.
        assert inspect() == discovered
        assert inspect(rescan=True) == refreshed
        assert fresh.ping() == refreshed
        assert host_imports() == ["before", "after!"]
    finally:
        manager.unload()
        host.shutdown()


@pytest.mark.platforms("any")
def test_extraction_keeps_nested_lookups_bundled_only_and_registration_private(tmp_path, monkeypatch):
    home = _home_with_plugins(tmp_path, monkeypatch, {})
    root = home / "plugins" / "model-providers"
    unrelated = root / "unrelated"
    unrelated.mkdir(parents=True)
    (unrelated / "__init__.py").write_text(
        'import os\nfrom pathlib import Path\n'
        '(Path(os.environ["HERMES_HOME"]) / "unrelated-imported").touch()\n', encoding="utf-8")
    target = root / "capture-only"
    target.mkdir()
    (target / "__init__.py").write_text(
        'from providers import get_provider_profile, list_providers, register_provider\n'
        'from providers.base import ProviderProfile\n'
        'assert list_providers()\n'
        'register_provider(ProviderProfile(name="capture-only"))\n'
        'assert get_provider_profile("capture-only") is None\n', encoding="utf-8")

    profiles = load_hosted_profiles(target, "_hermes_capture_only")
    assert [profile.name for profile in profiles] == ["capture-only"]
    assert not (home / "unrelated-imported").exists()

"""A live plugin host serves a profile proxy the source it was extracted from.

A rewritten model-provider plugin used to stay on the host's first capture: a freshly extracted
proxy carried the new fields while its methods still ran the old code, or raised AttributeError for
a method it had just gained. The rewrite below keeps every path, byte length and nanosecond mtime
unchanged while replacing both the package body and a relative helper. Freshness must come from
source bytes, not filesystem metadata, timestamp-validated bytecode or stale ``sys.modules``.
"""

import os
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from hermes_cli import plugins as plugins_mod
from hermes_cli.plugin_host_profiles import load_hosted_profiles

_ADDED_METHOD = "    def added(self):\n        return {'code': 'new'}\n"
_FILLER = "# " + "x" * (len(_ADDED_METHOD) - 3) + "\n"
_HELPER_V1 = 'MARK = "alpha"\n'
_HELPER_V2 = 'MARK = "omega"\n'

_PLUGIN = """import os, time
from pathlib import Path
from providers import register_provider
from providers.base import ProviderProfile
from .helper import MARK

_KEY = "%(key)s"
with (Path(os.environ["HERMES_HOME"]) / "freshness-imports.txt").open("a", encoding="utf-8") as marker:
    marker.write(f"{os.getpid()}:{_KEY}\\n")
time.sleep(0.2)  # keep concurrent first-use requests inside the recapture window

class Freshness(ProviderProfile):
    def ping(self):
        return {"code": "%(version)s"}

    def helper_mark(self):
        return MARK

%(extra)sregister_provider(Freshness(name="freshness", base_url="https://freshness.example/v1",
                                   env_vars=("%(key)s",)))
"""


def _source(version, key, extra):
    return _PLUGIN % {"version": version, "key": key, "extra": extra}


@pytest.mark.platforms("any")  # the host is a child process: it writes the marker itself
def test_live_host_recaptures_a_rewritten_profile_source(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir(parents=True)
    (home / "config.yaml").write_text("plugins:\n  isolation: host\n", encoding="utf-8")
    plugin_dir = home / "plugins" / "model-providers" / "freshness"
    plugin_dir.mkdir(parents=True)
    manifest = plugin_dir / "plugin.yaml"
    manifest.write_text("name: freshness\n", encoding="utf-8")
    helper = plugin_dir / "helper.py"
    helper.write_text(_HELPER_V1, encoding="utf-8")
    source = plugin_dir / "__init__.py"
    monkeypatch.setenv("HOME", str(tmp_path / "os-home"))
    monkeypatch.setenv("HERMES_HOME", str(home))

    v1 = _source("v1", "OLD_KEY", _FILLER)
    v2 = _source("v2", "NEW_KEY", _ADDED_METHOD)
    assert len(v1) == len(v2)  # the pyc's size check must not be what keeps this honest
    second = 1_700_000_000  # fixed whole second: the v1 pyc's timestamp check must accept the v2 source
    source.write_bytes(v1.encode("utf-8"))  # bytes, so no platform newline translation skews sizes
    os.utime(source, ns=(second * 10**9 + 100_000_000,) * 2)
    os.utime(helper, ns=(second * 10**9 + 100_000_000,) * 2)

    module_name = "_hermes_freshness_provider"
    host = plugins_mod.get_plugin_manager()._plugin_host()
    try:
        old = {p.name: p for p in load_hosted_profiles(plugin_dir, module_name)}["freshness"]
        assert tuple(old.env_vars) == ("OLD_KEY",)
        assert old.ping() == {"code": "v1"}  # first use starts the host, which captures v1 here
        assert old.helper_mark() == "alpha"

        source.write_bytes(v2.encode("utf-8"))
        os.utime(source, ns=(second * 10**9 + 100_000_000,) * 2)
        helper.write_text(_HELPER_V2, encoding="utf-8")
        os.utime(helper, ns=(second * 10**9 + 100_000_000,) * 2)

        fresh = {p.name: p for p in load_hosted_profiles(plugin_dir, module_name)}["freshness"]
        assert tuple(fresh.env_vars) == ("NEW_KEY",)  # a new extraction, not the cached v1 payload
        barrier = threading.Barrier(4)

        def call(_index):
            barrier.wait(timeout=10)
            return fresh.ping(), fresh.added(), fresh.helper_mark()

        with ThreadPoolExecutor(max_workers=4) as pool:
            assert list(pool.map(call, range(4))) == [
                ({"code": "v2"}, {"code": "new"}, "omega")] * 4
    finally:
        host.shutdown()

    # Exactly one import per source per process: the extraction child and one host capture. A
    # recapture on every concurrent first call would leave extra lines here.
    lines = (home / "freshness-imports.txt").read_text(encoding="utf-8").splitlines()
    assert sorted(line.split(":", 1)[1] for line in lines) == ["NEW_KEY", "NEW_KEY", "OLD_KEY", "OLD_KEY"]

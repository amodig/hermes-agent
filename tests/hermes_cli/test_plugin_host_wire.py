"""Plugin-host value tags cannot consume ordinary dictionaries, including across real RPC."""

import json
import os
from dataclasses import dataclass

import pytest

from hermes_cli.plugin_host_wire import Opaque, Record, decode, encode
from providers.base import OMIT_TEMPERATURE


_TAG_SHAPES = (
    {"__bytes__": "aGVsbG8="},
    {"__record__": "Payload", "fields": {"value": 1}},
    {"__opaque__": "Client", "repr": "<client>"},
    {"__callable__": 1},
    {"__object__": 1},
    {"__handle__": 1},
    {"__sentinel__": "OMIT_TEMPERATURE"},
    {"__dict__": {"__bytes__": "aGVsbG8="}},
)


def _dictionary_payload():
    return {
        "exact": list(_TAG_SHAPES),
        "nested": [{**shape, "status": 200, "children": [shape, {"value": shape}]}
                   for shape in _TAG_SHAPES],
        "invalid_tags": [{next(iter(shape)): None} for shape in _TAG_SHAPES],
    }


def test_dictionary_escapes_preserve_data_without_changing_live_value_tags():
    def reject_reference(ref):
        pytest.fail(f"ordinary data reached reference resolution: {ref!r}")

    payload = _dictionary_payload()
    for value in (*_TAG_SHAPES, *payload["invalid_tags"], payload):
        encoded = json.loads(json.dumps(encode(value)))
        assert decode(encoded) == value
        assert decode(encoded, reject_reference) == value
    ordinary = {"empty": {}, "values": [None, True, 1, "text", {"key": "value"}]}
    assert encode(ordinary) == ordinary

    @dataclass
    class Payload:
        __bytes__: str
        data: dict

    value = {
        "__record__": "ordinary data",
        "bytes": b"\x00\xff",
        "record": Payload("not a bytes tag", payload),
        "opaque": Opaque("Client", "<client>"),
        "temperature": OMIT_TEMPERATURE,
    }
    restored = decode(json.loads(json.dumps(encode(value))))
    assert restored["__record__"] == value["__record__"]
    assert restored["bytes"] == value["bytes"]
    assert isinstance(restored["record"], Record)
    assert restored["record"].__bytes__ == "not a bytes tag"
    assert restored["record"].data == payload
    assert isinstance(restored["opaque"], Opaque)
    assert (restored["opaque"].type_name, restored["opaque"].text) == ("Client", "<client>")
    assert restored["temperature"] is OMIT_TEMPERATURE

    for tag in ("__callable__", "__object__", "__handle__"):
        marker = object()
        reference = {tag: 7, "name": "live"}

        def refs(obj):
            return reference if obj is marker else None

        def resolve(ref):
            assert ref == reference
            return marker

        assert encode(marker, refs) is reference
        assert decode(reference, resolve) is marker
        encoded = json.loads(json.dumps(encode([payload, marker], refs)))
        restored = decode(encoded, resolve)
        assert restored[0] == payload
        assert restored[1] is marker


@pytest.mark.platforms("any")  # real child process, provider RPC, and nested parent facade calls
def test_reserved_dictionaries_round_trip_through_host_and_parent_state(tmp_path, monkeypatch):
    from agent import image_gen_registry
    from agent.image_gen_provider import ImageGenProvider
    from hermes_cli.plugins import PluginManager
    from tests.hermes_cli.test_plugin_host import _home_with_plugins

    _home_with_plugins(tmp_path, monkeypatch, {"wireecho": '''
import os
from agent.image_gen_provider import ImageGenProvider

class Echo(ImageGenProvider):
    def __init__(self, ctx):
        self._ctx = ctx
    @property
    def name(self):
        return "wireecho"
    def generate(self, prompt, aspect_ratio="landscape", **kwargs):
        self._ctx.state.set("payload", kwargs["payload"])
        return {"pid": os.getpid(), "payload": self._ctx.state.get("payload"),
                "attribute": self.attribute}

def register(ctx):
    handle = ctx.register_hook("post_tool_call", lambda **kwargs: kwargs)
    assert handle.active
    handle.dispose()
    ctx.register_image_gen_provider(Echo(ctx))
'''})
    manager = PluginManager()
    host = manager._plugin_host()
    try:
        manager.discover_and_load()
        assert manager._plugins["wireecho"].error is None
        assert host.pid != os.getpid() and host.alive
        provider = image_gen_registry.get_provider("wireecho")
        assert isinstance(provider, ImageGenProvider)
        payload = _dictionary_payload()
        attribute = {"__opaque__": "ordinary metadata", "status": 200}
        provider.attribute = attribute
        assert provider.generate("echo", payload=payload) == {
            "pid": host.pid, "payload": payload, "attribute": attribute,
        }
    finally:
        try:
            manager.unload()
        finally:
            host.shutdown()

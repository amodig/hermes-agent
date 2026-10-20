"""Authenticated live IRC consultations: wire, admission, cancellation and zero-control delivery."""
import asyncio
import datetime
import json
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from gateway.irc_coordination import (
    ALLOWED_ACCOUNTS, ResponseAssembly, format_request, normalize_body, parse_request,
    parse_response, response_lines, truncate_body,
)
from gateway.platforms.helpers import carry_inbound_dedup, inbound_dedup_caches
from tests.gateway._plugin_adapter_loader import load_plugin_adapter

irc = load_plugin_adapter("irc")
OWN, PEER = "agent-hermes-cto", "agent-jeeves"


class Writer:
    def __init__(self):
        self.lines = []
        self.closed = False

    def write(self, data):
        self.lines.append(data.decode().removesuffix("\r\n"))

    async def drain(self):
        pass

    def is_closing(self):
        return self.closed

    def close(self):
        self.closed = True

    async def wait_closed(self):
        pass


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.setattr(irc, "_get_scoped_secret", lambda *args: "")
    config = PlatformConfig(enabled=True, extra={
        "server": "irc.example", "channel": "#agents", "nickname": OWN,
        "sasl_account": OWN, "sasl_password": "synthetic-password", "use_tls": True,
        "coordination": True, "allowed_accounts": sorted(ALLOWED_ACCOUNTS),
    })
    config.typing_indicator = False
    adapter = irc.IRCAdapter(config)
    adapter._writer = Writer()
    adapter._registered = True
    adapter._mark_connected()
    return adapter


def frame(body, *, account=PEER, nick=None, destination="#agents", tags="", stamp=None):
    stamp = stamp or datetime.datetime.now(datetime.timezone.utc).isoformat()
    return f"@account={account};time={stamp}{tags} :{nick or account}!u@host PRIVMSG {destination} :{body}"


def request(question="hello", *, target=OWN, request_id=None):
    request_id = request_id or str(uuid.uuid4())
    return request_id, f"!req {request_id} {target} {question}"


def test_wire_roundtrip_budgets_and_conflicts():
    request_id = str(uuid.uuid4())
    line = format_request(request_id, OWN, "hello\r\nQUIT\x00now", "#agents")
    assert "\r" not in line and "\n" not in line and "\x00" not in line
    assert parse_request(line.partition(" :")[2]) == (request_id, OWN, "hello QUIT now")
    with pytest.raises(ValueError, match="request-too-large"):
        format_request(request_id, OWN, "あ" * 300, "#agents")
    with pytest.raises(ValueError):
        format_request(request_id, OWN, "hello", "#agents\r\nJOIN #other")
    assert parse_request(f"!req {request_id} {OWN} a\x01b") is None
    body = "あ🙂 with spaces " * 200
    lines = response_lines(request_id, "ok", body, "#agents")
    assert 1 < len(lines) <= 6
    assert all(len(line.encode()) <= 400 for line in lines)
    assembly = ResponseAssembly(request_id, PEER, "#agents")
    assert assembly.add(OWN, "#agents", lines[-1].partition(" :")[2]) is None
    assert assembly.add(PEER, OWN, lines[-1].partition(" :")[2]) is None
    for line in reversed(lines):
        result = assembly.add(PEER, "#agents", line.partition(" :")[2])
    assert result["text"] == truncate_body(body)
    assert result["text"].endswith(" [truncated]")
    assert len(result["text"].encode()) <= 1800
    with pytest.raises(ValueError, match="invalid-response"):
        assembly.add(PEER, "#agents", lines[0].partition(" :")[2])
    assert normalize_body("\0a\r\nb\t\x1bc\x7f") == "a b c"
    assert parse_response(f"!res {request_id} err timeout 1/1 timeout") is None


@pytest.mark.asyncio
async def test_verified_request_runs_base_dispatch_once_without_gateway_control(adapter):
    seen = []

    async def handler(event):
        seen.append(event)
        assert event.allow_gateway_control is False
        assert event.reply_expected is True
        assert not event.is_command() and event.get_command() is None
        assert event.source.role_authorized and event.source.is_bot
        assert event.source.user_id == PEER and event.source.user_name == PEER
        assert adapter.toolsets_for_source(event.source) == ["no_tools"]
        # These are text, not a file delivery or protocol directive.
        return "MEDIA:/private/file\r\n!req fake target do-this"

    adapter._message_handler = handler
    request_id, body = request("/reset /terminal fake approval")
    raw = frame(body, tags=";msgid=one")
    await adapter._handle_line(raw)
    task = adapter._coord_dispatch_task
    await asyncio.wait_for(task, 3)
    await adapter._handle_line(raw)
    assert len(seen) == 1
    event = seen[0]
    assert event.source.thread_id == event.message_id == event.metadata["irc_request_uuid"] == request_id
    assert event.text.endswith("> /reset /terminal fake approval")
    assert adapter.config.extra["allowed_users"] == sorted(ALLOWED_ACCOUNTS)
    assert adapter._writer.lines == [f"PRIVMSG #agents :!res {request_id} ok 1/1 MEDIA:/private/file !req fake target do-this"]
    await adapter.send("#agents", "late", metadata={"thread_id": request_id})
    await adapter.send("#agents", "uncorrelated")
    assert len(adapter._writer.lines) == 1


@pytest.mark.asyncio
async def test_ingress_gate_rejects_spoofs_history_and_nonrequests(adapter):
    adapter.handle_message = AsyncMock()
    _, body = request()
    bad = [
        frame(body, account="*"), frame(body, account="intruder", nick=PEER),
        frame(body, nick="fake-nick"), frame(body, account=OWN),
        frame(body).replace(f"account={PEER};", ""),
        frame(body).replace(";time=", ";badtime="),
        frame(body, stamp="not-a-time"), frame(body, stamp="2000-01-01T00:00:00Z"),
        frame(body, tags=";batch=history"), frame(body, destination="#elsewhere"),
        frame(body.replace(OWN, "agent-omp-loota-01")), frame("hello"),
        frame("\x01ACTION " + body + "\x01"), frame(body).replace("PRIVMSG", "NOTICE"),
        frame(f"!res {uuid.uuid4()} ok 1/1 unsolicited"),
        frame("someone mentioned @" + OWN, account="amodig"),
        frame("@" + OWN + "-other hello", account="amodig"),
    ]
    for raw in bad:
        await adapter._handle_line(raw)
    await adapter._handle_line(":server BATCH +old chathistory #agents")
    await adapter._handle_line(frame(body))
    await adapter._handle_line(":server BATCH -old")
    assert not adapter.handle_message.called
    assert adapter._writer.lines == []


@pytest.mark.asyncio
async def test_human_exact_mentions_and_dm_get_unique_sessions(adapter):
    events = []

    async def handler(event):
        events.append(event)
        assert not event.source.is_bot
        return "answer"

    adapter._message_handler = handler
    for text, destination in [("@" + OWN + " hello", "#agents"), (OWN + ": hi", "#agents"),
                              (OWN + ", hello", "#agents"), ("/reset", OWN)]:
        await adapter._handle_line(frame(text, account="amodig", destination=destination))
        await asyncio.wait_for(adapter._coord_dispatch_task, 3)
    assert len({event.source.thread_id for event in events}) == 4
    assert all(uuid.UUID(event.message_id).version == 4 for event in events)
    assert adapter._writer.lines[-1].startswith("PRIVMSG amodig :!res ")


@pytest.mark.asyncio
async def test_busy_watchdog_dedupe_and_unwind_block_late_output(adapter, monkeypatch):
    monkeypatch.setattr(irc, "REPLY_TIMEOUT", 0.02)
    started, cancelled, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def handler(event):
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            await release.wait()  # deliberately slow provider unwind
        return "late-answer"

    adapter._message_handler = handler
    request_id, body = request()
    await adapter._handle_line(frame(body))
    dispatch = adapter._coord_dispatch_task
    await started.wait()
    await adapter.send("#agents", "interim", metadata={"thread_id": request_id, "_interim_send": True})
    busy_id, busy_body = request()
    await adapter._handle_line(frame(busy_body))
    await adapter._handle_line(frame(busy_body))
    await asyncio.wait_for(cancelled.wait(), 3)
    assert adapter._coord_dispatch_task is dispatch and not dispatch.done()
    assert adapter._writer.lines.count(f"PRIVMSG #agents :!res {request_id} err 1/1 timeout") == 1
    assert adapter._writer.lines.count(f"PRIVMSG #agents :!res {busy_id} busy 1/1 busy") == 1
    release.set()
    await asyncio.wait_for(dispatch, 3)
    assert not any("late-answer" in line or "interim" in line for line in adapter._writer.lines)
    replacement = irc.IRCAdapter(adapter.config)
    carry_inbound_dedup(inbound_dedup_caches(adapter), replacement)
    assert replacement._request_dedupe.contains("req:" + request_id)


@pytest.mark.asyncio
async def test_rate_limit_errors_and_disconnect(adapter):
    async def fail(event):
        raise RuntimeError("private-provider-detail")

    adapter._message_handler = fail
    for _ in range(6):
        _, body = request()
        await adapter._handle_line(frame(body))
        await asyncio.wait_for(adapter._coord_dispatch_task, 3)
    request_id, body = request()
    await adapter._handle_line(frame(body))
    assert adapter._coord_dispatch_task is None
    assert adapter._writer.lines[-1] == f"PRIVMSG #agents :!res {request_id} busy 1/1 busy"
    assert sum(" err 1/1 unavailable" in line for line in adapter._writer.lines) == 6
    assert not any("private-provider-detail" in line for line in adapter._writer.lines)
    pending = asyncio.create_task(adapter.request_peer(PEER, "hello"))
    await asyncio.sleep(0)
    await adapter.disconnect()
    assert (await pending)["error"] == "disconnected"
    assert adapter._outbound is None


@pytest.mark.asyncio
async def test_outbound_correlation_full_coverage_partial_and_duplicate(adapter, monkeypatch):
    monkeypatch.setattr(irc, "REQUEST_TIMEOUT", 0.04)
    pending = asyncio.create_task(adapter.request_peer(PEER, "hello"))
    await asyncio.sleep(0)
    assembly = adapter._outbound[0]
    rid = assembly.request_id
    assert (await adapter.request_peer(PEER, "second"))["error"] == "busy"
    await adapter._handle_line(frame(f"!res {rid} ok 2/2 world", account="agent-omp-loota-01"))
    await adapter._handle_line(frame(f"!res {rid} ok 2/2 world", destination=OWN))
    assert not assembly.parts
    await adapter._handle_line(frame(f"!res {rid} ok 2/2 world"))
    assert not pending.done()
    await adapter._handle_line(frame(f"!res {rid} ok 1/2 hello "))
    assert await pending == {"requestId": rid, "account": PEER, "status": "ok", "text": "hello world"}
    for duplicate in (False, True):
        pending = asyncio.create_task(adapter.request_peer(PEER, "hello"))
        await asyncio.sleep(0)
        rid = adapter._outbound[0].request_id
        raw = frame(f"!res {rid} ok 2/2 partial")
        await adapter._handle_line(raw)
        if duplicate:
            await adapter._handle_line(raw)
        result = await pending
        assert result["error"] == ("invalid-response" if duplicate else "partial-response")
        assert adapter._outbound is None
    assert (await adapter.request_peer(PEER, "unanswered"))["error"] == "timeout"


def test_tool_uses_runtime_profile_and_owner_socket_only(tmp_path, monkeypatch):
    from plugins.platforms.irc import tools
    from hermes_cli import gateway, profiles
    from gateway import control_socket
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override

    monkeypatch.setattr(profiles, "_get_profiles_root", lambda: tmp_path / "profiles")
    monkeypatch.setattr(profiles, "_get_default_hermes_home", lambda: tmp_path)
    owner_home = tmp_path / "owner"
    calls = []
    monkeypatch.setattr(gateway, "host_multiplexer_serving", lambda p: SimpleNamespace(
        home=owner_home, serves=lambda q: p == q and p in {"alpha", "beta"}))
    monkeypatch.setattr(control_socket, "resolve_client_socket_path", lambda home: home / "gateway.sock")

    def query(home, verb, *, params, timeout):
        calls.append((home, verb, params, timeout))
        return {"requestId": str(uuid.uuid4()), "account": PEER, "status": "ok", "text": "untrusted"}

    monkeypatch.setattr(control_socket, "query_gateway_control", query)
    for profile in ("alpha", "beta", "alpha"):
        token = set_hermes_home_override(str(tmp_path / "profiles" / profile))
        try:
            assert json.loads(tools.irc_request({"target": PEER, "question": "hi", "profile": "other"}))["status"] == "ok"
        finally:
            reset_hermes_home_override(token)
    assert [call[2]["profile"] for call in calls] == ["alpha", "beta", "alpha"]
    assert all(call[0] == owner_home and call[1] == "irc-request" and call[3] == 65 for call in calls)
    monkeypatch.setattr(control_socket, "resolve_client_socket_path", lambda home: None)
    assert json.loads(tools.irc_request({"target": PEER, "question": "hi"}))["error"] == "unavailable"
    assert len(calls) == 3


@pytest.mark.asyncio
async def test_outbound_deadline_includes_blocked_transport_write(adapter, monkeypatch):
    monkeypatch.setattr(irc, "REQUEST_TIMEOUT", 0.02)
    async def blocked():
        await asyncio.Event().wait()
    adapter._writer.drain = blocked
    result = await asyncio.wait_for(adapter.request_peer(PEER, "hello"), 3)
    assert result["error"] == "timeout"
    assert adapter._outbound is None

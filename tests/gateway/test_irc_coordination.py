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
    ALLOWED_ACCOUNTS, REQUIRED_CAPS, ResponseAssembly, format_request, normalize_body, parse_request,
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
        "server": "z300.barracuda-trout.ts.net", "port": 6697, "channel": "#agents", "nickname": OWN,
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
@pytest.mark.parametrize("exit_mode", ["normal", "cancelled", "inactivity"])
async def test_watchdog_holds_admission_until_executor_worker_finishes(adapter, monkeypatch, exit_mode):
    import threading
    import gateway.run_turn as turn_module

    monkeypatch.setattr(irc, "REPLY_TIMEOUT", 0.02)
    started, release, worker_done = asyncio.Event(), threading.Event(), threading.Event()
    loop = asyncio.get_running_loop()
    turns = []
    interrupt = adapter.interrupt_session_activity
    adapter.interrupt_session_activity = AsyncMock(wraps=interrupt)

    async def short_poll(tasks, *, timeout):
        return await asyncio.wait(tasks, timeout=min(timeout, 0.01))

    monkeypatch.setattr(turn_module, "asyncio", SimpleNamespace(
        wait=short_poll, shield=asyncio.shield, CancelledError=asyncio.CancelledError))
    runner = SimpleNamespace(
        _delivery_adapter_for=lambda source: adapter,
        _run_agent_backup_interrupt_check=AsyncMock(),
        _run_agent_timeout_result=lambda *args: {"final_response": "late-answer"})

    def run_sync():
        loop.call_soon_threadsafe(started.set)
        try:
            release.wait(5)
            return {"final_response": "late-answer"}
        finally:
            worker_done.set()

    async def handler(event):
        turns.append(event)
        worker = SimpleNamespace(
            executor_task=asyncio.create_task(asyncio.to_thread(run_sync)),
            agent_timeout=1 if exit_mode == "inactivity" else None,
            timeout_fired=threading.Event())
        worker.timeout_fired.set()
        ctx = SimpleNamespace(source=event.source, agent_holder=[None])
        result = await turn_module.GatewayTurnMixin._run_agent_await_turn_worker(
            runner, worker, ctx, asyncio.Event(), None)
        return result["final_response"]

    adapter._message_handler = handler
    request_id, body = request()
    await adapter._handle_line(frame(body))
    dispatch = adapter._coord_dispatch_task
    try:
        await asyncio.wait_for(started.wait(), 3)
        event = turns[0]
        key = adapter._event_session_key(event)
        await asyncio.wait_for(adapter._active_sessions[key].wait(), 3)
        adapter.interrupt_session_activity.assert_awaited_once()
        if exit_mode == "cancelled":
            # Cancellation of the base coroutine must not free its executor worker.
            task = adapter._session_tasks[key]
            adapter._expected_cancelled_tasks.add(task)
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
        await adapter.send("#agents", "interim", metadata={"thread_id": request_id, "_interim_send": True})
        busy_id, busy_body = request()
        await adapter._handle_line(frame(busy_body))
        await adapter._handle_line(frame(busy_body))
        assert not worker_done.is_set() and len(turns) == 1
        assert adapter._coord_dispatch_task is dispatch and not dispatch.done()
        assert adapter._writer.lines.count(f"PRIVMSG #agents :!res {request_id} err 1/1 timeout") == 1
        assert adapter._writer.lines.count(f"PRIVMSG #agents :!res {busy_id} busy 1/1 busy") == 1
    finally:
        release.set()
        await asyncio.wait_for(dispatch, 3)
    assert worker_done.is_set() and adapter._coord_dispatch_task is None
    assert not any("late-answer" in line or "interim" in line for line in adapter._writer.lines)
    replacement = irc.IRCAdapter(adapter.config)
    carry_inbound_dedup(inbound_dedup_caches(adapter), replacement)
    assert replacement._request_dedupe.contains("req:" + request_id)
    monkeypatch.setattr(irc, "REPLY_TIMEOUT", 55.0)  # The fresh turn is not a timeout probe.
    adapter._message_handler = AsyncMock(return_value="fresh-answer")
    await adapter._handle_line(frame(request()[1]))
    await asyncio.wait_for(adapter._coord_dispatch_task, 3)
    adapter._message_handler.assert_awaited_once()
    assert adapter._writer.lines[-1].endswith(" ok 1/1 fresh-answer")


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
    original_query = control_socket.query_gateway_control

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
    monkeypatch.setattr(control_socket, "query_gateway_control", original_query)
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


@pytest.mark.asyncio
@pytest.mark.parametrize("capability", sorted(REQUIRED_CAPS))
async def test_required_capability_withdrawal_disconnects_and_stops_admission(adapter, capability):
    adapter.handle_message = AsyncMock()
    notified = AsyncMock()
    adapter.set_fatal_error_handler(notified)
    writer = adapter._writer
    await adapter._handle_line(f":server CAP {OWN} DEL :unrelated-cap")
    assert adapter.is_connected and not writer.closed
    future = asyncio.get_running_loop().create_future()
    adapter._outbound = (ResponseAssembly(str(uuid.uuid4()), PEER, "#agents"), future)
    dispatch = asyncio.create_task(asyncio.Event().wait())
    adapter._coord_dispatch_task = dispatch
    await adapter._handle_line(f":server CAP {OWN} DEL :unrelated-cap {capability}")
    assert not adapter.is_connected and writer.closed
    assert adapter.fatal_error_code == "capability_removed"
    assert adapter.fatal_error_retryable is False
    assert future.cancelled() and dispatch.cancelled()
    notified.assert_awaited_once_with(adapter)
    await adapter._handle_line(frame(request()[1]))
    adapter.handle_message.assert_not_awaited()
    assert not any(line.startswith("PRIVMSG ") for line in writer.lines)


@pytest.mark.asyncio
async def test_timestamp_warnings_are_bounded_and_private(adapter, monkeypatch, caplog):
    adapter.handle_message = AsyncMock()
    now = datetime.datetime.now(datetime.timezone.utc)
    clock = [100.0]
    monkeypatch.setattr(irc, "time", SimpleNamespace(time=lambda: now.timestamp(), monotonic=lambda: clock[0]))
    adapter._connection_started = now.timestamp() - 1
    stamps = {
        "missing": None,
        "malformed": "10000-01-01T00:00:00Z",
        "naive": now.replace(tzinfo=None).isoformat(),
        "future": (now + datetime.timedelta(seconds=6)).isoformat(),
        "before-connection-start": (now - datetime.timedelta(seconds=2)).isoformat(),
    }
    body = request("private-question-not-for-logs")[1]
    frames = [
        f"@account={PEER}" + (f";time={stamp}" if stamp is not None else "")
        + f" :{PEER}!u@host PRIVMSG #agents :{body}"
        for stamp in stamps.values()
    ]
    for _ in range(100):
        for raw in frames:
            await adapter._handle_line(raw)
    expected = {f"IRC: rejected server-time account={PEER} category={category}" for category in stamps}
    warnings = [r.message for r in caplog.records if r.name == irc.logger.name]
    assert len(warnings) == 5 and set(warnings) == expected
    assert set(adapter._timestamp_warning_at) == set(stamps)
    clock[0] += 59
    for raw in frames:
        await adapter._handle_line(raw)
    assert len([r for r in caplog.records if r.name == irc.logger.name]) == 5
    clock[0] += 1
    for raw in frames:
        await adapter._handle_line(raw)
    assert len([r for r in caplog.records if r.name == irc.logger.name]) == 10
    # Unauthenticated traffic does not produce even timestamp diagnostics.
    await adapter._handle_line(frames[0].replace(PEER, "unknown"))
    assert len([r for r in caplog.records if r.name == irc.logger.name]) == 10
    assert "private-question-not-for-logs" not in caplog.text
    adapter.handle_message.assert_not_awaited()
    assert adapter._coord_dispatch_task is None and adapter._writer.lines == []


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [OverflowError, OSError])
async def test_timestamp_conversion_failures_warn_without_dispatch(adapter, monkeypatch, caplog, error):
    raw = frame(request("private-conversion-question")[1])

    def timestamp():
        raise error("private-platform-diagnostic")

    stamp = SimpleNamespace(tzinfo=datetime.timezone.utc, timestamp=timestamp)
    monkeypatch.setattr(irc, "datetime", SimpleNamespace(
        datetime=SimpleNamespace(fromisoformat=lambda value: stamp)))
    await adapter._handle_line(raw)
    assert f"account={PEER} category=malformed" in caplog.text
    assert "private-" not in caplog.text
    assert adapter._coord_dispatch_task is None and adapter._writer.lines == []


def test_tool_availability_merges_mixed_platform_sections(monkeypatch):
    from plugins.platforms.irc import tools
    config = {"platforms": {"telegram": {"enabled": True}},
              "gateway": {"platforms": {"irc": {"enabled": True, "extra": {"coordination": True}}}}}
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: config)
    assert tools._available()
    config["platforms"]["irc"] = {"enabled": False}
    assert not tools._available()  # Same top-level precedence as gateway configuration.
    config["platforms"]["irc"] = {"enabled": True}
    assert tools._available()  # Nested extra survives the top-level override.


def test_tool_reaches_windows_pipe_without_unix_socket(tmp_path, monkeypatch):
    from plugins.platforms.irc import tools
    from gateway import control_socket

    calls = []
    monkeypatch.setattr("hermes_cli.profiles.current_profile_name", lambda: "default")
    monkeypatch.setattr("hermes_cli.gateway.host_multiplexer_serving", lambda profile: None)
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(control_socket, "_IS_WINDOWS", True)

    def no_unix_socket(home):
        raise AssertionError("Unix socket lookup must not gate Windows pipes")

    def pipe_query(home, payload, timeout):
        calls.append((home, json.loads(payload), timeout))
        return json.dumps({"ok": True, "result": {
            "requestId": str(uuid.uuid4()), "account": PEER, "status": "ok", "text": "peer data"}}).encode()

    monkeypatch.setattr(control_socket, "resolve_client_socket_path", no_unix_socket)
    monkeypatch.setattr(control_socket, "_query_windows_pipe", pipe_query)
    assert json.loads(tools.irc_request({"target": PEER, "question": "hello"}))["status"] == "ok"
    assert len(calls) == 1 and calls[0][0] == tmp_path and calls[0][2] == 65
    assert calls[0][1]["params"] == {"profile": "default", "target": PEER, "question": "hello"}

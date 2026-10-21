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


@pytest.fixture(autouse=True)
def isolate_coordination_process_state(monkeypatch):
    import gateway.irc_coordination as coordination

    monkeypatch.setattr(coordination, "_coordination_states", {})


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
@pytest.mark.parametrize("exit_mode", ["normal", "cancelled", "inactivity", "disconnect", "shutdown"])
async def test_watchdog_holds_admission_until_executor_worker_finishes(adapter, monkeypatch, exit_mode):
    import threading
    import gateway.run_turn as turn_module
    from unittest.mock import Mock

    monkeypatch.setattr(irc, "REPLY_TIMEOUT", 0.02)
    started, release, worker_done = asyncio.Event(), threading.Event(), threading.Event()
    loop = asyncio.get_running_loop()
    turns = []
    receiver, writers = adapter, [adapter._writer]
    release_lock = Mock()
    monkeypatch.setattr(adapter, "_release_platform_lock", release_lock)
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
            release.wait(15)
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
        if exit_mode in {"disconnect", "shutdown"}:
            await asyncio.wait_for(adapter.disconnect(), 1)
            await asyncio.wait_for(adapter.disconnect(), 1)  # Repeated cancellation must retain ownership.
            assert writers[0].closed and adapter._writer is None and not adapter.is_connected
            if exit_mode == "shutdown":
                await asyncio.wait_for(adapter.cancel_background_tasks(), 6)
            assert not worker_done.is_set() and not adapter._active_requests
            release_lock.assert_not_called()
            receiver = irc.IRCAdapter(adapter.config)
            carry_inbound_dedup(inbound_dedup_caches(adapter), receiver)
            open_connection = AsyncMock()
            monkeypatch.setattr(irc, "_open_connection", open_connection)
            assert not await receiver.connect(is_reconnect=True)
            assert receiver.fatal_error_code == "coordination_busy" and receiver.fatal_error_retryable
            open_connection.assert_not_awaited()
            # Even a replacement with a live transport cannot bypass the account's worker lease.
            receiver._writer = Writer()
            receiver._registered = True
            receiver._mark_connected()
            writers.append(receiver._writer)
        await adapter.send("#agents", "interim", metadata={"thread_id": request_id, "_interim_send": True})
        busy_id, busy_body = request()
        await receiver._handle_line(frame(busy_body))
        await receiver._handle_line(frame(busy_body))
        assert not worker_done.is_set() and len(turns) == 1
        assert adapter._coord_dispatch_task is dispatch and not dispatch.done()
        lines = [line for writer in writers for line in writer.lines]
        assert lines.count(f"PRIVMSG #agents :!res {request_id} err 1/1 timeout") == 1
        assert lines.count(f"PRIVMSG #agents :!res {busy_id} busy 1/1 busy") == 1
    finally:
        release.set()
        await asyncio.wait_for(dispatch, 3)
    assert worker_done.is_set() and adapter._coord_dispatch_task is None
    assert not any("late-answer" in line or "interim" in line for writer in writers for line in writer.lines)
    if exit_mode in {"disconnect", "shutdown"}:
        release_lock.assert_called_once()
    replacement = irc.IRCAdapter(adapter.config)
    carry_inbound_dedup(inbound_dedup_caches(adapter), replacement)
    assert replacement._request_dedupe.contains("req:" + request_id)
    adapter = receiver
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


@pytest.mark.asyncio
async def test_gateway_proxy_refuses_coordination_without_model_or_tools(adapter, monkeypatch):
    from unittest.mock import Mock
    from gateway.run_turn import GatewayTurnMixin

    monkeypatch.setattr("agent.secret_scope.get_secret", lambda name: "")
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: {
        "gateway": {"proxy_url": "https://remote-model.invalid"}})
    remote_tool = Mock()

    async def remote_turn(**kwargs):
        remote_tool()
        return {"final_response": "ordinary proxy answer"}

    runner = SimpleNamespace(
        _delivery_adapter_for=lambda source: adapter,
        _run_agent_via_proxy=AsyncMock(side_effect=remote_turn),
        _run_agent_display_settings=Mock(side_effect=AssertionError("local model must not run")))
    runner._get_proxy_url = lambda: GatewayTurnMixin._get_proxy_url(runner)
    results, sources = [], []

    async def handler(event):
        sources.append(event.source)
        result = await GatewayTurnMixin._run_agent_inner(
            runner, event.text, "", [], event.source, "synthetic-session")
        results.append(result)
        return result["final_response"]

    adapter._message_handler = handler
    request_id, body = request()
    await adapter._handle_line(frame(body))
    await asyncio.wait_for(adapter._coord_dispatch_task, 3)
    await adapter._handle_line(frame(body))
    assert adapter._writer.lines == [f"PRIVMSG #agents :!res {request_id} err 1/1 unavailable"]
    assert not adapter._active_requests and adapter._coord_dispatch_task is None
    assert results[0]["failed"] and results[0]["api_calls"] == 0 and results[0]["tools"] == []
    runner._run_agent_via_proxy.assert_not_awaited()
    runner._run_agent_display_settings.assert_not_called()
    remote_tool.assert_not_called()
    ordinary = irc.IRCAdapter(PlatformConfig(extra={"server": "irc.example", "channel": "#ordinary"}))
    runner._delivery_adapter_for = lambda source: ordinary
    result = await GatewayTurnMixin._run_agent_inner(runner, "hello", "", [], sources[0], "ordinary-session")
    assert result["final_response"] == "ordinary proxy answer"
    runner._run_agent_via_proxy.assert_awaited_once()
    remote_tool.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("interim_enabled", [False, True])
async def test_streaming_previews_never_consume_coordination_response(adapter, monkeypatch, interim_enabled):
    from unittest.mock import Mock
    from gateway.config import StreamingConfig
    from gateway.run_turn import GatewayTurnMixin
    from gateway.run_turn_runner import TurnRunner
    from gateway.turn_context import TurnContext
    from gateway.stream_consumer import GatewayStreamConsumer

    factory = Mock(wraps=GatewayStreamConsumer)
    monkeypatch.setattr("gateway.stream_consumer.GatewayStreamConsumer", factory)
    runner = SimpleNamespace(
        config=SimpleNamespace(streaming=StreamingConfig(enabled=True)),
        _delivery_adapter_for=lambda source: adapter)
    runner._build_stream_consumer_config = lambda *args, **kwargs: (
        GatewayTurnMixin._build_stream_consumer_config(runner, *args, **kwargs))
    contexts = []

    async def handler(event):
        ctx = TurnContext(
            source=event.source, user_config={}, resolve_display_setting=lambda *args: True,
            interim_assistant_messages_enabled=interim_enabled, _run_still_current=lambda: True,
            _status_adapter=adapter, _status_chat_id=event.source.chat_id,
            _status_thread_metadata={"thread_id": event.message_id})
        contexts.append(ctx)
        assert TurnRunner(runner, ctx)._setup_stream_consumer("irc") == (None, None, None, False)
        assert adapter.SUPPORTS_MESSAGE_EDITING is False
        # A stray/older consumer must also be unable to close this request.
        for text in ("preview one", "preview two", "preview three"):
            await adapter.send("#agents", text, metadata={"thread_id": event.message_id, "expect_edits": True})
            assert event.message_id in adapter._active_requests
        return "complete final answer"

    adapter._message_handler = handler
    request_id, body = request()
    await adapter._handle_line(frame(body))
    await asyncio.wait_for(adapter._coord_dispatch_task, 3)
    factory.assert_not_called()
    assert adapter._writer.lines == [f"PRIVMSG #agents :!res {request_id} ok 1/1 complete final answer"]
    ordinary = irc.IRCAdapter(PlatformConfig(extra={"server": "irc.example", "channel": "#ordinary"}))
    assert getattr(ordinary, "SUPPORTS_MESSAGE_EDITING", True) is True
    runner._delivery_adapter_for = lambda source: ordinary
    consumer, delta, _, _ = TurnRunner(runner, contexts[0])._setup_stream_consumer("irc")
    assert consumer is not None and delta is not None
    factory.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("voice_mode", [None, "all"])
@pytest.mark.parametrize("voice_input", [False, True])
async def test_auto_tts_cannot_replace_coordination_answer(adapter, monkeypatch, voice_mode, voice_input):
    from unittest.mock import Mock
    from gateway.run_turn import GatewayTurnMixin
    from gateway.run_voice import GatewayVoiceMixin
    from gateway.platforms.event import MessageType

    synthesis = Mock(side_effect=AssertionError("coordination must not synthesize speech"))
    monkeypatch.setattr("tools.tts_tool.text_to_speech_tool", synthesis)
    adapter._auto_tts_default = True
    adapter._auto_tts_enabled_chats.add("#agents")
    runner = SimpleNamespace(
        _voice_mode={"irc:#agents": voice_mode} if voice_mode else {},
        _voice_key_for_source=lambda source: "irc:#agents",
        _delivery_adapter_for=lambda source: adapter)
    runner._should_send_voice_reply = lambda *args, **kwargs: (
        GatewayVoiceMixin._should_send_voice_reply(runner, *args, **kwargs))

    async def send_voice(*args):
        return await GatewayVoiceMixin._send_voice_reply(runner, *args)

    runner._send_voice_reply = AsyncMock(wraps=send_voice)
    events = []

    async def handler(event):
        events.append(event)
        if voice_input:
            event.message_type = MessageType.VOICE  # Exercise the base voice fallback too.
        assert adapter._should_auto_tts_for_chat("#agents") is False
        return await GatewayTurnMixin._hmwa_deliver_turn_response(
            runner, event, event.source, SimpleNamespace(session_id="synthetic"),
            adapter._event_session_key(event), None, {}, [], "model answer", "", False)

    adapter._message_handler = handler
    request_id, body = request()
    await adapter._handle_line(frame(body))
    await asyncio.wait_for(adapter._coord_dispatch_task, 3)
    synthesis.assert_not_called()
    runner._send_voice_reply.assert_not_awaited()
    assert adapter._writer.lines == [f"PRIVMSG #agents :!res {request_id} ok 1/1 model answer"]
    ordinary = irc.IRCAdapter(PlatformConfig(extra={"server": "irc.example", "channel": "#ordinary"}))
    ordinary._auto_tts_default = True
    runner._delivery_adapter_for = lambda source: ordinary
    assert ordinary._should_auto_tts_for_chat("#agents")
    assert runner._should_send_voice_reply(events[0], "ordinary answer", [], already_sent=True)


@pytest.mark.asyncio
async def test_coordination_reference_payload_stays_literal_without_io(adapter, monkeypatch, tmp_path):
    from unittest.mock import Mock
    import agent.context_references as references
    import gateway.run_inbound as inbound
    from gateway.platforms.event import MessageEvent

    (tmp_path / "literal.txt").write_text("LOCAL_SENTINEL")
    (tmp_path / "folder").mkdir()
    read_file = Mock(wraps=references._read_file_reference)
    list_folder = Mock(return_value="FOLDER_SENTINEL")
    git = Mock(return_value=(None, "GIT_SENTINEL"))
    fetch = AsyncMock(return_value="URL_SENTINEL")
    expand = AsyncMock(wraps=references.preprocess_context_references_async)
    rehome = Mock(wraps=inbound.rehome_inbound_media)
    monkeypatch.setattr(references, "_read_file_reference", read_file)
    monkeypatch.setattr(references, "_build_folder_listing", list_folder)
    monkeypatch.setattr(references, "_expand_git_reference", git)
    monkeypatch.setattr(references, "_fetch_url_content", fetch)
    monkeypatch.setattr(references, "preprocess_context_references_async", expand)
    monkeypatch.setattr(inbound, "rehome_inbound_media", rehome)
    monkeypatch.setattr("tools.terminal_scope.terminal_env",
                        lambda key, default=None: str(tmp_path) if key == "TERMINAL_CWD" else default)
    runner = SimpleNamespace(
        _session_key_for_source=lambda source: "synthetic-session",
        _consume_pending_native_image_paths=Mock(),
        _prefix_inbound_sender_context=lambda event, source, text: text,
        _classify_inbound_media=lambda *args: ([], [], [], []),
        _prepend_inbound_media_file_notes=lambda text, *args: text,
        _prepend_inbound_document_notes=lambda event, text: text,
        _prepend_inbound_reply_context=lambda event, source, text: text,
        _inbound_model_context_length=AsyncMock(return_value=128000))
    runner._expand_inbound_context_references = lambda *args: (
        inbound.GatewayInboundMixin._expand_inbound_context_references(runner, *args))
    question = "@file:literal.txt @folder:folder @url:https://example.invalid/doc @git:1 git@example.invalid:repo.git"
    events, prepared = [], []

    async def handler(event):
        events.append(event)
        text = await inbound.GatewayInboundMixin._prepare_inbound_message_text(
            runner, event=event, source=event.source, history=[])
        prepared.append(text)
        assert text.encode() == event.text.encode()
        assert text.partition("\n> ")[2] == question
        return "literal answer"

    adapter._message_handler = handler
    request_id, body = request(question)
    await adapter._handle_line(frame(body))
    await asyncio.wait_for(adapter._coord_dispatch_task, 3)
    assert len(prepared) == 1
    expand.assert_not_awaited()
    fetch.assert_not_awaited()
    for probe in (read_file, list_folder, git, rehome, runner._consume_pending_native_image_paths):
        probe.assert_not_called()
    assert adapter._writer.lines == [f"PRIVMSG #agents :!res {request_id} ok 1/1 literal answer"]
    # Ordinary IRC events still use the real reference parser/expander.
    ordinary = MessageEvent(text=question, source=events[0].source)
    text = await inbound.GatewayInboundMixin._prepare_inbound_message_text(
        runner, event=ordinary, source=ordinary.source, history=[])
    assert all(marker in text for marker in ("LOCAL_SENTINEL", "FOLDER_SENTINEL", "GIT_SENTINEL", "URL_SENTINEL"))
    expand.assert_awaited_once()
    fetch.assert_awaited_once()
    for probe in (read_file, list_folder, git, rehome, runner._consume_pending_native_image_paths):
        probe.assert_called_once()


@pytest.mark.asyncio
async def test_lifecycle_notices_cannot_seal_coordination_response(adapter, monkeypatch):
    from unittest.mock import Mock
    from gateway.run_turn_runner import TurnRunner
    from gateway.turn_context import TurnContext

    async def status_send(current, chat_id, kind, text, metadata):
        return await current.send(chat_id, text, metadata=metadata)

    status_sink = AsyncMock(side_effect=status_send)
    monkeypatch.setattr("gateway.run._send_or_update_status_coro", status_sink)
    monkeypatch.setattr("gateway.run._prepare_gateway_status_message", lambda platform, kind, text: text)
    monkeypatch.setattr("gateway.run.render_notice_line", lambda notice: notice)
    monkeypatch.setattr("gateway.warning_notifications.render_notification", lambda present, **kwargs: present())
    monkeypatch.setattr("gateway.warning_notifications.is_diagnostic_notice", lambda notice: False)
    agent = SimpleNamespace(tools=[], status_callback=Mock(), notice_callback=Mock())
    events = []

    async def emit(event, current):
        scheduled = []

        async def notice_send(source, text):
            return await current.send(source.chat_id, text, metadata={"thread_id": source.thread_id})

        runner = SimpleNamespace(_service_tier=None, _consume_pending_turn_sidecar_notes=lambda key: [],
                                 _deliver_platform_notice=AsyncMock(side_effect=notice_send))
        ctx = TurnContext(
            source=event.source, user_config={}, session_key=current._event_session_key(event),
            _hooks_ref=SimpleNamespace(loaded_hooks=[]), _run_still_current=lambda: True,
            _status_adapter=current, _status_chat_id=event.source.chat_id,
            _status_thread_metadata={"thread_id": event.message_id})
        turn = TurnRunner(runner, ctx)
        ctx._status_callback_sync = turn._status_callback_sync
        turn._attach_session_title_callback = lambda *args: None

        def schedule(coro, *args):
            task = asyncio.create_task(coro)
            scheduled.append(task)
            return task

        turn._schedule = schedule
        interim = lambda text: schedule(current.send(event.source.chat_id, text, metadata={"_interim_send": True}))
        turn._wire_turn_agent_callbacks(agent, {}, None, None, interim, not current.coordination)
        if agent.status_callback:
            agent.status_callback("memory_recall", "lifecycle status")
        if agent.notice_callback:
            agent.notice_callback("lifecycle notice")
        if agent.interim_assistant_callback:
            agent.interim_assistant_callback("ordinary interim")
        await asyncio.gather(*scheduled)
        return runner

    async def handler(event):
        events.append(event)
        runner = await emit(event, adapter)
        assert agent.status_callback is None and agent.notice_callback is None
        runner._deliver_platform_notice.assert_not_awaited()
        status_sink.assert_not_awaited()
        return "complete model answer"

    adapter._message_handler = handler
    request_id, body = request()
    await adapter._handle_line(frame(body))
    await asyncio.wait_for(adapter._coord_dispatch_task, 3)
    assert adapter._writer.lines == [f"PRIVMSG #agents :!res {request_id} ok 1/1 complete model answer"]
    ordinary = irc.IRCAdapter(PlatformConfig(extra={"server": "irc.example", "channel": "#ordinary"}))
    ordinary._writer = Writer()
    runner = await emit(events[0], ordinary)  # Reusing the same agent restores its normal callbacks.
    status_sink.assert_awaited_once()
    runner._deliver_platform_notice.assert_awaited_once()
    assert sorted(ordinary._writer.lines) == sorted(
        f"PRIVMSG #agents :{text}" for text in ("lifecycle status", "lifecycle notice", "ordinary interim"))


def test_request_boundary_reserves_relay_prefix_headroom():
    from gateway.irc_coordination import IRC_BYTES, PREFIX_HEADROOM

    request_id = str(uuid.uuid4())
    envelope = f"PRIVMSG #agents :!req {request_id} {OWN} "
    room = IRC_BYTES - PREFIX_HEADROOM - len(envelope.encode())
    question = "é" * (room // 2) + "x" * (room % 2)
    line = format_request(request_id, OWN, question, "#agents")
    assert len(line.encode()) + PREFIX_HEADROOM == IRC_BYTES
    prefix = f":{OWN}!hermes@relay.example "
    assert len((prefix + line).encode()) <= IRC_BYTES
    received = irc._parse_irc_message(prefix + line)
    assert parse_request(received["params"][1]) == (request_id, OWN, question)
    with pytest.raises(ValueError, match="request-too-large"):
        format_request(request_id, OWN, question + "x", "#agents")


@pytest.mark.asyncio
async def test_only_explicit_final_response_can_close_request(adapter, monkeypatch):
    from unittest.mock import Mock
    import gateway.run_turn as turn_module

    intervals = []

    async def advance_interval(seconds):
        intervals.append(seconds)

    monkeypatch.setattr("gateway.run._float_env", lambda key, default: 30)
    monkeypatch.setattr(turn_module, "asyncio", SimpleNamespace(sleep=advance_interval))
    disp = SimpleNamespace(
        _display_surface_mode=lambda *args, **kwargs: "full", user_config={}, platform_key="irc",
        resolve_display_setting=lambda *args: False)
    runner = SimpleNamespace(
        _delivery_adapter_for=lambda source: adapter,
        _should_emit_long_running_notification=Mock(side_effect=[True, True, False]),
        _agent_activity_summary=lambda agent: {})

    async def handler(event):
        for metadata in ({"thread_id": event.message_id},
                         {"thread_id": event.message_id, "notify": True},
                         {"thread_id": event.message_id, "_final_response": "true"}):
            await adapter.send("#agents", "not a final answer", metadata=metadata)
            assert event.message_id in adapter._active_requests
        ctx = SimpleNamespace(
            source=event.source, session_key="synthetic-session", agent_holder=[None],
            _status_thread_metadata={"thread_id": event.message_id}, _cleanup_progress=False)
        await turn_module.GatewayTurnMixin._run_agent_notify_long_running(runner, disp, ctx, [None])
        assert intervals == [30, 30] and adapter._writer.lines == []
        assert event.message_id in adapter._active_requests
        return "the final answer"

    adapter._message_handler = handler
    request_id, body = request()
    await adapter._handle_line(frame(body))
    await asyncio.wait_for(adapter._coord_dispatch_task, 3)
    assert adapter._writer.lines == [f"PRIVMSG #agents :!res {request_id} ok 1/1 the final answer"]
    await adapter.send("#agents", "late final", metadata={"thread_id": request_id, "_final_response": True})
    assert len(adapter._writer.lines) == 1 and not adapter._active_requests


@pytest.mark.asyncio
@pytest.mark.parametrize("coordination", [True, False])
@pytest.mark.parametrize("outcome", ["success", "provider-failure", "exception", "overflow-exception"])
async def test_handled_turn_failures_are_errors_not_successes(adapter, coordination, outcome):
    from dataclasses import replace
    from gateway.run_turn import GatewayTurnMixin

    if not coordination:
        adapter = irc.IRCAdapter(PlatformConfig(extra={
            "server": "irc.example", "channel": "#ordinary", "nickname": "ordinary-bot"}))
        adapter.config.typing_indicator = False
        adapter._writer = Writer()
    runner = SimpleNamespace(
        _delivery_adapter_for=lambda source: adapter, _should_send_voice_reply=lambda *args, **kwargs: False,
        _hmwa_stop_typing_for_turn=AsyncMock(), _STATUS_HINTS=GatewayTurnMixin._STATUS_HINTS)
    runner._hmwa_add_failed_turn_notice = lambda *args: GatewayTurnMixin._hmwa_add_failed_turn_notice(runner, *args)
    returned = []

    async def handler(original):
        # pre_gateway_dispatch uses replace(): the outcome must reach the original adapter event too.
        event = replace(original, text=original.text)
        key = adapter._event_session_key(event)
        if outcome in {"exception", "overflow-exception"}:
            try:
                error = RuntimeError("private-provider-detail")
                if outcome == "overflow-exception":
                    error.status_code = 400
                raise error
            except RuntimeError as error:
                text = await GatewayTurnMixin._hmwa_agent_error_reply(
                    runner, error, event, event.source, None, key,
                    SimpleNamespace(history=[{}] * 60, message_text=None))
        else:
            failed = outcome == "provider-failure"
            text = await GatewayTurnMixin._hmwa_deliver_turn_response(
                runner, event, event.source, SimpleNamespace(session_id="synthetic"), key, None,
                {"failed": failed}, [], "private-provider-detail" if failed else "successful answer", "", False)
        returned.append(text)
        return text

    adapter._message_handler = handler
    if coordination:
        request_id, body = request()
        await adapter._handle_line(frame(body))
        await asyncio.wait_for(adapter._coord_dispatch_task, 3)
        status, text = ("ok", "successful answer") if outcome == "success" else ("err", "unavailable")
        assert adapter._writer.lines == [f"PRIVMSG #agents :!res {request_id} {status} 1/1 {text}"]
    else:
        await adapter._handle_privmsg("amodig", "#ordinary", "ordinary-bot: hello")
        await asyncio.wait_for(next(iter(adapter._session_tasks.values())), 3)
        assert adapter._writer.lines == [
            f"PRIVMSG #ordinary :{line}" for line in adapter._split_message(returned[0], "#ordinary")]
    assert len(returned) == 1


@pytest.mark.parametrize("shape", ["flat", "extra"])
@pytest.mark.parametrize("enabled,coordination", [(True, True), (False, True), (True, False)])
def test_tool_catalog_matches_normalized_gateway_config(adapter, monkeypatch, shape, enabled, coordination):
    from plugins.platforms.irc import tools
    from tools.registry import ToolRegistry, invalidate_check_fn_cache

    settings = {**adapter.config.extra, "coordination": coordination}
    block = {"enabled": enabled, **(settings if shape == "flat" else {"extra": settings})}
    config = {"gateway": {"platforms": {"irc": block}}, "platform_toolsets": {"cli": ["irc_coord"]}}
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: config)
    normalized = PlatformConfig.from_dict(block)
    gateway_adapter = irc.IRCAdapter(normalized)
    expected = normalized.enabled and gateway_adapter.coordination
    assert gateway_adapter._config_error() is None
    catalog = ToolRegistry()
    tools.register_tools(SimpleNamespace(register_tool=catalog.register))
    invalidate_check_fn_cache()
    try:
        selected = set(catalog.get_tool_names_for_toolset(config["platform_toolsets"]["cli"][0]))
        definitions = catalog.get_definitions(selected)
        assert {item["function"]["name"] for item in definitions} == ({"irc_request"} if expected else set())
        assert tools._available() is expected
    finally:
        invalidate_check_fn_cache()


@pytest.mark.asyncio
async def test_admission_window_survives_adapter_replacement(adapter, monkeypatch):
    import time

    now = [1000.0]
    monkeypatch.setattr(irc, "time", SimpleNamespace(time=time.time, monotonic=lambda: now[0]))
    handler = AsyncMock(return_value="accepted")
    adapter._message_handler = handler
    bodies = []
    for _ in range(6):
        _, body = request()
        bodies.append(body)
        await adapter._handle_line(frame(body))
        await asyncio.wait_for(adapter._coord_dispatch_task, 3)
    assert handler.await_count == 6
    caches = inbound_dedup_caches(adapter)
    await adapter.disconnect()
    replacement = irc.IRCAdapter(adapter.config)
    carry_inbound_dedup(caches, replacement)
    replacement._writer = Writer()
    replacement._registered = True
    replacement._mark_connected()
    replacement._message_handler = handler
    await replacement._handle_line(frame(bodies[0]))  # Fresh timestamp, same old UUID: still deduplicated.
    assert not replacement._writer.lines
    busy_id, busy_body = request()
    await replacement._handle_line(frame(busy_body))
    assert replacement._writer.lines == [f"PRIVMSG #agents :!res {busy_id} busy 1/1 busy"]
    assert handler.await_count == 6 and replacement._coord_dispatch_task is None
    now[0] += 61
    await replacement._handle_line(frame(busy_body))  # The busy UUID itself cannot become queued work.
    fresh_id, fresh_body = request()
    await replacement._handle_line(frame(fresh_body))
    await asyncio.wait_for(replacement._coord_dispatch_task, 3)
    assert handler.await_count == 7
    assert replacement._writer.lines == [
        f"PRIVMSG #agents :!res {busy_id} busy 1/1 busy",
        f"PRIVMSG #agents :!res {fresh_id} ok 1/1 accepted"]


@pytest.mark.asyncio
@pytest.mark.parametrize("coordination", [True, False])
@pytest.mark.parametrize("runtime_failure", [True, False])
async def test_coordination_refuses_unsafe_or_unavailable_runtime(adapter, coordination, runtime_failure):
    from unittest.mock import MagicMock, Mock
    from gateway.run_turn import GatewayTurnMixin
    from gateway.run_turn_runner import TurnRunner
    from gateway.turn_context import TurnContext

    if not coordination:
        adapter = irc.IRCAdapter(PlatformConfig(extra={
            "server": "irc.example", "channel": "#ordinary", "nickname": "ordinary-bot"}))
        adapter.config.typing_indicator = False
        adapter._writer = Writer()
    execute = Mock(return_value={"final_response": "ordinary Codex answer", "messages": []})

    def make_agent(**kwargs):
        return SimpleNamespace(
            model=kwargs["model"], api_mode=kwargs["api_mode"], session_id=kwargs["session_id"],
            tools=[], context_compressor=SimpleNamespace(last_prompt_tokens=0, context_length=200_000),
            session_prompt_tokens=0, session_completion_tokens=0, run_conversation=execute)

    factory = Mock(side_effect=make_agent)
    runner = MagicMock()
    runner.config = SimpleNamespace(streaming=None)
    runner._provider_routing = {}
    runner._agent_cache_lock = runner._session_db = runner._prefill_messages = None
    runner._agent_cache = {}
    runner._pending_model_notes = {}
    runner._pending_skills_reload_notes = {}
    runner.session_store._entries = {}
    runner._get_system_prompt_for_channel.return_value = None
    runner._resolve_session_agent_runtime.side_effect = lambda **kwargs: (
        "synthetic-codex", {"api_mode": kwargs["user_config"]["model"]["openai_runtime"]})
    if runtime_failure:
        runner._resolve_session_agent_runtime.side_effect = RuntimeError("synthetic private credential failure")
    runner._resolve_session_reasoning_config.return_value = None
    runner._resolve_session_service_tier.return_value = None
    runner._resolve_turn_agent_config.side_effect = lambda *args: GatewayTurnMixin._resolve_turn_agent_config(runner, *args)
    runner._agent_config_signature.return_value = ("synthetic-signature",)
    runner._extract_cache_busting_config.return_value = {}
    runner._refresh_fallback_model.return_value = None
    runner._consume_pending_native_image_paths.return_value = []
    runner._consume_pending_turn_sidecar_notes.return_value = []
    runner._is_telegram_topic_lane.return_value = False
    runner._is_discord_auto_thread_lane.return_value = False
    runner._is_relay_discord_channel_lane.return_value = False
    runner._delivery_adapter_for.return_value = adapter
    runner._should_send_voice_reply.return_value = False
    runner._hmwa_stop_typing_for_turn = AsyncMock()
    runner._STATUS_HINTS = GatewayTurnMixin._STATUS_HINTS
    runner._hmwa_add_failed_turn_notice.side_effect = lambda *args: GatewayTurnMixin._hmwa_add_failed_turn_notice(runner, *args)
    results = []

    async def handler(event):
        key = adapter._event_session_key(event)
        ctx = TurnContext(
            source=event.source, message=event.text, history=[], session_id="synthetic", session_key=key,
            user_config={"model": {"openai_runtime": "codex_app_server"}},
            AIAgent=factory, enabled_toolsets=[] if coordination else ["terminal"],
            resolve_display_setting=lambda *args: False, _run_still_current=lambda: True,
            _hooks_ref=SimpleNamespace(loaded_hooks=False))
        result = await asyncio.to_thread(TurnRunner(runner, ctx).run_sync)
        results.append(result)
        return await GatewayTurnMixin._hmwa_deliver_turn_response(
            runner, event, event.source, SimpleNamespace(session_id="synthetic"), key, None,
            result, [], result["final_response"], "", False)

    adapter._message_handler = handler
    if coordination:
        request_id, body = request()
        await adapter._handle_line(frame(body))
        await asyncio.wait_for(adapter._coord_dispatch_task, 3)
        await adapter._handle_line(frame(body))
        assert adapter._writer.lines == [f"PRIVMSG #agents :!res {request_id} err 1/1 unavailable"]
        assert not adapter._active_requests and adapter._coord_dispatch_task is None
        assert results[0]["failed"] and results[0]["api_calls"] == 0 and results[0]["tools"] == []
        factory.assert_not_called()
        execute.assert_not_called()
    else:
        await adapter._handle_privmsg("amodig", "#ordinary", "ordinary-bot: hello")
        await asyncio.wait_for(next(iter(adapter._session_tasks.values())), 3)
        if runtime_failure:
            assert adapter._writer.lines == [
                f"PRIVMSG #ordinary :{line}" for line in adapter._split_message(results[0]["final_response"], "#ordinary")]
            factory.assert_not_called()
            execute.assert_not_called()
        else:
            assert adapter._writer.lines == ["PRIVMSG #ordinary :ordinary Codex answer"]
            factory.assert_called_once()
            assert factory.call_args.kwargs["api_mode"] == "codex_app_server"
            execute.assert_called_once()
    assert len(results) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["coord-online", "coord-offline", "coord-aged", "coord-no-uuid", "ordinary", "telegram"])
async def test_restart_discards_consultations_but_resumes_ordinary_sessions(adapter, monkeypatch, tmp_path, case):
    from unittest.mock import Mock
    from gateway.config import GatewayConfig, Platform
    from gateway.session import SessionStore
    from tests.gateway.restart_test_helpers import RestartTestAdapter, make_restart_runner

    coordination = case.startswith("coord-")
    if case == "ordinary":
        adapter = irc.IRCAdapter(PlatformConfig(extra={"server": "irc.example", "channel": "#ordinary"}))
        adapter.config.typing_indicator = False
        adapter._writer = Writer()
    elif case == "telegram":
        adapter = RestartTestAdapter()
    runner, _ = make_restart_runner(adapter)
    handler = AsyncMock(return_value="resumed ordinary session")
    adapter.set_message_handler(handler)
    runner.adapters = {} if case == "coord-offline" else {adapter.platform: adapter}
    runner._persist_active_agents = Mock()
    runner._resume_owner_authorized = Mock(return_value=True)
    breaker = Mock(return_value=False)
    monkeypatch.setattr("gateway.restart_loop_guard.check_and_record", breaker)
    store = SessionStore(sessions_dir=tmp_path / "sessions", config=GatewayConfig())
    runner.session_store = store
    source = adapter.build_source(
        chat_id="#agents" if coordination else "ordinary-chat", chat_type="group", user_id=PEER,
        thread_id=str(uuid.uuid4()) if coordination and case != "coord-no-uuid" else None)
    entry = store.get_or_create_session(source)
    store.mark_resume_pending(entry.session_key, reason="restart_timeout")
    if case == "coord-aged":
        store._update_entry(entry.session_key, lambda row: setattr(
            row, "last_resume_marked_at", datetime.datetime.now() - datetime.timedelta(days=1)))

    scheduled = runner._schedule_resume_pending_sessions()
    if runner._background_tasks:
        await asyncio.wait_for(asyncio.gather(*runner._background_tasks), 3)
    if coordination:
        assert scheduled == 0 and not runner._running_agents and not adapter._session_tasks
        handler.assert_not_awaited()
        runner._persist_active_agents.assert_not_called()
        breaker.assert_not_called()
        reloaded = SessionStore(sessions_dir=tmp_path / "sessions", config=GatewayConfig()).get_or_create_session(source)
        assert not reloaded.resume_pending and reloaded.resume_reason is None and reloaded.last_resume_marked_at is None
    else:
        assert scheduled == 1
        handler.assert_awaited_once()
        event = handler.await_args.args[0]
        assert event.internal and event.text == "" and event.source.platform == adapter.platform
        assert adapter.platform in (Platform("irc"), Platform.TELEGRAM)


@pytest.mark.asyncio
async def test_synthetic_or_replayed_events_cannot_bypass_live_admission(adapter):
    from dataclasses import replace
    from gateway.platforms.event import MessageEvent

    entered, release = asyncio.Event(), asyncio.Event()
    events = []

    async def handler(event):
        events.append(event)
        entered.set()
        await release.wait()
        return "one live answer"

    adapter.set_message_handler(handler)
    source = adapter.build_source(chat_id="#agents", chat_type="group", user_id=PEER, thread_id=str(uuid.uuid4()))
    restored = MessageEvent(text="", source=source, internal=True)
    await adapter.handle_message(restored)
    assert not restored._gateway_accepted and not adapter._session_tasks and not events
    request_id, body = request()
    await adapter._handle_line(frame(body))
    dispatch = adapter._coord_dispatch_task
    await asyncio.wait_for(entered.wait(), 3)
    replay = replace(events[0])
    await adapter.handle_message(replay)  # Even an exact event copy cannot borrow the admitted dispatch.
    assert not replay._gateway_accepted and len(events) == 1
    release.set()
    await asyncio.wait_for(dispatch, 3)
    await adapter.handle_message(replace(events[0]))  # Closed UUID cannot be revived.
    assert len(events) == 1 and not adapter._active_requests
    assert adapter._writer.lines == [f"PRIVMSG #agents :!res {request_id} ok 1/1 one live answer"]


@pytest.fixture
def reload_irc_adapter(monkeypatch, tmp_path):
    import sys
    from pathlib import Path
    from hermes_cli.plugins import PluginManager, PluginManifest
    from hermes_cli.plugins_loader import _evict_modules

    manager = PluginManager(scope_key=str(tmp_path))
    manifest = PluginManifest(name="irc", path=str(Path(irc.__file__).parent), source="bundled", kind="platform")
    namespace = "hermes_plugins.irc_coordination_reload_test"

    def load(config):
        # Production loader evicts the package AND adapter submodule before executing both again.
        package = manager._load_directory_module(manifest, module_name=namespace)
        module = sys.modules[package.__name__ + ".adapter"]
        monkeypatch.setattr(module, "_get_scoped_secret", lambda *args: "")
        instance = module.IRCAdapter(config)
        instance.config.typing_indicator = False
        instance._writer = Writer()
        instance._registered = True
        instance._mark_connected()
        return instance, module

    try:
        yield load
    finally:
        _evict_modules(namespace)


@pytest.mark.asyncio
async def test_plugin_reload_preserves_worker_lease_and_expired_response(adapter, reload_irc_adapter, monkeypatch):
    first, first_module = reload_irc_adapter(adapter.config)
    started, release = asyncio.Event(), asyncio.Event()
    events = []

    async def handler(event):
        events.append(event)
        started.set()
        await release.wait()
        return "late generation-A answer"

    first.set_message_handler(handler)
    monkeypatch.setattr(first_module, "REPLY_TIMEOUT", 0.02)
    request_id, body = request()
    await first._handle_line(frame(body))
    dispatch = first._coord_dispatch_task
    try:
        await asyncio.wait_for(started.wait(), 3)
        second, second_module = reload_irc_adapter(adapter.config)
        assert first_module is not second_module and type(first) is not type(second)
        assert first._coordination_state is second._coordination_state
        second.set_message_handler(AsyncMock(return_value="new generation answer"))
        open_connection = AsyncMock()
        monkeypatch.setattr(second_module, "_open_connection", open_connection)
        assert not await second.connect(is_reconnect=True)
        assert second.fatal_error_code == "coordination_busy" and second.fatal_error_retryable
        open_connection.assert_not_awaited()
        second._mark_connected()  # Even a pre-existing live transport cannot bypass the lease.
        busy_id, busy_body = request()
        await second._handle_line(frame(busy_body))
        await second._handle_line(frame(busy_body))
        second._message_handler.assert_not_awaited()
        assert second._writer.lines == [f"PRIVMSG #agents :!res {busy_id} busy 1/1 busy"]
        assert len(events) == 1 and not dispatch.done()

        other_account = "agent-omp-loota-01"
        other = type(second)(PlatformConfig(extra={
            **adapter.config.extra, "nickname": other_account, "sasl_account": other_account}))
        other.config.typing_indicator = False
        other._writer = Writer()
        other._registered = True
        other._mark_connected()
        other.set_message_handler(AsyncMock(return_value="independent answer"))
        other_id, other_body = request(target=other_account)
        await other._handle_line(frame(other_body))
        await asyncio.wait_for(other._coord_dispatch_task, 3)
        assert other._writer.lines == [f"PRIVMSG #agents :!res {other_id} ok 1/1 independent answer"]
        assert not dispatch.done()
        await asyncio.wait_for(first._active_sessions[first._event_session_key(events[0])].wait(), 3)
        assert first._writer.lines == [f"PRIVMSG #agents :!res {request_id} err 1/1 timeout"]
    finally:
        release.set()
        await asyncio.wait_for(dispatch, 3)
    assert first._writer.lines == [f"PRIVMSG #agents :!res {request_id} err 1/1 timeout"]
    assert second._coordination_state.dispatch is None
    fresh_id, fresh_body = request()
    await second._handle_line(frame(fresh_body))
    await asyncio.wait_for(second._coord_dispatch_task, 3)
    await first.send("#agents", "late old generation", metadata={"thread_id": request_id, "_final_response": True})
    second._message_handler.assert_awaited_once()
    assert second._writer.lines[-1] == f"PRIVMSG #agents :!res {fresh_id} ok 1/1 new generation answer"
    assert len(first._writer.lines) == 1


@pytest.mark.asyncio
async def test_plugin_reload_preserves_rate_window_and_replay_ids(adapter, reload_irc_adapter):
    first, first_module = reload_irc_adapter(adapter.config)
    first.set_message_handler(AsyncMock(return_value="accepted"))
    bodies = []
    for index in range(6):
        _, body = request()
        bodies.append(body)
        await first._handle_line(frame(body, tags=f";msgid=reload-{index}"))
        await asyncio.wait_for(first._coord_dispatch_task, 3)
    assert first._message_handler.await_count == 6
    await first.disconnect()
    second, second_module = reload_irc_adapter(adapter.config)
    assert first_module is not second_module
    second.set_message_handler(AsyncMock(return_value="must not run"))
    await second._handle_line(frame(bodies[0]))  # Same UUID, fresh server timestamp.
    await second._handle_line(frame(request()[1], tags=";msgid=reload-0"))  # New UUID, old msgid.
    assert second._writer.lines == []  # Replays are silent, not even busy/error responses.
    seventh_id, seventh_body = request()
    await second._handle_line(frame(seventh_body))
    await second._handle_line(frame(seventh_body))
    assert second._writer.lines == [f"PRIVMSG #agents :!res {seventh_id} busy 1/1 busy"]
    second._message_handler.assert_not_awaited()
    assert second._coord_dispatch_task is None

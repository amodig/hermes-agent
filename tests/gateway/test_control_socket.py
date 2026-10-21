"""Tests for the gateway control socket (#92091 migration step 1)."""

import asyncio
import json
import socket
import sys
from pathlib import Path

import pytest

from gateway.control_socket import (
    CONTROL_PROTOCOL_VERSION,
    GatewayControlServer,
    identify_gateway,
    query_gateway_control,
    resolve_client_socket_path,
    resolve_server_socket_path,
    windows_pipe_name,
)

pytestmark = pytest.mark.platforms("posix")  # Unix-socket transport; the named-pipe half is covered on the wine2e lane


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture()
def home(tmp_path: Path) -> Path:
    d = tmp_path / "home" / ".hermes"
    d.mkdir(parents=True)
    return d


def _serve(home: Path, handlers=None):
    """Context helper: start a server in a fresh loop, yield inside coro."""
    return GatewayControlServer(home, verb_handlers=handlers)


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def test_short_home_binds_in_home(tmp_path: Path):
    # A home short enough for sun_path binds in-home with no pointer.
    # tmp_path can exceed the limit on CI runners, so build one in the
    # system temp root directly.
    import tempfile

    try:
        short_root = Path(tempfile.mkdtemp(prefix="hgw-", dir="/tmp"))
    except OSError:
        pytest.skip("/tmp not writable on this host")
    try:
        short_home = short_root / ".hermes"
        short_home.mkdir()
        assert len(str(short_home / "gateway.sock").encode()) <= 100
        bind, pointer = resolve_server_socket_path(short_home)
        assert bind == short_home / "gateway.sock"
        assert pointer is None
    finally:
        import shutil

        shutil.rmtree(short_root, ignore_errors=True)


def test_long_home_uses_pointer_fallback(tmp_path: Path):
    deep = tmp_path / ("x" * 120) / ".hermes"
    deep.mkdir(parents=True)
    bind, pointer = resolve_server_socket_path(deep)
    assert bind != deep / "gateway.sock"
    assert len(str(bind).encode()) <= 100
    assert pointer == deep / "gateway.sock.path"


def test_client_resolution_prefers_direct_then_pointer(home: Path, tmp_path: Path):
    assert resolve_client_socket_path(home) is None
    # pointer file to an existing socket-ish file
    target = tmp_path / "elsewhere.sock"
    target.touch()
    (home / "gateway.sock.path").write_text(str(target))
    assert resolve_client_socket_path(home) == target
    # direct file wins over pointer
    direct = home / "gateway.sock"
    direct.touch()
    assert resolve_client_socket_path(home) == direct


def test_windows_pipe_name_is_stable_and_home_scoped(tmp_path: Path):
    a = windows_pipe_name(tmp_path / "a")
    b = windows_pipe_name(tmp_path / "b")
    assert a.startswith(r"\\.\pipe\hermes-gateway-")
    assert a != b
    assert a == windows_pipe_name(tmp_path / "a")


# ---------------------------------------------------------------------------
# Server lifecycle + verbs (real sockets, real event loop)
# ---------------------------------------------------------------------------

def test_server_answers_identify_and_status(home: Path):
    async def scenario():
        server = GatewayControlServer(
            home,
            verb_handlers={
                "identify": lambda: {"pid": 4242, "code_sha": "abc123", "protocol": 1},
                "status": lambda: {"gateway_state": "running"},
            },
        )
        assert await server.start()
        try:
            loop = asyncio.get_running_loop()
            ident = await loop.run_in_executor(
                None, lambda: query_gateway_control(home, "identify")
            )
            status = await loop.run_in_executor(
                None, lambda: query_gateway_control(home, "status")
            )
            return ident, status
        finally:
            await server.stop()

    ident, status = _run(scenario())
    assert ident == {"pid": 4242, "code_sha": "abc123", "protocol": 1}
    assert status == {"gateway_state": "running"}


def test_unknown_verb_and_malformed_request(home: Path):
    async def scenario():
        server = GatewayControlServer(
            home, verb_handlers={"identify": lambda: {"pid": 1}}
        )
        assert await server.start()
        try:
            loop = asyncio.get_running_loop()
            unknown = await loop.run_in_executor(
                None, lambda: query_gateway_control(home, "restart")
            )

            def raw_garbage():
                path = resolve_client_socket_path(home)
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                    s.settimeout(2)
                    s.connect(str(path))
                    s.sendall(b"this is not json\n")
                    return s.recv(65536)

            garbage_reply = await loop.run_in_executor(None, raw_garbage)
            return unknown, garbage_reply
        finally:
            await server.stop()

    unknown, garbage_reply = _run(scenario())
    # unknown verb → ok:false → client returns None (fallback signal)
    assert unknown is None
    payload = json.loads(garbage_reply.decode())
    assert payload["ok"] is False
    assert payload["protocol"] == CONTROL_PROTOCOL_VERSION


def test_verb_handler_receives_params(home: Path):
    """A handler declaring a ``params`` argument is called with the request's params dict; a bare
    handler is still called with no args (backward compat for identify/status/rescan)."""
    received = {}

    def with_params(params):
        received.update(params)
        return {"echo": params}

    def bare():
        return {"ok": 1}

    async def scenario():
        server = GatewayControlServer(
            home, verb_handlers={"with-params": with_params, "bare": bare})
        assert await server.start()
        try:
            loop = asyncio.get_running_loop()
            got = await loop.run_in_executor(
                None, lambda: query_gateway_control(
                    home, "with-params", params={"old": "a", "new": "b"}))
            bare_ok = await loop.run_in_executor(
                None, lambda: query_gateway_control(home, "bare"))
            return got, bare_ok
        finally:
            await server.stop()

    got, bare_ok = _run(scenario())
    assert got == {"echo": {"old": "a", "new": "b"}}
    assert received == {"old": "a", "new": "b"}
    assert bare_ok == {"ok": 1}


def test_stop_removes_socket_and_pointer(home: Path):
    async def scenario():
        server = GatewayControlServer(
            home, verb_handlers={"identify": lambda: {"pid": 1}}
        )
        assert await server.start()
        bind, _ = resolve_server_socket_path(home)
        assert bind.exists()
        await server.stop()
        return bind

    bind = _run(scenario())
    assert not bind.exists()
    assert resolve_client_socket_path(home) is None
    # queries after stop cleanly return None
    assert query_gateway_control(home, "identify") is None


def test_stale_socket_file_is_replaced_on_bind(home: Path):
    # Plant the stale file at wherever the server will actually bind
    # (in-home OR the temp-dir fallback, depending on path length).
    bind, _ = resolve_server_socket_path(home)
    bind.parent.mkdir(parents=True, exist_ok=True)
    bind.touch()  # crashed predecessor's leftover

    async def scenario():
        server = GatewayControlServer(
            home, verb_handlers={"identify": lambda: {"pid": 7}}
        )
        assert await server.start()
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, lambda: identify_gateway(home))
        finally:
            await server.stop()

    assert _run(scenario()) == {"pid": 7}


def test_long_home_end_to_end_via_pointer(tmp_path: Path):
    deep = tmp_path / ("p" * 120) / ".hermes"
    deep.mkdir(parents=True)

    async def scenario():
        server = GatewayControlServer(
            deep, verb_handlers={"identify": lambda: {"pid": 9}}
        )
        assert await server.start()
        try:
            assert (deep / "gateway.sock.path").is_file()
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, lambda: identify_gateway(deep))
        finally:
            await server.stop()

    assert _run(scenario()) == {"pid": 9}
    assert not (deep / "gateway.sock.path").exists()


def test_no_socket_returns_none_fast(home: Path):
    assert identify_gateway(home) is None
    assert query_gateway_control(home, "status") is None


def test_default_identify_payload_shape(home: Path, monkeypatch):
    """The real identify handler carries the fleet-consumer contract fields."""
    monkeypatch.setenv("HERMES_HOME", str(home))

    async def scenario():
        server = GatewayControlServer(home)  # default handlers
        assert await server.start()
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, lambda: identify_gateway(home))
        finally:
            await server.stop()

    ident = _run(scenario())
    assert ident is not None
    assert ident["protocol"] == CONTROL_PROTOCOL_VERSION
    assert ident["pid"] == __import__("os").getpid()
    # contract keys exist even when values are None/absent-degradable
    for key in ("hermes_home", "supervisor", "kind", "start_time"):
        assert key in ident
    assert ident["supervisor"] in {
        "systemd",
        "launchd",
        "desktop",
        "external",
        "manual",
    }


# ---------------------------------------------------------------------------
# Consumer integration: fleet matrix + inventory prefer socket, fall back
# ---------------------------------------------------------------------------

def _fake_identity(pid: int, sha: str):
    return {
        "protocol": 1,
        "pid": pid,
        "code_sha": sha,
        "code_version": "9.9.9",
        "supervisor": "systemd",
        "kind": "hermes-gateway",
    }


def test_collect_fleet_versions_prefers_socket(tmp_path: Path, monkeypatch):
    import hermes_cli.update_receipt as ur

    home = tmp_path / ".hermes"
    home.mkdir()

    monkeypatch.setattr(
        "hermes_cli.version_info.get_code_identity",
        lambda refresh=False: {"sha": "HEADSHA", "version": "1.0"},
    )
    monkeypatch.setattr(
        "hermes_cli.profiles._get_default_hermes_home", lambda: home
    )
    monkeypatch.setattr(
        "hermes_cli.profiles._get_profiles_root", lambda: tmp_path / "no-profiles"
    )
    # stale state file that would report a WRONG pid — socket must win
    (home / "gateway_state.json").write_text(
        json.dumps({"pid": 1, "code_sha": "stalefile", "kind": "hermes-gateway"})
    )
    monkeypatch.setattr(
        "gateway.control_socket.identify_gateway",
        lambda h, **kw: _fake_identity(31337, "HEADSHA"),
    )

    fleet = ur.collect_fleet_versions()
    assert len(fleet) == 1
    entry = fleet[0]
    assert entry["pid"] == 31337
    assert entry["state"] == "current"
    assert entry["source"] == "socket"


def test_collect_fleet_versions_falls_back_to_state_file(tmp_path: Path, monkeypatch):
    """Without a socket answer the state file is a fallback claim, not an identity.

    A live PID that is not the home's verified gateway (here: this pytest
    process wrote the record) stays visible as ``unknown`` with no
    self-reported sha; only the verified gateway PID is classified
    ``current``/``stale`` from the file (#110420).
    """
    import os

    import hermes_cli.update_receipt as ur

    home = tmp_path / ".hermes"
    home.mkdir()

    monkeypatch.setattr(
        "hermes_cli.version_info.get_code_identity",
        lambda refresh=False: {"sha": "HEADSHA", "version": "1.0"},
    )
    monkeypatch.setattr(
        "hermes_cli.profiles._get_default_hermes_home", lambda: home
    )
    monkeypatch.setattr(
        "hermes_cli.profiles._get_profiles_root", lambda: tmp_path / "no-profiles"
    )
    monkeypatch.setattr(
        "gateway.control_socket.identify_gateway", lambda h, **kw: None
    )
    (home / "gateway_state.json").write_text(
        json.dumps(
            {
                "pid": os.getpid(),  # a live pid so _pid_exists passes
                "code_sha": "OLDSHA",
                "kind": "hermes-gateway",
            }
        )
    )

    fleet = ur.collect_fleet_versions()
    assert len(fleet) == 1
    assert fleet[0]["pid"] == os.getpid()
    assert fleet[0]["state"] == "unknown"
    assert fleet[0]["code_sha"] is None
    assert "source" not in fleet[0]

    # Same file, but the profile's identity resolver verifies this PID as the
    # gateway: the fallback may now classify from the stamped sha.
    monkeypatch.setattr(
        "gateway.status.live_gateway_pid_for_home", lambda h: os.getpid()
    )
    fleet = ur.collect_fleet_versions()
    assert len(fleet) == 1
    assert fleet[0]["state"] == "stale"
    assert fleet[0]["code_sha"] == "OLDSHA"
    assert "source" not in fleet[0]


def test_runtime_inventory_dedupes_same_pid_across_homes(tmp_path: Path, monkeypatch):
    """One multiplex gateway answering identify for two profile homes must
    yield exactly ONE runtime record (reviewer point on #92447)."""
    import hermes_cli.update_inventory as ui

    home = tmp_path / ".hermes"
    home.mkdir()
    profiles_root = tmp_path / "profiles"
    (profiles_root / "coder").mkdir(parents=True)

    monkeypatch.setattr(
        "hermes_cli.profiles._get_default_hermes_home", lambda: home
    )
    monkeypatch.setattr(
        "hermes_cli.profiles._get_profiles_root", lambda: profiles_root
    )
    monkeypatch.setattr(
        "hermes_cli.gateway._get_service_pids", lambda all_profiles=False: set()
    )
    monkeypatch.setattr(
        "hermes_cli.gateway.find_profile_gateway_processes", lambda: []
    )
    monkeypatch.setattr(
        "gateway.control_socket.identify_gateway",
        lambda h, **kw: _fake_identity(777, "SHA777"),
    )

    plan = ui.collect_runtime_inventory()
    gws = [r for r in plan.runtimes if r.kind == "gateway"]
    assert len(gws) == 1, [r.__dict__ for r in gws]
    assert gws[0].pid == 777


def test_runtime_inventory_prefers_socket_supervisor(tmp_path: Path, monkeypatch):
    import hermes_cli.update_inventory as ui

    home = tmp_path / ".hermes"
    home.mkdir()

    monkeypatch.setattr(
        "hermes_cli.profiles._get_default_hermes_home", lambda: home
    )
    monkeypatch.setattr(
        "hermes_cli.profiles._get_profiles_root", lambda: tmp_path / "no-profiles"
    )
    monkeypatch.setattr(
        "hermes_cli.gateway._get_service_pids", lambda all_profiles=False: set()
    )
    monkeypatch.setattr(
        "hermes_cli.gateway.find_profile_gateway_processes", lambda: []
    )
    monkeypatch.setattr(
        "gateway.control_socket.identify_gateway",
        lambda h, **kw: _fake_identity(555, "SHA555"),
    )

    plan = ui.collect_runtime_inventory()
    gws = [r for r in plan.runtimes if r.kind == "gateway"]
    assert len(gws) == 1
    assert gws[0].pid == 555
    # supervisor comes from the gateway's own declaration, not a PID scan
    assert gws[0].supervisor == "systemd"
    assert gws[0].code_sha == "SHA555"


def test_windows_pipe_query_is_bounded_when_the_peer_never_answers(home: Path, monkeypatch):
    """#132547: a pipe handle read cannot time out on its own; the query must still return None at its bound."""
    import threading
    import time

    from gateway import control_socket

    released = threading.Event()

    class _SilentPipe:  # accepts the request, never answers
        def write(self, _data):
            return None

        def read(self, _n):
            released.wait(10)
            return b""

        def close(self):
            return None

    monkeypatch.setattr(control_socket, "open", lambda *_a, **_k: _SilentPipe(), raising=False)
    start = time.monotonic()
    try:
        assert control_socket._query_windows_pipe(home, b"{}\n", 0.3) is None
        assert time.monotonic() - start < 2.0
    finally:
        released.set()


def test_irc_request_native_socket_profile_scope_and_disconnect(home, monkeypatch):
    from contextlib import asynccontextmanager
    from types import SimpleNamespace
    from gateway.irc_coordination import irc_request_verb
    from hermes_constants import get_hermes_home, set_hermes_home_override, reset_hermes_home_override
    import gateway.run

    @asynccontextmanager
    async def scope(profile_home):
        token = set_hermes_home_override(str(profile_home))
        try:
            yield
        finally:
            reset_hermes_home_override(token)

    monkeypatch.setattr(gateway.run, "_async_profile_runtime_scope", scope)

    async def scenario():
        calls = []
        started, unwound = asyncio.Event(), asyncio.Event()

        async def request_peer(target, question, *, request_id):
            calls.append((get_hermes_home(), target))
            if question == "wait":
                started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    unwound.set()
            return {"requestId": request_id, "account": target, "status": "ok", "text": "untrusted peer data"}

        adapter = SimpleNamespace(coordination=True, is_connected=True, request_peer=request_peer)
        homes = {name: home / name for name in ("alpha", "beta")}

        def authorization_adapter(platform, profile):
            assert platform.value == "irc"
            assert get_hermes_home() == homes[profile]
            return adapter

        runner = SimpleNamespace(
            served_profile_names=lambda: list(homes), _served_profile_homes=homes,
            _primary_profile_name="alpha", _authorization_adapter=authorization_adapter,
            _async_scope_or_null=lambda factory, path: factory(path))
        handler = irc_request_verb(runner, asyncio.get_running_loop())
        server = GatewayControlServer(home, verb_handlers={"irc-request": handler})
        assert await server.start()
        try:
            for profile in ("alpha", "beta", "alpha"):
                result = await asyncio.to_thread(
                    query_gateway_control, home, "irc-request",
                    params={"profile": profile, "target": "agent-jeeves", "question": "hello"}, timeout=3)
                assert result["status"] == "ok" and result["account"] == "agent-jeeves"
            assert [call[0] for call in calls] == [homes["alpha"], homes["beta"], homes["alpha"]]
            for params, error in [
                ({"profile": "other", "target": "agent-jeeves", "question": "hello"}, "unavailable"),
                ({"profile": "alpha", "target": "agent-*", "question": "hello"}, "invalid-request"),
                ({"profile": "alpha", "target": "agent-jeeves", "question": "x" * 1801}, "question-too-large"),
            ]:
                result = await asyncio.to_thread(query_gateway_control, home, "irc-request", params=params, timeout=3)
                assert result["error"] == error
            reader, writer = await asyncio.open_unix_connection(str(resolve_client_socket_path(home)))
            writer.write(json.dumps({"protocol": 1, "verb": "irc-request", "params": {
                "profile": "alpha", "target": "agent-jeeves", "question": "wait"}}).encode() + b"\n")
            await writer.drain()
            await asyncio.wait_for(started.wait(), 3)
            writer.close()
            await writer.wait_closed()
            await asyncio.wait_for(unwound.wait(), 3)
        finally:
            await server.stop()

    _run(scenario())


def test_legacy_verb_answers_half_closed_client(home):
    async def scenario():
        server = GatewayControlServer(home, verb_handlers={"legacy": lambda: {"value": 1}})
        assert await server.start()
        try:
            reader, writer = await asyncio.open_unix_connection(str(resolve_client_socket_path(home)))
            writer.write(b'{"verb":"legacy","protocol":1}\n')
            await writer.drain()
            writer.write_eof()
            response = json.loads(await asyncio.wait_for(reader.readline(), 3))
            assert response["result"] == {"value": 1}
            writer.close()
            await writer.wait_closed()
        finally:
            await server.stop()
    _run(scenario())


def test_pipe_handler_runs_off_loop_and_processes_one_request(home):
    import threading
    from types import SimpleNamespace
    from gateway.control_socket import _PipeControlProtocol

    async def scenario():
        loop = asyncio.get_running_loop()
        main_thread = threading.get_ident()
        started, release, closed = asyncio.Event(), asyncio.Event(), asyncio.Event()
        calls, responses = [], []

        async def on_gateway_loop(value):
            started.set()
            await release.wait()
            return {"value": value}

        def handler(params, *, cancelled):
            calls.append(threading.get_ident())
            future = asyncio.run_coroutine_threadsafe(on_gateway_loop(params["value"]), loop)
            try:
                return future.result(timeout=1)
            finally:
                if not future.done():
                    future.cancel()

        server = GatewayControlServer(home, verb_handlers={"blocking": handler})
        protocol = _PipeControlProtocol(server)
        protocol.connection_made(SimpleNamespace(write=responses.append, close=closed.set))
        try:
            protocol.data_received(b'{"verb":"blocking","params":')
            protocol.data_received(b'{"value":42}}\n')
            protocol.data_received(b'{"verb":"blocking","params":{"value":99}}\n')
            await asyncio.wait_for(started.wait(), 3)
            assert calls and calls[0] != main_thread
            assert not responses and not closed.is_set()
        finally:
            release.set()
            await asyncio.wait_for(protocol._response_task, 3)
        assert len(calls) == 1 and closed.is_set()
        assert len(responses) == 1
        assert json.loads(responses[0])["result"] == {"value": 42}

    _run(scenario())


def test_pipe_disconnect_cancels_opted_in_handler_without_late_write(home):
    from types import SimpleNamespace
    from gateway.control_socket import _PipeControlProtocol

    async def scenario():
        loop = asyncio.get_running_loop()
        started = asyncio.Event()
        responses = []

        def handler(*, cancelled):
            loop.call_soon_threadsafe(started.set)
            assert cancelled.wait(3)
            return {"error": "cancelled"}

        server = GatewayControlServer(home, verb_handlers={"blocking": handler})
        protocol = _PipeControlProtocol(server)
        protocol.connection_made(SimpleNamespace(write=responses.append, close=lambda: None))
        protocol.data_received(b'{"verb":"blocking"}\n')
        await asyncio.wait_for(started.wait(), 3)
        protocol.connection_lost(None)
        await asyncio.wait_for(protocol._response_task, 3)
        assert not responses

    _run(scenario())


@pytest.mark.parametrize("exit_mode", ["success", "error", "timeout", "disconnect", "cancel", "shutdown"])
def test_concurrent_irc_waits_leave_default_executor_and_other_verbs_live(home, monkeypatch, exit_mode):
    import concurrent.futures
    import threading
    from contextlib import asynccontextmanager
    from types import SimpleNamespace
    import gateway.control_socket as control
    import gateway.irc_coordination as coordination
    import gateway.run

    assert coordination.CONTROL_TIMEOUT == 65
    if exit_mode == "timeout":
        monkeypatch.setattr(coordination, "CONTROL_TIMEOUT", 5.0)

    async def scenario():
        loop = asyncio.get_running_loop()
        default_workers = 2
        loop.set_default_executor(concurrent.futures.ThreadPoolExecutor(
            max_workers=default_workers, thread_name_prefix="shared-default"))
        private_workers = control._IRC_WAIT_WORKERS
        count = max(default_workers, private_workers) * 2
        release, started = asyncio.Event(), asyncio.Event()
        scope_threads, wait_threads, submitted, handlers, clients = [], [], [], [], []
        request_tasks = []
        active = set()
        fail = exit_mode == "error"

        @asynccontextmanager
        async def scope(profile_home):
            # The real scope initializer also needs to_thread: this is the deadlock edge.
            await asyncio.to_thread(lambda: scope_threads.append(threading.current_thread().name))
            yield

        monkeypatch.setattr(gateway.run, "_async_profile_runtime_scope", scope)

        async def request_peer(target, question, *, request_id):
            request_tasks.append(asyncio.current_task())
            active.add(request_id)
            if len(active) == private_workers:
                started.set()
            try:
                await release.wait()
                if fail:
                    raise RuntimeError("synthetic provider failure")
                return {"requestId": request_id, "account": target, "status": "ok", "text": "synthetic reply"}
            finally:
                active.discard(request_id)

        adapter = SimpleNamespace(coordination=True, is_connected=True, request_peer=request_peer)
        homes = {f"profile-{index}": home / f"profile-{index}" for index in range(count)}
        runner = SimpleNamespace(
            served_profile_names=lambda: list(homes), _served_profile_homes=homes,
            _primary_profile_name="profile-0", _authorization_adapter=lambda *args: adapter,
            _async_scope_or_null=lambda factory, path: factory(path))
        verb = coordination.irc_request_verb(runner, loop)

        def handler(params, *, cancelled):
            wait_threads.append(threading.current_thread().name)
            return verb(params, cancelled=cancelled)

        server = GatewayControlServer(home, verb_handlers={
            "irc-request": handler, "status": lambda: {"alive": True}})
        # Record real concurrent futures, so cancellation assertions wait for actual thread exit.
        pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=private_workers, thread_name_prefix="hermes-irc-control")
        submit = pool.submit

        def record_submit(*args, **kwargs):
            future = submit(*args, **kwargs)
            submitted.append(future)
            return future

        monkeypatch.setattr(pool, "submit", record_submit)
        server._irc_executor = pool
        run_request = server._run_request_line

        async def tracked(raw, *, cancelled):
            if json.loads(raw).get("verb") == "irc-request":
                handlers.append(asyncio.current_task())
            return await run_request(raw, cancelled=cancelled)

        monkeypatch.setattr(server, "_run_request_line", tracked)
        assert await server.start()

        async def send(verb, index=0):
            reader, writer = await asyncio.open_unix_connection(str(resolve_client_socket_path(home)))
            clients.append(writer)
            writer.write(json.dumps({"protocol": 1, "id": index, "verb": verb, "params": {
                "profile": f"profile-{index}", "target": "agent-jeeves", "question": "hold"}}).encode() + b"\n")
            await writer.drain()
            return asyncio.create_task(reader.readline())

        try:
            replies = await asyncio.gather(*(send("irc-request", index) for index in range(count)))
            await asyncio.wait_for(started.wait(), 3)
            assert len(scope_threads) == private_workers
            assert all(name.startswith("shared-default") for name in scope_threads)
            assert all(name.startswith("hermes-irc-control") for name in wait_threads)
            assert await asyncio.wait_for(asyncio.to_thread(lambda: "default pool live"), 1) == "default pool live"
            status = await send("status")
            assert json.loads(await asyncio.wait_for(status, 1))["result"] == {"alive": True}
            assert len(active) == private_workers  # Other verbs replied while all private waits were outstanding.
            early = asyncio.as_completed(replies)
            for _ in range(count - private_workers):
                assert json.loads(await asyncio.wait_for(next(early), 1))["error"] == "busy"
            if exit_mode == "disconnect":
                for writer in clients:
                    writer.close()
            elif exit_mode == "cancel":
                for task in handlers:
                    if not task.done():
                        task.cancel()
            elif exit_mode == "shutdown":
                await asyncio.wait_for(server.stop(), 3)
            elif exit_mode != "timeout":
                release.set()
            raw = await asyncio.wait_for(asyncio.gather(*replies), 7)
            responses = [json.loads(line) if line else None for line in raw]
            assert sum(reply is not None and reply.get("error") == "busy" for reply in responses) == count - private_workers
            for reply in responses:
                if reply is None:
                    assert exit_mode in {"disconnect", "cancel"}
                elif reply.get("ok"):
                    result = reply["result"]
                    if exit_mode == "success":
                        assert result["status"] == "ok"
                    else:
                        assert result["error"] in {"cancelled", "timeout", "unavailable"}
                else:
                    assert reply["error"] in {"busy", "timeout"}
            await asyncio.wait_for(asyncio.gather(
                *(asyncio.wrap_future(future) for future in submitted), return_exceptions=True), 3)
            # Cancelling a concurrent future schedules loop cancellation; thread exit is not
            # acknowledgement that the coroutine's finally has run. Await that separately.
            _, pending = await asyncio.wait(request_tasks, timeout=3)
            assert not pending
            assert not active and all(future.done() for future in submitted)
            if exit_mode == "shutdown":
                assert await server.start()
            # All four permits—not just one—must be usable after every exit path.
            started.clear()
            release.clear()
            fail = False
            fresh = await asyncio.gather(*(send("irc-request", index) for index in range(private_workers)))
            await asyncio.wait_for(started.wait(), 3)
            release.set()
            results = await asyncio.wait_for(asyncio.gather(*fresh), 3)
            assert all(json.loads(line)["result"]["status"] == "ok" for line in results)
        finally:
            release.set()
            for writer in clients:
                writer.close()
            await server.stop()

    _run(scenario())


def test_irc_private_wait_pipe_returns_success(home):
    from types import SimpleNamespace
    from gateway.control_socket import _PipeControlProtocol

    async def scenario():
        replies = []
        server = GatewayControlServer(home, verb_handlers={
            "irc-request": lambda params, cancelled: {"status": "ok"}})
        protocol = _PipeControlProtocol(server)
        protocol.connection_made(SimpleNamespace(write=replies.append, close=lambda: None))
        try:
            protocol.data_received(b'{"verb":"irc-request","params":{}}\n')
            await asyncio.wait_for(protocol._response_task, 3)
            assert json.loads(replies[0])["result"] == {"status": "ok"}
        finally:
            await server.stop()

    _run(scenario())

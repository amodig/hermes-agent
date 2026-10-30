"""Tests for the IRC platform adapter plugin."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from tests.gateway._plugin_adapter_loader import load_plugin_adapter

# Load plugins/platforms/irc/adapter.py under a unique module name
# (plugin_adapter_irc) so it cannot collide with other plugin adapters
# loaded by sibling tests in the same process.
_irc_mod = load_plugin_adapter("irc")

_parse_irc_message = _irc_mod._parse_irc_message
_extract_nick = _irc_mod._extract_nick
IRCAdapter = _irc_mod.IRCAdapter
check_requirements = _irc_mod.check_requirements
validate_config = _irc_mod.validate_config
register = _irc_mod.register
_standalone_send = _irc_mod._standalone_send
is_connected = _irc_mod.is_connected
_env_enablement = _irc_mod._env_enablement


class TestIRCProtocolHelpers:

    def test_parse_simple_command(self):
        msg = _parse_irc_message("PING :server.example.com")
        assert msg["command"] == "PING"
        assert msg["params"] == ["server.example.com"]
        assert msg["prefix"] == ""


    def test_extract_nick_full_prefix(self):
        assert _extract_nick("nick!user@host") == "nick"


# ── IRC Adapter ──────────────────────────────────────────────────────────




class TestIRCAdapterLockConflict:

    @pytest.mark.asyncio
    async def test_connect_fails_when_identity_lock_held(self, monkeypatch):
        """``acquire_scoped_lock`` returns ``(acquired, existing)``; a live foreign holder must stop
        connect() before any socket is opened (the tuple is truthy, so a bare ``if not`` never fired)."""
        import gateway.status as gateway_status
        from gateway.config import PlatformConfig

        monkeypatch.setattr(
            gateway_status, "acquire_scoped_lock",
            lambda scope, identity, metadata=None: (False, {"pid": 4242, "profile": "other"}))
        opened = []

        async def _no_connect(*a, **k):
            opened.append(a)
            raise AssertionError("socket must not be opened on a lock conflict")
        monkeypatch.setattr(asyncio, "open_connection", _no_connect)
        adapter = IRCAdapter(PlatformConfig(enabled=True, extra={
            "server": "irc.example", "nickname": "hermes", "channel": "#x"}))
        assert await adapter.connect() is False
        assert adapter._fatal_error_code == "irc_lock"
        assert "other" in adapter._fatal_error_message
        assert opened == []


class TestIRCAdapterSend:

    @pytest.fixture
    def adapter(self, monkeypatch):
        for key in ("IRC_SERVER", "IRC_PORT", "IRC_NICKNAME", "IRC_CHANNEL", "IRC_USE_TLS"):
            monkeypatch.delenv(key, raising=False)
        from gateway.config import PlatformConfig
        cfg = PlatformConfig(
            enabled=True,
            extra={
                "server": "localhost",
                "port": 6667,
                "nickname": "testbot",
                "channel": "#test",
                "use_tls": False,
            },
        )
        return IRCAdapter(cfg)


    @pytest.mark.asyncio
    async def test_send_success(self, adapter):
        writer = MagicMock()
        writer.is_closing = MagicMock(return_value=False)
        writer.write = MagicMock()
        writer.drain = AsyncMock()
        adapter._writer = writer

        result = await adapter.send("#test", "hello world")
        assert result.success is True
        assert result.message_id is not None
        # Verify PRIVMSG was sent
        writer.write.assert_called()
        sent_data = writer.write.call_args[0][0]
        assert b"PRIVMSG #test :hello world" in sent_data


class TestIRCAdapterMessageParsing:

    @pytest.fixture
    def adapter(self, monkeypatch):
        for key in ("IRC_SERVER", "IRC_PORT", "IRC_NICKNAME", "IRC_CHANNEL", "IRC_USE_TLS"):
            monkeypatch.delenv(key, raising=False)
        from gateway.config import PlatformConfig
        cfg = PlatformConfig(
            enabled=True,
            extra={
                "server": "localhost",
                "port": 6667,
                "nickname": "hermes",
                "channel": "#test",
                "use_tls": False,
            },
        )
        a = IRCAdapter(cfg)
        a._current_nick = "hermes"
        a._registered = True
        return a


    @pytest.mark.asyncio
    async def test_handle_addressed_channel_message(self, adapter):
        """Messages addressed to the bot (nick: msg) should be dispatched."""
        handler = AsyncMock(return_value="response")
        adapter._message_handler = handler

        # Mock handle_message to capture the event
        dispatched = []
        original_dispatch = adapter._dispatch_message

        async def capture_dispatch(**kwargs):
            dispatched.append(kwargs)

        adapter._dispatch_message = capture_dispatch

        await adapter._handle_line(":user!u@host PRIVMSG #test :hermes: hello there")
        assert len(dispatched) == 1
        assert dispatched[0]["text"] == "hello there"
        assert dispatched[0]["chat_id"] == "#test"

    @pytest.mark.asyncio
    async def test_ignores_unaddressed_channel_message(self, adapter):
        dispatched = []

        async def capture_dispatch(**kwargs):
            dispatched.append(kwargs)

        adapter._dispatch_message = capture_dispatch
        adapter._message_handler = AsyncMock()

        await adapter._handle_line(":user!u@host PRIVMSG #test :just talking")
        assert len(dispatched) == 0


    @pytest.mark.asyncio
    async def test_ctcp_action_converted(self, adapter):
        """CTCP ACTION (/me) should be converted to text."""
        dispatched = []

        async def capture_dispatch(**kwargs):
            dispatched.append(kwargs)

        adapter._dispatch_message = capture_dispatch
        adapter._message_handler = AsyncMock()

        await adapter._handle_line(":user!u@host PRIVMSG hermes :\x01ACTION waves\x01")
        assert len(dispatched) == 1
        assert dispatched[0]["text"] == "* user waves"


    @pytest.mark.asyncio
    async def test_unauthorized_user_blocked(self, monkeypatch):
        """Nicks not in allowlist should be ignored."""
        for key in ("IRC_SERVER", "IRC_PORT", "IRC_NICKNAME", "IRC_CHANNEL", "IRC_USE_TLS"):
            monkeypatch.delenv(key, raising=False)
        from gateway.config import PlatformConfig
        cfg = PlatformConfig(
            enabled=True,
            extra={
                "server": "localhost",
                "port": 6667,
                "nickname": "hermes",
                "channel": "#test",
                "use_tls": False,
                "allowed_users": ["Admin", "BOB"],
            },
        )
        adapter = IRCAdapter(cfg)
        adapter._current_nick = "hermes"
        adapter._registered = True
        dispatched = []

        async def capture_dispatch(**kwargs):
            dispatched.append(kwargs)

        adapter._dispatch_message = capture_dispatch
        adapter._message_handler = AsyncMock()

        await adapter._handle_line(":eve!u@host PRIVMSG #test :hermes: hello")
        assert len(dispatched) == 0


class TestIRCAdapterSplitting:

    def test_split_respects_byte_limit(self):
        """Multi-byte characters should not exceed IRC byte limit."""
        # 100 japanese chars = 300 bytes in utf-8
        text = "あ" * 100
        from gateway.config import PlatformConfig
        cfg = PlatformConfig(enabled=True, extra={"server": "x", "channel": "#x"})
        adapter = IRCAdapter(cfg)
        adapter._current_nick = "bot"
        lines = adapter._split_message(text, "#test")
        for line in lines:
            overhead = len(f"PRIVMSG #test :{line}\r\n".encode("utf-8"))
            assert overhead <= 512, f"line over 512 bytes: {overhead}"


class TestIRCProtocolHelpersExtra:

    def test_parse_malformed_no_space(self):
        """A line starting with : but no space should not crash."""
        msg = _parse_irc_message(":justaprefix")
        assert msg["prefix"] == "justaprefix"
        assert msg["command"] == ""
        assert msg["params"] == []


class TestIRCAdapterMarkdown:


    def test_strip_link(self):
        result = IRCAdapter._strip_markdown("[click here](https://example.com)")
        assert result == "click here (https://example.com)"

    def test_strip_image(self):
        result = IRCAdapter._strip_markdown("![alt](https://example.com/img.png)")
        assert result == "https://example.com/img.png"


# ── Requirements / validation ────────────────────────────────────────────


class TestIRCRequirements:

    def test_check_requirements_with_env(self, monkeypatch):
        monkeypatch.setenv("IRC_SERVER", "irc.test.net")
        monkeypatch.setenv("IRC_CHANNEL", "#test")
        load = MagicMock(side_effect=RuntimeError("config unavailable"))
        monkeypatch.setattr("hermes_cli.config.load_config_readonly", load)
        assert check_requirements() is True
        load.assert_not_called()


    @pytest.mark.parametrize("shape", ["extra", "flat"])
    @pytest.mark.parametrize("case,expected", [
        ("enabled", True), ("disabled", False), ("missing-enabled", False),
        ("missing-server", False), ("missing-channel", False),
    ])
    def test_config_requirements_gate_actual_adapter_creation(self, monkeypatch, tmp_path, shape, case, expected):
        import json
        from gateway.config import PlatformConfig
        from gateway.platform_registry import PlatformEntry, PlatformRegistry
        from hermes_constants import set_hermes_home_override, reset_hermes_home_override

        monkeypatch.setattr(_irc_mod, "_get_scoped_secret", lambda *args: "")
        settings = dict(_coordination_config().extra)
        if case.startswith("missing-") and case != "missing-enabled":
            settings.pop(case.removeprefix("missing-"))
        block = {"enabled": case != "disabled", **({"extra": settings} if shape == "extra" else settings)}
        if case == "missing-enabled":
            block.pop("enabled")
        # A top-level sibling must not hide the documented nested IRC block.
        config = {"gateway": {"platforms": {"irc": block}}, "platforms": {"telegram": {"enabled": False}}}
        (tmp_path / "config.yaml").write_text(json.dumps(config))
        registry = PlatformRegistry()
        registry.register(PlatformEntry(
            name="irc", label="IRC", adapter_factory=IRCAdapter,
            check_fn=check_requirements, validate_config=validate_config))
        token = set_hermes_home_override(str(tmp_path))
        try:
            assert check_requirements() is expected
            created = registry.create_adapter("irc", PlatformConfig.from_dict(block))
            assert isinstance(created, IRCAdapter) is expected
            if expected:
                assert created.server == settings["server"] and created.channel == settings["channel"]
                assert created.coordination is True
        finally:
            reset_hermes_home_override(token)

    @pytest.mark.parametrize("failure", ["load", "import"])
    def test_check_requirements_config_errors_fail_closed(self, monkeypatch, failure):
        import sys

        monkeypatch.setattr(_irc_mod, "_get_scoped_secret", lambda *args: "")
        if failure == "import":
            monkeypatch.setitem(sys.modules, "gateway.config_loader", None)
        else:
            monkeypatch.setattr("hermes_cli.config.load_config_readonly",
                                MagicMock(side_effect=RuntimeError("synthetic load failure")))
        assert check_requirements() is False

    def test_check_requirements_follows_effective_profile_config(self, monkeypatch, tmp_path):
        import json
        from hermes_constants import set_hermes_home_override, reset_hermes_home_override

        monkeypatch.setattr(_irc_mod, "_get_scoped_secret", lambda *args: "")
        homes = [tmp_path / name for name in ("alpha", "beta")]
        for index, home in enumerate(homes):
            home.mkdir()
            (home / "config.yaml").write_text(json.dumps({"platforms": {"irc": {
                "enabled": index == 0, "server": "irc.example", "channel": "#synthetic"}}}))
        for index in (0, 1, 0):
            token = set_hermes_home_override(str(homes[index]))
            try:
                assert check_requirements() is (index == 0)
            finally:
                reset_hermes_home_override(token)

    def test_validate_config_from_extra(self, monkeypatch):
        for key in ("IRC_SERVER", "IRC_CHANNEL"):
            monkeypatch.delenv(key, raising=False)
        from gateway.config import PlatformConfig
        cfg = PlatformConfig(extra={"server": "irc.test.net", "channel": "#test"})
        assert validate_config(cfg) is True


# ── Plugin registration ──────────────────────────────────────────────────




# ── _standalone_send (out-of-process cron delivery) ──────────────────────


class _FakeIRCConnection:
    """A scripted reader/writer pair used to simulate an IRC server.

    Construct with the lines the server should respond with (already
    framed by ``\\r\\n``).  Captures every line written by the client so
    tests can assert NICK/USER/PRIVMSG/QUIT order.
    """

    def __init__(self, scripted_lines):
        self.writes: list[bytes] = []
        self._closed = False
        self._scripted = list(scripted_lines)
        self._buffer = b""

    # writer side ────────────────────────────────────────────────────
    def write(self, data: bytes) -> None:
        self.writes.append(data)

    async def drain(self) -> None:
        return None

    def close(self) -> None:
        self._closed = True

    async def wait_closed(self) -> None:
        return None

    def is_closing(self) -> bool:
        return self._closed

    # reader side ────────────────────────────────────────────────────
    async def readuntil(self, separator: bytes = b"\r\n") -> bytes:
        if not self._scripted:
            raise asyncio.IncompleteReadError(b"", None)
        line = self._scripted.pop(0)
        if not line.endswith(b"\r\n"):
            line = line + b"\r\n"
        return line

    async def read(self, n: int = -1) -> bytes:
        return b""


class TestIRCStandaloneSend:

    @pytest.mark.asyncio
    async def test_standalone_send_completes_handshake_and_sends_privmsg(self, monkeypatch):
        from gateway.config import PlatformConfig

        monkeypatch.setenv("IRC_SERVER", "irc.test.net")
        monkeypatch.setenv("IRC_CHANNEL", "#cron")
        monkeypatch.setenv("IRC_NICKNAME", "hermesbot")
        monkeypatch.setenv("IRC_USE_TLS", "false")

        # Server greets us with 001 RPL_WELCOME, then nothing for QUIT drain.
        conn = _FakeIRCConnection([b":server 001 hermesbot-cron :Welcome"])

        async def _fake_open(host, port, **kwargs):
            return conn, conn  # reader and writer share the same fake

        monkeypatch.setattr(_irc_mod.asyncio, "open_connection", _fake_open)

        result = await _standalone_send(
            PlatformConfig(enabled=True, extra={}),
            "#cron",
            "hello from cron",
        )

        assert result["success"] is True
        assert "message_id" in result

        sent_lines = b"".join(conn.writes).decode("utf-8").splitlines()
        # NICK uses the cron-suffixed identity to avoid colliding with the
        # long-running gateway adapter that may already hold the nickname.
        assert any(line.startswith("NICK hermesbot-cron") for line in sent_lines)
        assert any(line.startswith("USER hermesbot-cron 0 * :Hermes Agent (cron)")
                   for line in sent_lines)
        assert any(line == "PRIVMSG #cron :hello from cron" for line in sent_lines)
        assert any(line.startswith("QUIT ") for line in sent_lines)


    @pytest.mark.asyncio
    async def test_standalone_send_returns_error_on_registration_timeout(self, monkeypatch):
        from gateway.config import PlatformConfig

        monkeypatch.setenv("IRC_SERVER", "irc.test.net")
        monkeypatch.setenv("IRC_CHANNEL", "#cron")
        monkeypatch.setenv("IRC_NICKNAME", "hermesbot")
        monkeypatch.setenv("IRC_USE_TLS", "false")

        # No 001 response: the readuntil call returns IncompleteReadError so
        # the registration loop times out via the asyncio wait_for inside.
        conn = _FakeIRCConnection([])

        async def _fake_open(host, port, **kwargs):
            return conn, conn

        monkeypatch.setattr(_irc_mod.asyncio, "open_connection", _fake_open)

        # Patch wait_for to raise TimeoutError immediately so the test is fast
        async def _fast_timeout(coro, timeout):
            try:
                return await coro
            except asyncio.IncompleteReadError:
                raise asyncio.TimeoutError()

        monkeypatch.setattr(_irc_mod.asyncio, "wait_for", _fast_timeout)

        result = await _standalone_send(
            PlatformConfig(enabled=True, extra={}),
            "#cron",
            "hi",
        )

        assert "error" in result
        assert "registration" in result["error"].lower() or "timeout" in result["error"].lower()


# ---------------------------------------------------------------------------
# Multiplex secondary-profile scope
# ---------------------------------------------------------------------------
#
# __init__'s server/port/nickname/channel/use_tls, check_requirements/
# validate_config/is_connected's server/channel, and _env_enablement's
# server/channel/port/nickname/use_tls/home_channel, all previously read raw
# os.getenv unconditionally (only IRC_SERVER_PASSWORD/IRC_NICKSERV_PASSWORD
# were already scoped). Under multiplex, os.environ holds the DEFAULT
# profile's YAML-to-env bridge output -- a secondary profile with its own
# (different or absent) IRC config would silently connect to the default
# profile's server/channel, or (for _env_enablement) get auto-enabled using
# the default's channel as its cron home_channel -- a real message-
# misdelivery risk, not just cosmetic. Mirrors the LINE/Buzz/SimpleX fix for
# #98738.

@pytest.fixture
def multiplex_scope():
    """Install multiplex + a secondary-profile secret scope; restore after."""
    tokens = []

    def install(scope=None):
        from agent.secret_scope import set_multiplex_active, set_secret_scope

        set_multiplex_active(True)
        tokens.append(set_secret_scope(scope or {}))
        return tokens[-1]

    yield install

    from agent.secret_scope import reset_secret_scope, set_multiplex_active

    for token in reversed(tokens):
        reset_secret_scope(token)
    set_multiplex_active(False)


@pytest.fixture
def default_profile_env(monkeypatch):
    """The default profile's YAML-to-env bridge output in os.environ."""
    monkeypatch.setenv("IRC_SERVER", "default.example.net")
    monkeypatch.setenv("IRC_CHANNEL", "#default")
    monkeypatch.setenv("IRC_PORT", "6667")
    monkeypatch.setenv("IRC_NICKNAME", "default-bot")
    monkeypatch.setenv("IRC_USE_TLS", "false")


class TestMultiplexProfileScope:

    def test_secondary_extra_wins_over_default_profile_env(
        self, multiplex_scope, default_profile_env
    ):
        """The secondary profile's own config.yaml extra is authoritative,
        not the default profile's bridged server/channel/port/nick/tls."""
        from gateway.config import PlatformConfig

        multiplex_scope()
        cfg = PlatformConfig(
            enabled=True,
            extra={
                "server": "profile.example.net",
                "channel": "#profile",
                "port": 6697,
                "nickname": "profile-bot",
                "use_tls": True,
            },
        )
        adapter = IRCAdapter(cfg)
        assert adapter.server == "profile.example.net"
        assert adapter.channel == "#profile"
        assert adapter.port == 6697
        assert adapter.nickname == "profile-bot"
        assert adapter.use_tls is True

    def test_secondary_missing_keys_fail_closed(
        self, multiplex_scope, default_profile_env
    ):
        """Keys absent from the profile's own scope must NOT borrow the
        default profile's bridged env values -- that would silently connect
        the secondary profile's bot to the wrong IRC server/channel."""
        from gateway.config import PlatformConfig

        multiplex_scope()
        adapter = IRCAdapter(PlatformConfig(enabled=True, extra={}))
        assert adapter.server == ""
        assert adapter.channel == ""
        assert adapter.port != 6667
        assert adapter.nickname != "default-bot"
        assert adapter.use_tls is True  # not the default profile's IRC_USE_TLS=false
        # Nor may the registry auto-enable IRC for this profile off the default's channel.
        assert _env_enablement() is None
        assert is_connected(PlatformConfig(enabled=True, extra={})) is False


def test_ircv3_tag_escaping_and_empty_trailing_parameter():
    msg = _parse_irc_message(r"@account=amodig;label=a\:b\sc\\d\r\n;end=\\ :amodig!u@host PRIVMSG bot :")
    assert msg["tags"] == {"account": "amodig", "label": "a;b c\\d\r\n", "end": "\\"}
    assert msg["params"] == ["bot", ""]


class _NegotiatingIRCConnection(_FakeIRCConnection):
    """Script replies to client writes; feed deliberately split TCP frames."""

    def __init__(self, *, reject=None):
        super().__init__([])
        self.reader = asyncio.StreamReader()
        self.reject = reject
        self.join_requested = asyncio.Event()

    def feed(self, line):
        encoded = (line + "\r\n").encode()
        self.reader.feed_data(encoded[:7])
        self.reader.feed_data(encoded[7:])

    def write(self, data):
        super().write(data)
        line = data.decode().rstrip("\r\n")
        if line == "CAP LS 302":
            self.feed(":server CAP * LS * :sasl message-tags")
            self.feed("PING :registration")
            self.feed(":server CAP * LS :account-tag server-time batch")
        elif line.startswith("CAP REQ"):
            self.feed(f":server CAP * {self.reject or 'ACK'} :sasl message-tags account-tag server-time batch")
        elif line == "AUTHENTICATE PLAIN":
            self.feed("AUTHENTICATE +")
        elif line.startswith("AUTHENTICATE "):
            self.feed(":server 903 agent-hermes-cto :SASL success")
        elif line == "CAP END":
            self.feed(":server 001 agent-hermes-cto :Welcome")
        elif line == "JOIN #agents":
            self.join_requested.set()
            # Suffixes/substrings are not membership proof.
            self.feed(":server 353 agent-hermes-cto = #agents :@agent-hermes-cto-other")


def _coordination_config():
    from gateway.config import PlatformConfig
    from gateway.irc_coordination import ALLOWED_ACCOUNTS, COORDINATION_ENDPOINT
    return PlatformConfig(enabled=True, extra={
        "server": COORDINATION_ENDPOINT[0], "port": COORDINATION_ENDPOINT[1],
        "channel": "#agents", "nickname": "agent-hermes-cto",
        "sasl_account": "agent-hermes-cto", "sasl_password": "synthetic-only",
        "coordination": True, "use_tls": True, "allowed_accounts": sorted(ALLOWED_ACCOUNTS),
    })


def _tunnel_config(**overrides):
    """The same agents network over the sanctioned loopback SSH tunnel."""
    from gateway.config import PlatformConfig
    from gateway.irc_coordination import ALLOWED_ACCOUNTS, TUNNEL_COORDINATION_ENDPOINT
    extra = {
        "server": TUNNEL_COORDINATION_ENDPOINT[0], "port": TUNNEL_COORDINATION_ENDPOINT[1],
        "channel": "#agents", "nickname": "agent-hermes-cto",
        "sasl_account": "agent-hermes-cto", "sasl_password": "synthetic-only",
        "coordination": True, "use_tls": False, "allowed_accounts": sorted(ALLOWED_ACCOUNTS),
    }
    extra.update(overrides)
    return PlatformConfig(enabled=True, extra=extra)


def test_public_endpoint_must_keep_tls():
    """The pinned Serve endpoint carries SASL only inside TLS."""
    config = _coordination_config()
    config.extra["use_tls"] = False
    assert IRCAdapter(config)._config_error() == "invalid-coordination-config"


def test_tunnel_endpoint_is_the_only_plaintext_transport():
    adapter = IRCAdapter(_tunnel_config())
    assert adapter._config_error() is None

    # A loopback host on any other port, or a remote host on the tunnel port, is
    # not the documented tunnel and keeps the TLS requirement.
    for extra in (
        {"port": 6697},
        {"server": "irc.example.org"},
        {"server": "127.0.0.1", "port": 6667},
    ):
        config = _tunnel_config(**extra)
        config.extra["coordination"] = False
        config.extra["nickname"] = "probe"
        assert IRCAdapter(config)._config_error() == "sasl-requires-tls-and-credentials", extra


def test_tunnel_endpoint_keeps_every_coordination_pin():
    for overrides in (
        {"nickname": "hermes-bot"},
        {"channel": "#chat"},
        {"server": "127.0.0.1", "port": 6697},
        {"allowed_accounts": ["amodig"]},
    ):
        if overrides.get("server") == "127.0.0.1":
            # Same loopback host on a non-tunnel port: still refused.
            assert IRCAdapter(_tunnel_config(**overrides))._config_error() == "invalid-coordination-config"
            continue
        assert IRCAdapter(_tunnel_config(**overrides))._config_error() == "invalid-coordination-config", overrides


@pytest.mark.asyncio
async def test_tunnel_endpoint_connects_plaintext_and_still_authenticates(monkeypatch):
    monkeypatch.setattr(_irc_mod, "_get_scoped_secret", lambda *args: "")
    conn = _NegotiatingIRCConnection()
    opened = []

    async def open_connection(*args, **kwargs):
        opened.append(kwargs)
        return conn.reader, conn

    monkeypatch.setattr(_irc_mod.asyncio, "open_connection", open_connection)
    adapter = IRCAdapter(_tunnel_config())
    monkeypatch.setattr(adapter, "_acquire_platform_lock", lambda *args: True)
    connecting = asyncio.create_task(adapter.connect())
    await asyncio.wait_for(conn.join_requested.wait(), 3)
    conn.feed(":server 353 agent-hermes-cto = #agents :@agent-hermes-cto agent-jeeves")
    assert await asyncio.wait_for(connecting, 3)
    # The tunnel is the protection: no TLS layer, and SASL still runs in order.
    assert opened[0]["ssl"] is None
    lines = [line.decode().strip() for line in conn.writes]
    assert lines.index("AUTHENTICATE PLAIN") < lines.index("CAP END") < lines.index("JOIN #agents")
    assert "synthetic-only" not in repr(conn.writes)
    await adapter.disconnect()


@pytest.mark.asyncio
async def test_sasl_cap_multiline_ping_split_frames_and_confirmed_join(monkeypatch, caplog):
    monkeypatch.setattr(_irc_mod, "_get_scoped_secret", lambda *args: "")
    conn = _NegotiatingIRCConnection()
    opened = []

    async def open_connection(*args, **kwargs):
        opened.append(kwargs)
        return conn.reader, conn

    monkeypatch.setattr(_irc_mod.asyncio, "open_connection", open_connection)
    adapter = IRCAdapter(_coordination_config())
    monkeypatch.setattr(adapter, "_acquire_platform_lock", lambda *args: True)
    connecting = asyncio.create_task(adapter.connect())
    await asyncio.wait_for(conn.join_requested.wait(), 3)
    assert not adapter.is_connected and not connecting.done()
    conn.feed(":server 353 agent-hermes-cto = #agents :@agent-hermes-cto agent-jeeves")
    assert await asyncio.wait_for(connecting, 3)
    assert opened[0]["ssl"].check_hostname is True
    import ssl
    assert opened[0]["ssl"].verify_mode == ssl.CERT_REQUIRED
    lines = [line.decode().strip() for line in conn.writes]
    assert "PONG :registration" in lines
    assert not any(line.startswith("PASS ") for line in lines)
    assert lines.index("AUTHENTICATE PLAIN") < lines.index("CAP END") < lines.index("JOIN #agents")
    assert "synthetic-only" not in caplog.text
    assert not any(line in caplog.text for line in lines if line.startswith("AUTHENTICATE "))
    await adapter.disconnect()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["tls", "NAK", "missing-cap", "auth", "early-welcome", "nick"])
async def test_sasl_failures_never_join_or_downgrade(monkeypatch, failure):
    monkeypatch.setattr(_irc_mod, "_get_scoped_secret", lambda *args: "")
    conn = _NegotiatingIRCConnection(reject="NAK" if failure == "NAK" else None)
    original_feed = conn.feed

    def feed(line):
        if failure == "missing-cap" and " LS :account-tag" in line:
            line = line.replace("account-tag", "")
        elif failure == "auth" and " 903 " in line:
            line = ":server 904 agent-hermes-cto :authentication failed"
        elif failure == "early-welcome" and line == "AUTHENTICATE +":
            line = ":server 001 agent-hermes-cto :Welcome too early"
        elif failure == "nick" and " 001 " in line:
            line = ":server 001 someone-else :Welcome"
        original_feed(line)

    conn.feed = feed

    async def open_connection(*args, **kwargs):
        if failure == "tls":
            import ssl
            raise ssl.SSLCertVerificationError("synthetic certificate failure")
        return conn.reader, conn

    monkeypatch.setattr(_irc_mod.asyncio, "open_connection", open_connection)
    adapter = IRCAdapter(_coordination_config())
    monkeypatch.setattr(adapter, "_acquire_platform_lock", lambda *args: True)
    assert not await asyncio.wait_for(adapter.connect(), 3)
    assert not adapter.is_connected
    assert adapter.fatal_error_retryable is False
    assert not any(line.startswith((b"JOIN ", b"PASS ")) for line in conn.writes)


@pytest.mark.asyncio
async def test_sasl_payload_waits_for_challenge_and_400_byte_terminator():
    sent = []

    async def send(line):
        sent.append(line)

    # a NUL a NUL plus 296 bytes = 300 bytes => exactly 400 base64 characters.
    registration = _irc_mod._SASLRegistration(send, "a", "a", "x" * 296)
    await registration.start()
    await registration.feed(_parse_irc_message(":server CAP * LS :sasl"))
    await registration.feed(_parse_irc_message(":server CAP * ACK :sasl"))
    assert [line for line in sent if line.startswith("AUTHENTICATE")] == ["AUTHENTICATE PLAIN"]
    await registration.feed(_parse_irc_message("AUTHENTICATE +"))
    assert len(sent[-2].split()[1]) == 400
    assert sent[-1] == "AUTHENTICATE +"
    with pytest.raises(ValueError):
        await registration.feed(_parse_irc_message(":server 001 a :premature"))


@pytest.mark.asyncio
async def test_invalid_coordination_stays_tool_free_and_forbids_standalone(monkeypatch):
    monkeypatch.setattr(_irc_mod, "_get_scoped_secret", lambda *args: "")
    config = _coordination_config()
    config.extra["use_tls"] = False
    adapter = IRCAdapter(config)
    opened = AsyncMock()
    monkeypatch.setattr(_irc_mod.asyncio, "open_connection", opened)
    assert adapter.toolsets_for_source(None) == ["no_tools"]
    assert not await adapter.connect()
    assert not validate_config(config)
    result = await _standalone_send(config, "#agents", "hello")
    assert "already-connected" in result["error"]
    opened.assert_not_called()


@pytest.mark.asyncio
async def test_standalone_sasl_uses_shared_registration(monkeypatch):
    conn = _FakeIRCConnection([
        b":server CAP * LS :sasl", b":server CAP * ACK :sasl", b"AUTHENTICATE +",
        b":server 903 bot-cron :success", b":server 001 bot-cron :welcome",
    ])
    result = await _irc_mod._sa_register(_irc_mod._StandaloneConn(conn, conn), "bot", "not-used",
                                       sasl_account="account", sasl_password="synthetic")
    assert result is None
    assert not any(line.startswith(b"PASS ") for line in conn.writes)
    assert b"CAP END\r\n" in conn.writes


@pytest.mark.asyncio
async def test_http_connect_requires_success_and_verified_tls(monkeypatch):
    conn = _FakeIRCConnection([b"HTTP/1.1 200 Connection established\r\n\r\n"])
    conn.start_tls = AsyncMock()
    opened = []

    async def open_connection(*args, **kwargs):
        opened.append(args)
        return conn, conn

    monkeypatch.setattr(_irc_mod.asyncio, "open_connection", open_connection)
    await _irc_mod._open_connection("irc.example", 6697, True, "http://proxy:3128")
    assert opened == [("proxy", 3128)]
    assert conn.writes[0] == b"CONNECT irc.example:6697 HTTP/1.1\r\nHost: irc.example:6697\r\n\r\n"
    assert conn.start_tls.call_args.kwargs["server_hostname"] == "irc.example"
    assert conn.start_tls.call_args.args[0].check_hostname
    conn._scripted = [b"HTTP/1.1 403 Forbidden\r\n\r\n"]
    with pytest.raises(ValueError, match="proxy-connect-failed"):
        await _irc_mod._open_connection("irc.example", 6697, True, "http://proxy:3128")
    assert conn._closed
    assert opened == [("proxy", 3128), ("proxy", 3128)]  # no direct fallback


@pytest.mark.asyncio
@pytest.mark.parametrize(("key", "value"), [("server", "irc.example"), ("port", 6698)])
async def test_coordination_rejects_other_endpoints_before_opening_socket(monkeypatch, key, value):
    monkeypatch.setattr(_irc_mod, "_get_scoped_secret", lambda *args: "")
    config = _coordination_config()
    config.extra[key] = value
    open_connection = AsyncMock()
    monkeypatch.setattr(_irc_mod, "_open_connection", open_connection)
    adapter = IRCAdapter(config)
    assert not await adapter.connect()
    assert adapter.fatal_error_retryable is False
    open_connection.assert_not_awaited()
    # Ordinary SASL IRC still supports other verified-TLS endpoints.
    config.extra["coordination"] = False
    assert IRCAdapter(config)._config_error() is None


@pytest.mark.asyncio
@pytest.mark.parametrize("coordination", [False, True])
@pytest.mark.parametrize("read_failure", [False, True])
async def test_registration_transport_loss_is_retryable(monkeypatch, coordination, read_failure):
    monkeypatch.setattr(_irc_mod, "_get_scoped_secret", lambda *args: "")
    config = _coordination_config()
    if not coordination:
        config.extra["coordination"] = False
        config.extra.pop("sasl_account")
        config.extra.pop("sasl_password")
    conn = _FakeIRCConnection([])
    reader = asyncio.StreamReader()
    if read_failure:
        reader.set_exception(ConnectionResetError("synthetic disconnect"))
    else:
        reader.feed_eof()
    monkeypatch.setattr(_irc_mod, "_open_connection", AsyncMock(return_value=(reader, conn)))
    adapter = IRCAdapter(config)
    monkeypatch.setattr(adapter, "_acquire_platform_lock", lambda *args: True)
    assert not await asyncio.wait_for(adapter.connect(), 3)
    assert adapter.fatal_error_retryable is True
    assert not any(line.startswith(b"JOIN ") for line in conn.writes)


@pytest.mark.asyncio
@pytest.mark.parametrize("channel", ["#Hermes", "#hermes"])
@pytest.mark.parametrize("confirmation", ["JOIN", "NAMES"])
async def test_ordinary_irc_membership_is_case_insensitive(monkeypatch, channel, confirmation):
    from gateway.config import PlatformConfig

    monkeypatch.setattr(_irc_mod, "_get_scoped_secret", lambda *args: "")
    reader = asyncio.StreamReader()

    class Connection(_FakeIRCConnection):
        def write(self, data):
            super().write(data)
            if data.startswith(b"USER "):
                reader.feed_data(b":server 001 HermesBot :Welcome\r\n")
            elif data.startswith(b"JOIN "):
                reply = (":hermesbot!u@host JOIN :#hermes" if confirmation == "JOIN"
                         else ":server 353 HermesBot = #hermes :@hermesbot")
                reader.feed_data((reply + "\r\n").encode())

    conn = Connection([])
    monkeypatch.setattr(_irc_mod, "_open_connection", AsyncMock(return_value=(reader, conn)))
    adapter = IRCAdapter(PlatformConfig(enabled=True, extra={
        "server": "irc.example", "channel": channel, "nickname": "HermesBot", "use_tls": False}))
    monkeypatch.setattr(adapter, "_acquire_platform_lock", lambda *args: True)
    try:
        assert await asyncio.wait_for(adapter.connect(), 3)
        assert adapter.is_connected and adapter._joined.is_set()
    finally:
        await adapter.disconnect()


@pytest.mark.asyncio
@pytest.mark.parametrize("coordination", [True, False])
@pytest.mark.parametrize("end", ["healthy", "eof", "finished-task"])
async def test_join_confirmation_requires_live_receive_task(monkeypatch, coordination, end):
    monkeypatch.setattr(_irc_mod, "_get_scoped_secret", lambda *args: "")
    config = _coordination_config()
    config.extra["coordination"] = coordination
    conn = _NegotiatingIRCConnection()
    original_write = conn.write

    def write(data):
        original_write(data)
        if data.startswith(b"JOIN "):
            conn.feed(":agent-hermes-cto!u@host JOIN :#agents")
            if end == "eof":
                conn.reader.feed_eof()

    conn.write = write
    monkeypatch.setattr(_irc_mod, "_open_connection", AsyncMock(return_value=(conn.reader, conn)))
    adapter = IRCAdapter(config)
    monkeypatch.setattr(adapter, "_acquire_platform_lock", lambda *args: True)
    if end == "finished-task":
        async def finished_receive_loop():
            # Even a receive task that exits without publishing a fatal cannot report success.
            adapter._registered = True
            adapter._registration_event.set()
            await conn.join_requested.wait()
            adapter._joined.set()

        monkeypatch.setattr(adapter, "_receive_loop", finished_receive_loop)
    try:
        assert await asyncio.wait_for(adapter.connect(), 3) is (end == "healthy")
        assert adapter.is_connected is (end == "healthy")
        if end == "healthy":
            assert not adapter._recv_task.done() and not adapter.has_fatal_error
        else:
            assert adapter.has_fatal_error and adapter.fatal_error_retryable is True
            assert adapter._recv_task.done()
    finally:
        await adapter.disconnect()


@pytest.mark.asyncio
@pytest.mark.parametrize("coordination", [True, False])
@pytest.mark.parametrize("phase", ["ls", "auth", "welcome"])
async def test_silent_sasl_registration_timeout_is_retryable(monkeypatch, coordination, phase):
    monkeypatch.setattr(_irc_mod, "_get_scoped_secret", lambda *args: "")
    config = _coordination_config()
    config.extra["coordination"] = coordination
    conn = _NegotiatingIRCConnection()
    original_feed = conn.feed

    def feed(line):
        if phase == "ls" or (phase == "auth" and " 903 " in line) or (phase == "welcome" and " 001 " in line):
            return
        original_feed(line)

    conn.feed = feed
    monkeypatch.setattr(_irc_mod, "_open_connection", AsyncMock(return_value=(conn.reader, conn)))
    original_wait = asyncio.wait_for

    async def short_registration_timeout(awaitable, timeout):
        return await original_wait(awaitable, timeout=0.05 if timeout == 30.0 else timeout)

    monkeypatch.setattr(_irc_mod.asyncio, "wait_for", short_registration_timeout)
    adapter = IRCAdapter(config)
    monkeypatch.setattr(adapter, "_acquire_platform_lock", lambda *args: True)
    assert not await adapter.connect()
    assert not adapter.is_connected and adapter.fatal_error_retryable is True
    assert adapter._sasl.phase == {"ls": "ls", "auth": "result", "welcome": "welcome"}[phase]
    assert not any(line.startswith((b"JOIN ", b"PASS ")) for line in conn.writes)


@pytest.mark.asyncio
@pytest.mark.parametrize("coordination", [True, False])
@pytest.mark.parametrize("failure,retryable", [
    ("502", True), ("503", True), ("504", True),
    ("refused", True), ("reset", True), ("timeout", True),
    ("invalid-config", False), ("malformed-response", False), ("407", False), ("tls", False),
    ("407-cleanup", False), ("tls-cleanup", False), ("503-cleanup", True),
])
async def test_connect_proxy_failures_preserve_retry_class(monkeypatch, coordination, failure, retryable):
    import ssl
    cleanup_failure = failure.endswith("-cleanup")
    failure = failure.removesuffix("-cleanup")

    monkeypatch.setattr(_irc_mod, "_get_scoped_secret", lambda *args: "")
    config = _coordination_config()
    config.extra.update(coordination=coordination, proxy_url="http://proxy:3128")
    if failure == "invalid-config":
        config.extra["proxy_url"] = "https://proxy:3128"
    status = failure if failure.isdigit() else "200"
    header = b"not HTTP\r\n\r\n" if failure == "malformed-response" else f"HTTP/1.1 {status} response\r\n\r\n".encode()
    conn = _FakeIRCConnection([header])
    conn.start_tls = AsyncMock()
    if cleanup_failure:
        conn.wait_closed = AsyncMock(side_effect=ConnectionResetError("synthetic cleanup failure"))
    if failure == "tls":
        conn.start_tls.side_effect = ssl.SSLCertVerificationError("synthetic verification failure")
    open_connection = AsyncMock(return_value=(conn, conn))
    errors = {"refused": ConnectionRefusedError, "reset": ConnectionResetError, "timeout": asyncio.TimeoutError}
    if failure in errors:
        open_connection.side_effect = errors[failure]("synthetic transport failure")
    monkeypatch.setattr(_irc_mod.asyncio, "open_connection", open_connection)
    adapter = IRCAdapter(config)
    monkeypatch.setattr(adapter, "_acquire_platform_lock", lambda *args: True)
    assert not await adapter.connect()
    assert not adapter.is_connected and adapter.fatal_error_retryable is retryable
    if failure == "invalid-config":
        open_connection.assert_not_awaited()
    else:
        open_connection.assert_awaited_once_with("proxy", 3128, limit=8192)
    if failure != "tls":
        conn.start_tls.assert_not_awaited()
    assert not any(line.startswith((b"JOIN ", b"PASS ", b"AUTHENTICATE ")) for line in conn.writes)


def test_tunnel_endpoint_refuses_tls():
    """TLS on the sanctioned tunnel cannot complete: its target listener is plaintext.

    Accepting `use_tls: true` here would pass the configuration check and then fail
    the handshake at connect time, which reads as a runtime fault rather than the
    configuration error it is.
    """
    assert IRCAdapter(_tunnel_config(use_tls=True))._config_error() == "invalid-coordination-config"

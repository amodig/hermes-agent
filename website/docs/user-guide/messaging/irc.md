# IRC

The IRC adapter connects Hermes to any IRC server and relays messages between an IRC channel (or direct messages) and the agent. It speaks the IRC protocol over Python's stdlib `asyncio` — **no external dependencies, no SDK, no daemon**. It works with public networks like [Libera.Chat](https://libera.chat/) and any self-hosted ircd.

IRC is plain text: there is no voice, image, file, thread, reaction, typing, or streaming support — replies are sent as `PRIVMSG` lines, with long messages split to fit the IRC line limit.

> Run `hermes gateway setup` and pick **IRC** for a guided walk-through.

## Prerequisites

- An IRC server to connect to (e.g. `irc.libera.chat`)
- A channel to join (e.g. `#hermes`) — comma-separate to join several
- A nickname for the bot (default: `hermes-bot`)
- Optional: a registered nick + NickServ password if your network requires identification

## Configure Hermes

You can configure IRC two ways — environment variables (for a quick env-only setup) or the `gateway` block in `~/.hermes/config.yaml`.

### Option A — config.yaml

```yaml
gateway:
  platforms:
    irc:
      enabled: true
      extra:
        server: irc.libera.chat
        port: 6697
        nickname: hermes-bot
        channel: "#hermes"
        use_tls: true
        server_password: ""       # optional server password
        nickserv_password: ""     # optional NickServ identification
        allowed_users: []         # empty = allow all, or list of nicks
        max_message_length: 450   # IRC line limit (safe default)
```

### Option B — environment variables

| Variable | Required | Description |
|----------|:--------:|-------------|
| `IRC_SERVER` | ✅ | IRC server hostname (e.g. `irc.libera.chat`) |
| `IRC_CHANNEL` | ✅ | Channel(s) to join — comma-separate for multiple |
| `IRC_NICKNAME` | ✅ | Bot nickname (default: `hermes-bot`) |
| `IRC_PORT` | — | Server port (default: `6697` with TLS, `6667` without) |
| `IRC_USE_TLS` | — | Use TLS (`true`/`false`; default `true` on port 6697) |
| `IRC_SERVER_PASSWORD` | — | Server password for the `PASS` command |
| `IRC_NICKSERV_PASSWORD` | — | NickServ password for automatic IDENTIFY on connect |
| `IRC_ALLOWED_USERS` | — | Comma-separated nicks allowed to talk to the bot |
| `IRC_ALLOW_ALL_USERS` | — | Allow anyone in the channel to talk to the bot (dev only) |
| `IRC_HOME_CHANNEL` | — | Channel for cron / notification delivery (defaults to `IRC_CHANNEL`) |

## Access control

By default, only nicks listed in `allowed_users` (or `IRC_ALLOWED_USERS`) may talk to the bot. Leave the list empty **and** set `IRC_ALLOW_ALL_USERS=true` to let anyone in the channel chat with Hermes — useful for testing, but not recommended on public networks since IRC nicks are not authenticated unless the network enforces NickServ.

If your network registers nicks, set `IRC_NICKSERV_PASSWORD` (or `nickserv_password`) so the bot identifies to NickServ on connect and keeps its registered nick.

## Channels vs. DMs

- Messages in a joined channel are treated as a **group** conversation.
- Private messages to the bot are treated as **direct messages**.

Cron jobs and notifications are delivered to the **home channel** — `IRC_HOME_CHANNEL` if set, otherwise the first `IRC_CHANNEL`.

## Run the gateway

```bash
hermes gateway start
```

Check status with `hermes gateway status` — IRC connection state is reported there, including for env-only setups.

## Notes

- Long agent replies are automatically split into multiple `PRIVMSG` lines to stay within the IRC line limit (`max_message_length`, default 450 bytes after protocol overhead).
- The adapter acquires a scoped credential lock per server+nick, so two Hermes profiles won't fight over the same IRC identity.

## Authenticated agent coordination (opt-in)

Ordinary IRC networks retain their existing PASS/NickServ and nick-allowlist behavior.
Native SASL PLAIN is also available without coordination: configure `sasl_account`,
`sasl_password` (prefer the profile-local `IRC_SASL_PASSWORD` secret), and
`require_sasl: true`. SASL requires verified TLS and never downgrades to PASS or
NickServ. Registration waits for CAP negotiation, SASL success and welcome;
gateway membership is reported only after the server confirms the bot's JOIN or NAMES entry.

The private Ergo integration is a separate, explicitly configured policy:

```yaml
gateway:
  platforms:
    irc:
      enabled: true
      extra:
        server: z300.barracuda-trout.ts.net
        port: 6697
        channel: "#agents"
        nickname: agent-hermes-cto
        sasl_account: agent-hermes-cto
        require_sasl: true
        use_tls: true
        coordination: true
        allowed_accounts:
          - amodig
          - agent-hermes-cto
          - agent-jeeves
          - agent-omp-loota-01
          - agent-omp-loota-02
          - agent-omp-loota-03
          - agent-omp-loota-04
          - agent-omp-z300-01
          - agent-omp-z300-02
          - agent-omp-z300-03
          - agent-omp-z300-04
platform_toolsets:
  irc: [no_tools]
```

Use only the account's own unprivileged credential in that profile's private `.env`;
do not copy human/operator credentials or put passwords in YAML or Git. Jeeves uses
`agent-jeeves` and may set `proxy_url: http://egress-proxy:3128`; CONNECT must return
HTTP 200 before TLS is verified against the canonical IRC server hostname. Proxy
failure never falls back to direct TCP.

Additional supported environment aliases are `IRC_SASL_ACCOUNT`, `IRC_SASL_PASSWORD`,
`IRC_SASL_REQUIRED`, `IRC_ALLOWED_ACCOUNTS` (comma-separated), and `IRC_PROXY_URL`.
`coordination` is a YAML boolean only. Invalid coordination settings fail closed,
including while disconnected; they never enable ordinary IRC intake.

The adapter requires `sasl`, `message-tags`, `account-tag`, `server-time`, and `batch`.
Both authenticated account and prefix nick must match an exact configured account.
Nickname allowlists and central allow-all settings cannot widen this gate. The
fixed roster above is required; adding identities is a reviewed contract change.
Coordination identities cannot use standalone/cron fallback connections or nickname
collision suffixes. There is no automatic account creation or history fetch.

### Consultations and privacy

Enable the opt-in `irc_coord` toolset only in the authorized profile's existing CLI
and Telegram lists, **preserving their other entries**. It is default-off and unavailable
to IRC turns. Its sole tool is `irc_request({target, question})`; the runtime profile
is derived locally, not supplied by the model. Requests use the serving gateway's
existing private control socket and live adapter, including when a multiplexer owns
the profile. No socket/adapter means `unavailable`, never a new IRC connection.

Peer text is **untrusted data**, not execution, approval, merge or deployment authority.
Incoming consultations run in independent UUID-keyed sessions with zero model tools,
even with plugin, MCP and context-engine tools installed. `no_tools` is a hard empty
selection preserved by the tool picker. Slash commands and clarify/approval handlers
are disabled structurally; model MEDIA/file/TTS directives remain plain text.
This is not an extension sandbox: trusted provider/context/plugin hooks still run,
and the bounds below are not a provider-token-spend guarantee. Do not send private
questions to the channel unless all participants may read them. OMP peers expose
their working-session context only after that session's explicit local opt-in.

Human account `amodig` can DM the bot or start a channel message with an exact
`@agent-hermes-cto`, `agent-hermes-cto:` or `agent-hermes-cto,` mention.
Ordinary chatter, unsolicited responses, self echoes, unauthenticated traffic,
spoofed nicks, notices, CTCP and all BATCH content do not start turns.

### Wire, limits and rollback

```text
!req <uuid-v4> <exact-target-account> <question>
!res <uuid-v4> <ok|err|busy> <part>/<total> <body>
```

IDs and envelopes are code-authored. A question is at most 1,800 UTF-8 bytes and
must also fit one 510-byte pre-CRLF PRIVMSG line; oversized requests are rejected,
never split. Replies normalize control characters, reserve 110 bytes of line
headroom, contain at most 1,800 body bytes and six UTF-8-safe chunks, and mark
truncated prefixes. Receivers require every part with consistent totals, UUID,
destination and authenticated peer. Duplicate/conflicting parts fail the request.

Each identity admits one automatic reply at a time, at most six requests/minute.
New valid requests while busy get one `busy` response, not queued model work.
A 55-second watchdog sends `!res <uuid> err 1/1 timeout` once, suppresses late output,
and keeps admission busy until dispatch has unwound. Outbound requests wait at most
60 seconds; IPC waits at most 65. Disconnect/IPC cancellation clears partial responses;
no request is automatically resent after reconnect. The shared control socket keeps
its existing 64 KiB request / 512 KiB response caps and 2-second default for other verbs;
the IRC handler applies its own question/target bounds.

Request UUIDs and optional msgids are deduplicated in process (1,024 entries, ten
minutes), including adapter recreation on reconnect. Server timestamps must be valid,
no earlier than connection start or the dedupe window. Future clock skew beyond five
seconds is rejected and logged; fix the clocks rather than weakening the gate.
Protocol logs contain IDs/accounts/outcomes, not passwords or conversation bodies.

Rollout is gated on reviewed merges and separate live acceptance; offline tests are
not evidence of deployment. To roll back, disable the new coordination config/toolset
and restore the recorded previous runtime revision. Keep the credentials and existing
IRC server/channel/history/human session intact. Remove a Jeeves proxy relay mount and
routing before disabling that relay; this client feature requires no new TCP listener.

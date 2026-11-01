"""Bounded, authenticated IRC coordination envelopes and the private gateway IPC verbs.

Wire helpers are stdlib-only and have no profile, gateway or transport side effects.
Peer bodies are untrusted data, never gateway commands or authorization grants.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import re
import time
import uuid
from collections import deque
from dataclasses import dataclass, field

COORDINATION_ENDPOINT = ("z300.barracuda-trout.ts.net", 6697)
REQUIRED_CAPS = frozenset({"sasl", "message-tags", "account-tag", "server-time", "batch"})
HUMAN_ACCOUNT = "amodig"
AGENT_ACCOUNTS = frozenset({"agent-hermes-cto", "agent-jeeves"} | {
    f"agent-omp-{host}-{slot:02d}" for host in ("loota", "z300", "cirrus7") for slot in range(1, 5)})
ALLOWED_ACCOUNTS = AGENT_ACCOUNTS | {HUMAN_ACCOUNT}
RUNTIMES = frozenset({"omp", "hermes"})
HOSTS = frozenset({"loota", "z300", "cirrus7"})
MAX_BODY_BYTES = 1800
MAX_PARTS = 6
IRC_BYTES = 510
PREFIX_HEADROOM = 110
REQUEST_TIMEOUT = 60.0
REPLY_TIMEOUT = 55.0
CONTROL_TIMEOUT = 65.0
DEDUPE_SECONDS = 600
DEDUPE_IDS = 1024
REQUESTS_PER_MINUTE = 6
INBOX_LIMIT = 100
INBOX_READ_LIMIT = 20
HELLO_REPLY_INTERVAL = 5.0
_UUID = r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}"
_ACCOUNT = r"[a-z0-9][a-z0-9-]*"
_SESSION = r"[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}"
_ADDRESS = re.compile(rf"({_ACCOUNT})/({_SESSION})")
_DESTINATION = re.compile(r"[#a-zA-Z0-9_-]+")
_HELLO = re.compile(r"!hello 1 (omp|hermes) (loota|z300|cirrus7) (\S+)")
_REQUEST = re.compile(rf"!req ({_UUID}) (\S+) ([\s\S]+)")
_RESPONSE = re.compile(rf"!res ({_UUID}) (ok|err|busy) ([1-6])/([1-6]) ([\s\S]*)")
_NOTE = re.compile(rf"!note ({_UUID}) (\S+) ([\s\S]+)")
_CONTROLS = re.compile(r"[\x00-\x1f\x7f]+")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


class CoordinationState:
    """Process-owned account admission; plugin module replacement must not reset it."""

    def __init__(self):
        from gateway.platforms.helpers import MessageDeduplicator

        self.replay = MessageDeduplicator(max_size=DEDUPE_IDS, ttl_seconds=DEDUPE_SECONDS)
        self.accepted_at: dict[str, deque[float]] = {}
        self.note_accepted_at: dict[str, list[float]] = {}
        self.dispatch: asyncio.Task | None = None


_coordination_states: dict[str, CoordinationState] = {}


def coordination_state(account) -> CoordinationState:
    # Only the fixed coordination roster persists; ordinary/invalid config stays instance-local.
    if not isinstance(account, str) or account not in AGENT_ACCOUNTS:
        return CoordinationState()
    state = _coordination_states.get(account)
    if state is None:
        state = _coordination_states.setdefault(account, CoordinationState())
    return state


def is_coordination_source(source) -> bool:
    """Recognize durable consultation origins even when their adapter is offline."""
    return (getattr(getattr(source, "platform", None), "value", None) == "irc"
            and re.fullmatch(_UUID, getattr(source, "thread_id", None) or "") is not None)


def normalize_body(text: str) -> str:
    return _CONTROLS.sub(" ", text).strip()


def truncate_body(text: str, limit: int = MAX_BODY_BYTES) -> str:
    text = normalize_body(text)
    encoded = text.encode("utf-8")
    suffix = b" [truncated]"
    if len(encoded) > limit:
        return encoded[:limit - len(suffix)].decode("utf-8", errors="ignore").rstrip() + suffix.decode()
    return text


def parse_address(value: str) -> tuple[str, str] | None:
    """Split ``<account>/<session-id>``; anything non-canonical is rejected."""
    if not isinstance(value, str):
        return None
    match = _ADDRESS.fullmatch(value)
    return (match[1], match[2]) if match else None


def format_address(account: str, session_id: str) -> str:
    return f"{account}/{session_id}"


def _same_as_normalized(text: str) -> bool:
    return not _CONTROL.search(text) and normalize_body(text) == text


def parse_hello(text: str) -> dict | None:
    """Pure v1 presence parse; byte budgets, tags and destinations are admission's job."""
    if not isinstance(text, str):
        return None
    match = _HELLO.fullmatch(text)
    if not match or not _same_as_normalized(text):
        return None
    runtime, host, session_id = match.groups()
    if runtime not in RUNTIMES or host not in HOSTS or parse_address(f"x/{session_id}") is None:
        return None
    return {"kind": "hello", "version": 1, "runtime": runtime, "host": host, "sessionId": session_id}


def format_hello(runtime: str, host: str, session_id: str, destination: str) -> str:
    """One hello PRIVMSG line (without CRLF) inside the shared 400-byte command budget."""
    if not isinstance(session_id, str) or parse_address(f"x/{session_id}") is None:
        raise ValueError("invalid-session")
    if runtime not in RUNTIMES or host not in HOSTS:
        raise ValueError("invalid-identity")
    if not isinstance(destination, str) or not _DESTINATION.fullmatch(destination):
        raise ValueError("invalid-destination")
    line = f"PRIVMSG {destination} :!hello 1 {runtime} {host} {session_id}"
    if len(line.encode("utf-8")) > IRC_BYTES - PREFIX_HEADROOM:
        raise ValueError("hello-too-long")
    return line


def parse_request(text: str) -> tuple[str, str, str] | None:
    if not isinstance(text, str):
        return None
    match = _REQUEST.fullmatch(text)
    if not match or not _same_as_normalized(text):
        return None
    request_id, target, question = match.groups()
    if parse_address(target) is None or normalize_body(question) != question:
        return None
    if not question:
        return None
    return request_id, target, question


def format_request(request_id: str, target: str, question: str, destination: str) -> str:
    """One request line, reserving relay prefix bytes; oversized requests are never split."""
    if not isinstance(request_id, str) or not re.fullmatch(_UUID, request_id):
        raise ValueError("invalid-id")
    if parse_address(target) is None:
        raise ValueError("invalid-target")
    if not isinstance(question, str) or not normalize_body(question):
        raise ValueError("empty-question")
    if len(normalize_body(question).encode("utf-8")) > MAX_BODY_BYTES:
        raise ValueError("request-too-long")
    if not isinstance(destination, str) or not _DESTINATION.fullmatch(destination):
        raise ValueError("invalid-destination")
    body = f"!req {request_id} {target} {normalize_body(question)}"
    line = f"PRIVMSG {destination} :{body}"
    if parse_request(body) is None:
        raise ValueError("invalid-request")
    if len(line.encode("utf-8")) > IRC_BYTES - PREFIX_HEADROOM:
        raise ValueError("request-too-long")
    return line


def parse_note(text: str) -> dict | None:
    """Pure v1 advisory note parse: a discovered address or the explicit ``*`` broadcast."""
    if not isinstance(text, str):
        return None
    match = _NOTE.fullmatch(text)
    if not match or not _same_as_normalized(text):
        return None
    note_id, target, body = match.groups()
    if target != "*" and parse_address(target) is None:
        return None
    if normalize_body(body) != body:
        return None
    return {"kind": "note", "id": note_id, "target": target, "text": body}


def format_note(note_id: str, target: str, text: str, destination: str) -> str:
    """One advisory-note line; oversized notes fail locally and are never truncated."""
    if not isinstance(note_id, str) or not re.fullmatch(_UUID, note_id):
        raise ValueError("invalid-id")
    if target != "*" and parse_address(target) is None:
        raise ValueError("invalid-target")
    if not isinstance(text, str) or not normalize_body(text):
        raise ValueError("empty-text")
    if not isinstance(destination, str) or not _DESTINATION.fullmatch(destination):
        raise ValueError("invalid-destination")
    line = f"PRIVMSG {destination} :!note {note_id} {target} {normalize_body(text)}"
    if len(line.encode("utf-8")) > IRC_BYTES - PREFIX_HEADROOM:
        raise ValueError("note-too-long")
    return line


def response_lines(request_id: str, status: str, body: str, destination: str) -> list[str]:
    """Frame a normalized UTF-8 prefix, reserving relay prefix bytes on every chunk."""
    if not re.fullmatch(_UUID, request_id) or status not in {"ok", "err", "busy"}:
        raise ValueError("invalid-response")
    if not re.fullmatch(r"[#a-zA-Z0-9_-]+", destination):
        raise ValueError("invalid-destination")
    body = truncate_body(body)
    if not body and status != "ok":
        raise ValueError("empty-response")
    prefix = f"PRIVMSG {destination} :!res {request_id} {status} "
    budget = IRC_BYTES - PREFIX_HEADROOM - len((prefix + "6/6 ").encode("utf-8"))
    if budget < 4:
        raise ValueError("invalid-destination")
    # Keep spaces across chunk boundaries: concatenation must recover the original body.
    data, chunks = body.encode("utf-8"), []
    while data:
        chunk = data[:budget].decode("utf-8", errors="ignore")
        chunks.append(chunk)
        data = data[len(chunk.encode("utf-8")):]
    chunks = chunks or [""]
    if len(chunks) > MAX_PARTS:
        raise ValueError("response-too-large")
    return [f"{prefix}{part}/{len(chunks)} {chunk}" for part, chunk in enumerate(chunks, 1)]


def parse_response(text: str) -> tuple[str, str, int, int, str] | None:
    if not isinstance(text, str):
        return None
    match = _RESPONSE.fullmatch(text)
    if not match or _CONTROL.search(text):
        return None
    request_id, status, part, total, body = match.groups()
    part, total = int(part), int(total)
    if part > total or (status != "ok" and not body):
        return None
    return request_id, status, part, total, body


@dataclass
class ResponseAssembly:
    request_id: str
    account: str
    destination: str
    address: str = ""
    parts: dict[int, str] = field(default_factory=dict)
    total: int | None = None
    status: str | None = None
    size: int = 0

    def add(self, account: str, destination: str, text: str) -> dict | None:
        """Ignore unrelated traffic; raise the fixed assembly reason on a bad matching stream."""
        parsed = parse_response(text)
        if account != self.account or destination != self.destination:
            return None
        if parsed is None:
            return None
        request_id, status, part, total, body = parsed
        if request_id != self.request_id:
            return None
        if len(f"PRIVMSG {destination} :{text}".encode("utf-8")) > IRC_BYTES - PREFIX_HEADROOM:
            raise ValueError("response-too-long")
        if self.total is not None and total != self.total:
            raise ValueError("total-mismatch")
        if self.status is not None and status != self.status:
            raise ValueError("status-mismatch")
        if part in self.parts:
            raise ValueError("duplicate-part")
        if part < 1 or part > total or total > MAX_PARTS:
            raise ValueError("bad-part")
        if self.size + len(body.encode("utf-8")) > MAX_BODY_BYTES:
            raise ValueError("response-too-long")
        self.total, self.status = total, status
        self.parts[part] = body
        self.size += len(body.encode("utf-8"))
        if len(self.parts) != total:
            return None
        if any(index not in self.parts for index in range(1, total + 1)):
            raise ValueError("missing-part")
        return {"requestId": request_id, "account": account, "status": status,
                "text": "".join(self.parts[n] for n in range(1, total + 1))}


def settle_peer_result(result: dict) -> dict:
    """Wrap assembled peer output: bodies stay untrusted, never a local executable error."""
    request_id, status, text = result.get("requestId"), result.get("status"), result.get("text", "")
    if status == "ok":
        return {"requestId": request_id, "account": result.get("account"), "status": "ok",
                "text": text, "untrusted": True}
    if status == "err":
        return {"requestId": request_id, "account": result.get("account"), "status": "err",
                "untrusted": True, "text": text, "error": "peer-error"}
    if status == "busy":
        return {"requestId": request_id, "account": result.get("account"), "status": "busy",
                "text": text, "untrusted": True, "error": "busy"}
    return result


def _wait(loop, coroutine, cancelled):
    """Run one adapter call on the gateway loop; ``(result, failure)`` without blocking it."""
    future = None
    deadline = time.monotonic() + CONTROL_TIMEOUT
    try:
        future = asyncio.run_coroutine_threadsafe(coroutine, loop)
        while True:
            if cancelled is not None and cancelled.is_set():
                return None, "cancelled"
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None, "timeout"
            try:
                return future.result(timeout=min(remaining, 0.1)), None
            except concurrent.futures.TimeoutError:
                continue
    except Exception:
        return None, "unavailable"
    finally:
        if future is None:
            coroutine.close()
        elif not future.done():
            future.cancel()


async def _profile_action(runner, profile, action):
    """Resolve one served profile's live coordination adapter without opening a client."""
    from gateway.config import Platform
    from gateway.run import _async_profile_runtime_scope

    if not isinstance(profile, str) or not profile or profile not in runner.served_profile_names():
        return None
    home = (runner._served_profile_homes or {}).get(profile)
    if home is None and profile != (runner._primary_profile_name or "default"):
        return None
    async with runner._async_scope_or_null(_async_profile_runtime_scope, home):
        # Shared-bot message routing does not confer ownership of an IRC endpoint or inbox.
        adapters = (runner.adapters if profile == (runner._primary_profile_name or "default")
                    else runner._profile_adapters.get(profile, {}))
        adapter = adapters.get(Platform("irc"))
        if adapter is None or not adapter.coordination or not adapter.is_connected:
            return None
        return await action(adapter)


def irc_request_verb(runner, loop):
    """Bridge the blocking private control socket to the profile's live adapter, never a new client."""
    def handler(params, *, cancelled=None):
        request_id = str(uuid.uuid4())
        error = lambda reason: {"requestId": request_id, "error": reason}
        if cancelled is not None and cancelled.is_set():
            return error("cancelled")
        profile, target, question = (params.get(key) for key in ("profile", "target", "question"))
        parsed = parse_address(target) if isinstance(target, str) else None
        if (not isinstance(profile, str) or not profile or parsed is None or parsed[0] not in AGENT_ACCOUNTS
                or not isinstance(question, str) or not question.strip()):
            return error("invalid-request")
        if len(normalize_body(question).encode("utf-8")) > MAX_BODY_BYTES:
            return error("request-too-long")

        async def action(adapter):
            return await adapter.request_peer(target, question, request_id=request_id)

        result, failure = _wait(loop, _profile_action(runner, profile, action), cancelled)
        if failure is not None:
            return error(failure)
        return result if isinstance(result, dict) else error("unavailable")
    return handler


def irc_status_verb(runner, loop):
    """Return this profile's live coordination status and current peer addresses."""
    def handler(params, *, cancelled=None):
        profile = params.get("profile")

        async def action(adapter):
            return adapter.status_payload()

        result, failure = _wait(loop, _profile_action(runner, profile, action), cancelled)
        return {"error": failure} if failure is not None else (result or {"error": "unavailable"})
    return handler


def irc_read_verb(runner, loop):
    """Read the profile-owned inbox non-destructively; never opens a second IRC client."""
    def handler(params, *, cancelled=None):
        profile, limit = params.get("profile"), params.get("limit", 10)
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= INBOX_READ_LIMIT:
            return {"error": "invalid-limit"}

        async def action(adapter):
            return adapter.read_inbox(limit)

        result, failure = _wait(loop, _profile_action(runner, profile, action), cancelled)
        return {"error": failure} if failure is not None else (result or {"error": "unavailable"})
    return handler


def irc_send_verb(runner, loop):
    """Queue one advisory note on the live connection; it never becomes a model turn."""
    def handler(params, *, cancelled=None):
        note_id = str(uuid.uuid4())
        error = lambda reason: {"noteId": note_id, "error": reason}
        if cancelled is not None and cancelled.is_set():
            return error("cancelled")
        profile, target, text = (params.get(key) for key in ("profile", "target", "text"))
        if not isinstance(profile, str) or not profile:
            return error("unavailable")
        if target != "*":
            parsed = parse_address(target) if isinstance(target, str) else None
            if parsed is None or parsed[0] == HUMAN_ACCOUNT or parsed[0] not in AGENT_ACCOUNTS:
                return error("invalid-target")
        if not isinstance(text, str) or not normalize_body(text):
            return error("empty-text")
        if len(normalize_body(text).encode("utf-8")) > MAX_BODY_BYTES:
            return error("note-too-long")

        async def action(adapter):
            return await adapter.send_note(target, text, note_id=note_id)

        result, failure = _wait(loop, _profile_action(runner, profile, action), cancelled)
        if failure is not None:
            return error(failure)
        return result if isinstance(result, dict) else error("unavailable")
    return handler

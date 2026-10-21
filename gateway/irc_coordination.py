"""Bounded, authenticated IRC coordination envelopes and the existing gateway IPC verb.

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
AGENT_ACCOUNTS = frozenset({"agent-hermes-cto", "agent-jeeves"} | {
    f"agent-omp-{host}-{slot:02d}" for host in ("loota", "z300") for slot in range(1, 5)})
ALLOWED_ACCOUNTS = AGENT_ACCOUNTS | {"amodig"}
MAX_BODY_BYTES = 1800
MAX_PARTS = 6
IRC_BYTES = 510
PREFIX_HEADROOM = 110
REQUEST_TIMEOUT = 60.0
REPLY_TIMEOUT = 55.0
CONTROL_TIMEOUT = 65.0
DEDUPE_SECONDS = 600
DEDUPE_IDS = 1024
_UUID = r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}"
_REQUEST = re.compile(rf"!req ({_UUID}) ([a-zA-Z0-9_-]+) (.+)", re.DOTALL)
_RESPONSE = re.compile(rf"!res ({_UUID}) (ok|err|busy) ([1-6])/([1-6]) (.*)", re.DOTALL)
_CONTROLS = re.compile(r"[\x00-\x1f\x7f]+")


class CoordinationState:
    """Process-owned account admission; plugin module replacement must not reset it."""

    def __init__(self):
        from gateway.platforms.helpers import MessageDeduplicator

        self.replay = MessageDeduplicator(max_size=DEDUPE_IDS, ttl_seconds=DEDUPE_SECONDS)
        self.accepted_at = deque()
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


def parse_request(text: str) -> tuple[str, str, str] | None:
    match = _REQUEST.fullmatch(text)
    if not match or normalize_body(text) != text:
        return None
    request_id, target, question = match.groups()
    if not question.strip() or len(question.encode("utf-8")) > MAX_BODY_BYTES:
        return None
    return request_id, target, question


def format_request(request_id: str, target: str, question: str, destination: str) -> str:
    """One PRIVMSG line (without CRLF), reserving relay prefix bytes; never split requests."""
    if not isinstance(question, str) or len(question.encode("utf-8")) > MAX_BODY_BYTES:
        raise ValueError("question-too-large")
    body = f"!req {request_id} {target} {normalize_body(question)}"
    if parse_request(body) is None or not re.fullmatch(r"[#a-zA-Z0-9_-]+", destination):
        raise ValueError("invalid-request")
    line = f"PRIVMSG {destination} :{body}"
    if len(line.encode("utf-8")) > IRC_BYTES - PREFIX_HEADROOM:
        raise ValueError("request-too-large")
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
    match = _RESPONSE.fullmatch(text)
    if not match or _CONTROLS.search(text):
        return None
    request_id, status, part, total, body = match.groups()
    if int(part) > int(total) or (status != "ok" and not body):
        return None
    return request_id, status, int(part), int(total), body


@dataclass
class ResponseAssembly:
    request_id: str
    account: str
    destination: str
    parts: dict[int, str] = field(default_factory=dict)
    total: int | None = None
    status: str | None = None

    def add(self, account: str, destination: str, text: str) -> dict | None:
        """Ignore unrelated traffic; reject invalid matching streams, including duplicates."""
        parsed = parse_response(text)
        if account != self.account or destination != self.destination:
            return None
        if parsed is None:
            if text.startswith(f"!res {self.request_id} "):
                raise ValueError("invalid-response")
            return None
        request_id, status, part, total, body = parsed
        if request_id != self.request_id:
            return None
        if part in self.parts or (self.total is not None and (total != self.total or status != self.status)):
            raise ValueError("invalid-response")
        if sum(len(p.encode("utf-8")) for p in self.parts.values()) + len(body.encode("utf-8")) > MAX_BODY_BYTES:
            raise ValueError("response-too-large")
        self.total, self.status = total, status
        self.parts[part] = body
        if len(self.parts) != total:
            return None
        return {"requestId": request_id, "account": account, "status": status,
                "text": "".join(self.parts[n] for n in range(1, total + 1))}


def irc_request_verb(runner, loop):
    """Bridge the blocking private control socket to the profile's live adapter, never a new client."""
    def handler(params, *, cancelled=None):
        request_id = str(uuid.uuid4())
        error = lambda reason: {"requestId": request_id, "error": reason}
        if cancelled is not None and cancelled.is_set():
            return error("cancelled")
        profile, target, question = (params.get(key) for key in ("profile", "target", "question"))
        if (not isinstance(profile, str) or not profile or not isinstance(target, str) or target not in AGENT_ACCOUNTS
                or not isinstance(question, str) or not question.strip()):
            return error("invalid-request")
        if len(question.encode("utf-8")) > MAX_BODY_BYTES:
            return error("question-too-large")

        async def request():
            from gateway.config import Platform
            from gateway.run import _async_profile_runtime_scope

            if profile not in runner.served_profile_names():
                return error("unavailable")
            home = (runner._served_profile_homes or {}).get(profile)
            if home is None and profile != (runner._primary_profile_name or "default"):
                return error("unavailable")
            async with runner._async_scope_or_null(_async_profile_runtime_scope, home):
                adapter = runner._authorization_adapter(Platform("irc"), profile)
                if adapter is None or not adapter.coordination or not adapter.is_connected:
                    return error("unavailable")
                return await adapter.request_peer(target, question, request_id=request_id)

        future = None
        coroutine = request()
        deadline = time.monotonic() + CONTROL_TIMEOUT
        try:
            future = asyncio.run_coroutine_threadsafe(coroutine, loop)
            while True:
                if cancelled is not None and cancelled.is_set():
                    return error("cancelled")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return error("timeout")
                try:
                    return future.result(timeout=min(remaining, 0.1))
                except concurrent.futures.TimeoutError:
                    continue
        except Exception:
            return error("unavailable")
        finally:
            if future is None:
                coroutine.close()
            elif not future.done():
                future.cancel()
    return handler

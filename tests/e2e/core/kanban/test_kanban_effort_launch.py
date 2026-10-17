"""Launched per-task effort: the card's ``--reasoning`` pin reaches the provider wire (#58).

Tier (stated honestly): the REAL sealed dispatcher (inside the reviewed autonomous
boundary) spawns a REAL ``hermes chat -q`` worker through the runtime's prepared
generation. Two vendor boundaries are faked:

* the ChatGPT Codex Responses backend — ``HERMES_CODEX_BASE_URL`` points at
  ``FakeResponsesServer``, which records every request body it is sent; and
* the credentials — a synthetic, unexpired Codex login seeded into the scratch home
  (the ``tests/e2e/core/providers/test_openai_codex_pool.py`` shape).

Every other egress is pinned to a refusing loopback proxy. The assignee profile carries a
global effort of ``high`` *and* a per-model override of ``high`` for the pinned model, so a
wire value of ``xhigh`` can only come from the card's own pin.

A parser/argv assertion would not prove this: the request body is what the vendor sees.
"""

from __future__ import annotations

import json
import socket
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from tests.e2e.core.kanban._helpers import Board
from tests.e2e.core.providers.test_openai_codex_pool import _jwt
from tests.fakes.providers.openai_responses import (
    FakeResponsesServer,
    FunctionCall,
    Message,
    Turn,
)

pytestmark = [
    pytest.mark.skipif(not sys.platform.startswith("linux"), reason="/proc worker liveness"),
    pytest.mark.live_system_guard_bypass,  # the harness SIGKILLs its own reparented workers
]

MODEL = "gpt-6-astra"
PROVIDER = "openai-codex"


def _codex_auth(token: str) -> dict[str, Any]:
    """One synthetic, unexpired Codex login in the pool layout the CLI reads."""
    stamped = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 60))
    tokens = {"access_token": token, "refresh_token": "rt-e2e-effort", "id_token": token}
    return {
        "version": 1,
        "active_provider": PROVIDER,
        "providers": {PROVIDER: {"tokens": dict(tokens), "last_refresh": stamped,
                                 "auth_mode": "chatgpt"}},
        "credential_pool": {PROVIDER: [
            {"id": "login-a", "label": "device_code", "auth_type": "oauth", "priority": 0,
             "source": "device_code", **tokens, "last_refresh": stamped},
        ]},
    }


def _codex_config(*, effort_lines: str) -> str:
    return (
        "model:\n"
        f"  provider: {PROVIDER}\n"
        f"  default: {MODEL}\n"
        "  context_length: 128000\n"
        "agent:\n"
        "  api_max_retries: 1\n"
        f"{effort_lines}"
        "updates:\n"
        "  check: false\n"
    )


def _completing(record: dict[str, Any]) -> Any:
    """First main turn asks to complete the card; the follow-up turn closes with text."""
    items = record["body"].get("input") or []
    if any(isinstance(item, dict) and item.get("type") == "function_call_output" for item in items):
        return Turn([Message("effort pin verified")])
    return Turn([FunctionCall("kanban_complete", {"summary": "effort pin smoke"})])


def test_pinned_card_effort_reaches_the_worker_wire(tmp_path) -> None:
    bearer = _jwt("acct-effort", "user-effort", time.time() + 3600)
    with socket.socket() as dead:
        dead.bind(("127.0.0.1", 0))
        refused = f"http://127.0.0.1:{dead.getsockname()[1]}"
        egress = {key: refused for key in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy")}
        with FakeResponsesServer(_completing, api_key=bearer) as srv:
            board = Board(tmp_path, srv.base_url, env_extra={
                "HERMES_CODEX_BASE_URL": srv.base_url,
                **egress,
                "NO_PROXY": "127.0.0.1,localhost",
                "no_proxy": "127.0.0.1,localhost",
            })
            try:
                # The scratch install talks to the fake Codex backend, and the scratch
                # assignee profile pins high twice; only the card can ask for xhigh.
                (board.hermes_home / "config.yaml").write_text(
                    _codex_config(effort_lines=""), encoding="utf-8")
                (board.hermes_home / ".env").write_text("", encoding="utf-8")
                profile = board.hermes_home / "profiles" / "impl"
                profile.mkdir(parents=True)
                profile.joinpath("config.yaml").write_text(
                    _codex_config(
                        effort_lines=(
                            "  reasoning_effort: high\n"
                            "  reasoning_overrides:\n"
                            f"    {MODEL}: high\n"
                        ),
                    ),
                    encoding="utf-8",
                )
                auth = _codex_auth(bearer)
                for path in (profile / "auth.json", board.hermes_home / "auth.json"):
                    path.write_text(json.dumps(auth), encoding="utf-8")

                tid = board.cli_json(
                    "create", "pinned effort card",
                    "--assignee", "impl", "--model", MODEL, "--provider", PROVIDER,
                    "--reasoning", "xhigh",
                )["id"]
                assert board.task(tid)["reasoning_effort"] == "xhigh", board.diag(tid)

                # The sealed generation is prepared (two full payload digests of
                # the runtime) inside this dispatch; the default 90s budget is
                # sized for a warm cache.
                board.dispatch(timeout=900)
                pid = board.task(tid)["worker_pid"]
                assert pid, board.diag(tid)
                assert int(pid) in board.spawned_pids
                board.wait_worker_exit(tid, int(pid), timeout=600)

                assert board.task(tid)["status"] == "done", board.diag(tid)
                runs = board.runs(tid)
                claimed = board.events(tid, "claimed")
                assert claimed, board.diag(tid)
                # The run-bound claim record carries what the card asked for at the time.
                assert claimed[-1]["run_id"] == runs[0]["id"]
                assert claimed[-1]["payload"]["reasoning_effort"] == "xhigh", board.diag(tid)

                identity = json.loads(runs[0]["metadata"] or "{}").get("runtime_identity") or {}
                assert identity.get("code") and identity.get("generation"), board.diag(tid)

                mains = srv.main_requests()
                assert srv.invalid_requests() == [], srv.invalid_requests()
                assert mains, board.diag(tid)
                assert (mains[0].get("reasoning") or {}).get("effort") == "xhigh", (
                    json.dumps(mains[0].get("reasoning")) + "\n" + board.diag(tid))
                assert mains[0].get("model") == MODEL, mains[0].get("model")
            finally:
                board.kill_workers()

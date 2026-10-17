"""Bounded, typed control operations that never compete for a worker slot.

A supervised CTO process needs a handful of cheap, fixed operations (read the GitHub issue
queue, mint a status line, inventory its own runtime) while both heavy-worker slots are
occupied.  Every existing local execution path charges a worker slot (foreground terminal and
background/PTY alike), so the only alternatives were "raise the cap" or "exempt arbitrary
commands" -- both of which widen the unattended surface this boundary exists to bound.

Instead this tool runs ONE operation from a fixed table inside its own transient systemd
service, placed in the reviewed aggregate slice beside the workers:

* Fixed argv only.  No executable, argv, environment, cwd, timeout, URL, unit, shell or
  ``gh api`` parameter is accepted; unknown or operation-inapplicable keys are refused.
* The wrapper ``~/.local/bin/hermes-github`` (absolute, owner-owned, non-symlink, not
  group/other-writable) stays the sole owner of App credential minting.  There is no
  plain-``gh`` fallback.
* A clean environment is constructed rather than inherited: no tokens, askpass, aliases,
  shell startup, editor, proxy or provider settings.
* systemd owns single-flight (a same-name transient unit cannot start twice) and descendant
  teardown (``--service-type=exec`` + ``ExitType=main`` + ``KillMode=control-group`` +
  ``RuntimeMaxSec``), so there is no lock, lease, PID roster or name-based reaper here.
* A stdlib-only gate inside the service proves placement and limits, then hands the operation
  to the wrapper only after the parent grants one byte.  A refusal, a lost grant or an
  identity mismatch means nothing runs.

The operation never takes a ``ConcurrencyHardMax`` slot and never changes worker capacity.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import select
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from tools.autonomous_resources import (
    AutonomousResourceUnavailable,
    check_ancestors,
    check_autonomous_placement,
    live_contract,
    process_cgroup,
)
from tools.environments.base_output import _BoundedOutputCollector, _start_drain_thread
from tools.registry import registry, tool_error

TOOL_NAME = "bounded_control"
TOOLSET = "bounded_control"

#: One fixed unit name for every control operation: the same-name transient start refusal IS
#: the single-flight authority, so a concurrent launch loses without guessing.
CONTROL_UNIT = "hermes-control.service"
CONTROL_UNIT_STEM = "hermes-control"
MEMORY_BYTES = 512 * 1024 * 1024
TASKS_MAX = 32
RUNTIME_MAX_SECONDS = 45
TIMEOUT_START_SECONDS = 5
TIMEOUT_STOP_SECONDS = 5
#: Whole client budget: the service's own 45s deadline plus start/stop slack.
CLIENT_DEADLINE_SECONDS = 55
READINESS_SECONDS = 6
OUTPUT_CHARS = 20_000

_REPO_RE = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")
_STATES = ("open", "closed", "all")
_MAX_TITLE = 200
_MAX_BODY = 20_000
_MAX_LIMIT = 50
_DEFAULT_LIMIT = 20

_OPERATIONS: Dict[str, Mapping[str, frozenset]] = {
    "github_issue_list": {"required": frozenset({"repo"}), "optional": frozenset({"state", "limit"})},
    "github_issue_view": {"required": frozenset({"repo", "number"}), "optional": frozenset()},
    "github_issue_create": {"required": frozenset({"repo", "title"}), "optional": frozenset({"body"})},
    "github_issue_comment": {"required": frozenset({"repo", "number", "body"}), "optional": frozenset()},
    "github_status": {"required": frozenset(), "optional": frozenset()},
    "github_help": {"required": frozenset(), "optional": frozenset()},
    "runtime_diagnostics": {"required": frozenset(), "optional": frozenset()},
}

_COMMON_KEYS = frozenset({"operation"})

#: Runs INSIDE the transient service: announce identity, wait for the parent's grant, exec.
#: Stdlib only, no imports of this package (the point is that nothing else is trusted here).
_GATE_SOURCE = r'''
import json
import os
import sys

nonce = sys.argv[1]
argv = json.loads(sys.argv[2])
with open("/proc/self/stat", "rb") as handle:
    fields = handle.read().rsplit(b")", 1)[1].split()
identity = {"nonce": nonce, "pid": os.getpid(), "start_time": int(fields[19])}
sys.stdout.write(json.dumps(identity) + "\n")
sys.stdout.flush()
if sys.stdin.buffer.read(1) != b"1":
    sys.exit(41)
os.execve(argv[0], argv, dict(os.environ))
'''


class _Refuse(Exception):
    """A refusal: nothing ran, and the message says why in the operator's terms."""


# --------------------------------------------------------------------------
# Argument validation
# --------------------------------------------------------------------------

def _text(value: Any, field: str, *, limit: int, required: bool) -> str:
    if not isinstance(value, str):
        raise _Refuse(f"{field} must be a string")
    if "\x00" in value:
        raise _Refuse(f"{field} must not contain NUL bytes")
    trimmed = value.strip()
    if required and not trimmed:
        raise _Refuse(f"{field} must not be blank")
    if len(value) > limit:
        raise _Refuse(f"{field} must be at most {limit} characters")
    return value


def _count(value: Any, field: str, *, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _Refuse(f"{field} must be an integer")
    if value < 1 or value > maximum:
        raise _Refuse(f"{field} must be between 1 and {maximum}")
    return value


def _repo(value: Any) -> str:
    text = _text(value, "repo", limit=200, required=True)
    if not _REPO_RE.match(text):
        raise _Refuse("repo must be OWNER/NAME (letters, digits, '.', '_', '-')")
    if any(part.startswith("-") for part in text.split("/", 1)):
        raise _Refuse("repo components must not start with '-'")
    return text


def _validate(args: Mapping[str, Any]) -> Tuple[str, Dict[str, Any]]:
    if not isinstance(args, Mapping):
        raise _Refuse("arguments must be an object")
    operation = args.get("operation")
    if not isinstance(operation, str) or operation not in _OPERATIONS:
        raise _Refuse(
            "operation must be one of " + ", ".join(sorted(_OPERATIONS))
        )
    spec = _OPERATIONS[operation]
    allowed = set(_COMMON_KEYS) | set(spec["required"]) | set(spec["optional"])
    unknown = sorted(set(args) - allowed)
    if unknown:
        raise _Refuse(
            f"{operation}: parameter(s) not applicable: {', '.join(unknown)}. "
            f"Valid parameters: {', '.join(sorted(allowed))}. Nothing ran."
        )
    missing = sorted(spec["required"] - set(args))
    if missing:
        raise _Refuse(f"{operation}: missing required parameter(s): {', '.join(missing)}")

    clean: Dict[str, Any] = {"operation": operation}
    if "repo" in args:
        clean["repo"] = _repo(args["repo"])
    if "number" in args:
        clean["number"] = _count(args["number"], "number", maximum=1_000_000_000)
    if "limit" in args:
        clean["limit"] = _count(args["limit"], "limit", maximum=_MAX_LIMIT)
    else:
        clean["limit"] = _DEFAULT_LIMIT
    if "state" in args:
        state = args["state"]
        if state not in _STATES:
            raise _Refuse("state must be one of " + ", ".join(_STATES))
        clean["state"] = state
    else:
        clean["state"] = "open"
    if "title" in args:
        clean["title"] = _text(args["title"], "title", limit=_MAX_TITLE, required=True)
    if "body" in args:
        clean["body"] = _text(
            args["body"], "body", limit=_MAX_BODY,
            required=operation == "github_issue_comment",
        )
    elif operation == "github_issue_create":
        clean["body"] = ""
    return operation, clean


# --------------------------------------------------------------------------
# Fixed execution table
# --------------------------------------------------------------------------

def _wrapper() -> Path:
    """The App-credential wrapper, validated as an owner-owned regular executable."""
    path = Path.home() / ".local" / "bin" / "hermes-github"
    if not path.is_absolute():
        raise _Refuse(f"{path} is not absolute")
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise _Refuse(f"{path} is unavailable ({exc}); refusing to fall back to plain gh") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise _Refuse(f"{path} must be a regular file, never a symlink or special file")
    if metadata.st_uid != os.getuid():
        raise _Refuse(f"{path} is not owned by the current user")
    if metadata.st_mode & 0o022:
        raise _Refuse(f"{path} must not be group/other writable")
    if not os.access(path, os.X_OK):
        raise _Refuse(f"{path} is not executable")
    return path


def _operation_argv(operation: str, clean: Mapping[str, Any], wrapper: Path) -> List[str]:
    """The exact argv for *operation*; every element is fixed by the table."""
    w = str(wrapper)
    if operation == "github_issue_list":
        return [
            w, "gh", "issue", "list", "--repo", clean["repo"],
            "--state", clean["state"], "--limit", str(clean["limit"]),
            "--json", "number,title,state,updatedAt,labels,url",
        ]
    if operation == "github_issue_view":
        return [
            w, "gh", "issue", "view", str(clean["number"]), "--repo", clean["repo"],
            "--json", "number,title,state,body,labels,url",
        ]
    if operation == "github_issue_create":
        return [
            w, "gh", "issue", "create", "--repo", clean["repo"],
            "--title", clean["title"], "--body", clean["body"],
        ]
    if operation == "github_issue_comment":
        return [
            w, "gh", "issue", "comment", str(clean["number"]), "--repo", clean["repo"],
            "--body", clean["body"],
        ]
    if operation == "github_status":
        return [w, "status"]
    if operation == "github_help":
        return [w, "gh", "--help"]
    raise _Refuse(f"{operation}: no fixed execution")


def _payload_env(home: Path) -> Dict[str, str]:
    """The child's entire environment. Nothing inherited: no tokens, proxy or shell setup."""
    return {
        "HOME": str(home),
        "PATH": f"{home}/.local/bin:/usr/local/bin:/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "GH_PROMPT_DISABLED": "1",
        "GH_PAGER": "cat",
        "GIT_TERMINAL_PROMPT": "0",
    }


def _resolve(name: str) -> List[str]:
    from hermes_platform.resolver import locate_command

    command = list(locate_command(name).command or ())
    if not command:
        raise _Refuse(f"{name} could not be resolved on this host")
    return command


#: ``check_ancestors`` reports kernel limits under the controller's own names.
_EFFECTIVE_KEYS = {
    "MemoryHigh": "memory.high",
    "MemoryMax": "memory.max",
    "MemorySwapMax": "memory.swap.max",
}


def _memory_bounds(workers_effective: Mapping[str, Any]) -> Tuple[int, int]:
    """512MiB, clamped down to whatever finite bound the worker slice already carries."""
    bound = MEMORY_BYTES
    for systemd_key in ("MemoryHigh", "MemoryMax"):
        try:
            numeric = int(workers_effective.get(_EFFECTIVE_KEYS[systemd_key]))
        except (TypeError, ValueError):
            continue
        bound = min(bound, numeric)
    return bound, bound


def _service_argv(
    payload: List[str], *, slice_unit: str, memory_high: int, memory_max: int, home: Path,
) -> List[str]:
    return [
        *_resolve("systemd-run"),
        "--user", "--quiet", "--pipe", "--wait", "--collect",
        "--service-type=exec", "--expand-environment=no",
        f"--unit={CONTROL_UNIT}", f"--slice={slice_unit}",
        "-p", "ExitType=main",
        "-p", "MemoryAccounting=yes",
        "-p", f"MemoryHigh={memory_high}",
        "-p", f"MemoryMax={memory_max}",
        "-p", "MemorySwapMax=0",
        "-p", f"TasksMax={TASKS_MAX}",
        "-p", f"RuntimeMaxSec={RUNTIME_MAX_SECONDS}",
        "-p", f"TimeoutStartSec={TIMEOUT_START_SECONDS}",
        "-p", f"TimeoutStopSec={TIMEOUT_STOP_SECONDS}",
        "-p", "KillMode=control-group",
        "-p", f"WorkingDirectory={home}",
        "--",
        *payload,
    ]


# --------------------------------------------------------------------------
# Launch: readiness gate, placement proof, then the grant
# --------------------------------------------------------------------------

def _systemctl(argv: List[str]) -> Optional[subprocess.CompletedProcess]:
    from tools.process_registry import systemd_user_bus_env

    try:
        return subprocess.run(
            argv, capture_output=True, text=True, timeout=10, stdin=subprocess.DEVNULL,
            env=systemd_user_bus_env(),
        )
    except (OSError, subprocess.SubprocessError):
        return None


def _unit_properties() -> Optional[Dict[str, str]]:
    """One read of the control service's own properties, or None when unreadable."""
    completed = _systemctl([
        *_resolve("systemctl"), "--user", "show", CONTROL_UNIT,
        "--property=LoadState", "--property=ActiveState", "--property=MainPID",
        "--property=InvocationID", "--property=ControlGroup",
        "--property=MemoryHigh", "--property=MemoryMax", "--property=MemorySwapMax",
    ])
    if completed is None or completed.returncode != 0:
        return None
    return dict(
        line.split("=", 1) for line in completed.stdout.splitlines() if "=" in line
    )


def _read_line(stream, timeout: float) -> Optional[str]:
    """Read the readiness line within *timeout*; None on EOF/timeout/oversize.

    Byte at a time on purpose: ``systemd-run --pipe`` is a shared stream, so a bulk read
    would swallow the beginning of the payload's own output. The announcement is capped at
    1KiB, so dropping an oversized one is a refusal, not a partial parse.
    """
    deadline = time.monotonic() + timeout
    buffered = b""
    while True:
        if len(buffered) > 1024:
            return None
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        try:
            ready, _, _ = select.select([stream], [], [], remaining)
        except (OSError, ValueError):
            return None
        if not ready:
            return None
        chunk = os.read(stream.fileno(), 1)
        if not chunk:
            return None
        if chunk == b"\n":
            return buffered.decode("utf-8", "replace")
        buffered += chunk


def _verify_launch(
    readiness: Mapping[str, Any], *, nonce: str, placement: Mapping[str, Any],
    memory_high: int, memory_max: int,
) -> Optional[str]:
    """Return the refusal reason, or None when the service is exactly where it must be."""
    if readiness.get("nonce") != nonce:
        return "readiness nonce does not match this launch"
    pid = readiness.get("pid")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return "readiness did not report a usable PID"
    from hermes_cli.kanban_runtime import process_start_time

    try:
        actual_start = process_start_time(pid)
    except Exception as exc:  # noqa: BLE001 - unreadable identity is a refusal
        return f"service PID {pid} start time is unreadable ({exc})"
    if int(readiness.get("start_time") or 0) != int(actual_start):
        return f"service PID {pid} start time changed after readiness"

    properties = _unit_properties()
    if properties is None:
        return f"{CONTROL_UNIT} properties are unreadable"
    if properties.get("LoadState") != "loaded":
        return f"{CONTROL_UNIT} is not loaded ({properties.get('LoadState')!r})"
    if properties.get("MainPID") != str(pid):
        return f"{CONTROL_UNIT} MainPID is {properties.get('MainPID')!r}, not {pid}"
    if not properties.get("InvocationID"):
        return f"{CONTROL_UNIT} has no InvocationID"

    expected_group = f"{placement['aggregate']['cgroup']}/{CONTROL_UNIT}"
    if properties.get("ControlGroup") != expected_group:
        return (
            f"{CONTROL_UNIT} runs in {properties.get('ControlGroup')!r}, "
            f"not directly in {expected_group!r}"
        )
    for key, expected in (("MemoryHigh", memory_high), ("MemoryMax", memory_max),
                          ("MemorySwapMax", 0)):
        if properties.get(key) != str(expected):
            return f"{CONTROL_UNIT} {key} is {properties.get(key)!r}, not {expected}"
    try:
        effective = check_ancestors(
            Path("/sys/fs/cgroup"), expected_group,
            {"MemoryHigh": memory_high, "MemoryMax": memory_max, "MemorySwapMax": 0},
        )["effective_limits"]
    except Exception as exc:  # noqa: BLE001 - unreadable limits are a refusal, not a crash
        return f"effective limits for {expected_group} are unreadable ({exc})"
    for key, expected in (("MemoryHigh", memory_high), ("MemoryMax", memory_max),
                          ("MemorySwapMax", 0)):
        actual = effective.get(_EFFECTIVE_KEYS[key])
        if actual is None or int(actual) > expected:
            return (
                f"{CONTROL_UNIT} effective {key} {actual!r} does not fit the "
                f"requested {expected}"
            )
    return None


def _runtime_fields() -> Dict[str, Any]:
    """Code/generation identity of the running runtime. Never environment or secrets."""
    generation = os.environ.get("HERMES_KANBAN_RUNTIME_GENERATION")
    fields: Dict[str, Any] = {"module_root": str(Path(__file__).resolve().parent.parent)}
    if generation:
        try:
            from hermes_cli.kanban_runtime_generation import generation_manifest

            identity = generation_manifest(Path(generation), verify=False)["identity"]
            fields["code"] = identity.get("code_sha")
            fields["generation"] = identity.get("generation")
            return fields
        except Exception:  # noqa: BLE001 - identity is evidence, never a launch gate
            fields["generation"] = generation
    try:
        from hermes_cli.kanban_runtime import _git_sha

        fields["code"] = _git_sha(Path(fields["module_root"]))
    except Exception:  # noqa: BLE001
        fields["code"] = None
    return fields


def _run_operation(operation: str, clean: Mapping[str, Any]) -> Dict[str, Any]:
    wrapper = _wrapper()
    home = Path.home()
    try:
        placement = check_autonomous_placement(pid=os.getpid())
    except AutonomousResourceUnavailable as exc:
        raise _Refuse(
            "bounded control requires a supervised process inside the autonomous aggregate "
            f"({exc}). An interactive session must not fall back to its own cgroup."
        ) from exc
    memory_high, memory_max = _memory_bounds(placement["workers_effective_limits"])
    nonce = secrets.token_hex(16)
    argv = _operation_argv(operation, clean, wrapper)
    payload = [
        *_resolve("env"), "-i",
        *(f"{key}={value}" for key, value in _payload_env(home).items()),
        sys.executable, "-I", "-c", _GATE_SOURCE, nonce, json.dumps(argv),
    ]
    service = _service_argv(
        payload, slice_unit=live_contract().aggregate_unit,
        memory_high=memory_high, memory_max=memory_max, home=home,
    )
    from tools.process_registry import systemd_user_bus_env

    try:
        proc = subprocess.Popen(
            service, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=systemd_user_bus_env(),
        )
    except OSError as exc:
        raise _Refuse(f"{CONTROL_UNIT} could not be started ({exc})") from exc

    deadline = time.monotonic() + CLIENT_DEADLINE_SECONDS
    output = _BoundedOutputCollector(OUTPUT_CHARS, spill_path=None)
    granted = False
    refusal = None
    try:
        line = _read_line(proc.stdout, READINESS_SECONDS)
        if line is None:
            refusal = "the service never announced readiness"
        else:
            try:
                readiness = json.loads(line)
            except ValueError:
                refusal = "the service announcement was not JSON"
            else:
                refusal = _verify_launch(
                    readiness, nonce=nonce, placement=placement,
                    memory_high=memory_high, memory_max=memory_max,
                )
                if refusal is None:
                    proc.stdin.write(b"1")
                    proc.stdin.flush()
                    granted = True
    finally:
        try:
            proc.stdin.close()
        except OSError:
            pass

    drain = _start_drain_thread(proc, output)
    errors: List[bytes] = []

    def _collect_stderr() -> None:
        try:
            while len(b"".join(errors)) < 8192:
                chunk = proc.stderr.read(1024) if proc.stderr is not None else b""
                if not chunk:
                    return
                errors.append(chunk)
        except (OSError, ValueError):
            return

    err_thread = threading.Thread(target=_collect_stderr, daemon=True)
    err_thread.start()
    remaining = max(1.0, deadline - time.monotonic())
    cleanup_complete = True
    try:
        exit_code: Optional[int] = proc.wait(timeout=remaining)
    except subprocess.TimeoutExpired:
        exit_code = None
        cleanup_complete = False
    drain.join(timeout=5)
    err_thread.join(timeout=2)
    stderr = b"".join(errors)
    text = output.render()
    truncated = getattr(output, "_total_chars", 0) > output.max_chars
    result: Dict[str, Any] = {
        "ok": exit_code == 0 and refusal is None,
        "operation": operation,
        "unit": CONTROL_UNIT,
        "exit_code": exit_code,
        "output": text,
        "truncated": truncated,
        "granted": granted,
        "cleanup_complete": cleanup_complete,
        "runtime": _runtime_fields(),
    }
    if refusal is not None:
        result["error"] = f"refused before execution: {refusal}"
    elif exit_code is None:
        result["error"] = (
            f"{CONTROL_UNIT} did not finish within {CLIENT_DEADLINE_SECONDS}s; "
            "its own deadline bounds the remaining work"
        )
    elif exit_code != 0:
        result["error"] = (stderr.strip().decode("utf-8", "replace")
                           or text.strip()[-600:] or f"exit code {exit_code}")
    return result


# --------------------------------------------------------------------------
# Read-only diagnostics
# --------------------------------------------------------------------------

def _diagnostics() -> Dict[str, Any]:
    """Inventory the boundary and the control service; never mutate, never dispatch."""
    payload: Dict[str, Any] = {
        "operation": "runtime_diagnostics",
        "unit": None,
        "caller": {"pid": os.getpid()},
        "reads": {},
    }
    try:
        payload["caller"]["cgroup"] = process_cgroup(Path("/proc"), os.getpid())
    except Exception as exc:  # noqa: BLE001 - per-read error, never a hard failure
        payload["reads"]["caller_cgroup"] = str(exc)
    try:
        placement = check_autonomous_placement(pid=os.getpid())
        payload["caller"]["placement_ok"] = True
        payload["caller"]["cgroup"] = placement.get("launcher_cgroup")
        payload["aggregate"] = placement["aggregate"]
        payload["worker_slot_occupied"] = placement["worker_slot_occupied"]
        payload["concurrency_hard_max"] = placement["concurrency_hard_max"]
        payload["worker_scopes"] = list(placement["worker_scopes"])[:10]
        payload["effective_limits"] = {
            "aggregate": placement["aggregate_effective_limits"],
            "workers": placement["workers_effective_limits"],
        }
    except Exception as exc:  # noqa: BLE001
        payload["caller"]["placement_ok"] = False
        payload["reads"]["placement"] = str(exc)
        try:
            inventory = check_autonomous_placement(pid=None)
            payload["aggregate"] = inventory["aggregate"]
            payload["worker_slot_occupied"] = inventory["worker_slot_occupied"]
            payload["concurrency_hard_max"] = inventory["concurrency_hard_max"]
            payload["worker_scopes"] = list(inventory["worker_scopes"])[:10]
        except Exception as inventory_error:  # noqa: BLE001
            payload["reads"]["inventory"] = str(inventory_error)

    properties = _unit_properties()
    payload["control_service"] = properties
    if properties is None:
        payload["reads"]["control_service"] = f"{CONTROL_UNIT} properties are unreadable"
    payload["limits"] = {
        "memory_high": MEMORY_BYTES, "memory_max": MEMORY_BYTES, "memory_swap_max": 0,
        "tasks_max": TASKS_MAX, "runtime_max_seconds": RUNTIME_MAX_SECONDS,
    }
    payload["runtime"] = _runtime_fields()
    payload["next"] = (
        "Start/repair/deployment actions stay with the existing #57 workflow and the operator; "
        "this inventory neither dispatches, stops workers nor unblocks cards."
    )
    rendered = json.dumps(payload, ensure_ascii=False)
    if len(rendered) > OUTPUT_CHARS:
        rendered = rendered[:OUTPUT_CHARS]
    return {
        **payload,
        "ok": True,
        "exit_code": None,
        "output": rendered,
        "truncated": len(rendered) >= OUTPUT_CHARS,
        "granted": False,
        "cleanup_complete": True,
    }


# --------------------------------------------------------------------------
# Tool surface
# --------------------------------------------------------------------------

BOUNDED_CONTROL_SCHEMA = {
    "name": TOOL_NAME,
    "description": (
        "Run one bounded, typed control operation from a fixed table in its own transient "
        "systemd service inside the autonomous boundary. It never consumes a heavy-worker "
        "slot (workers stay capped), never runs arbitrary commands, and never accepts an "
        "executable, argv, environment, cwd, timeout, URL, unit or shell. Use it for the "
        "bounded GitHub issue read/comment operations and for read-only runtime inventory "
        "when both worker slots are busy; everything else still goes through the normal "
        "approval path. A refusal never falls back to terminal or a worker slot."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": sorted(_OPERATIONS),
                "description": (
                    "Fixed operation. github_issue_list/view/status/help are reads; "
                    "github_issue_create/comment mutate GitHub and still pass the normal "
                    "tool-approval hooks; runtime_diagnostics returns a read-only inventory."
                ),
            },
            "repo": {
                "type": "string",
                "description": "OWNER/NAME for the issue operations.",
            },
            "number": {
                "type": "integer",
                "description": "Issue number, for view/comment.",
            },
            "state": {
                "type": "string",
                "enum": list(_STATES),
                "description": "Issue state filter for github_issue_list (default open).",
            },
            "limit": {
                "type": "integer",
                "description": f"Issue page size for github_issue_list, 1-{_MAX_LIMIT} (default {_DEFAULT_LIMIT}).",
            },
            "title": {
                "type": "string",
                "description": f"Issue title for github_issue_create (nonblank, <= {_MAX_TITLE} characters).",
            },
            "body": {
                "type": "string",
                "description": f"Issue/comment body (<= {_MAX_BODY} characters; required for comment).",
            },
        },
        "required": ["operation"],
        "additionalProperties": False,
    },
}


def _handle(args: Mapping[str, Any], **_kwargs) -> str:
    try:
        from tools.kanban_tools import (
            _reject_delegated_child_mutation,
            _require_orchestrator_tool,
        )

        _reject_delegated_child_mutation(TOOL_NAME)
        _require_orchestrator_tool(TOOL_NAME)
        if sys.platform != "linux":
            raise _Refuse("bounded control needs Linux cgroups and the user systemd manager")
        operation, clean = _validate(args)
        if operation == "runtime_diagnostics":
            return json.dumps(_diagnostics(), ensure_ascii=False)
        return json.dumps(_run_operation(operation, clean), ensure_ascii=False)
    except _Refuse as exc:
        return tool_error(f"{TOOL_NAME}: {exc}")
    except Exception as exc:  # noqa: BLE001 - tool results never raise
        return tool_error(f"{TOOL_NAME}: {exc}")


registry.register(
    name=TOOL_NAME,
    toolset=TOOLSET,
    schema=BOUNDED_CONTROL_SCHEMA,
    handler=_handle,
    emoji="🎛",
)

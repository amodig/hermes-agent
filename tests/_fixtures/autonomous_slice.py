"""Disposable transient systemd slices for tests that need a REAL boundary.

The production slices -- ``autonomous.slice`` and ``autonomous-workers.slice`` --
are owned by chezmoi. A test must never create, start, stop or otherwise touch
them, nor the interactive ``agents*`` slices, the gateways, or the desktop units.

These helpers build uniquely named TRANSIENT slices through the same systemd
manager API the production units are created with, hand the runtime a private
contract pointing at them, and tear down exactly the units they created. The
concurrency behaviour is the real one: ``ConcurrencyHardMax`` on the fixture
workers slice is enforced by the same systemd code path as production, which is
what makes a fixture result evidence rather than a simulation.

Requires Linux, cgroup v2 and a reachable user bus; callers skip otherwise.
Callers must also carry ``@pytest.mark.live_system_guard_bypass``: creating
systemd units is deliberately outside the default test guard.
"""
from __future__ import annotations

import contextlib
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

#: Default fixture budgets, for tests whose payloads are explicit sleepers or
#: bounded allocators. Large enough that systemd reports them back unchanged,
#: small enough that nothing here can pressure the host.
HIGH_BYTES = 64 * 1024 ** 2
MAX_BYTES = 128 * 1024 ** 2
SWAP_BYTES = 0

#: Budgets for tests that boot a REAL Hermes worker into the fixture slice: the
#: production memory shape, because a cold Hermes boot does not fit (and would be
#: throttled or killed) inside the sleeper-sized defaults. These are ceilings, not
#: reservations -- the fixture only ever runs the one worker the test drives.
HERMES_BOOT_HIGH_BYTES = 2 * 1024 ** 3
HERMES_BOOT_MAX_BYTES = 3 * 1024 ** 3
HERMES_BOOT_SWAP_BYTES = 0

CGROUP_ROOT = Path("/sys/fs/cgroup")

UNIT_PROPERTIES = (
    "LoadState", "ActiveState", "ControlGroup",
    "MemoryHigh", "MemoryMax", "MemorySwapMax",
    "ConcurrencyHardMax", "ConcurrencySoftMax",
)


def supports_concurrency_hard_max() -> bool:
    """Whether this systemd manager knows the property at all (>= 258).

    Read from the root slice rather than from a fixture unit, so the answer does not
    depend on anything this fixture creates. The ``--`` is required: the unit name
    starts with a dash.
    """
    completed = _run(
        # ``--property`` must precede the ``--``: the unit name starts with a dash,
        # so the separator has to come last.
        ["systemctl", "--user", "show", "--property=LoadState,ConcurrencyHardMax",
         "--", "-.slice"],
    )
    properties = dict(
        line.split("=", 1) for line in completed.stdout.splitlines() if "=" in line
    )
    return properties.get("LoadState") == "loaded" and bool(properties.get("ConcurrencyHardMax"))


def require_user_bus() -> None:
    """Skip (never pass) when the real-kernel fixture cannot run here."""
    from tools import process_registry

    if not process_registry._IS_LINUX:
        pytest.skip("disposable systemd slices require Linux")
    if not process_registry._systemd_run_user_scope_available():
        pytest.skip("systemd-run --user --scope is unavailable on this host")
    if not supports_concurrency_hard_max():
        pytest.skip(
            "this systemd does not support ConcurrencyHardMax (needs >= 258), so the "
            "native slot cannot be created here"
        )


def require_free_boundary() -> None:
    """Skip unless the host presents a FREE autonomous boundary.

    Autonomous dispatch refuses unless the caller is inside a deployed aggregate
    whose single worker slot is free -- on CI, and on any undeployed or held host,
    that is the product working correctly, not a test failure. Scenarios that drive
    a real dispatcher (or a manual gateway with an embedded one) therefore state the
    precondition instead of asserting against a boundary the host does not have.
    """
    if sys.platform != "linux":
        pytest.skip("the autonomous boundary is Linux-only")
    properties = {}
    try:
        completed = _run([
            "systemctl", "--user", "show", "autonomous-workers.slice",
            "--property=LoadState,ConcurrencyHardMax",
        ])
    except OSError:
        pass
    else:
        properties = dict(
            line.split("=", 1) for line in completed.stdout.splitlines() if "=" in line
        )
    if properties.get("LoadState") != "loaded" or properties.get("ConcurrencyHardMax") != "1":
        pytest.skip(
            "no free autonomous boundary on this host (LoadState="
            f"{properties.get('LoadState')!r}, ConcurrencyHardMax="
            f"{properties.get('ConcurrencyHardMax')!r}); place this scenario inside "
            "`systemd-run --user --scope --slice=autonomous.slice ...` before enabling it"
        )


def user_bus_env() -> Dict[str, str]:
    """Environment reaching the user manager, derived exactly as the runtime does."""
    from tools import process_registry

    return process_registry.systemd_user_bus_env()


def _run(argv: List[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(
        argv, capture_output=True, text=True, timeout=60, env=user_bus_env(), **kwargs,
    )


def _start_transient_slice(name: str, properties: List[Tuple[str, str, str]]) -> None:
    """``StartTransientUnit`` for a slice, with D-Bus-correct property types.

    ``MemoryHigh``/``MemoryMax``/``MemorySwapMax`` are ``t`` (uint64);
    ``ConcurrencyHardMax`` is ``u`` (uint32). Sending the wrong signature is the
    difference between a slice that exists and a misleading "unexpected message
    contents", so the caller states the signature explicitly.
    """
    argv = [
        "busctl", "--user", "call", "org.freedesktop.systemd1",
        "/org/freedesktop/systemd1", "org.freedesktop.systemd1.Manager",
        "StartTransientUnit", "ssa(sv)a(sa(sv))", name, "replace", str(len(properties)),
    ]
    for key, signature, value in properties:
        argv += [key, signature, value]
    argv.append("0")
    completed = _run(argv)
    assert completed.returncode == 0, (
        f"StartTransientUnit {name} failed: {completed.stderr.strip()}"
    )


def show(unit: str) -> Dict[str, str]:
    completed = _run([
        "systemctl", "--user", "show", unit, "--property=" + ",".join(UNIT_PROPERTIES),
    ])
    assert completed.returncode == 0, f"systemctl show {unit} failed: {completed.stderr.strip()}"
    return dict(
        line.split("=", 1) for line in completed.stdout.splitlines() if "=" in line
    )


def cgroup_of(pid: int) -> str:
    for line in Path(f"/proc/{pid}/cgroup").read_text(encoding="utf-8").splitlines():
        if line.startswith("0::"):
            return line[3:]
    raise AssertionError(f"PID {pid} has no unified cgroup")


@dataclass(frozen=True)
class DisposableBoundary:
    """A created fixture topology. Always torn down by ``disposable_boundary``."""

    token: str
    aggregate_unit: str
    workers_unit: str
    workers_concurrency_hard_max: int = 1
    high_bytes: int = HIGH_BYTES
    max_bytes: int = MAX_BYTES
    swap_bytes: int = SWAP_BYTES

    def contract_kwargs(self) -> Dict[str, object]:
        limits = {
            "MemoryHigh": self.high_bytes,
            "MemoryMax": self.max_bytes,
            "MemorySwapMax": self.swap_bytes,
        }
        return {
            "aggregate_unit": self.aggregate_unit,
            "workers_unit": self.workers_unit,
            "aggregate_limits": dict(limits),
            "worker_limits": dict(limits),
            "gateway_limits": dict(limits),
            "workers_concurrency_hard_max": self.workers_concurrency_hard_max,
            # The fixture has no gateway unit, and the HOST's real gateways belong
            # to the host's topology, not to this one: enumerating them would assert
            # host placement from a fixture run.
            "gateway_unit_prefix": f"fixture-gateway-{self.token}",
        }

    def contract(self):
        from tools.autonomous_resources import _Contract

        return _Contract(**self.contract_kwargs())

    def install(self, monkeypatch) -> None:
        """Point the runtime's live contract at this fixture for THIS test.

        Goes through monkeypatch rather than assigning the module attribute: a
        permanently retargeted ``_LIVE_CONTRACT`` would silently retarget every
        later test in the same process, so a fixture slice name would leak into an
        unrelated launch and the real contract would never be exercised again.
        """
        from tools import autonomous_resources

        monkeypatch.setattr(autonomous_resources, "_LIVE_CONTRACT", self.contract())

    @property
    def aggregate_group(self) -> str:
        return show(self.aggregate_unit)["ControlGroup"]

    @property
    def workers_group(self) -> str:
        return show(self.workers_unit)["ControlGroup"]

    def launcher_argv(self, argv: List[str], *, unit_suffix: str) -> List[str]:
        """Run *argv* inside the fixture aggregate, as a real systemd scope."""
        unit = f"fixture-launcher-{self.token}-{unit_suffix}.scope"
        return [
            "systemd-run", "--user", "--scope", "--quiet", "--collect",
            f"--slice={self.aggregate_unit}", "--unit", unit, "--expand-environment=no",
            "--", *argv,
        ]

    def launch(self, argv: List[str], **kwargs) -> subprocess.Popen:
        return subprocess.Popen(
            self.launcher_argv(argv, unit_suffix=uuid.uuid4().hex[:8]),
            start_new_session=True, env=user_bus_env(), **kwargs,
        )

    def _worker_scope_units(self) -> List[str]:
        """Worker scopes currently living directly inside the fixture worker slice."""
        try:
            group = self.workers_group
        except Exception:
            return []
        directory = CGROUP_ROOT / group.lstrip("/")
        if not directory.is_dir():
            return []
        return [child.name for child in sorted(directory.glob("hermes-worker-*.scope"))]

    def stop_unit(self, unit: str) -> subprocess.CompletedProcess:
        return _run(["systemctl", "--user", "stop", unit])

    def tear_down(self) -> None:
        """Remove exactly this fixture's units, innermost first.

        Worker scopes are stopped before their slices: a slice stop can be refused
        while a transient scope inside it is busy, and in this suite a leftover
        scope is not merely untidy -- the next run's validator would see a live
        `hermes-worker-*.scope` in someone else's slice and fail as if a worker had
        escaped the reviewed boundary.
        """
        for scope in self._worker_scope_units():
            self.stop_unit(scope)
        self.stop_unit(self.workers_unit)
        self.stop_unit(self.aggregate_unit)


def verify(boundary: DisposableBoundary) -> None:
    """The kernel must really carry what the fixture asked for."""
    expected = (
        (boundary.aggregate_unit, str(boundary.high_bytes), str(boundary.max_bytes),
         str(boundary.swap_bytes), None),
        (boundary.workers_unit, str(boundary.high_bytes), str(boundary.max_bytes),
         str(boundary.swap_bytes), str(boundary.workers_concurrency_hard_max)),
    )
    for unit, high, maximum, swap, hard_max in expected:
        properties = show(unit)
        assert properties.get("LoadState") == "loaded", (unit, properties)
        assert properties.get("MemoryHigh") == high, (unit, properties)
        assert properties.get("MemoryMax") == maximum, (unit, properties)
        assert properties.get("MemorySwapMax") == swap, (unit, properties)
        assert properties.get("ControlGroup"), (unit, properties)
        if hard_max is not None:
            assert properties.get("ConcurrencyHardMax") == hard_max, (unit, properties)
    workers_group = show(boundary.workers_unit)["ControlGroup"]
    aggregate_group = show(boundary.aggregate_unit)["ControlGroup"]
    assert workers_group == f"{aggregate_group}/{boundary.workers_unit}", (
        f"{boundary.workers_unit} is not nested inside {boundary.aggregate_unit}"
    )


def create_boundary(
    *, concurrency_hard_max: int = 1,
    high_bytes: int = HIGH_BYTES, max_bytes: int = MAX_BYTES, swap_bytes: int = SWAP_BYTES,
) -> DisposableBoundary:
    """Create the fixture topology; on any failure the units are torn down first."""
    token = uuid.uuid4().hex[:10]
    boundary = DisposableBoundary(
        # A dash here is what makes systemd nest `<agg>-workers.slice` inside
        # `<agg>.slice`; the token itself is hex so it adds no nesting of its own.
        token=token,
        aggregate_unit=f"fixturea{token}.slice",
        workers_unit=f"fixturea{token}-workers.slice",
        workers_concurrency_hard_max=concurrency_hard_max,
        high_bytes=high_bytes,
        max_bytes=max_bytes,
        swap_bytes=swap_bytes,
    )
    try:
        _start_transient_slice(boundary.aggregate_unit, [
            ("MemoryHigh", "t", str(high_bytes)),
            ("MemoryMax", "t", str(max_bytes)),
            ("MemorySwapMax", "t", str(swap_bytes)),
        ])
        _start_transient_slice(boundary.workers_unit, [
            ("MemoryHigh", "t", str(high_bytes)),
            ("MemoryMax", "t", str(max_bytes)),
            ("MemorySwapMax", "t", str(swap_bytes)),
            ("ConcurrencyHardMax", "u", str(concurrency_hard_max)),
        ])
        verify(boundary)
    except BaseException:
        boundary.tear_down()
        raise
    return boundary


@contextlib.contextmanager
def disposable_boundary(
    *, concurrency_hard_max: int = 1,
    high_bytes: int = HIGH_BYTES, max_bytes: int = MAX_BYTES, swap_bytes: int = SWAP_BYTES,
):
    """Yield a created fixture topology and always remove exactly its own units."""
    boundary = create_boundary(
        concurrency_hard_max=concurrency_hard_max,
        high_bytes=high_bytes, max_bytes=max_bytes, swap_bytes=swap_bytes,
    )
    try:
        yield boundary
    finally:
        boundary.tear_down()
        # The fixture must not leave a live worker scope behind for the next test
        # (or for the host) to find.
        state = show(boundary.workers_unit).get("ActiveState")
        assert state in ("inactive", "failed", "not-found"), (
            f"fixture slice {boundary.workers_unit} survived teardown: {state}"
        )
        leftover = boundary._worker_scope_units()
        assert not leftover, f"fixture worker scopes survived teardown: {leftover}"


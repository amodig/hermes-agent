#!/usr/bin/env python3
"""Host boundary that keeps autonomous Hermes work out of the interactive cgroup.

An interactive OMP session and its agent controls share one memory cgroup, so a
runaway autonomous worker inside that cgroup can OOM the editor and the desktop
app next to it (amodig/hermes-cto#50).  Autonomy therefore gets its own
aggregate, placed as a *sibling* of the interactive slices directly under the
user manager, and exactly one worker may run in it at a time.

This module is the single decision point:

* :func:`check_autonomous_boundary` -- launch admission.  Every autonomous
  launch calls it (with its own PID) before expensive preparation, and again
  immediately before it hands the worker its grant.
* :func:`check_autonomous_worker` -- post-start verification that a PID really
  landed in its own worker scope inside that aggregate.

Both read real systemd properties and real cgroup membership; neither trusts
``Slice=``, a cached capability probe, or a unit-file declaration.  The slot
itself is enforced by systemd (``ConcurrencyHardMax=1``), not here: this module
refuses a launch when the slot is already busy, but the atomic authority that
makes two racing launchers resolve to exactly one worker is the kernel-level
concurrency limit on the slice.

Failure policy is fail-closed.  A missing unit, a pending daemon reload, an
unsupported property, an unreadable cgroup, or incomplete cgroup visibility all
refuse the launch rather than fall back to the interactive slice.  There is no
hostname check and no environment variable that disables the boundary.

This module is deliberately stdlib-only, because hermes-cto's read-only host
verifier imports it by file path without booting Hermes.  It must not import
``tools.process_registry`` or any other runtime package.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional

GIB = 1024 ** 3
PROC_ROOT = Path("/proc")
CGROUP_ROOT = Path("/sys/fs/cgroup")

LIMITS = ("MemoryHigh", "MemoryMax", "MemorySwapMax")
CGROUP_LIMITS = ("memory.high", "memory.max", "memory.swap.max")
UNLIMITED = 4294967295  # systemd's "no limit" for uint32 unit properties

#: Aggregate holding every autonomous Hermes process.  Sibling of the
#: interactive slices, directly beneath the user manager.
AUTONOMOUS_SLICE = "autonomous.slice"
#: Worker sub-slice.  Nested inside the aggregate on purpose: its only bounded
#: parent is autonomy, never an interactive slice.  systemd nests ``foo-bar.slice``
#: under ``foo.slice`` from the name alone.
AUTONOMOUS_WORKERS_SLICE = "autonomous-workers.slice"

#: Gateway and worker slice both carry the full aggregate budget; the effective
#: bound is the minimum along the tree, so the workers slice cannot widen it.
AUTONOMOUS_LIMITS = {"MemoryHigh": 2 * GIB, "MemoryMax": 3 * GIB, "MemorySwapMax": 1 * GIB}
WORKER_SLICE_LIMITS = dict(AUTONOMOUS_LIMITS)
GATEWAY_LIMITS = dict(AUTONOMOUS_LIMITS)
#: One autonomous worker at a time.  Soft stays unlimited so a second launch
#: fails immediately instead of queueing behind a claimed board card.
WORKERS_CONCURRENCY_HARD_MAX = 1
WORKERS_CONCURRENCY_SOFT_MAX = UNLIMITED
#: Human-readable form of the soft-limit requirement, for diagnostics.
SOFT_MAX_UNLIMITED_LABEL = f"unlimited ({WORKERS_CONCURRENCY_SOFT_MAX})"

GATEWAY_UNIT_PREFIX = "hermes-gateway"
GATEWAY_UNIT_SUFFIX = ".service"
WORKER_SCOPE_PREFIX = "hermes-worker-"
WORKER_SCOPE_SUFFIX = ".scope"
GATEWAY_OOMPOLICY = "kill"

UNIT_PROPERTIES = (
    *LIMITS,
    "LoadState",
    "ActiveState",
    "NeedDaemonReload",
    "Slice",
    "ControlGroup",
    "MainPID",
    "OOMPolicy",
    "ConcurrencyHardMax",
    "ConcurrencySoftMax",
    "ManagedOOMMemoryPressure",
    "ManagedOOMSwap",
)
OOM_POLICY_PROPERTIES = ("ManagedOOMMemoryPressure", "ManagedOOMSwap")
#: ``auto`` is systemd's "unset"; only an explicit/effective ``kill`` is policy.
OOM_KILL = "kill"
#: The only value that takes an ancestor out of systemd-oomd's reach.
OOM_CONTINUE = "continue"

BOUNDARY_DIAGNOSTIC = "autonomous resource boundary unavailable"
SLOT_DIAGNOSTIC = "autonomous worker slot occupied"


class AutonomousResourceUnavailable(RuntimeError):
    """The host does not present the reviewed autonomous boundary.

    Infrastructure, never a property of the work being launched: callers that
    keep per-job retry budgets must not charge it (a card parked ``blocked``
    because the host policy drifted is not a task failure).
    """


class AutonomousWorkerBusy(AutonomousResourceUnavailable):
    """The single autonomous worker slot is already occupied.

    A hold, not a fault: the surviving worker keeps running, and the next
    launch succeeds once the whole scope has emptied.
    """


def _fail(detail: str, *, slot: bool = False) -> None:
    prefix = SLOT_DIAGNOSTIC if slot else BOUNDARY_DIAGNOSTIC
    raise (AutonomousWorkerBusy if slot else AutonomousResourceUnavailable)(
        f"{prefix}: {detail}"
    )


def _require(condition, detail: str, *, slot: bool = False) -> None:
    if not condition:
        _fail(detail, slot=slot)


# --------------------------------------------------------------------------
# Generic cgroup readers (also imported by hermes-cto's read-only host verifier)
# --------------------------------------------------------------------------

def process_cgroup(proc, pid) -> str:
    """Unified (cgroup v2) cgroup path of *pid*, read from its ``/proc`` entry.

    Raises ``FileNotFoundError`` when the PID has exited: callers walking a
    live tree treat that as a short-lived descendant, not as a policy failure.
    """
    for line in (Path(proc) / str(pid) / "cgroup").read_text(encoding="utf-8").splitlines():
        if line.startswith("0::"):
            return line[3:]
    _fail(f"PID {pid}: unified cgroup missing")


def contained(path: str, group: str) -> bool:
    """Path-boundary containment; a bare prefix sibling (``foo.slice`` under
    ``foo``) is not a child."""
    _require(path.startswith("/") and group.startswith("/"), "invalid cgroup path")
    group = group.rstrip("/")
    return path == group or path.startswith(group + "/")


def check_ancestors(root, group: str, expected: Mapping[str, object]) -> dict:
    """Walk *group*'s ancestors to the host root, reporting every limit.

    The innermost cgroup's kernel limits must equal *expected* exactly, so a
    systemd property that never reached the kernel is a failure, and every
    finite ancestor is reported so the caller can see which one wins.
    """
    _require(group.startswith("/") and ".." not in Path(group).parts,
             "invalid cgroup path")
    root = Path(root)
    directory = root / group.lstrip("/")
    effective = [None, None, None]
    rows: List[dict] = []
    while directory != root:
        try:
            values = [(directory / key).read_text(encoding="utf-8").strip() for key in CGROUP_LIMITS]
        except OSError as exc:
            _fail(f"{group}: cgroup limits unreadable ({exc})")
        if not rows:
            _require(
                values == [
                    "max" if expected[key] == "infinity" else str(expected[key])
                    for key in LIMITS
                ],
                f"{group}: kernel limits differ from systemd properties",
            )
        for index, value in enumerate(values):
            if value != "max":
                try:
                    number = int(value)
                except ValueError:
                    _fail(f"{group}: unrecognised kernel limit {value!r}")
                effective[index] = number if effective[index] is None else min(effective[index], number)
        rows.append({
            "cgroup": "/" + str(directory.relative_to(root)),
            **dict(zip(CGROUP_LIMITS, values)),
            "memory.current": _read_or_none(directory / "memory.current"),
        })
        directory = directory.parent
    return {"ancestors": rows, "effective_limits": dict(zip(CGROUP_LIMITS, effective))}


def _read_or_none(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def user_manager_root(group: str) -> str:
    """Cgroup path of the systemd user manager that owns *group*.

    Taken as the prefix through the first ``.service`` component, which is the
    manager instance (``user@1000.service``) above every slice in the hierarchy.
    """
    parts = Path(group).parts
    for index, part in enumerate(parts):
        if part.endswith(".service"):
            return "/" + "/".join(parts[1:index + 1])
    _fail(f"{group}: no user-manager service in the cgroup path")


def _cgroup_populated(directory: Path) -> Optional[bool]:
    """Whether *directory* holds processes, directly or through descendants.

    ``cgroup.procs`` alone cannot answer this: a scope whose only processes sit in
    a nested child reports an empty ``cgroup.procs`` while it is still busy, and
    treating that as "empty" would hand the single slot to a second worker.
    ``cgroup.events`` reports ``populated`` for the whole subtree.

    ``None`` means the question could not be answered at all. Callers must treat
    that as a failure rather than as a free slot: reading "unknown" as "empty" is
    exactly how a surviving worker is handed away, and a missing worker is not
    observable afterwards.
    """
    try:
        for line in (directory / "cgroup.events").read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition(" ")
            if key == "populated":
                return value.strip() == "1" if value.strip() in ("0", "1") else None
    except OSError:
        pass
    # No `cgroup.procs` fallback: an empty parent process list says nothing about a
    # nested descendant, and `cgroup.events` is the only per-subtree answer. Falling
    # back to it would read a busy scope as free exactly when a survivor matters.
    return None


def live_worker_scope_cgroups(root, manager_group: str) -> List[str]:
    """Cgroups of LIVE Hermes worker scopes under *manager_group*.

    Discovered from the cgroup tree rather than from gateway ancestry: a granted
    worker outlives the gateway that launched it, and reparenting takes it out of
    the replacement gateway's process tree -- exactly the survivor the slot must
    keep counting.
    """
    _require(
        manager_group.startswith("/") and ".." not in Path(manager_group).parts,
        "invalid cgroup path",
    )
    root = Path(root)
    directory = root / manager_group.lstrip("/")
    found: List[str] = []
    for candidate in sorted(directory.rglob(f"{WORKER_SCOPE_PREFIX}*{WORKER_SCOPE_SUFFIX}")):
        if not candidate.is_dir():
            continue
        populated = _cgroup_populated(candidate)
        _require(
            populated is not None,
            f"{candidate}: population evidence is unreadable, so the worker slot "
            "cannot be shown to be free",
        )
        if populated:
            found.append("/" + str(candidate.relative_to(root)))
    return found


# --------------------------------------------------------------------------
# Topology and host snapshot
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class _Contract:
    """The unit names and limits a validation run expects.

    Production uses :data:`_LIVE_CONTRACT`.  Tests and the disposable smoke pass
    their own through the private validators so a fixture's UUID-named transient
    slices can be checked with real cgroup data; the public entry points never
    accept a contract, so nothing in production can relax the reviewed values.
    """

    aggregate_unit: str = AUTONOMOUS_SLICE
    workers_unit: str = AUTONOMOUS_WORKERS_SLICE
    aggregate_limits: Mapping[str, object] = field(default_factory=lambda: dict(AUTONOMOUS_LIMITS))
    worker_limits: Mapping[str, object] = field(default_factory=lambda: dict(WORKER_SLICE_LIMITS))
    gateway_limits: Mapping[str, object] = field(default_factory=lambda: dict(GATEWAY_LIMITS))
    workers_concurrency_hard_max: int = WORKERS_CONCURRENCY_HARD_MAX
    gateway_unit_prefix: str = GATEWAY_UNIT_PREFIX


_LIVE_CONTRACT = _Contract()


def live_contract() -> _Contract:
    """The contract autonomous launches must satisfy.

    Read at every call rather than captured at import, so the scope target
    follows the contract instead of a constant that a fixture cannot retarget.
    Callers may never pass a contract of their own: the reviewed production
    values stay the only ones the real entry points use.
    """
    return _LIVE_CONTRACT


@dataclass(frozen=True)
class _Snapshot:
    """Everything a validation run is allowed to read.

    Tests build this over private cgroup/proc fixtures and a synthetic systemd
    property map, so the rules are exercised without a live user manager.
    Production reads it from the host through :func:`_live_snapshot`.
    """

    proc_root: Path = PROC_ROOT
    cgroup_root: Path = CGROUP_ROOT
    units: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    oomd_active: bool = False


def _show_units(names, *, system: bool = False) -> Dict[str, Dict[str, str]]:
    """Read *names*' properties from the user manager, or the system one.

    An unknown unit is reported by systemd as ``LoadState=not-found`` with exit
    status 0, so it stays in the map and fails the loaded check like any other
    missing policy -- an absent unit must not be indistinguishable from an
    unqueried one.
    """
    units: Dict[str, Dict[str, str]] = {}
    for name in names:
        try:
            completed = subprocess.run(
                ["systemctl", *(("show",) if system else ("--user", "show")), name,
                 "--property=" + ",".join(UNIT_PROPERTIES)],
                capture_output=True, text=True, timeout=30, stdin=subprocess.DEVNULL,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            _fail(f"cannot read {name} from the user manager ({exc})")
        properties = dict(
            line.split("=", 1) for line in completed.stdout.splitlines() if "=" in line
        )
        _require(properties, f"systemctl returned no properties for {name}")
        units[name] = properties
    return units


def _known_gateway_units(contract: _Contract) -> List[str]:
    try:
        completed = subprocess.run(
            ["systemctl", "--user", "list-units", "--all", "--no-legend", "--plain",
             f"{contract.gateway_unit_prefix}*"],
            capture_output=True, text=True, timeout=30, stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        _fail(f"cannot enumerate {contract.gateway_unit_prefix}* units ({exc})")
    names = []
    for line in completed.stdout.splitlines():
        parts = line.split()
        if parts and parts[0].endswith(GATEWAY_UNIT_SUFFIX):
            names.append(parts[0])
    return sorted(set(names))


def _oomd_active() -> bool:
    """Whether systemd-oomd is running here.

    ``ManagedOOMMemoryPressure=kill`` on the user slices is only a *candidate
    marking* for systemd-oomd; with the daemon stopped it kills nothing, and
    refusing over an inactive policy would be a false positive.
    """
    try:
        completed = subprocess.run(
            ["systemctl", "is-active", "systemd-oomd"],
            capture_output=True, text=True, timeout=15, stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.stdout.strip() == "active"


def _ancestor_unit_names(manager_group: str) -> List[str]:
    """UNIT NAMES on the path from the host root down to the user manager.

    Derived from the cgroup path rather than hardcoded, so a different UID or a
    differently named manager is inspected as it actually is. Each ancestor
    cgroup's owning unit is named by its last path component -- and the unit map is
    keyed by unit name, so returning full cgroup paths here would silently miss
    every ancestor when the OOM policy is looked up.
    """
    parts = Path(manager_group).parts
    return [parts[index] for index in range(1, len(parts))]


def _ancestor_units(names: List[str]) -> Dict[str, Dict[str, object]]:
    """OOM policy for the units ABOVE the user manager, read from their OWNER.

    ``user.slice``, ``user-1000.slice`` and ``user@1000.service`` are SYSTEM units.
    The user manager answers for names that look like them -- its own
    ``user.slice``/``user-1000.slice``, which live at a DIFFERENT cgroup -- so
    reading through ``systemctl --user`` would attribute an unrelated unit's policy
    to the ancestor, hiding a real one and inventing an imaginary one. The system
    manager owns these units and is also the view ``systemd-oomd`` reads, so it is
    the only source used here.

    Unreadable is not clean. Kernel memory limits say nothing about a userspace OOM
    killer, so an ancestor whose owning manager cannot be read refuses the launch
    rather than being assumed safe.
    """
    units: Dict[str, Dict[str, object]] = {}
    for name in names:
        try:
            properties = _show_units([name], system=True).get(name, {})
        except AutonomousResourceUnavailable as exc:
            _fail(f"{name}: its owning system manager could not be read ({exc})")
        _require(
            properties.get("LoadState") == "loaded",
            f"{name}: the owning system manager does not report this shared ancestor",
        )
        units[name] = {**properties, "views": {"system": properties}}
    return units


def _live_snapshot(contract: _Contract = _LIVE_CONTRACT) -> _Snapshot:
    names = [contract.aggregate_unit, contract.workers_unit, *_known_gateway_units(contract)]
    units = _show_units(names)
    aggregate = units[contract.aggregate_unit].get("ControlGroup", "")
    if aggregate.startswith("/") and aggregate != "/":
        units.update(_ancestor_units(_ancestor_unit_names(user_manager_root(aggregate))))
    return _Snapshot(units=units, oomd_active=_oomd_active())


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def _unit(snapshot: _Snapshot, name: str) -> Mapping[str, str]:
    properties = snapshot.units.get(name)
    _require(properties is not None, f"host policy for {name} was not read")
    return properties


def _require_unit_policy(
    snapshot: _Snapshot, name: str, limits: Mapping[str, object], label: Optional[str] = None,
) -> Mapping[str, str]:
    """*name* must be loaded, settled, and carry exactly *limits*."""
    label = label or name
    unit = _unit(snapshot, name)
    _require(unit.get("LoadState") == "loaded", f"missing host policy: {label}")
    _require(unit.get("NeedDaemonReload") == "no", f"{label}: daemon reload pending")
    # A unit the manager only knows as a stub (queried once, never configured)
    # reports LoadState=loaded with unlimited defaults and no cgroup. Say what it
    # is: the rollout gate reads this message, and "drift" would send an operator
    # looking for a file that was never installed.
    if not unit.get("ControlGroup") and all(
        unit.get(key) == "infinity" for key in limits
    ):
        _fail(f"{label} is not installed on this host ({name})")
    for key, value in limits.items():
        _require(unit.get(key) == str(value), f"{label}: {key} drift (expected {value})")
    return unit


def _require_control_group(unit: Mapping[str, str], label: str) -> str:
    group = unit.get("ControlGroup", "")
    _require(
        group.startswith("/") and group != "/" and ".." not in Path(group).parts,
        f"{label}: no usable cgroup",
    )
    return group


def _require_memory_controller(snapshot: _Snapshot) -> None:
    controllers = _read_or_none(snapshot.cgroup_root / "cgroup.controllers")
    _require(
        controllers is not None and "memory" in controllers.split(),
        "cgroup v2 memory controller is unavailable",
    )


def _require_visible(snapshot: _Snapshot, group: str) -> Path:
    """The cgroup we were told about must exist in the tree we can see.

    A sliced view of the hierarchy would otherwise let every property check pass
    while the process walk silently observed nothing.
    """
    directory = snapshot.cgroup_root / group.lstrip("/")
    _require(directory.is_dir(), f"{group}: not visible in this cgroup namespace")
    return directory


def _require_shared_ancestors_clean(
    snapshot: _Snapshot, group: str, expected: Mapping[str, object],
) -> dict:
    """No ancestor above the aggregate may share its failure domain.

    ``MemoryHigh``/``MemoryMax``/``MemorySwapMax`` above autonomy are shared
    budgets: OOM there can kill interactive work, which is the incident this
    boundary exists to prevent.  The same holds for an *active* systemd-oomd
    kill policy.  Both are reported, never repaired.
    """
    report = check_ancestors(snapshot.cgroup_root, group, expected)
    shared = [row for row in report["ancestors"] if row["cgroup"] != group]
    for row in shared:
        for key in CGROUP_LIMITS:
            _require(
                row[key] == "max",
                f"shared ancestor {row['cgroup']} imposes {key}={row[key]} above {group}",
            )
    kill_policy = []
    observed_policy = []
    for row in shared:
        # Each ancestor cgroup's owning unit is named by its last path component,
        # which is how `_ancestor_units` keys the map.
        unit = snapshot.units.get(Path(row["cgroup"]).name)
        if unit is None:
            continue
        values = {key: unit.get(key) for key in OOM_POLICY_PROPERTIES}
        observed_policy.append({
            "cgroup": row["cgroup"],
            "unit": Path(row["cgroup"]).name,
            "effective": values,
            "views": unit.get("views"),
        })
        # `auto` is systemd's default and resolves to a kill marking for the user
        # slices, so with systemd-oomd running it is NOT a promise that the ancestor
        # is outside oomd's reach. Only an explicit opt-out counts as out of reach.
        reachable = {
            key: value for key, value in values.items()
            if value and value != OOM_CONTINUE
        }
        if reachable:
            kill_policy.append({"cgroup": row["cgroup"], "properties": reachable})
    if snapshot.oomd_active and kill_policy:
        _fail(
            f"systemd-oomd is active and every shared ancestor of {group} it can "
            f"reach is a kill target: {kill_policy}; stop and disable the daemon, or "
            "have its owner mark the ancestors out of reach through #26"
        )
    return {
        **report, "shared": shared, "oomd_kill_candidates": kill_policy,
        # What was actually read for each shared ancestor, so the acceptance report
        # names the policy and which manager answered instead of only its absence.
        "shared_oom_policy": observed_policy,
    }


def _require_gateways(snapshot: _Snapshot, aggregate: str, contract: _Contract) -> List[dict]:
    """Every active gateway lives directly in the aggregate.

    A second gateway left behind in the interactive controls slice would keep
    dispatching workers into the interactive failure domain, so it fails here
    even when this process's own placement is fine.
    """
    gateways = []
    for name in sorted(snapshot.units):
        if not (name.startswith(contract.gateway_unit_prefix)
                and name.endswith(GATEWAY_UNIT_SUFFIX)):
            continue
        unit = snapshot.units[name]
        if unit.get("LoadState") != "loaded" or unit.get("ActiveState") != "active":
            continue
        _require(unit.get("NeedDaemonReload") == "no", f"{name}: daemon reload pending")
        for key, value in contract.gateway_limits.items():
            _require(unit.get(key) == str(value), f"{name}: {key} drift (expected {value})")
        _require(unit.get("Slice") == contract.aggregate_unit,
                 f"{name} must use {contract.aggregate_unit}")
        _require(unit.get("OOMPolicy") == GATEWAY_OOMPOLICY,
                 f"{name}: OOMPolicy must be {GATEWAY_OOMPOLICY}")
        group = _require_control_group(unit, name)
        _require(group == f"{aggregate}/{name}", f"{name} cgroup is outside {aggregate}")
        main_pid = int(unit.get("MainPID") or 0)
        _require(main_pid > 0, f"{name} has no main PID")
        _require(
            process_cgroup(snapshot.proc_root, main_pid) == group,
            f"{name} process is not in its service cgroup",
        )
        gateways.append({"unit": name, "cgroup": group, "main_pid": main_pid})
    return gateways


def _validate_policy(snapshot: _Snapshot, contract: _Contract = _LIVE_CONTRACT) -> dict:
    """Placement, limits and shared-ancestor checks common to both entry points."""
    _require_memory_controller(snapshot)
    aggregate_unit = _require_unit_policy(
        snapshot, contract.aggregate_unit, contract.aggregate_limits, "autonomous aggregate",
    )
    aggregate = _require_control_group(aggregate_unit, "autonomous aggregate")
    _require_visible(snapshot, aggregate)

    manager = user_manager_root(aggregate)
    _require(
        aggregate == f"{manager}/{contract.aggregate_unit}",
        f"{contract.aggregate_unit} must sit directly beneath {manager}, found {aggregate}",
    )

    workers_unit = _require_unit_policy(
        snapshot, contract.workers_unit, contract.worker_limits, "autonomous worker slice",
    )
    _require(
        workers_unit.get("ConcurrencyHardMax") == str(contract.workers_concurrency_hard_max),
        "autonomous worker slice: ConcurrencyHardMax must be "
        f"{contract.workers_concurrency_hard_max}, got {workers_unit.get('ConcurrencyHardMax')!r}",
    )
    _require(
        workers_unit.get("ConcurrencySoftMax") == str(WORKERS_CONCURRENCY_SOFT_MAX),
        "autonomous worker slice: ConcurrencySoftMax must stay "
        f"{SOFT_MAX_UNLIMITED_LABEL} so a second launch is refused instead of "
        f"queued, got {workers_unit.get('ConcurrencySoftMax')!r}",
    )
    workers_group = f"{aggregate}/{contract.workers_unit}"
    _require_visible(snapshot, workers_group)
    _require(
        workers_unit.get("ControlGroup") == workers_group,
        f"autonomous worker slice cgroup {workers_unit.get('ControlGroup')!r} is not {workers_group!r}",
    )

    aggregate_report = _require_shared_ancestors_clean(
        snapshot, aggregate, contract.aggregate_limits,
    )
    workers_report = check_ancestors(snapshot.cgroup_root, workers_group, contract.worker_limits)
    gateways = _require_gateways(snapshot, aggregate, contract)

    scopes = live_worker_scope_cgroups(snapshot.cgroup_root, manager)
    # A worker scope belongs DIRECTLY inside the worker slice; one nested deeper,
    # or living in one of the interactive slices, is a misplaced survivor.
    misplaced = [path for path in scopes if Path(path).parent.as_posix() != workers_group]
    return {
        "aggregate": {"unit": contract.aggregate_unit, "cgroup": aggregate},
        "workers": {"unit": contract.workers_unit, "cgroup": workers_group},
        "manager": manager,
        "aggregate_ancestors": aggregate_report["ancestors"],
        "aggregate_effective_limits": aggregate_report["effective_limits"],
        "shared_ancestors": aggregate_report["shared"],
        "oomd_active": snapshot.oomd_active,
        "oomd_kill_candidates": aggregate_report["oomd_kill_candidates"],
        "shared_ancestor_oom_policy": aggregate_report["shared_oom_policy"],
        "workers_ancestors": workers_report["ancestors"],
        "workers_effective_limits": workers_report["effective_limits"],
        "gateways": gateways,
        "worker_scopes": scopes,
        "misplaced_worker_scopes": misplaced,
        "concurrency_hard_max": contract.workers_concurrency_hard_max,
    }


def check_autonomous_boundary(*, pid: Optional[int] = None) -> dict:
    """Validate the autonomous boundary and, for a launch, the free slot.

    ``pid`` is the *launching* process.  Pass ``os.getpid()`` on every launch
    path: the caller's real cgroup must already be inside the aggregate, so a
    dispatcher that started in an interactive slice refuses instead of launching
    autonomy next to the editor.  ``pid=None`` is inventory only -- it reports
    placement and occupancy without asserting the caller's own position, which is
    what hermes-cto's read-only host verifier needs.
    """
    contract = live_contract()
    return _validate_boundary(_live_snapshot(contract), pid=pid, contract=contract)


def _validate_boundary(
    snapshot: _Snapshot, *, pid: Optional[int], contract: _Contract = _LIVE_CONTRACT,
) -> dict:
    report = _validate_policy(snapshot, contract)
    _require(
        not report["misplaced_worker_scopes"],
        "Hermes worker scope(s) outside the autonomous worker slice: "
        f"{report['misplaced_worker_scopes']}",
    )
    report["worker_slot_occupied"] = bool(report["worker_scopes"])
    report["launcher_cgroup"] = None
    if pid is None:
        return report

    group = process_cgroup(snapshot.proc_root, pid)
    _require_visible(snapshot, group)
    _require(
        contained(group, report["aggregate"]["cgroup"]),
        f"launcher PID {pid} is in {group}, outside {report['aggregate']['cgroup']}",
    )
    report["launcher_cgroup"] = group
    if report["worker_slot_occupied"]:
        _fail(
            f"{report['workers']['cgroup']} already holds {report['worker_scopes']}; "
            "wait for the running worker's whole scope to empty",
            slot=True,
        )
    return report


def check_autonomous_worker(pid: int) -> dict:
    """Verify that *pid* is an admitted worker in its own scope inside the boundary.

    Called after bootstrap and again immediately before the grant: a worker that
    cannot be shown to sit in its own live ``hermes-worker-*.scope`` directly
    inside the workers slice gets nothing.
    """
    contract = live_contract()
    return _validate_worker(_live_snapshot(contract), pid=pid, contract=contract)


def _validate_worker(
    snapshot: _Snapshot, *, pid: int, contract: _Contract = _LIVE_CONTRACT,
) -> dict:
    report = _validate_policy(snapshot, contract)
    _require(
        not report["misplaced_worker_scopes"],
        "Hermes worker scope(s) outside the autonomous worker slice: "
        f"{report['misplaced_worker_scopes']}",
    )
    workers_group = report["workers"]["cgroup"]
    group = process_cgroup(snapshot.proc_root, pid)
    _require_visible(snapshot, group)
    name = Path(group).name
    _require(
        name.startswith(WORKER_SCOPE_PREFIX) and name.endswith(WORKER_SCOPE_SUFFIX),
        f"worker PID {pid} is in {group}, not in a {WORKER_SCOPE_PREFIX}*{WORKER_SCOPE_SUFFIX}",
    )
    _require(
        Path(group).parent.as_posix() == workers_group,
        f"worker scope {group} is not directly inside {workers_group}",
    )
    scopes = report["worker_scopes"]
    _require(
        scopes == [group],
        f"expected exactly the admitted worker scope {group}, found {scopes}",
    )
    report["worker_cgroup"] = group
    report["worker_pid"] = pid
    return report

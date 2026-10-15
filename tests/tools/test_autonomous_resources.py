"""Boundary decisions in ``tools/autonomous_resources.py``.

These are the rules that decide whether an autonomous worker may start at all and
where it ended up. They run against private cgroup/proc fixtures and a synthetic
systemd property map, so every rejection is exercised deterministically rather
than depending on the runner's host policy; the real-kernel proof that systemd
enforces the slot lives in the disposable-slice fixture used at the bottom of
this file and in dotfiles' ``scripts/check_autonomous_isolation.py``.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tools import autonomous_resources as ar
from tests._fixtures import autonomous_slice as fixture

MANAGER = "/user.slice/user-1000.slice/user@1000.service"
AGGREGATE = MANAGER + "/autonomous.slice"
WORKERS = AGGREGATE + "/autonomous-workers.slice"
GATEWAY = "hermes-gateway.service"
GATEWAY_GROUP = f"{AGGREGATE}/{GATEWAY}"

LIMITS = {"MemoryHigh": 2 * ar.GIB, "MemoryMax": 3 * ar.GIB, "MemorySwapMax": 1 * ar.GIB}


def _unit(limits, group, *, active="active", reload_state="no", **extra):
    properties = {key: str(value) for key, value in limits.items()}
    properties.update({
        "LoadState": "loaded",
        "ActiveState": active,
        "NeedDaemonReload": reload_state,
        "Slice": ar.AUTONOMOUS_SLICE,
        "ControlGroup": group,
        "MainPID": "0",
        "OOMPolicy": "kill",
        "ConcurrencyHardMax": str(ar.UNLIMITED),
        "ConcurrencySoftMax": str(ar.UNLIMITED),
        "ManagedOOMMemoryPressure": "auto",
        "ManagedOOMSwap": "auto",
    })
    properties.update({key: str(value) for key, value in extra.items()})
    return properties


class Host:
    """A synthetic host: a cgroup tree, a proc tree, and a unit property map."""

    def __init__(self, tmp_path: Path, *, oomd_active: bool = False):
        self.root = tmp_path / "cgroup"
        self.proc = tmp_path / "proc"
        self.units = {}
        self.oomd_active = oomd_active
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "cgroup.controllers").write_text("cpuset cpu memory pids\n")
        for relative, values in (
            ("user.slice", ("max", "max", "max")),
            ("user.slice/user-1000.slice", ("max", "max", "max")),
            (MANAGER.lstrip("/"), ("max", "max", "max")),
            (AGGREGATE.lstrip("/"), self._kernel_values(LIMITS)),
            (WORKERS.lstrip("/"), self._kernel_values(LIMITS)),
            (GATEWAY_GROUP.lstrip("/"), self._kernel_values(LIMITS)),
        ):
            self.add_cgroup(relative, values)
        self.units[ar.AUTONOMOUS_SLICE] = _unit(LIMITS, AGGREGATE)
        self.units[ar.AUTONOMOUS_WORKERS_SLICE] = _unit(
            LIMITS, WORKERS, ConcurrencyHardMax=ar.WORKERS_CONCURRENCY_HARD_MAX,
        )
        self.units[GATEWAY] = _unit(
            LIMITS, GATEWAY_GROUP, MainPID="5000", OOMPolicy="kill",
        )
        self.add_process(5000, GATEWAY_GROUP)
        # Shared ancestors above autonomy, exactly as the live host reports them.
        self.units["user.slice"] = {"ManagedOOMMemoryPressure": "kill", "ManagedOOMSwap": "auto"}
        self.units["user-1000.slice"] = {"ManagedOOMMemoryPressure": "kill", "ManagedOOMSwap": "auto"}
        self.units["user@1000.service"] = {"ManagedOOMMemoryPressure": "auto", "ManagedOOMSwap": "auto"}

    @staticmethod
    def _kernel_values(limits):
        return tuple(str(limits[key]) for key in ar.LIMITS)

    def add_cgroup(self, relative: str, values):
        directory = self.root / relative
        directory.mkdir(parents=True, exist_ok=True)
        for key, value in zip(ar.CGROUP_LIMITS, values):
            (directory / key).write_text(str(value))
        (directory / "memory.current").write_text("1024")
        return directory

    def add_scope(self, group: str, pid: int | None, *, populated: bool | None = None,
                  nested_pid: int | None = None):
        directory = self.root / group.lstrip("/")
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "cgroup.procs").write_text("" if pid is None else f"{pid}\n")
        if nested_pid is not None:
            nested = directory / "nested"
            nested.mkdir(exist_ok=True)
            (nested / "cgroup.procs").write_text(f"{nested_pid}\n")
        if populated is None:
            populated = pid is not None or nested_pid is not None
        (directory / "cgroup.events").write_text(
            f"populated {1 if populated else 0}\nfrozen 0\n"
        )
        return directory

    def ensure_cgroup(self, group: str):
        """Create *group* with default limits when the fixture does not define it.

        The validator refuses a cgroup it cannot see, so a fixture process placed in
        an undeclared group would fail for the wrong reason.
        """
        if not (self.root / group.lstrip("/")).is_dir():
            # lstrip: joining an absolute path onto the fixture root would escape it.
            self.add_cgroup(group.lstrip("/"), ("max", "max", "max"))
        return group

    def add_process(self, pid: int, group: str, children=()):
        self.ensure_cgroup(group)
        directory = self.proc / str(pid)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "cgroup").write_text("0::" + group + "\n")
        task = directory / "task" / str(pid)
        task.mkdir(parents=True, exist_ok=True)
        (task / "children").write_text(" ".join(str(child) for child in children))
        for child in children:
            self.add_process(child, group)
        return pid

    def snapshot(self):
        return ar._Snapshot(
            proc_root=self.proc, cgroup_root=self.root, units=self.units,
            oomd_active=self.oomd_active,
        )

    def admit(self, pid: int | None):
        return ar._validate_boundary(self.snapshot(), pid=pid)

    def verify_worker(self, pid: int):
        return ar._validate_worker(self.snapshot(), pid=pid)


@pytest.fixture()
def host(tmp_path):
    return Host(tmp_path)


def test_exact_topology_and_limits_admit_a_launcher_inside_the_aggregate(host):
    launcher = host.add_process(6000, f"{AGGREGATE}/launcher.scope")
    report = host.admit(launcher)
    assert report["aggregate"]["cgroup"] == AGGREGATE
    assert report["workers"]["cgroup"] == WORKERS
    assert report["worker_slot_occupied"] is False
    assert [entry["unit"] for entry in report["gateways"]] == [GATEWAY]
    assert report["launcher_cgroup"] == f"{AGGREGATE}/launcher.scope"
    # Reported, not enforced: the marking is only a policy for systemd-oomd.
    assert report["oomd_kill_candidates"], report
    assert report["oomd_active"] is False


def test_inventory_does_not_require_the_caller_to_be_inside(host):
    outsider = host.add_process(7000, "/app.slice/omp-session.scope")
    report = host.admit(None)
    assert report["launcher_cgroup"] is None
    assert report["worker_slot_occupied"] is False
    # ...but the same caller is refused the moment it actually launches.
    with pytest.raises(ar.AutonomousResourceUnavailable, match="outside"):
        host.admit(outsider)


def test_missing_worker_slice_refuses(host):
    del host.units[ar.AUTONOMOUS_WORKERS_SLICE]
    with pytest.raises(ar.AutonomousResourceUnavailable, match="was not read"):
        host.admit(None)


def test_not_installed_slice_is_named_as_such(host):
    host.units[ar.AUTONOMOUS_SLICE] = {
        "LoadState": "loaded", "NeedDaemonReload": "no", "ControlGroup": "",
        "MemoryHigh": "infinity", "MemoryMax": "infinity", "MemorySwapMax": "infinity",
    }
    with pytest.raises(ar.AutonomousResourceUnavailable, match="is not installed"):
        host.admit(None)


@pytest.mark.parametrize(("unit", "key", "value"), [
    (ar.AUTONOMOUS_SLICE, "NeedDaemonReload", "yes"),
    (ar.AUTONOMOUS_SLICE, "MemoryMax", str(4 * ar.GIB)),
    (ar.AUTONOMOUS_SLICE, "MemorySwapMax", "infinity"),
    (ar.AUTONOMOUS_WORKERS_SLICE, "NeedDaemonReload", "yes"),
    (ar.AUTONOMOUS_WORKERS_SLICE, "MemoryHigh", "infinity"),
])
def test_policy_drift_refuses(host, unit, key, value):
    host.units[unit][key] = value
    with pytest.raises(ar.AutonomousResourceUnavailable) as excinfo:
        host.admit(None)
    assert str(excinfo.value).startswith(ar.BOUNDARY_DIAGNOSTIC)


@pytest.mark.parametrize("hard_max", [str(ar.UNLIMITED), "0", "1", "3"])
def test_native_slot_must_match_the_reviewed_capacity(host, hard_max):
    host.units[ar.AUTONOMOUS_WORKERS_SLICE]["ConcurrencyHardMax"] = hard_max
    with pytest.raises(ar.AutonomousResourceUnavailable, match="drain workers and align") as excinfo:
        host.admit(None)
    assert f"must be {ar.WORKERS_CONCURRENCY_HARD_MAX}, got {hard_max!r}" in str(excinfo.value)


def test_soft_limit_must_not_queue_a_second_launch(host):
    host.units[ar.AUTONOMOUS_WORKERS_SLICE]["ConcurrencySoftMax"] = "1"
    with pytest.raises(ar.AutonomousResourceUnavailable, match="ConcurrencySoftMax"):
        host.admit(None)


def test_aggregate_must_sit_directly_beneath_the_user_manager(host):
    host.units[ar.AUTONOMOUS_SLICE]["ControlGroup"] = "/app.slice/autonomous.slice"
    host.add_cgroup("app.slice/autonomous.slice", Host._kernel_values(LIMITS))
    # Outside the user manager there is no owning manager to bound the aggregate.
    with pytest.raises(ar.AutonomousResourceUnavailable, match="no user-manager service"):
        host.admit(None)


def test_worker_slice_must_be_directly_inside_the_aggregate(host):
    host.units[ar.AUTONOMOUS_WORKERS_SLICE]["ControlGroup"] = f"{AGGREGATE}/deeper/{ar.AUTONOMOUS_WORKERS_SLICE}"
    with pytest.raises(ar.AutonomousResourceUnavailable, match="worker slice cgroup"):
        host.admit(None)


def test_kernel_limits_must_match_the_systemd_properties(host):
    (host.root / AGGREGATE.lstrip("/") / "memory.max").write_text(str(6 * ar.GIB))
    with pytest.raises(ar.AutonomousResourceUnavailable, match="differ from systemd properties"):
        host.admit(None)


def test_unreadable_cgroup_limits_refuse(host):
    (host.root / AGGREGATE.lstrip("/") / "memory.high").unlink()
    (host.root / AGGREGATE.lstrip("/") / "memory.high").mkdir()
    with pytest.raises(ar.AutonomousResourceUnavailable, match="unreadable"):
        host.admit(None)


def test_missing_memory_controller_refuses(host):
    (host.root / "cgroup.controllers").write_text("cpuset cpu pids\n")
    with pytest.raises(ar.AutonomousResourceUnavailable, match="memory controller"):
        host.admit(None)


def test_invisible_cgroup_tree_refuses(host):
    import shutil

    shutil.rmtree(host.root / AGGREGATE.lstrip("/"))
    with pytest.raises(ar.AutonomousResourceUnavailable, match="not visible"):
        host.admit(None)


@pytest.mark.parametrize("key", list(ar.CGROUP_LIMITS))
def test_finite_shared_ancestor_refuses(host, key):
    (host.root / MANAGER.lstrip("/") / key).write_text(str(2 * ar.GIB))
    with pytest.raises(ar.AutonomousResourceUnavailable, match="shared ancestor"):
        host.admit(None)


def test_oomd_kill_policy_on_a_shared_ancestor_refuses_only_while_oomd_is_active(host):
    candidates = host.admit(None)["oomd_kill_candidates"]
    assert candidates, "a `kill` marking on an ancestor must be reported"
    assert {entry["cgroup"] for entry in candidates} >= {
        "/user.slice", "/user.slice/user-1000.slice",
    }
    active = Host(host.proc.parent, oomd_active=True)
    with pytest.raises(ar.AutonomousResourceUnavailable, match="systemd-oomd is active"):
        ar._validate_boundary(active.snapshot(), pid=None)


def test_ancestor_units_are_keyed_by_unit_name_not_cgroup_path():
    """Regression: the OOM-policy lookup keys on the unit name.

    Keying the ancestor map by cgroup path made every shared-ancestor lookup miss,
    so an active OOM-kill policy above the aggregate was never reported or
    rejected while the fixtures -- which key by unit name -- kept passing.
    """
    assert ar._ancestor_unit_names("/user.slice/user-1000.slice/user@1000.service") == [
        "user.slice", "user-1000.slice", "user@1000.service",
    ]


def test_active_gateway_outside_the_aggregate_refuses(host):
    host.units["hermes-gateway-other.service"] = _unit(
        LIMITS, f"{MANAGER}/agents-controls.slice/hermes-gateway-other.service",
        MainPID="5100",
    )
    with pytest.raises(ar.AutonomousResourceUnavailable, match="cgroup is outside"):
        host.admit(None)


def test_active_gateway_with_a_migrated_main_pid_refuses(host):
    host.units[GATEWAY]["ControlGroup"] = AGGREGATE + "/" + GATEWAY
    host.add_process(5000, f"{AGGREGATE}/launcher.scope")
    with pytest.raises(ar.AutonomousResourceUnavailable, match="not in its service cgroup"):
        host.admit(None)


def test_worker_scopes_admit_up_to_capacity_then_release_one_slot(host):
    first = f"{WORKERS}/hermes-worker-kanban-t-run-7.scope"
    second = f"{WORKERS}/hermes-worker-kanban-t-run-8.scope"
    launcher = host.add_process(6000, AGGREGATE)
    host.add_scope(first, 6100)
    host.add_process(6100, first)
    report = host.admit(launcher)
    assert report["worker_slot_occupied"] is False
    assert report["worker_scopes"] == [first]
    assert host.verify_worker(6100)["worker_cgroup"] == first

    host.add_scope(second, 6200)
    host.add_process(6200, second)
    assert host.verify_worker(6100)["worker_cgroup"] == first
    assert host.verify_worker(6200)["worker_cgroup"] == second
    with pytest.raises(ar.AutonomousWorkerBusy) as excinfo:
        host.admit(launcher)
    assert str(excinfo.value).startswith(ar.SLOT_DIAGNOSTIC)
    assert host.admit(None)["worker_scopes"] == [first, second]
    assert host.admit(None)["worker_slot_occupied"] is True

    # Only the completed scope empties; the second remains accounted and valid.
    host.add_scope(first, None, populated=False)
    report = host.admit(launcher)
    assert report["worker_scopes"] == [second]
    assert report["worker_slot_occupied"] is False
    assert host.verify_worker(6200)["worker_cgroup"] == second


def test_scope_with_only_a_nested_descendant_still_occupies_the_slot(host):
    """An empty ``cgroup.procs`` on the scope is not proof the slot is free."""
    host.add_scope(f"{WORKERS}/hermes-worker-run-7.scope", None, nested_pid=6200)
    host.add_scope(f"{WORKERS}/hermes-worker-run-8.scope", 6300)
    report = host.admit(None)
    assert report["worker_slot_occupied"] is True
    with pytest.raises(ar.AutonomousWorkerBusy):
        host.admit(host.add_process(6000, AGGREGATE))


def test_unreadable_population_evidence_refuses(host):
    """Unknown is not empty: the slot must not be handed away on missing evidence."""
    scope = host.root / f"{WORKERS}/hermes-worker-run-6.scope".lstrip("/")
    scope.mkdir(parents=True)
    with pytest.raises(ar.AutonomousResourceUnavailable, match="population evidence"):
        host.admit(None)


def test_collected_scope_without_processes_frees_the_slot(host):
    host.add_scope(f"{WORKERS}/hermes-worker-run-5.scope", None, populated=False)
    report = host.admit(host.add_process(6000, AGGREGATE))
    assert report["worker_slot_occupied"] is False
    assert report["worker_scopes"] == []


def test_survivor_outside_the_worker_slice_is_drift_not_busy(host):
    """A misplaced survivor is a policy failure, never a reason to kill it."""
    host.add_scope(f"{MANAGER}/agents-controls.slice/hermes-worker-legacy.scope", 6300)
    with pytest.raises(ar.AutonomousResourceUnavailable) as excinfo:
        host.admit(host.add_process(6000, AGGREGATE))
    assert not isinstance(excinfo.value, ar.AutonomousWorkerBusy)
    assert ar.SLOT_DIAGNOSTIC not in str(excinfo.value)


def test_worker_verification_requires_its_own_scope_immediately_inside(host):
    scope = f"{WORKERS}/hermes-worker-kanban-t-run-7.scope"
    host.add_scope(scope, 6400)
    host.add_process(6400, scope)
    report = host.verify_worker(6400)
    assert report["worker_cgroup"] == scope
    assert report["worker_pid"] == 6400


@pytest.mark.parametrize(("group", "match"), [
    # A scope directly in the workers slice that is not a worker scope at all.
    (f"{WORKERS}/helper.service", "not in a hermes-worker"),
    # A PID that never reached the workers slice.
    (f"{AGGREGATE}/launcher.scope", "not in a hermes-worker"),
    # Nested deeper, or living in an interactive slice: caught by the misplaced
    # rule that runs before placement, which is the rule that names the survivor.
    (f"{WORKERS}/nested/hermes-worker-x.scope", "outside the autonomous worker slice"),
    (f"{MANAGER}/agents-controls.slice/hermes-worker-x.scope", "outside the autonomous worker slice"),
])
def test_worker_verification_refuses_any_other_placement(host, group, match):
    # Real population evidence: an unreadable scope is refused earlier, and these
    # cases are about placement, not about missing evidence.
    host.add_scope(group, 6500)
    pid = host.add_process(6500, group)
    with pytest.raises(ar.AutonomousResourceUnavailable, match=match):
        host.verify_worker(pid)


def test_worker_verification_refuses_more_than_the_native_capacity(host):
    for run, pid in ((7, 6400), (8, 6401), (9, 6402)):
        scope = f"{WORKERS}/hermes-worker-kanban-t-run-{run}.scope"
        host.add_scope(scope, pid)
        host.add_process(pid, scope)
    with pytest.raises(ar.AutonomousResourceUnavailable, match="within capacity"):
        host.verify_worker(6400)


def test_worker_verification_refuses_drift_in_the_boundary(host):
    scope = f"{WORKERS}/hermes-worker-kanban-t-run-7.scope"
    host.add_scope(scope, 6400)
    host.add_process(6400, scope)
    host.units[ar.AUTONOMOUS_WORKERS_SLICE]["ConcurrencyHardMax"] = str(ar.UNLIMITED)
    with pytest.raises(ar.AutonomousResourceUnavailable, match="ConcurrencyHardMax"):
        host.verify_worker(6400)


def test_errors_are_infrastructure_and_carry_the_documented_prefix(host):
    del host.units[ar.AUTONOMOUS_SLICE]
    with pytest.raises(ar.AutonomousResourceUnavailable) as excinfo:
        host.admit(None)
    assert isinstance(excinfo.value, RuntimeError)
    assert str(excinfo.value).startswith(ar.BOUNDARY_DIAGNOSTIC)
    assert issubclass(ar.AutonomousWorkerBusy, ar.AutonomousResourceUnavailable)


def test_process_cgroup_raises_filenotfound_for_a_vanished_pid(host):
    """Callers walking a live tree rely on this to skip short-lived descendants."""
    with pytest.raises(FileNotFoundError):
        ar.process_cgroup(host.proc, 999999)


def test_contained_rejects_a_bare_prefix_sibling():
    assert ar.contained("/a/b/c", "/a/b")
    assert not ar.contained("/a/bc", "/a/b")
    assert ar.contained("/a/b", "/a/b")
    assert ar.contained("/a", "/")


def test_live_worker_scope_discovery_is_anchored_at_the_manager(host):
    host.add_scope(f"{WORKERS}/hermes-worker-a.scope", 6400)
    host.add_scope(f"{MANAGER}/app.slice/hermes-worker-b.scope", 6401)
    host.add_scope(f"{MANAGER}/other.slice/helper.service", 6402)
    assert ar.live_worker_scope_cgroups(host.root, MANAGER) == [
        f"{MANAGER}/app.slice/hermes-worker-b.scope",
        f"{WORKERS}/hermes-worker-a.scope",
    ]


@pytest.mark.platforms("linux")
@pytest.mark.live_system_guard_bypass  # creates a disposable transient slice
def test_real_transient_slices_satisfy_the_boundary_for_a_real_launcher(tmp_path, monkeypatch):
    """The validator accepts a real systemd topology, not just a fixture tree.

    This is the bridge between the synthetic cases above and the production
    deployment: the same code decides, reading real properties and a real cgroup
    tree, and the launcher is really inside the aggregate.
    """
    fixture.require_user_bus()
    # A worker executing the test is itself a real scope outside the disposable
    # aggregate, so the isolation validator correctly refuses this fixture.
    if "hermes-worker-" in Path("/proc/self/cgroup").read_text(encoding="utf-8"):
        pytest.skip("the test runner is an autonomous worker outside the fixture slice")
    with fixture.disposable_boundary(concurrency_hard_max=ar.WORKERS_CONCURRENCY_HARD_MAX) as boundary:
        # The hermetic runner strips the user-bus environment. The production
        # validator reads ambient env while the fixture's setup passes it explicitly.
        for key in ("XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS"):
            if key in fixture.user_bus_env():
                monkeypatch.setenv(key, fixture.user_bus_env()[key])
        boundary.install(monkeypatch)
        report = ar.check_autonomous_boundary(pid=None)
        assert report["aggregate"]["cgroup"] == boundary.aggregate_group
        assert report["workers"]["cgroup"] == boundary.workers_group
        assert report["worker_slot_occupied"] is False
        assert report["gateways"] == []

        code = (
            "import os;"
            "from tools import autonomous_resources as ar;"
            "ar._LIVE_CONTRACT = ar._Contract(**%r);"
            "report = ar.check_autonomous_boundary(pid=os.getpid());"
            "print(report['launcher_cgroup'])"
        ) % (boundary.contract_kwargs(),)
        completed = subprocess.run(
            boundary.launcher_argv([sys.executable, "-c", code], unit_suffix="adm"),
            capture_output=True, text=True, timeout=60, env=fixture.user_bus_env(),
            cwd=str(Path(ar.__file__).resolve().parents[1]),
        )
        assert completed.returncode == 0, completed.stderr
        assert completed.stdout.strip().startswith(boundary.aggregate_group + "/")


@pytest.mark.platforms("linux")
@pytest.mark.live_system_guard_bypass  # creates disposable transient slices
def test_real_native_slot_refuses_the_third_worker_and_releases_one(tmp_path):
    """The real manager admits two scoped workers, then refuses a third.

    The read-only precheck can be raced; native systemd capacity cannot. After
    one scope empties, another starts while the surviving scope remains live.
    """
    fixture.require_user_bus()
    with fixture.disposable_boundary(concurrency_hard_max=ar.WORKERS_CONCURRENCY_HARD_MAX) as boundary:
        def start(name: str, command: list[str]) -> subprocess.Popen[bytes]:
            return subprocess.Popen(
                ["systemd-run", "--user", "--scope", "--quiet", "--collect",
                 f"--slice={boundary.workers_unit}", "--unit", name,
                 "--", *command],
                env=fixture.user_bus_env(), start_new_session=True,
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            )

        first_unit = f"hermes-worker-{boundary.token}-first.scope"
        second_unit = f"hermes-worker-{boundary.token}-second.scope"
        holders = [
            start(name, [sys.executable, "-c", "import time; time.sleep(60)"])
            for name in (first_unit, second_unit)
        ]
        try:
            for unit, proc in zip((first_unit, second_unit), holders):
                deadline = time.monotonic() + 20
                while time.monotonic() < deadline:
                    if proc.poll() is not None:
                        assert proc.stderr is not None
                        pytest.fail(f"{unit} was refused: " + proc.stderr.read().decode(errors="replace"))
                    if os.path.exists(f"/sys/fs/cgroup{boundary.workers_group}/{unit}"):
                        break
                    time.sleep(0.05)
                else:
                    pytest.fail(f"{unit} never entered its scope")

            third = start(f"hermes-worker-{boundary.token}-third.scope", ["/bin/true"])
            assert third.wait(timeout=30) != 0, "native capacity admitted a third worker"
            assert third.stderr is not None
            assert b"Concurrency limit" in third.stderr.read()

            holders[0].terminate()
            holders[0].wait(timeout=10)
            # Scope collection is asynchronous. Retry only a disposable launch;
            # the surviving worker must remain live throughout.
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                assert holders[1].poll() is None, "releasing one scope killed the other"
                replacement = start(f"hermes-worker-{boundary.token}-replacement.scope", ["/bin/true"])
                if replacement.wait(timeout=30) == 0:
                    break
                time.sleep(0.1)
            else:
                pytest.fail("native capacity did not reopen after one scope emptied")
            assert holders[1].poll() is None
        finally:
            for proc in holders:
                if proc.poll() is None:
                    proc.terminate()
                    proc.wait(timeout=10)


@pytest.mark.platforms("linux")
@pytest.mark.live_system_guard_bypass  # creates a disposable transient slice
def test_user_manager_inventory_derives_bus_without_changing_system_manager(monkeypatch):
    """A scrubbed test or system-level launcher still reads its user manager."""
    fixture.require_user_bus()
    with fixture.disposable_boundary(concurrency_hard_max=ar.WORKERS_CONCURRENCY_HARD_MAX) as boundary:
        monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
        monkeypatch.delenv("DBUS_SESSION_BUS_ADDRESS", raising=False)
        original_run = subprocess.run
        calls = []

        def observed_run(argv, **kwargs):
            calls.append((argv, kwargs.get("env")))
            return original_run(argv, **kwargs)

        monkeypatch.setattr(ar.subprocess, "run", observed_run)
        properties = ar._show_units([boundary.aggregate_unit, boundary.workers_unit])
        assert properties[boundary.aggregate_unit]["ControlGroup"] == boundary.aggregate_group
        assert properties[boundary.workers_unit]["ConcurrencyHardMax"] == str(ar.WORKERS_CONCURRENCY_HARD_MAX)
        # The system manager is still the owner of shared ancestors; do not
        # route its request through the user manager just to make this pass.
        assert ar._show_units(["user.slice"], system=True)["user.slice"]["LoadState"] == "loaded"
        assert isinstance(ar._known_gateway_units(boundary.contract()), list)
        expected_bus = fixture.user_bus_env()["DBUS_SESSION_BUS_ADDRESS"]
        assert all(env and env.get("DBUS_SESSION_BUS_ADDRESS") == expected_bus
                   for argv, env in calls if "--user" in argv)
        assert all(env is None for argv, env in calls if "--user" not in argv)


def test_ancestor_units_refuse_when_no_manager_reports_the_ancestor(monkeypatch):
    """Unknown policy is incomplete visibility, not a clean bill of health.

    Kernel memory limits cannot establish the absence of a systemd-oomd policy, so
    an ancestor neither manager can describe must refuse the launch.
    """
    def fake_show(names, *, system=False):
        if system:
            raise ar.AutonomousResourceUnavailable("no system bus here")
        return {name: {"LoadState": "not-found"} for name in names}

    monkeypatch.setattr(ar, "_show_units", fake_show)
    with pytest.raises(ar.AutonomousResourceUnavailable, match="could not be read"):
        ar._ancestor_units(["user@1000.service"])


def test_shared_ancestor_oom_policy_is_reported(host):
    """The boundary report names the policy it read, not only its absence."""
    report = host.admit(None)
    observed = {entry["unit"]: entry for entry in report["shared_ancestor_oom_policy"]}
    assert set(observed) >= {"user.slice", "user-1000.slice", "user@1000.service"}
    assert observed["user.slice"]["effective"]["ManagedOOMMemoryPressure"] == "kill"
def test_ancestor_units_read_the_owning_system_manager_only(monkeypatch):
    """The user manager's same-named units are NOT the ancestors.

    ``systemctl --user show user.slice`` answers with the user manager's own
    ``user.slice`` at a different cgroup, whose policy would be attributed to the
    real ancestor -- inventing one and hiding the other. The system manager owns
    these units and is the view systemd-oomd reads, so it is the only source.
    """
    seen = []

    def fake_show(names, *, system=False):
        seen.append((tuple(names), system))
        return {name: {
            "LoadState": "loaded",
            "ManagedOOMMemoryPressure": "auto",
            "ManagedOOMSwap": "auto",
        } for name in names}

    monkeypatch.setattr(ar, "_show_units", fake_show)
    units = ar._ancestor_units(["user.slice", "user@1000.service"])
    assert seen == [
        (("user.slice",), True), (("user@1000.service",), True),
    ], "the user manager must not be consulted for an ancestor it does not own"
    assert units["user.slice"]["views"]["system"]["ManagedOOMMemoryPressure"] == "auto"


def test_ancestor_units_refuse_when_the_owning_manager_is_unreadable(monkeypatch):
    """Incomplete visibility is failure: kernel limits say nothing about oomd."""
    def fake_show(names, *, system=False):
        raise ar.AutonomousResourceUnavailable("no system bus here")

    monkeypatch.setattr(ar, "_show_units", fake_show)
    with pytest.raises(ar.AutonomousResourceUnavailable, match="owning system manager could not be read"):
        ar._ancestor_units(["user.slice"])


def test_ancestor_units_refuse_when_the_owner_does_not_report_the_ancestor(monkeypatch):
    def fake_show(names, *, system=False):
        return {name: {"LoadState": "not-found"} for name in names}

    monkeypatch.setattr(ar, "_show_units", fake_show)
    with pytest.raises(ar.AutonomousResourceUnavailable, match="does not report this shared ancestor"):
        ar._ancestor_units(["user@1000.service"])


def test_auto_only_ancestors_are_kill_targets_while_oomd_runs(host):
    """`auto` alone must trigger the refusal: this systemd has no opt-out.

    The host reading is `auto` on all three ancestors, so a test that only ever sees
    an explicit `kill` would pass even if `auto` were treated as safe. Both snapshots
    are set to literal `auto` and the readings asserted, so the refusal can only come
    from the `auto` values.
    """
    for name in ("user.slice", "user-1000.slice", "user@1000.service"):
        host.units[name] = {
            "ManagedOOMMemoryPressure": "auto", "ManagedOOMSwap": "auto",
        }
    report = host.admit(None)
    readings = {entry["unit"]: entry["effective"] for entry in report["shared_ancestor_oom_policy"]}
    assert set(readings) == {"user.slice", "user-1000.slice", "user@1000.service"}, readings
    assert all(value == "auto" for entry in readings.values() for value in entry.values()), readings
    assert {entry["cgroup"] for entry in report["oomd_kill_candidates"]} == {
        "/user.slice", "/user.slice/user-1000.slice", "/user.slice/user-1000.slice/user@1000.service",
    }, report["oomd_kill_candidates"]

    active = Host(host.proc.parent, oomd_active=True)
    for name in ("user.slice", "user-1000.slice", "user@1000.service"):
        active.units[name] = {
            "ManagedOOMMemoryPressure": "auto", "ManagedOOMSwap": "auto",
        }
    with pytest.raises(ar.AutonomousResourceUnavailable, match="systemd-oomd is active"):
        ar._validate_boundary(active.snapshot(), pid=None)


def test_ancestors_are_clean_only_without_oomd(host):
    """The daemon's state, not the ancestor's marking, is what makes it safe.

    This is the positive control for the refusal above: the same ancestors pass with
    the daemon stopped, so the refusal comes from the daemon and not from the shape of
    the fixture.
    """
    report = host.admit(None)
    assert report["oomd_active"] is False
    assert report["oomd_kill_candidates"], "the readings are still reported"
    assert host.admit(host.add_process(6000, AGGREGATE))["worker_slot_occupied"] is False

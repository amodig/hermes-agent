"""Boundary regressions for the fixed Podman verification executor.

These pin attestation and cleanup decisions against crafted engine JSON so a weakened
check fails loudly. The real container/namespace/readiness proof is
``scripts/verify_worker_execution.py``.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import worker_execution_podman as podman

pytestmark = pytest.mark.platforms("linux")  # procfs, namespaces and rootless Podman are Linux-only

IMAGE = "docker.io/library/postgres@sha256:" + "b" * 64
IMAGE_ID = "c" * 64


def _spec(tmp_path: Path) -> podman._Spec:
    identity = {
        "passwd": tmp_path / "passwd",
        "group": tmp_path / "group",
        "hosts": tmp_path / "hosts",
        "resolv": tmp_path / "resolv",
    }
    for path in identity.values():
        path.write_text("x\n", encoding="utf-8")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    return podman._Spec(
        name="verifier", kind="verifier", image_reference=IMAGE, image_tag="localhost/worker-execution:pg",
        entrypoint=["/bin/sh"], command=["-c", "exec true"],
        env={"HOME": "/scratch/home", "TMPDIR": "/scratch/tmp", "PATH": "/usr/bin", "LANG": "C.UTF-8", "CI": "1"},
        mounts=[(scratch, "/scratch", True), (identity["passwd"], "/etc/passwd", False)],
        network="container:" + "d" * 64, timeout_seconds=30, log_path=tmp_path / "verifier.log",
    )


def _inspect_for(spec: podman._Spec, *, overrides=None) -> dict:
    container_id = "a" * 64
    data = {
        "Id": container_id,
        "Image": IMAGE_ID,
        "State": {
            "Status": "running", "Pid": os.getpid(), "ConmonPid": os.getpid(), "ExitCode": 0,
        },
        "Config": {
            "Entrypoint": list(spec.entrypoint),
            "Cmd": list(spec.command),
            "User": spec.user,
            "Tty": False,
            "OpenStdin": False,
            "Env": [f"{key}={value}" for key, value in spec.env.items()] + [f"HOSTNAME={container_id[:12]}"],
            "Healthcheck": {"Test": ["NONE"]},
            "StopSignal": "SIGKILL",
        },
        "HostConfig": {
            "ReadonlyRootfs": True,
            "CapAdd": [],
            "CapDrop": sorted(podman.DEFAULT_ROOTLESS_CAPS),
            "SecurityOpt": ["no-new-privileges"],
            "PortBindings": {},
            "Privileged": False,
            "PublishAllPorts": False,
            "RestartPolicy": {"Name": "no"},
            "AutoRemove": False,
            "LogConfig": {"Type": "k8s-file", "Path": str(spec.log_path)},
            "NetworkMode": spec.network,
            "Ulimits": [{"Name": "RLIMIT_NOFILE", "Soft": 1024, "Hard": 1024}],
        },
        "NetworkSettings": {"SandboxKey": f"/proc/{os.getpid()}/ns/net"},
        "Mounts": [
            {"Source": str(source), "Destination": destination, "RW": writable}
            for source, destination, writable in spec.mounts
        ],
    }
    for path, value in (overrides or {}).items():
        target = data
        keys = path.split(".")
        for key in keys[:-1]:
            target = target[key]
        target[keys[-1]] = value
    return data


class _FakeEngine:
    def __init__(self, payload: dict, run_root: Path) -> None:
        self.payload = payload
        self.run_root = run_root

    def text(self, args, *, timeout=60.0, deadline=None, supervise=None):  # noqa: ANN001 - test double
        return json.dumps([self.payload])


def _container() -> podman.Container:
    container = podman.Container("verifier", "verifier", IMAGE, "localhost/worker-execution:pg")
    container.id = "a" * 64
    return container


def _attest(tmp_path, spec, *, host_pid_inode: int = 1001, **overrides):
    payload = _inspect_for(spec, overrides=overrides)
    engine = _FakeEngine(payload, tmp_path / "run")
    engine.run_root.mkdir(parents=True, exist_ok=True)
    container = _container()
    # Real conmon pidfile contents for the fake conmon pid.
    (engine.run_root / (spec.name + ".conmon.pid")).write_text(str(payload["State"]["ConmonPid"]))
    real_namespace_inode = podman._namespace_inode
    podman._namespace_inode = lambda pid, name: (
        5000 if name == "pid" else real_namespace_inode(pid, name)
    )
    try:
        podman._attest(
            engine, container, spec, image_id=IMAGE_ID, keeper_id="d" * 64,
            keeper_net_inode=os.stat(f"/proc/{os.getpid()}/ns/net").st_ino,
            host_pid_inode=host_pid_inode,
            worker_cgroup=podman._process_cgroup(os.getpid()) or "",
            expect_waiting_gate=False,
        )
    finally:
        podman._namespace_inode = real_namespace_inode
    return container


def test_attest_accepts_the_constructed_boundary(tmp_path):
    spec = _spec(tmp_path)
    container = _attest(tmp_path, spec)
    assert container.pid == os.getpid()
    assert container.pid_inode == 5000


def test_attest_refuses_a_container_in_the_host_pid_namespace(tmp_path):
    spec = _spec(tmp_path)
    with pytest.raises(podman.VerificationFailure) as excinfo:
        _attest(tmp_path, spec, host_pid_inode=5000)
    assert excinfo.value.code == "attestation_failed"


@pytest.mark.parametrize(
    "overrides",
    [
        {"Config.Env": ["HOME=/scratch/home", "HERMES_HOME=/live"]},
        {"Config.Healthcheck": {"Test": ["CMD", "true"]}},
        {"HostConfig.ReadonlyRootfs": False},
        {"HostConfig.CapAdd": ["CAP_SYS_ADMIN"]},
        {"HostConfig.CapDrop": []},
        {"HostConfig.SecurityOpt": []},
        {"HostConfig.PortBindings": {"8080/tcp": [{"HostPort": "8080"}]}},
        {"HostConfig.NetworkMode": "host"},
        {"HostConfig.LogConfig": {"Type": "journald", "Path": ""}},
        {"HostConfig.Ulimits": [{"Name": "RLIMIT_NOFILE", "Soft": 65536, "Hard": 65536}]},
        {"Mounts": [{"Source": "/etc", "Destination": "/etc/passwd", "RW": True}]},
        {"Config.Entrypoint": ["/bin/bash"]},
        {"Config.Cmd": ["-c", "true"]},
        {"Config.User": "0:0"},
        {"State.Pid": 1},
        {"Image": "e" * 64},
    ],
)
def test_attest_refuses_any_boundary_drift(tmp_path, overrides):
    spec = _spec(tmp_path)
    with pytest.raises(podman.VerificationFailure):
        _attest(tmp_path, spec, **overrides)


def test_create_args_are_fixed_and_generated(tmp_path):
    spec = _spec(tmp_path)
    engine = podman.Engine(tmp_path / "run")
    args = podman._create_args(engine, spec)
    joined = " ".join(args)
    for flag in (
        "--pull=never", "--cgroups=disabled", "--read-only", "--read-only-tmpfs=false",
        "--restart=no", "--no-healthcheck", "--image-volume=ignore", "--http-proxy=false",
        "--unsetenv-all", "--userns=keep-id:uid=1000,gid=1000", "--pid=private",
        "--ipc=private", "--uts=private", "--no-hosts", "--cap-drop=ALL",
        "--security-opt=no-new-privileges", "--user=1000:1000", "--stop-signal=SIGKILL",
        "--ulimit=nofile=1024:1024", "--log-driver=k8s-file", "--log-opt=max-size=1mb",
    ):
        assert flag in args, flag
    assert "--network" in args and spec.network in args
    assert "--dns" not in args and "docker.sock" not in joined
    env_values = [args[index + 1] for index, item in enumerate(args) if item == "--env"]
    assert all(value.split("=", 1)[0].isupper() for value in env_values)
    assert not any(value.startswith("HERMES_") for value in env_values)
    assert args[-len(spec.command):] == spec.command


def test_service_environment_rejects_colliding_ids():
    assert podman._service_env([{"id": "db", "port": 5432}]) == {
        "VERIFY_SERVICE_DB_HOST": "127.0.0.1", "VERIFY_SERVICE_DB_PORT": "5432",
    }
    with pytest.raises(podman.VerificationFailure):
        podman._service_env([{"id": "a-b", "port": 1}, {"id": "a_b", "port": 2}])


def test_bounded_collector_keeps_a_trailing_window():
    collector = podman.BoundedCollector(limit=8)
    collector.feed(b"0123456789")
    assert collector.truncated is True
    assert collector.total == 10
    assert collector.text() == "23456789"


def test_image_store_refuses_tampering(tmp_path):
    store = tmp_path / "images"
    store.mkdir(mode=0o700)
    (store / "inventory.json").write_text('{"images": {}}', encoding="utf-8")
    info = os.lstat(store)
    authority = SimpleNamespace(
        image_store={
            "path": str(store), "device": int(info.st_dev), "inode": int(info.st_ino),
            "inventory_digest": "sha256:" + "0" * 64,
        },
    )
    with pytest.raises(podman.VerificationFailure) as excinfo:
        podman.ImageStore(authority)
    assert excinfo.value.code == "image_store_changed"

    authority.image_store["inventory_digest"] = None
    assert podman.ImageStore(authority).entry  # member exists
    with pytest.raises(podman.VerificationFailure):
        podman.ImageStore(authority).entry(IMAGE)


def test_cleanup_journal_requires_unchanged_root_identity(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    journal = tmp_path / "journal.json"
    payload = {
        "protocol": podman.JOURNAL_PROTOCOL, "collection": "x",
        "run_root": {"path": str(root), "device": 1, "inode": 2},
        "run_root_path": str(root), "phase": "created", "containers": [],
        "host_epoch": podman._host_epoch(),
    }
    journal.write_text(json.dumps(payload), encoding="utf-8")
    ok, reason = podman.cleanup_journal(journal)
    assert ok is False and reason == "path_identity_changed"
    assert journal.exists()

    info = os.lstat(root)
    payload["run_root"] = {"path": str(root), "device": int(info.st_dev), "inode": int(info.st_ino)}
    journal.write_text(json.dumps(payload), encoding="utf-8")
    # A real (empty) private engine store so the inventory is readable.
    podman.Engine(root).prepare()
    ok, reason = podman.cleanup_journal(journal)
    assert ok is True and reason == "cleaned"


class _UnstartableEngine:
    """An engine whose executable cannot be started at all."""

    def __init__(self, run_root):
        self.run_root = run_root

    def run(self, *_args, **_kwargs):
        raise podman.VerificationFailure("engine_unavailable", "podman is not installed")


def test_cleanup_removes_the_tree_only_when_nothing_was_allocated(tmp_path):
    run_root = tmp_path / "run"
    run_root.mkdir()
    engine = _UnstartableEngine(run_root)
    ok, reasons = podman._cleanup(engine, [])
    assert ok is True and reasons == ["engine_unavailable_no_allocation:engine_unavailable"]
    assert not run_root.exists()

    run_root.mkdir()
    ok, reasons = podman._cleanup(engine, [], creates=("verifier",))
    assert ok is False and reasons == ["engine_unavailable:engine_unavailable"]
    assert run_root.exists()


def test_cleanup_removes_the_tree_when_the_engine_answers_with_an_error(tmp_path):
    class FailingEngine:
        def __init__(self, run_root):
            self.run_root = run_root

        def run(self, argv, **_kwargs):
            return subprocess.CompletedProcess(argv, 125, b"", b"cannot connect to the engine")

    run_root = tmp_path / "run"
    run_root.mkdir()
    ok, reasons = podman._cleanup(FailingEngine(run_root), [])
    assert ok is True and reasons == ["listing_failed_no_allocation"]
    assert not run_root.exists()

    run_root.mkdir()
    ok, reasons = podman._cleanup(FailingEngine(run_root), [], creates=("verifier",))
    assert ok is False and reasons == ["listing_failed", "listing_failed_after_rm"]
    assert run_root.exists()


def test_cleanup_survives_an_engine_that_dies_mid_removal(tmp_path):
    class VanishingEngine:
        def __init__(self, run_root):
            self.run_root = run_root
            self.calls = 0

        def run(self, argv, **_kwargs):
            self.calls += 1
            if argv[0] == "ps":
                return subprocess.CompletedProcess(argv, 0, b"a" * 64 + b"\n", b"")
            raise podman.VerificationFailure("engine_unavailable", "podman disappeared")

    run_root = tmp_path / "run"
    run_root.mkdir()
    container = podman.Container("verifier", "verifier", "image", "tag")
    container.id = "a" * 64
    ok, reasons = podman._cleanup(VanishingEngine(run_root), [container])
    assert ok is False
    assert reasons[:2] == ["rm_failed:verifier:engine_unavailable", "still_listed:verifier"]
    assert run_root.exists()


def test_absent_run_root_keeps_a_journal_while_recorded_process_lives(tmp_path):
    journal = tmp_path / "journal.json"
    journal.write_text(json.dumps({
        "protocol": podman.JOURNAL_PROTOCOL, "collection": "x", "phase": "started",
        "host_epoch": podman._host_epoch(),
        "run_root_path": str(tmp_path / "missing"), "run_root": {"device": 1, "inode": 2},
        "containers": [{
            "name": "verifier", "pid": os.getpid(),
            "pid_start": podman._process_start_time(os.getpid()),
        }],
    }), encoding="utf-8")
    ok, reason = podman.cleanup_journal(journal)
    assert ok is False and reason == "root_absent_processes_alive"
    assert journal.exists()


@pytest.mark.parametrize(
    ("recorded", "current"),
    [("unknown", "unknown"), ("", "unknown"), ("a-previous-boot", ""), ("", "")],
)
def test_unavailable_boot_identities_retain_the_journal(tmp_path, monkeypatch, recorded, current):
    journal = tmp_path / "journal.json"
    journal.write_text(json.dumps({
        "protocol": podman.JOURNAL_PROTOCOL, "collection": "x", "phase": "started",
        "run_root_path": str(tmp_path / "missing"), "run_root": {"device": 1, "inode": 2},
        "containers": [], "host_epoch": recorded,
    }), encoding="utf-8")
    monkeypatch.setattr(podman, "_host_epoch", lambda: current)
    ok, reason = podman.cleanup_journal(journal)
    assert ok is False and reason == "boot_identity_unknown"
    assert journal.exists()


def test_unknown_boot_identity_retains_the_journal(tmp_path, monkeypatch):
    journal = tmp_path / "journal.json"
    journal.write_text(json.dumps({
        "protocol": podman.JOURNAL_PROTOCOL, "collection": "x", "phase": "started",
        "host_epoch": "a-previous-boot",
        "run_root_path": str(tmp_path / "missing"), "run_root": {"device": 1, "inode": 2},
        "containers": [{
            "name": "verifier", "pid": os.getpid(),
            "pid_start": podman._process_start_time(os.getpid()),
        }],
    }), encoding="utf-8")
    monkeypatch.setattr(podman, "_host_epoch", lambda: "unknown")
    ok, reason = podman.cleanup_journal(journal)
    assert ok is False and reason == "boot_identity_unknown"
    assert journal.exists()


def test_stale_boot_never_reuses_recorded_process_identity(tmp_path):
    journal = tmp_path / "journal.json"
    journal.write_text(json.dumps({
        "protocol": podman.JOURNAL_PROTOCOL, "collection": "x", "phase": "started",
        "host_epoch": "a-previous-boot",
        "run_root_path": str(tmp_path / "missing"), "run_root": {"device": 1, "inode": 2},
        "containers": [{
            "name": "verifier", "pid": os.getpid(),
            "pid_start": podman._process_start_time(os.getpid()),
        }],
    }), encoding="utf-8")
    ok, reason = podman.cleanup_journal(journal)
    assert ok is True and reason == "absent"
    assert not journal.exists()


def test_retained_prefix_cannot_starve_a_later_journal(tmp_path, monkeypatch):
    import hermes_cli.kanban_worker_runtime as runtime

    home = tmp_path / "home"
    prep = home / "worker-executions" / "prep"
    prep.mkdir(parents=True)
    for index in range(podman.MAX_REAP_JOURNALS):
        (prep / f"{index:03d}.json").write_text("{not json", encoding="utf-8")
    eligible = prep / "zzz.json"
    eligible.write_text(json.dumps({
        "protocol": podman.JOURNAL_PROTOCOL, "phase": "started",
        "host_epoch": podman._host_epoch(), "deadline_epoch": 0,
        "run_root_path": str(tmp_path / "missing"), "run_root": {"device": 1, "inode": 2},
        "containers": [],
    }), encoding="utf-8")
    monkeypatch.setattr("hermes_constants.get_process_hermes_home", lambda: home)

    runtime._reap_worker_execution_journals()
    assert eligible.exists()  # the retained prefix owns this pass's whole budget
    runtime._reap_worker_execution_journals()
    assert not eligible.exists()


def test_dispatch_reap_rotates_across_preparation_roots(tmp_path, monkeypatch):
    import hermes_cli.kanban_worker_runtime as runtime
    from hermes_cli import worker_execution_podman

    home = tmp_path / "home"
    for profile in ("alpha", "beta"):
        (home / "profiles" / profile / "worker-executions" / "prep").mkdir(parents=True)

    calls: list[str] = []

    def fake(directory, *, max_journals):
        calls.append(str(directory))
        return [], 1

    monkeypatch.setattr(worker_execution_podman, "reap_journals", fake)
    monkeypatch.setattr("hermes_constants.get_process_hermes_home", lambda: home)
    runtime._reap_worker_execution_journals()
    first = list(calls)
    calls.clear()
    runtime._reap_worker_execution_journals()
    second = list(calls)
    assert first and second
    assert len(set(first)) == 2
    assert first[0] != second[0]


def test_reap_counts_skipped_journals_and_rotates_oldest_first(tmp_path):
    (tmp_path / "a.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "b.json").write_text(json.dumps({
        "protocol": podman.JOURNAL_PROTOCOL, "worker_pid": os.getpid(),
        "worker_start": podman._process_start_time(os.getpid()),
        "host_epoch": podman._host_epoch(), "deadline_epoch": 0,
        "run_root_path": str(tmp_path / "missing"), "run_root": {}, "containers": [],
    }), encoding="utf-8")
    results, examined = podman.reap_journals(tmp_path, max_journals=1)
    assert examined == 1
    assert [row["journal"] for row in results] == ["a.json"]

    results, examined = podman.reap_journals(tmp_path, max_journals=1)
    assert examined == 1
    assert results == []


def test_reap_skips_live_owner_and_unelapsed_deadlines(tmp_path):
    journal_dir = tmp_path / "journals"
    journal_dir.mkdir()
    alive = {
        "protocol": podman.JOURNAL_PROTOCOL, "worker_pid": os.getpid(),
        "worker_start": podman._process_start_time(os.getpid()),
        "host_epoch": podman._host_epoch(), "deadline_epoch": 0,
        "run_root_path": str(tmp_path / "missing"), "run_root": {}, "containers": [],
    }
    (journal_dir / "alive.json").write_text(json.dumps(alive), encoding="utf-8")
    results, _examined = podman.reap_journals(journal_dir)
    assert results == []
    assert (journal_dir / "alive.json").exists()

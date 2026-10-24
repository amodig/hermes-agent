"""Boundary regressions for the fixed Podman verification executor.

These pin attestation and cleanup decisions against crafted engine JSON so a weakened
check fails loudly. The real container/namespace/readiness proof is
``scripts/verify_worker_execution.py``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import worker_execution_podman as podman

pytestmark = pytest.mark.skipif(not __import__("sys").platform.startswith("linux"), reason="Linux-only boundary")

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
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def text(self, args, *, timeout=60.0):  # noqa: ANN001 - test double
        return json.dumps([self.payload])


def _container() -> podman.Container:
    container = podman.Container("verifier", "verifier", IMAGE, "localhost/worker-execution:pg")
    container.id = "a" * 64
    return container


def _attest(tmp_path, spec, **overrides):
    payload = _inspect_for(spec, overrides=overrides)
    engine = _FakeEngine(payload)
    container = _container()
    podman._attest(
        engine, container, spec, image_id=IMAGE_ID, keeper_id="d" * 64,
        keeper_net_inode=payload["NetworkSettings"]["SandboxKey"] and os.stat(
            f"/proc/{os.getpid()}/ns/net").st_ino,
        worker_cgroup=podman._process_cgroup(os.getpid()) or "",
        expect_waiting_gate=False,
    )
    return container


def test_attest_accepts_the_constructed_boundary(tmp_path):
    spec = _spec(tmp_path)
    container = _attest(tmp_path, spec)
    assert container.pid == os.getpid()


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
    }
    journal.write_text(json.dumps(payload), encoding="utf-8")
    ok, reason = podman.cleanup_journal(journal)
    assert ok is False and reason == "path_identity_changed"
    assert journal.exists()

    info = os.lstat(root)
    payload["run_root"] = {"path": str(root), "device": int(info.st_dev), "inode": int(info.st_ino)}
    journal.write_text(json.dumps(payload), encoding="utf-8")
    ok, reason = podman.cleanup_journal(journal)
    assert ok is True and reason == "cleaned"


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
    results = podman.reap_journals(journal_dir)
    assert results == []
    assert (journal_dir / "alive.json").exists()

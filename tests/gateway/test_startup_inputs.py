"""Gateway input evidence ignores status ticks, never reloads away startup history."""
import json
import shutil

import pytest

from hermes_cli import gateway_startup_inputs as inputs
from hermes_cli.env_loader import load_hermes_dotenv


def test_startup_input_snapshot_is_published_once_before_reload(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    secondary = home / "profiles" / "secondary"
    secondary.mkdir(parents=True)
    managed = tmp_path / "managed"
    managed.mkdir()
    project_env = tmp_path / "runtime.env"
    (home / ".env").write_text(f"HERMES_MANAGED_DIR={managed}\n")
    (home / "config.yaml").write_text("{}")
    (secondary / ".env").write_text("FIXTURE_SECONDARY=metadata-fixture-value\n")
    (secondary / ".op.env").write_text("FIXTURE_BOOTSTRAP=metadata-fixture-value\n")
    (managed / ".env").write_text("FIXTURE_MANAGED=metadata-fixture-value\n")
    (managed / "config.yaml").write_text("{}")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(tmp_path / "not-managed"))
    for name, value in (("_started", False), ("_published", False), ("_pending", None), ("_context", None)):
        monkeypatch.setattr(inputs, name, value)

    inputs.begin_gateway_startup_inputs(home, project_env)
    snapshot_path = home / "gateway_startup_inputs.json"
    assert not snapshot_path.exists()  # A competing startup has not claimed ownership.
    load_hermes_dotenv(hermes_home=home, project_env=project_env, load_external_secrets=False)
    inputs.publish_gateway_startup_inputs()
    original = snapshot_path.read_bytes()
    snapshot = json.loads(original)
    identity = {"pid": snapshot["pid"], "start_time": snapshot["start_time"]}
    assert "metadata-fixture-value" not in original.decode()
    assert snapshot["inputs"][str(home / ".op.env")]["state"] is None
    for profile in (home, secondary):
        for name in (".env", ".op.env", "config.yaml", "gateway.json"):
            assert str(profile / name) in snapshot["inputs"]
        assert snapshot["inputs"][str(profile / "gateway.json")]["state"] is None
    assert str(managed / ".env") in snapshot["inputs"]
    assert str(managed / "config.yaml") in snapshot["inputs"]
    assert snapshot_path.stat().st_mode & 0o777 == 0o600

    status = home / "gateway_state.json.tmp"
    status.write_text('{"gateway_state":"running"}')
    status.replace(home / "gateway_state.json")
    inputs.require_unchanged_startup_inputs(home, project_env, [secondary], **identity)
    (home / ".op.env").write_text("FIXTURE_NEW=value\n")
    inputs.begin_gateway_startup_inputs(home, project_env)
    load_hermes_dotenv(hermes_home=home, project_env=project_env, load_external_secrets=False)
    inputs.publish_gateway_startup_inputs()
    assert snapshot_path.read_bytes() == original
    with pytest.raises(ValueError, match="changed after process start"):
        inputs.require_unchanged_startup_inputs(home, project_env, [secondary], **identity)


@pytest.mark.parametrize("change", ["created", "deleted", "replaced", "hash", "managed_deleted", "missing", "pid", "start_time"])
def test_startup_snapshot_refuses_changed_inputs_and_identity(tmp_path, monkeypatch, change):
    home = tmp_path / "home"
    home.mkdir()
    project_env = tmp_path / "runtime.env"
    managed = tmp_path / "managed"
    managed.mkdir()
    (managed / "config.yaml").write_text("{}")
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    source = home / ".op.env"
    if change != "created":
        source.write_text("FIXTURE=before\n")
    snapshot = inputs.capture_startup_inputs(home, project_env)
    inputs.write_startup_inputs(home, snapshot)
    identity = {"pid": snapshot["pid"], "start_time": snapshot["start_time"]}
    snapshot_path = home / "gateway_startup_inputs.json"
    if change == "created":
        source.write_text("FIXTURE=after\n")
    elif change == "deleted":
        source.unlink()
    elif change == "replaced":
        replacement = home / "replacement"
        shutil.copy2(source, replacement)
        replacement.replace(source)
    elif change == "hash":
        source.write_text("FIXTURE=after!\n")
        # Even equal filesystem metadata cannot stand in for the recorded bytes.
        state = snapshot["inputs"][str(source)]["state"]
        changed = inputs.capture_startup_inputs(home, project_env)["inputs"][str(source)]["state"]
        snapshot["inputs"][str(source)]["state"] = {**changed, "sha256": state["sha256"]}
        snapshot_path.write_text(json.dumps(snapshot))
    elif change == "managed_deleted":
        shutil.rmtree(managed)
    elif change == "missing":
        snapshot_path.unlink()
    else:
        identity[change] += 1
    with pytest.raises((OSError, ValueError)):
        inputs.require_unchanged_startup_inputs(home, project_env, **identity)


@pytest.mark.parametrize("ignored", [".backup", "default", "ghost", "deleted"])
def test_startup_snapshot_ignores_non_live_profiles(tmp_path, monkeypatch, ignored):
    from hermes_constants import mark_named_profile_deleted

    home = tmp_path / "home"
    live = home / "profiles" / "reviewer"
    live.mkdir(parents=True)
    (live / ".env").write_text("FIXTURE_LIVE=value\n")
    stray = home / "profiles" / ignored
    stray.mkdir()
    # .op.env is an input, but not an identity marker.
    (stray / (".op.env" if ignored == "ghost" else ".env")).symlink_to("missing")
    if ignored == "deleted":
        mark_named_profile_deleted(stray)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(tmp_path / "managed"))
    for name, value in (("_started", False), ("_published", False), ("_pending", None), ("_context", None)):
        monkeypatch.setattr(inputs, name, value)

    project_env = tmp_path / "runtime.env"
    inputs.begin_gateway_startup_inputs(home, project_env)
    inputs.publish_gateway_startup_inputs()
    snapshot_path = home / "gateway_startup_inputs.json"
    assert snapshot_path.is_file(), "an irrelevant profile directory must not prevent publication"
    snapshot = json.loads(snapshot_path.read_bytes())
    assert snapshot["inputs"][str(live / ".env")]["state"] is not None
    assert not any(str(stray) + "/" in path for path in snapshot["inputs"])
    inputs.require_unchanged_startup_inputs(
        home, project_env, [live], pid=snapshot["pid"], start_time=snapshot["start_time"])

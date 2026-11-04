"""Gateway input evidence ignores status ticks, never reloads away startup history."""
import json
import shutil

import pytest

from hermes_cli import gateway_startup_inputs as inputs
from hermes_cli.env_loader import load_hermes_dotenv


@pytest.mark.parametrize("unterminated", [None, "home", "project", "managed", "secondary"])
@pytest.mark.parametrize("edit_before_publish", [False, True])
def test_startup_input_snapshot_is_published_once_before_reload(
    tmp_path, monkeypatch, unterminated, edit_before_publish,
):
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
    project_env.write_text("FIXTURE_PROJECT=metadata-fixture-value\n")
    if unterminated:
        source = {"home": home / ".env", "project": project_env,
                  "managed": managed / ".env", "secondary": secondary / ".env"}[unterminated]
        source.write_text(source.read_text().rstrip("\n"))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(tmp_path / "not-managed"))
    for name, value in (("_started", False), ("_published", False), ("_pending", None), ("_context", None)):
        monkeypatch.setattr(inputs, name, value)

    inputs.begin_gateway_startup_inputs(home, project_env)
    snapshot_path = home / "gateway_startup_inputs.json"
    assert not snapshot_path.exists()  # A competing startup has not claimed ownership.
    load_hermes_dotenv(hermes_home=home, project_env=project_env, load_external_secrets=False)
    if edit_before_publish:
        (home / ".env").write_text("FIXTURE_EDIT=after-start\n")
    inputs.publish_gateway_startup_inputs()
    if edit_before_publish:
        assert not snapshot_path.exists(), "publication must refuse a genuine post-start edit"
        return
    assert snapshot_path.is_file(), "loader normalization must not invalidate startup evidence"
    if unterminated:
        assert source.read_bytes().endswith(b"\n")
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


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("selector", ["HERMES_MANAGED_DIR", "HERMES_HOME"])
def test_secret_selected_authority_is_captured_by_the_startup_writer(tmp_path, monkeypatch, selector):
    from hermes_cli import env_loader
    from hermes_cli.managed_scope import get_managed_dir

    home = tmp_path / "home"
    managed = tmp_path / "managed"
    alternate = tmp_path / "alternate"
    for path in (home, managed, alternate):
        path.mkdir()
    helper = tmp_path / "helper"
    helper.write_text(f"#!/bin/sh\nprintf '{selector}={alternate}\\n'\n")
    helper.chmod(0o700)
    (home / "config.yaml").write_text(
        "secrets:\n  command:\n    enabled: true\n    override_existing: true\n"
        f"    command: {helper}\n")
    (alternate / "config.yaml").write_text("{}")
    (alternate / ".env").write_text("FIXTURE_SELECTED=value")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    monkeypatch.delenv("FIXTURE_SELECTED", raising=False)
    for name, value in (("_started", False), ("_published", False), ("_pending", None), ("_context", None)):
        monkeypatch.setattr(inputs, name, value)
    env_loader.reset_secret_source_cache()
    project = tmp_path / "runtime.env"
    try:
        inputs.begin_gateway_startup_inputs(home, project)
        load_hermes_dotenv(hermes_home=home, project_env=project)
        inputs.publish_gateway_startup_inputs()
        snapshot = json.loads((home / "gateway_startup_inputs.json").read_text())
        assert snapshot["authority"] == {
            "home": str(alternate if selector == "HERMES_HOME" else home),
            "managed_dir": str(get_managed_dir()),
        }
        for name in (".env", "config.yaml"):
            assert snapshot["inputs"][str(alternate / name)]["state"]["sha256"] == (
                inputs._file_metadata(alternate / name)["sha256"])
        # Verification starts with exec-time selectors and deliberately skips sources.
        monkeypatch.setenv(selector, str(managed if selector == "HERMES_MANAGED_DIR" else home))
        with pytest.raises(ValueError, match="authority selection differs"):
            inputs.require_unchanged_startup_inputs(
                home, project, pid=snapshot["pid"], start_time=snapshot["start_time"])
    finally:
        env_loader.reset_secret_source_cache()


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("secondary", [False, True])
@pytest.mark.parametrize("proxy", ["", "http://proxy.example:8080/", "http://user:fixture-credential@proxy.example"])
def test_startup_route_snapshot_uses_hydrated_adapter_inputs(tmp_path, monkeypatch, secondary, proxy):
    from agent.secret_scope import (
        build_profile_secret_scope, reset_multiplex_context, reset_secret_scope,
        set_multiplex_context, set_secret_scope,
    )
    from gateway.config import PlatformConfig
    from hermes_cli import env_loader
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from plugins.platforms.irc.adapter import IRCAdapter

    home = tmp_path / "home"
    profile = home / "profiles/secondary" if secondary else home
    profile.mkdir(parents=True)
    helper = tmp_path / "helper"
    helper.write_text(
        "#!/bin/sh\nprintf 'IRC_SERVER=irc.example.org\\nIRC_PORT=6697\\nIRC_USE_TLS=true\\n"
        f"IRC_PROXY_URL={proxy}\\nIRC_SASL_PASSWORD=fixture-credential\\n"
        "IRC_SERVER_PASSWORD=fixture-credential\\nIRC_NICKSERV_PASSWORD=fixture-credential\\n'\n")
    helper.chmod(0o700)
    (profile / "config.yaml").write_text(
        "secrets:\n  command:\n    enabled: true\n    override_existing: true\n"
        f"    command: {helper}\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(tmp_path / "managed"))
    for key in ("IRC_SERVER", "IRC_PORT", "IRC_USE_TLS", "IRC_PROXY_URL",
                "IRC_SASL_PASSWORD", "IRC_SERVER_PASSWORD", "IRC_NICKSERV_PASSWORD"):
        monkeypatch.setenv(key, "")
    for name, value in (("_started", False), ("_published", False), ("_pending", None), ("_context", None)):
        monkeypatch.setattr(inputs, name, value)
    env_loader.reset_secret_source_cache()
    project = tmp_path / "runtime.env"
    inputs.begin_gateway_startup_inputs(home, project)
    if secondary:
        env_loader.hydrate_profile_secret_sources(profile)
    else:
        load_hermes_dotenv(hermes_home=home, project_env=project)
    home_token = set_hermes_home_override(profile)
    multiplex_token = set_multiplex_context(secondary)
    secret_token = set_secret_scope(build_profile_secret_scope(profile) if secondary else None)
    try:
        adapter = IRCAdapter(PlatformConfig(enabled=True, extra={
            "server": "127.0.0.1", "port": 16697, "use_tls": False, "channel": "#agents",
        }))
        inputs.publish_gateway_startup_inputs()
        path = home / "gateway_startup_inputs.json"
        original = path.read_bytes()
        snapshot = json.loads(original)
        route = snapshot["routes"][str(profile)]["irc"]
        assert route == adapter.startup_route()
        assert route == {
            "server": "irc.example.org", "port": 6697, "use_tls": True,
            "proxy": ({"invalid": True} if "fixture-credential" in proxy else
                      {"scheme": "http", "hostname": "proxy.example", "port": 8080} if proxy else None),
        }
        assert b"fixture-credential" not in original
        assert not {"sasl_account", "sasl_password", "server_password", "nickserv_password"} & route.keys()
        adapter.server = "later.example"
        inputs.record_gateway_startup_route("irc", adapter.startup_route())
        inputs.publish_gateway_startup_inputs()
        assert path.read_bytes() == original
    finally:
        reset_secret_scope(secret_token)
        reset_multiplex_context(multiplex_token)
        reset_hermes_home_override(home_token)
        env_loader.reset_secret_source_cache()

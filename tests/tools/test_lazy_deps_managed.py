"""Managed-install guard in :func:`tools.lazy_deps.ensure` (#48628).

A package-manager install (NixOS, and anything else shipping Hermes from a
read-only store) cannot receive lazy pip installs: the venv's site-packages
lives in the store, so the uv -> pip -> ensurepip ladder burns ~15s
bootstrapping ensurepip only to fail. ``ensure()`` must fail fast instead.
"""

import importlib

import pytest


FEATURE = "provider.anthropic"


@pytest.fixture
def lazy_deps():
    """Resolve the live module after update tests may reload it."""
    return importlib.import_module("tools.lazy_deps")


@pytest.fixture(autouse=True)
def _missing_and_installable(monkeypatch, lazy_deps):
    """Reach the guard: deps missing, installs allowed, no durable target.

    ``_allow_lazy_installs`` is patched explicitly so the suite does not
    depend on the host's ~/.hermes/config.yaml (a local
    ``allow_lazy_installs: false`` otherwise short-circuits with a different
    rejection reason).
    """
    monkeypatch.setattr(lazy_deps, "feature_missing", lambda _f: ("some-pkg==1.0",))
    monkeypatch.setattr(lazy_deps, "_allow_lazy_installs", lambda: True)
    monkeypatch.setattr(lazy_deps, "_lazy_install_target", lambda: None)


def _no_installer(monkeypatch, lazy_deps):
    """Fail loudly if the guard lets execution reach the install ladder."""
    def _boom(*_a, **_kw):
        raise AssertionError("guard let execution reach the install ladder")

    monkeypatch.setattr(lazy_deps.subprocess, "run", _boom)


def _sentinel_installer(monkeypatch, lazy_deps, *, success=False):
    """Replace the installer with a deterministic, in-process result."""
    calls = []

    def _install(specs, **_kwargs):
        calls.append(specs)
        return lazy_deps._InstallResult(success, "", "sentinel installer")

    monkeypatch.setattr(lazy_deps, "_venv_pip_install", _install)
    return calls


def test_nixos_install_fails_fast_without_touching_the_installer(monkeypatch, lazy_deps):
    monkeypatch.setattr("hermes_cli.config.get_managed_system", lambda: "nixos")
    _no_installer(monkeypatch, lazy_deps)

    with pytest.raises(lazy_deps.FeatureUnavailable) as excinfo:
        lazy_deps.ensure(FEATURE, prompt=False)

    assert "nixos" in excinfo.value.reason
    # The managed-runtime reason is part of ensure()'s failure contract.
    assert excinfo.value.reason.startswith("unsupported ")


def test_managed_guard_is_classified_as_skipped_by_refresh(monkeypatch, lazy_deps):
    monkeypatch.setattr("hermes_cli.config.get_managed_system", lambda: "nixos")
    monkeypatch.setattr(lazy_deps, "active_features", lambda: [FEATURE])
    _no_installer(monkeypatch, lazy_deps)

    result = lazy_deps.refresh_active_features()

    assert result[FEATURE].startswith("skipped:")
    assert "nixos-managed installs" in result[FEATURE]


def test_unmanaged_install_is_not_blocked_by_the_guard(monkeypatch, lazy_deps):
    """On a normal pip install the guard must be transparent."""
    monkeypatch.setattr("hermes_cli.config.get_managed_system", lambda: None)
    _no_installer(monkeypatch, lazy_deps)
    calls = _sentinel_installer(monkeypatch, lazy_deps)

    with pytest.raises(lazy_deps.FeatureUnavailable) as excinfo:
        lazy_deps.ensure(FEATURE, prompt=False)

    assert calls == [("some-pkg==1.0",)]
    assert excinfo.value.reason == "pip install failed: sentinel installer"


def test_durable_install_target_overrides_the_guard(monkeypatch, lazy_deps, tmp_path):
    """A writable target lets managed deployments use the install path."""
    monkeypatch.setattr("hermes_cli.config.get_managed_system", lambda: "nixos")
    monkeypatch.setattr(lazy_deps, "_lazy_install_target", lambda: tmp_path)
    _no_installer(monkeypatch, lazy_deps)
    calls = _sentinel_installer(monkeypatch, lazy_deps)

    with pytest.raises(lazy_deps.FeatureUnavailable) as excinfo:
        lazy_deps.ensure(FEATURE, prompt=False)

    assert calls == [("some-pkg==1.0",)]
    assert excinfo.value.reason == "pip install failed: sentinel installer"
    assert "nixos" not in excinfo.value.reason.lower()


def test_platform_unsupported_takes_precedence(monkeypatch, lazy_deps):
    """A platform-specific reason is more actionable than 'managed install'.

    Also required for consistency: refresh_active_features pre-checks
    _unsupported_feature_reason before calling ensure().
    """
    monkeypatch.setattr("hermes_cli.config.get_managed_system", lambda: "nixos")
    monkeypatch.setattr(
        lazy_deps, "_unsupported_feature_reason", lambda _f: "unsupported on win32"
    )
    _no_installer(monkeypatch, lazy_deps)

    with pytest.raises(lazy_deps.FeatureUnavailable) as excinfo:
        lazy_deps.ensure(FEATURE, prompt=False)

    assert excinfo.value.reason == "unsupported on win32"


def test_unreadable_config_fails_open(monkeypatch, lazy_deps):
    """A broken config must not block installs on a normal pip install."""
    def _raise():
        raise RuntimeError("config unreadable")

    monkeypatch.setattr("hermes_cli.config.get_managed_system", _raise)
    _no_installer(monkeypatch, lazy_deps)
    calls = _sentinel_installer(monkeypatch, lazy_deps)

    with pytest.raises(lazy_deps.FeatureUnavailable) as excinfo:
        lazy_deps.ensure(FEATURE, prompt=False)

    assert calls == [("some-pkg==1.0",)]
    assert excinfo.value.reason == "pip install failed: sentinel installer"



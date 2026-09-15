"""Observable generation boundaries; all mutations target synthetic installations."""
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading

import pytest

from hermes_cli import kanban_runtime as runtime
from hermes_cli import kanban_runtime_generation as generation


REPOSITORY = Path(__file__).resolve().parents[2]


def _write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


@pytest.fixture
def runtime_storage(tmp_path, monkeypatch):
    with tempfile.TemporaryDirectory(prefix="runtime-storage-", dir=tmp_path) as raw:
        storage = Path(raw)
        monkeypatch.setattr(generation, "_runtime_storage_root", lambda: storage)
        yield storage


@pytest.fixture
def installation(tmp_path, monkeypatch, runtime_storage):
    source = tmp_path / "checkout"
    dependencies = tmp_path / "site-packages"
    plugins = tmp_path / "home" / "plugins"
    external = tmp_path / "external"
    native_probe = tmp_path / "native_probe.py"
    for name in ("kanban_runtime.py", "kanban_runtime_generation.py"):
        destination = source / "hermes_cli" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPOSITORY / "hermes_cli" / name, destination)
    for name in ("__init__.py", "base.py"):
        destination = source / "providers" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPOSITORY / "providers" / name, destination)
    _write(source / "hermes_cli" / "__init__.py", '__version__ = "fixture"\n')
    _write(source / "hermes_constants.py", "import os\nfrom pathlib import Path\ndef get_hermes_home(): return Path(os.environ['HERMES_HOME'])\n")
    _write(source / "early.py", "VALUE = 'old'\n")
    _write(source / "late.py", "VALUE = 'old'\n")
    _write(dependencies / "startup_sdk.py", "VALUE = 'old'\n")
    _write(dependencies / "unseeded_sdk" / "__init__.py", "")
    _write(dependencies / "unseeded_sdk" / "selected.py", "VALUE = 'old'\n")
    (dependencies / "unseeded_sdk" / "data.bin").write_bytes(b"old\x00bytes")
    _write(dependencies / "unseeded_sdk-1.0.dist-info" / "METADATA", "Metadata-Version: 2.1\nName: unseeded-sdk\nVersion: 1.0\n")
    # Deliberately incomplete RECORD: provisioning must not select an import closure.
    _write(dependencies / "unseeded_sdk-1.0.dist-info" / "RECORD", "unseeded_sdk/__init__.py,,\n")
    _write(external / "external_plugin" / "__init__.py", "from pathlib import Path\ndef value(): return Path(__file__).with_name('resource.txt').read_text()\n")
    _write(external / "external_plugin" / "resource.txt", "old")
    _write(external / "implementation.py", "VALUE = 'old'\n")
    _write(dependencies / "editable_hook.py", f"""
import importlib.util
from pathlib import Path
import sys
MAPPING = {{'editable_plugin': {str(external / 'implementation')!r}}}
class Finder:
    def find_spec(self, fullname, path=None, target=None):
        if fullname in MAPPING:
            candidate = Path(MAPPING[fullname]).with_suffix('.py')
            if candidate.is_file():
                return importlib.util.spec_from_file_location(fullname, candidate)
sys.meta_path.insert(0, Finder())
""")
    _write(dependencies / "external.pth", str(external) + "\nimport external_plugin\nimport editable_hook\n")
    _write(plugins / "model-providers" / "fixture" / "__init__.py", """
from pathlib import Path
from providers import register_provider
from providers.base import ProviderProfile
class Fixture(ProviderProfile):
    def resolve_aux_model(self, *, vision=False):
        return Path(__file__).with_name('resource.txt').read_text()
register_provider(Fixture(name='generation-fixture'))
""")
    _write(plugins / "model-providers" / "fixture" / "resource.txt", "old")
    _write(native_probe, "VALUE = 'old'\n")
    _write(source / "hermes_cli" / "main.py", """
import importlib
import importlib.metadata
import importlib.resources
import json
import os
from pathlib import Path
import sys
import early
import startup_sdk
from hermes_cli.kanban_runtime import runtime_identity, assert_runtime_import_root
print(json.dumps({'source': early.VALUE, 'dependency': startup_sdk.VALUE}), flush=True)
name = sys.stdin.readline().strip()
late = importlib.import_module('late')
sdk = importlib.import_module(name)
import external_plugin
import editable_plugin
import native_probe
import providers
profile = providers.get_provider_profile('generation-fixture')
assert_runtime_import_root()
print(json.dumps({
    'dynamic': sdk.VALUE,
    'late': late.VALUE,
    'resource': importlib.resources.files('unseeded_sdk').joinpath('data.bin').read_bytes().hex(),
    'metadata': importlib.metadata.version('unseeded-sdk'),
    'external': external_plugin.value(),
    'editable': editable_plugin.VALUE,
    'provider': profile.resolve_aux_model(),
    'native': native_probe.VALUE,
    'prefix': sys.prefix,
    'stdlib': json.__file__,
    'identity': runtime_identity().as_dict(),
}), flush=True)
""")
    monkeypatch.setenv("HERMES_HOME", str(plugins.parent))
    monkeypatch.delenv("HERMES_KANBAN_RUNTIME_GENERATION", raising=False)
    monkeypatch.delenv("HERMES_ENABLE_PROJECT_PLUGINS", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOOTSTRAP_PATH", raising=False)
    for _, variable in runtime._RUNTIME_RESOURCE_ROOTS:
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setattr(generation, "_runtime_import_roots", lambda root: [dependencies, external, plugins])
    native_inputs = generation._python_runtime_inputs
    def native_with_probe():
        paths, excluded, executable = native_inputs()
        original_stdlib = generation._stdlib_roots()[0]
        origin = next(origin for origin in sorted(paths, key=len, reverse=True) if original_stdlib.is_relative_to(origin))
        stdlib = Path(paths[origin]) / original_stdlib.relative_to(origin)
        paths[str(native_probe)] = str(Path(stdlib) / "native_probe.py")
        return paths, excluded, executable
    monkeypatch.setattr(generation, "_python_runtime_inputs", native_with_probe)
    def prepare():
        return generation.prepare_runtime_generation(runtime.runtime_identity(source))
    return source, dependencies, plugins, external, native_probe, prepare


def _launch(prepared):
    return subprocess.Popen(
        prepared.command_prefix, env={**os.environ, **prepared.env},
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )


def _finish(process):
    output, error = process.communicate("unseeded_sdk.selected\n", timeout=60)
    assert process.returncode == 0, error
    return json.loads(output.splitlines()[-1])


def test_cold_generation_preserves_dynamic_imports_plugins_and_native_runtime(installation):
    source, dependencies, plugins, external, native_probe, prepare = installation
    prepared = prepare()
    child = _launch(prepared)
    try:
        assert json.loads(child.stdout.readline()) == {"source": "old", "dependency": "old"}
        # In-place writes, not rename-only installers: shared source inodes would leak.
        for path in (source / "early.py", source / "late.py", dependencies / "startup_sdk.py", dependencies / "unseeded_sdk" / "selected.py", native_probe, external / "implementation.py"):
            path.write_text("VALUE = 'new'\n", encoding="utf-8")
        (dependencies / "unseeded_sdk" / "data.bin").write_bytes(b"new\x00bytes")
        (external / "external_plugin" / "resource.txt").write_text("new", encoding="utf-8")
        (plugins / "model-providers" / "fixture" / "resource.txt").write_text("new", encoding="utf-8")
        old = _finish(child)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
    assert {key: old[key] for key in ("dynamic", "late", "external", "editable", "provider", "native")} == dict.fromkeys(("dynamic", "late", "external", "editable", "provider", "native"), "old")
    assert old["resource"] == b"old\x00bytes".hex()
    assert old["metadata"] == "1.0"
    assert Path(old["prefix"]).is_relative_to(prepared.root / "python")
    assert Path(old["stdlib"]).is_relative_to(prepared.root / "python")
    assert runtime.same_code_identity(prepared.identity, old["identity"])
    following = prepare()
    newer = _launch(following)
    try:
        assert json.loads(newer.stdout.readline()) == {"source": "new", "dependency": "new"}
        fresh = _finish(newer)
    finally:
        if newer.poll() is None:
            newer.kill()
            newer.wait()
    assert {key: fresh[key] for key in ("dynamic", "late", "external", "editable", "provider", "native")} == dict.fromkeys(("dynamic", "late", "external", "editable", "provider", "native"), "new")
    assert fresh["resource"] == b"new\x00bytes".hex()
    assert prepared.identity.dependency_fingerprint != following.identity.dependency_fingerprint


def test_editable_namespace_path_hook_keeps_delayed_imports_in_generation(installation):
    source, dependencies, _, external, _, prepare = installation
    namespace_source = external / "namespace-source"
    uncaptured = external.parent / "uncaptured"
    sentinel = "fixture-namespace.path-token"
    _write(namespace_source / "selected.py", "VALUE = 'old'\n")
    _write(uncaptured / "selected.py", "raise AssertionError('mutable namespace loaded')\n")
    # Setuptools uses a PathEntryFinder with an opaque placeholder, including
    # virtual parent namespaces that have no physical search directory at all.
    _write(dependencies / "namespace_hook.py", f"""
from importlib.machinery import ModuleSpec
import sys
PATH_PLACEHOLDER = {sentinel!r}
NAMESPACES = {{
    'fixture_namespace': [],
    'fixture_namespace.branch': [{str(namespace_source)!r}],
    'fixture_namespace.uncaptured': [{str(uncaptured)!r}],
}}
class NamespaceFinder:
    @classmethod
    def path_hook(cls, path):
        if path == PATH_PLACEHOLDER:
            return cls
        raise ImportError
    @classmethod
    def find_spec(cls, fullname, target=None):
        if fullname in NAMESPACES:
            spec = ModuleSpec(fullname, None, is_package=True)
            spec.submodule_search_locations = [*NAMESPACES[fullname], PATH_PLACEHOLDER]
            return spec
sys.path_hooks.append(NamespaceFinder.path_hook)
sys.path.append(PATH_PLACEHOLDER)
""")
    _write(dependencies / "namespace.pth", "import namespace_hook\nimport startup_sdk\n")
    _write(source / "hermes_cli" / "main.py", """
import json
import sys
from hermes_cli.kanban_runtime import RuntimeIdentityError
print('ready', flush=True)
sys.stdin.readline()
import fixture_namespace.branch.selected as selected
import fixture_namespace
import fixture_namespace.branch
import editable_plugin
try:
    import fixture_namespace.uncaptured.selected
except RuntimeIdentityError:
    refused = True
else:
    refused = False
print(json.dumps({
    'namespace': selected.VALUE,
    'editable': editable_plugin.VALUE,
    'files': [selected.__file__, editable_plugin.__file__],
    'parent_paths': list(fixture_namespace.__path__),
    'child_paths': list(fixture_namespace.branch.__path__),
    'uncaptured_refused': refused,
}), flush=True)
""")
    prepared = prepare()
    child = _launch(prepared)
    try:
        assert child.stdout.readline().strip() == "ready"
        shutil.rmtree(namespace_source)
        (external / "implementation.py").unlink()
        old = _finish(child)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
    assert old["namespace"] == old["editable"] == "old"
    assert all(Path(path).is_relative_to(prepared.root) for path in old["files"])
    assert old["parent_paths"] == [sentinel]
    assert sentinel in old["child_paths"]
    assert all(Path(path).is_relative_to(prepared.root) for path in old["child_paths"] if path != sentinel)
    assert old["uncaptured_refused"] is True


def test_fresh_home_freezes_absent_plugins_until_next_generation(installation):
    source, _, plugins, _, _, prepare = installation
    pending = plugins.with_name("pending-plugins")
    plugins.rename(pending)
    _write(source / "hermes_cli" / "main.py", """
import json
import os
from pathlib import Path
import sys
print('ready', flush=True)
sys.stdin.readline()
from hermes_constants import get_hermes_home
from hermes_cli.kanban_runtime_generation import generation_runtime_path
from providers import get_provider_profile
directory = generation_runtime_path(get_hermes_home() / 'plugins')
assert directory.is_relative_to(Path(os.environ['HERMES_KANBAN_RUNTIME_GENERATION']))
profile = get_provider_profile('generation-fixture')
print(json.dumps({
    'directory': directory.exists(),
    'provider': profile.resolve_aux_model() if profile else None,
}), flush=True)
""")
    prepared = prepare()
    child = _launch(prepared)
    try:
        assert child.stdout.readline().strip() == "ready"
        pending.rename(plugins)
        assert _finish(child) == {"directory": False, "provider": None}
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
    following = prepare()
    newer = _launch(following)
    try:
        assert newer.stdout.readline().strip() == "ready"
        assert _finish(newer) == {"directory": True, "provider": "old"}
    finally:
        if newer.poll() is None:
            newer.kill()
            newer.wait()
    assert prepared.identity.dependency_fingerprint != following.identity.dependency_fingerprint


def test_preloaded_external_plugin_is_captured_without_repointing_parent(installation, monkeypatch):
    source, dependencies, _, external, _, prepare = installation
    _write(external / "external_plugin" / "__init__.py", """
from pathlib import Path
import unseeded_sdk
def value():
    return Path(__file__).with_name('resource.txt').read_text()
def lazy_value():
    from unseeded_sdk.lazy import selected
    return selected.VALUE
""")
    _write(dependencies / "unseeded_sdk" / "lazy" / "__init__.py", "")
    _write(dependencies / "unseeded_sdk" / "lazy" / "selected.py", "VALUE = 'old'\n")
    modules = {}
    for name, directory in (("unseeded_sdk", dependencies), ("external_plugin", external)):
        spec = importlib.util.spec_from_file_location(name, directory / name / "__init__.py")
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        modules[name] = module
    plugin = modules["external_plugin"]
    parent_paths = {name: (module.__file__, list(module.__path__)) for name, module in modules.items()}
    assert plugin.value() == "old"
    _write(source / "hermes_cli" / "main.py", """
import json
import external_plugin
print('ready', flush=True)
input()
value = external_plugin.lazy_value()
from unseeded_sdk.lazy import selected
try:
    import unseeded_sdk.new_only
except ModuleNotFoundError:
    refused = True
else:
    refused = False
print(json.dumps({
    'external': external_plugin.value(),
    'lazy': value,
    'files': [external_plugin.__file__, external_plugin.unseeded_sdk.__file__, selected.__file__],
    'uncaptured_refused': refused,
}), flush=True)
""")
    prepared = prepare()
    assert {name: (module.__file__, list(module.__path__)) for name, module in modules.items()} == parent_paths
    (external / "external_plugin" / "resource.txt").write_text("new", encoding="utf-8")
    assert plugin.value() == "new"
    child = _launch(prepared)
    try:
        assert child.stdout.readline().strip() == "ready"
        shutil.rmtree(external)
        shutil.rmtree(dependencies)
        _write(dependencies / "unseeded_sdk" / "lazy" / "selected.py", "VALUE = 'new'\n")
        _write(dependencies / "unseeded_sdk" / "new_only.py", "VALUE = 'mutable'\n")
        old = _finish(child)
        assert old["external"] == old["lazy"] == "old"
        assert all(Path(path).is_relative_to(prepared.root) for path in old["files"])
        assert old["uncaptured_refused"] is True
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()


def test_partial_or_rewritten_manifest_never_reaches_worker_code(installation):
    _, _, _, _, _, prepare = installation
    prepared = prepare()
    (prepared.root / "imports" / "0" / "unseeded_sdk" / "selected.py").unlink()
    child = _launch(prepared)
    output, error = child.communicate("unseeded_sdk.selected\n", timeout=60)
    assert child.returncode != 0, error
    assert output == ""
    generation.cleanup_runtime_generation(prepared.root)
    assert not prepared.root.exists()
    second = prepare()
    manifest_path = second.root / "generation.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["imports"].append("/tmp/mutable-import-root")
    # Replace the lease manifest, never mutate a shared cache inode.
    manifest_path.unlink()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    child = _launch(second)
    output, error = child.communicate("unseeded_sdk.selected\n", timeout=60)
    assert child.returncode != 0, error
    assert output == ""
    generation.cleanup_runtime_generation(second.root)
    assert not second.root.exists()


def test_reused_generation_cannot_be_deleted_while_another_worker_is_live(installation, runtime_storage):
    source, _, _, _, _, prepare = installation
    first, second = prepare(), prepare()
    assert first.identity.generation == second.identity.generation
    published, = (runtime_storage / "generations").iterdir()
    assert first.root.parent == second.root.parent == runtime_storage / "workers"
    if os.name != "nt":
        assert (first.root / "source" / "late.py").samefile(second.root / "source" / "late.py")
        assert (second.root / "source" / "late.py").samefile(published / "source" / "late.py")
    assert not (second.root / "source" / "late.py").samefile(source / "late.py")
    child = _launch(second)
    try:
        child.stdout.readline()
        generation.cleanup_runtime_generation(first.root, force=True)
        assert published.is_dir()
        (source / "late.py").write_text("VALUE = 'new'\n", encoding="utf-8")
        following = prepare()
        assert following.identity.generation != second.identity.generation
        assert not published.exists()
        retained, = (runtime_storage / "generations").iterdir()
        assert generation.generation_manifest(retained)["identity"]["generation"] == following.identity.generation
        generation.sweep_runtime_generations()
        assert second.root.exists()
        observed = _finish(child)
        assert observed["dynamic"] == observed["late"] == "old"
        generation.cleanup_runtime_generation(second.root)
        assert not second.root.exists()
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()


@pytest.mark.linux_only
def test_runtime_storage_uses_persistent_user_cache(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    assert generation._runtime_storage_root().is_relative_to(home / ".cache")
    override = tmp_path / "xdg-cache"
    monkeypatch.setenv("XDG_CACHE_HOME", str(override))
    assert generation._runtime_storage_root().is_relative_to(override)


def test_generation_cleanup_is_confined_to_owned_worker_roots(runtime_storage, tmp_path, monkeypatch):
    live = runtime_storage / "workers" / "hermes-kanban-runtime-live"
    stale = runtime_storage / "workers" / "hermes-kanban-runtime-stale"
    outside = tmp_path / "hermes-kanban-runtime-outside"
    publication = runtime_storage / "generations" / "hermes-kanban-runtime-content"
    unrecognized = runtime_storage / "workers" / "unrelated"
    for root in (live, stale, outside, publication, unrecognized):
        root.mkdir(parents=True)
        generation.write_runtime_generation_owner(root)
    (live / ".hermes-kanban-runtime-owner.json").write_text(json.dumps({"pid": os.getpid()}), encoding="utf-8")
    generation.write_runtime_generation_owner(stale, start_time=runtime.process_start_time() + 1)
    for root in (outside, publication, unrecognized):
        generation.cleanup_runtime_generation(root, force=True)
    if os.name != "nt":
        alias = live.parent / "hermes-kanban-runtime-alias"
        alias.symlink_to(outside, target_is_directory=True)
        generation.cleanup_runtime_generation(alias, force=True)
        assert alias.is_symlink()
    generation.sweep_runtime_generations()
    assert not stale.exists()
    assert all(root.is_dir() for root in (live, outside, publication, unrecognized))
    generation.write_runtime_generation_owner(live, start_time=runtime.process_start_time() + 1)
    monkeypatch.setenv("HERMES_KANBAN_RUNTIME_GENERATION", str(live))
    generation.sweep_runtime_generations()
    assert live.is_dir()


def test_generation_waits_for_source_and_dependency_transaction(installation):
    source, _, _, _, _, _ = installation
    expected = runtime.runtime_identity(source)
    writer = subprocess.Popen(
        [sys.executable, "-c", """
import sys
from pathlib import Path
from hermes_cli.kanban_runtime_generation import installation_mutation_lock
root = Path(sys.argv[1])
with installation_mutation_lock(root):
    (root / 'early.py').write_text("VALUE = 'new'\\n")
    print('half-written', flush=True)
    sys.stdin.readline()
    (root / 'late.py').write_text("VALUE = 'new'\\n")
""", str(source)],
        env={**os.environ, "PYTHONPATH": str(source), "HERMES_HOME": str(source / "other-profile")},
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    started = threading.Event()
    def prepare_while_locked():
        started.set()
        return generation.prepare_runtime_generation(expected)
    try:
        assert writer.stdout.readline().strip() == "half-written"
        probe = subprocess.run(
            [sys.executable, "-c", """
import sys
from hermes_cli.kanban_runtime_generation import installation_mutation_lock
try:
    with installation_mutation_lock(sys.argv[1], blocking=False):
        print('unexpected acquisition')
except BlockingIOError:
    print('busy')
""", str(source)],
            env={**os.environ, "PYTHONPATH": str(source), "HERMES_HOME": str(source / "third-profile")},
            stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=10,
        )
        assert probe.returncode == 0, probe.stderr
        assert probe.stdout.strip() == "busy"
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(prepare_while_locked)
            assert started.wait(5)
            try:
                with pytest.raises(TimeoutError):
                    future.result(timeout=0.2)
            finally:
                _, error = writer.communicate("commit\n", timeout=20)
                assert writer.returncode == 0, error
            with pytest.raises(runtime.RuntimeIdentityError):
                future.result(timeout=20)
    finally:
        if writer.poll() is None:
            writer.kill()
            writer.wait()


def test_generation_refuses_storage_inside_a_captured_import_root(installation, monkeypatch):
    _, dependencies, _, _, _, prepare = installation
    storage = dependencies / "runtime-cache"
    monkeypatch.setattr(generation, "_runtime_storage_root", lambda: storage)
    with pytest.raises(runtime.RuntimeIdentityError) as error:
        prepare()
    assert str(storage) in str(error.value)
    assert str(dependencies) in str(error.value)
    assert not storage.exists()


def test_incomplete_installation_is_not_publishable(installation):
    source, _, _, _, _, prepare = installation
    (source / ".update-incomplete").write_text("interrupted", encoding="utf-8")
    with pytest.raises(runtime.RuntimeIdentityError):
        prepare()


def test_installed_metadata_can_locate_data_outside_site_packages(installation, monkeypatch):
    source, dependencies, _, _, _, prepare = installation
    monkeypatch.setattr(sys, "prefix", str(dependencies.parent))
    data = dependencies.parent / "share" / "fixture.txt"
    _write(data, "old installed resource")
    _write(dependencies / "unseeded_sdk-1.0.dist-info" / "RECORD", "../share/fixture.txt,,\n")
    _write(source / "hermes_cli" / "main.py", """
from importlib.metadata import distribution
print(distribution('unseeded-sdk').locate_file('../share/fixture.txt').read_text())
""")
    prepared = prepare()
    data.write_text("new installed resource", encoding="utf-8")
    child = _launch(prepared)
    output, error = child.communicate(timeout=60)
    assert child.returncode == 0, error
    assert output.strip() == "old installed resource"

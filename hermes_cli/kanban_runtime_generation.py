"""Materialized worker runtimes, published before the worker imports Hermes.

Only stdlib imports belong here: this file also runs as the isolated bootstrap.
"""
from __future__ import annotations

from contextlib import contextmanager, ExitStack
from dataclasses import dataclass, replace
from functools import cache
import errno
import importlib.machinery
import importlib.metadata
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import sysconfig
import tempfile
import threading
import time
import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hermes_cli.kanban_runtime import RuntimeIdentity

_GENERATION_ENV = "HERMES_KANBAN_RUNTIME_GENERATION"
_MANIFEST = "generation.json"
_OWNER = ".hermes-kanban-runtime-owner.json"
_TREE_EXCLUDES = {".git", ".hg", ".svn", "__pycache__"}
_SOURCE_EXCLUDES = {"venv", ".venv", ".pytest_cache", ".ruff_cache", ".mypy_cache"}


def _runtime_storage_root() -> Path:
    """Keep copied runtimes and their leases together, outside RAM-backed tempdirs."""
    home = Path.home()
    if sys.platform == "darwin":
        cache_root = home / "Library" / "Caches"
    elif sys.platform == "win32":
        cache_root = Path(os.environ.get("LOCALAPPDATA") or home / "AppData" / "Local")
    else:
        cache_root = Path(os.environ.get("XDG_CACHE_HOME") or home / ".cache")
    return (cache_root.expanduser() / "hermes" / "kanban-runtime").resolve()


def _error(message: str):
    from hermes_cli.kanban_runtime import RuntimeIdentityError
    return RuntimeIdentityError(message)


@contextmanager
def installation_mutation_lock(module_root=None, *, blocking=True):
    """Serialize managed source/dependency writes across profiles and worktrees.

    The UI update marker is deliberately not reused: it is profile-local,
    read-then-write, and fails open. OS locks are released on process death.
    """
    if os.environ.get(_GENERATION_ENV):
        raise _error("immutable workers cannot mutate their installed runtime")
    root = Path(module_root or Path(__file__).resolve().parents[1]).resolve()
    keys = {root, Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve()}
    target = os.environ.get("HERMES_LAZY_INSTALL_TARGET", "").strip()
    if target:
        keys.add(Path(target).resolve())
    # Survives the updater evicting and reimporting first-party modules.
    state = getattr(sys, "_hermes_installation_locks", None)
    if state is None:
        state = (threading.RLock(), {})
        sys._hermes_installation_locks = state
    mutex, held = state
    if not mutex.acquire(blocking=blocking):
        raise BlockingIOError(errno.EWOULDBLOCK, "runtime installation is being mutated")
    with ExitStack() as stack:
        stack.callback(mutex.release)
        for key in sorted(map(str, keys)):
            if key in held:
                continue
            digest = hashlib.sha256(key.encode()).hexdigest()
            lock_dir = Path(tempfile.gettempdir()) / "hermes-installation-locks"
            lock_dir.mkdir(mode=0o700, exist_ok=True)
            handle = stack.enter_context((lock_dir / digest).open("a+b"))
            if os.name == "nt":
                import msvcrt
                handle.seek(0)
                if not handle.read(1):
                    handle.write(b"\0")
                    handle.flush()
                while True:
                    try:
                        handle.seek(0)
                        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                        break
                    except OSError as exc:
                        if exc.errno not in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                            raise
                        if not blocking:
                            raise BlockingIOError(exc.errno, "runtime installation is being mutated") from exc
                        time.sleep(0.05)
                def unlock(handle=handle):
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
                def unlock(handle=handle):
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            held[key] = handle
            stack.callback(held.pop, key, None)
            stack.callback(unlock)
        yield stack.close


def _members(root: Path, *, source=False, exclude=()):
    """Walk materialized files, preserving package data and extension libraries."""
    if root.is_file():
        yield "", root
        return
    def unreadable(error):
        raise error
    for base, directories, files in os.walk(root, followlinks=True, onerror=unreadable):
        base = Path(base)
        resolved = base.resolve()
        ancestors = {parent.resolve() for parent in base.parents if parent.is_relative_to(root)}
        if resolved in ancestors:
            raise _error(f"cyclic runtime directory: {base}")
        directories[:] = sorted(
            name for name in directories
            if name not in _TREE_EXCLUDES and name not in exclude
            and not (source and (name in _SOURCE_EXCLUDES or (base / name / "pyvenv.cfg").is_file()))
        )
        relative = base.relative_to(root).as_posix()
        yield relative + "/", base
        for name in sorted(files):
            if (source and name.endswith((".pyc", ".pyo"))) or name == ".env" or name.startswith(".env."):
                continue
            path = base / name
            if path.is_file():
                yield path.relative_to(root).as_posix(), path


def source_members(root: Path):
    yield from _members(root, source=True)


def _digest_members(members):
    digest = hashlib.sha256()
    for name, path in members:
        digest.update(name.encode())
        digest.update(b"\0")
        if path.is_file():
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()


@cache
def _stdlib_roots():
    roots = [Path(sysconfig.get_path("stdlib")).resolve()]
    dynamic = sysconfig.get_config_var("DESTSHARED")
    if dynamic:
        path = Path(dynamic).resolve()
        if path.is_relative_to(Path(sys.base_prefix).resolve()) and path.is_dir() and not path.is_relative_to(roots[0]):
            roots.append(path)
    dlls = Path(sys.base_prefix).resolve() / "DLLs"
    if dlls.is_dir():
        roots.append(dlls)
    return tuple(roots)


def _is_stdlib(path: Path):
    return any(path.is_relative_to(root) for root in _stdlib_roots()) and not any(
        part in {"site-packages", "dist-packages"} for part in path.parts
    )




def _python_runtime_inputs():
    """Preserve CPython's install layout and native loader companions."""
    base = Path(sys.base_prefix).resolve()
    executable = Path(getattr(sys, "_base_executable", sys.executable)).resolve()
    executable_relative = executable.relative_to(base) if executable.is_relative_to(base) else (Path("python.exe") if os.name == "nt" else Path("bin") / executable.name)
    # Managed standalone/framework distributions carry their own native libraries
    # and resources. Preserve that layout rather than guessing library dependencies.
    if base not in {Path("/"), Path("/usr"), Path("/usr/local")}:
        return {str(base): "python"}, {str(base): {"site-packages", "dist-packages"}}, f"python/{executable_relative.as_posix()}"
    mappings = {str(executable): f"python/{executable_relative.as_posix()}"}
    excluded = {}
    for stdlib in _stdlib_roots():
        relative = stdlib.relative_to(base)
        mappings[str(stdlib)] = f"python/{relative.as_posix()}"
        excluded[str(stdlib)] = {"site-packages", "dist-packages"}
    for directory in (base, base / "DLLs", Path(sysconfig.get_config_var("LIBDIR") or base / "lib")):
        if not directory.is_dir():
            continue
        for pattern in ("*.dll", "libpython*.so*", "libpython*.dylib", "python*.zip"):
            for library in directory.glob(pattern):
                relative = library.relative_to(base) if library.is_relative_to(base) else Path("lib") / library.name
                mappings[str(library)] = f"python/{relative.as_posix()}"
    library = str(sysconfig.get_config_var("LDLIBRARY") or "")
    for candidate in (base / library, Path(sysconfig.get_config_var("LIBDIR") or base / "lib") / library):
        if library and candidate.is_file():
            relative = candidate.relative_to(base) if candidate.is_relative_to(base) else Path("lib") / candidate.name
            mappings[str(candidate)] = f"python/{relative.as_posix()}"
    if (base / "DLLs").is_dir():
        mappings[str(base / "DLLs")] = "python/DLLs"
    return mappings, excluded, f"python/{executable_relative.as_posix()}"


def _optional_plugin_roots(workspace=None) -> set[Path]:
    from hermes_constants import get_hermes_home
    roots = {(get_hermes_home() / "plugins").resolve()}
    if os.environ.get("HERMES_ENABLE_PROJECT_PLUGINS", "").lower() in {"1", "true", "yes", "on"}:
        roots.update(
            (Path(directory) / ".hermes" / "plugins").resolve()
            for directory in (Path.cwd(), workspace) if directory is not None
        )
    return roots


def _runtime_import_roots(source: Path) -> list[Path]:
    """All active import roots, including editable and file-loaded plugins."""
    candidates = [Path(entry or os.getcwd()).resolve() for entry in sys.path]
    for key in ("purelib", "platlib"):
        candidates.append(Path(sysconfig.get_path(key)).resolve())
    target = os.environ.get("HERMES_LAZY_INSTALL_TARGET", "").strip()
    if target:
        candidates.append(Path(target).resolve())
    plugin_roots = _optional_plugin_roots()
    candidates.extend(plugin_roots)
    for module in tuple(sys.modules.values()):
        # Editable installers may expose packages only through a meta finder.
        for location in (getattr(module, "MAPPING", {}) or {}).values() if isinstance(getattr(module, "MAPPING", None), dict) else ():
            candidates.append(Path(location).resolve().parent)
        namespaces = getattr(module, "NAMESPACES", None)
        if isinstance(namespaces, dict):
            for locations in namespaces.values():
                candidates.extend(Path(location).resolve() for location in locations)
        for location in getattr(module, "__path__", ()) or ():
            path = Path(location).resolve()
            if path.exists():
                candidates.append(path.parent)
        location = getattr(module, "__file__", None)
        if location:
            path = Path(location).resolve()
            if path.is_file():
                candidates.append(path.parent)
    roots = []
    for path in candidates:
        in_source = path.is_relative_to(source) and not any(part in _SOURCE_EXCLUDES for part in path.relative_to(source).parts)
        if in_source or _is_stdlib(path) or (not path.exists() and path not in plugin_roots):
            continue
        if not any(path == root or path.is_relative_to(root) for root in roots):
            roots.append(path)
    return roots


def _copy_members(source: Path, destination: Path, *, first_party=False, exclude=()):
    if source.is_dir():
        destination.mkdir(parents=True, exist_ok=True)
    for name, path in _members(source, source=first_party, exclude=exclude):
        target = destination / name if name else destination
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)  # never link files from the mutable installation


def _payload_digest(root: Path):
    return _digest_members((name, path) for name, path in _members(root) if name not in {_MANIFEST, _OWNER} and not name.startswith(f".{_OWNER}."))


def _generation_digest(root: Path, manifest):
    identity = {key: value for key, value in manifest["identity"].items() if key not in {"generation", "pid", "start_time"}}
    attestation = {**manifest, "identity": identity, "payload": _payload_digest(root)}
    return hashlib.sha256(json.dumps(attestation, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _json_write(path: Path, value):
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    os.replace(temporary, path)


def write_runtime_generation_owner(root: Path, *, pid=None, start_time=None):
    from hermes_cli.kanban_runtime import process_start_time
    pid = int(pid or os.getpid())
    _json_write(Path(root) / _OWNER, {"pid": pid, "start_time": int(start_time or process_start_time(pid))})


def cleanup_runtime_generation(root, *, force=False):
    if not root:
        return
    root = Path(root)
    if root.is_symlink():
        return
    root = root.resolve()
    if root.parent != _runtime_storage_root().resolve() / "workers" or not root.name.startswith("hermes-kanban-runtime-"):
        return
    if not root.is_dir():
        return
    if not force:
        from hermes_cli.kanban_runtime import process_start_time
        try:
            owner = json.loads((root / _OWNER).read_text(encoding="utf-8"))
            pid = int(owner["pid"])
            started = process_start_time(pid)
            # Bootstrap initially has only stdlib available. Until it verifies
            # the payload and records a start time, any live PID retains its lease.
            if "start_time" not in owner or started == int(owner["start_time"]):
                return
        except (OSError, ValueError, KeyError, TypeError):
            return
        except Exception:
            # Only positive OS evidence of exit permits cleanup when probing fails.
            if sys.platform.startswith("linux"):
                if Path(f"/proc/{pid}").exists():
                    return
            else:
                try:
                    import psutil
                    if psutil.pid_exists(pid):
                        return
                except ImportError:
                    return
    _remove_runtime_tree(root)


def _remove_runtime_tree(root: Path):
    def remove_readonly(function, path, exc):
        if os.name != "nt":
            raise exc[1]
        os.chmod(path, 0o700)
        function(path)
    try:
        shutil.rmtree(root, onerror=remove_readonly)
    except OSError:
        return  # Cleanup is best effort; a retained generation remains safe.


def sweep_runtime_generations():
    # Immutable children receive a concrete lease, not their launcher's cache settings.
    if os.environ.get(_GENERATION_ENV):
        return
    for root in (_runtime_storage_root() / "workers").glob("hermes-kanban-runtime-*"):
        cleanup_runtime_generation(root)


def _prune_runtime_content(current: Path, source: Path):
    # The caller holds the source installation lock; other sources are independent.
    # Worker leases own sealed inodes, so obsolete publications need no live-owner scan.
    for candidate in current.parent.iterdir():
        if candidate == current or candidate.is_symlink() or not candidate.is_dir():
            continue
        if len(candidate.name) != 64 or any(char not in "0123456789abcdef" for char in candidate.name):
            continue
        try:
            manifest = json.loads((candidate / _MANIFEST).read_text(encoding="utf-8"))
            if manifest["identity"]["module_root"] != str(source):
                continue
        except (OSError, ValueError, KeyError, TypeError):
            continue
        _remove_runtime_tree(candidate)


@dataclass(frozen=True)
class RuntimeGeneration:
    root: Path
    command_prefix: list[str]
    env: dict[str, str]
    identity: RuntimeIdentity


def _installed_root_location(root: Path, index: int):
    for prefix, destination in ((Path(sys.prefix).resolve(), "python"), (Path(sys.base_prefix).resolve(), "base-python")):
        if root.is_relative_to(prefix):
            return prefix, destination, str(Path(destination) / root.relative_to(prefix))
    return None, None, f"imports/{index}"


def _installed_resources(roots, mappings):
    """Include installed data/scripts outside site-packages, preserving RECORD paths."""
    for index, root in enumerate(roots):
        prefix, destination, _ = _installed_root_location(root, index)
        if prefix is None or not root.is_dir():
            continue
        for distribution in importlib.metadata.distributions(path=[str(root)]):
            for entry in distribution.files or ():
                path = Path(os.path.abspath(distribution.locate_file(entry)))
                if not path.is_file() or not path.is_relative_to(prefix) or path.is_relative_to(root):
                    continue
                if path.name == ".env" or path.name.startswith(".env."):
                    continue
                mappings.setdefault(str(path), str(Path(destination) / path.relative_to(prefix)))


def prepare_runtime_generation(expected_identity, *, workspace=None):
    from hermes_cli.kanban_runtime import RuntimeIdentity, runtime_identity, same_code_identity, _RUNTIME_RESOURCE_ROOTS
    expected = expected_identity if isinstance(expected_identity, RuntimeIdentity) else RuntimeIdentity.from_value(expected_identity)
    source = Path(expected.module_root).resolve()
    with installation_mutation_lock(source):
        for marker in (".update-incomplete", ".lazy-refresh-incomplete"):
            if (source / marker).exists():
                raise _error(f"runtime installation needs recovery: {marker}")
        current = runtime_identity(source)
        if not same_code_identity(expected, current):
            raise _error("source runtime changed before generation preparation")
        roots = _runtime_import_roots(source)
        optional_roots = _optional_plugin_roots(workspace)
        for plugins in sorted(optional_roots):
            if not any(plugins.is_relative_to(root) for root in (source, *roots)):
                roots.append(plugins)
        native_paths, exclusions, executable = _python_runtime_inputs()
        mappings = {**native_paths, str(source): "source"}
        mappings.update((str(root), _installed_root_location(root, index)[2]) for index, root in enumerate(roots))
        _installed_resources(roots, mappings)
        resources = {}
        for directory, variable in _RUNTIME_RESOURCE_ROOTS:
            origin = Path(os.environ.get(variable) or source / directory).resolve()
            if origin.is_dir():
                if not origin.is_relative_to(source):
                    mappings.setdefault(str(origin), f"resources/{directory}")
                resources[variable] = str(origin)
        storage = _runtime_storage_root().resolve()
        for origin in mappings:
            if storage.is_relative_to(Path(origin).resolve()):
                raise _error(
                    f"runtime generation storage {storage} is inside captured root {origin}; "
                    "move the user cache outside runtime source, import, and resource roots"
                )
        optional_origins = {str(root) for root in optional_roots} - native_paths.keys() - {str(source), *resources.values()}
        def fingerprint(origin):
            path = Path(origin)
            if origin in optional_origins:
                try:
                    path.lstat()
                except FileNotFoundError:
                    return None  # Freeze discovery absence, not an empty or mutable directory.
            return _digest_members(_members(path, source=origin == str(source), exclude=exclusions.get(origin, ())))
        fingerprints = {origin: fingerprint(origin) for origin in mappings}
        dependency_fingerprint = hashlib.sha256(json.dumps(
            {origin: digest for origin, digest in fingerprints.items() if origin != str(source)}, sort_keys=True
        ).encode()).hexdigest()
        cache_key = hashlib.sha256(json.dumps({"files": fingerprints, "paths": mappings, "resources": resources}, sort_keys=True).encode()).hexdigest()
        cache_base = storage / "generations"
        worker_base = storage / "workers"
        storage.mkdir(mode=0o700, parents=True, exist_ok=True)
        for directory in (cache_base, worker_base):
            if directory.is_symlink():
                raise _error("runtime storage directories must not be symlinks")
            directory.mkdir(mode=0o700, exist_ok=True)
        cache = cache_base / cache_key
        if not cache.exists():
            staging = Path(tempfile.mkdtemp(prefix=".preparing-", dir=cache_base))
            try:
                for origin, destination in mappings.items():
                    if fingerprints[origin] is None:
                        continue
                    _copy_members(Path(origin), staging / destination, first_party=origin == str(source), exclude=exclusions.get(origin, ()))
                    copied = ((name, staging / destination / name if name else staging / destination) for name, _ in _members(Path(origin), source=origin == str(source), exclude=exclusions.get(origin, ())))
                    if _digest_members(copied) != fingerprints[origin]:
                        raise _error("runtime changed while copying a generation")
                shutil.copy2(__file__, staging / "bootstrap.py")
                identity = replace(current, dependency_fingerprint=dependency_fingerprint)
                manifest = {"identity": identity.as_dict(), "paths": mappings, "imports": ["source", *(mappings[str(root)] for root in roots)], "resources": resources, "executable": executable}
                identity = replace(identity, generation=_generation_digest(staging, manifest))
                manifest["identity"] = identity.as_dict()
                _json_write(staging / _MANIFEST, manifest)
                for _, path in _members(staging):
                    if path.is_file():
                        path.chmod(path.stat().st_mode & ~0o222)
                # Publish only a complete copy; concurrent manual changes fail closed.
                if any(fingerprint(origin) != digest for origin, digest in fingerprints.items()):
                    raise _error("runtime changed before generation publication")
                os.replace(staging, cache)
            except BaseException:
                _remove_runtime_tree(staging)
                raise
        manifest = json.loads((cache / _MANIFEST).read_text(encoding="utf-8"))
        identity = RuntimeIdentity.from_value(manifest["identity"])
        if _generation_digest(cache, manifest) != identity.generation:
            raise _error("published runtime generation is corrupt")
        root = Path(tempfile.mkdtemp(prefix="hermes-kanban-runtime-", dir=worker_base)).resolve()
        try:
            write_runtime_generation_owner(root)
            # Cache files are sealed copies, so sharing their inodes cannot observe an installer write.
            def link_copy(origin, destination):
                try:
                    if os.name == "nt":
                        raise OSError("Windows generation files need independently deletable copies")
                    os.link(origin, destination)
                except OSError:
                    shutil.copy2(origin, destination)
                return destination
            shutil.copytree(cache, root, dirs_exist_ok=True, copy_function=link_copy)
            if any(fingerprint(origin) is not None for origin, digest in fingerprints.items() if digest is None):
                raise _error("optional plugin root appeared before generation publication")
        except BaseException:
            _remove_runtime_tree(root)
            raise
        _prune_runtime_content(cache, source)
        env = {_GENERATION_ENV: str(root), "HERMES_DISABLE_LAZY_INSTALLS": "1", "HERMES_LAZY_INSTALL_TARGET": ""}
        library_variable = "PATH" if os.name == "nt" else ("DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH")
        library_roots = [root / "python"] if os.name == "nt" else [root / "python" / "lib", root / "python" / "lib64"]
        env[library_variable] = os.pathsep.join([*map(str, library_roots), os.environ.get(library_variable, "")])
        for variable, origin in resources.items():
            env[variable] = str(_mapped_path(Path(origin), root, mappings))
        return RuntimeGeneration(root, [str(root / executable), "-I", "-S", str(root / "bootstrap.py")], env, identity)


def _mapped_path(path: Path, root: Path, mappings):
    path = Path(os.path.abspath(path))
    # Match captured names before following links added to the mutable installation.
    for resolve in (False, True):
        if resolve:
            path = path.resolve()
        if path.is_relative_to(root):
            return path
        for origin in sorted(mappings, key=len, reverse=True):
            if path.is_relative_to(origin):
                return root / mappings[origin] / path.relative_to(origin)
    if _is_stdlib(path):
        return path
    raise _error(f"import path was not captured in the runtime generation: {path}")


def generation_runtime_path(path):
    raw = os.environ.get(_GENERATION_ENV)
    if not raw:
        return Path(path)
    root = Path(raw).resolve()
    manifest = json.loads((root / _MANIFEST).read_text(encoding="utf-8"))
    return _mapped_path(Path(path), root, manifest["paths"])


def generation_manifest(root=None, *, verify=False):
    root = Path(root or os.environ[_GENERATION_ENV]).resolve()
    manifest = json.loads((root / _MANIFEST).read_text(encoding="utf-8"))
    if verify and _generation_digest(root, manifest) != manifest["identity"]["generation"]:
        raise _error("runtime generation is incomplete or changed")
    return manifest


def _mapped_search_path(entry, root, mappings):
    if not isinstance(entry, str):
        return entry  # PathFinder ignores non-string search entries.
    # Path hooks also own opaque tokens, not just filesystem paths. Keep their
    # exact spelling; any origins/locations they return still cross the fence.
    # Probe current hooks: deleted filesystem roots can retain cached FileFinders.
    if entry and not os.path.exists(entry) and importlib.machinery.PathFinder._path_hooks(entry) is not None:
        return entry
    return str(_mapped_path(Path(entry or os.getcwd()), root, mappings))


class _GenerationFinder:
    """Relocate installed editable/.pth finders without importing mutable files."""
    def __init__(self, root, mappings):
        self.root, self.mappings = root, mappings

    def find_spec(self, fullname, path=None, target=None):
        search = path if path is not None else sys.path
        mapped = [_mapped_search_path(entry, self.root, self.mappings) for entry in search]
        if path is None:
            sys.path[:] = mapped
        for finder in tuple(sys.meta_path):
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            # Editable finders probe candidate existence before returning a spec.
            # Relocate their captured install mapping before that mutable lookup.
            module = sys.modules.get(getattr(finder, "__module__", ""))
            editable = getattr(module, "MAPPING", None)
            if isinstance(editable, dict):
                for name in (fullname, fullname.rpartition(".")[0]):
                    if name in editable:
                        editable[name] = str(_mapped_path(Path(editable[name]), self.root, self.mappings))
            spec = finder.find_spec(fullname, mapped if path is not None else None, target)
            if spec is None:
                continue
            if spec.origin in {"built-in", "frozen"}:
                return spec
            locations = spec.submodule_search_locations
            if locations is not None:
                locations = [_mapped_search_path(entry, self.root, self.mappings) for entry in locations]
            if spec.origin:
                origin = _mapped_path(Path(spec.origin), self.root, self.mappings)
                if str(origin) != spec.origin:
                    return importlib.util.spec_from_file_location(
                        fullname, origin, submodule_search_locations=locations,
                    )
            spec.submodule_search_locations = locations
            return spec
        return None


class _GenerationMetaPath(list):
    def insert(self, index, finder):
        # Installed .pth hooks may prepend their finder; preserve its priority
        # behind the relocation fence, not ahead of it.
        super().insert(max(1, index if index >= 0 else len(self) + index), finder)


def _bootstrap():
    root = Path(__file__).resolve().parent
    # Adopt only an allocated lease at our script root, not an env-supplied path.
    # A rejected bootstrap must not leave its lease owned by the live dispatcher.
    if root.parent.name == "workers" and root.name.startswith("hermes-kanban-runtime-") and (root / _OWNER).is_file():
        _json_write(root / _OWNER, {"pid": os.getpid()})
    if Path(os.environ.get(_GENERATION_ENV, "")).resolve() != root:
        raise RuntimeError("runtime generation bootstrap path mismatch")
    if not Path(sys.prefix).resolve().is_relative_to(root / "python"):
        raise RuntimeError("generation interpreter did not select its copied standard library")
    # No Hermes or third-party code has been imported by -I -S at this point.
    manifest = json.loads((root / _MANIFEST).read_text(encoding="utf-8"))
    if _generation_digest(root, manifest) != manifest["identity"]["generation"]:
        raise RuntimeError("runtime generation is incomplete or changed")
    sys.dont_write_bytecode = True
    stdlib = [entry for entry in sys.path if _is_stdlib(Path(entry)) or Path(entry).name == f"python{sys.version_info.major}{sys.version_info.minor}.zip"]
    sys.path[:] = [str(root / entry) for entry in manifest["imports"]] + stdlib
    # The absent stdlib zip is part of Python's standard search path, not an install root.
    mappings = dict(manifest["paths"])
    mappings.update((entry, entry) for entry in stdlib)
    sys.meta_path = _GenerationMetaPath([_GenerationFinder(root, mappings), *sys.meta_path])
    write_runtime_generation_owner(root)
    import site
    for entry in manifest["imports"][1:]:
        if (root / entry).is_dir():
            site.addsitedir(str(root / entry))
    sys.path[:] = [_mapped_search_path(entry, root, mappings) for entry in sys.path]
    import runpy
    runpy.run_module("hermes_cli.main", run_name="__main__")


if __name__ == "__main__":
    _bootstrap()

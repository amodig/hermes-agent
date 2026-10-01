# Architecture Decision Records

## 2026-07-13: Scope plugin manager state by Hermes home/profile (keyed cache)

Status: Accepted

Context:
Hermes supports multiple profiles via different Hermes home directories.
Homes are switched two ways in a running process: the `HERMES_HOME`
environment variable (single-profile CLI/gateway processes), and the
context-local `set_hermes_home_override()` (`hermes_constants.py`), which
the multiplexed gateway worker (`gateway/run.py`'s `_profile_scope`) and
subagent/embedded callers use to serve several profiles from one
long-lived process. The override is a `ContextVar` and deliberately does
**not** mutate `os.environ`, since that would leak one profile's home
into every other concurrent task in the same process.

The plugin manager was a process-global single-slot singleton
(`_plugin_manager`). User-installed plugins are discovered from
`get_hermes_home() / "plugins"`, and context-engine plugins (e.g.
`hermes-lcm`) capture profile-scoped state — such as the LCM database
path — at registration time. A single-slot cache meant:

1. Switching homes via `set_hermes_home_override()` was invisible to a
   naive "did `HERMES_HOME` change" check, so the singleton silently kept
   serving the first profile's manager to every other profile in the
   process.
2. Even when a fresh `PluginManager` *was* created for a new home, plugin
   modules are imported into `sys.modules` as `hermes_plugins.<slug>` by
   `_load_directory_module`, and only that top-level module was ever
   replaced. A same-slug plugin's *relative* imports
   (`from . import state`) are cached separately under
   `hermes_plugins.<slug>.<submodule>`, and Python's import machinery
   resolves those from `sys.modules` first — so a profile switch could
   silently keep serving a previous profile's already-imported submodule
   code/state instead of re-executing the new profile's plugin.

Decision:
- Replace the single-slot singleton with a cache keyed on the *resolved*
  Hermes home path (`_plugin_managers_by_home: Dict[Path, PluginManager]`).
  `get_plugin_manager()` resolves the current home via `get_hermes_home()`
  (which itself already consults `get_hermes_home_override()` before
  `os.environ`), so both the env-var and context-local override paths are
  covered uniformly.
- `_plugin_manager` (the old single-slot name) is kept as a thin "last
  manager returned" pointer purely for backward compatibility with
  existing test code that does
  `monkeypatch.setattr(plugins_mod, "_plugin_manager", some_manager)`.
  When that name is monkeypatched to a manager the keyed cache doesn't
  know about, `get_plugin_manager()` treats it as an explicit injection
  and adopts it into the cache under the *current* resolved home, rather
  than discarding it.
- Both `PluginManager._load_directory_module` (initial/`force=True`
  reload within the same home) and the shared `_clear_plugin_submodules`
  helper (profile switch / test teardown) evict `sys.modules[module_name]`
  **and every name prefixed with `module_name + "."`** before a plugin
  slug is (re-)imported, so relative-import submodules can never survive
  a reload or a home switch.
- Test isolation (`tests/conftest.py`'s `_hermetic_environment` fixture)
  calls a new `_reset_plugin_managers_for_tests()` helper that drops the
  entire keyed cache and purges every plugin submodule from `sys.modules`
  between tests, instead of only resetting the single-slot pointer.

Consequences:
- Per-profile LCM instances (and any other context-engine plugin) use
  their own `{home}/lcm.db` regardless of whether the profile switch went
  through `HERMES_HOME` or `set_hermes_home_override()`.
- Plugin discovery remains cached within a profile for normal
  performance, and re-entering a previously-seen profile reuses its
  cached manager instead of rebuilding from scratch.
- Sequential *and* interleaved profile switching — in tests, the gateway
  multiplexer worker, or embedded callers using the context-local
  override — no longer leaks context-engine state, plugin module state,
  or stale relative-import submodules across profiles.
- Regression coverage exercises the real production path
  (`set_hermes_home_override()`) rather than only the env-var path, and
  includes a dedicated relative-import leak test.


## 2026-10-01: Distinguish a replaced host from a dead worker, and grant nothing while draining

Status: Accepted

Context:
A `running` Kanban card whose worker is gone had no distinguishable causes. An
ordinary worker crash counted against the wrong failure limit, because the
dispatcher's configured `kanban.failure_limit` reached the reclaim phase but was
dropped before crash accounting, so every crash used the built-in default. After
a reboot or container restart, a recorded worker PID may be gone or reused by an
unrelated process, and nothing recorded which host instantiation had owned the
claim. Separately, a gateway that was already draining kept launching workers
through the shutdown window, and `systemd-run --user --scope` placed workers in
`app.slice`, outside the gateway slice whose budget the operator was reading.

Decision:
Record the host instantiation epoch (`<boot_id>:<pid1_start>`) on each new run as
additive metadata beside the sealed runtime identity, never backfilling history.
Before every other reclaim path, pause a locally owned `running` card only on a
positive epoch verdict (differing valid boot UUID, or differing valid PID 1
start ticks within the same valid boot); missing, malformed or incomparable
components are unknown, never "changed". The pause is one compare-and-swap that
closes the run as `interrupted` and emits the same sticky `blocked` event an
explicit operator block uses, with reason `host_restarted`, refusing NULL or
foreign run claims. It charges and resets nothing, never probes or signals the
stale PID, and creates no successor card; `unblock_task` stays the only resume.

Thread the configured `failure_limit` through to crash accounting (per-task
`max_retries` still wins; systemic, protocol-violation and quota-wall policies
keep their precedence). Set the drain flag synchronously at signal/stop entry and
poll a `should_stop` predicate at every long launch step — before preparation,
after preparation and before the claim, during bootstrap waits, and immediately
before the grant. A cancelled launch charges no failure; a cancellation after the
claim cancels the ungranted worker and records a sticky `gateway_stopping` pause
on that exact run; a cancellation after the grant cancels nothing, so a granted
worker survives a gateway restart. Accept the user-manager race, where scope
creation fails before the gateway's own signal, only on the positive evidence of
`systemctl --user is-system-running` reporting `stopping`, evaluated at the native
scope/bootstrap failure site so that only the failure it was sampled beside is
converted — dispatcher handlers exempt nothing, keeping runtime-identity,
workspace and database failures charged. Report a launch refusal with its real
phase instead of always `runtime_identity`, and isolate post-grant bookkeeping so
a failed hook or PID write cannot cancel a worker that already holds its grant.

Add `--slice-inherit` to the shared scope argv so a worker keeps its own cgroup
while staying inside the caller's slice, and expose interruptions separately from
crashes in dispatcher tick telemetry.

Consequences:
- A host interruption is an operator decision: the card waits in the blocked lane
  until `hermes kanban unblock`, with implementation/review phase, candidate,
  graph, goal revision and model pins preserved.
- Receipts written before host epochs existed keep their previous reclaim
  behavior and are never exempted from accounting on missing provenance.
- Ordinary crashes now respect the operator's configured limit; the incident's
  historical third failure would have tripped either the default or the
  configured limit, so this fix is not a claim about what caused that block.
- Workers no longer escape the gateway's ancestor budget. A host budget, slice,
  drop-in, routing or profile value does not change, and systemd without the flag
  keeps the restart-safe route unavailable rather than unmanaged.
- True reboot and user-manager-shutdown-contention acceptance still needs a
  disposable systemd host; the evidence gathered on `loota` covers a real
  user-service restart, slice membership, and injected-epoch behavior.

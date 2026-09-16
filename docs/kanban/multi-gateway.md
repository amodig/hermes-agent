# Multi-gateway deployment

Hermes supports multiple gateway processes running concurrently — one per profile
(default, writer, admin, coder, researcher). Each gateway opens its own connection
to platform APIs and delivers messages for its profile's subscribers.

Task subscriptions also cover review feedback. A `changes_requested` review
event is delivered as an actionable review-BLOCK notification. Subscriptions
using `notify+wake` additionally wake the exact originating chat/thread/session
so the controller inspects the existing card and current run; `notify` remains
passive-only and `wake` remains wake-only. Review feedback never creates,
unblocks, requeues, or otherwise mutates a task.

## Single-dispatcher posture

Only one gateway owns the kanban dispatcher. The owning gateway keeps
`kanban.dispatch_in_gateway: true` (the default); every other gateway sets it
to `false`.

**Why this matters:** dispatching is single-owner so multiple gateways do not
race to spawn the same work. Notification delivery is profile-owned instead:
each gateway polls only subscriptions for profiles whose platform adapters it
hosts. The atomic event claim prevents duplicate delivery across watcher
processes.

## Worker runtime generations

Before granting a worker claim, the dispatcher prepares an immutable generation
containing Hermes, installed dependencies and package resources, and the Python
interpreter and standard library. The worker starts inside that generation before
importing Hermes. Its identity records both source provenance and the captured
dependency content.

Managed updates, dependency installation, and recovery coordinate with generation
preparation. Existing workers retain their captured runtime across updates and
gateway restarts; new workers use the current installation. Install missing
features in the mutable installation, then start a new worker: sealed workers
cannot lazy-install dependencies.

Published generations and worker leases share the platform's persistent cache
(`$XDG_CACHE_HOME/hermes/kanban-runtime`, or `~/.cache/hermes/kanban-runtime`, on
Linux). Obsolete publications can be removed without removing surviving worker
leases. Lease cleanup checks the actual worker PID and process start time; do not
manually delete runtime storage while workers are alive.

An explicit `HERMES_BIN` must identify this installation's canonical entrypoint.
Arbitrary wrappers and other installations are rejected rather than executed
outside the generation. Runtime generations are not an OS sandbox: task
workspaces and operating-system libraries remain external.

## Configuration

On the dispatch-owning gateway (typically the `default` profile), no change is
needed. On every other profile gateway, add to `~/.hermes/config.yaml`:

```yaml
kanban:
  dispatch_in_gateway: false
```

Or set the env var: `HERMES_KANBAN_DISPATCH_IN_GATEWAY=false`

## What each gateway does

| Gateway role | dispatch_in_gateway | Opens subscribed board DBs? | Dispatcher | Notifier |
|---|---|---|---|---|
| default (confirmed dispatch-lock owner) | true (default) | yes | yes | owned profiles + legacy unstamped subscriptions |
| writer, admin, coder, etc. | false | yes, when the profile has subscriptions | no | that gateway's owned profiles |

Non-dispatch gateways still deliver messages for their own platform adapters
(Telegram, Discord, etc.). They do not dispatch tasks, and they skip boards
that have no subscriptions owned by their profiles.

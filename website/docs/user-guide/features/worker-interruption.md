# Worker interruption, restart survival, and failure accounting

A Kanban card that stops making progress has three very different causes, and
conflating them is what turns a stalled card into a silent one. This page is the
operator view of how the dispatcher tells them apart. The durable decision is
recorded in the [architecture decisions](../../developer-guide/architecture-decisions.md).

## Gateway restart versus host restart

They are not the same event, and they have opposite correct responses.

**A gateway restart must not disturb granted work.** A worker is launched in its
own transient scope and is granted its claim only after its runtime identity is
verified. Past that grant the worker owns the card: the dispatcher never cancels
it, it keeps its PID, its process start time, and its claim, and a replacement
dispatcher leaves it alone. Existing workers therefore survive an intentional
gateway restart or an installation replacement.

**A host restart must not be mistaken for a dead worker.** When the machine is
re-instantiated (reboot, container restart), a `running` card's recorded PID may
be gone — or reused by an unrelated process. Signalling it would be wrong, and
counting the interruption as a crash would spend the card's retry budget on
something the task did not do.

To tell them apart, every new run records the host instantiation epoch
(`<boot_id>:<pid1_start>`) in `task_runs.metadata["host_epoch"]`, beside the
sealed runtime identity. Before any other reclaim path, the dispatcher compares
that recorded epoch with the live one and pauses the card only on a **positive**
verdict:

| Recorded | Live | Verdict |
|---|---|---|
| valid boot A | valid boot B, A ≠ B | host was re-instantiated |
| valid boot A | valid boot A, valid PID1 start differs | same boot, new container instantiation |
| valid boot A | valid boot A, identical PID1 start | not evidence |
| missing / malformed / unreadable | anything | not evidence — existing behavior preserved |

Runs written before host epochs existed have no recorded value and are never
exempted from normal accounting; they simply keep the behavior they had.

## What a pause looks like

A positive verdict produces exactly one outcome per interruption:

- the run closes with outcome/status `interrupted`;
- the card moves to `blocked` through the same sticky `blocked` event an explicit
  `kanban_block` uses, with reason `host_restarted`;
- the event carries the recorded and live epochs plus the resume phase;
- `consecutive_failures` is neither charged nor reset;
- the stale PID is never probed or signalled, and no successor card is created;
- implementation/review phase, candidate, graph, goal revision, and model pins
  are preserved.

Because the block is sticky, ordinary dispatcher ticks leave the card exactly
where the operator found it: `recompute_ready` does not promote it, and the
dispatcher does not re-pause it. **`hermes kanban unblock <task>` is the only
resume path**, and it keeps its established semantics (including resetting the
dispatcher failure counter).

```
hermes kanban list --status blocked     # find the paused card
hermes kanban show <task>               # blocked reason: host_restarted
hermes kanban unblock <task>            # resume in its recorded phase
```

Dispatcher telemetry, gateway logs and CLI output distinguish `interrupted`
(a sticky-blocked run requiring operator recovery), `cancelled` (an unclaimed
launch left queued), and `crashed` (a worker failure).

## Gateway shutdown

A gateway that is draining or stopping must not hand work to a worker it will
not outlive. The drain flag is set synchronously at signal/stop entry — before
diagnostics — and the dispatcher polls it before preparation, after preparation
and before the claim, during the bootstrap waits, and immediately before the
grant. The final stop check and grant share a boundary lock with drain/stop
transitions, so a stop cannot land between that check and the grant. A grant
that wins the boundary remains final.
The native grant write is nonblocking with a one-second deadline. A partial
frame cannot grant ownership, and receipt cleanup happens outside the boundary.

- **Cancelled before the claim:** the prepared worker is cancelled and the card
  is left queued and uncharged. Nothing about it is written.
- **Cancelled after the claim, before the grant:** the ungranted worker is
  cancelled and that exact run closes as `interrupted` with the sticky
  `gateway_stopping` pause.
- **Cancelled after the grant:** nothing happens. See "gateway restart" above.

The same treatment applies when the *user manager* is already stopping — the
race that can precede the gateway's own signal, where scope creation fails before
any drain flag is set. That is accepted only on positive evidence: `systemctl
--user is-system-running` reporting `stopping`. A timeout, a missing
`systemctl`, an inaccessible bus, `degraded`, or arbitrary error text is not
evidence, and a genuine launch failure still fails closed and counts normally.
The probe decodes output as UTF-8 with replacement for invalid bytes; malformed
output cannot become positive shutdown evidence.

The evidence is evaluated at the native scope/bootstrap failure site and converts
only the failure it was sampled beside. Dispatcher exception handlers exempt
nothing, so a runtime-identity, workspace or database failure stays charged and
keeps its real phase (`runtime_identity` only for identity errors, `launch`
otherwise) even while a shutdown is concurrent.

Bookkeeping after a successful grant — the claimed hook, the durable PID write,
the spawned hook — is isolated per step and never cancels the worker, pauses its
run, or charges a retry. That run belongs to a live, verified worker, so a failed
hook or a failed PID write is logged and the spawn still counts rather than
requeueing the card beside a worker that is already running.

## Failure accounting

Ordinary worker crashes are counted against the configured
`kanban.failure_limit`. Precedence, highest first:

1. per-task `max_retries`;
2. `kanban.failure_limit`;
3. the built-in default.

Three deliberate policies outrank the configured limit and are unchanged:

- **systemic** — three or more identical error fingerprints in one tick trip
  immediately;
- **protocol violation** — a bounded violation-only budget, independent of
  `consecutive_failures`;
- **quota wall** — a rate-limited exit releases the card without counting a
  failure at all.

An interruption (`host_restarted`, `gateway_stopping`) is never counted by any of
these.

## Budget placement

Workers are launched with `systemd-run --user --scope --slice-inherit`, so a
worker keeps its own cgroup while remaining a sibling of the gateway service
inside the gateway's slice. A worker is therefore bounded by the slice's budget
and every ancestor of it, not by the gateway service's own `MemoryMax`; an OOM
inside a worker cannot take the gateway down.

That flag needs systemd ≥ 248, and the capability is probed separately from
"can we make a scope at all". Without it, workers retain their legacy managed
scopes, restart survival and per-worker memory isolation. A warning reports
that the shared slice budget cannot be honoured. Required workers still refuse
to launch when no transient scope can be created.

# Working this fork — verification and review rules

Fork-local guidance for `loota-main`. `AGENTS.md` stays upstream's; this file
records the rules this fork's agents learned the hard way. `hermes-cto/AGENTS.md`
points here, so it applies to work done in this checkout *and* in its worktrees
(`hermes-agent-wt/…`), which sit outside the repo and therefore never load
`AGENTS.md` on their own.

## Match verification scope to blast radius

`scripts/run_tests.sh` with no paths is the contract, and a curated file list is
only a supplement to it. Seven runs of a 14-file matrix passed while three
consumer suites were broken: a worker-bootstrap import, a handoff-cancellation
fixture and a version-fenced update. Every one of them was visible in the full
suite and invisible in the matrix.

- Behavioural change to a shared module → run the full suite before pushing.
- Documentation or a new standalone script → its own tests plus the project
  linters are proportionate; say which you ran.
- The project's own linters are part of the gate: `scripts/check-windows-footguns.py`
  (pass explicit paths or `--staged`), the contributor audit, and case-collision
  checks. They cost a minute; skipping them costs a CI cycle.

## Enumerate call sites before changing a shared helper

Introducing a local binding inside a function (a guarded import, a captured
name) silently breaks every other call site in that function. One `grep` for the
symbol — or LSP references where a server is available — before editing, and the
report should say how many call sites were checked. A tolerant import is still a
behaviour change.

## Read the failure yield, not the pass count

A review loop's findings change kind over time. Early rounds find defects in the
feature; later rounds find regressions introduced by the earlier fixes. When the
majority of findings are self-inflicted, more micro-review is the wrong tool —
freeze the surface and run the widest integration check available (full suite,
merged-base CI, deploy rehearsal). Micro-review at that point just manufactures
the next regression.

## Known pre-existing failures

`.github/known-pre-existing-failures.txt` lists tests that already fail on the
base branch, each with the commit and date it was observed. After a CI run:

```bash
scripts/check_known_failures.py <pytest-output-or-junit.xml>
```

It prints which failures are known and which are `UNEXPECTED` (exit 1). Before
claiming a branch broke something, run the same file against a base worktree —
`git worktree add --detach /tmp/base origin/loota-main` — because "fails here"
and "fails on base" are different claims and only the second one exonerates.

## Deploy-adjacent changes

Code that the gateway, dispatcher or worker bootstrap touches is verified once it
runs on the host. Before deploying, run the private ops repo's
`scripts/rehearse_deploy.sh`; it checks the host contracts (unit properties,
process environment, startup evidence, receipt provenance) that no source gate
covers.

# M9 — launch resolution, workspace bootstrap, and resume

Parent plan: [`00-orchestrator-plan.md`](00-orchestrator-plan.md)

Depends on: M2, M5, and M8.

## Objective

Resolve one compatible expert/snapshot/adapter/runtime tuple before spend,
transactionally build the live workspace, and make that identity durable before
orchestrator construction. Resume verifies the original pin rather than following
new GitHub pointers.

## Owned responsibilities

- `LaunchResolver` compatibility and trust decision.
- Empty-scope E0/S-EMPTY bootstrap orchestration.
- `TaskAdapterManifest` validation/materialization.
- `LaunchManifest` and pre-orchestrator `BootstrapPin` persistence.
- Atomic expert workspace and read-only snapshot/adapter construction.
- `Kapso.evolve`, CLI, `ExperimentWorkspace`, and `RunCheckpoint` integration.
- Resume identity/reconciliation and fresh denylist enforcement.
- Direct replacement/removal of active `initial_repo`/starter-selection behavior.

## Proposed code surface

```text
src/kapso/cross_run/launch/
  __init__.py
  resolver.py
  workspace.py
  revocation.py

src/kapso/
  kapso.py
  cli.py

src/kapso/execution/
  orchestrator.py
  run_checkpoint.py

src/kapso/execution/experiment_workspace/
  experiment_workspace.py
  experiment_session.py

tests/
  test_launch_resolver.py
  test_bootstrap_pin.py
  test_cross_run_workspace.py
  test_cross_run_resume.py
  test_cross_run_kapso_api.py
```

## Launch request and resolver

The caller supplies a complete launch request:

```text
scope/task-family identity
task adapter identity
goal/input/target contracts
starting artifact requirements
runtime/dependency/hardware envelope
requested coding-agent/search mode
budget/fidelity envelope
```

`LaunchResolver`:

- [ ] Validates the request against the current `ExpertScopeContract`.
- [ ] Uses M2 to resolve expert and knowledge discovery pointers once at protected
      branch commits.
- [ ] Resolves and verifies the exact task adapter.
- [ ] Checks release/module preconditions, task-family bindings, context dimensions,
      dependency/runtime/hardware compatibility, expiration/revalidation, and
      denylist generation.
- [ ] Verifies the chosen expert release and knowledge snapshot were tested as an
      eligible combination or under an explicit compatibility policy.
- [ ] Creates one immutable `LaunchManifest` binding all identities, digests,
      publications, expected source composition hash, and request hash.
- [ ] Never exposes independently mutable current pointers to the run.

If no release/snapshot exists, the resolver invokes the explicit bootstrap
workflows: M7/M8 must publish validated E0 and M5 must publish validated EMPTY.
Absence, auth failure, or corrupt state cannot activate bootstrap implicitly.

## Transactional workspace builder

For a fresh launch:

1. materialize expert source and knowledge/search packages through M2 cache;
2. materialize/verify the read-only task adapter;
3. stage a new workspace from the expert source archive;
4. bind adapter/evaluation/data interfaces without copying mutable shared state;
5. validate source composition hash, module contracts, semantic book, and task
   adapter boundary;
6. flush and atomically rename the complete workspace;
7. write/flush `BootstrapPin` containing launch ID, local paths/tree hashes,
   denylist generation, and verification receipts; and
8. only then construct `ExperimentWorkspace`, strategy, and coding agent.

- [ ] Partial workspace construction has no bootstrap marker and is never resumed.
- [ ] Snapshot/search and shared adapter roots remain read-only.
- [ ] Local experiments branch only inside the new workspace.
- [ ] `RepoMemory` is rebuilt from the actual composed workspace, not reused from
      the expert release or another run.
- [ ] No model/embedding/evaluator call begins before step 7.

## API and CLI integration

- [ ] Add explicit config-path/scope/task-adapter launch inputs to `Kapso.evolve`
      and `kapso evolve` while keeping operational defaults in config.
- [ ] Replace direct `initial_repo` cloning and starter-repository selection with
      `LaunchResolver`/`StarterWorkspaceBuilder`.
- [ ] Delete the old active arguments, config keys, selectors, cloning helpers,
      prompts, tests, and docs when the new path activates; do not retain aliases.
- [ ] Preserve an explicit task starting-artifact contract through the task adapter
      rather than a generic repository escape hatch.
- [ ] Return launch/snapshot/expert/task-adapter identities in result metadata.

M9 owns these high-conflict files until M10 performs final cleanup/activation.

## Checkpoint and resume

- [ ] Replace `RunCheckpoint` and Generic state with exact new schemas carrying the
      launch ID and bootstrap-pin digest.
- [ ] On resume, require the original `BootstrapPin`, workspace tree, read-only
      snapshot package, adapter, checkpoint, IdeaArchive, experiment store, journal,
      and branches to reconcile.
- [ ] Never re-resolve expert/knowledge `CURRENT` or replace a pinned component.
- [ ] Refresh and authenticate only the current security/contamination denylist.
- [ ] If a new performance revocation exists, preserve reproducibility but mark run
      output/promotion eligibility under policy.
- [ ] If a security/contamination revocation affects the pin or derivatives, fail
      closed before agent execution/evaluation/publication.
- [ ] Checkpoint the observed denylist generation and every derivative taint.
- [ ] Old checkpoint/bootstrap shapes fail explicitly; no migration.

## Failure and trust behavior

- Missing/corrupt/unauthorized/incompatible/expired artifacts fail before spend.
- Network failure during fresh resolution fails; no local unpinned substitute.
- A verified local cache may support normal offline scientific work only after one
  exact launch is resolved and pinned.
- Resume still requires configured fresh security-denylist state; performance-only
  state may use the immutable offline pin under policy.
- GitHub bytes are untrusted until publisher, attestation, artifact identity, and
  digest verification completes.

## Tests

- Resolve compatible expert/snapshot/adapter tuples and reject every incompatible
  dimension independently.
- Reject torn pairs, substituted manifests, cross-task launch reuse, stale release,
  forged publisher, and denylisted component.
- Bootstrap explicit E0/EMPTY; prove missing remote does not trigger it.
- Inject death after every download/stage/rename/pin/checkpoint boundary.
- Prove no coding-agent/embedding/evaluator call occurs before `BootstrapPin`.
- Resume after remote pointers advance and require original local pin.
- Corrupt each local component/receipt/tree and require fail-loud resume.
- Exercise performance and security revocation differences.
- Verify expert repo is writable only inside the run workspace and snapshot/adapter
  roots remain read-only.
- Verify old `initial_repo` and checkpoint paths are absent after activation.

## Definition of done

- Every run begins from one verified atomic launch identity.
- Fresh startup and resume expose no partial/torn component combination.
- The current run remains reproducible when GitHub pointers advance.
- Security/contamination freshness is enforced before dangerous operations.
- The old starter/`initial_repo` path and old persisted shapes no longer exist.

## Non-goals

- Building expert or knowledge releases.
- Capturing/publishing run evidence.
- Task-specific adapter business logic.
- Providing a fallback non-cross-run startup mode.

# M9 — launch resolution, workspace bootstrap, and resume

Parent plan: [`00-orchestrator-plan.md`](00-orchestrator-plan.md)

Depends on: M2, M5, and M8.

Status: **in progress**. The transactional launch resolver and exact authority
contracts, atomic workspace/bootstrap-pin installation, exact run-checkpoint
contracts, the protected checkpoint CAS store, the immutable derived-generation
contract/layout, and the dependency-pure reconciled archive/history/journal
projection and retained bundle layer, single-lock checkpoint/generation
publisher, mutable-view promotion, current-frontier action reservation, receipt-pinned
create-only action store, mandatory action-ledger projection, and durable
reservation/recovery composition, descriptor-safe runtime reopen, and deterministic
nonterminal action recovery are implemented. The pinned Docker CLI, daemon, and
image authority is now a domain-neutral runtime reusable by M8 evaluators and
M9 action adapters, with shared host/runtime authority owned once by top-level
`cross_run.docker` rather than by either consumer. Boundary-specific production
adapters and the mechanically constrained supervisor remain, while the
content-addressed Docker execution-policy, deterministic preparation-claim,
inert-evidence, prepared-execution contracts, and durable eight-event action prefix
are implemented. Every Docker volume-subpath now has preparation-time physical
authority, and post-spawn request/credential delivery plus pre-start volume
reobservation are crash-atomic and descriptor-bound. Full event-5 receipt
assembly, committed-container reinspection and token-sealed start, terminal
Docker inspection, and adopted-release result authority are implemented.
Descriptor-bound result capture, the closed typed-termination evidence
contracts, terminal event-6 persistence/replay, the closed semantic control
topology, and exact crash adoption of a published timeout directive are
implemented. Physical timeout publication and provider termination, cleanup, OS
executor activation, explicit E0/S-EMPTY provisioning orchestration, full policy
refresh on resume, and API/runner activation remain.

## Objective

Resolve one compatible expert/snapshot/adapter/runtime tuple before spend,
transactionally build the live workspace, and make that identity durable before
orchestrator construction. Resume verifies the original pin rather than following
new GitHub pointers.

## Owned responsibilities

- `CrossRunTaskBinding` to `ScopeRegistry` resolution.
- `LaunchResolver` compatibility and trust decision.
- Empty-scope E0/S-EMPTY bootstrap orchestration.
- `TaskAdapterManifest` validation/materialization.
- `LaunchManifest` and pre-orchestrator `BootstrapPin` persistence.
- Atomic expert workspace and read-only snapshot/adapter construction.
- `Kapso.evolve`, CLI, `ExperimentWorkspace`, and `RunCheckpoint` integration.
- PostTrainBench/RelBench scope-binding config and runner integration.
- Resume identity/reconciliation and fresh denylist enforcement.
- Direct replacement/removal of active `initial_repo`/starter-selection behavior.

## Proposed code surface

```text
src/kapso/cross_run/launch/
  __init__.py
  resolver.py
  workspace.py
  revocation.py
  run_action_reservation_contracts.py
  run_action_spawn_contracts.py
  run_action_supervisor_contracts.py
  run_action_activation_delivery.py
  run_action_runtime_volume.py
  run_action_docker_barrier.py
  run_action_result_capture.py
  run_action_docker_adapter.py

src/kapso/
  kapso.py
  cli.py

src/kapso/execution/
  orchestrator.py
  run_checkpoint.py

src/kapso/execution/experiment_workspace/
  experiment_workspace.py
  experiment_session.py

benchmarks/posttrain/
  config.yaml
  runner.py

benchmarks/relbench/
  config.yaml
  runner.py

tests/
  test_launch_resolver.py
  test_bootstrap_pin.py
  test_cross_run_workspace.py
  test_cross_run_resume.py
  test_cross_run_kapso_api.py
```

## Launch request and resolver

The caller supplies a complete launch request. It names a scope/task binding, not
GitHub repositories:

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

- [x] Resolves `scope_id` through M1's canonical `ScopeRegistry`; reject any
      caller-supplied or benchmark-level repository override.
- [x] Passes the resolved `ScopeRepositorySettings` to M2, resolves the two
      scientific discovery pointers once at exact default-branch commits, and
      freshly authenticates the security pointer through M8's denylist authority.
- [x] Verifies repository publication records, expert release, knowledge snapshot,
      and task adapter all name the requested scope lineage.
- [x] Validates `task_family_id` and `task_adapter_id` against the pinned current
      `ExpertScopeContract` before materialization.
- [x] Resolves and verifies the exact task adapter.
- [x] Checks the accepted expert release-matrix adapter authority, task-family
      capability, input/target and artifact interfaces, exact capability tags,
      method, toolchain, transfer dimensions, hardware envelope, artifact mount
      layout, OCI/dependency runtime, artifact TTL, release-use state, and complete
      security-denylist closure.
- [x] Verifies the chosen expert release and knowledge snapshot were tested as an
      eligible combination or under an explicit compatibility policy.
- [x] Creates one immutable `LaunchManifest` binding all identities, digests,
      publications, the security snapshot/generation, the scope
      repository-binding hash, expected source composition hash, and request hash.
- [x] Never exposes independently mutable current pointers to the run.

The implemented v1 compatibility policy deliberately permits only new
starting-artifact content within an otherwise exact verified release-matrix
interface. A launch-specific materialization receipt binds each declared artifact
ID to its verified file bytes, tree, reference, and mount before compatibility is
issued. The receipt records the compatible adapter cases, accepted matrix adapter
authority, and exact typed context. Free-text module preconditions and resource
notes remain evidence for agents, not executable admission predicates; any
constraint that must gate launch must first become a typed task-context or adapter
contract field and be covered by the release matrix. Expert and knowledge
`CURRENT`, their immutable intent and activation refs, and the active adapter are
all re-read after policy checks; any movement aborts launch.

`LaunchManifest` is immutable evidence, not a standalone in-process capability.
Only a resolver-sealed `ResolvedLaunch` may cross into workspace bootstrap; M9's
`BootstrapPin` will bind the complete manifest digest so immutable evidence identity
is never confused with live one-shot issuance authority.

If no release/snapshot exists, the resolver fails before spend. The launch
coordinator must invoke explicit administrative bootstrap workflows—M7/M8 publish
validated E0 and M5 publishes validated S-EMPTY—and then restart ordinary
resolution from GitHub `CURRENT`. Absence, auth failure, or corrupt state cannot
activate bootstrap implicitly, and bootstrap authority never substitutes for a
missing scientific `CURRENT`.

## Transactional workspace builder

For a fresh launch:

1. require and atomically consume the resolver's live one-shot `ResolvedLaunch`;
2. pin the owned, non-shared destination parent and private staging root by file
   descriptor, and reject any pathname/inode substitution;
3. copy the verified in-memory expert, knowledge, adapter-runtime, and task-input
   byte closures—never links or mutable cache paths;
4. revalidate expert topology, ownership, generated controls, semantic book, task
   adapter projection, and expected source composition;
5. keep knowledge, adapter runtime, and task inputs under one exact-mode read-only
   envelope outside the writable expert repository; reject links, shared inodes,
   special files, extra paths, or writable descendants;
6. construct deterministic raw Git blobs/trees/commit/index over exactly the
   expert source, with no porcelain, hooks, global filters, clock, or host identity;
7. mint a `WorkspaceInstallationReceipt` that pins the knowledge-cache tree join
   and exact Git object/index closure, embed it and the full `LaunchManifest` in
   `BootstrapPin`, enforce launch-specific byte bounds for every staged/reopened
   source, knowledge, Git, and control file, and write the pin last;
8. fsync every file and directory, atomically no-replace rename the complete run
   root, fsync its parent, and reopen every layout path through descriptor-relative
   no-follow walks;
9. issue one non-clonable `PreparedLaunchWorkspace`; burn its authority before
   repeating the complete descriptor-safe closure check at orchestration handoff;
   and
10. only then construct `ExperimentWorkspace`, strategy, or any paid dependency.

- [x] Partial construction remains hidden staging and is never resumable.
- [x] Every visible fresh run root contains the complete self-contained pin.
- [x] Symlinked component/control ancestors, extra run-root state, replaced inodes,
      and cloned or replayed prepared handles fail closed.
- [x] Snapshot/search, adapter runtime, and task-input roots are fresh read-only
      copies outside the writable Git repository.
- [x] The baseline commit, tree, and index are deterministic across local paths,
      umasks, and host Git configuration; reopen requires the exact config,
      identity, ref, loose-object, index, file-mode, and directory closure.
- [ ] Before activation, route every coding-agent/evaluator process through a
      fail-closed execution permit whose OS sandbox cannot resolve the run-root
      control plane; a writable working directory alone is not isolation.
- [ ] `RepoMemory` is rebuilt from the actual composed workspace, not reused from
      the expert release or another run.
- [ ] API activation proves no model/embedding/evaluator call begins before step 9
      completes.

## API and CLI integration

- [ ] Add explicit config-path plus `scope_id`/`task_family_id`/`task_adapter_id`
      launch inputs to `Kapso.evolve` and `kapso evolve`. Never accept expert or
      knowledge/security repository coordinates through the API/CLI.
- [ ] Make benchmark runners obtain those three values from their typed
      `cross_run_binding`; callers normally select a benchmark mode rather than
      repeat the binding on every task.
- [ ] Bind PostTrainBench to
      `ml_ai/language_model_post_training/posttrain` and RelBench to
      `ml_ai/relational_tabular_prediction/relbench`; neither benchmark file may
      name any repository.
- [ ] Replace direct `initial_repo` cloning and starter-repository selection with
      `LaunchResolver`/`StarterWorkspaceBuilder`.
- [ ] Delete the old active arguments, config keys, selectors, cloning helpers,
      prompts, tests, and docs when the new path activates; do not retain aliases.
- [ ] Preserve an explicit task starting-artifact contract through the task adapter
      rather than a generic repository escape hatch.
- [ ] Return launch/snapshot/expert/task-adapter identities in result metadata.

M9 owns these high-conflict files until M10 performs final cleanup/activation.

## Checkpoint and resume

The implemented checkpoint foundation governs publication, reservation, and
action recovery, while production provider execution remains dormant until the
supervisor and transactional edit-promotion work below land. One exact
`RunCheckpoint` owns strategy, safety, cost, feedback, termination, bootstrap,
derivative-frontier, and derived-generation authority. A candidate can advance
only the exact predecessor; Generic and tree strategy states enforce their own
exact revision transitions.

Every checkpoint transition embeds one content-addressed
`RunDerivedStateGeneration`. It binds the exact predecessor checkpoint-head,
checkpoint and evidence frontier, a bootstrap-pinned strategy-specific state
layout, and the old/new digest, revision, and size of every state projection.
Generic owns `idea_archive`, `experiment_history`, and `execution_journal`; tree
owns only the latter two. The archive bytes must exactly equal the archive snapshot
inside Generic strategy state, while history and journal revisions must agree with
the exact number of executed node revisions.

The builder creates separate private object and staging directories outside the
writable expert repository. A retained `generation-<digest>.bundle` will contain
the canonical manifest plus the exact archive/history/event bytes, including
non-reconstructible embeddings and timestamps. Public JSON/JSONL files are
read-only, repairable views—not peer authorities. Generation objects remain
reachable until a future safe-GC proof shows no checkpoint/capture dependency.
The current checkpoint staging cleaner cannot inspect or delete this separate
generation store.

The checkpoint pathname is a replaceable current projection. Its monotonic local
floor is a builder-created append-only journal on one inode whose device/inode,
lock inode, and exact launch-settings identity are sealed into
`WorkspaceInstallationReceipt` and therefore `BootstrapPin`. Every hash-chained
journal record retains the full canonical checkpoint, so recovery can rerun the
complete strategy/archive/safety predecessor relation rather than trusting an ID.
`RunStatePublisher` runs the complete transition under that pinned lock and:

1. reopens the canonical journal/checkpoint lineage, referenced generation, and
   every strategy-owned view through descriptor-relative no-follow paths;
2. binds a one-shot publication permit to the observed checkpoint, journal head and byte
   position, candidate, generation, and exact bundle digest/size;
3. validates the candidate's complete predecessor closure and rejects journal,
   generation, view, store-entry, or staging exhaustion before authoritative
   mutation;
4. publishes and reopens the immutable bundle with no-replace semantics, replaces
   and reopens the checkpoint, and appends/fsyncs/reopens its monotonic journal
   head;
5. promotes every repairable view solely from that retained bundle; and
6. returns a live non-clonable `ReconciledRunFrontier` only after reopening and
   jointly verifying checkpoint, journal, bundle, projection, and views.

On explicit reconciled reopen, only one authoritative crash seam is repairable:
the checkpoint is exactly one fully validated successor ahead of the journal and
its exact retained bundle is already durable. A partial final journal record is
repairable only when it is an exact prefix of that deterministic successor record;
safe missing or byte-stale views are rebuilt afterward. All unrelated tails,
missing/corrupt referenced bundles, missing/stale checkpoints, skipped revisions,
canonical rollback substitutions, unsafe views, journal/lock inode changes, and
malformed records fail loudly.

The process UID remains inside the local filesystem trust boundary: an unrestricted
same-UID process can rewrite an append-only-by-protocol regular file in place.
Therefore M9 must not activate this layout through the legacy unsandboxed coding
agent. Descriptor-safe execution permits and policy probes must prove that agent
and evaluator processes cannot resolve `.kapso` or any path outside the writable
expert workspace before the new runtime becomes reachable.

- [x] Replace `RunCheckpoint` and Generic state with exact new schemas carrying the
      launch ID, bootstrap pin, strategy-specific state layout, and immutable
      derived-generation dependency.
- [x] Build canonical dependency-pure archive, experiment-history, and execution
      journal projections; require one mutually reconciled frontier, typed
      embedding-space/input proof, exact strategy/idea/outcome/artifact lineage,
      and retain its complete immutable generation bundle.
- [x] Make one shared-lock publisher stage/fsync the complete generation, CAS the
      checkpoint, idempotently promote every mutable view, verify all bytes/refs,
      and return a non-clonable reconciled-frontier receipt. A durable checkpoint
      alone must never authorize capture or another paid/dangerous action.
- [x] Require a live reconciled receipt for every non-genesis state publication,
      and persist one exact action reservation from complete request bytes.
      Recovery reopens the pinned workspace by descriptor, reconciles source and
      Git state to checkpoint evidence, and authenticates the current denylist
      before issuing sealed transition capabilities; state publication retains
      the exclusive checkpoint lock.
- [x] Make the action store authoritative across processes: compare-and-swap each
      reservation against the exact predecessor ledger, persist intent/claim/
      prepared/spawn/result/decision/acceptance events, hold the receipt-pinned workspace
      lock, require terminal live state for publication, and derive branch
      accounting from durable events.
- [x] Recover one exact final nonterminal prefix without replaying a committed
      spawn; bind recovery to the complete frontier and an issued exact
      implementation catalog, burn preparation and activation authority once, and
      replay terminal accepted bytes without execution or interpretation access.
- [ ] On resume, require the original `BootstrapPin`, workspace tree, read-only
      snapshot package, adapter, checkpoint, IdeaArchive, experiment store, journal,
      and branches to reconcile.
- [ ] Never replace a pinned expert or scientific knowledge component. A narrow
      policy-only reader may resolve current knowledge `CURRENT` solely to refresh
      the release-use projection.
- [ ] Refresh and authenticate the current security/contamination denylist and,
      when online eligibility is required, the release-use policy projection.
- [ ] If a new performance revocation exists, preserve reproducibility but mark run
      output/promotion eligibility under policy.
- [ ] If a security/contamination revocation affects the pin or derivatives, fail
      closed before agent execution/evaluation/publication.
- [ ] Require the live authenticated snapshot to equal or descend from both the
      bootstrap pin and durable local floor; checkpoint its exact identity,
      generation, publication, pointer, and every derivative taint.
- [ ] Old checkpoint/bootstrap shapes fail explicitly; no migration.

`RunStatePublisher` is now the only publication API for the new run-state
authority. The checkpoint-only permit and durable receipt were deleted. Its
one-shot permit binds the observed checkpoint, journal head and byte position,
candidate, generation, and exact bundle digest/size. Under the pinned lock it
publishes the immutable bundle with no-replace semantics, commits the checkpoint
and monotonic journal, promotes all strategy-owned views, reopens the full closure,
and issues a live non-clonable `ReconciledRunFrontier`.

Construction and ordinary inspection never repair checkpoint state. Explicit
`load_reconciled()` may repair only an exactly adjacent checkpoint after the
referenced bundle has decoded and reconciled, and may rebuild safe missing or
byte-stale views solely from that bundle. Missing/corrupt referenced objects,
unrelated journal tails, unsafe views, stale permits, and inode substitutions fail
loudly. A bundle published before a failed checkpoint CAS remains an inert orphan
and can be reused only by a later authorized candidate naming it exactly.

`RunFrontierActionGate` has one mutation: it durably reserves event 1 for a
coding-agent, embedding, or evaluator call. It derives an immutable intent from
the complete canonical request bytes and binds it to the exact checkpoint,
safety, generation, journal, bundle, view, and workspace closure. The same
operation or content-derived intent cannot be reserved twice. Only active,
non-yielded checkpoints act, and the capability matrix is exact:
ideation/evaluation coding agents and evaluators read the workspace,
implementation coding agents edit it, and embeddings receive no workspace
binding.

Reservation holds the checkpoint and descriptor-safe workspace locks only long
enough to revalidate the complete frontier and append `INTENT_RESERVED`. It
proves that the owner-private workspace has the checkpointed branch head, a clean
source tree equal to the commit tree, an exact flag-free Git index, one exact
configured ref, no replace/alternate/shallow/graft state, and a bounded
self-contained loose object store equal to the complete reachable
commit/parent/tree/blob closure. Every later transition belongs exclusively to
`RunActionRecoveryCoordinator`, which reacquires and reproves the authorities
needed for that exact durable prefix. Checkpoint publication remains exclusive.

An edit is exclusive across processes against all workspace readers and other
edits. A successful result must finish as one clean direct-successor commit with
canonical Git header grammar; a failed result may terminate only with the exact
unchanged predecessor workspace. A successful edit spends the predecessor
frontier, so no later action or publication may use it until a checkpoint
successor records exactly one authorized `RunBranchAdvance` to that commit.
Reconstructed gates and publishers derive this state from durable event prefixes
rather than process-local memory.

Successful edit promotion uses immutable whole-workspace generations, not an
in-place file transaction. The public workspace pathname is stable, but its leaf
inode is dynamic; its ancestors, the run root, the workspace lock, and a private
same-filesystem promotion-staging root remain receipt-pinned. Before event 7, the
coordinator copies and fully reconciles the isolated result into that staging
root without changing the public workspace. The source is a process-bound
descriptor lease reopened directly from event 3's prepared keeper/volume/workspace
proof and event 6's terminal result-capture volume and generation-sentinel
evidence. The lease retains and re-proves the keeper process, mounted root,
sentinel pathname/content/inode, and workspace pathname/inode; neither the
execution adapter nor the result interpreter receives promotion authority. Event
7 requires the promotion exactly for a successful edit and binds its result
receipt, prepared-workspace proof, and staged clean direct successor. Recovery
then admits exactly `(public=predecessor, stage=successor)` or
`(public=successor, stage=predecessor)` and performs at most one atomic
`RENAME_EXCHANGE`. Event 8 accepts exactly that candidate only after both parents
are fsynced and the public successor is fully re-inspected.

The retired predecessor remains in staging until event 8 is durable. Cleanup is
bounded, descriptor-relative, mount-ID confined, identity-reproved, and
idempotent; recovery retries partial cleanup and also reconciles the latest
accepted promotion when event 8 is already included in the checkpoint projection.
Ledger ownership is checked first: a newer durable event-7 stage is never treated
as older cleanup residue, while pre-event-7 staging can be discarded and derived
again from event 3/event 6. Before event 7, an interrupted temporary tree is
validated and removed, while a fully renamed candidate is reproved and reused. No
per-path write-ahead log or mutable selector exists.

This protocol assumes a local filesystem that implements crash-atomic,
same-filesystem `renameat2(RENAME_EXCHANGE)` and persists the exchange after both
parent directories are fsynced. Production activation therefore requires a
destructive VM/filesystem power-loss test, not only injected process exits. The
receipt-pinned workspace flock is the cooperating-process exclusion boundary:
the lock pathname and both exchange-parent paths are rebound immediately before
mutation, while a hostile unrestricted same-UID process that ignores the lock is
outside the runtime trust boundary and must be excluded by the provider sandbox.

A live authenticated denylist descendant must equal the observation already
checkpointed in the safety state; any advance requires a durable safety-state
successor before work can begin. Security-blocked state cannot act.
Reproducibility-only state may continue scientific work but remains ineligible
for promotion. The generic provider APIs and OS sandbox are not yet wrapped by
this gate, so the new runtime remains dormant.

The durable foundation is now receipt-pinned and create-only. It stores complete
untruncated request bytes, provider result bytes, accepted canonical result bytes,
predecessor-linked operation events, and a metadata-only `ACTION_LEDGER` in every
derived-state layout. The ledger is an explicit mandatory projection and
derivative-evidence authority; terminal prefixes cannot be extended or rolled
back. The registry, workspace, and lifetime-runtime locks are fixed and
inode-bound by the workspace receipt. An operation's immutable first event is
also its lock inode, so a losing or capacity-rejected reservation cannot strand
an empty permanent lock. Structurally valid orphan staging files and final blobs
without a referencing event are cleaned under the pinned registry lock before
reuse.

Fresh activation and restart now converge on one process-bound
`ActiveLaunchWorkspace`. `StarterWorkspaceBuilder.reopen()` reads the bounded,
canonical local `BootstrapPin` without consulting a resolver or remote
`CURRENT`, validates its exact configured layout and settings identity, acquires
the receipt-pinned runtime lock with nonblocking exclusive semantics, and only
then repeats the full immutable/control closure verification. The authority
retains both the root and runtime-lock descriptors for its lifetime; checkpoint,
workspace, and action-store access begins from a duplicate of that root
descriptor. Explicit close, process death, or garbage collection releases the
lease. Forked children close inherited descriptors, lose both prepared and active
registries, and cannot use or release the parent's authority. Resume deliberately
does not require the initial checkpoint journal or initial writable expert tree:
the publisher and action-recovery layers own those evolved authorities.

The gate persists `INTENT_RESERVED` before returning a reservation. The recovery
coordinator alone persists
`SPAWN_COMMITTED` before exposing bounded delivery/revalidation authority,
`ACTIVATION_COMMITTED` before provider-start authority, complete raw results
before interpretation, complete interpreted decisions before workspace
completion, and terminal acceptance with the exact post-action workspace. A
crash deliberately leaves the last complete durable prefix for
resume. Admission is a ledger compare-and-swap, so reconstructed gates and
concurrent processes cannot both reserve against the same live floor. The old
permit, lease, context manager, and direct gate transition methods have been
deleted; store transition entry points are internal and sealed to recovery
authority. Provider execution
IDs and invocation nonces are unique across the full store, not merely within
one operation. Every reservation and spawn also pins one content-addressed
boundary identity. The boundary jointly embeds an action-kind-bound,
content-addressed execution-lifecycle identity and an action-kind-bound,
content-addressed pure result-interpreter identity. A substituted lifecycle,
interpreter, implementation method, recovery protocol, or sandbox policy is
rejected against the exact process-local composition.

`RunActionRecoveryCoordinator` classifies the exact action-ledger suffix under
the current checkpoint and workspace locks. The suffix must be one predecessor
chain with at most one final nonterminal operation, and every reservation must
match the complete current frontier, including all mutable-view digests.
Terminal operations replay their complete accepted bytes without contacting an
execution adapter, result interpreter, or provider.

Recovery is an explicit eight-event state machine:

```text
INTENT_RESERVED
  → PREPARATION_ALLOCATED
  → EXECUTION_PREPARED
  → SPAWN_COMMITTED
  → ACTIVATION_COMMITTED
  → RESULT_RECEIVED
  → RESULT_DECIDED
  → RESULT_ACCEPTED
```

`INTENT_RESERVED` may cancel before logical allocation when its frontier is stale.
Allocated or prepared work may close as `FRONTIER_INVALIDATED` before spawn only
after the reserved workspace is re-proved unchanged and another frontier
authority is stale. A workspace mismatch itself remains unresolved: this event
cannot claim that an unbound external successor is the action's result. The
terminal records the reservation's exact unchanged workspace binding (or no
workspace for a workspace-free action) because no request, credential, or start
authority existed. Physical resource loss is not a terminal fact: an absent,
unknown, or temporarily unreachable occurrence remains unresolved until
preparation can return complete exact evidence. A persisted prepared occurrence
is never replaced. `SPAWN_COMMITTED` means the exact
occurrence has no selected activation receipt and may only stage an inert
activation or remain unresolved. `ACTIVATION_COMMITTED` selects the sole receipt
that may precede start and may terminate only as a result or provider
interruption. `RESULT_DECIDED` binds the exact result receipt, pure interpreter
identity, disposition, and complete accepted bytes. `RESULT_ACCEPTED` binds that
decision to the final workspace proof. Every event repeats the exact reservation
and predecessor, while allocation, prepared, spawn, activation, result, decision,
and acceptance payloads bind their immediate durable authority.
The store rejects skipped/reordered phases, old intent-to-spawn records, identity
splices, multiple nonterminal operations, and global reuse of claim, allocation,
volume, generation, sentinel, prepared, container, keeper, file, provider, or
invocation identities.

Before any physical Docker mutation, recovery durably appends one logical
allocation containing the deterministic claim and its unpredictable generation-
bound runtime-volume authority. The generation-derived sentinel identity is also
present in the issued Docker volume labels, so even a pre-sentinel volume is
observable as that exact allocation. Recovery rechecks workspace and security
after the event-2 fsync. A same-process/same-thread preparation capability
distinguishes first materialization, allocation reopen, and exact prepared-
occurrence revalidation; it carries the live
workspace descriptor and daemon-visible source path when the policy requires a
workspace. Before physical allocation, the exact lifecycle adapter must twice produce the
same conservative bound for the complete prepared-event encoding; the coordinator
rejects a nonpositive, nondeterministic, or over-limit envelope. The returned
prepared event must then fit that declared bound. Prepared evidence is persisted
before spawn. Security and workspace are checked again immediately before the
durable spawn commit, and the reservation boundary must still equal the
checkpoint safety boundary. Only after that commit does a separate single-use
staging capability expose the complete request and a capability-owned duplicate
workspace descriptor for delivery plus inert revalidation. The adapter declares
the activation-event envelope twice before delivery; the returned receipt must
fit it. The coordinator then rechecks workspace and security and durably selects
that exact receipt as event 5. Read-only inspection follows every fresh or
recovered event-5 selection. Only a distinct process-bound continuation
capability bound to that inspection may start, wait, or capture. The staging
capability burns on every exit and closes its owned
workspace descriptor; the descriptor-free continuation capability also burns on
every exit.
Request bytes are unreadable from the action session before spawn.
Preparation returns one typed state: exact prepared with an origin compatible
with its create/reopen/revalidation mode, or unknown.
Allocation reopen may physically materialize only after positive total absence,
or may re-open only the completely reobserved event-2 occurrence. Prepared
revalidation can return only the identical occurrence or uncertainty; it cannot
allocate. Partial, substituted, absent, or ambiguous event-2 resources remain
unresolved and never receive replacement authority. Administrative frontier
invalidation may close the allocation or prepared prefix only after re-proving
its workspace unchanged, without changing the normal eight-event success chain.

An event-4 query admits only exact inert or unknown state; running or exited state
cannot be adopted without durable activation. Exact inert state may restage and
select event 5. Every event-5 path now uses the same token-sealed continuation
protocol. Read-only inspection returns only `INERT_CONTINUABLE`,
`RUNNING_CONTINUABLE`, `TERMINAL_CONTINUABLE`, or `UNKNOWN`, plus one exact
observation digest for states that may continue. It cannot return result bytes
or terminalize an operation. A distinct
same-process/same-thread single-use capability binds the complete event-2
allocation, event-5 activation event, and exact observation object. Its sole
`continue_committed_once` call must revalidate that observation before it may
start, wait, or capture. Pending work leaves event 5 unchanged; a captured result
is checked against the complete activation evidence before event 6. No
proof-only quiescence or resource-loss state may terminalize the operation.
`UNKNOWN` receives no capability. Immediate post-event-5 execution and restart
therefore share one path, and a failed continuation is retried only after a new
inspection. Recovery never routes committed work through allocation or
preparation and never replaces the event-5 selection.

### Durable Docker supervisor boundary

The supervisor identity graph is acyclic and separates reproducible intent from
one host-local Docker occurrence:

```text
DockerExecutionPolicy
        ↓
PreparationClaim
        ↓
PreparedExecution
        ↓
SpawnCommit → staging capability → ActivationCommitted
                                        ↓
                  inspection token → continuation capability
                                        ↓
             TerminalObservation
                      ↓
             ResultCaptureReceipt
```

`DockerExecutionPolicy` is content-addressed and owned by the process-bound
execution lifecycle: the lifecycle identity contains its exact policy ID, and the
preparation path resolves it rather than accepting a caller-selected policy.
The policy binds the action kind, command-template implementation ID,
content-addressed image and volume-keeper helper authorities, pinned
runtime-settings digest, closed raw-field schema/projection versions,
value-constrained non-secret static environment, UID/GID, filesystem and
activation-network policies, non-secret credential policy, sandbox profiles and
controls, the Docker controls that this host can actually enforce, and separate
supervisor-only time/result bounds. No argv or dynamic string argument is
durable. The adapter renders a fixed lifecycle-owned command in memory from the
template and admitted in-container paths; requests travel only through the input
file and credentials only through the post-commit credential file.
`PreparationClaim` is deterministic before Docker allocation and embeds the
complete validated reservation and exact policy. Independently valid IDs cannot
be spliced across reservations, policies, or same-kind lifecycle implementations.

`PreparedExecution` is deliberately an occurrence receipt, not a reusable
semantic identity. One deterministic local Docker volume name identifies one
random generation. The volume is a keeper-mounted `tmpfs` with exact byte and
inode ceilings, UID/GID and mode, and mandatory `nosuid,nodev,noswap`; execution
remains allowed because coding actions may run workspace outputs. A physical
generation sentinel proves the reopened generation by no-follow
mount/device/inode identity, regular-file type, owner, mode, link count, size,
and content digest. Exact Docker volume name, labels, scope, driver and options
must agree with the issued authority. `statfs` supplies effective byte/inode
limits and actual allocated usage; logical file sizes are not treated as storage
enforcement.

A running, network-free keeper is the sole mount owner. The execution policy
pins the helper source path and digest. Keeper evidence proves the read-only,
nonrecursive bind resolves to the exact root-owned, singly linked, static ELF
file by mount/device/inode, mode, digest, and absence of an interpreter or
dynamic dependencies. The main container never sees the volume root or sentinel.
It receives only prefix-disjoint named-volume subpaths for workspace, input,
result, optional credential delivery, and temporary storage. Input and
credential mounts are read-only; result and temporary mounts are writable;
workspace access exactly follows the reservation.
Input and optional credential final names are absent before spawn; their exact
private parent directories are the durable delivery-slot authority. Only the
result is a pre-created empty, private, singly linked regular file.
`PreparedExecution` pins every path Docker will resolve at start: the input
parent, optional credential parent, result parent and result inode, temporary
root, optional workspace root, volume root, and sentinel by mount/device/inode.
The workspace is copied into the same bounded generation and its observed tree
digest, Git-closure digest, entry count, and byte count must equal the durable
frontier binding.

The deterministic volume, keeper, and main-container names and complete role
labels derive only from `PreparationClaim`, avoiding a back-edge from Docker
state to `PreparedExecution`. Main evidence requires `created`, PID/restart count
zero, zero start/finish timestamps, restart `no`, auto-remove false, network
`none`, no healthcheck, no anonymous or plugin mount, and no Docker-socket mount.
Keeper evidence requires a running, never-restarted, network-free occurrence and
the exact helper and full-volume mount. Both create/inspect projections must
match in both directions. Their pinned raw-field schemas classify every raw path
as projected, required literal, runtime evidence, or explicitly
nonauthoritative; an unknown path fails. Unsupported paper controls—NanoCPU,
real-time CPU, swappiness, abstract block-I/O rules, per-file project quotas, and
`RLIMIT_FSIZE` as a storage boundary—are absent rather than falsely claimed.
Supervisor time and result-byte limits remain supervisor facts rather than
Docker inspection facts. Credential records contain opaque authority and file
shape only—never secret bytes or host credential paths.

The action store now atomically persists one
`PreparationAllocation → PreparedExecution` occurrence before `SpawnCommit`.
Reservation admission accounts for the complete remaining lifecycle: at most eight
event files and three content blobs per operation, plus the configured crash
staging allowance and two fixed lock files. Every append and reopen rechecks its
remaining event/blob headroom, so an accepted intent cannot strand an
irreversible spawn solely because the store later reaches its configured byte or
entry bound. This capacity proof is separate from the lifecycle adapter's
pre-allocation serialization envelope: the former reserves store space, while the
latter proves that the complete concrete prepared record can occupy one event
file before any Docker resource is created. Before `SpawnCommit`, the
production supervisor may only create or reopen that exact inert resource;
request bytes and credential leases remain absent. A post-spawn, single-use
staging capability binds the whole `PreparedExecution`, spawn commit, and either
exact credential-lease receipts or a no-credentials proof. Only the supervisor
may consume it to attach the admitted broker network, populate the prepared
delivery slots, and derive an `ActivationRevalidationReceipt` after re-inspecting
the volume, physical generation sentinel, running keeper, copied workspace,
every prepared mounted-subpath directory, delivered input/credential files, the
pre-created result parent and file, empty temporary root, optional workspace
root, and still-never-started main container. The result-file observation carries
its exact prepared parent identity; temporary and workspace roots have distinct
activation observations. Immutable
volume facts must equal preparation; allocated usage and actual available
blocks/bytes/inodes must form the exact delivery-delta `statfs` relationship, and
the fresh observation must retain positive result-plus-temporary headroom. Workspace
and sentinel observations are distinct activation-time contracts bound to the
exact spawn commit and their preparation evidence IDs; replaying a preparation
object is not revalidation. Event 5 durably selects one full receipt with
create-only publication. Revalidation itself grants no workload authority. After
publication, a new process-bound single-use capability must reobserve the same
inert occurrence, reproduce the selected receipt exactly, and retain the live
resource authority and workspace lock.
It embeds the exact typed spawn commit and delivery predecessors. Request delivery
proves the fixed regular-file name, digest, size, owner/group, read-only mode, and
single link; credential delivery proves the same structural facts plus its opaque
broker lease authority and size, but stores no credential digest or bytes.
Every delivery/proof record binds the exact spawn-commit content ID, including its
invocation nonce; a semantically similar second fence cannot reuse prior delivery.
A crash before event 5 may stage a new candidate. A crash after event 5 may only
revalidate the selected receipt; it can neither overwrite nor append a second
selection.

### Resolved-mount workload barrier

Docker resolves each named-volume `VolumeSubpath` during container start, after
event-5 descriptors were observed. Directly starting the provider command would
therefore leave an uncloseable substitution interval. The sole path instead adds
an empty prepared `control` subpath mounted read-only into the main container and
uses the pinned static supervisor BusyBox as a fixed barrier entrypoint. The
intended command remains policy-bound and is passed as positional arguments;
the fixed shell program never interpolates it. With Docker `--init`, the daemon's
state PID names init, so the supervisor proves the exact direct BusyBox child
rather than misidentifying init as the wrapper. The host authority pins the
configured static `docker-init` source path and digest; both main and keeper
projections carry that descriptor-proven source evidence, and filesystem policy
reserves `/sbin/docker-init` from every workload mount. This is launch-source
authority only. The post-start process proof must separately establish that
Docker's state PID executes those exact bytes with the expected init argv before
release.

One continuation capability performs at most one physical transition and always
returns for fresh inspection:

```text
created + release absent
        ↓ start wrapper
barrier running + release absent
        ↓ prove /proc/<init-pid>/root and mountinfo
        ↓ link exact release record
workload released + release present
        ↓ wait / stop / kill under one absolute deadline
terminal + release present
        ↓ terminal observation and descriptor result capture
```

Before the irrevocable release link, the supervisor sandwiches the init and
wrapper process generations/cgroups, validates the pinned helper bind, and joins
every actual in-container mount root to event 5 by mount/device/inode and access
mode. It separately reproves the input digest/inode, credential inode/size and
opaque lease authority, original empty result inode and parent, empty temporary
root, optional workspace frontier, and empty control root. The content-addressed
workload-release receipt binds that complete resolved-mount observation, the
durable event-5 event ID, prepared/spawn/provider identities, barrier protocol,
credential validity, host boot identity, and fixed execution/grace deadlines.

The complete canonical receipt is fsynced in an anonymous file before
`linkat(AT_EMPTY_PATH)` publishes `control/release` without replacement. The
link itself is workload authorization: the wrapper may observe it before the
directory fsync or caller return, so every scientific and security check occurs
before the link. Exact presence always wins on recovery and is adopted; no path
releases or starts again. The release file remains the crash-surviving authority
until event 6 embeds the complete receipt, after which exact cleanup may destroy
the runtime volume.

Security and credential freshness gate only transitions that could newly publish
release. Once release exists, stale security must not abandon a running process:
wait, deadline enforcement, containment, terminal inspection, and typed
termination-receipt registration remain authorized. A recovered wait uses the
receipt's same-boot absolute deadline; it never receives a fresh timeout. Zero or
multiple matching resources, substituted mounts/labels/runtime/image, PID reuse,
an unproved wrapper, or an unexplained disappearance classify as `UNKNOWN`; they
never authorize recreation. A present but invalid release file fails loud as
corrupt, also without recreation. A typed terminal-failure or resource-loss
receipt, not a proof-free enum, is required before interruption.

Provider completion has three evidence-bearing outcomes. `RESULT_CAPTURED`
requires exit zero, no OOM, the original bounded result inode, and the exact
prepared result-parent authority. `PROVIDER_TERMINATED` carries a
content-addressed receipt whose disposition is `FAILED` for provider failures
or `INTERRUPTED` for supervisor containment. That receipt disposition is not the
deleted overloaded execution-event kind: the ledger event remains
`PROVIDER_TERMINATED`. Its reason names timeout, OOM, empty result, barrier exit,
security/credential containment, or positive pre-release loss. It embeds either
exact terminal-plus-empty-result evidence or a narrowly pre-release
resource-loss observation and includes the full workload release receipt when
release occurred. Any wrong/unstable/oversized result inode, released resource
disappearance, or mixed Docker inventory remains `UNKNOWN`; it is never
mislabeled as a provider failure. Event 6 stores the complete result or
termination evidence before cleanup.

Pre-commit cleanup is allowed only for the unique exact never-started occurrence
when no spawn commit exists. Terminal observation precedes result capture: the
observation never refers forward to a capture, while the capture may bind that
exact predecessor observation.

The lifecycle-owned policy, allocation, bounded-volume/sentinel/workspace, closed
projection, mount, inert-evidence, prepared-execution, activation-revalidation contracts,
durable allocation/prepared index, eight-event store embedding, and process-bound
preparation/activation capabilities and bounded runtime-volume contracts are
implemented. The shared Docker host authority now also pins its daemon root,
systemd cgroup driver, single-sourced static BusyBox helper, and configured
static `docker-init` source executable. Both host sources are descriptor-proven
as root-owned, singly linked, content-pinned ELF code with no dynamic loader or
dependency table, and their source evidence is bound into both main and keeper
projections. Runtime injection remains a post-start process-proof obligation.
The structural
raw-schema identity, strict action-image admission, exact bounded tmpfs-volume
request, and exact keeper/main create requests are implemented. The shared static
supervisor helper's running keeper bind target is then re-read through the
inspected keeper PID, bound to
that container's cgroup, and required to retain the issued source device, inode,
and digest. Shared descriptor-safe `/proc` primitives now parse the exact live
PID/state/parent/start generation, full byte-exact NUL-separated argv, unified
cgroup, process root, executable, and mount/PID namespace identities without
pathname fallback. The canonical resolved-workload graph now preserves the
complete validated Docker-inspection digest, exact init/wrapper generations and
argv, exact full-EOF mountinfo payload/length/digest and reparsed normalized
records, derived effective access, source-to-resolved inode joins, and logical
file/workspace observations; it rejects stacked or nested overlays and
intentionally carries the exact activation receipt rather than an unverified
event-ID string. The runtime assembly now consumes only the recovery coordinator's
active, one-shot `RUNNING_CONTINUABLE` capability, joins its token-sealed Docker
inspection to the actual typed durable event 5, retains one host `/proc` occurrence,
and owns the exact init, wrapper, namespace, executable, mount-root, and logical-file
descriptors in a process-and-thread-bound lease. Every reuse performs a closed
forward/reverse sandwich around the logical mount proof, admits only scheduler
`R`/`S` movement within the same PID/parent/start generation, and revalidates the
closed Docker semantics while treating changes limited to enumerated
nonauthoritative raw fields as new audit observations rather than new container
occurrences. The process-snapshot bound comes only from `LaunchSettings`; executable
authority uses its independent configured content digest and static-ELF proof.
The release authority contracts now join the complete resolved graph to the
actual typed durable event 5, bind the exact denylist observation and a
credential-validity interval that spans containment without exceeding broker
policy, and derive same-boot execution and containment deadlines from one
authorization instant. The independently configured canonical receipt bound
strictly contains the process-snapshot bound and leaves event-envelope space;
it is carried in the supervisor policy and reserved in both pre-mutation and
post-materialization tmpfs capacity proofs. Before allocation, the sealed
lifecycle adapter must repeat one deterministic conservative release-envelope
bound. Recovery joins it to the canonical policy/config cap before spawn and
requires the actual activation-event bound plus the complete process-snapshot
bound to fit strictly before delivery or event 5.
The committed continuation's release security, broker-validity authority, and
system clock live in a private coordinator-issued registry, never in
adapter-supplied publication arguments. Only the trusted leaf publisher may open
them, and only while the owner process/thread is inside the exact
`RUNNING_CONTINUABLE` callback. Publication also requires the exact registered
`RunActionBlockedWorkloadLease`; an ordinary or same-token reminted resolved graph
has no authority. The publisher serializes and fsyncs one exact receipt into a
registered anonymous inode. The final gate accepts only that frozen candidate,
derives its receipt, deadline, and inode internally, performs the second broker
validity observation, then queries the checkpointed denylist scope, contract,
subjects, and ancestor. An exact result samples coordinator-owned BOOTTIME and
invokes that candidate's fixed no-replace link; there is no caller-supplied
callback, clock, receipt label, path, or inode tuple. Denial, malformed output,
expiry, exception, callback return, or link failure burns both authorities. The
configured, policy-bound commit window begins before receipt construction, so
preparation latency consumes rather than extends the budget.

Before every event-5 continuation, the coordinator reopens the exact
keeper-mounted control directory and classifies it as empty, exactly
`control/release`, or corrupt. Empty permits the normal fresh authorization.
The sole canonical `0400` release file is full-EOF parsed, joined to the actual
typed event 5, descriptor/path revalidated, fsynced, and adopted without repeating
an already-irreversible security decision. Any other topology or byte/identity
state fails loud and cannot republish. Result event 6 embeds the content-addressed
adoption, including the complete receipt and linked parent/file identity, with
the terminal observation and capture receipt. Recovery at event 6 therefore no
longer depends on Docker or the runtime volume.
Volume, never-started keeper, running keeper, and never-started main inspections now
require complete nested raw schemas and normalize only enumerated daemon
identities and ordering; issued and observed projections are equal in repeated
Docker 29.1.3 runs with a
digest-pulled loopback OCI image. A durable preparation allocation binds the
claim to one unpredictable runtime-volume generation before Docker allocation.
The deterministic volume name remains claim-bound, while its exact label set
also carries the generation's sentinel content identity; keeper and main labels
remain claim-bound. The race-safe name/label resource manager therefore accepts
the complete preparation allocation and has a twice-stable name-only,
name-plus-label, and label-only inventory with inspect-by-ID container rebinding
and full-inspection volume occurrence digests. A stale volume from another
generation of the same claim is a substitution before sentinel publication.
After the keeper starts, the supervisor now opens one descriptor-bound
`/proc/<pid>` process generation and resolves its cgroup, root, mountinfo, and
stat records relative to that lease. Before the first volume mutation it requires
the exact issued tmpfs mount/device/options/owner/mode, stable block and inode
capacity accounting, a live non-zombie keeper generation, and zero root entries;
the real Docker lifecycle proves this boundary before creating any layout path.
That durable volume evidence now names the exact keeper evidence, container,
process, Linux start-time generation, policy-derived systemd cgroup path, root
mount, device, and inode used for the observation. The mounted-helper, keeper,
volume, prepared-execution, and activation-reobservation contracts all preserve
that process generation. The prepared layout names that exact
content-addressed volume evidence, so neither a recycled PID, a cgroup-parent
substitution, a valid observation from another keeper, nor a valid layout from
another physical root can be spliced into a prepared execution.
Workspace staging now has a descriptor-only copy prerequisite that inventories
the complete source and `.git` topology before mutation, includes physical Git
bytes and entries in capacity planning, sandwiches every copied inode against
source metadata, normalizes copied directories to private mode, and independently
reconciles both source and destination frontiers after the copy.
Runtime-volume preparation now holds that same keeper/process/root descriptor
lease from the empty proof through publication and final observation. It
pre-admits physical workspace and `.git` bytes/inodes, transient staging, every
future delivery/result/temporary reservation, and requires strict residual
headroom. Private directories, one empty result file, and the complete workspace
are built under an unpublished staging directory. Input and optional credential
directories are persisted as exact empty delivery slots: their final filenames
do not exist at preparation, so no authoritative writable payload inode can
contain a torn write. Activation writes a complete payload into an anonymous
`O_TMPFILE`, validates and fsyncs it, changes it to read-only mode, publishes it
exactly once with `linkat(AT_EMPTY_PATH)` and no replacement, fsyncs the slot
directory, and retains the exact linked descriptor through aggregate
construction. Immediately before the candidate can return, activation joins the
sole final pathname back to that retained mount/device/inode, metadata, and
complete bytes, then rechecks the retained descriptor before releasing it. A
crash before the link leaves the slot empty; a crash after the link leaves only
the complete final file. Unsupported filesystems and collisions fail loud; there
is no named staging or pathname-based fallback. A completed
read-only sentinel inode moves through a nonce-bound pending name and is
atomically published with `RENAME_NOREPLACE` only after every final directory is
in place and staging is gone; that rename is the final namespace mutation.
Preparation also creates a distinct empty `control` directory, reserves the
future release inode, and binds that directory into the prepared layout and
global subpath-identity graph. Projection protocol v4 mounts the exact control
subpath read-only at `/kapso-supervisor/control`, mounts the same pinned static
helper read-only and non-recursively into the main container, and replaces direct
target execution with the fixed positional-argument barrier. Main inspection
requires that exact helper bind, control mount, wrapper program, poll setting,
target command, and mount count. Its running observation also closes the exact
Docker lifecycle occurrence, including PID/start timestamp, zero restart count,
and safe runtime-normalized daemon fields, without claiming proc-generation or
resolved-mount authority. Activation proves the prepared control inode is still
empty both before and after payload delivery. Real-Docker validation starts the
wrapper, parses that running observation, waits two configured poll intervals,
and proves the target's first-write marker and a shell-metacharacter argument
marker remain absent while `control/release` is absent. It then publishes and
descriptor-adopts the exact receipt, proves the target runs with positional
arguments rather than shell interpolation, and observes exit without a restart.
The post-start boundary can now reopen that exact generation through the keeper
as a process-bound `RunActionControlDirectoryLease`: it retains the mounted root,
sentinel, and control descriptors; reproves their original physical identities;
reopens the keeper's current mount path on every use; and admits only the exact
empty or sole-release namespace. The raw descriptor is not public: release
publication must consume the process-bound lease through a guarded operation.
Read-only reopen is available only from a durable `PreparedExecution` and
reproves the exact root topology, child topology, file shapes, workspace/Git
frontier, sentinel inode/content, keeper process generation, mount, and stable
`statvfs` accounting. After an event-2 crash, allocation-bound recovery may create
only after proven total absence or may adopt only a completely reobserved exact
occurrence. Partial or ambiguous resources remain unresolved until positive
terminal cleanup exists; no path mints replacement allocation authority.
The result boundary no longer admits provider bytes alone. Result event 6 now
carries the release adoption, terminal main-container observation, and
descriptor-capture receipt
that binds the terminal fence, prepared result file, exact runtime generation,
fresh physical volume evidence, file metadata, size, and digest. `RESULT_RECEIVED`
embeds all three authorities before the existing atomic blob-to-event publication
and revalidates their complete prepared/spawn/container/volume/sentinel/file graph;
durable recovery therefore replays the captured bytes without contacting the
provider or discarding terminal provenance. Pure interpretation then publishes
the accepted bytes and `RESULT_DECIDED` atomically; terminal acceptance carries
no duplicate result blob. Concrete terminal Docker inspection, including exact
Docker 29.1.3 exited-state normalization, is now implemented. The trusted
terminal leaf retains and revalidates the adopted release inode, same host boot,
exact three-resource inventory, and two complete typed terminal snapshots. Its
semantic digest normalizes only the three explicitly order-nonauthoritative
Docker lists, and its capability-bound path must reproduce the read-only
inspection token exactly once. The continuation capability retains that exact
trusted terminal: a terminal adapter cannot return even `PENDING` without
completing the leaf, and `RESULT_CAPTURED` must carry the retained observation
unchanged. Exit zero, nonzero, and OOM remain typed terminal facts; only the
adopted-release result join admits zero/no-OOM capture from the same released
container and start timestamp. Descriptor-bound result capture is now
implemented as a second private trusted leaf. It retains the adopted release,
same host boot, exact Docker inventory, and terminal occurrence around a
keeper-root descriptor read; reopens the original prepared result inode without
following links or blocking on special files; and sandwiches its parent,
sentinel, keeper generation, root topology, and `statvfs` evidence. The
configured result bound is checked against the policy and the complete payload
is read through EOF. A zero-byte original inode now remains exact descriptor
evidence for `EMPTY_RESULT`, while `RunActionProviderResult` and durable result
authority remain strictly nonempty. Capture authority can be taken only after
the trusted terminal is observed and no later than the original release
execution deadline. Once authorized, bounded descriptor I/O may cross that
deadline; host I/O latency cannot change provider success. The capability
registers the exact `RunActionProviderResult`: a fabricated result, a substituted
equal-shape capture, or returning `PENDING` after consuming capture authority is
rejected.

Typed provider termination now has one closed immutable evidence graph. Timeout
requires a fresh running-container observation sampled after the release-derived
deadline and a descriptor-read, no-replace `control/timeout` publication;
publication outranks every later exit fact. Without that authority, OOM,
nonzero exit, and descriptor-proved empty result are mutually exclusive failed
outcomes. The only pre-release interruption is a stable, same-boot proof that
the exact volume and keeper remain, the main alone is absent, and release is
absent under the exact control authority. The contracts bind the activation
event, release adoption, terminal occurrence, immutable runtime-volume
occurrence, deadlines, and publication inode. Recovery reopens the exact
`EMPTY`, `RELEASED`, or `TIMED_OUT` control topology. A timed-out query carries
the reconstructed publication receipt itself, joined to event 5 and its adopted
release; it cannot collapse to an ordinary released query. The store persists a
complete receipt only as terminal event 6 after a pure join to durable allocation
and activation. That event publishes no result blobs, cannot be extended, reopens
across either publication crash side, and replays through recovery without
invoking an adapter or interpreter.

Recovery now has a sealed registration boundary for that receipt. A dedicated
pre-release-main-loss observation state cannot be confused with inert, running,
terminal, or unknown Docker state. One process/thread-bound continuation
capability admits exactly `PENDING`, its privately captured result, or its
privately registered termination receipt. Result capture and termination
registration consume each other's authority, and a returned but unregistered or
cross-occurrence receipt is rejected. Released termination must reproduce the
retained trusted terminal and release adoption; timeout additionally must
reproduce the exact adopted timeout publication; pre-release loss must reproduce
the exact loss-observation content ID. Before terminal publication the
coordinator reproves the unchanged host workspace and reopens the reason-specific
topology—`EMPTY` for pre-release loss, `TIMED_OUT` for timeout, and `RELEASED`
for every other released failure. It retains that descriptor lease across the
event append and checks it immediately before and after. Successful result
publication similarly retains `RELEASED` across its append, so a concurrent
timeout cannot become success. A crash before either append leaves event 5 and
requires fresh physical evidence; a crash after event 6 replays without adapter
or interpreter use.

The reader and recovery seam deliberately do not publish a missing timeout.
They adopt only a canonical, owner/mode/link/mount/device/inode-exact
`control/timeout` file under a retained release/control sandwich; malformed or
spliced visible state fails loud. The normal physical producer remains later:
timeout publication, Docker containment, terminal-failure/empty-result evidence,
positive pre-release-loss inspection, cleanup, and production adapters are not
yet wired. Ordinary adapters therefore remain pending rather than fabricating
termination. No production caller can receive Docker start authority from the
reservation gate; M9 activation must continue to route every post-reservation
transition through the coordinator's sealed capabilities.

The timeout payload envelope is now single-sourced before physical publication
lands. Launch configuration and the durable supervisor policy carry the same
2 MiB directive bound; configuration requires the process-snapshot bound to fit
strictly inside it and the release, snapshot, and timeout bounds together to fit
strictly inside one terminal event. Recovery, release adoption, and timeout
adoption reject a policy/config mismatch before provider mutation. Preparation,
activation revalidation, and runtime-volume planning reserve the allocated
timeout bytes and a second immutable control-file inode in addition to release
headroom. The real Docker lifecycle continues to pass with that stricter capacity
accounting.

The recovery surface is now fail-closed ahead of typed termination. The
proof-only `PROVEN_RESOURCE_LOST`, `QUIESCENT_RECHECKABLE`, and
`PROVEN_QUIESCENT_WITHOUT_RESULT` states are deleted, as are the overloaded
provider `INTERRUPTED` reasons and store mutation. Cancellation is expressed by
its event kind, and a distinct `FRONTIER_INVALIDATED` event can close only an
allocated or prepared pre-spawn prefix after its unchanged workspace is
re-proved. A workspace mismatch remains unresolved. After durable event 5, the
continuation admits only `PENDING`, a privately registered descriptor-captured
result, or a privately registered typed termination. Because no normal adapter
yet owns the physical termination leaf, current production behavior still
selects only the first two branches. A complete pre-existing event-6 termination
receipt is terminal and replayable; ambiguity cannot become terminal.

The coordinator owns one process-bound, non-clonable implementation catalog fixed
at composition; `recover()` accepts no caller-selected implementation. Each
catalog entry exact-object binds one execution adapter and one result interpreter
to the two identities in its durable boundary. Execution adapters own only
prepare, stage, read-only inspection, and token-sealed continuation. Result
interpreters receive only the complete
request and raw-result bytes: no workspace binding, descriptor, provider object,
or execution method. A `RESULT_RECEIVED` tail resolves and invokes only the
interpreter. Interpretation is repeated to detect nondeterminism, while the
coordinator alone reconciles the unchanged host workspace before and after the
durable decision. A `RESULT_DECIDED` tail invokes neither adapter nor interpreter;
it revalidates the workspace and appends only terminal acceptance. A crash after
spawn commit therefore reopens only as committed work; a crash after raw-result
persistence re-runs only local interpretation; a crash after the decision reruns
only workspace completion. Successful editing results remain at
`RESULT_RECEIVED` until the isolated staging/promotion protocol lands.

Publication takes the locks in checkpoint → workspace → registry order and
retains them through bundle/checkpoint/view commit. The candidate `ACTION_LEDGER`
must equal the live store exactly, every new prefix must be terminal and bind the
current frontier, and old terminal prefixes are immutable. Zero workspace edits
requires unchanged branch evidence. The current eight-event foundation admits
only workspace-free actions, unchanged read-only actions, and failed unchanged
editing actions. A successful editing result cannot mutate or advance the live
host; it remains nonterminal until the isolated promotion slice supplies exact
write-ahead authority. Missing post-crash workspace identity stays blocked rather
than being guessed. Read-only
and otherwise unchanged terminals still form one exact full workspace-identity
chain, including source and admitted Git closure digests, whose final identity
must equal the live workspace. Resume can now classify and reconcile each
nonterminal prefix without blindly reinvoking a committed provider. Production
adapter/supervisor implementations and their OS isolation remain required before
this path is activated.

## Failure and trust behavior

- Missing/corrupt/unauthorized/incompatible/expired artifacts fail before spend.
- Network failure during fresh resolution fails; no local unpinned substitute.
- A verified local cache may support normal offline scientific work only after one
  exact launch is resolved and pinned.
- Resume still requires configured fresh security-denylist state; performance-only
  state may use the immutable offline pin under policy.
- The bootstrap/pin floor detects deletion or substitution relative to the run but
  never authorizes offline; every dangerous boundary still makes a live request.
- GitHub bytes are untrusted until publisher, attestation, artifact identity, and
  digest verification completes.

## Tests

- Resolve PostTrainBench and RelBench bindings through `ml_ai` to the same
  expert/knowledge/security repository triple while retaining distinct
  family/adapter identities.
- Reject unknown scope, unknown family/adapter, repository overrides, a repository
  triple whose publications name another scope, and an expert/snapshot scope mismatch.
- Resolve compatible expert/snapshot/adapter tuples and reject every incompatible
  dimension independently.
- Reject torn pairs, substituted manifests, cross-task launch reuse, stale release,
  forged publisher, and denylisted component.
- Bootstrap explicit E0/EMPTY; prove missing remote does not trigger it.
- Inject death after every download/stage/rename/pin/checkpoint boundary.
- Prove no coding-agent/embedding/evaluator call occurs before `BootstrapPin`.
- Reject stopped/completed action frontiers, duplicate reservations, request
  changes, invalid boundary/capability combinations, and dirty or substituted
  workspace state at reservation and recovery.
- Inject death at reserved, allocated, prepared, spawn-committed,
  activation-committed, raw-result, and result-decided prefixes; prove allocation
  creation, allocation reopen, and prepared revalidation are distinct, committed
  work is never freshly replayed, an exact result decision accepts without provider
  or interpreter access, ambiguous provider state remains unresolved, implementation
  catalogs and single-use capabilities reject clone/fork/reuse, security movement
  before allocation cancels, security movement after allocation remains
  cleanup-blocked, and workspace mutation during local interpretation never becomes
  a terminal event.
- Prove embeddings receive no workspace capability; prove edits exclude parallel
  edits/readers across processes, poison stale reservations/publication candidates, and
  become usable only after an exact branch-advance checkpoint successor.
- Reject replace refs, alternates, shallow/graft state, packed or unreachable Git
  objects, missing reachable objects, index behavior flags, malformed commit
  headers, admitted-metadata changes, and workspace/Git entry-limit exhaustion.
- Resume after remote pointers advance and require original local pin.
- Hold one runtime in one process and prove same-process and cross-process reopen
  fail nonblocking until close or process death; prove a forked child cannot use
  the copied authority or release the parent's lease.
- Publish a durable frontier, close the original runtime, reopen solely from the
  local pin, and reconcile that exact checkpoint, generation, journal, and views.
- Corrupt each local component/receipt/tree and require fail-loud resume.
- Exercise performance and security revocation differences.
- Prove crash-before-link, crash-after-link adoption, concurrent no-replace
  publication, unsupported anonymous-file/link syscalls, parent/inode
  substitution, exact delivery `statfs` deltas, and retained result/temporary
  headroom.
- Inject death before wrapper start, after an ambiguous start response, while
  barrier-blocked, after resolved-mount proof, immediately after the release
  link, while running, during deadline stop/kill, after terminal inspection, and
  during result capture. Every mutation requires a fresh inspection.
- Substitute every Docker `VolumeSubpath` after event 5 but before start and
  require the post-start namespace proof to leave release absent. Prove stale
  security/credentials block release but never block containment of an already
  released workload.
- In real Docker, create the inert container before delivery, activate a complete
  request plus credential and copied Git workspace, then start only after the
  all-subpath observation closes. Commit the exact activation receipt as durable
  event 5, have the recovery coordinator issue its one-shot continuation
  capability, resolve and revalidate the blocked workload twice, and prove the
  release, target command, and result remain absent while the ledger stays at
  five events.
- After the released target exits, reopen event 5, descriptor-adopt the same
  release, parse the exact exited Docker occurrence twice, consume one sealed
  terminal reinspection capability, descriptor-capture the original bounded
  result inode, and prove the exact result graph is interpreted and durably
  accepted through event 8 without cleanup.
- Admit only `()`, `(release)`, and `(release, timeout)` as real control-directory
  topologies; reject timeout-only/extra entries and every retained topology
  mutation. Reject malformed, over-bound, substituted, and cross-occurrence
  timeout bytes.
- Reopen an event-5 run with an existing timeout, reconstruct its exact
  publication receipt without republishing, carry it through the committed query
  and sealed terminal capability, persist timeout event 6 under a retained
  `TIMED_OUT` fence, and replay without implementation access.
- Verify expert repo is writable only inside the run workspace and snapshot/adapter
  roots remain read-only.
- Verify old `initial_repo` and checkpoint paths are absent after activation.

## Definition of done

- Every run begins from one verified atomic launch identity.
- Every fresh run reaches GitHub only through the scope registry mapping; task
  inputs never carry repository coordinates.
- Fresh startup and resume expose no partial/torn component combination.
- The current run remains reproducible when GitHub pointers advance.
- Security/contamination freshness is enforced before dangerous operations.
- The old starter/`initial_repo` path and old persisted shapes no longer exist.

## Non-goals

- Building expert or knowledge releases.
- Capturing/publishing run evidence.
- Task-specific adapter business logic.
- Providing a fallback non-cross-run startup mode.

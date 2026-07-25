# M9 — launch resolution, workspace bootstrap, and resume

Parent plan: [`00-orchestrator-plan.md`](00-orchestrator-plan.md)

Depends on: M2, M5, and M8.

Status: **in progress**. The transactional launch resolver and exact authority
contracts, atomic workspace/bootstrap-pin installation, exact run-checkpoint
contracts, the protected checkpoint CAS store, the immutable derived-generation
contract/layout, and the dependency-pure reconciled archive/history/journal
projection and retained bundle layer, single-lock checkpoint/generation
publisher, mutable-view promotion, current-frontier action lease, receipt-pinned
create-only action store, mandatory action-ledger projection, and durable
gate/publication composition, descriptor-safe runtime reopen, and deterministic
nonterminal action recovery are implemented. The pinned Docker CLI, daemon, and
image authority is now a domain-neutral runtime reusable by M8 evaluators and
M9 action adapters, with shared host/runtime authority owned once by top-level
`cross_run.docker` rather than by either consumer. Boundary-specific production
adapters and the mechanically constrained supervisor remain, while the
content-addressed Docker execution-policy, deterministic preparation-claim,
inert-evidence, prepared-execution contracts, and durable six-event action prefix
are implemented. OS executor activation, explicit E0/S-EMPTY provisioning
orchestration, full policy refresh on resume, and API/runner activation remain.

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

The implemented checkpoint foundation is deliberately dormant until the execution
permit and transactional view-promotion work below lands. One exact
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
2. binds a one-shot permit to the observed checkpoint, journal head and byte
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
      and issue one-shot action permits from complete request bytes. Permit
      consumption holds a shared checkpoint lock for the entire external action,
      reopens the pinned workspace by descriptor, reconciles the source tree,
      Git branch, commit tree, and index to checkpoint evidence, and authenticates
      the current denylist as the exact checkpointed observation; state
      publication retains the exclusive lock.
- [x] Make the action store authoritative across processes: compare-and-swap each
      reservation against the exact predecessor ledger, persist intent/claim/
      prepared/spawn/result/acceptance events, hold the receipt-pinned workspace
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

`RunFrontierActionGate` is the only new-run authority for coding-agent, embedding,
and evaluator calls. It derives an immutable intent from the complete canonical
request bytes, binds it to the exact checkpoint/safety/generation/journal/bundle/
view closure, and burns the permit before revalidation. The same operation or
content-derived intent cannot be issued twice. Only active, non-yielded
checkpoints act, and the capability matrix is exact: ideation/evaluation coding
agents and evaluators read the workspace, implementation coding agents edit it,
and embeddings receive no workspace descriptor.

Consumption holds a shared descriptor-safe frontier lock through boundary
completion, while checkpoint publication requires the exclusive lock. Before
use, the gate proves that the owner-private workspace has the checkpointed branch
head, a clean source tree equal to the commit tree, an exact flag-free Git index,
one exact configured ref, no replace/alternate/shallow/graft state, and a bounded
self-contained loose object store equal to the complete reachable
commit/parent/tree/blob closure. The returned identity includes the canonical
digest of every admitted Git metadata file, so read-only actions must leave the
full source and Git frontier unchanged.

An edit is exclusive across processes against all workspace readers and other
edits. A successful result must finish as one clean direct-successor commit with
canonical Git header grammar; a failed result may terminate only with the exact
unchanged predecessor workspace. A successful edit spends the predecessor
frontier, so no later action or publication may use it until a checkpoint
successor records exactly one authorized `RunBranchAdvance` to that commit.
Reconstructed gates and publishers derive this state from durable event prefixes
rather than process-local memory.

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

The gate now persists `INTENT_RESERVED` before returning a permit,
`SPAWN_COMMITTED` before exposing provider/workspace authority, complete raw
results before interpretation, and complete accepted results with the exact
post-action workspace. Normal context exit requires a terminal durable prefix;
exceptional exit deliberately leaves ambiguous spawn or received-result state
for resume. Admission is a ledger compare-and-swap, so reconstructed gates and
concurrent processes cannot both reserve against the same live floor.
Mutation entry points are internal and sealed to the gate. Provider execution
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

Recovery is an explicit six-event state machine:

```text
INTENT_RESERVED
  → PREPARATION_CLAIMED
  → EXECUTION_PREPARED
  → SPAWN_COMMITTED
  → RESULT_RECEIVED
  → RESULT_ACCEPTED
```

`INTENT_RESERVED` may cancel before allocation when its frontier is stale.
Claimed or prepared work may terminally interrupt before spawn when its frontier
is invalidated or the supervisor positively proves its resource was lost. This
terminal records the reservation's exact unchanged workspace binding (or no
workspace for a workspace-free action) because no request, credential, or start
authority existed. It is durable before cleanup; cleanup is idempotent,
nonblocking garbage collection, and an inert orphan has no spawn authority.
An unknown or temporarily unreachable resource remains unresolved. A persisted
prepared occurrence is never replaced. `SPAWN_COMMITTED` may terminate only as
a result or provider interruption.
Every event repeats the exact reservation and predecessor, while claim, prepared,
spawn, result, and acceptance payloads bind their immediate durable authority.
The store rejects skipped/reordered phases, old intent-to-spawn records, identity
splices, multiple nonterminal operations, and global reuse of claim, prepared,
container, slot, quota, filesystem, provider, or invocation identities.

Before allocating, recovery durably appends the deterministic claim. A
same-process/same-thread preparation capability distinguishes first allocation,
claim reopen, and exact prepared-occurrence revalidation; it carries the live
workspace descriptor and daemon-visible source path when the policy requires a
workspace. Before allocation, the exact lifecycle adapter must twice produce the
same conservative bound for the complete prepared-event encoding; the coordinator
rejects a nonpositive, nondeterministic, or over-limit envelope. The returned
prepared event must then fit that declared bound. Prepared evidence is persisted
before spawn. Security and workspace are checked again immediately before the
durable spawn commit, and the reservation boundary must still equal the
checkpoint safety boundary. Only after that commit does a separate single-use
activation capability expose the complete request and a capability-owned
duplicate workspace descriptor for exactly one execution-adapter invocation.
Both capabilities burn on success or exception and close their descriptors.
Request bytes are unreadable from the action session before spawn.
Preparation returns one typed state: exact prepared with an origin compatible
with its allocation/reopen/revalidation mode, positively lost, or unknown.
Claim reopen may allocate only after positive exact absence. Prepared
revalidation can return only the identical occurrence, positive loss, or
uncertainty; it cannot allocate. Early interruption is admitted after the claim
or prepared event without changing the normal six-event success chain.

A committed spawn receives only its durable prepared execution and spawn
identity: `RESULT_AVAILABLE` records the exact bytes,
`RUNNING_REATTACHABLE` may reattach only under the unchanged security
observation, and an exact `INERT_ACTIVATABLE` observation rebuilds activation
authority for the same prepared execution and spawn after rechecking workspace
then security. Proven quiescence interrupts, and `UNKNOWN` remains unresolved.
Recovery never routes committed work through claim or preparation and never
mints a second activation fence.

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
SpawnCommit → single-use ActivationGrant
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
Input/result/credential files are empty, private, singly linked regular files
before spawn. The workspace is copied into the same bounded generation and its
observed tree digest, Git-closure digest, entry count, and byte count must equal
the durable frontier binding.

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
`PreparationClaim → PreparedExecution` occurrence before `SpawnCommit`.
Reservation admission accounts for the complete remaining lifecycle: at most six
event files and three content blobs per operation, plus the configured crash
staging allowance and two fixed lock files. Every append and reopen rechecks its
remaining event/blob headroom, so an accepted intent cannot strand an
irreversible spawn solely because the store later reaches its configured byte or
entry bound. This capacity proof is separate from the lifecycle adapter's
pre-allocation serialization envelope: the former reserves store space, while the
latter proves that the complete concrete prepared record can occupy one event
file before any Docker or slot resource is created. Before `SpawnCommit`, the
production supervisor may only create or reopen that exact inert resource;
request bytes and credential leases remain absent. A post-commit, single-use
`ActivationGrant` will bind the whole
`PreparedExecution`, spawn commit, and either exact credential-lease receipts or
a no-credentials proof. Only the supervisor may consume it to attach the
admitted broker network, populate the prepared delivery files, and start the
same container once. Immediately before start, it must issue a new
`ActivationRevalidationReceipt` after re-inspecting the volume, physical
generation sentinel, running keeper, copied workspace, delivered
input/result/credential files, and still-never-started main container. Immutable
volume facts must equal preparation; allocated usage and actual available
blocks/bytes/inodes must form the exact `statfs` capacity relationship, and the
fresh observation must retain positive result-plus-temporary headroom. Workspace
and sentinel observations are distinct activation-time contracts bound to the
exact spawn commit and their preparation evidence IDs; replaying a preparation
object is not revalidation.
The serialized receipt is evidence only: a process-bound single-use lease keeps
the live resource authority and workspace lock and is the sole authority that
may synchronously start.
It embeds the exact typed spawn commit and delivery predecessors. Request delivery
proves the fixed regular-file name, digest, size, owner/group, read-only mode, and
single link; credential delivery proves the same structural facts plus its opaque
broker lease authority and size, but stores no credential digest or bytes.
Every delivery/proof record binds the exact spawn-commit content ID, including its
invocation nonce; a semantically similar second fence cannot reuse prior delivery.
Crash or lease loss requires a complete new revalidation. Zero or multiple matching resources,
missing positive state evidence, substituted mounts/labels/runtime/image, or an
unexplained exit classify as `UNKNOWN`; they never authorize recreation.
Pre-commit cleanup is allowed only for the unique exact never-started occurrence
when no spawn commit exists. Terminal observation precedes result capture: the
observation never refers forward to a capture, while the capture may bind that
exact predecessor observation.

The lifecycle-owned policy, claim, bounded-volume/sentinel/workspace, closed
projection, mount, inert-evidence, prepared-execution, activation-revalidation contracts,
durable claim/prepared index, six-event store embedding, and process-bound
preparation/activation capabilities and bounded runtime-volume contracts are
implemented. The shared Docker host authority now also pins its daemon root,
systemd cgroup driver, and single-sourced static BusyBox helper. The structural
raw-schema identity, strict action-image admission, exact bounded tmpfs-volume
request, and exact keeper/main create requests are implemented. The static
keeper helper is descriptor-proven as root-owned, singly linked, content-pinned
ELF code with no dynamic loader or dependency table. Its running bind target is
then re-read through the inspected keeper PID, bound to that container's cgroup,
and required to retain the issued source device, inode, and digest. Volume,
never-started main, and running keeper inspections now require complete nested
raw schemas and normalize only enumerated daemon identities and ordering; issued
and observed projections are equal in repeated Docker 29.1.3 runs with a
digest-pulled loopback OCI image. The race-safe name/label resource manager,
now has a twice-stable name-only, name-plus-label, and label-only inventory with
inspect-by-ID container rebinding and full-inspection volume occurrence digests.
Allocation/reopen reconciliation, concrete activation/result receipts, positive
cleanup authority, and production adapters remain the next slices.

The coordinator owns one process-bound, non-clonable implementation catalog fixed
at composition; `recover()` accepts no caller-selected implementation. Each
catalog entry exact-object binds one execution adapter and one result interpreter
to the two identities in its durable boundary. Execution adapters own only
prepare/start/inspect/reattach. Result interpreters receive only the complete
request and raw-result bytes: no workspace binding, descriptor, provider object,
or execution method. A `RESULT_RECEIVED` tail resolves and invokes only the
interpreter. Interpretation is repeated to detect nondeterminism, while the
coordinator alone reconciles the workspace before, during, and after durable
acceptance. A crash after spawn commit therefore reopens only as committed work;
a crash after raw-result persistence re-runs only local interpretation.

Publication takes the locks in checkpoint → workspace → registry order and
retains them through bundle/checkpoint/view commit. The candidate `ACTION_LEDGER`
must equal the live store exactly, every new prefix must be terminal and bind the
current frontier, and old terminal prefixes are immutable. Zero workspace edits
requires unchanged branch evidence. One successfully accepted or concretely
interrupted workspace-changing edit requires the live workspace and exactly one
authorized branch advance to match its durable before/after identities. Missing
post-crash workspace identity stays blocked rather than being guessed. Read-only
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
- Reject stopped/completed action frontiers, duplicate operations, cloned permits,
  request changes, invalid boundary/capability combinations, and workspace
  mutation between issuance and consumption.
- Inject death at reserved, claimed, prepared, spawn-committed, and raw-result
  prefixes; prove allocation, claim reopen, and prepared revalidation are
  distinct, committed work is never freshly replayed, ambiguous provider state
  remains unresolved, implementation catalogs and single-use capabilities reject
  clone/fork/reuse, security movement before allocation cancels, security movement
  after allocation remains cleanup-blocked, and workspace mutation during local
  interpretation never becomes a terminal event.
- Prove embeddings receive no workspace capability; prove edits exclude parallel
  edits/readers across processes, poison old permits/publication candidates, and
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

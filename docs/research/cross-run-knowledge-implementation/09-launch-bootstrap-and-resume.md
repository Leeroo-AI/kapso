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
inert-evidence, and prepared-execution contracts are implemented. OS executor
activation, explicit E0/S-EMPTY provisioning orchestration, full policy refresh on
resume, and API/runner activation remain.

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
      reservation against the exact predecessor ledger, persist spawn/result/
      acceptance events, hold the receipt-pinned workspace lock, require terminal
      live state for publication, and derive branch accounting from durable events.
- [x] Recover one exact final nonterminal prefix without replaying a committed
      spawn; bind recovery to the complete frontier and an issued exact
      implementation catalog, burn fresh-spawn authority once, and replay terminal
      accepted bytes without execution or interpretation access.
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

Recovery is an explicit state machine. An unspawned reservation may allocate
locally, but it receives neither request bytes nor workspace authority; security
and workspace are checked again immediately before the durable spawn commit, and
the reservation boundary must still equal the checkpoint safety boundary. Only
then does a one-shot, same-process/same-thread capability expose the complete
request and a capability-owned duplicate workspace descriptor for exactly one
execution-adapter invocation. The capability burns on success or exception and
closes that descriptor. A committed spawn receives only its durable execution
identity: `RESULT_AVAILABLE` records the exact bytes, `RUNNING_REATTACHABLE` may
reattach only under the unchanged security observation, proven quiescence
interrupts, and `UNKNOWN` remains unresolved. Recovery never routes a committed
spawn through fresh preparation or start.

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
content-addressed image authority, pinned runtime-settings digest, closed raw-field
schema/projection versions, value-constrained non-secret static environment,
UID/GID, filesystem and activation-network policies, non-secret credential
policy, sandbox profiles and controls, all Docker-visible resource controls, and
separate supervisor-only time/result bounds. No argv or dynamic string argument
is durable. The adapter renders a fixed lifecycle-owned command in memory from
the template and admitted in-container paths; requests travel only through the
input slot and credentials only through the post-commit credential slot.
`PreparationClaim` is deterministic before Docker allocation and embeds the
complete validated reservation and exact policy. Independently valid IDs cannot
be spliced across reservations, policies, or same-kind lifecycle implementations.

`PreparedExecution` is deliberately an occurrence receipt, not a reusable
semantic identity. It embeds the claim, empty supervisor-owned input/result/
credential slot receipts, and exact Docker inspection evidence for one named
container. Each slot receipt includes an ordered `openat2`-beneath/no-symlink
root-to-leaf walk with mount, device, inode and inode-generation identities;
private ownership/mode and ACL/link facts; and an enabled, enforced, exclusive
logical-quota observation bound to the exact claim, slot kind, leaf identity,
mount/device, filesystem authority, and deterministic scope. Its byte limit
equals the admitted payload bound and its entry limit is exactly one; scopes are
pairwise distinct across slots.
All slot walks share one private claim-root prefix and distinct leaves. The
workspace has a separately rooted walk tied back to the durable workspace
device/inode binding.
The name and complete label set derive only from `PreparationClaim`, avoiding a
back-edge from Docker state to `PreparedExecution`. The evidence requires
`created`, PID/restart count zero, zero start/finish timestamps, restart `no`,
auto-remove false, network `none`, no healthcheck, no volume-plugin mount, no
Docker-socket mount, and an exact bind-mount bijection to the workspace and
prepared slots. Full descriptor-walk ancestry proves that no bind source equals,
contains, or is contained by another; inode generation prevents same-inode reuse
from passing a reopen, every bind has type `bind` and recursively-private
propagation, read-only mounts are recursively read-only, and nested mounts are
absent. This
prevents a writable result mount from aliasing or enclosing a read-only workspace,
input, or credential directory. Raw create/inspect objects are validated in
memory and are not persisted. One versioned, content-addressed, normalized
create/inspect projection must match exactly in both directions. Its pinned
raw-field schema classifies every raw path as projected, required-literal,
runtime-evidence, or explicitly nonauthoritative; an unknown/unclassified path
fails. The projection covers the policy, template ID, static environment,
UID/GID, hostname, working directory, CPU/NanoCPU/realtime,
memory/reservation/swap/OOM/PID/block-I/O/ulimit/shm/tmpfs controls and flags,
namespaces, devices, groups, cgroup parent, capabilities, sysctls, security
profiles/options, protected system paths, runtime, logging, init, mount policy,
ports/network adjuncts, volumes/storage adjuncts, process flags, and privilege
policy. Unsafe adjuncts are explicit empty/false fields, never omissions.
Supervisor time and output/result byte limits do not pretend to be Docker
inspection facts.
Credential records contain policy and slot authority only—never secret bytes or
host credential paths.

The supervisor store will atomically enforce one
`PreparationClaim → PreparedExecution` occurrence. Before `SpawnCommit`, it may
only create or reopen that exact inert resource; request bytes and credential
leases remain absent. A post-commit, single-use `ActivationGrant` will bind the
whole `PreparedExecution`, spawn commit, and either exact credential-lease
receipts or a no-credentials proof. Only the supervisor may consume it to attach
the admitted broker network, populate the already-mounted private delivery
slots, and start the same container once. Immediately before start, it must issue
a new `ActivationRevalidationReceipt` after rewalking every slot/workspace,
rechecking quota enforcement and delivered input/result/credential shapes, and
re-inspecting the still-never-started container. The serialized receipt is
evidence only: a process-bound single-use lease keeps the reopened descriptors and
workspace lock live and is the sole authority that may synchronously start.
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

The lifecycle-owned policy, claim, prepared-slot/walk/quota, closed projection,
mount, inert-evidence, prepared-execution, and activation-revalidation contracts
are implemented. The durable claim index,
mechanically constrained Docker supervisor, activation/result records, action
store embedding, and production adapters remain the next slices.

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
- Inject death at reserved, spawn-committed, and raw-result prefixes; prove
  committed work is never freshly replayed, ambiguous provider state remains
  unresolved, implementation catalogs and fresh capabilities reject
  clone/fork/reuse, security movement before commit cancels, and workspace
  mutation during local interpretation never becomes a terminal event.
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

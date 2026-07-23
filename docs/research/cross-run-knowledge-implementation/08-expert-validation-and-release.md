# M8 — expert validation, composition, release, and revocation

Parent plan: [`00-orchestrator-plan.md`](00-orchestrator-plan.md)

Depends on: M2, M3, and M7. M8 consumes a shared verified task-adapter provider
that M9 also uses; M9 does not own the provider implementation.

## Objective

Certify or reject expert candidates through an ordered evaluator cascade, compose
only compatible approved changes, and publish one immutable history-free expert
release. No proposer, score, or single task can grant promotion authority.

## Owned responsibilities

- Candidate eligibility and evaluator-cascade state machine.
- Static/security/sanitation/identity/dependency/license gates.
- Source replay, synthetic fresh-task, development-anchor, cross-family, sealed
  canary, cost, and release-wide regression evidence.
- Independent automated reviewer assertions and Pareto-aware promotion decision.
- Candidate rebase/composition against current stable release.
- Final source/map/contracts/book assembly and immutable GitHub publication.
- Performance/security/contamination revocation behavior.

## Proposed code surface

```text
src/kapso/cross_run/expert/
  validation.py
  validation_store.py
  composition.py
  release.py
  publisher.py

src/kapso/cross_run/
  source_archives.py
  task_adapters.py
  task_adapter_store.py

src/kapso/cross_run/launch/
  revocation.py

tests/
  test_expert_validation.py
  test_expert_evaluator_cascade.py
  test_expert_composition.py
  test_expert_release_publisher.py
  test_expert_revocation.py
```

Target expert repositories may also receive generated workflow definitions for
validation and release publication. The autonomous agent may change them directly;
their exact content is included in the candidate identity and must pass the same
automated evaluator cascade before activation.

The validation substrate is separate from catalog admission. It uses
`ExpertValidationTrack`, `ExpertCandidateEligibilityDecision`,
`ExpertValidationAttempt`, `ExpertEvaluatorRun`,
`ExpertEvaluatorAttestation`, its rotatable signature envelope, and
`ExpertCandidateValidationState`. Sealed-canary runs persist only a strict
aggregate result; hidden case details cannot enter the validation or reviewer
closure. The exact,
typed validation policy excludes the local state-store path from scientific
identity while the runtime configuration fingerprint retains it for audit.

The validation track is derived from trusted M7 records, never accepted from a
caller or proposer: repository changes are architecture, the exact
`mechanically_general_fix` trigger is mechanical, and other capability changes
are behavioral. Task-specific, confounded/noisy, and unsafe/specialized are
review dispositions, not proposer-selected tracks.

Evaluator execution requires two injected, read-only verified providers:

- a bundle reader that resolves the complete sanitized M3 `RunBundle` artifact
  closure needed for faithful replay; and
- a task-adapter reader that resolves a `TaskAdapterManifest`, exact source tree,
  and verification receipt.

Episode summaries and manifest hashes are not substitutes for either byte
closure. Missing bytes fail loud and make the attempt ineligible.

Source-run replay selection is itself content-addressed. Trigger evidence is
traversed only through typed episode, observation, contradiction-claim, and
bundle edges. Evidence attached to changed module contracts may add explicit
coverage cases, but snapshot membership, parent-episode links, and "all packet
episodes" are never implicit selection authority. Every selected episode records
whether it is causal or coverage evidence and why it was included. A replay-required
candidate with no such episode is ineligible; configured episode and bundle limits
fail the whole selection rather than clipping it.

Selection is not a caller-facing choice. Enrollment derives it from the exact
store-validated candidate and the typed validation settings. Its identity binds
the candidate tree and commit, trigger packet and deterministic decision,
KnowledgeSnapshot, validation policy, configuration fingerprint, selected
episodes, reasons, and source bundles. Eligibility and the validation attempt
embed that exact selection, and reducer replay reopens the validated candidate
store and re-derives it. A source base that is not current, or an unavailable validation facility, takes
precedence and produces no selection. Generic evaluator-result construction and
reduction reject source replay until the typed executor can prove exact
episode-and-adapter coverage for every selected case.

Captured `ArtifactEnvironment` records name the exact scientific adapter
manifest and verification receipt, plus the exact content ID for every starting
artifact. An opaque adapter hash or logical artifact reference is not replay
authority. Enrollment keeps current adapter pins for fresh validation separate
from historical adapter-package pins grouped over the selected source episodes;
multiple versions of one logical adapter may coexist. Missing historical bytes
fail before an attempt can become eligible.

Once the source stage is current, deterministic preflight reopens the candidate
and validation state, observes CURRENT, resolves concrete verified bundle lineages,
materializes the exact source-base tree, copies the candidate
tree into an immutable byte closure, and materializes every captured starting
artifact. A lineage provider must retain every root-to-tip `RunBundle` byte
closure; preflight ignores its claimed projection and deterministically rebuilds
every generation from those bytes. Generation-zero, adjacent supersession, and
the reprojected tip are checked again when the runtime-only prepared object is
constructed.

One monotonic deadline and one aggregate entry/byte budget govern the whole
preflight, not each provider call. Every provider receives only the remaining
time and capacity; candidate, source base, historical adapter packages, all retained
bundle generations, and deduplicated contexts debit that same budget. Reused
content IDs count once, while the same ID with different bytes fails loud. Each
case is an explicit matched pair: a source-base-control leg and candidate leg share
one historical adapter, context, evaluator binding, and artifact closure. The
aggregate request, every case, and every leg are content-addressed with exact
dependency closure. The prepared object rechecks its settings/config identity,
selection cases, byte-derived lineages, and aggregate limits, and owns all bytes;
a sandbox may not resolve IDs or mutable pointers.

Each case also owns one immutable `ExpertSourceReplayComputeBinding`. It copies
the configured execution-provider, paired-protocol, and sandbox-policy versions;
the exact per-leg wall time, termination grace, CPU, memory, shared-memory,
process, open-file, writable tmpfs inode/allocated-storage, output, stream, and
accelerator limits; and a
deterministically counterbalanced two-leg order. Its content ID is part of the
case dependency closure and matched-compute digest. Both legs therefore receive
the same authorized allocation, but their observed duration and consumption may
differ and belong only in execution receipts. Scientific repeats remain solely
the exact evaluation fingerprints: the executor runs each named leg once and may
not add an observation-dependent retry. The sandbox policy is structural, not a
set of caller-controlled booleans: its version must dispatch to an implementation
that guarantees offline direct execution, read-only inputs, fresh private
writable roots, and the adapter-declared fixed non-secret environment.

Provider selection is one exact composite dispatch over the paired-execution
protocol, execution-provider ID, version, and canonical settings digest,
sandbox-policy version, historical adapter runtime-protocol version, and
historical task-evaluator protocol version.
The registry pre-resolves every case in the aggregate prepared request before
reservation or filesystem work and fails if any complete key is absent. There
are no wildcards, aliases, compatible-version ranges, per-field lookup, or
defaults. Image/platform identities and resource ceilings remain exact case
inputs checked by the selected provider; they do not select an implementation.
The resolved provider advertises the same full key again immediately before and
after execution so registry mutation or provider substitution cannot bypass
dispatch. Resolution keeps the provider private. After the durable spawn marker,
a guarded one-shot capability invokes that exact provider with no caller-supplied
arguments; the registry binds both verified expert trees, while the capability
passes only the exact source-base or candidate byte closure selected by the allocated
leg. Only its session-registered sealed completion may enter the journal.

The Docker provider executes a digest-namespaced private copy of the pinned CLI
against only the configured root-owned Unix socket, an empty private Docker
configuration, and a minimal fixed host environment. Every leg freshly requires
the exact client/server API, host platform, storage and cgroup authorities,
default runtime, isolation capabilities, security-option set, and required local
drivers. Local image admission permits no pull: manifest/config/platform and the
adapter-declared environment must match exactly, while inherited commands,
volumes, and healthchecks are forbidden. The adapter contract must declare a
non-empty `PATH`. Container creation explicitly passes every declared variable
and overlays only the provider-owned `HOME=/kapso/home` and
`HOSTNAME=kapso-task-evaluation`; exact container inspection therefore proves the
complete direct-evaluator environment without relying on Docker's implicit Linux
defaults. The host kernel, root-owned dynamic
loader/libraries and daemon, and same-UID processes able to mutate the private
provider root are explicit trusted computing base; evaluator code is not.
Registry bootstrap derives every distinct full dispatch key from the prepared
historical request, requires every implementation-selecting dimension to match
the concrete provider's code-owned v1 constants, and resolves the full request
before fd-relative/no-follow initialization of the configured private hierarchy.
The registry is permanently bound to that complete prepared byte authority, not
only its dispatch key. All resolved providers share one lazy pinned runtime created
only by execution or interrupted cleanup; deterministic received-result acceptance
therefore has no Docker dependency. Bootstrap never reloads current settings. A
kernel lock on the authorized workspace root serializes configured-hierarchy
creation across processes, while a lock on the owner-private provider root
serializes pinned runtime-authority publication; crashes release both locks with
their descriptors.
Daemon resource names use the complete unpredictable provider-handle digest and
carry exact handle/role labels. The writable root is a fresh local-driver tmpfs
volume whose size and inode options are derived from the matched compute binding.
Cleanup validates every extant resource before mutation, removes containers by
their inspected immutable IDs, rechecks the labelled volume before removal, and
treats absence as idempotent success without any start or exec operation.
For a live leg, the provider materializes only the journal-selected expert tree,
verified adapter, captured task artifacts, and canonical blinded request; captures
the pinned BusyBox helper into that private workspace; starts a read-only keeper;
and exact-inspects both keeper and evaluator before use. The evaluator runs by
direct absolute entrypoint under the matched wall, stream, CPU, memory, shared
memory, process, open-file, and writable-root limits. It is stopped and removed
before the keeper emits the bounded tar snapshot. Only a completed zero exit may
enter strict result parsing; every other bounded outcome returns no result bytes.
Normal return and failure both reap daemon resources before fd-safe workspace
deletion.

The explicitly invoked production check is
`pytest -q tests/live_expert_replay_docker.py -s`. It serves a deterministic,
digest-addressed scratch image from a loopback-only OCI registry, pulls that
exact digest once, then runs both counterbalanced legs through the request-bound
registry, fresh-authority coordinator, journal, and concrete Docker provider. It
requires zero subsequent registry requests, accepts the exact source-base-control and
candidate scores, publishes the typed source-stage result into the validation
journal, replays the completed stage without another provider bootstrap, and
proves that every handle-owned container, volume, and workspace is gone. The
fixture synthesizes task-adapter, current-release, and denylist authority; it does
not exercise their production GitHub transports.

The corresponding task-matrix production check is
`pytest -q tests/live_expert_task_evaluation_docker.py -s`. One digest-pinned
loopback image executes both semantic legs of a control comparison and the sole
candidate leg of a bootstrap matrix through the request-bound concrete registry,
double-reopen fresh authority, and four-event durable journal. It asserts the
candidate/control values by semantic leg kind, reconstructs each completed store
without a provider or Docker object, proves no registry traffic after the initial
pull, and leaves no handle-labelled container, volume, or workspace. As above,
the fixture synthesizes adapter, current-release, and denylist transports while
exercising their production typed boundaries.

The task evaluator is a blinded scientific ABI, not a view of validation
authority. Its request/result contracts, canonical parser, stable aggregation,
mount paths, and opaque `task_evaluation_invocation_*` namespace now live in the
domain-neutral `task_evaluator_protocol` module. Source replay retains only its
source-specific allocation authority above that ABI; adapter-owned matrix cases
use `TaskEvaluationInvocationAllocation` with exact task reservation/case/leg
namespaces. There is no replay-named ABI alias or old invocation format. Protocol
v1 writes one canonical request to
`/kapso/input/request.json`, mounts the selected expert at
`/kapso/input/expert`, the verified adapter at `/kapso/input/adapter`, captured
task artifacts beneath `/kapso/input/task`, and accepts only
`/kapso/writable/result.json`. These paths are protocol constants rather than
request-controlled fields. Before a spawn, the local execution journal mints a
CSPRNG nonce, durably allocates it to the exact reservation/case/leg, and never
reuses that allocation; protocol construction accepts only this typed allocation
and derives the exported opaque invocation ID from its complete private binding.
After that ID is redacted, both legs receive byte-identical
request bodies containing only the input/target contract fingerprints, complete
source `EvaluationFingerprint` records, the exact transfer dimensions named by
the adapter's `consumed_dimension_ids`, and logical starting-artifact
reference/mount pairs. Bundle history, source scores and effects, score-of-record
selection, episode and validation provenance, compute policy, leg order/kind,
and candidate/source-base/tree identities remain in trusted Kapso receipts and never
enter the adapter sandbox.

The sole native evaluator result echoes the protocol and opaque invocation ID
and contains one ID-sorted row per requested fingerprint: a finite floating-point
aggregate and one finite floating-point value for every exact seed/replicate ID.
It contains no pass/fail, winner, delta, costs, diagnostics, or general artifact
map. Kapso rejects noncanonical JSON, unknown fields, missing/extra/substituted
fingerprints or replicates, unsupported aggregation protocols, and aggregates
that differ from a trusted recomputation beyond the policy-pinned tolerance.
Parsing occurs only after a successful bounded process outcome and a trusted
freeze of the sole regular result file; stdout is never a result channel. The
trusted paired reducer, not the adapter, applies metric direction and scale to
produce factual deltas. A later decision reducer alone applies scientific
thresholds and chooses a validation outcome.

Scientific comparison authority is nevertheless explicit before execution. Each
verified task adapter binds every supported evaluator-fingerprint/metric pair to
one domain-neutral promotion dimension, the exact objective direction, and a
finite positive metric scale. Preflight requires every selected
`EvaluationFingerprint` to match exactly one binding and requires that binding's
dimension and direction to exist unchanged in central promotion policy. The
adapter therefore defines metric semantics and scale, while central policy alone
defines noise floors, hard-regression ratios, and repeat requirements. The paired
reducer divides direction-aligned deltas by the bound scale; it must never
infer that a metric such as accuracy is `quality`, divide by an observed control
score, or reinterpret aggregate-recomputation tolerance as scientific noise.

Preflight re-observes CURRENT after source-base materialization, but its request
is evidence, not an execution lease. Source execution remains fail-closed until
the executor can atomically reserve the unchanged validation head and, immediately
before process spawn, recheck CURRENT, candidate/release revocation,
and every historical adapter package's verifier authority, trust, and revocation
state. It must produce exactly one result for each named leg and publish a typed
paired-comparison receipt against the same reservation. If `CURRENT` changes
while an attempt is validating, preflight returns no executable request: it
publishes a typed authority-invalidation record against the observed validation
state ID and returns the resulting terminal state. That content-addressed,
compare-and-swap transition closes the attempt as `FAILED`; the candidate can
no longer execute or be re-enrolled as valid because its immutable manifest pins
the source base that is no longer current. Evolution must rebase the change into a new successor candidate
against current authority and enroll that new identity; retrying the old candidate
deterministically remains ineligible.

The reservation substrate records admission as one immutable, content-addressed
operation alias on the unchanged authorization transition. It does not mint a
second scientific validation state, mutable lease, owner, nonce, or expiry.
Rebuilding preflight after a crash therefore produces the same request against
the same state: an identical reservation replays exactly, while a different
request cannot reserve that head. The journal lock makes reservation versus
invalidation atomic, orphan objects before journal publication are harmless, and
one authorization transition admits at most one request. A local execution lock
may avoid duplicate paid work but is never authority; final receipt publication
remains fenced by validation-head compare-and-swap.

The execution journal is a private create-only event directory derived from the
validation-store root. Each reservation has one exclusive lock, one canonical
hash-chained event prefix, and no mutable head file. The legal leg order is
derived from the persisted request and each case's counterbalanced `leg_order`;
the caller cannot select work. Each successful leg has exactly four events:
`INVOCATION_ALLOCATED -> SPAWN_COMMITTED -> RESULT_RECEIVED -> RESULT_ACCEPTED`.
The allocation mints one 128-bit CSPRNG nonce for the exact
reservation/case/leg and reuses it after restart. `SPAWN_COMMITTED` persists the
fresh authority fence, exact provider key, canonical evaluator request, and
comparison tolerance immediately before the provider side effect; only the
runtime capability returned after its directory fsync may start execution.
`RESULT_RECEIVED` persists bounded process observations and references a private
immutable raw-result blob whose bytes are committed before the event.
`RESULT_ACCEPTED` is minted only after those exact bytes are reparsed against the
persisted request and tolerance. Only an accepted tail advances the schedule.

A complete prefix is converted to reducer authority only through a sealed,
runtime-only `CompletedExpertSourceReplayExecution`. The live reservation session
rereads the durable journal, reruns its complete event/blob validation, requires
exactly four events for every scheduled leg, and compares that disk view with its
in-memory prefix before minting the capability. The detached capability is bound
to its store instance and creator process; it is immutable and reconstructable
after restart, but it is neither serialized nor fresh publication authority.
This avoids a second durable completion state machine and lets later network
checks and validation CAS run without holding the execution lock.

The factual reducer accepts only that sealed capability plus the exact
reservation and prepared request. It identifies control and candidate by their
leg IDs rather than counterbalanced position, then pairs results by evaluation
fingerprint and replicate IDs. Each fingerprint row embeds its exact fingerprint,
adapter-owned metric binding, both accepted result rows, candidate-minus-control
delta, direction-aligned delta, and scale-normalized effect. Mathematical zero
is canonicalized to positive zero and every derived value must remain finite.
The content-addressed receipt preserves case/fingerprint rows without averaging,
the score-of-record identity, both accepted-event IDs, the chronological complete
journal chain, the aggregate-recomputation tolerance, and the expanded
reservation/request dependency projections from which its exact closure is
rederived on every parse. It records no threshold, pass/fail,
winner, noise estimate, Pareto decision, retry, or promotion state.

The pure source-stage decision reducer consumes that factual receipt and the
runtime-only `PreparedExpertSourceReplayRequest` authority. From the latter it
derives the exact content-addressed execution request and the validation policy
explicitly pinned by that request. It requires the receipt's full request-dependency projection,
case identities, score-of-record identities, and aggregate tolerance to match.
Each complete fingerprint body must equal the source episode's terminal attempt,
and each adapter-owned metric binding must equal the binding in the verified
historical task-adapter manifest before policy is applied. Every
case/fingerprint comparison remains an independent hard-regression constraint:
the candidate fails iff any scale-normalized, already direction-aligned effect
is strictly below the negative central `hard_regression_ratio`; equality with
the bound passes. The reducer never averages rows, reapplies metric direction,
or lets a gain compensate for a regression. Score of record remains the primary
reporting identity but does not exempt auxiliary governed metrics from the hard
gate. A complete factual receipt therefore yields only `passed` or
`candidate_failed`. Noise floors, repeat sufficiency, independent contexts, and
positive-benefit support belong to later promotion aggregation, not this
non-regression stage. The durable decision stores only exact hard-regression
case/fingerprint references and the expanded receipt/policy dependency closure;
fresh external authority and validation-head compare-and-swap remain a separate
publication step.

That final step uses a distinct `SourceReplayDecisionPublicationFence`; a
per-leg spawn fence cannot authorize it because it predates the scientific
result and binds only one invocation. The final fence has no invocation
allocation. It records a newly fetched `CURRENT`, freshly reverified
historical adapters and verifier identities, and one exact denylist observation
covering the reservation/request, full factual receipt and decision, every
execution event, and every nested spawn-fence, adapter, verifier, provider-handle,
and denylist dependency. The resulting
`ExpertSourceReplayStageResultRecord` is self-contained: it nests the factual
receipt, policy decision, and final fence and binds them to the exact validation
attempt, authorization transition/state, candidate tree, reservation, policy,
and configuration. It is source-stage evidence, not a fabricated evaluator run
or attestation.

Publication idempotency is keyed only by the reservation command: candidate,
the reservation's authorization transition, and reservation ID. Receipt,
decision, and fence identities do not enter the operation ID. The store replays
that operation before external work and again under its exclusive lock, so the
first valid fresh fence wins and concurrent or lost-response retries return its
exact source result. A process/store/coordinator-bound one-shot permit is required
for a new commit; serialized fence bytes alone are not authority. Receipt,
decision, fence, source result, target state, operation, and transition are
written as immutable objects before the candidate journal is atomically replaced
as the sole visibility point.

Publication fsyncs a private staging file, renames it atomically without
replacement, and fsyncs both directories. Event, request, result, staging-entry,
and structural event/blob counts are config-bounded before allocation or parse.
Unsafe ownership or modes, links, unexpected entries, noncanonical bytes, forks,
gaps, phase substitutions, reused invocation identities, or result-digest
changes fail loud. An allocation-only tail is resumable. A reopened spawn tail is
permanently interrupted and can never execute again. Its persisted typed provider
handle dispatches idempotent cleanup of daemon resources; repeated cleanup may
remove only resources bearing that exact handle and never invokes the evaluator.
A received-result tail may resume deterministic parsing without provider
authority, including after a crash. A technical, missing, or invalid result
remains durable and terminal for that invocation rather than silently becoming
another trial.

The executor reopens a reservation through a public read-only store boundary.
The journal itself requires and reconstructs the complete prepared byte authority
and exact pinned policy before taking its lock; one short shared-lock
read then requires the exact journal-bound reservation, stored request, current
transition/state/attempt, candidate, and observed CURRENT. GitHub `CURRENT`,
historical adapter re-verification, and the live security denylist are checked
outside the validation lock. A second identical reopen after those external
checks closes validation-head races before the local spawn boundary is written.
The store lock is global, so it never encloses archive verification, network
access, an execution-journal lock, a callback, workspace work, or provider start.
Enrollment, evaluator-result publication, reservation admission, and source-base
invalidation all use the same pattern: shared exact-replay and head observation,
unlocked reducer/provider/verifier work, then exclusive exact-replay-first and
unchanged-snapshot compare-and-swap before any write. Identical concurrent
operations converge on one transition; a different winner makes the stale
operation fail without publishing its reduced objects into the journal.
No local lock can make GitHub, a denylist, and process creation transactional;
safety instead comes from the double reopen, the durable at-most-once marker,
and a final fresh-authority plus validation-head CAS before accepting receipts.

The fresh spawn-authority coordinator implements that external interval as one
aggregate operation. It retains a verified GitHub observation containing the
exact expert publication, `CURRENT` pointer digest, branch commit, repository
identity, and validation closure; reopens every unique prepared historical
adapter and records its verifier/version and complete proof dependency closure;
and submits an internally derived subject set to one authenticated denylist
provider. That set includes the reservation and request, the request's entire
dependency closure, source-base-release and candidate dependencies, adapter proofs,
and a content-addressed synthetic identity for each verifier authority. The
provider must echo the exact sorted subject set with no denied subject. The
resulting content-addressed fence persists the exact checked-subject tuple and
binds an invocation allocation owned by the coordinator's exact execution-store
instance under its live per-reservation lock, current release, adapter
observations, denylist snapshot/publication/repository/pointer/attestation, and
generation. Callers cannot mint an allocation, omit subjects, construct a fence
from top-level candidate/release IDs alone, or retain fresh authorization for a
later spawn. The coordinator's final reopen flows directly into the create-only
spawn append and returns only the post-fsync execution capability. The journal
rederives the provider key, handle, evaluator request, adapter observations, and
security subject tuple from the prepared closure on every reopen; possession of
serialized fence bytes is not runtime authority.

The reservation API accepts only the runtime-only prepared closure, reconstructs
it to rerun all byte, lineage, context, artifact, adapter, source-base, candidate, and
aggregate-budget invariants, independently re-derives every compute binding from
the persisted settings, and then persists its request. A self-consistent
content contract without those prepared authorities is not executable admission.

Reservation admission reopens the candidate, rechecks `CURRENT`, re-resolves
every historical adapter package through its retained trusted verifier, and
revalidates the accepted evaluator prefix before binding the operation. A
short shared-lock read first captures the expected validation head; all candidate,
GitHub, archive, and verifier work then runs without a validation lock; and a
final exclusive-lock compare-and-swap either binds the unchanged head or replays
an identical concurrently committed reservation. A changed head or different
request fails rather than carrying stale external validation into admission. A
historical reservation stays auditable after authority changes, while execution
must perform fresh CURRENT, revocation, and verifier observations immediately
before each spawn and again before receipt publication. Adapter dependency
closure includes every sanitation and validation proof reference, so later taint
or revocation cannot miss a proof that package verification consumed.

Implemented validation substrate:

- enrollment reopens the exact M7 candidate, resolves the current release through
  a GitHub reader that verifies repository policy and the observed immutable
  release identity, and resolves every trigger binding through the provider's
  trusted active adapter index rather than accepting caller-selected manifests;
- validation track and stage plan are recomputed from trusted candidate records;
- exact source bundle IDs resolve through a bounded predecessor walk and a
  retained root-to-tip byte closure and independent projection replay;
  generation-zero roots, monotonic adjacent frontiers, immutable Git/source
  proof, and the complete sanitation closure are required without consulting a
  mutable current pointer;
- every exact trigger adapter binding is enrolled under an unambiguous canonical
  identity and an exact `TaskAdapterPackagePin`, while configured task families
  remain explicit stage-policy inputs; reducer replay resolves those immutable
  pins even if the active publisher attestation rotates;
- attempts retain the complete eligibility, adapter, and verification dependency
  closure;
- ordinary executable-stage results are bounded before identity, stored as typed
  `ExpertEvaluatorResultRecord` contracts, signature-verified through an injected
  fail-closed verifier, and accepted only in exact prefix order; and
- retry lineage preserves both the current state and latest historical attempt,
  and restarts at stage one. An intervening ineligible state cannot reset attempt
  identity; approved, released, validating, and revoked states cannot be retried.
- durable validation history stores immutable content-addressed decisions,
  configurations, attempts, signed evaluator-result envelopes, states, operations,
  and transitions behind one atomic per-candidate journal;
- validation states and transitions carry one ordered typed stage-result prefix:
  each state reference binds the exact `ExpertValidationStage` to its result-record
  namespace, and journal replay proves record identity, outcome, candidate,
  attempt, and canonical stage-plan order. Ordinary stages retain evaluator result
  records; source replay has a distinct reserved stage-result namespace for its
  later receipt/decision/publication-authority record rather than a fabricated
  evaluator run;
- operation-to-transition bindings make lost-response retries exact, while the
  journal head provides compare-and-swap publication with no fork, merge, or
  rollback behavior; provider and attestation work runs outside that lock and an
  exact retry returns before repeating external verification;
- immutable source-replay reservation aliases admit exactly one byte-closed
  prepared execution request without changing the validation head and replay
  exactly across process/store recovery;
- the local source-replay execution journal durably allocates the first exact
  scheduled invocation once and enforces the complete four-event lifecycle for
  every leg; sealed runtime-only allocation, spawn-authorization, and spawn
  capabilities prevent serialized authority from crossing process boundaries;
  the private version-dispatched provider is called once through a guarded
  no-argument executor, and raw process results or evaluator bytes are never an
  admission API;
  immutable bounded result bytes precede deterministic typed acceptance; restart
  resumes allocation or result acceptance but never a committed spawn; persisted
  provider handles support idempotent daemon-resource cleanup without reexecution;
  concurrent sessions serialize; and corrupt, forked, substituted, over-bound, or
  unsafe journal state fails loud without a mutable execution snapshot; and
- source replay and task evaluation now consume one neutral task-evaluation Docker
  sandbox rather than owning parallel runtime machinery. Pinned CLI/daemon/image
  authority, handle-labelled resources, owner-private workspaces, verified byte-tree
  publication, the complete container lifecycle, cleanup, stream bounding, and strict
  BusyBox result-snapshot parsing are single-sourced under task-evaluation names. The
  sandbox accepts only a content-addressed handle, explicit byte closures and limits;
  source replay and task evaluation are thin projections from their independently
  sealed invocations. Their complete request-specific key sets are resolved before
  filesystem or Docker work and share one lazy runtime authority plus one race-safe
  trusted-root initializer. Neither producer's request, candidate, provenance, or
  reservation types become sandbox authority, and the source-named runtime, resource,
  filesystem implementations and labels are removed; and
- the task-evaluation journal now has its own minimal four-event contract and
  pure offline prefix reducer. Its sole schedule follows canonical request-case
  order and each case's semantic counterbalanced leg order, so repeated leg IDs
  across cases cannot alias. Every spawn is rederived from the exact reservation,
  prepared bytes, fresh fence, provider key and handle, blinded evaluator request,
  and configured tolerance. Referenced result bytes are digest-, size-, policy-,
  and compute-bounded; accepted events are reparsed from those bytes, while a
  malformed received-result tail remains safely reopenable and non-advancing; and
- source replay and task evaluation now share one semantics-free private journal
  filesystem. It validates owner-private roots and digest-only paths, distinguishes
  absent or empty partial layouts from corrupt durable partial state with
  `lexists`, bounds directory enumeration and no-follow reads, validates the whole
  staging set before cleanup, publishes immutable numbered events and result
  blobs with no-replace rename plus file/directory fsync, and carries process-local
  flock ownership. Source replay retains its own schedule, event reducer, permits,
  and completed capability; no generic callback state machine or legacy store
  error/result-blob alias remains; and
- task evaluation now has a concrete task-specific store on that filesystem.
  It owns the exact case-scoped schedule, same-nonce allocation restart,
  coordinator-sealed spawn append, one-shot provider capability, sealed-completion-
  only result admission, deterministic received-result acceptance, cleanup-only
  interrupted spawns, and store/process/live-lock-bound completed proof. The exact
  registry alone resolves `(case_id, leg_id)`, constructs the provenance-erased
  invocation under its private seal, checks provider identity before and after one
  call, and requires the exact completion handle. Every append is validated by the
  pure reducer before publication and poisons its session across an uncertain
  create-only response. Both task evaluation and source replay validate raw
  stream-limit counts against the exact runner outcome, then persist only the
  canonical `limit + 1` sentinel for the triggering stream; offline replay
  requires that exact bounded representation; and
- fresh task-evaluation spawn authority is one non-splittable
  `R0 → C0 → adapters → denylist → C1 → R1 → spawn` operation. `R0`
  must equal the live allocation's complete reservation snapshot, every distinct
  prepared adapter is reverified once in canonical order, `C1` must equal `C0` in
  full, and `R1` must equal `R0`. The coordinator exposes neither the fence nor a
  provider selection and returns a spawn permit only after the store fsyncs the
  at-most-once marker; and
- a complete journal can mint only one deterministic store/process-bound completed
  capability. The factual reducer reopens the exact durable task reservation,
  resolves accepted results by case plus semantic leg, and emits the task-owned
  rows in global plan order. The report embeds a content-addressed proof projection
  of the request, reservation, complete chronological journal, case/fingerprint
  coverage, accepted-event lineage, and exact expanded dependency closure, but no
  effects, winner, threshold, or promotion decision; and
- fresh spawn authority performs the exact double reopen around rich GitHub
  `CURRENT`, complete historical-adapter trust, transitive denylist, and verifier
  authority observations and returns one invocation-bound typed fence; and
- one production source-stage orchestrator composes preflight, complete provider
  resolution before reservation, reservation replay, durable journal-tail
  recovery, fresh-authority execution, and final atomic publication. Empty and
  allocated tails continue, received results are parsed without provider work,
  completed publications replay without external work, and a reopened spawn is
  idempotently cleaned then reported as permanently non-reexecutable. One
  canonical execution-journal child and a candidate-scoped kernel lock prevent
  alternate local journals or concurrent orchestrators from duplicating paid
  work, while a partially created empty journal layout remains restartable; and
- automated review is a distinct typed validation stage rather than an evaluator
  alias. A self-contained candidate source input and the complete ordered accepted
  evidence prefix are rendered without truncation to every configured independent
  reviewer in an empty read-only workspace with no tools or prior-knowledge MCP.
  The v2 packet dispatches on the candidate derivation: direct proposals retain
  their real authoring operation, while deterministic compositions retain their
  materialization, plan, source authorities, and origin-principal union without
  inventing an outer proposer invocation. Durable replay stores and reopens that
  same discriminated union.
  The coordinator constructs the configured CLI runner from the authorized
  workspace root and rejects runner, configured-path, prompt-budget, or sealed
  artifact-budget substitution. Deterministic operation identities make paid calls
  restart-safe; exact agent artifacts, receipts, assertions, and
  unanimous/rejected/disputed adjudication are persisted before one packet-keyed
  journal CAS. Restart re-derives the full closure, accepted reviews advance only
  to the release matrix, and rejected or conflicting reviews terminate without
  appending a passed-stage reference; and
- current-release-authority invalidation is a content-addressed terminal transition
  that preserves accepted-stage history and proves expected versus observed
  `CURRENT`. It covers source-base advancement or disappearance and a release appearing
  after bootstrap absence, so stale attempts cannot accept their remaining work.

Terminal publication eligibility, composition, and release remain separate later
slices; neither automated review, the factual matrix reducer, nor the pure Pareto
decision can synthesize their authority.

The shared adapter trust boundary now separates stable scientific manifest
identity from exact verified package identity. A typed verification receipt binds
the full manifest bytes (including its excluded publisher attestation), exact
source archive/tree, extraction receipt, sanitation and validation proof closure,
and verifier identity. Eligibility records that receipt closure transitively. The
receipt-keyed immutable package store re-extracts the hardened canonical tar/zstd
archive and re-verifies exact source/proof/publisher bytes and configured authority
on publication and read. Signed activation records move one logical binding under
compare-and-swap while historical pins remain replayable. A generic content ID is
not accepted as production verification. The configured trust registry selects
one active authority for new records, retains explicit historical verifier
versions for replay, and treats removal as revocation; deployment supplies those
authority implementations behind the typed fail-closed protocol.

The scientific manifest itself has no opaque evaluator, context, or runtime maps.
It contains exactly three domain-neutral contracts:

- `TaskEvaluatorBinding`: a protocol version and normalized adapter-relative
  executable path, a sorted allowlist of exact protected evaluation-tree
  fingerprints, and sorted exact evaluator/metric comparison bindings. Each
  binding cross-checks objective direction and supplies one central promotion
  dimension plus a finite positive task-semantic scale. The package verifier
  attests those compatibility and comparison claims; the protocol fixes
  request/result layout and direct, no-shell invocation.
- `TaskAdapterContextBinding`: a sorted allowlist of transfer-dimension IDs the
  evaluator consumes. It may be empty, must be a subset of the exact scope schema,
  and every replay context must contain the declared dimensions.
- `TaskAdapterRuntimeContract`: runtime protocol, normalized registry-qualified
  image repository, platform-manifest and image-config digests,
  dependency-lock path/digest, operating system, architecture, and optional OCI
  variant, plus the key-sorted non-secret environment that must exactly match
  the pinned image configuration. The derived `repository@manifest-digest` reference is the sole
  executable image authority; the config digest disambiguates a platform image
  from an OCI index. A bare local image ID or mutable tag is not authority.

Package verification requires an executable mode of `100755`, proves the runtime
lock against the extracted source bytes, and rejects mutable image locations.
Metric/direction/fidelity/replicate/aggregation authority stays in the exact
`EvaluationFingerprint`; resource ceilings and the exact paired execution envelope
belong to validation config and the execution request respectively. The replay
case pins every source fingerprint plus its score of record and hashes the source
revision, context receipt, starting artifacts, adapter tree, evaluator ABI,
context allowlist, and runtime proof into one matched-compute binding. This keeps
post-training, relational tabular prediction, and later ML task families on the
same core contract without teaching Kapso task-specific metric or budget names.
M9 must reject a launch whose possible evaluator IDs are absent from the pinned
adapter; evaluator evolution outside the allowlist stays local-only until a newly
verified adapter package attests the new protected evaluation-tree fingerprint.

## Evaluator cascade

Implement one durable state machine:

```text
contract/schema
-> identity/secrets/license/dependency scan
-> static/unit/security/resource tests
-> synthetic fresh-task smoke
-> source-run replay
-> visible development anchors
-> configured cross-family transfer matrix
-> sealed canary attestation
-> independent automated reviewer decision
-> complete release matrix
-> publication eligibility
```

- [ ] Persist every attempt, exact candidate/tree, config/policy, evaluator identity,
      inputs, outputs, cost, duration, and attestation.
- [ ] Require each stage before the next; no best-effort continuation after failure.
- [ ] Distinguish infrastructure fixes, behavioral capabilities, architecture
      changes, task-specific changes, and revocations with explicit policies.
- [ ] Validate scope conformance, repository-map graph, module contracts, book
      digest, adapter isolation, and capability lineage.
- [ ] Run source replays against faithful source contexts and fresh-task smokes
      against identity-disjoint public/synthetic adapters.
- [ ] Keep sealed examples/details outside proposer/reviewer prompts; consume only a
      signed aggregate attestation from the promotion service.
- [x] Use the configured Pareto dimensions and hard regression bounds across
      quality, robustness, cost, portability, reproducibility, and security.
- [x] Treat noise-floor gains as inconclusive until configured repeat evidence
      exists.

Stage applicability is deterministic. Bootstrap omits replay, anchors, transfer,
and canary because no source-base evidence exists. Mechanical fixes use deterministic
gates, fresh-task smoke, source replay when a source base exists, review, release
matrix, and publication. Behavioral changes additionally require development
anchors, cross-family transfer when more than one family is bound, and a sealed
canary. Architecture changes use the release-wide path and require a canary only
when the typed policy says so. An unavailable required canary makes the candidate
ineligible; it is never treated as a skipped or passed stage.

## Automated review and decision

- [x] Accept reviewer assertions only from configured autonomous identities/roles.
- [x] Require exact candidate, evidence, evaluator-run, rubric, and source-base-release
      references.
- [x] Preserve conflicting reviews as disputed; do not overwrite by time.
- [x] A separate coding-agent/service role reviews each proposal; the proposing
      invocation cannot review its own output or transition state.
- [x] Review direct proposals and deterministic compositions through their real,
      mutually exclusive derivation evidence; compositions have no fake proposer.
- [ ] Supported task-specific improvements remain knowledge/task-adapter candidates,
      never expert core.
- [ ] Failed or non-dominated candidates stay immutable in the candidate archive;
      they are not installed into runs.

Promotion states are explicit: `ineligible`, `validating`, `failed`, `disputed`,
`pareto_retained`, `approved`, `released`, and `revoked` as frozen by M1. There is
no implicit promotion from a direct Git commit; only a completely validated
immutable release referenced by `CURRENT.json` is active.

## Typed release matrix and publication eligibility

The final Pareto decision must not consume `ExpertEvaluatorRun.measurements`
directly. That generic map has no control pairing, adapter-owned scale/direction,
context lineage, repeat identity, or comparability authority. The release matrix
now reduces into one typed factual comparative closure and is accepted as its own
validation stage before Pareto reduction. The pure decision and its sealed
`PUBLICATION_ELIGIBILITY` transition are implemented. This transition approves an
exact candidate for later release assembly; it does not mutate GitHub or grant a
release lease.

Minimal ownership:

- `promotion_contracts.py` owns the immutable adapter authority, provenance,
  full-fingerprint cell, precommitted evaluation plan, factual row, and report;
- `promotion_plan.py` derives cells from the frozen attempt, accepted source
  replay, and adapter-owned cases, then `validation_store.py` reserves exactly one
  plan against the unchanged `RELEASE_MATRIX` validation head before any process
  starts;
- the task-evaluation substrate generalizes source replay's request, journal,
  bounded execution, accepted-result, and fresh-adapter-authority machinery for
  cells that cannot reuse exact accepted replay evidence;
- `promotion_evidence.py` requires a store/process-sealed completed task journal,
  independently reopens its exact durable reservation, resolves every referenced
  accepted event, reuses a source comparison only on exact
  candidate/source-base/adapter/context/fingerprint identity, and mints the report; it
  never accepts evaluator-authored rows or effects;
- `promotion_stage_contracts.py` embeds that report in the exact accepted-stage
  authority, while `promotion_stage.py` preserves the process-local execution
  capability through one validation-store CAS. Raw reports and caller-minted
  result records cannot publish; after acceptance, restart and audit reopen only
  the self-contained stored result and do not depend on the disposable task
  journal;
- `promotion.py` re-derives per-replicate direction-aligned normalized effects and
  computes a content-addressed Pareto decision without weighted scores. This pure
  decision has no transition authority by itself; terminal publication consumes
  the exact accepted stage reference from the validation store;
- `promotion_authority.py` proves fresh `CURRENT` or authenticated bootstrap
  absence plus exact adapter/verifier/denylist authority and seals publication;
  and
- the terminal promotion coordinator, `validation.py`, and `validation_store.py`
  reduce and persist `PUBLICATION_ELIGIBILITY` through one journal
  compare-and-swap.

Terminal authority is outcome-sensitive. `FAILED` and `PARETO_RETAINED` are pure
local reductions: they make no CURRENT, adapter, or denylist call, preserve the
accepted matrix prefix, and persist the decision as terminal evidence. Only
`APPROVED` reopens the exact accepted matrix and candidate, observes authenticated
CURRENT (or bootstrap absence), re-resolves every matrix adapter package including
source-only historical packages, derives the exact adapter-observation set, checks
the complete candidate/history/decision/adapter closure against the denylist,
observes the identical CURRENT authority again, and reopens the unchanged local
head before the sealed store CAS. A changed release identity produces the durable
generalized CURRENT-authority invalidation; same-release metadata movement,
substitution, denial, or a local-head race fails closed without eligibility.
For a composed candidate, the denylist projection includes the outer derivation,
materialization, assessment, and plan plus every direct source candidate, commit,
validation-context dependency, trigger, operation, receipt, workspace record,
sanitation report, ancestor closure, and source dependency. Composition therefore
cannot launder a denied source through a new outer candidate identity.

The authority fence, decision, result, operation, state, and transition are
content-addressed and journaled atomically. The store independently re-derives the
decision, exact adapter observations, security closure, terminal result, and state
under the attempt's persisted configuration. Durable replay is resolved from the
matrix-result identity before any external authority call, so concurrent and
post-restart replay is offline and byte-identical. Only approval appends the typed
publication-eligibility result to accepted history; retained and failed results
remain transition evidence outside the accepted prefix.

The generic `ExpertEvaluatorRun` route is fail-closed for source replay,
automated review, release matrix, and publication eligibility; each requires its
typed stage path. There is no legacy flat-payload admission path. Durable plan
reservation, rather than merely embedding a plan in the later report, supplies
temporal precommit authority.
Operational stage publication additionally requires adapter-owned task evidence;
therefore a structurally valid source-only factual report remains useful for
analysis but cannot become the accepted `RELEASE_MATRIX` result.
The plan binds the exact candidate and optional source-base trees, full verified adapter
packages, task contexts, source lineage or adapter-owned case, complete
`EvaluationFingerprint` including every seed/replicate, metric authority, and exact
dependency closure. Source replay and adapter-owned-case provenance are orthogonal
to control-comparison versus bootstrap mode: a control matrix may mix both channels,
while bootstrap requires adapter-owned standalone cases and never fabricates a zero
control. Fresh provider verification is required at plan admission, before each
spawn, and before terminal publication; embedded authority exists for durable
offline revalidation, not as a substitute for freshness.

Adapter-owned execution uses neutral `task-evaluation-*` contracts rather than a
second source-replay dialect. One content-addressed case binds its adapter/provenance,
signed case, context, independence identity, complete cell/fingerprint set, artifact
IDs, compute envelope, and semantic expert legs. Control-comparison requests require
exactly one candidate and one source-base-control leg per case; bootstrap requests
require exactly one candidate leg and prohibit every source-base/control field. Each
leg is bound to the exact candidate or source-base artifact, source receipt, and tree. A reservation in turn binds
the unchanged matrix-plan operation, request, validation head, candidate, and
authenticated present/absent CURRENT state before any execution can begin.
The durable request binds the configured evaluator ID, role, and version. A separate
runtime closure joins it to the exact reserved plan and rejects omitted, substituted,
or foreign adapter provenances, cells, fingerprints, cases, contexts, independence
identities, artifacts, or compute. Compute is derived only from the release-matrix
evaluator timeout and the shared configured provider, sandbox, resource, output,
stream, and accelerator authority. Bootstrap has one candidate leg; control comparison
counterbalances candidate-first and source-base-control-first order deterministically across the
complete adapter-provenance set. That plan join is deliberately not a spawn
capability: materialization must still prove exact package bytes and fresh adapter
authority.

Request construction is also derived rather than caller-authored. The reserved plan,
configured release-matrix evaluator, immutable stored candidate, exact candidate
bytes, and optional candidate-bound source-base receipt/bytes produce the complete
canonical request, cases, legs, compute bindings, and dependency closure; the result
is immediately rejoined to the plan. Source-reuse provenances never become new
evaluator cases because their already-accepted rows are reduced separately.

The shared operational authority is single-sourced as
`validation.task_evaluation_provider` plus `task_evaluation_*` policy fields.
Execution protocol, provider identity/version/settings digest, sandbox, resource and
stream ceilings, journal/result bounds, accelerator, and aggregate materialization
limits are common to source replay and adapter-owned matrix cases. Aggregate
recomputation uses the one configured `task_evaluation_aggregate_tolerance`; the
former source-only name is removed. Evaluator identity,
role, version, and leg timeout remain stage-specific in `evaluators`; source selection,
bundle/episode limits, historical context materialization, stage decision, and score
comparison tolerance remain explicitly source-replay policy. The former generic
`source_replay_*` execution keys and provider block are rejected rather than aliased.

One provenance record represents exactly one declared evaluation case. Source
provenance names the accepted source-stage result, paired-comparison receipt,
execution case, selection, episode, complete bundle lineage, materialization receipt,
and every case-declared fingerprint. It cannot substitute a merely namespaced bundle
or select a subset of the accepted comparison. Multiple cases may legitimately share
the same task context or lineage root; they remain separate cases, while later power
analysis counts distinct context and independence identities without pretending that
same-root repeats are independent. Exact historical and active package versions of
one logical adapter binding may coexist. Coverage is the exact fingerprint set
declared by each case, never fingerprints invented from the adapter's metric catalog.
The eventual report must name the validation-operation alias that reserved the plan,
so embedding an otherwise identical plan cannot fabricate temporal precommit.

`TaskAdapterManifest` now owns a non-empty canonical set of signed release-matrix
cases. Each case embeds its full task context and fingerprints, an explicit
statistical independence group, and any starting-artifact byte trees stored under
the package's reserved asset subtree. Package verification rejects missing,
substituted, overlapping, hidden, or executable/runtime-aliased asset bytes. Exact
scope validation happens at enrollment and again at plan admission. The structural
anti-inflation rule prevents budget, runtime, metric, fidelity, or replicate changes
from turning one dataset origin into multiple claimed independence groups.
Evaluator containers never receive that full package tree: `/input/adapter` is the
exact verified runtime projection with the entire `release_matrix_assets/` subtree
removed. A matrix executor may expose only the selected case's declared artifact
closures under `/input/task`; source replay continues to expose only its independently
materialized historical task context. This prevents one case or source replay from
reading another signed case's fixtures through the adapter mount.

The byte boundary is now shared without conflating authorization producers.
`VerifiedTaskEvaluationCandidate` and `VerifiedTaskEvaluationSourceBase` prove exact
expert-tree bytes for both source replay and matrix execution.
`VerifiedTaskEvaluationAdapterRuntime` revalidates the manifest, verification
receipt, extraction receipt, executable, lock, and exact fixture-free runtime
projection. `materialize_task_evaluation_starting_artifacts` accepts only a case
embedded in that verified manifest and copies only its declared package paths into
immutable artifact closures. These objects prove bytes; the later materialized-case
closure must still join them to the reserved request, adapter authority, signed case,
and fresh provider observation before they become a spawn capability.

`PreparedTaskEvaluationRequest` now supplies that pure byte join. Every canonical
request case retains its full `VerifiedTaskAdapter`, proves the exact embedded plan
authority, derives the fixture-free runtime again, and matches only the signed case's
artifact closures. Historical source-only adapter versions are not executable cases.
Aggregate accounting includes candidate, optional source base, and each distinct full
adapter package exactly once; runtime and fixture projections are views of already
counted package bytes. It also retains the exact typed current-release observation on
which preflight relied. This prepared object is still neither a durable reservation
nor an execution capability.

Provider resolution now begins from that exact prepared closure. Its executable-case
projection removes plan, provenance, cell, and validation identities while retaining
the signed context, fingerprints, selected fixture bytes, fixture-free adapter
runtime, compute envelope, and semantic candidate/source-base legs. Bootstrap projects
only a candidate leg; control-comparison mode binds each semantic leg to its exact verified tree
and receipt. A full-key registry resolves every case and runs deterministic support
checks before reservation. The key contains only implementation-selecting protocol,
provider/settings, sandbox, adapter-runtime, and evaluator-protocol identities; mode,
leg order, scientific identity, resource limits, image, and accelerator requirements
remain exact case inputs. Missing, duplicate, or mutable provider identities fail
closed. Resolved providers stay private. Leg invocation is seal-only, derives its
complete canonical evaluator request and handle internally, carries only the selected
expert leg, and has no provider-execution surface until the journal owns its one-shot
spawn capability.

Fresh task-evaluation authority also has one shared observation vocabulary. The
adapter-verifier and authenticated denylist observations are now neutral contracts
used by both source replay and adapter-owned matrix execution; the old source-named
formats are removed. A minimal matrix spawn fence binds only the reservation, request,
allocation, one stable fresh current-or-absent observation, the exact sorted set of
all prepared adapter trust observations, and the exact denylist observation. Its pure
projection derives the complete checked-subject set from the reservation/request
closures, allocated case/leg pair, candidate and optional source-base dependencies,
current publication/validation closure, and every adapter/verifier dependency. Missing
or extra checked subjects, denied subjects, foreign scope/release/absence, substituted
adapters, or a cross-case allocation fail closed. A harmless branch-head advance after
admission may produce a new observation; the later coordinator must require its two
fresh observations to equal each other while preserving the reservation's exact
source-base release or bootstrap absence.

`TaskEvaluationPreflightCoordinator` is the sole producer that turns a reserved plan
into that byte-closed request. Its order is fixed:

1. reopen the exact plan alias at the unchanged local validation head;
2. reopen the exact candidate and join its manifest, trigger packet, decision, commit,
   tree, plan subjects, and configured validation fingerprint before any external read;
3. authenticate `C0`, which is either the expected source-base `CURRENT` or repository-head-
   bound bootstrap absence;
4. materialize the exact source base only in control-comparison mode, then derive the
   request immediately so a substituted source base fails before adapter acquisition;
5. resolve each distinct adapter-case package by its pinned manifest and receipt under
   the one configured entry, byte, and monotonic deadline budget; source-only historical
   packages are evidence and are not reacquired as executable packages;
6. authenticate `C1` and require the full observation to equal `C0`, including repository
   identity, default-branch head, pointer digest, publication, and validation closure;
7. reopen the exact plan alias again and require it to equal the first reopen; and
8. derive fixture-free runtimes and only the selected signed-case artifacts.

Bootstrap never calls the source-base provider. Exact observation equality detects a
release appearing during bootstrap and a source base restored under a new branch head
after intermediate movement. No network call occurs under the validation-store lock.
`TaskEvaluationReservation` then durably binds the exact request, plan alias,
authorization transition/state/attempt, candidate tree, scope contract and logical
scope, and the required content-addressed current-or-absent observation. The store
persists request, observation, reservation, and operation as one alias on the existing
release-matrix transition; it creates no new validation state. The exact prepared
closure is reconstructed before the exclusive local commit, and the journal reopens
the complete request/plan/configuration/observation join without GitHub, candidate,
adapter, or evaluator calls. Concurrent identical commits produce one winner. A later
identical replay preserves the first admission observation even if a harmless branch
commit produced a newer observation; spawn freshness is a separate authority. A
different request or changed validation/plan head conflicts. Every later spawn must
still reacquire live package, current-release, and denylist authority.

The planner now derives a mixed control matrix from the complete accepted source
replay plus every signed case in every immutable attempt-pinned active adapter.
Historical package versions needed by source evidence remain exact authorities but
do not contribute adapter-owned cases unless they are also active pins. Bootstrap
derives only active adapter cases, names no source base or control, needs no source
request, and reserves/reopens through the same unchanged-head journal alias.
Control-comparison mode requires accepted source authority and directly binds the
plan source-base tree to the candidate's verified source-base closure.

Plan reservation performs no evaluator spend. Admission reopens the candidate and
every exact package, checks current release authority before and after adapter
resolution, and rejects a release that appears during bootstrap admission. The
durable alias intentionally replays offline; authenticated current/empty authority
must be reacquired before evaluator spawn and again before terminal publication.
Identical concurrent admission replays the same alias, while a different plan, stale
head, changed current authority, corrupt durable join, or foreign adapter case fails
closed.

Accepted source replay is reused from the validation store, not by reopening its
local execution workspace. Under one shared lock, the store revalidates the current
matrix-plan alias, the complete accepted transition history, the historical source
reservation alias, and the separately persisted request, reservation, receipt, and
stage-result objects. The source-row reducer then copies only the accepted event IDs
and exact replicate maps for cells whose case, context, adapter, fingerprint, metric
binding, candidate, and source base all match the reserved plan. It performs no preflight,
provider call, freshness lookup, or fallback. Missing or corrupt accepted objects
fail loud, while loss of the already-published execution journal does not erase or
rerun accepted scientific evidence.

The report covers reserved cells exactly once and in canonical order, with complete
candidate and, when source-base-backed, control replicate maps. Observation-event
namespaces must match their provenance channel. Every fingerprint from one case
shares the case's single candidate/control accepted-event pair; an event cannot be
reused by another case or leg role. Adapter-owned rows additionally retain the
exact reservation and request IDs, both dependency projections, aggregate
recomputation tolerance, every chronological journal-event ID, globally unique
request-exact case IDs, complete case/fingerprint coverage, and the semantic
candidate/control accepted-event mapping. A source-only factual contract omits
that task projection; every operational plan containing adapter cases requires it.
Missing/extra dimensions, contexts,
fingerprints or replicates; reused lineage authority; nonfinite or signed-zero
values; metric direction/scale substitution; stale packages; or omitted dependencies
fail loud. Effects, thresholds, winner labels, and promotion state are absent from
the factual report and derived only by the trusted decision reducer.

For a source-base-backed cell and exact replicate ID, the decision reducer computes
`raw = candidate - control`, multiplies by `+1` for maximize or `-1` for minimize,
then divides by the adapter-owned positive comparison scale. It normalizes
mathematical zero to positive zero and rejects nonfinite arithmetic. It never uses
the control value as a denominator, reapplies direction, or aggregates before the
hard gate.

Decision order is fixed:

1. prove exact dimension/adapter/context/replicate coverage and comparability;
2. apply every per-replicate hard-regression bound (`effect < -bound`; equality
   passes), before power or benefit analysis;
3. classify strict `effect > noise_floor` gains, strict `effect < -noise_floor`
   material regressions, and the inclusive interval as ties, using exact comparison
   rather than floating-point closeness;
4. require at least `minimum_replicates_per_cell` in every governed cell. An
   independent evidence unit is one edge between a task-context ID and an
   independence-lineage ID; the dimension must admit a maximum one-to-one matching
   of at least `minimum_distinct_context_lineage_pairs`, so repeated metrics,
   contexts, or lineages cannot manufacture power;
5. call a cell gain-supporting only when every one of its precommitted replicates is
   a strict gain. A dimension is confirmed only when its gain-supporting cells admit
   the same configured context-lineage matching; and
6. approve source-base-backed evidence only with no material regression and at least one
   confirmed dimension. A trusted `MECHANICAL_GENERAL_FIX` attempt is the sole
   exception: once fully powered, complete non-regression is sufficient.

Any hard regression is `FAILED`, even when the rest of the matrix is underpowered.
After the hard gate, underpowered evidence is `PARETO_RETAINED`; a non-hard
gain/regression trade-off or partial/inconsistent gain is also retained. Fully
powered material regression without gain, or fully powered all-tie evidence, is
`FAILED`. Malformed or noncomparable input raises and produces no decision; it is
not converted into a retained candidate.

Bootstrap never fabricates a source base, zero control, delta, or effect. It establishes
the first baseline only when every candidate-only cell meets the replicate minimum
and every dimension meets the same independent context-lineage matching, yielding
`APPROVED` with an explicit standalone-coverage reason; insufficient bootstrap
coverage is `PARETO_RETAINED`. Absolute-quality floors remain upstream evaluator
and reviewer authority until an adapter contract explicitly defines them.

With the current `hard_regression_ratio = 0` and positive noise floor, every
negative effect is a hard failure, including a negative value inside the noise
interval. This strict precedence is intentional and tested.

Only `APPROVED` appends an
accepted `PUBLICATION_ELIGIBILITY` reference; retained/failed candidates preserve
the accepted prefix and cite the decision as terminal evidence. `PARETO_RETAINED`
means retained relative to the named source base and matrix, never membership in a
mutable global frontier and never permission to publish. Approval is also not a
release lease: a later expected-parent GitHub CAS may still lose, requiring a new
rebased/composed candidate and full revalidation.

## Rebase and composition

Composition executes only from sealed terminal approvals. The implemented planning
foundation separates scientific identity from time-sensitive admission:

- `ExpertCompositionBaseReference` is a stable release projection: release, scope,
  source-tree, map, module, semantic-book, and configuration identities, with no
  publisher attestation, branch head, cache receipt, or pointer metadata.
- `ExpertCompositionSourceReference` is the stable candidate/commit/source-base/patch/
  proposed-topology projection. Different historical source bases are intentional: this
  is the input required to rebase an approved stale candidate, not a claim that it
  was built from current source.
- `ExpertCompositionPlan` binds one base and a canonical non-empty source set. It
  deliberately contains neither approval records nor `CURRENT.json` observations,
  so harmless authority refreshes cannot change the scientific plan or successor
  identity.
- `ExpertCompositionAssessment` is a pure, content-addressed classification:
  `CLEAN`, `ALREADY_PRESENT`, `CONFLICTED`, or `REQUIRES_RESTRUCTURE`. Typed conflict
  records form a complete disjoint partition of the plan's sources; their subject
  syntax and structural-versus-nonstructural outcome are derived rather than
  caller-selected.
- `ExpertCompositionBaseClosure` verifies the release/tree/map/module/book joins and
  exact source bytes, but is deliberately not current-release authority.
  `GitHubExpertCompositionBaseProvider` authenticates `CURRENT`, materializes and
  re-verifies the immutable package plus isolated source archive under the cache
  lease, rebuilds the typed closure, observes `CURRENT` again, and only then issues a
  process-local `CurrentExpertCompositionBase`. A newer branch head is harmless only
  when the complete pointer, repository binding, and policy are unchanged; admission
  then derives the base revocation closure from that newer observation rather than
  retaining resolve-time branch metadata.
- The implemented source resolver reopens every exact candidate package and complete
  validation-store history, independently rederives terminal `APPROVED`, and returns
  a process-local capability whose exposed candidate and approval values are defensive
  copies. Serialized IDs alone are never approval authority.
- A later admission fence must own those source capabilities, a sealed current-base
  capability, fresh denylist and adapter-trust observations, an identical
  pre/post-denylist `CURRENT` observation, and the exact security closure. Temporal
  facts authorize one persistence attempt and never enter scientific identity.

The implemented deterministic reducer treats each patch path relative to current source as
follows: current equals `before` means applicable, current equals `after` means
already present, and any third value means conflict. Multiple approved sources that
touch the same path conflict even if their resulting bytes match; automatic merging
must not invent shared provenance. Only capability changes may compose mechanically.
Architecture candidates, topology ambiguity, incompatible module ownership,
dependency cycles, adapter leakage, cross-source capability incompatibility, or an
aggregate source limit require a fresh architect/generalizer proposal. A coding agent
that resolves any conflict also
creates a fresh proposal rather than laundering that judgment through the pure
reducer. A clean materialization embeds both source-base and successor tree manifests,
rederives the exact patch, regenerates every control file and semantic book, and reruns
the shared topology and ownership validators. Module resource-bound maps remain opaque
domain contracts; the generic reducer enforces configured aggregate tree limits rather
than inventing arithmetic across unrelated schemas.

Clean reductions now project to a distinct deterministic-composition candidate
derivation; they never fabricate a coding-agent invocation. The composition plan owns
the active task-binding set and a distinct bounded source count, and every source reference binds its agent derivation,
validation context, and origin principal. The candidate package codec retains exact
commit-checked source packages plus the current source-base bytes. Reopen recomputes the full
replay-evidence union, reconstructs the verified base, reruns the pure reducer, and
requires its assessment, materialization, and successor bytes to match exactly. This
also makes arbitrary sanitized edits and causal-evidence omission fail closed. Sources
are deliberately limited to directly approved agent proposals for now; composed-source
recursion remains rejected until a bounded flattening contract is designed. Generic
store persistence rejects composition candidates; only the sealed admission capability
may cross that side-effect boundary.

Composition admission is now one deterministic coordinator path. It resolves the
authenticated current base and canonical approved-source set itself, unions active task
bindings across every source, builds the plan, reruns the reducer, and admits only
`CLEAN`. It then takes the validation store's shared lease, verifies every terminal
source head, and takes the candidate store's exclusive persistence lease without
releasing the validation lease. The source heads therefore cannot advance while the
candidate store reopens every exact source package, re-verifies every exact source
adapter, observes `CURRENT`, checks the complete transitive subject set against the
authenticated denylist, and observes the identical `CURRENT` again. There is no
validation replay under the candidate lock, so the lock order is acyclic. A
coordinator-sealed exact authority wrapper is bound to one candidate store; the source
resolver separately seals an active process-local validation-lease capability over the
exact source objects. The store requires both capabilities when it mints the one-shot
permit and again after burning that permit under the exclusive lease. An arbitrary
callback cannot register as admission authority, and a permit cannot outlive the
validation lease. Generic persistence remains unable to store compositions.

The atomic candidate directory contains canonical `ADMISSION.json` beside
`COMMITTED.json`. The admission fence binds the computed candidate commit, plan,
materialization, base, terminal source heads, fresh `CURRENT`, adapter observations,
and exact denylist observation. It is deliberately excluded from the scientific
candidate commit and candidate/plan IDs: authority refresh cannot change scientific
identity, while deletion, substitution, or noncanonical bytes make reopen fail loud.
Later validation and publication never reuse this historical fence as live authority;
they pin its identity and complete dependency closure in eligibility/attempt evidence,
rerun the full cascade, and acquire fresh publication authority. Publication's denylist
projection likewise includes the fence and its checked closure so a compromised
admission authority remains transitively revocable.

Before release:

- [x] Define stable base/source references, composition-plan identity, typed conflict,
      and complete deterministic assessment contracts.
- [x] Resolve every source reference through the candidate/validation stores and
      issue process-local approved-source capabilities.
- [x] Resolve the latest stable expert release through M2 and compare it with every
      candidate's source-base commit/tree.
- [x] If the source base moved, deterministically rebase compatible capability effects into
      an exact new tree identity; never patch the old release in place.
- [x] Detect overlapping paths, module-contract conflicts, topology drift, adapter
      leakage, cross-source capability incompatibility, and configured aggregate tree
      limits.
- [x] Project clean reductions into origin-neutral composition candidates, retain
      complete commit-checked source packages, rerun reduction during validation, and
      round-trip the exact package codec without fake agent history.
- [x] Seal the authenticated GitHub `CURRENT` base and consume fresh source/denylist/
      adapter/current observations atomically when persisting the composed candidate.
- [ ] Route conflicts and architecture changes through a fresh architect/generalizer
      proposal rather than treating them as reducer output.
- [ ] Rerun the complete cascade/release matrix on the exact composed tree.
- [ ] Preserve all candidate ancestry/evidence in the release manifest.
- [x] Serialize remote publication through the explicit expected-parent/CAS protocol.

## Release assembly and GitHub publication

`ExpertReleasePublisher`:

1. verifies the exact approved source tree and validation closure;
2. regenerates `EXPERT_REPO.md` from map/contracts and verifies its digest;
3. strips candidate branches, run histories, logs, data, weights, caches, hidden
   evaluation, Git metadata, and task outputs;
4. builds the history-free source archive, dependency lock, release manifest,
   checksums, and test-matrix summary;
5. prepares the exact validated source commit from the expected parent without
   moving the default branch;
6. invokes M2's immutable-release transaction with a sealed expert activation gate;
7. refreshes exact CURRENT, task-adapter, denylist, reservation, intent, identity,
   package, and activation-commit authority; and
8. advances expert `CURRENT.json` only through M2's sole expected-parent CAS.

- [x] A release ID is the source tree plus exact manifest/contract closure, not a
      sequential display label.
- [x] `E000007` is display/version order; launch pins content ID, commit, tag, asset
      IDs/digests, and publication record.
- [x] Publication retry with identical content is idempotent.
- [x] CAS conflict requires re-resolution/revalidation, not force push. A durable
      stale outcome retains an authenticated losing intent/identity and exact
      prepared commit, while the post-CAS success witness prevents historically
      active releases from being misclassified.

Remote activation and local lifecycle completion are deliberately separate crash
boundaries. The implemented recovery verifies the write-once preparation and
distinct post-CAS success witness, authenticates the exact publication identity and
a stable current observation, then persists that witness. Every successor must
witness its exact predecessor before CAS, closing the post-CAS/local-crash gap
without an ancestry horizon. One publisher/store-bound capability atomically writes
the activation receipt, operation, `RELEASED` state, and lifecycle transition while
clearing the reservation. Offline replay returns that first durable outcome. A
witnessed release remains `RELEASED` after any number of successors; a prepared
loser becomes stale only after a witnessed competitor remains stable and the loser
witness is rechecked absent. Commit ancestry is intentionally insufficient.

Release proof edges are now categorized rather than inferred from one
undifferentiated closure. `ExpertBaseReleaseManifest.consumed_dependency_ids` is
the exact taint-propagating source, candidate, validation, and evidence closure;
an ordinary release has no control dependencies. `ExpertReleaseActivationReceipt`
separately persists the consumed release/publication proof and remaining
control-only planned/current evidence after exact overlap removal. Their sets are
exact and disjoint. Emergency
revocation projects only the consumed set, while both sets remain
content-addressed audit evidence. This is the required substrate for clean forward
recovery: a revoked CURRENT predecessor may later be retained as ordering evidence
without being laundered into, or re-tainting, a clean source lineage.

Release lineage is explicit and no longer encoded by an overloaded parent field.
`ExpertReleaseLineage.source_base_release_id` names the immutable bytes and
scientific comparison base; `activation_predecessor_release_id` names the exact
authenticated GitHub `CURRENT` that publication is ordered after. The accepted
publication-eligibility fence freezes the latter before deterministic assembly,
so the manifest is self-contained and can be audited offline. The publication
plan reproduces that exact lineage, binds `CURRENT` and its pointer only to the
activation predecessor, and derives generation from that predecessor. Ordinary
manifests and plans require both IDs to be equal; bootstrap requires both absent.
The old `parent_release_id`/`parent_pointer` release fields are removed rather
than retained as aliases. The same semantic split now runs through candidate,
proposal, composition, validation, task-evaluation, source-replay, and promotion
contracts: `source_base_*` always names immutable scientific inputs;
`expected_current_*` names a planned temporal fence; `observed_current_*` names a
fresh authority observation. Experimental rows and legs use `control_*` and
`source_base_control`, never `parent`, for the comparison arm. Superseded wire keys,
enum values, and package paths are rejected rather than aliased.

Source-replay execution reservation deliberately stores
`expected_current_release_id`: reserving the already prepared request performs no
new GitHub read and therefore cannot claim an observation. Task-evaluation
reservation stores `observed_current_release_id` because its coordinator obtains
and binds the fresh CURRENT observation at that boundary. Ordinary validation
still requires candidate source base, reserved/observed CURRENT, and the final
fresh CURRENT to be equal; clean recovery must enter through its own typed path.

Publication-plan construction is production-owned. The only orchestration entry
point is `ExpertReleasePublisher.reserve(candidate_id, committed_at)`: it reopens
an existing durable reservation before any remote read, otherwise rebuilds the
approved package, sandwiches the exact resolver `CurrentPointerState` between two
authenticated CURRENT observations, and requires all three to equal the accepted
publication-eligibility fence. The bound assembler privately derives generation,
tag, assets, manifest and tree digests, categorized dependencies, and the exact
validation closure. The publisher then performs a publisher-bound private store
CAS, which independently re-derives the plan from the exact package before the
lock/journal commit. Caller-supplied plans, the plan permit, assembler authorization
method, and public store reservation method are deleted. A lost reservation response replays
the first intent and timestamp without GitHub access; CURRENT movement after the
local reservation remains safe because publication preflight, final revalidation,
and expected-head CAS are still mandatory.

Clean recovery will be a dedicated whole-tree rollback-as-forward path, not an
exception in the normal zero-match gate. It separates the independently clean
source/comparison base from the actual activation predecessor. Normal evolution
requires those releases to be equal. Recovery selects the newest separately
authenticated clean historical base, or the canonical empty tree, reruns the full
applicable matrix, publishes a new immutable release, and CAS-advances from the
durably revoked CURRENT release. A typed recovery plan and fresh authority fence
must prove that every consumed subject is clear and that every allowed match is an
exact control-only revocation of the publication barrier. Partial repair from
revoked source bytes remains unsupported until content-level consuming-edge
provenance can prove exclusion.

## Revocation

The authenticated security control plane is implemented as one dedicated
scope-bound repository and three cooperating responsibilities:

1. `SecurityDenylistPublisher` is the only normal security publication entry
   point. Static validation requires canonical supported-version manifest and
   evidence-bundle bytes, exact proof closure, exact scope repository binding, and
   a deterministic generation tag before a remote write. Its activation gate
   authenticates the active predecessor, requires generation-zero bootstrap or
   one adjacent cumulative successor, then re-resolves the same predecessor after
   immutable release/identity publication and immediately before the final
   expected-parent `CURRENT.json` compare-and-swap. The focused transport seals an
   owner-bound authorization capability; the generic publisher rejects absent,
   foreign, or caller-supplied lookalike gates. Publication also rejects a
   generation at or beyond the configured finite lineage horizon.
2. `GitHubSecurityDenylistSnapshotProvider` resolves current or exact historical
   write-once identities through M2, materializes the complete release, and
   rechecks canonical manifest/evidence bytes, repository binding, tag, artifact
   identity, attestation reference, and exact validation closure.
3. `AuthenticatedSecurityDenylistAuthority` makes a live current request for each
   exact bounded subject tuple, authenticates every predecessor needed to reach a
   private local floor (or generation zero on first use), rejects rollback/fork or
   revocation removal, atomically advances the compact checkpoint, and returns the
   exact matched revocation records rather than only lossy subject IDs. Each
   observation preserves the kind, revocation identity, reason, evidence
   identities, and timestamp for the checked-subject intersection. Count and byte
   bounds are enforced before content validation and sorting. Its checkpoint is
   never offline authorization.

The checkpoint store requires an owner-private trusted root, private real
directories and lock/checkpoint files, bounded canonical bytes, per-scope locking,
fsynced staging/replace, and fail-loud corruption handling. A checkpoint stores
only the authenticated floor identity and authority, never cumulative revocation
arrays, so every policy-admitted snapshot remains checkpointable. The process UID
is inside the trust boundary; hostile same-UID code remains outside this
mechanism's threat model.

The expert lifecycle now consumes that authenticated emergency authority through
`ExpertReleaseRevocationCoordinator`. It reopens the exact historical activation,
projects the release's complete proof-consumption closure, performs one fresh
denylist observation without holding the journal lock, and seals an owner-bound
one-shot permit only if at least one security/contamination record matches. The
store compare-and-swaps the unchanged `RELEASED` head and atomically appends the
receipt, operation, `REVOKED` state, and transition. The activation transition is
never rewritten: historical activation reopens as `RELEASED`, while the current
lifecycle state reopens as `REVOKED`. Exact retry is offline and publication retry
fails before touching GitHub.

- [x] Append authenticated security/contamination revocation receipts from the
      signed cumulative emergency lineage.
- [ ] Append separately authenticated performance/compatibility revocation events;
      they must not enter the fail-closed emergency lineage.
- [ ] Performance revocation prevents new launch/promotion and marks existing run
      outputs ineligible while preserving offline reproducibility.
- [x] Security/contamination revocation is checked from the fresh emergency
      denylist before M8 agent execution/evaluation/publication boundaries.
- [ ] Extend the same fresh check to M9 launch and resume.
- [x] Propagate emergency taint through exact module/candidate/release proof
      dependencies and persist the matched root records in the release receipt.
- [x] Separate taint-propagating consumed dependencies from control-only activation
      evidence, with exact disjoint closures and fail-loud replay validation.
- [x] Split release source-base lineage from activation-predecessor ordering while
      retaining the ordinary equality invariant.
- [x] Make release-plan construction and first-writer reservation a bound publisher
      operation with offline durable replay.
- [x] Name immutable scientific inputs, temporal CURRENT authority, and experimental
      controls distinctly across all expert contracts and persisted artifacts.
- [ ] Publish a clean successor/rollback pointer; never move or delete the old
      immutable release as the history mechanism.

## Tests

- Exercise every candidate class and evaluator transition.
- Inject failures at each cascade stage and prove later stages do not execute.
- Verify sealed details never enter agent/reviewer artifacts.
- Test noisy gain, mean gain with hard regression, cost regression, task-specific
  winner, mechanically provable fix, and architecture benefit fixtures.
- Compose disjoint candidates; reject overlapping/conflicting/cyclic candidates.
- Force source-base advancement and require new identity plus full revalidation.
- Build the same approved release twice and require identical archive/manifest IDs.
- Inject release publication/CAS failures and prove old current remains launchable.
- Exercise performance versus security revocation and complete taint closure.

## Definition of done

- Only an exact fully approved tree can become a release.
- No model output or benchmark score can bypass validation/review.
- Concurrent candidates cannot mutate or silently compose into stable state.
- Published source is history-free, reconstructable, and paired with complete
  validation provenance.
- Revocation blocks unsafe future use without destroying reproducibility.

## Non-goals

- Candidate proposal generation (M7).
- Knowledge snapshot publication.
- Live workspace startup (M9).
- Defining task-specific benchmark evaluators beyond their adapter contract.

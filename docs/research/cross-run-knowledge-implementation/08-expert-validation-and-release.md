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
store and re-derives it. A stale parent or unavailable validation facility takes
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
and validation state, observes the current parent release, resolves concrete
verified bundle lineages, materializes the exact parent tree, copies the candidate
tree into an immutable byte closure, and materializes every captured starting
artifact. A lineage provider must retain every root-to-tip `RunBundle` byte
closure; preflight ignores its claimed projection and deterministically rebuilds
every generation from those bytes. Generation-zero, adjacent supersession, and
the reprojected tip are checked again when the runtime-only prepared object is
constructed.

One monotonic deadline and one aggregate entry/byte budget govern the whole
preflight, not each provider call. Every provider receives only the remaining
time and capacity; candidate, parent, historical adapter packages, all retained
bundle generations, and deduplicated contexts debit that same budget. Reused
content IDs count once, while the same ID with different bytes fails loud. Each
case is an explicit matched pair: a parent-control leg and candidate leg share
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
passes only the exact parent or candidate byte closure selected by the allocated
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
`HOSTNAME=kapso-source-replay`; exact container inspection therefore proves the
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
requires zero subsequent registry requests, accepts the exact parent and
candidate scores, publishes the typed source-stage result into the validation
journal, replays the completed stage without another provider bootstrap, and
proves that every handle-owned container, volume, and workspace is gone. The
fixture synthesizes task-adapter, current-release, and denylist authority; it does
not exercise their production GitHub transports.

The task evaluator is a blinded scientific ABI, not a view of validation
authority. Protocol v1 writes one canonical request to
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
and candidate/parent/tree identities remain in trusted Kapso receipts and never
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

Preflight re-observes the current parent after materialization, but its request
is evidence, not an execution lease. Source execution remains fail-closed until
the executor can atomically reserve the unchanged validation head and, immediately
before process spawn, recheck the current parent, candidate/release revocation,
and every historical adapter package's verifier authority, trust, and revocation
state. It must produce exactly one result for each named leg and publish a typed
paired-comparison receipt against the same reservation. If `CURRENT` changes
while an attempt is validating, preflight returns no executable request: it
publishes a typed authority-invalidation record against the observed validation
state ID and returns the resulting terminal state. That content-addressed,
compare-and-swap transition closes the attempt as `FAILED`; the candidate can
no longer execute or be re-enrolled as valid because its immutable manifest pins
the stale parent. Evolution must rebase the change into a new successor candidate
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
allocation. It records a newly fetched parent `CURRENT`, freshly reverified
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
transition/state/attempt, candidate, and observed parent. GitHub `CURRENT`,
historical adapter re-verification, and the live security denylist are checked
outside the validation lock. A second identical reopen after those external
checks closes validation-head races before the local spawn boundary is written.
The store lock is global, so it never encloses archive verification, network
access, an execution-journal lock, a callback, workspace work, or provider start.
Enrollment, evaluator-result publication, reservation admission, and parent
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
dependency closure, parent-release and candidate dependencies, adapter proofs,
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
it to rerun all byte, lineage, context, artifact, adapter, parent, candidate, and
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
must perform fresh parent, revocation, and verifier observations immediately
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
- a complete journal can mint only one deterministic store/process-bound completed
  capability, from which the factual reducer produces a canonical paired receipt
  with semantic control/candidate assignment, adapter-declared dimension/scale,
  direction-aligned and normalized effects, accepted-event lineage, and exact
  expanded dependency closure but no scientific or promotion decision; and
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
  The coordinator constructs the configured CLI runner from the authorized
  workspace root and rejects runner, configured-path, prompt-budget, or sealed
  artifact-budget substitution. Deterministic operation identities make paid calls
  restart-safe; exact agent artifacts, receipts, assertions, and
  unanimous/rejected/disputed adjudication are persisted before one packet-keyed
  journal CAS. Restart re-derives the full closure, accepted reviews advance only
  to the release matrix, and rejected or conflicting reviews terminate without
  appending a passed-stage reference; and
- parent-authority invalidation is a content-addressed terminal transition that
  preserves accepted-stage history, proves expected versus observed `CURRENT`,
  and makes stale attempts recoverable without accepting their remaining work.

The Pareto promotion-decision, composition, and release paths remain separate
later slices; automated review cannot synthesize their authority.

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
- [ ] Use the configured Pareto dimensions and hard regression bounds across
      quality, robustness, cost, portability, reproducibility, and security.
- [ ] Treat noise-floor gains as inconclusive until configured repeat evidence
      exists.

Stage applicability is deterministic. Bootstrap omits replay, anchors, transfer,
and canary because no parent evidence exists. Mechanical fixes use deterministic
gates, fresh-task smoke, source replay when a parent exists, review, release
matrix, and publication. Behavioral changes additionally require development
anchors, cross-family transfer when more than one family is bound, and a sealed
canary. Architecture changes use the release-wide path and require a canary only
when the typed policy says so. An unavailable required canary makes the candidate
ineligible; it is never treated as a skipped or passed stage.

## Automated review and decision

- [x] Accept reviewer assertions only from configured autonomous identities/roles.
- [x] Require exact candidate, evidence, evaluator-run, rubric, and parent-release
      references.
- [x] Preserve conflicting reviews as disputed; do not overwrite by time.
- [x] A separate coding-agent/service role reviews each proposal; the proposing
      invocation cannot review its own output or transition state.
- [ ] Supported task-specific improvements remain knowledge/task-adapter candidates,
      never expert core.
- [ ] Failed or non-dominated candidates stay immutable in the candidate archive;
      they are not installed into runs.

Promotion states are explicit: `ineligible`, `validating`, `failed`, `disputed`,
`pareto_retained`, `approved`, `released`, and `revoked` as frozen by M1. There is
no implicit promotion from a direct Git commit; only a completely validated
immutable release referenced by `CURRENT.json` is active.

## Rebase and composition

Before release:

- [ ] Resolve the latest stable expert release through M2 and compare it with every
      candidate's parent commit/tree.
- [ ] If the parent moved, rebase or compose into a new candidate tree with a new
      identity; never patch the old release in place.
- [ ] Detect overlapping paths, module-contract conflicts, capability-lineage
      conflicts, dependency cycles, adapter leakage, and incompatible resource
      bounds.
- [ ] Rerun the complete cascade/release matrix on the exact composed tree.
- [ ] Preserve all candidate ancestry/evidence in the release manifest.
- [ ] Serialize final publication through the explicit expected-parent/CAS protocol.

## Release assembly and GitHub publication

`ExpertReleasePublisher`:

1. verifies the exact approved source tree and validation closure;
2. regenerates `EXPERT_REPO.md` from map/contracts and verifies its digest;
3. strips candidate branches, run histories, logs, data, weights, caches, hidden
   evaluation, Git metadata, and task outputs;
4. builds the history-free source archive, dependency lock, release manifest,
   checksums, and test-matrix summary;
5. commits the exact validated source directly from the expected parent;
6. invokes M2's immutable-release transaction; and
7. advances expert `CURRENT.json` only after immutable publication verifies.

- [ ] A release ID is the source tree plus exact manifest/contract closure, not a
      sequential display label.
- [ ] `E000007` is display/version order; launch pins content ID, commit, tag, asset
      IDs/digests, and publication record.
- [ ] Publication retry with identical content is idempotent.
- [ ] CAS conflict requires re-resolution/revalidation, not force push.

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
   exact denial intersection. Count and byte bounds are enforced before content
   validation and sorting. Its checkpoint is never offline authorization.

The checkpoint store requires an owner-private trusted root, private real
directories and lock/checkpoint files, bounded canonical bytes, per-scope locking,
fsynced staging/replace, and fail-loud corruption handling. A checkpoint stores
only the authenticated floor identity and authority, never cumulative revocation
arrays, so every policy-admitted snapshot remains checkpointable. The process UID
is inside the trust boundary; hostile same-UID code remains outside this
mechanism's threat model.

- [ ] Append signed performance, security, contamination, or compatibility
      revocation events.
- [ ] Performance revocation prevents new launch/promotion and marks existing run
      outputs ineligible while preserving offline reproducibility.
- [ ] Security/contamination revocation enters the fresh emergency denylist checked
      at launch, resume, before agent execution/evaluation/publication.
- [ ] Propagate taint through module/candidate/release evidence dependencies.
- [ ] Publish a clean successor/rollback pointer; never move or delete the old
      immutable release as the history mechanism.

## Tests

- Exercise every candidate class and evaluator transition.
- Inject failures at each cascade stage and prove later stages do not execute.
- Verify sealed details never enter agent/reviewer artifacts.
- Test noisy gain, mean gain with hard regression, cost regression, task-specific
  winner, mechanically provable fix, and architecture benefit fixtures.
- Compose disjoint candidates; reject overlapping/conflicting/cyclic candidates.
- Force parent advancement and require new identity plus full revalidation.
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

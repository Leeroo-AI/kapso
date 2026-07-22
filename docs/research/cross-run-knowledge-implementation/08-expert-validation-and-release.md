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
process, open-file, writable-store, output, stream, and accelerator limits; and a
deterministically counterbalanced two-leg order. Its content ID is part of the
case dependency closure and matched-compute digest. Both legs therefore receive
the same authorized allocation, but their observed duration and consumption may
differ and belong only in execution receipts. Scientific repeats remain solely
the exact evaluation fingerprints: the executor runs each named leg once and may
not add an observation-dependent retry. The sandbox policy is structural, not a
set of caller-controlled booleans: its version must dispatch to an implementation
that guarantees offline direct execution, read-only inputs, fresh private
writable roots, and a fixed non-secret environment.

Provider selection is one exact composite dispatch over the paired-execution
protocol, execution-provider ID and version, sandbox-policy version, historical
adapter runtime-protocol version, and historical task-evaluator protocol version.
The registry pre-resolves every case in the aggregate prepared request before
reservation or filesystem work and fails if any complete key is absent. There
are no wildcards, aliases, compatible-version ranges, per-field lookup, or
defaults. Image/platform identities and resource ceilings remain exact case
inputs checked by the selected provider; they do not select an implementation.
The resolved provider advertises the same full key again immediately before
execution so registry mutation or provider substitution cannot bypass dispatch.

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
future paired reducer, not the adapter, applies metric direction, comparison
policy, and validation outcome.

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
the caller cannot select work. The first durable boundary allocates one 128-bit
CSPRNG nonce to the exact reservation/case/leg and reuses it after restart.
Publication fsyncs a private staging file, renames it atomically without
replacement, and fsyncs both directories. Unsafe modes, links, unexpected
entries, noncanonical bytes, forks, gaps, or identity substitutions fail loud. A
later spawn event is the at-most-once boundary: an allocation-only tail is
resumable, while a reopened spawn-marker tail is permanently interrupted and can
be cleaned up but never executed again.

The executor reopens a reservation through a public read-only store boundary.
Prepared byte authority is reconstructed before the lock; one short shared-lock
read then requires the exact journal-bound reservation, stored request, current
transition/state/attempt, candidate, and observed parent. GitHub `CURRENT`,
historical adapter re-verification, and the live security denylist are checked
outside the validation lock. A second identical reopen after those external
checks closes validation-head races before the local spawn boundary is written.
The store lock is global, so it never encloses archive verification, network
access, an execution-journal lock, a callback, workspace work, or provider start.
No local lock can make GitHub, a denylist, and process creation transactional;
safety instead comes from the double reopen, the durable at-most-once marker,
and a final fresh-authority plus validation-head CAS before accepting receipts.

The reservation API accepts only the runtime-only prepared closure, reconstructs
it to rerun all byte, lineage, context, artifact, adapter, parent, candidate, and
aggregate-budget invariants, independently re-derives every compute binding from
the persisted settings, and then persists its request. A self-consistent
content contract without those prepared authorities is not executable admission.

Reservation admission reopens the candidate, rechecks `CURRENT`, re-resolves
every historical adapter package through its retained trusted verifier, and
revalidates the accepted evaluator prefix before binding the operation. A
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
- executable-stage results are bounded before identity, signature-verified through
  an injected fail-closed verifier, and accepted only in exact prefix order; and
- retry lineage preserves both the current state and latest historical attempt,
  and restarts at stage one. An intervening ineligible state cannot reset attempt
  identity; approved, released, validating, and revoked states cannot be retried.
- durable validation history stores immutable content-addressed decisions,
  configurations, attempts, signed evaluator-result envelopes, states, operations,
  and transitions behind one atomic per-candidate journal;
- operation-to-transition bindings make lost-response retries exact, while the
  journal head provides compare-and-swap publication with no fork, merge, or
  rollback behavior;
- immutable source-replay reservation aliases admit exactly one byte-closed
  prepared execution request without changing the validation head and replay
  exactly across process/store recovery;
- the local source-replay execution journal durably allocates the first exact
  scheduled invocation once, survives restart, serializes concurrent reservation
  sessions, and rejects corrupt, forked, substituted, or unsafe journal state;
  spawn and result events extend the same create-only prefix rather than adding a
  mutable execution snapshot; and
- parent-authority invalidation is a content-addressed terminal transition that
  preserves accepted-stage history, proves expected versus observed `CURRENT`,
  and makes stale attempts recoverable without accepting their remaining work.

The automated-review, promotion-decision, composition, and release paths remain
separate later slices; the executable reducer and store cannot synthesize their
authority.

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
  executable path plus a sorted allowlist of exact protected evaluation-tree
  fingerprints. The package verifier attests that compatibility claim; the
  protocol fixes request/result layout and direct, no-shell invocation.
- `TaskAdapterContextBinding`: a sorted allowlist of transfer-dimension IDs the
  evaluator consumes. It may be empty, must be a subset of the exact scope schema,
  and every replay context must contain the declared dimensions.
- `TaskAdapterRuntimeContract`: runtime protocol, normalized registry-qualified
  image repository, platform-manifest and image-config digests,
  dependency-lock path/digest, operating system, architecture, and optional OCI
  variant. The derived `repository@manifest-digest` reference is the sole
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

- [ ] Accept reviewer assertions only from configured autonomous identities/roles.
- [ ] Require exact candidate, evidence, evaluator-run, rubric, and parent-release
      references.
- [ ] Preserve conflicting reviews as disputed; do not overwrite by time.
- [ ] A separate coding-agent/service role reviews each proposal; the proposing
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

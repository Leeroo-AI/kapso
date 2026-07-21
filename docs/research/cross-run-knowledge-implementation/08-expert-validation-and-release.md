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

Implemented validation substrate:

- enrollment reopens the exact M7 candidate, resolves the current release through
  a GitHub reader that verifies repository policy and the observed immutable
  release identity, and resolves every trigger binding through the provider's
  trusted active adapter index rather than accepting caller-selected manifests;
- validation track and stage plan are recomputed from trusted candidate records;
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
  and transitions behind one atomic per-candidate journal; and
- operation-to-transition bindings make lost-response retries exact, while the
  journal head provides compare-and-swap publication with no fork, merge, or
  rollback behavior.

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

# M4 — catalog, episodes, prior ideas, reviews, and claims

Status: **implemented; authenticated live Codex validation blocked by account quota**

Parent plan: [`00-orchestrator-plan.md`](00-orchestrator-plan.md)

Depends on: M1 and M3.

## Objective

Project sanitized bundles into immutable cross-run evidence, preserve independent
review and interpretation, and publish deterministic catalog generations. This is
the scientific-memory authority; it never mutates local run stores and never
promotes expert code.

## Owned responsibilities

- Disjoint `TransferEpisode`/`PriorIdea` projection.
- Bundle/capture supersession and attempt deduplication.
- Append-only `ReviewAssertion` registry.
- Structured coding-agent `ClaimProposer` with exact evidence provenance.
- Deterministic admission, dispute, supersession, taint, and revocation closure.
- Grow-only immutable catalog facts, reduced generations, and rebaseable deltas.

## Proposed code surface

```text
src/kapso/cross_run/catalog/
  __init__.py
  agent_operations.py
  store.py
  projector.py
  assertions.py
  claims.py
  admission.py
  reviews.py
  reducer.py
  service.py

src/kapso/cross_run/prompts/
  claim_proposer.md
  catalog_reviewer.md

tests/
  test_cross_run_catalog_store.py
  test_cross_run_episode_projector.py
  test_cross_run_assertions.py
  test_cross_run_claim_proposer.py
  test_cross_run_catalog_reviewer.py
  test_cross_run_catalog_service.py
  test_cross_run_admission.py
  cross_run_catalog_agents_live.py
```

## Catalog store

- [x] Store immutable content objects and additive `CatalogInputDelta` facts; only
      the `current` generation pointer is mutable.
- [x] Make `CatalogGenerationManifest` name its parent, exact complete object
      closure, applied input deltas, latest valid bundle frontier, active subject
      states, configuration fingerprint, and generation number.
- [x] Make `CatalogDeltaManifest` describe one exact base/target transition without
      participating in the target identity preimage.
- [x] Compare-and-swap the pointer under a process lock using expected generation
      number and ID; stage, fsync, rename, and fsync the directory.
- [x] On a stale CAS, reload the winning generation, union additive facts, rerun the
      deterministic reducer, and retry. Never merge precomputed entry states.
- [x] Make identical operation replay idempotent, conflicting operation reuse fail,
      and every immutable object write verify existing bytes byte-for-byte.
- [x] Keep attestation envelopes separate from scientific payload IDs.
- [x] Preserve exact bundle, assertion, revocation, supersession, taint, operation,
      and derivation closure within one scope-contract lineage.
- [x] Emit a canonical reviewable delta consumed by M2 publication.

## Deterministic projection

For each latest admitted capture frontier:

```text
source idea linked to any node/revision -> one TransferEpisode
source idea never linked to a node      -> one PriorIdea
```

- [x] Join bundle archive, experiment projection, journal, checkpoint, and branch
      refs by exact qualified identity.
- [x] Validate the bundle manifest, capture descriptor, sanitation report, every
      checksum, and the admitted supersession frontier before projection.
- [x] Fold all zero-based journal revisions of one idea/node into one ordered
      episode attempt list; retain every evaluator fingerprint used historically.
- [x] Distinguish completed/failed/interrupted execution, valid/invalid/partial
      evaluation, and comparable/incomparable/inconclusive comparison.
- [x] Compute a relative effect only from real comparable parent measurements.
- [x] Preserve exact per-attempt branch artifacts and confounders; mark intervention
      structure `undetermined` unless explicit source evidence proves otherwise.
- [x] Project only never-linked rejected/deferred/unexecuted ideas as `PriorIdea`.
- [x] Use latest admitted bundle supersession frontier so periodic/final captures do
      not double-count observations.
- [x] Let a later projection supersede either prior-idea or episode payload for the
      same globally qualified source idea; never concatenate attempt prefixes.
- [x] Mint no lesson, mechanism, applicability rule, or success label beyond the
      deterministic structured outcome.

## Review assertions

- [x] Assign reviewer identity/role/rubric from the configured slot and bind each
      assertion to an immutable structured-call operation receipt; never trust
      identity returned by the model.
- [x] Require exact subject/evidence refs, judgment, rationale, and
      optional same-reviewer superseded assertion.
- [x] Preserve conflicting assertions; never overwrite by recency.
- [x] Count at most one active current-rubric vote per distinct trusted principal;
      proposer and reviewer principals must differ.
- [x] Resolve configured approval/rejection quorums mechanically; mixed quorum is
      disputed, neither quorum is inconclusive/quarantined, and stale rubrics remain
      historical without counting.
- [x] Make disputed evidence ineligible for exploit anchoring and expert promotion.
- [x] Propagate later revocation/taint to every dependent catalog subject.

Reviewer assertions come from a separately configured autonomous coding-agent role
or review service. The pipeline never pauses for human approval, and the
claim-proposing invocation cannot review or admit its own output.

## Claim proposer

`ClaimProposer` uses the shared durable `CodingAgentCallRunner`. The shipped role
uses Codex; both the role and its reviewer slots are fully typed config. Reviewers
use `gpt-5.6-sol` at `xhigh` reasoning and never invoke Claude Code.

Input packet:

```text
pinned scope contract
complete selected episodes and prior ideas
existing claim revisions
supporting and contradicting evidence sets
review assertions and trust state
required strict output schema
```

Tasks:

- [x] Build packets by explicit complete-record selection; never truncate model-bound
      episode, idea, claim, rationale, or user content.
- [x] Derive the operation ID from the complete packet, template, schema, and agent
      settings. Persist prompt, schema, invocation, stdout/stderr, final response,
      and cryptographic artifact digests.
- [x] Require mechanism, applicability predicates, exclusions, exact support, and
      exact contradiction references.
- [x] Validate every returned reference and registered context predicate.
- [x] Exclude claim IDs, state, admission, reviews, attestations, and arbitrary
      provenance from model output. Framework code mints claim lineage/revision IDs;
      only `CatalogEntryState` owns trust state.
- [x] Preserve malformed or semantically invalid call artifacts outside the active
      catalog and propagate the operation failure.
- [x] Perform no direct chat/completions/responses call.

Each claim proposal and review also emits a `CatalogAgentOperationRecord`. It
binds the exact packet/template/schema/agent preimage, exact `final.json` bytes,
receipt, and framework-minted object IDs. Reduction reparses the authenticated
output and rejects receipt replay against altered claims, evidence closures, or
assertions. `ClaimEvidenceClosure` preserves every episode assessment—including
`not_applicable` rationales—and reviewers receive the complete evaluated universe.
Admission independently requires every review of a claim revision to reference
that complete universe; an authenticated review over a hidden subset is rejected.
Reduction reconstructs the typed packet, verifies all nested records against their
catalog bytes, and accepts only the shipped prompt template and response schema.
The operation's complete secret-free catalog configuration is bound to the input
delta that published it, so historical model/effort/rubric settings remain valid
after rotation without granting stale reviews current quorum authority.
First publication also binds the packet to the exact parent generation, its fact
membership, and active state mapping. Prompt/schema equality is checked at that
publication boundary; later reductions trust the already-validated immutable
historical bytes rather than comparing them with a newer checkout.

## Admission and revocation

- [x] Express admission rules as deterministic policy over sanitation, provenance,
      assertion, comparability, lineage-diversity, and scope state.
- [x] Never infer trust from score magnitude, recency, similarity, or model prose.
- [x] Publish new immutable claim revisions when evidence or applicability changes;
      state-only changes create entry states, never claim revisions.
- [x] Close selected claims over supporting/contradicting episodes and assertions.
- [x] Require admitted, untainted, isolated, comparable supporting episodes from the
      configured minimum independent runs/task contexts before support admission.
- [x] Append typed revocation and taint events; revoking a contradiction taints its
      dependent claim instead of strengthening it. Never delete payload identity.
- [x] Apply state precedence `revoked/tainted > superseded > disputed > admitted or
      quarantined` and recompute transitive proof/derivation taint to a fixed point.
- [x] Reject any catalog generation with an incomplete dependency closure.

The current capture authority has no typed ablation/isolation fact, so the
production projector deliberately emits `undetermined`. With isolated support
required, projected episodes are still admitted and retrievable, while causal
mechanism claims remain quarantined. A later module may add a separately validated
identification fact; neither diffs nor model prose may upgrade this field.

Successor projections may reuse historical immutable derivation-event IDs. The
reducer requires exact closure per manifest and rejects orphan events, but does not
misclassify legitimate repeated references as duplicate evidence. Unknown fact
namespaces fail reduction rather than entering proof closure without validation.

Review packets carry only the active assertion head per configured reviewer.
Superseded history remains in the grow-only catalog, while bounded re-review does
not deadlock as assertion history grows; the configured record limit applies to
complete scientific evidence records, never by truncating one.

## Tests

- Project positive, negative, technical-failure, interrupted, invalid-evaluation,
  partial, and unmeasured-baseline cases.
- Prove projection disjointness for every source idea state.
- Prove multiple revisions/captures cannot become independent evidence.
- Shuffle input ordering and require byte-identical generation output.
- Exercise assertion conflict, supersession, unauthenticated reviewer, and stale
  rubric cases.
- Run deterministic fake Codex claim/review results through strict validation.
- Reject hallucinated evidence IDs, unregistered predicates, missing exclusions,
  omitted contradictions, and self-supporting claims.
- Propagate contamination through episode -> claim -> candidate dependency fixtures.
- Race OS processes on one expected generation, inject crashes at each atomic write,
  and prove stale deltas are rebased by fact union plus full reduction.
- Project real PostTrain-shaped and RelBench-shaped M3 bundles without core domain
  conditionals.
- Invoke the authenticated Codex CLI for one real proposal and two independent
  reviews, prove cache replay executes once, and verify the source workspace remains
  byte-identical. The executable production harness is
  `tests/cross_run_catalog_agents_live.py`; the 2026-07-21 attempt reached the
  authenticated Codex CLI and was rejected by its usage limit (reset reported for
  2026-07-25 11:05 UTC), so this final external validation remains pending.

## Definition of done

- Sanitized bundles deterministically project into auditable cross-run evidence.
- Coding-agent interpretation is attributable and cannot admit itself.
- Catalog generations preserve complete review/revocation/taint history.
- No local run authority is opened for write or merged into the catalog.
- M5 can build a snapshot from one exact generation without inspecting raw runs.

## Non-goals

- Embeddings, semantic search, or prompt packet ranking.
- Expert candidate generation.
- GitHub transport implementation.
- Editing current-run evidence or policy state.

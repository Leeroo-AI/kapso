# M4 — catalog, episodes, prior ideas, reviews, and claims

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
- Codex/Claude `ClaimProposer` with exact evidence provenance.
- Deterministic admission, dispute, supersession, taint, and revocation closure.
- Immutable `CrossRunCatalog` generations and candidate deltas.

## Proposed code surface

```text
src/kapso/cross_run/catalog/
  __init__.py
  store.py
  projector.py
  assertions.py
  claims.py
  admission.py

src/kapso/cross_run/prompts/
  claim_proposer.md

tests/
  test_cross_run_catalog_store.py
  test_cross_run_episode_projector.py
  test_cross_run_assertions.py
  test_cross_run_claim_proposer.py
  test_cross_run_admission.py
```

## Catalog store

- [ ] Persist ordered immutable generations containing content IDs and separate
      `CatalogEntryState` records.
- [ ] Accept an expected generation for every mutation/publication.
- [ ] Use canonical total ordering ending in content ID.
- [ ] Make identical replay idempotent and conflicting replay fail.
- [ ] Keep object payloads immutable; new review/state publishes new assertions or
      catalog generations rather than rewriting evidence.
- [ ] Preserve exact bundle, assertion, revocation, supersession, and taint closure.
- [ ] Validate that every referenced object belongs to the configured scope lineage.
- [ ] Emit a reviewable catalog delta for M2 publication.

## Deterministic projection

For each latest admitted capture frontier:

```text
source idea linked to any node/revision -> one TransferEpisode
source idea never linked to a node      -> one PriorIdea
```

- [ ] Join bundle archive, experiment projection, journal, checkpoint, and branch
      refs by exact qualified identity.
- [ ] Fold all revisions of one idea/node into one ordered episode attempt list.
- [ ] Distinguish completed/failed/interrupted execution, valid/invalid/partial
      evaluation, and comparable/incomparable/inconclusive comparison.
- [ ] Compute a relative effect only from real comparable parent measurements.
- [ ] Preserve coupled-intervention status and confounders.
- [ ] Project only never-linked rejected/deferred/unexecuted ideas as `PriorIdea`.
- [ ] Use latest admitted bundle supersession frontier so periodic/final captures do
      not double-count observations.
- [ ] Mint no lesson, mechanism, applicability rule, or success label beyond the
      deterministic structured outcome.

## Review assertions

- [ ] Validate reviewer identity/role against configured trust roots.
- [ ] Require rubric version, exact subject/evidence refs, judgment, rationale,
      timestamp, attestation, and optional superseded assertion.
- [ ] Preserve conflicting assertions; never overwrite by recency.
- [ ] Resolve configured adjudication mechanically.
- [ ] Derive `disputed`/`inconclusive` when required agreement is absent.
- [ ] Make disputed evidence ineligible for exploit anchoring and expert promotion.
- [ ] Propagate later revocation/taint to every dependent catalog subject.

Reviewer assertions may come from authorized humans or a separately configured
review service. The claim-proposing coding-agent invocation cannot review or admit
its own output.

## Claim proposer

`ClaimProposer` reuses the ideation `CodingAgentCallRunner` contract and supports
configured Codex or Claude Code roles.

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

- [ ] Build packets by complete-record selection; never truncate model-bound
      episode, idea, claim, rationale, or user content.
- [ ] Persist prompt, schema, CLI/model/effort/tool configuration, full response,
      and structured proposal provenance.
- [ ] Require mechanism, applicability predicates, exclusions, exact support, and
      exact contradiction references.
- [ ] Validate every returned reference and registered context predicate.
- [ ] Reject self-certified states; model output enters only as `proposed` or a
      policy-allowed provisional revision.
- [ ] Preserve malformed or semantically invalid call artifacts outside the active
      catalog and propagate the operation failure.
- [ ] Perform no direct chat/completions/responses call.

## Admission and revocation

- [ ] Express admission rules as deterministic policy over sanitation, provenance,
      assertion, comparability, lineage-diversity, and scope state.
- [ ] Never infer trust from score magnitude, recency, similarity, or model prose.
- [ ] Publish new immutable claim revisions when evidence/applicability/state
      changes.
- [ ] Close selected claims over supporting/contradicting episodes and assertions.
- [ ] Append revocation and taint events; never delete historical payload identity.
- [ ] Reject any catalog generation with an incomplete dependency closure.

## Tests

- Project positive, negative, technical-failure, interrupted, invalid-evaluation,
  partial, and unmeasured-baseline cases.
- Prove projection disjointness for every source idea state.
- Prove multiple revisions/captures cannot become independent evidence.
- Shuffle input ordering and require byte-identical generation output.
- Exercise assertion conflict, supersession, unauthenticated reviewer, and stale
  rubric cases.
- Run deterministic fake Codex/Claude claim results through strict validation.
- Reject hallucinated evidence IDs, unregistered predicates, missing exclusions,
  omitted contradictions, and self-supporting claims.
- Propagate contamination through episode -> claim -> candidate dependency fixtures.
- Simulate two catalog deltas from one generation and deterministic union after CAS
  conflict.

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

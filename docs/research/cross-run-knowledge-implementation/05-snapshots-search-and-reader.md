# M5 — knowledge snapshots, portable search, and the reader gate

Parent plan: [`00-orchestrator-plan.md`](00-orchestrator-plan.md)

Depends on: M1, M2, and M4.

## Objective

Publish one immutable, independently readable scientific-memory release from an
exact catalog generation, then expose deterministic structured/lexical/semantic
retrieval through a read-only local boundary. Historical task workspaces and raw
traces are not runtime dependencies.

## Owned responsibilities

- Snapshot admission/proof closure and canonical package assembly.
- Extraction of one shared OpenAI embedding provider boundary.
- Portable metadata, lexical, vector, and optional ANN sidecars.
- Compatibility-first hybrid retrieval and deterministic packet budgets.
- `PriorKnowledgeSnapshot` persistence/verification.
- `PriorKnowledgeGate` MCP tools over a pinned packet/materialization.
- Immutable GitHub knowledge release and `CURRENT.json` advancement via M2.

## Proposed code surface

```text
src/kapso/cross_run/knowledge/
  __init__.py
  package.py
  index.py
  retrieval.py
  publisher.py
  access.py

src/kapso/core/
  embeddings.py

src/kapso/gated_mcp/gates/
  prior_knowledge_gate.py

src/kapso/gated_mcp/
  presets.py
  server.py

tests/
  test_knowledge_snapshot_package.py
  test_cross_run_index.py
  test_cross_run_retrieval.py
  test_prior_knowledge_gate.py
  test_knowledge_snapshot_publisher.py
  test_shared_embeddings.py
```

## Shared embedding boundary

- [ ] Move the current isolated OpenAI embedding implementation to one shared
      `kapso.core.embeddings` module and update existing ideation callers directly.
- [ ] Delete the superseded provider implementation; retain no alias/import shim.
- [ ] Keep the OpenAI import at module top in the feature module and preserve
      provider-SDK default credential discovery.
- [ ] Ensure coding-agent and MCP subprocess environments receive no embedding
      credential.
- [ ] Key every vector by provider, model, dimensions, canonicalizer version, and
      complete input hash (`EmbeddingSpaceId`).
- [ ] Batch calls under config-owned limits without truncating canonical source
      text. Window complete calls if provider limits require it.
- [ ] Attribute embedding cost/latency separately from coding-agent telemetry.
- [ ] Propagate missing credentials, provider errors, dimension mismatch, and
      malformed responses. Explicit `enabled: false` is the only no-embedding mode.

## Snapshot package

From one exact catalog generation, `KnowledgeSnapshotPublisher`:

- [ ] Selects admitted, non-revoked objects under the configured scope/retrieval
      policy.
- [ ] Includes every claim/relative-effect proof dependency, assertion, active
      state, revocation, and sanitation reference required for audit.
- [ ] Includes complete canonical JSON records; IDs-only placeholders are not
      sufficient for runtime retrieval.
- [ ] Includes the pinned scope contract and policy identities.
- [ ] Builds a deterministic file order, archive metadata, checksums, manifest,
      and snapshot content ID.
- [ ] Verifies that extraction of the package recreates the declared record and
      proof closure byte-for-byte.
- [ ] Packages raw sanitized audit deltas separately; the runtime package contains
      safe normalized knowledge only.
- [ ] Deterministically shards release assets only at the config-owned size bound.

An explicit `EMPTY` snapshot is built and validated through the same path; missing
remote state is never interpreted as empty.

## Search sidecars

Canonical JSON is truth. Search artifacts are rebuildable:

```text
metadata index
lexical index
vectors/<EmbeddingSpaceId>/ids
vectors/<EmbeddingSpaceId>/float32 data
optional ANN index
index-manifest.json
```

- [ ] Index scope/task family, context dimensions, trust/state, outcome,
      evaluation identity, lineage, timestamps, mechanism, applicability,
      exclusions, and record type.
- [ ] Provide exact-term/identifier lexical search alongside semantic vectors.
- [ ] Use exact cosine over compact vectors as the initial implementation.
- [ ] Build an ANN sidecar only after the configured corpus/latency threshold; it
      may generate candidates but cannot change canonical filters or final ordering.
- [ ] Pin sidecars to the exact snapshot record closure and embedding space.
- [ ] Reject stale/corrupt/mismatched sidecars. Rebuilding creates new assets and a
      new snapshot publication, not an in-place mutation.
- [ ] Do not put indexes or growing databases in Git history.

## Retrieval pipeline

`CrossRunRetriever` accepts one pinned snapshot plus:

```text
task context binding
problem/objective
current local gaps
ideation or consumer directive
configured outcome/diversity/byte budgets
```

It performs, in order:

1. authorization, revocation, taint, and trust filtering;
2. scope/task-family and registered context compatibility classification;
3. evaluation-fingerprint comparability checks where effects are requested;
4. lexical and semantic candidate rank within compatible tiers;
5. evidence quality, retrieval utility, and recency rank;
6. diversity caps by run, lineage, approach family, outcome, and record type;
7. proof-closure expansion; and
8. complete-record packet admission under configured record/byte budgets.

Rules:

- [ ] Semantic similarity never determines truth, sign, novelty, or admission.
- [ ] Exact-context records rank separately from analogies; incompatible records
      are absent.
- [ ] Positive, negative, inconclusive, and frontier slots remain explicit.
- [ ] A top-level record whose proof closure cannot fit is skipped as a whole.
- [ ] Deterministic total-order tie breaking ends in content ID.
- [ ] The resulting `PriorKnowledgeSnapshot` includes query/policy/source identity,
      exact records, proofs, and digest.
- [ ] Repeating the same query over the same pin produces byte-identical output.

## MCP reader

`PriorKnowledgeGate` is a local read-only MCP gate with the minimum v1 tools:

```text
list_prior_knowledge()
get_prior_knowledge_record(record_id)
```

- [ ] The gate receives an explicit packet/materialization path from the trusted
      launcher; it never resolves GitHub or `CURRENT`.
- [ ] Validate the packet digest and record/proof membership before serving.
- [ ] Permit only IDs present in the persisted packet for live ideation v1.
- [ ] Return complete schema-rendered records, not clipped display summaries.
- [ ] Treat record prose/code as untrusted data with explicit provenance labels.
- [ ] Log exact tool call and returned IDs in coding-agent invocation artifacts.
- [ ] Mount no write, network, raw artifact, or current-run memory authority.

An interactive full-snapshot `search_prior_knowledge` tool is explicitly deferred.
It requires a durable access-session schema that seals every query/response into
the consuming operation before selection; it must not be added as an unlogged
shortcut.

## Knowledge release publication

- [ ] Submit the reviewable snapshot manifest/catalog delta through M2's candidate
      PR boundary.
- [ ] Required CI rebuilds canonical package and search sidecars from the merged
      exact catalog generation.
- [ ] Compare built IDs/digests with the proposed manifest.
- [ ] Use M2's stable transaction to publish the immutable knowledge release, then
      CAS `CURRENT.json`.
- [ ] On CAS conflict, reload the catalog base, rebuild the deterministic union,
      rerun checks, and publish a new identity.

## Tests

- Build the same snapshot under shuffled filesystem/catalog ordering and require
  identical bytes and IDs.
- Prove a clean temp directory can retrieve from only the materialized package.
- Verify complete proof closure and reject every missing dependency.
- Test exact/analogical/incompatible contexts and evaluator mismatch.
- Test positive/negative/inconclusive/frontier diversity and whole-record budgets.
- Test embedding-space separation, stale sidecars, corrupt float data, and explicit
  embedding disablement.
- Compare exact-cosine and ANN candidate modes for identical final policy output on
  fixed fixtures.
- Test MCP membership, complete rendering, injection-shaped content, and no write
  or GitHub access.
- Inject publication/rebuild/CAS failures and verify old `CURRENT` remains valid.

## Definition of done

- A snapshot is self-contained for normal retrieval and proof inspection.
- Search is local, deterministic, compatibility-first, and scale-ready.
- Ideation/claim/expert consumers can use one reader boundary without GitHub
  credentials.
- Canonical knowledge remains usable after deleting every search sidecar and
  rebuilding it.
- One immutable GitHub knowledge release can be resolved by M2 and read offline.

## Non-goals

- Live ideation lifecycle changes (M6).
- Claim proposal/admission (M4).
- Expert release construction.
- Raw historical task reproduction.

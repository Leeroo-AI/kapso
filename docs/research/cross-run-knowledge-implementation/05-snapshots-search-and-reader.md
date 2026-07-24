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
- Exact active release-use policy projection from irreversible catalog facts.
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

src/kapso/cross_run/
  record_contracts.py
  record_registry.py

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

- [x] Move the current isolated OpenAI embedding implementation to one shared
      dependency-pure `kapso.core.embedding_contracts` module, isolate the provider
      SDK in `kapso.core.embedding_provider`, and update existing ideation callers
      directly.
- [x] Delete the superseded provider implementation; retain no alias/import shim.
- [x] Keep the OpenAI import at module top in the feature module and preserve
      provider-SDK default credential discovery.
- [x] Ensure coding-agent and MCP subprocess environments receive no embedding
      credential.
- [x] Key every vector by provider, model, dimensions, canonicalizer version, and
      complete input hash (`EmbeddingSpaceId`).
- [x] Batch calls under config-owned count limits without truncating canonical
      source text; provider size-limit failures propagate rather than clipping an
      input.
- [x] Attribute embedding cost/latency separately from coding-agent telemetry.
- [x] Propagate missing credentials, provider errors, dimension mismatch, and
      malformed responses. Explicit `enabled: false` is the only no-embedding mode.

## Snapshot package

From one exact catalog generation, `KnowledgeSnapshotPublisher`:

- [x] Selects admitted, non-revoked objects under the configured scope/retrieval
      policy.
- [x] Includes every claim/relative-effect proof dependency, assertion, active
      state, revocation, and sanitation reference required for audit.
- [x] Projects every performance/compatibility release-use event as a distinct,
      sorted, proof-closed manifest field; never mixes it with scientific catalog
      revocation or retrieval roots.
- [x] Includes complete canonical JSON records; IDs-only placeholders are not
      sufficient for runtime retrieval.
- [x] Parses every envelope through the owning dependency-pure `StrictContract`
      from one registry shared with catalog reduction; a reminted malformed shape
      is rejected even when its content hash is internally consistent.
- [x] Includes the pinned scope contract and policy identities.
- [x] Builds a deterministic file order, archive metadata, checksums, manifest,
      and snapshot content ID.
- [x] Verifies that extraction of the package recreates the declared record and
      proof closure byte-for-byte.
- [x] Materializes the verified directory with an atomic no-replace commit, so a
      concurrent owner can never be overwritten during the staging window.
- [x] Keeps raw sanitized audit data outside the runtime package, which contains
      safe normalized knowledge only.
- [x] Deterministically shards release assets only at the config-owned size bound.

An explicit `EMPTY` snapshot is built and validated through the same path; missing
remote state is never interpreted as empty.

The release, publication, and activation-witness identities inside each event are
external GitHub authority references, not local package dependencies. The policy
reader must resolve the historical release identity, compare the exact publication
ID, reconstruct and compare the write-once activation witness, and match the
materialized expert manifest's scope before using the projection. A syntactically
valid event alone grants no release-use decision authority.

- [x] Resolve the current KnowledgeSnapshot twice around materialization and
      historical matching, rejecting a `CURRENT` race or missing `CURRENT`.
- [x] Authenticate exact absence from the complete package projection, including
      the empty-expert bootstrap case.
- [x] Group matching events by release and authenticate each historical activation
      once; unrelated broken external references cannot deny another release.
- [x] Return a content-addressed policy observation binding the current knowledge
      publication, pointer, repository, scope, checked releases, and exact matches.

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

- [x] Index scope/task family, context dimensions, trust/state, outcome,
      evaluation identity, lineage, timestamps, mechanism, applicability,
      exclusions, and record type.
- [x] Provide exact-term/identifier lexical search alongside semantic vectors.
- [x] Use exact cosine over compact vectors as the initial implementation.
- [ ] Build an ANN sidecar only after a measured corpus/latency threshold warrants
      it; the v1 implementation intentionally has no ANN mode. It
      may generate candidates but cannot change canonical filters or final ordering.
- [x] Pin sidecars to the exact catalog generation, record closure, canonical
      source input, and embedding space. The finalized snapshot binds every
      sidecar checksum, avoiding a snapshot-ID/index-manifest checksum cycle.
- [x] Reject stale/corrupt/mismatched sidecars. Rebuilding creates new assets and a
      new snapshot publication, not an in-place mutation.
- [x] Do not put indexes or growing databases in Git history.

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

- [x] Semantic similarity never determines truth, sign, novelty, or admission.
- [x] Exact-context records rank separately from analogies; incompatible records
      are absent.
- [x] Positive, negative, inconclusive, and frontier slots remain explicit.
- [x] A top-level record whose proof closure cannot fit is skipped as a whole.
- [x] Deterministic total-order tie breaking ends in content ID.
- [x] The resulting `PriorKnowledgeSnapshot` includes query/policy/source identity,
      exact records, typed proofs, digest, and per-record compatibility/outcome/rank
      metadata.
- [x] Repeating the same query over the same pin produces byte-identical output.

## MCP reader

`PriorKnowledgeGate` is a local read-only MCP gate with the minimum v1 tools:

```text
list_prior_knowledge()
get_prior_knowledge_record(record_id)
```

- [x] The gate receives an explicit packet/materialization path from the trusted
      launcher; it never resolves GitHub or `CURRENT`.
- [x] Validate canonical bytes, the packet digest, and record/proof membership
      before serving.
- [x] Permit only IDs present in the persisted packet for live ideation v1.
- [x] Return complete schema-rendered records, not clipped display summaries.
- [x] Treat record prose/code as untrusted data with explicit provenance labels.
- [ ] Seal the gate's exact tool calls and returned IDs into coding-agent
      invocation artifacts. M5 emits canonical audit events; M6 owns their durable
      capture with the consuming `IdeaBatch`.
- [x] Mount no write, network, raw artifact, or current-run memory authority.
- [x] Keep reader, gate, and server imports silent so coding-agent initialization
      cannot corrupt MCP's stdout JSON-RPC transport.

The trusted caller persists the access materialization as one canonical,
write-once local file with atomic publication and fsync before launching the MCP
subprocess. The configured materialization byte bound is enforced before parsing.

An interactive full-snapshot `search_prior_knowledge` tool is explicitly deferred.
It requires a durable access-session schema that seals every query/response into
the consuming operation before selection; it must not be added as an unlogged
shortcut.

## Knowledge release publication

- [x] Consume only the exact generation already produced by M4's deterministic
      automated admission state machine.
- [x] Rebuild the canonical package and search sidecars from the admitted exact
      catalog generation.
- [x] Compare built IDs/digests with the proposed manifest.
- [x] Use M2's autonomous transaction to commit directly, publish the immutable
      knowledge release, then CAS `CURRENT.json`.
- [x] Require a nonempty snapshot to name exactly the resolved current scientific
      snapshot as its sole parent; only explicit `EMPTY` has no parent.
- [x] Let an M2 CAS conflict fail loud without weakening the old `CURRENT`.
- [ ] Have the M10 operational loop resolve the new base, rebuild and revalidate,
      then retry as a new publication operation rather than mutating the loser.

## Tests

- Build the same snapshot under shuffled filesystem/catalog ordering and require
  identical bytes and IDs.
- Prove a clean temp directory can retrieve from only the materialized package.
- Verify complete proof closure and reject every missing dependency.
- Test exact/analogical/incompatible contexts and evaluator mismatch.
- Test positive/negative/inconclusive/frontier diversity and whole-record budgets.
- Test embedding-space separation, stale sidecars, corrupt float data, and explicit
  embedding disablement.
- Compare exact-cosine and ANN candidate modes only when the measured-scale ANN
  follow-up is implemented.
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

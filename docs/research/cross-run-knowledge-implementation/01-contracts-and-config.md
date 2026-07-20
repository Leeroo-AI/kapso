# M1 — contracts, canonical identity, and configuration

Parent plan: [`00-orchestrator-plan.md`](00-orchestrator-plan.md)

## Objective

Create the domain-neutral, strict substrate used by every cross-run module. This
module contains no GitHub access, filesystem publication, coding-agent call,
embedding call, strategy mutation, or task-family conditional.

## Owned responsibilities

- Every shared immutable schema named by the parent contract freeze.
- Canonical JSON, exact-key parsing, and content-addressed identifiers.
- Payload/envelope separation for attestations and GitHub locations.
- Scope-contract and registered context-dimension validation.
- Comparability, lineage, supersession, and proof-reference primitives.
- One validated `cross_run` configuration tree with a single source for defaults.
- Typed errors for incompatible, corrupt, missing-reference, and identity-conflict
  failures.

## Proposed code surface

```text
src/kapso/cross_run/
  __init__.py
  canonical.py
  contracts.py
  settings.py

src/kapso/
  config.yaml

src/kapso/core/
  config.py

tests/
  test_cross_run_canonical.py
  test_cross_run_contracts.py
  test_cross_run_settings.py
```

## Contract tasks

- [ ] Define immutable dataclasses/enums for every frozen contract.
- [ ] Reject unknown fields, missing fields, duplicate JSON keys, booleans used as
      integers, non-finite numbers, naive timestamps, empty identifiers, and
      unordered/gapped revision lists.
- [ ] Validate `ExpertScopeContract` task-family and context-dimension lineage.
- [ ] Validate `TaskContextBinding` exclusively against its pinned scope revision;
      do not add model/tokenizer/table/schema names to core code.
- [ ] Keep `EvaluationFingerprint` equivalence separate from transfer
      compatibility.
- [ ] Require globally qualified source identity on cross-run records while
      retaining local IDs only as provenance fields.
- [ ] Define exact supersession and capability split/merge lineage rules.
- [ ] Separate immutable content payloads from `CatalogEntryState`, attestations,
      and `GitHubPublicationRecord`.
- [ ] Require claim applicability, exclusions, support, contradiction, and state.
- [ ] Require knowledge/expert manifests to name complete dependency and checksum
      closures.
- [ ] Define `PriorKnowledgeSnapshot` as complete selected records plus query,
      policy, source snapshot, proof references, and digest.
- [ ] Define `BootstrapPin` independently of `RunCheckpoint` so startup can become
      durable before orchestrator construction.
- [ ] Use descriptive code identifiers; design index names remain documentation
      provenance only.

## Canonical identity

`canonical.py` owns one implementation for:

```text
canonical_json_bytes(payload)
content_id(namespace, payload_without_id)
tree_or_blob_digest(bytes)
verify_declared_content_id(payload)
```

Rules:

- UTF-8, sorted keys, fixed separators, and no non-finite numbers;
- sets are represented as sorted arrays under schema-specific total orders;
- timestamps are normalized UTC strings before hashing;
- an object's ID field is absent from its own hash preimage;
- signatures, publication locations, mutable admission state, and access telemetry
  are outside the content preimage;
- hash algorithms and canonicalizer versions are explicit structural identities;
  they are not silently negotiated; and
- parsing corrupt bytes raises. Only a documented missing optional file may yield
  an empty result.

Golden-byte fixtures pin canonical output so later refactors cannot silently mint
different IDs.

## Configuration tasks

- [ ] Add one `cross_run` tree to `src/kapso/config.yaml` containing all paths,
      budgets, thresholds, timeouts, branch/tag conventions, cache policy,
      embedding model, CLI roles, validation gates, and production-test settings.
- [ ] Add strict typed settings in `cross_run/settings.py`; module/dataclass defaults
      are sourced from the loaded canonical config rather than repeated literals.
- [ ] Validate repository coordinates, branch/tag prefixes, filesystem paths,
      non-negative budgets, ratios, timeouts, index settings, and enabled role
      combinations before any external call.
- [ ] Keep secrets and authentication material out of the config.
- [ ] Thread settings explicitly from `load_config()`/`load_mode_config()`; no new
      module reads process environment variables.
- [ ] Include the relevant cross-run configuration projection in launch, capture,
      catalog, snapshot, and validation fingerprints.

The plan intentionally does not freeze example numeric defaults. Each operational
value is chosen once in `config.yaml` during implementation and referenced from
there in code and tests.

## Tests

- Round-trip every contract through canonical JSON.
- Pin golden bytes and content IDs for representative records.
- Reject every missing, unknown, duplicate, wrongly typed, and non-finite field.
- Prove attestations and GitHub relocation do not change scientific content IDs.
- Prove registered post-training and relational-prediction context bindings both
  validate without core conditionals.
- Reject unregistered dimensions and incompatible scope revisions.
- Prove comparable evaluation fingerprints and analogical task contexts remain
  independent classifications.
- Validate episode attempt ordering, claim proof references, candidate lineage,
  snapshot closure, release maps, launch manifests, and bootstrap pins.
- Prove every settings value comes from the config tree and secrets are absent.

## Definition of done

- Parent-plan contract freeze is marked complete.
- All downstream modules can import contracts without importing GenericSearch,
  GitHub, OpenAI, or a coding-agent adapter.
- Canonical identity is deterministic across process restarts and input ordering.
- The config fails before work on every invalid operational value.
- No compatibility parser, schema migration, deprecated alias, or domain-specific
  field exists.

## Non-goals

- Persistence stores or atomic writes.
- GitHub/API command execution.
- Scientific interpretation or retrieval ranking.
- Task-adapter implementation.

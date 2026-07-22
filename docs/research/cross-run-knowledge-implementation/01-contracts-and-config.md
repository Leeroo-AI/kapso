# M1 — contracts, canonical identity, and configuration

Parent plan: [`00-orchestrator-plan.md`](00-orchestrator-plan.md)

Status: **complete**

## Objective

Create the domain-neutral, strict substrate used by every cross-run module. This
module contains no GitHub access, filesystem publication, coding-agent call,
embedding call, strategy mutation, or task-family conditional.

## Owned responsibilities

- Every shared immutable schema named by the parent contract freeze.
- Canonical JSON, exact-key parsing, and content-addressed identifiers.
- Payload/envelope separation for attestations and GitHub locations.
- Scope-contract and registered context-dimension validation.
- Typed `ScopeRegistry` repository routing and `CrossRunTaskBinding` validation.
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

- [x] Define immutable dataclasses/enums for every frozen contract.
- [x] Reject unknown fields, missing fields, duplicate JSON keys, booleans used as
      integers, non-finite numbers, naive timestamps, empty identifiers, and
      unordered/gapped revision lists.
- [x] Validate `ExpertScopeContract` task-family and context-dimension lineage.
- [x] Define strict `ScopeRepositorySettings` and `CrossRunTaskBindingSettings`:
      the former maps one `scope_id` to one expert/knowledge/security repository
      triple; the latter carries only `scope_id`, `task_family_id`, and
      `task_adapter_id`.
- [x] Canonically fingerprint each resolved scope repository entry so
      `LaunchManifest` can record the exact location binding used at startup.
- [x] Reject missing scopes, duplicate repository triples, any repository assigned
      to more than one scope, aliases within a triple, and unknown task-family or
      adapter bindings.
- [x] Validate `TaskContextBinding` exclusively against its pinned scope revision;
      do not add model/tokenizer/table/schema names to core code.
- [x] Keep `EvaluationFingerprint` equivalence separate from transfer
      compatibility.
- [x] Require globally qualified source identity on cross-run records while
      retaining local IDs only as provenance fields.
- [x] Define exact supersession and capability split/merge lineage rules.
- [x] Separate immutable content payloads from `CatalogEntryState`, attestations,
      and `GitHubPublicationRecord`.
- [x] Require claim applicability, exclusions, support, contradiction, and state.
- [x] Require knowledge/expert manifests to name complete dependency and checksum
      closures.
- [x] Define cumulative security-denylist evidence, revocation, snapshot, and
      evidence-bundle contracts with exact predecessor/proof closure and immutable
      snapshot identity in launch/bootstrap pins.
- [x] Define `PriorKnowledgeSnapshot` as complete selected records plus query,
      policy, source snapshot, proof references, and digest.
- [x] Define `BootstrapPin` independently of `RunCheckpoint` so startup can become
      durable before orchestrator construction.
- [x] Use descriptive code identifiers; design index names remain documentation
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

- [x] Add one `cross_run` tree to `src/kapso/config.yaml` containing all paths,
      budgets, thresholds, timeouts, branch/tag conventions, cache policy,
      embedding model, CLI roles, validation gates, and production-test settings.
- [x] Make `cross_run.scopes` the sole repository-location registry. For the first
      deployment it contains `ml_ai -> Leeroo-AI/kapso-expert +
      Leeroo-AI/kapso-knowledge + Leeroo-AI/kapso-security`; no other
      author-maintained config contains those coordinates.
- [x] Define and validate the typed `cross_run_binding` config shape; M9 owns
      populating and wiring the concrete PostTrainBench and RelBench bindings.
- [x] Add explicit config composition that copies the canonical scope registry into
      each generated self-contained runtime config with its source fingerprint.
      Benchmark config files and runners must not duplicate, override, infer, or
      hardcode repository coordinates.
- [x] Return one typed effective config containing the global `cross_run` tree and
      the selected workload mode. Keep the workload-only projection free of
      cross-run operator settings so they cannot enter the existing campaign
      resume fingerprint; M9 owns threading the typed effective config into launch.
- [x] Add strict typed settings in `cross_run/settings.py`; module/dataclass defaults
      are sourced from the loaded canonical config rather than repeated literals.
- [x] Validate repository coordinates, branch/tag prefixes, filesystem paths,
      non-negative budgets, ratios, timeouts, index settings, and enabled role
      combinations before any external call.
- [x] Keep secrets and authentication material out of the config.
- [x] Thread settings explicitly from `load_config()`/`load_effective_config()`; no new
      module reads process environment variables.
- [x] Include the relevant cross-run configuration projection in launch, capture,
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
- Prove both bindings resolve to the same configured `ml_ai` repository triple
  while preserving distinct task-family and adapter identities.
- Reject unknown scope, unknown family/adapter, duplicated repository ownership,
  within-scope aliasing, and any benchmark-level repository override.
- Reject a generated runtime config whose copied registry does not match its
  canonical source fingerprint.
- Prove repository relocation changes only settings/location records, not scope
  contracts, artifact content IDs, or persisted launch identities.
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
- Repository coordinates are single-sourced by scope and never accepted as a task
  or `Kapso.evolve` argument.
- No compatibility parser, schema migration, deprecated alias, or domain-specific
  field exists.

## Implementation and validation

Implemented in `kapso.cross_run.canonical`, `kapso.cross_run.contracts`,
`kapso.cross_run.settings`, and `kapso.core.config`, with the canonical settings
tree in `src/kapso/config.yaml`.

- 63 focused canonical/contract/config tests pass.
- 89 affected config, ideation, budget, and checkpoint tests pass.
- Canonical configuration identity is stable across fresh Python processes.
- Both configured private GitHub repositories resolve under the authenticated
  production identity with administrator permission.
- Installed-package loading, PostTrainBench/RelBench config composition,
  compilation, and the Generic/GitHub/OpenAI/coding-agent import boundary pass.
- Four independent `fable` reviews at maximum reasoning were applied; their
  integration, strictness, and causal-evidence findings are covered by regression
  tests.

## Non-goals

- Persistence stores or atomic writes.
- GitHub/API command execution.
- Scientific interpretation or retrieval ranking.
- Task-adapter implementation.

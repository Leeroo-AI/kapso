# M7 — expert candidates, repository architecture, and semantic book

Parent plan: [`00-orchestrator-plan.md`](00-orchestrator-plan.md)

Depends on: M1, M2, and M5.

## Objective

Produce isolated, evidence-linked expert candidates without mutating a stable
release. Support both empty-scope repository bootstrap and later capability or
topology evolution, while keeping the repository domain-neutral and its semantic
book mechanically synchronized.

## Owned responsibilities

- Evidence-backed capability and architecture trigger evaluation.
- Codex/Claude `ExpertRepoArchitect` bootstrap/restructure proposals.
- Codex/Claude `GeneralizationProposer` capability patches.
- Candidate workspace/store and immutable manifest.
- `ExpertRepositoryMap`, module contracts, and capability lineage.
- Deterministic `EXPERT_REPO.md` compilation and validation.
- Validated candidate handoff to M8's autonomous release path.

## Proposed code surface

```text
src/kapso/cross_run/expert/
  __init__.py
  triggers.py
  architect.py
  generalizer.py
  candidates.py
  book.py

src/kapso/cross_run/prompts/
  expert_repo_bootstrap.md
  expert_repo_restructure.md
  expert_capability_generalization.md

tests/
  test_expert_triggers.py
  test_expert_architect.py
  test_expert_generalizer.py
  test_expert_candidates.py
  test_expert_semantic_book.py
```

## Trigger policy

Capability candidates require one configured trigger:

- repeated difficulty across independent run lineages;
- repeated mechanism success across distinct registered contexts;
- a mechanically general infrastructure/reliability fix;
- a supported claim whose executable form removes repeated work; or
- valid contradiction/revocation evidence against a released capability.

Architecture candidates require:

- an admitted scope/task-family/artifact change that does not fit current bounds;
- repeated cross-module duplication suggesting shared ownership;
- dependency cycles, adapter leakage, or contract/topology mismatch; or
- semantic-book navigation/composition evidence showing structural failure.

- [x] Implement trigger calculation deterministically from pinned evidence and
      configured thresholds.
- [x] Count independent lineages, not cloned descendants.
- [x] Treat highest score, file-copy frequency, agent preference, or aesthetic
      rearrangement as insufficient.
- [ ] Classify task-specific improvements as knowledge/task-adapter work, never
      expert core.
- [x] Persist trigger decision and exact evidence packet even when no candidate is
      proposed.

## Bootstrap architect

When no eligible release exists for a scope, `ExpertRepoArchitect` receives:

```text
attested ExpertScopeContract
task-family/adapter contracts
runtime and repository constraints
representative public task contracts
explicit empty parent release state
strict candidate/map/module schemas
```

Through the configured Codex or Claude Code CLI it edits a clean local candidate
workspace and proposes:

- minimal capability boundaries and stable IDs;
- physical source/test layout;
- module contracts and dependency graph;
- task-adapter boundary;
- fresh-task smoke harness; and
- machine-readable `ExpertRepositoryMap` sufficient for book generation.

- [ ] Do not scaffold speculative empty task-family subsystems.
- [ ] Do not include datasets, weights, experiment memory, logs, hidden evaluation,
      Git history, benchmark answers, or identity-specific defaults.
- [ ] Preserve complete invocation artifacts and CLI/model/tool provenance.
- [ ] Treat output as a quarantined `repository_architecture` candidate, never E0.

## Later repository restructuring

- [x] Materialize the exact parent expert release and verification receipt.
- [ ] Provide the architect only the persisted trigger evidence packet and
      scope/task-family contracts.
- [ ] Require one atomic full-tree candidate covering every move/split/merge/rename/
      deletion plus updated tests, entrypoints, module contracts, dependency edges,
      and repository map.
- [ ] Keep capability IDs stable across path-only moves.
- [ ] Mint new IDs plus explicit old-to-new lineage for semantic split/merge.
- [ ] Record lineage for historical evidence interpretation, not runtime shims.
- [ ] Require measured structural benefit or admitted scope accommodation.

## Capability generalization

`GeneralizationProposer` receives one trigger, parent release, selected candidate
ancestors, and a proof-closed knowledge packet.

- [ ] Use the existing coding-agent runner; no direct generative API.
- [ ] Ask for the smallest task-general patch and complete module-contract update.
- [ ] Preserve known failures, preconditions, exclusions, resource bounds,
      dependencies/licenses, tests, replay refs, and evidence IDs.
- [ ] Keep benchmark/model/task identity out of generic defaults.
- [ ] Permit reuse of non-revoked candidate ancestors selected by evidence and
      lineage diversity, not simply latest/highest score.
- [ ] Preserve failed/non-dominated candidates as immutable future inputs.

## Candidate store and manifest

- [ ] Create an isolated candidate workspace from the exact parent tree hash.
- [ ] Compute patch, full candidate tree hash, module-contract refs, proposed
      repository-map ref, dependencies, lineage, source evidence, sanitation report,
      and coding-agent provenance.
- [ ] Reject candidate changes outside allowed source/test/contract roots.
- [ ] Preserve validation attempts as later immutable attachments; do not mutate
      candidate proposal identity.
- [ ] Identical candidate replay is idempotent; conflicting content under one ID
      fails.
- [ ] Hand only locally schema/sanitation-valid candidates to M8's automated
      validation state machine.

## Semantic book compiler

`EXPERT_REPO.md` is generated, never agent-authored.

- [ ] Render from the exact `ExpertRepositoryMap` and `ExpertModuleContract`s.
- [ ] Include purpose/invariants, one-screen architecture/stage flow, capability
      index, inputs/outputs, preconditions/incompatibilities, entrypoints/tests,
      dependencies/compositions, adapter boundary, validation commands, and
      external evidence/failure IDs.
- [ ] Validate every path, link, entrypoint, test, dependency edge, incompatibility,
      and evidence reference against the candidate tree/knowledge packet.
- [ ] Produce deterministic bytes/digest under input ordering changes.
- [ ] Reject manual edits or a manifest/book digest mismatch.

## Tests

- Empty scope produces a minimal complete candidate, not a promoted E0.
- Post-training-only first scope does not hardcode that ontology into framework
  contracts; adding relational prediction can restructure cleanly.
- Trigger fixtures cover independent evidence, correlated clones, one-off winner,
  repeated failure, new task family, duplication, cycle, adapter leakage, and
  aesthetic rename.
- Fake Codex/Claude outputs exercise valid patch, malformed map, missing test,
  task-specialized default, hidden benchmark constant, and speculative scaffold.
- Verify path moves preserve capability IDs and splits/merges require lineage.
- Compile the same book from shuffled inputs and require identical bytes.
- Break each path/link/dependency/test/evidence ref and require rejection.
- Prove proposer output alone cannot activate E0/E+1 and manual book edits fail
  deterministic validation.

## Definition of done

- Empty and non-empty scopes can produce isolated architecture candidates.
- Capability proposals are evidence-linked and task-general by contract.
- Repository structure can evolve atomically without fixed ML-domain folders.
- Every candidate has a valid repository map, module contracts, lineage, and
  deterministic semantic book.
- Stable releases remain untouched; M8 is the only promotion path.

## Non-goals

- Candidate certification or release publication.
- Hidden/sealed evaluator execution.
- Knowledge claim admission.
- Task-adapter implementation details.

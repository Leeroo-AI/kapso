# M7 — expert candidates, repository architecture, and semantic book

Parent plan: [`00-orchestrator-plan.md`](00-orchestrator-plan.md)

Depends on: M1, M2, and M5.

Status: **Complete; architect/generalizer proposal, deterministic candidate
sealing, exact ancestor inputs, and independent correctness review approved.**

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
  proposal.py
  proposal_contract.py
  candidates.py
  book.py
  store.py
  workspace.py

src/kapso/execution/coding_agents/
  structured_call.py
  workspace_delta.py

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
- [ ] Add M8's independent semantic classification of task-specific improvements;
      prompts already direct them to knowledge/task-adapter work, never
      expert core.
- [x] Persist trigger decision and exact evidence packet even when no candidate is
      proposed.

## Bootstrap architect

When no eligible release exists for a scope, `ExpertRepoArchitect` receives:

```text
attested ExpertScopeContract
active task-family/adapter binding identities
repository constraints and exact workspace limits
persisted trigger evidence and optional proof-closed prior knowledge
explicit empty parent release state
strict candidate/map/module schemas
```

The proposer does not receive raw task repositories or invent facts about an
adapter implementation. It proposes the expert-side interface from the attested
scope constraints. M8's fresh-task gates validate that interface against the
pinned public adapter contract; M9 later makes the `TaskAdapterManifest` an exact
launch input. A bootstrap proposal can therefore be quarantined before adapter
validation, but cannot certify its own harness or become E0.

Through the configured Codex or Claude Code CLI it edits a clean local candidate
workspace and proposes:

- minimal capability boundaries and stable IDs;
- physical source/test layout;
- module contracts and dependency graph;
- task-adapter boundary;
- fresh-task smoke harness; and
- machine-readable `ExpertRepositoryMap` sufficient for book generation.

- [x] Do not scaffold speculative empty task-family subsystems.
- [x] Do not include datasets, weights, experiment memory, logs, hidden evaluation,
      Git history, benchmark answers, or identity-specific defaults.
- [x] Preserve complete invocation artifacts and CLI/model/tool provenance,
      including explicit read/edit authority and the exact workspace delta.
- [x] Treat output as a quarantined `repository_architecture` candidate, never E0.

## Later repository restructuring

- [x] Materialize the exact parent expert release and verification receipt.
- [x] Provide the architect only the persisted trigger evidence packet and
      scope/task-family contracts.
- [x] Require one atomic full-tree candidate covering every move/split/merge/rename/
      deletion plus updated tests, entrypoints, module contracts, dependency edges,
      and repository map.
- [x] Keep capability IDs stable across path-only moves.
- [x] Mint new IDs plus explicit old-to-new lineage for semantic split/merge.
- [x] Record lineage for historical evidence interpretation, not runtime shims.
- [x] Require measured structural benefit or admitted scope accommodation.

## Capability generalization

`GeneralizationProposer` receives one trigger, parent release, selected candidate
ancestors, and a proof-closed knowledge packet.

- [x] Extend the existing coding-agent runner with explicit edit authority; no
      direct generative API.
- [x] Ask for the smallest task-general patch and complete module-contract update.
- [x] Preserve known failures, preconditions, exclusions, resource bounds,
      dependencies/licenses, tests, replay refs, and evidence IDs.
- [x] Enforce monotonic accumulated safety/provenance fields and exact fixed
      safety envelopes when deriving changed module contracts; prompt guidance
      alone is not authority. Module versions are positive integer `vN` values
      and must increase.
- [x] Keep benchmark/model/task identity out of generic defaults.
- [ ] Bind M8 admission/revocation and diversity selection state to persisted
      candidate ancestors selected by evidence and
      lineage diversity, not simply latest/highest score.
- [x] Preserve failed/non-dominated candidates as immutable future inputs.
- [x] Require every changed capability to own an edited/deleted non-control path
      and keep observation-triggered changes within the named capabilities.

## Candidate store and manifest

- [x] Create an isolated private candidate workspace from the canonical empty tree
      or the exact released-parent source receipt; validate generated controls,
      remove only those controls through pinned descriptors, and require the
      remaining editable tree to equal parent-minus-controls.
- [x] Serialize edits as a canonical create/modify/delete delta bound to both exact
      parent and edited tree hashes; validate and replay it without relying on
      subprocess side effects.
- [x] Lease a workspace across baseline inspection, CLI execution, and delta
      sealing so distinct operation IDs cannot race on one tree.
- [x] Thread the active expert lease's descriptor authority into editable coding
      agents; execute the CLI and inspect both sides of its delta through that
      descriptor so temporary path substitution cannot enter a candidate.
- [x] Bind cached results to exact invocation, `final.json`, audit, artifact set,
      and workspace delta; reject partial, public, linked, or substituted
      artifacts.
- [x] Package the complete coding-agent artifact closure with every candidate,
      replay prior-knowledge audit truth on reopen, require recognized security
      policies, and bind normalized MCP authority into operation identity.
- [x] Bind the configured proposer principal into operation identity and include
      every model-visible prior selection/snapshot/record/proof ID in the
      manifest's taint/revocation dependency closure.
- [x] Pin the complete proposer authority in each operation so future principal
      rotation changes new identities without corrupting historical candidates.
- [x] Compute patch, full candidate tree hash, module-contract refs, proposed
      repository-map ref, dependencies, lineage, source evidence, sanitation report,
      and coding-agent provenance.
- [x] Persist the sorted unique set of expert releases whose bytes or run outputs
      were model-visible: the immediate source base, every trigger/prior-knowledge
      episode, and the transitive lineage of every ancestor candidate. Reopen
      recomputes the set, and persistence re-resolves each direct ancestor from the
      immutable store under the candidate lock.
- [x] Reject candidate changes outside allowed source/test/contract roots and
      configured aggregate entry/byte limits.
- [x] Preserve validation attempts as later immutable attachments; do not mutate
      candidate proposal identity.
- [x] Identical candidate replay is idempotent; conflicting content under one ID
      fails.
- [x] Hand only locally schema/sanitation-valid candidates to M8's automated
      validation state machine.

## Semantic book compiler

`EXPERT_REPO.md` is generated, never agent-authored.

- [x] Render from the exact `ExpertRepositoryMap` and `ExpertModuleContract`s.
- [x] Include purpose/invariants, one-screen architecture/stage flow, capability
      index, inputs/outputs, preconditions/incompatibilities, entrypoints/tests,
      dependencies/compositions, adapter boundary, validation commands, and
      external evidence/failure IDs.
- [x] Validate every path, link, entrypoint, test, dependency edge, incompatibility,
      and evidence reference against the candidate tree/knowledge packet.
- [x] Produce deterministic bytes/digest under input ordering changes.
- [x] Reject manual edits or a manifest/book digest mismatch.

## Implemented proposal boundary

`ExpertRepositoryArchitect` accepts only deterministic `BOOTSTRAP` or
`RESTRUCTURE` decisions. Its coding agent edits ordinary source/tests and returns
the complete desired topology plus complete semantic module contracts.
`ExpertCapabilityGeneralizer` accepts only `GENERALIZE`; it returns complete
replacements for changed module contracts while Kapso preserves the parent
topology exactly.

Both delegate to one `ExpertCandidateProposalEngine`. The engine recomputes the
trigger, validates an optional proof-closed prior-knowledge materialization, reads
ancestor IDs only through the immutable candidate store, leases the exact
parent-minus-controls workspace, derives the operation ID from the full prompt,
schema, principal, MCP authority, ancestors, and trigger, and invokes the
configured coding agent with pinned descriptor authority. It then seals the durable workspace
delta, rejects generated-control edits or path declaration drift, reconstructs
the edited bytes, mints all framework identities, regenerates controls/book,
scans and validates the whole detached closure, successfully closes the lease,
and only then persists it.

Every selected ancestor is embedded as a content-identified
`ExpertCandidateAncestorInput`: manifest, scope, patch, exact tree and bytes,
repository map, module contracts, workspace delta, and sanitation report. The
exact admitted UTF-8 source is represented as readable text as well as verified
descriptors, so the coding agent can inspect implementations rather than opaque
base64. The proposal prompt and candidate package therefore remain reconstructible
without the original workspace. M8 will add validation/revocation eligibility state; M7
already refuses caller-injected closures and accepts only IDs that reopen from the
local immutable store.

The packaged operation must reproduce the fixed mode-specific prompt and JSON
schema under `kapso.expert_proposal.v1`. Reopen reparses `final.json`, rederives
the map/module/lineage result, revalidates the prior-knowledge closure, and rejects
any semantic substitution even when lower-level artifact checksums are internally
consistent.

The derived manifest dependency closure includes the exact prior-knowledge
selection artifact, its source snapshot, and every selected/proof record ID. A
candidate cannot reopen if those causal edges are removed. Generalization also
rejects removal of accumulated safety/provenance fields, changes to the fixed
resource/dependency envelope, or rewrites of existing license declarations.
Restructuring must make a real structural/path-interface change and applies the
same protections to every preserved semantic capability; path-reference
replacement remains legal only when removed refs are observed deletions and have
same-kind replacements among observed changed paths.

Release-use lineage is deliberately narrower than emergency taint. A published
expert release is a new scientific checkpoint: a later candidate consumes that
immediate release ID, not the unreleased inputs that once produced it. Before
publication, however, lineage is transitive through agent ancestors and
deterministic composition sources. Every episode shown to a coding agent counts,
even when the trigger reducer did not classify it as causal, because model
visibility itself is a consuming edge.

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
- Prove concurrent first construction, concurrent identical persistence,
  noncanonical commit records, checksum changes, hardlinks, symlinks, FIFOs, and
  public package modes cannot expose an invalid candidate.
- Prove editable calls reject wrong parents, symlinks, hardlinks, special files,
  oversize trees, path collisions, cross-operation workspace races, and tampered
  cached output/deltas.
- Prove construction failures remove created destinations; root/control/source
  swap races fail loud; cleanup handles read-only trees, FIFOs, sockets, links,
  and replacement names without following or changing outside targets.
- Prove a replace-edit-restore ABA sequence cannot redirect CLI execution or
  delta sealing, and an already-substituted path differs from outer authority.

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

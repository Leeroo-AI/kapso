# M6 — ideation-v3 and local-memory bridge

Parent plan: [`00-orchestrator-plan.md`](00-orchestrator-plan.md)

Depends on: M1 and M5.

## Objective

Connect one pinned `KnowledgeSnapshot` to live ideation without merging foreign
ideas/experiments into current-run authorities. Persist the exact prior packet used
by every batch so generation, analysis, selection, and resume are reproducible.

## Owned responsibilities

- New-only IdeaArchive and Generic checkpoint schemas.
- Prior retrieval placement in `IdeationEngine` transaction ordering.
- `PriorKnowledgeSnapshot` and foreign-reference provenance on batches/ideas.
- Separate local-evidence and prior-knowledge prompt sections.
- Cross-run advisory novelty/adaptation analysis.
- Packet-only MCP reader mounting for Codex/Claude ideation roles.
- Strict resume/reconciliation across checkpoint, archive, and prior packet.

## Proposed code surface

```text
src/kapso/execution/search_strategies/generic/ideation/
  types.py
  archive.py
  evidence.py
  analyzer.py
  generator.py
  selector.py
  engine.py
  coding_agents.py

src/kapso/execution/search_strategies/generic/
  strategy.py

src/kapso/execution/search_strategies/generic/prompts/
  ideation_v3_candidate.md
  ideation_v3_selector.md

tests/
  test_prior_knowledge_ideation_types.py
  test_prior_knowledge_ideation_engine.py
  test_prior_knowledge_analysis.py
  test_prior_knowledge_resume.py
  test_cross_run_ideation_integration.py
```

## Persisted shape changes

- [ ] Replace `kapso.ideation_archive.v3` with the one new archive shape containing
      a source snapshot ID and exact `PriorKnowledgeSnapshot` on each applicable
      `IdeaBatch`.
- [ ] Add typed `prior_knowledge_refs` to `IdeaRecord`; keep local `evidence_refs`,
      claim IDs, parent idea IDs, and node IDs local-only.
- [ ] Add launch/scope/snapshot/expert-release identities to the Generic strategy
      state and context hash.
- [ ] Replace the current Generic checkpoint schema directly; old checkpoints fail
      with an explicit restart requirement.
- [ ] Update every serializer, fixture, prompt artifact, result manifest, and
      reconciliation test in the same changes.
- [ ] Add no migration, dual reader, missing-field default, or compatibility alias.

## Engine transaction ordering

For a new batch:

1. reconcile checkpoint, `IdeaArchive`, `ExperimentHistoryStore`, and node history;
2. build current `CampaignEvidenceSnapshot`;
3. choose local BOOTSTRAP/EXPLORE/EXPLOIT policy, gaps, parent, and directive;
4. build the cross-run query from problem, pinned `TaskContextBinding`, local gaps,
   and directive;
5. retrieve exactly one proof-closed `PriorKnowledgeSnapshot` from M5;
6. create/persist the planned `IdeaBatch` with both local evidence and prior packet;
7. generate/analyze/select using those frozen inputs;
8. persist selected local idea before node creation; and
9. continue the existing implementation/evaluation path.

- [ ] If retrieval fails, the batch is not created and no model call starts.
- [ ] If the configured snapshot is explicit `EMPTY`, persist an explicit empty
      prior packet with its real identity/digest.
- [ ] Resume loads the persisted packet and never searches the snapshot again for
      that batch.
- [ ] Context hashes include full packet identity/content, not a clipped rendering.

## Prompt and CLI integration

- [ ] Render local current-run evidence and foreign prior knowledge in separately
      labelled mandatory sections.
- [ ] Preserve complete selected records; packet budgeting happened before prompt
      construction.
- [ ] State that foreign scores are not local incumbents, foreign failures do not
      close local gaps, and foreign records cannot be parents.
- [ ] Mount M5's `PriorKnowledgeGate` into generator and selector Codex/Claude CLI
      configurations against the persisted packet.
- [ ] Expose only packet list/get tools; implementation raw artifacts remain behind
      a separate future read-only gate.
- [ ] Persist MCP calls/results with the normal coding-agent invocation artifacts.
- [ ] Do not pass GitHub or OpenAI credentials to the agent/MCP subprocesses.

## Analysis and selection semantics

- [ ] Compute exact/descriptor/embedding neighbors against both local ideas and
      eligible prior records, but label their origins.
- [ ] Keep hard exact-duplicate rejection local-campaign-only.
- [ ] A close foreign record may require an explicit adaptation or changed-context
      rationale; it cannot make the local idea ineligible by itself.
- [ ] Cross-run evidence may influence BOOTSTRAP/EXPLORE proposals but cannot move
      local policy to EXPLOIT.
- [ ] EXPLOIT still requires a supported local lever under the current evidence
      snapshot.
- [ ] Selector criticism may cite prior records but must create/select one new local
      idea with valid local parent/artifact/capacity references.
- [ ] Generated foreign-reference IDs are validated against the persisted packet.

## Authority separation

The only connection is:

```text
local store/archive -> RunBundle -> cross-run snapshot
cross-run snapshot -> prior packet -> new local idea -> new local node
```

M6 must prove that foreign records never enter:

- `GenericSearch.node_history`;
- `ExperimentHistoryStore`;
- local evidence claims/gap closure/incumbent computation;
- local parent resolution;
- local `IdeaArchive` as foreign batches or foreign ideas; or
- checkpoint node prefixes.

A new local idea may cite a prior record but always receives a new local ID,
analysis, decision, node link, and evaluation.

## Tests

- Empty snapshot, positive prior, negative prior, contradiction, analogical prior,
  incompatible prior, and foreign unexecuted-idea fixtures.
- Verify directive planning precedes retrieval and retrieval precedes batch/model
  calls.
- Crash after packet persistence, generation, analysis, selection, and node link;
  resume must reuse exact prior content and issue no duplicate query.
- Reject a checkpoint/archive packet digest mismatch.
- Prove no foreign ID is accepted as local parent, node, claim, gap resolution, or
  incumbent.
- Prove global exact duplicate remains advisory while local exact duplicate remains
  a hard rule.
- Verify EXPLOIT cannot be anchored by cross-run support alone.
- Verify Codex and Claude configurations both mount packet-only MCP tools with no
  provider credentials.
- End-to-end deterministic run: prior failure changes the generated/selected local
  idea while the executed store still begins empty.

## Definition of done

- Every ideation batch is reproducible from its local evidence, prior packet, and
  coding-agent artifacts.
- Resume never re-queries prior knowledge for completed batch phases.
- Local and foreign authorities are mechanically unmergeable.
- Both supported coding-agent CLIs can inspect full selected prior records through
  MCP.
- No pre-cross-run archive/checkpoint path remains.

## Non-goals

- Publishing snapshots or generating embeddings.
- Dynamic full-snapshot search from inside a model call.
- Expert candidate generation/promotion.
- Changing local policy, budget, fidelity, or evaluator authority.

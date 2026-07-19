# M5 — experiment memory, retrieval, and outcomes

Parent plan: [`00-orchestrator-plan.md`](00-orchestrator-plan.md)

Depends on: M1. Final live integration depends on M4.

## Objective

Connect executed experiments back to their originating ideas while preserving
the strict separation between proposed work and executed evidence.

## Owned responsibilities

- Required idea provenance on the Generic SearchNode-to-ExperimentRecord projection.
- Objective-aware experiment retrieval.
- Final outcome update after orchestrator-side candidate evaluation.
- Complete frozen proposal-memory packets and separate executed-memory retrieval.
- Cross-store reconstruction and consistency rules.
- Strict replacement experiment JSON shape.

## Proposed code surface

```text
execution/memories/experiment_memory/store.py
execution/orchestrator.py
gated_mcp/gates/experiment_history_gate.py
gated_mcp/gates/__init__.py
gated_mcp/presets.py
gated_mcp/server.py
generic/ideation/outcomes.py

tests/
  test_experiment_idea_linkage.py
  test_ideation_outcomes.py
```

## Experiment projection tasks

- [x] Add required `idea_id`, `selection_batch_id`, parent lineage, objective
      direction/utility, fidelity, evaluation attempts, duration, and cost to
      `ExperimentRecord`.
- [x] Project fields from finalized SearchNodes without changing raw score.
- [x] Reject incompatible or unlinked generic experiment records.
- [x] Make top-experiment retrieval correct for minimizing objectives.
- [x] Preserve invalid evaluations as audit records while excluding them from
      top-result ranking and hypothesis outcomes.
- [x] Keep experiment similarity search limited to executed content.

## Outcome write-back tasks

- [x] Add a narrow strategy/orchestrator hook invoked after
      `_evaluate_candidates` and experiment-store persistence.
- [x] Build `IdeaOutcome` only from the finalized linked node.
- [x] Distinguish technical, invalid, inconclusive, valid-positive, and
      valid-negative outcomes.
- [x] Compute normalized delta against the selected idea's frozen comparison
      basis.
- [x] Author strict causal evidence by default from the complete code diff and
      evaluation/feedback provenance; require a current registered attempt for
      claims and targeted gap updates.
- [x] Allow an external evaluator to explicitly replace `ideation_evidence`
      while preserving immutable built-in author provenance and ordinary
      metadata.
- [x] Apply no claim or gap effect from score direction alone.
- [x] Make duplicate outcome writes idempotent.
- [x] If archive update fails after experiment persistence, leave enough link
      data for M6 reconciliation; do not rerun work.

## Retrieval tasks

Keep existing executed-memory tools:

```text
get_top_experiments
get_recent_experiments
search_similar_experiments
```

The original plan considered separate proposal-memory tools:

```text
get_recent_ideas
get_idea
get_idea_neighbors
get_idea_lineage
list_evaluation_gaps
```

They are intentionally not added. Every v3 generator and selector call already
receives the complete frozen idea archive, claims, gaps, and precomputed
neighbors in its mandatory packet. An MCP gate would be an unused second read
model with a separate environment/configuration path. Executed experiment tools
remain separate and expose only executed records.

- [x] Label executed results with stable idea and batch IDs.
- [x] Show experiment parent, validation tier/fidelity, and linked idea.
- [x] Keep idea lifecycle, lineage, gaps, and precomputed neighbors in the
      complete mandatory campaign packet, distinct from executed-memory tools.
- [x] Keep free-text idea similarity in the trusted Kapso process; MCP
      processes receive neither the embedding API key nor embedding authority.

## Tests

- Incompatible experiment JSON fails loudly.
- New projection round-trips every strict field.
- Minimize objective returns the correct top experiments.
- Invalid evaluations remain stored but do not rank or update ideas.
- One finalized node produces one linked ExperimentRecord and IdeaOutcome.
- Technical failure creates no normalized hypothesis delta.
- Experiment persisted/outcome missing can be reconstructed idempotently.
- Experiment tools never return unexecuted ideas.
- Mandatory packets clearly distinguish deferred, rejected, selected, and evaluated ideas.
- Similarity analysis and executed-record search remain separate concerns.
- Coding-agent and MCP subprocess environments contain no embedding API key.

## Definition of done

- The full link `IdeaRecord -> SearchNode -> ExperimentRecord -> IdeaOutcome`
  can be queried and audited.
- Every experiment-history caller constructs and consumes the strict v3 record.
- No candidate is inserted into ExperimentHistoryStore before node execution.
- Orchestrator write ordering matches the architecture design.

## Non-goals

- Candidate ranking.
- Parent selection.
- Embedding model selection for the candidate analyzer.
- Checkpoint default switching.

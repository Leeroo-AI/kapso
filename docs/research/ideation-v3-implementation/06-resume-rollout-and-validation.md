# M6 — resume, rollout, and system validation

Parent plan: [`00-orchestrator-plan.md`](00-orchestrator-plan.md)

Depends on: M1–M5.

## Objective

Prove the integrated system is crash-safe, strict, observable, free of the
superseded ideation implementation, and production-ready end to end.

## Owned responsibilities

- Strategy checkpoint references to the idea archive.
- Cross-store resume reconciliation.
- Removal of superseded code, persisted shapes, config, prompts, and tests.
- Failure injection at every durable boundary.
- Scenario replay and end-to-end validation.
- Production activation evidence.
- Operator-facing documentation and diagnostics.

## Implemented code surface

```text
generic/strategy.py                 # sequential post-M4 resume additions
generic/ideation/engine.py          # phase-exact batch continuation
configuration and docs surfaces

tests/
  test_ideation_resume.py
  test_ideation_engine.py
  test_ideation_coding_agents.py
  test_ideation_v3_integration.py
```

## Checkpoint tasks

- [x] Persist active batch ID and archive revision in strategy state or the
      appropriate checkpoint layer.
- [x] Persist CLI role/model configuration and embedding provider/model identity
      in the configuration fingerprint, never credentials.
- [x] Reject incompatible pre-v3 checkpoint/archive state explicitly.
- [x] Validate every v3 node's `idea_id` and `selection_batch_id`.
- [x] Reconcile archive state before permitting another iteration.
- [x] Preserve existing contiguous node-history and parent-lineage checks.

## Reconciliation matrix

| Durable state | Required resume action |
|---|---|
| No batch | Start a new batch |
| `PLANNED` | Resume generation only if context hash still matches |
| `GENERATED` | Reuse pool; analyze |
| `ANALYZED` | Reuse facts; select |
| `SELECTED`, no node | Idempotently bridge the selected idea |
| Linked node implementing | Resume/recover through existing experiment lifecycle |
| Finalized node, no ExperimentRecord | Recreate executed projection |
| ExperimentRecord exists, no IdeaOutcome | Reconstruct outcome; do not rerun |
| Conflicting idea/node link | Fail loudly with typed corruption error |
| Problem, iteration, parent snapshot, or frozen context changed | Fail loudly; never mutate or silently replace the active batch |

## Superseded-code removal

- [x] Delete the old single-solution generation and ephemeral ensemble path.
- [x] Delete the old selector implementation and superseded prompts.
- [x] Delete config fields that only supported the prior path.
- [x] Delete permissive old checkpoint/ExperimentRecord loaders.
- [x] Rewrite callers and tests to construct the strict v3 shapes.
- [x] Search the repository for legacy names and prove no reachable or dead
      compatibility path remains.

## Durable-seam validation

| Durable seam | Evidence |
|---|---|
| Completed coding-agent operation while a batch remains `PLANNED` | Coding-agent durable replay tests and engine planned-resume coverage |
| Candidate pool at `GENERATED` | `test_generated_batch_resume_reuses_the_persisted_candidate_pool` |
| Analysis at `ANALYZED` | Selector-failure resume test and credentialed analyzed-phase resume |
| Decision at `SELECTED` | Engine selection and parent-bridge tests |
| Node link at `BRIDGED` | Parent bridge and bridged same-node recovery tests |
| New recovery revision in experiment memory ahead of checkpoint | Store-ahead recovery reconciliation test |
| Finalized checkpoint node without record | `test_finalized_checkpoint_node_recreates_record_and_outcome` |
| Experiment record without outcome | `test_experiment_record_recreates_node_and_outcome_without_rerun` |
| Archive/outcome ahead of checkpoint | `test_generic_ideation_archive_memory_checkpoint_and_resume_are_one_lifecycle` |
| Conflicting or corrupt state | Strict archive, checkpoint, identity, and revision tests |

Candidate output is durable per operation as `result.json`; the archive then
persists the complete candidate pool atomically. There is deliberately no
partial-candidate-pool archive state.

## Scenario replay

Create compact serialized fixtures rather than depending on historical remote
worktrees:

- cold start with no comparable result;
- same-family candidate collapse;
- credible improvement with an atomic lever;
- narrative diagnosis contradicted by slice metrics;
- high-impact language/subgroup gap repeatedly deferred;
- proxy leader failing full validation;
- technical failure with recoverable committed branch;
- minimizing objective;
- delivery-grade incumbent at the reserve boundary (must finalize); and
- delivery-grade incumbent without enough post-reserve capacity.

Each fixture asserts mode, directive, operator mix, eligible pool, selected or
terminal action, and gap effects.

## Activation sequence

1. Land strict modules in dependency order.
2. Connect v3 as the sole ideation path.
3. Delete superseded code, prompts, config, fixtures, and tests.
4. Run unit, integration, resume, failure-injection, and scenario suites.
5. Run offline replay over captured historical candidate outputs where useful.
6. Run full end-to-end campaigns with deterministic evaluators and CLI-provider
   boundary fakes. The canonical archive-ahead crash window is covered by
   `tests/test_ideation_v3_integration.py`.
7. Run a production-environment smoke campaign through real configured CLIs
   and the OpenAI embedding provider when credentials and capacity permit.
8. Publish completion evidence against every design acceptance criterion.

The activation sequence is complete for the Generic ideation boundary. The
credentialed smoke used the configured Codex and Claude Code generators, the
real OpenAI embedding endpoint, and the configured Codex selector. It persisted
two eligible candidates, resumed from `ANALYZED` without repeating generation
or embeddings, and reached `SELECTED`. The deterministic E2E separately covers
the Generic execution bridge, ExperimentRecord/IdeaOutcome write ordering, stale
checkpoint reconciliation, and idempotent replay.

## Activation metrics

- best delivery-validated utility per wall-clock and cost;
- invalid/duplicate candidate rate;
- descriptor-collapse and repair rate;
- high-impact gap debt and closure rate;
- selector hard-rule violation count, which must remain zero;
- predicted versus realized gain/cost calibration;
- proxy-to-delivery divergence;
- technical versus hypothesis failure classification; and
- crash/resume consistency, which must be exact.

## Definition of done

- Exact v3 checkpoints, operation replay, phase-exact batch resume, and
  cross-store reconciliation have named automated tests.
- Incompatible pre-v3 state fails loudly and no Generic ideation compatibility
  path remains.
- The sole v3 path runs end to end without taking budget, fidelity, Git,
  implementation, or evaluation authority from their existing owners.
- Automated tests isolate external CLIs and OpenAI; a separate credentialed
  smoke proves the configured Codex, Claude Code, embedding, and selector path.
- Benchmark implementation/evaluator smoke remains a deployment gate outside
  this portable ideation suite.

## Non-goals

- Live shadow generation.
- Learned cross-campaign scheduling.
- Benchmark-specific policy branches.

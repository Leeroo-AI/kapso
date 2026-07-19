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

## Proposed code surface

```text
generic/strategy.py                 # sequential post-M4 resume additions
execution/run_checkpoint.py         # only if schema-level fields are needed
configuration and docs surfaces

tests/
  test_ideation_resume.py
  test_ideation_reconciliation.py
  test_ideation_v3_integration.py
  fixtures/ideation_v3/
```

## Checkpoint tasks

- [ ] Persist active batch ID and archive revision in strategy state or the
      appropriate checkpoint layer.
- [ ] Persist CLI role/model configuration and embedding provider/model identity
      in the configuration fingerprint, never credentials.
- [ ] Reject incompatible pre-v3 checkpoint/archive state explicitly.
- [ ] Validate every v3 node's `idea_id` and `selection_batch_id`.
- [ ] Reconcile archive state before permitting another iteration.
- [ ] Preserve existing contiguous node-history and parent-lineage checks.

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
| Context changed before execution | Retain and abandon old batch; create a new batch |

## Superseded-code removal

- [ ] Delete the old single-solution generation and ephemeral ensemble path.
- [ ] Delete the old selector implementation and superseded prompts.
- [ ] Delete config fields that only supported the prior path.
- [ ] Delete permissive old checkpoint/ExperimentRecord loaders.
- [ ] Rewrite callers and tests to construct the strict v3 shapes.
- [ ] Search the repository for legacy names and prove no reachable or dead
      compatibility path remains.

## Failure-injection tests

Inject termination after:

1. batch creation;
2. first candidate persistence;
3. full pool persistence;
4. analysis persistence;
5. selection persistence;
6. node linkage;
7. implementation commit;
8. internal evaluation;
9. external candidate evaluation;
10. ExperimentRecord persistence; and
11. IdeaOutcome persistence before checkpoint.

For every boundary assert no duplicated generation, node, experiment, outcome,
gap effect, or budget attribution.

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
- delivery-grade incumbent with one affordable complete opportunity probe; and
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
   boundary fakes.
7. Run a production-environment smoke campaign through real configured CLIs
   and the OpenAI embedding provider when credentials and capacity permit.
8. Publish completion evidence against every design acceptance criterion.

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

- Every design acceptance criterion has a named automated test or explicit
  evidence artifact.
- Every transaction boundary survives injected termination.
- Incompatible pre-v3 state fails loudly and no compatibility code remains.
- V3 can run end-to-end without changing existing execution authority.
- Tests prove every reasoning call uses a fake Codex/Claude CLI runner and only
  the embedding provider can reach the direct OpenAI client.
- Completion evidence proves the sole v3 path works end to end.

## Non-goals

- Live shadow generation.
- Learned cross-campaign scheduling.
- Benchmark-specific policy branches.

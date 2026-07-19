# M6 — resume, rollout, and system validation

Parent plan: [`00-orchestrator-plan.md`](00-orchestrator-plan.md)

Depends on: M1–M5.

## Objective

Prove the integrated system is crash-safe, backward-compatible, observable,
and ready to become the default without relying on live double-generation.

## Owned responsibilities

- Strategy checkpoint references to the idea archive.
- Cross-store resume reconciliation.
- Legacy compatibility and explicit optional backfill.
- Failure injection at every durable boundary.
- Scenario replay and end-to-end validation.
- Temporary mode configuration and activation evidence.
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

- [ ] Persist `ideation_mode`, active batch ID, archive schema version, and
      archive revision in strategy state or the appropriate checkpoint layer.
- [ ] Reject silent resume across different ideation modes.
- [ ] Load legacy state with no fabricated idea links.
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

## Legacy tasks

- [ ] Keep null idea links valid for pre-v3 nodes and ExperimentRecords.
- [ ] If backfill is enabled, create deterministic `LEGACY_EXECUTED` ideas from
      selected solution text with explicit projection provenance.
- [ ] Never infer historical candidate pools, operators, selector reasoning,
      or rejected ideas.
- [ ] Make backfill idempotent and separately auditable.
- [ ] Preserve legacy mode until v3 acceptance is complete.

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

## Rollout sequence

1. Land all modules with `legacy` default.
2. Run unit, integration, resume, and scenario suites.
3. Run offline replay over captured legacy candidate outputs where available.
4. Run explicit `v3` smoke campaigns with deterministic/mock evaluators.
5. Run bounded real campaigns and compare delivery utility, cost, duplicate
   rate, gap debt, and failure rate against legacy baselines.
6. Publish an activation report against the design acceptance criteria.
7. Change default to `v3` only after the report passes review.
8. Remove legacy mode later, in a separate cleanup after resume support policy
   permits it.

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
- Legacy and v3 resume semantics are deterministic and non-interchangeable.
- V3 can run end-to-end without changing existing execution authority.
- Activation report justifies the default switch using measured outcomes.

## Non-goals

- Removing legacy code in the activation change.
- Live shadow generation.
- Learned cross-campaign scheduling.
- Benchmark-specific policy branches.

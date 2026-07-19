# M2 — evidence, policy, and proposal operators

Parent plan: [`00-orchestrator-plan.md`](00-orchestrator-plan.md)

Depends on: M1 contract freeze.

## Objective

Implement the deterministic decision plane that converts campaign facts and
capacity into a reproducible `SearchDirective`. This module must be testable
without a coding-agent call, MCP server, or writable repository.

## Owned responsibilities

- Campaign evidence snapshot construction.
- Objective-direction normalization and comparable-result filtering.
- Claim provenance and contradiction detection inputs.
- Evaluation-gap state and priority computation.
- State-machine precedence.
- Operator allocation and concrete parent plans.
- Resurfacing eligibility for deferred ideas.

## Proposed code surface

```text
generic/ideation/
  evidence.py
  policy.py
  operators.py

tests/
  test_ideation_evidence.py
  test_ideation_policy.py
  test_ideation_operators.py
```

## Evidence builder tasks

- [x] Accept read-only `SearchNode` projections, idea archive snapshot,
      evaluation metadata, and capacity snapshot.
- [x] Normalize maximize/minimize objectives to utility.
- [x] Build the causal spine: incumbent, latest attempt, linked ideas, parent
      lineage, and relevant negative evidence.
- [x] Keep observations, hypotheses, and constraints distinct.
- [x] Mark missing evidence `INSUFFICIENT`; never zero-fill unknowns.
- [x] Determine comparability from evaluator/fidelity/seed metadata rather than
      score presence alone.
- [x] Estimate a noise floor from comparable repeats when available.
- [x] Identify proxy divergence, surprising gains, plateaus, and technical-only
      failures as typed facts.
- [x] Produce a content-addressed evidence snapshot ID.

## Gap ledger tasks

- [x] Accept typed gap declarations from evaluation profiles and outcomes.
- [x] Allow only evaluation outcomes to close or mark gaps inconclusive.
- [x] Compute priority from impact, evidence confidence, expected uncertainty
      reduction, and measured cost information.
- [x] Use age/deferral only as starvation-resistant tie-breakers.
- [x] Persist explicit conservative assumptions when a factor is unavailable.
- [x] Increment debt on justified selector deferral without forcing blind
      execution.

## Policy tasks

- [x] Implement precedence: `FINALIZE`, `RECOVER`, `BOOTSTRAP`, `VERIFY`,
      `EXPLOIT`, `EXPLORE`.
- [x] Consume fidelity/budget decisions; do not create a second runtime model.
- [x] Treat `FINALIZE` as an action, not a generator stance.
- [x] Admit an opportunity probe only under the design's incumbent, reserve,
      completion, and expected-value conditions.
- [x] Emit structured reasons referencing evidence IDs.
- [x] Guarantee identical inputs produce identical `PolicyDecision`.

## Operator and parent-plan tasks

- [x] Implement briefs for independent draft, target gap, atomic refine,
      ablation, mechanism shift, crossover, verify, and recover.
- [x] Assign genuinely distinct descriptors rather than persona wording.
- [x] Reserve one actionable high-priority gap slot when required.
- [x] Resolve candidate parent choices to typed plans, not mutable branches.
- [x] Ensure crossover has one implementation parent and explicit read-only
      sources.
- [x] Identify deferred ideas eligible for resurfacing only after a relevant
      condition changes.

## Tests

- Cold start chooses `BOOTSTRAP`.
- Recoverable timeout chooses `RECOVER` before bootstrap/explore.
- Proxy/full divergence chooses `VERIFY`.
- Credible comparable improvement with supported lever chooses `EXPLOIT`.
- Plateau or contradicted lever chooses `EXPLORE`.
- Insufficient terminal capacity chooses `FINALIZE`.
- Maximize and minimize histories produce equivalent normalized decisions.
- High-impact gap outranks merely old low-impact gap.
- Repeated deferral increases debt but never directly forces selection.
- Same-family history causes a mechanism-shift operator.
- Deferred idea resurfaces only when recorded conditions changed.

## Definition of done

- Policy output is a pure function over serialized fixtures.
- No code in this module imports an agent adapter or mutates strategy state.
- Scenario fixtures reproduce the design's run 16/17/19 lessons abstractly.
- Every directive reason is auditable back to evidence.

## Non-goals

- Prompt construction and coding-agent calls.
- Semantic embeddings.
- Branch materialization.
- Actual budget admission or fidelity selection.

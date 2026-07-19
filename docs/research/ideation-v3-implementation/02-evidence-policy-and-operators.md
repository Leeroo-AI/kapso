# M2 — evidence, policy, and proposal operators

Parent plan: [`00-orchestrator-plan.md`](00-orchestrator-plan.md)

Depends on: M1 contract freeze.

## Objective

Implement the deterministic decision plane that converts campaign facts and
capacity into a reproducible `SearchDirective`. This module must be testable
without an LLM, MCP server, or writable repository.

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

- [ ] Accept read-only `SearchNode` projections, idea archive snapshot,
      evaluation metadata, and capacity snapshot.
- [ ] Normalize maximize/minimize objectives to utility.
- [ ] Build the causal spine: incumbent, latest attempt, linked ideas, parent
      lineage, and relevant negative evidence.
- [ ] Keep observations, hypotheses, and constraints distinct.
- [ ] Mark missing evidence `INSUFFICIENT`; never zero-fill unknowns.
- [ ] Determine comparability from evaluator/fidelity/seed metadata rather than
      score presence alone.
- [ ] Estimate a noise floor from comparable repeats when available.
- [ ] Identify proxy divergence, surprising gains, plateaus, and technical-only
      failures as typed facts.
- [ ] Produce a content-addressed evidence snapshot ID.

## Gap ledger tasks

- [ ] Accept typed gap declarations from evaluation profiles and outcomes.
- [ ] Allow only evaluation outcomes to close or mark gaps inconclusive.
- [ ] Compute priority from impact, evidence confidence, expected uncertainty
      reduction, and measured cost information.
- [ ] Use age/deferral only as starvation-resistant tie-breakers.
- [ ] Persist explicit conservative assumptions when a factor is unavailable.
- [ ] Increment debt on justified selector deferral without forcing blind
      execution.

## Policy tasks

- [ ] Implement precedence: `FINALIZE`, `RECOVER`, `BOOTSTRAP`, `VERIFY`,
      `EXPLOIT`, `EXPLORE`.
- [ ] Consume fidelity/budget decisions; do not create a second runtime model.
- [ ] Treat `FINALIZE` as an action, not a generator stance.
- [ ] Admit an opportunity probe only under the design's incumbent, reserve,
      completion, and expected-value conditions.
- [ ] Emit structured reasons referencing evidence IDs.
- [ ] Guarantee identical inputs produce identical `PolicyDecision`.

## Operator and parent-plan tasks

- [ ] Implement briefs for independent draft, target gap, atomic refine,
      ablation, mechanism shift, crossover, verify, and recover.
- [ ] Assign genuinely distinct descriptors rather than persona wording.
- [ ] Reserve one actionable high-priority gap slot when required.
- [ ] Resolve candidate parent choices to typed plans, not mutable branches.
- [ ] Ensure crossover has one implementation parent and explicit read-only
      sources.
- [ ] Identify deferred ideas eligible for resurfacing only after a relevant
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

- Prompt construction and LLM calls.
- Semantic embeddings.
- Branch materialization.
- Actual budget admission or fidelity selection.

# Ideation v3 implementation — orchestrator plan

Status: **planning baseline**

Design authority: [`../ideation-v3-design.md`](../ideation-v3-design.md)

This is the controlling implementation plan for ideation v3. It coordinates
six module plans, records cross-module decisions, defines integration gates,
and is the only plan allowed to change dependency order or shared contracts.

The module plans are executable work packages. They may refine internal tasks,
but a change to a shared type, lifecycle transition, write ordering, or module
boundary must first be recorded here.

## Outcome

Replace GenericSearch's ephemeral "generate one solution string" stage with a
durable, evidence-directed ideation pipeline while preserving:

- orchestrator authority over budget and fidelity;
- SearchNode/checkpoint authority over executable campaign state;
- existing implementation, evaluation-integrity, and feedback behavior;
- ExperimentHistoryStore as the executed-memory projection;
- ref-correct Git lineage; and
- backward-compatible resume for legacy campaigns.

## Planning structure

| ID | Module plan | Responsibility | Depends on |
|---|---|---|---|
| M1 | [`01-domain-and-archive.md`](01-domain-and-archive.md) | Shared types, lifecycle rules, atomic `IdeaArchive` | — |
| M2 | [`02-evidence-policy-and-operators.md`](02-evidence-policy-and-operators.md) | Evidence snapshots, gaps, state policy, operator and parent plans | M1 |
| M3 | [`03-candidate-pipeline.md`](03-candidate-pipeline.md) | Generation adapters, analysis, duplicate alarms, bounded repair, selection | M1, M2 |
| M4 | [`04-generic-search-bridge.md`](04-generic-search-bridge.md) | `IdeationEngine`, GenericSearch integration, parent materialization, node linkage | M1, M2, M3 |
| M5 | [`05-experiment-memory-and-outcomes.md`](05-experiment-memory-and-outcomes.md) | Experiment projection, idea/experiment retrieval, finalized outcome write-back | M1; integrates after M4 |
| M6 | [`06-resume-rollout-and-validation.md`](06-resume-rollout-and-validation.md) | Checkpoint reconciliation, legacy compatibility, end-to-end tests, rollout | M1–M5 |

The plans intentionally group strongly coupled responsibilities. Creating one
plan per class would distribute single invariants across too many owners and
make integration harder to reason about.

## Dependency graph

```mermaid
flowchart LR
    M1["M1: Domain and Archive"]
    M2["M2: Evidence, Policy, Operators"]
    M3["M3: Candidate Pipeline"]
    M4["M4: GenericSearch Bridge"]
    M5["M5: Experiment Memory and Outcomes"]
    M6["M6: Resume, Rollout, Validation"]

    M1 --> M2
    M1 --> M3
    M1 --> M4
    M1 --> M5
    M2 --> M3
    M2 --> M4
    M3 --> M4
    M4 --> M5
    M4 --> M6
    M5 --> M6
```

M2 and the additive, M1-dependent portion of M5 may be developed in parallel.
M3 starts when M2's contracts are stable. M4 is the first point at which v3
can execute a real experiment. M6 begins only after the live write path is
complete.

## Shared contract freeze

M1 must land these names and semantics before dependent implementation begins:

```text
IdeaId
IdeaBatchId
IdeaRecord
IdeaBatch
IdeaDescriptor
IdeaOutcome
EvidenceClaim
EvaluationGap
CampaignEvidenceSnapshot
IdeationCapacityView
PolicyDecision
SearchDirective
OperatorBrief
ParentPlan
CandidateAnalysis
SelectionDecision
```

The freeze covers:

- ID stability and JSON representation;
- enum values and legal lifecycle transitions;
- optional versus required fields;
- objective-normalized utility semantics;
- `origin_batch_id` versus `selected_in_batch_id`;
- idea-to-node and batch-to-node join keys; and
- schema-version and migration behavior.

Fields may be added compatibly after the freeze. Renames, removals, semantic
changes, and new lifecycle edges require a decision recorded below.

`IdeationCapacityView` is a read-only adapter over the existing
`BudgetSnapshot`, `FidelityDecision`, and fidelity timing authority. It must not
introduce an ideation-owned runtime estimator or reserve calculation.

## Proposed package boundary

Keep v3 internal to GenericSearch until another strategy needs it:

```text
src/kapso/execution/search_strategies/generic/ideation/
  __init__.py
  types.py
  archive.py
  evidence.py
  policy.py
  operators.py
  generator.py
  analyzer.py
  selector.py
  engine.py
  outcomes.py
```

This avoids prematurely making strategy-specific assumptions part of Kapso's
global execution API. Shared promotion can happen later from demonstrated use.

## File ownership during implementation

To prevent parallel plans from repeatedly editing the same high-conflict files:

| Surface | Primary owner | Rule |
|---|---|---|
| New `generic/ideation/` types and archive | M1 | Dependent plans import; they do not redefine |
| `evidence.py`, `policy.py`, `operators.py` | M2 | Pure logic; no strategy mutation |
| Generator/analyzer/selector and ideation prompts | M3 | No direct `GenericSearch.run()` edits |
| `generic/strategy.py`, parent resolution, SearchNode link fields | M4 | Sole owner until M4 integration lands |
| Experiment store, experiment MCP gate, idea-history MCP wiring, orchestrator outcome hook | M5 | Sole owner of cross-store write order |
| Checkpoint reconciliation, compatibility flag, full integration fixtures | M6 | Starts after M4 and M5 to avoid overlapping scaffolding |

When sequential work requires a later module to touch an earlier owner's file,
the later plan records the exact integration edit and tests it against the
earlier module's contract.

## Delivery strategy

Use small mergeable slices, but do not expose a partially authoritative v3
path. The runtime compatibility switch is temporary:

```text
ideation_mode = legacy | v3
```

Initial default: `legacy`.

The flag and its value must be checkpointed or included in the configuration
fingerprint so resume cannot switch semantics silently. After all acceptance
gates pass and benchmark replay shows no regression, change the default to
`v3`. Remove the legacy path only in a later cleanup change.

Do not add a live shadow mode that doubles LLM generation cost. Use deterministic
fixtures, captured candidate pools, and offline replay for comparisons.

## Integration waves

### Wave 1 — persistence substrate

Deliver M1:

- immutable JSON-compatible domain types;
- transition validation;
- versioned, atomic archive;
- idempotent mutation operations;
- archive query primitives; and
- corruption and round-trip tests.

Gate: the archive can represent the full design lifecycle without any LLM or
GenericSearch dependency.

### Wave 2 — deterministic decision plane

Deliver M2 and the additive schema portion of M5:

- evidence normalization;
- maximize/minimize correctness;
- claim and gap state transitions;
- deterministic policy precedence;
- operator and parent-plan construction;
- optional idea linkage fields on executed records; and
- backward-compatible experiment-store loading.

Gate: frozen fixtures deterministically produce expected modes, gap priorities,
and directives.

### Wave 3 — candidate plane

Deliver M3:

- structured generator results;
- independent operator briefs;
- archived-idea resurfacing;
- exact duplicate and semantic-neighbor facts;
- feasibility and evidence checks;
- one bounded repair request; and
- structured selector decision and fallback behavior.

Gate: captured pools can be analyzed and selected without starting an
experiment, and every considered idea is persisted.

### Wave 4 — live search bridge

Deliver M4:

- engine transaction ordering;
- current agent adapters reused through the candidate interface;
- per-candidate parent snapshots;
- selected idea persisted before node creation;
- linked SearchNode construction; and
- existing implementation/evaluation path unchanged after the bridge.

Gate: one v3 iteration produces the same valid executed node shape plus stable
idea provenance.

### Wave 5 — outcome loop

Complete M5:

- finalized experiment projection carries idea links;
- outcome update runs after orchestrator-side evaluation;
- experiment and idea retrieval remain separate;
- missing outcome can be reconstructed idempotently; and
- MCP output labels executed versus proposed work unambiguously.

Gate: `IdeaRecord -> SearchNode -> ExperimentRecord -> IdeaOutcome` is
reconstructable after process restart.

### Wave 6 — reliability and activation

Deliver M6:

- checkpoint/archive reconciliation;
- legacy checkpoint and experiment-memory compatibility;
- failure-injection tests at every transaction boundary;
- abstract run 16/17/19 scenario replays;
- end-to-end v3 opt-in test; and
- default-switch evidence report.

Gate: all acceptance criteria from the design document pass before `v3`
becomes the default.

## Parallel execution guidance

The safe parallelization pattern is:

1. M1 runs alone until the contract freeze is reviewed.
2. M2 and M5's additive ExperimentRecord/schema work may run in parallel.
3. M3 starts after M2; M5 may continue on retrieval tests without touching the
   orchestrator hook.
4. M4 integrates the completed M1–M3 surfaces and owns `strategy.py`.
5. M5 then adds the live orchestrator outcome hook against M4's stable node
   links.
6. M6 runs as the integration/reliability owner after all write paths land.

Prefer one reviewable change set per module. If M5 is split, use one additive
memory-schema change and one later outcome-hook change under the same plan.
Never run M4 and M6 edits to `strategy.py` concurrently.

## Cross-module system flow

```mermaid
sequenceDiagram
    participant O as Orchestrator
    participant G as GenericSearch
    participant P as Evidence and Policy
    participant A as IdeaArchive
    participant C as Candidate Pipeline
    participant X as Experiment Lifecycle
    participant E as ExperimentHistoryStore

    O->>G: admitted iteration plus capacity/fidelity view
    G->>P: node history, idea history, capacity
    P-->>G: evidence snapshot and directive
    G->>A: create PLANNED batch
    G->>C: generate, resurface, analyze, select
    C->>A: persist each lifecycle boundary
    C-->>G: selected idea and frozen parent plan
    G->>X: create linked node and execute
    X-->>O: finalized strategy node
    O->>O: attach external evaluation
    O->>E: add executed projection
    O->>A: record final idea outcome
    O->>O: checkpoint node and archive revision
```

## Global invariants

All module plans must preserve these invariants:

1. A generated candidate is persisted before it can be selected.
2. A selected idea and decision are persisted before a SearchNode is created.
3. Every v3 SearchNode has exactly one `idea_id` and one
   `selection_batch_id`.
4. One idea links to at most one experiment node; recovery reuses that node.
5. Experiment memory contains only work that reached the experiment lifecycle.
6. Idea memory never supplies score truth independently of linked evaluation.
7. Budget/fidelity code is the only admission and runtime-estimation authority.
8. Parent branch, parent node, materialized ref, diff base, and feedback base
   describe the same immutable parent snapshot.
9. Technical failure does not update hypothesis value or close an evaluation
   gap.
10. An invalid or incomparable evaluation cannot update normalized utility.
11. Semantic similarity is an observation, not a reward or hard rejection.
12. Resume reuses durable work and fails loudly on conflicting links.

## Test strategy

Each module owns unit tests; M6 owns integration and failure-injection tests.

### Required test layers

| Layer | Purpose |
|---|---|
| Domain tests | Serialization, validation, state transitions, stable IDs |
| Store tests | Atomic writes, revision checks, migration, corruption handling |
| Pure-policy tests | Mode precedence, utility direction, gap priority, operator allocation |
| Candidate tests | Structured parsing, duplicate alarms, repair quota, selector hard rules |
| Bridge tests | Parent-ref correctness, node linkage, idempotent creation |
| Memory tests | Executed-only projection, idea links, objective-aware retrieval |
| Resume tests | Crash at every durable boundary and deterministic recovery |
| Scenario replay | Cold start, collapse, contradiction, subgroup gap, proxy divergence, terminal reserve |
| End-to-end | One opt-in v3 run using mocked agents and deterministic evaluator |

### Required adversarial fixtures

- minimizing objective where raw descending order is wrong;
- semantically similar idea with a materially different parent/test;
- exact duplicate with cosmetic rewording;
- all ensemble members returning one approach family;
- selected idea persisted immediately before a crash;
- experiment persisted immediately before idea outcome write;
- invalid evaluation that appears to improve the score;
- technical failure followed by successful recovery;
- high-impact gap repeatedly deferred; and
- delivery-grade incumbent with insufficient capacity for another comparable
  run.

## Definition of complete

The implementation is complete only when:

- all twelve design acceptance criteria pass;
- every module plan's definition of done is satisfied;
- legacy checkpoints load without fabricated history;
- v3 checkpoints cannot resume under legacy semantics or vice versa;
- the candidate population and selector decision survive a crash;
- experiment and idea retrieval never conflate proposed and executed work;
- benchmark scenario replay demonstrates the intended state transitions; and
- documentation for configuration, storage, and operator behavior is updated.

## Decision log

| ID | Decision | Reason |
|---|---|---|
| D1 | Keep v3 strategy-local under `generic/ideation/` | Avoid premature global API surface |
| D2 | Use six implementation plans | Match coupling and reduce coordination overhead |
| D3 | Keep archive separate from checkpoint and experiment JSON | Avoid mixed lifecycles and dual-write authority |
| D4 | Use explicit link IDs rather than embedded candidate pools | Preserve provenance without contaminating experiment records |
| D5 | Use `legacy|v3`, not live shadow generation | Avoid silently doubling campaign cost |
| D6 | Persist outcomes after orchestrator candidate evaluation | External metrics and validity must be attached first |
| D7 | Give `strategy.py` to M4 and `orchestrator.py` write ordering to M5 | Minimize high-conflict parallel edits |

## Progress ledger

Update this table as implementation work lands. Do not use prose status buried
inside module plans as the campaign-level source of truth.

| Module | Status | Implementation reference | Validation reference | Blocker |
|---|---|---|---|---|
| M1 Domain and Archive | Not started | — | — | — |
| M2 Evidence, Policy, Operators | Not started | — | — | M1 contract freeze |
| M3 Candidate Pipeline | Not started | — | — | M1, M2 |
| M4 GenericSearch Bridge | Not started | — | — | M1–M3 |
| M5 Experiment Memory and Outcomes | Not started | — | — | M1; live integration waits for M4 |
| M6 Resume, Rollout, Validation | Not started | — | — | M1–M5 |

## Plan-maintenance protocol

For every implementation change:

1. identify the owning module plan;
2. confirm its prerequisites are complete;
3. update the module checklist and tests;
4. record any cross-contract decision in this file before coding it;
5. update the progress ledger with the commit or pull request; and
6. run the integration gates for the current wave.

This file coordinates work. It should not duplicate the detailed task lists in
the module plans.

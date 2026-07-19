# Ideation v3 — evidence-directed exploration and exploitation

Status: **accepted design; implementation status is tracked in
[`ideation-v3-implemented-system.md`](ideation-v3-implemented-system.md)**

This document specifies the intended architecture and system flow for Kapso's
next ideation system. It supersedes the scheduling and persistence portions of
`worktree-posttrainbench:docs/research/ideation-v2-design.md`, while retaining
that proposal's strongest mechanisms: operator diversity, persistent candidate
pools, measured-gap coverage, embedding-based duplicate alarms, and
evidence-grounded selection.

Implementation planning is deliberately out of scope. The purpose of this
document is to establish stable responsibilities, contracts, invariants, and
end-to-end behavior before choosing files, phases, or migrations.

Implementation work is decomposed and coordinated by the
[ideation v3 implementation orchestrator](ideation-v3-implementation/00-orchestrator-plan.md).

## Executive decision

Kapso should use a **small, durable idea population controlled by a
deterministic evidence policy**.

The system does not try to maximize an abstract novelty score. It tries to
maximize delivery-validated improvement per unit of constrained experiment
capacity by:

1. recognizing whether the campaign needs bootstrap, recovery, verification,
   exploitation, or exploration;
2. generating candidates through distinct proposal operators;
3. retaining every candidate and its provenance;
4. treating semantic similarity only as a duplicate warning;
5. grounding causal claims and evaluation gaps in recorded evidence;
6. selecting one executable candidate under hard capacity and validation
   constraints; and
7. linking the selected idea to its experiment and observed outcome.

Kapso should not add MCTS, UCB, reinforcement learning, or a large population
scheduler at this stage. A typical campaign provides too few expensive
observations to estimate those policies reliably. Better proposal operators,
evidence quality, and lifecycle integrity are higher-leverage.

## Decisions relative to ideation v2

| Ideation v2 mechanism | Decision in v3 |
|---|---|
| Improve proposal operators before adding search-policy complexity | Retain |
| Persist selected and unselected candidates | Retain in a separate `IdeaArchive` |
| Use embeddings to detect repeated ideas | Retain as a warning, never a value or novelty score |
| Reserve a candidate for a measured gap | Retain, prioritized by impact and uncertainty rather than age alone |
| Audit selector diagnoses against score evidence | Retain and make it a structured selection output |
| Replace static ensemble personas with stance-specific briefs | Retain as operator briefs; task priors remain optional constraints |
| Derive `EXPLORE`, `EXPLOIT`, and `FINAL_GAMBLE` from experiment count | Replace with the evidence state machine; remove `FINAL_GAMBLE` |
| Estimate remaining full runs inside ideation | Reject; consume the existing fidelity/budget capacity contract |
| Keep every losing candidate on its associated experiment record | Reject; ideas and experiments have different lifecycles |
| Search similarity over selected and unselected text stored on experiments | Reject; expose separate idea and experiment retrieval |
| Never regenerate a collapsed candidate pool | Replace with one bounded diversity-repair round |
| Discard or rederive older stores on upgrade | Accept during pre-release development; v3 replaces the old shape without migration shims |

## The design in one diagram

```mermaid
flowchart LR
    subgraph ExistingAuthority["Existing control-plane authority"]
        ORCH["Orchestrator"]
        CAP["Budget and Fidelity Policy"]
        EXP["Experiment History"]
        EVAL["Evaluation Results"]
        REPO["Repository Memory"]
    end

    subgraph Ideation["Ideation subsystem inside GenericSearch"]
        EVID["Campaign Evidence Builder"]
        POLICY["Ideation Policy"]
        OPS["Operator Planner"]
        GEN["Candidate Generator"]
        ANALYZE["Candidate Analyzer"]
        SELECT["Candidate Selector"]
        ARCHIVE["Idea Archive"]
    end

    subgraph Execution["Existing experiment lifecycle"]
        BRIDGE["Experiment Bridge"]
        NODE["SearchNode and Git Branch"]
        RUN["Implement and Evaluate"]
        OUTCOME["Outcome Recorder"]
    end

    ORCH --> CAP
    CAP --> EVID
    EXP --> EVID
    EVAL --> EVID
    REPO --> EVID
    ARCHIVE --> EVID
    EVID --> POLICY
    POLICY --> OPS
    OPS --> GEN
    GEN --> ANALYZE
    ARCHIVE <--> GEN
    ANALYZE --> SELECT
    SELECT --> ARCHIVE
    SELECT --> BRIDGE
    BRIDGE --> NODE
    NODE --> RUN
    RUN --> EVAL
    EVAL --> EXP
    EXP --> OUTCOME
    OUTCOME --> ARCHIVE
    OUTCOME --> EVID
```

## Scope and boundaries

Ideation begins after the orchestrator admits another search action and ends
when a selected idea has been durably linked to an experiment node. It consumes
campaign evidence and capacity; it does not own either.

The following boundaries are mandatory:

- **Budget and fidelity own capacity.** Ideation consumes a capacity read model.
  It does not estimate full-run duration, alter the finalization reserve, or
  admit work independently.
- **SearchNode/checkpoint owns canonical executed state.** Scores, feedback,
  branches, failures, and evaluation attempts remain on executed nodes.
  `ExperimentHistoryStore` remains the durable executed-work retrieval
  projection for implementation/MCP workflows; ideation consumes the canonical
  node-history projection and does not treat the store as a second authority.
- **Idea history owns proposals.** Generated, rejected, selected, and
  not-yet-executed candidates remain idea records.
- **Evaluation owns truth.** A coding agent may identify an assumption or
  propose a test, but only recorded evaluation evidence can mark a gap tested
  or close a claim.
- **GenericSearch remains the strategy.** Ideation is an internal subsystem,
  not a competing search strategy or a second orchestrator.
- **Coding-agent CLIs propose and criticize. Deterministic code enforces.**
  Capacity, schema validity, lifecycle transitions, exact duplicates, evidence
  status, and resume behavior are not prompt responsibilities.

## Model invocation boundary

All generative or judgment-bearing ideation calls run through a coding-agent
CLI. The supported implementations are Codex CLI and Claude Code. Ideation must
not call a chat, completions, responses, or generic `LLMBackend` inference API
directly.

This boundary covers:

- initial candidate generation;
- diversity-repair generation;
- selector/critic judgment;
- optional claim or descriptor extraction that requires model reasoning; and
- any future model-assisted ideation operation.

The CLI runner contract supplies the prompt, read-only workspace, allowed
tools, model/effort configuration, timeout, telemetry, and structured output.
Codex and Claude Code may implement that contract differently, but downstream
ideation code receives the same result type. Candidate and selector roles may
independently choose either CLI.

The only direct model API boundary in ideation is embeddings:

```text
OpenAIEmbeddingProvider
  authentication: official SDK default credential discovery
  default model: text-embedding-3-small
  operation: embeddings only
  consumer: CandidateAnalyzer
```

`text-embedding-3-small` is the cost-oriented default for duplicate alarms; the
model remains configurable and every stored vector records its provider,
model, dimensions, and input hash. `text-embedding-3-large` may be selected
when multilingual or domain-specific calibration demonstrates a material
benefit. OpenAI documents both as current embedding models, with
[`text-embedding-3-large`](https://developers.openai.com/api/docs/models/text-embedding-3-large)
described as the more capable option and
[`text-embedding-3-small`](https://developers.openai.com/api/docs/models/text-embedding-3-small)
as the smaller option.

The provider uses the official OpenAI SDK's embeddings endpoint and default
credential discovery after Kapso's outer dotenv startup. Ideation code never
reads an environment variable or a `.env` path, logs or persists a key, or
passes the embedding credential to coding-agent subprocesses. Codex continues
to use CLI login or its dedicated CLI credentials, and Claude Code continues
to use its configured auth mode.

Embedding failure propagates and leaves the batch at its last durable state.
Similarity may be explicitly disabled in configuration, in which case exact
and descriptor checks run without semantic neighbors. There is no automatic
credential, timeout, rate-limit, or provider-error degradation. Cached vectors
are reused only when provider, model, dimensions, and input hash match.
Embedding call count, input tokens, duration, and model are recorded separately
from coding-agent telemetry.

## Connection to the current system

V3 replaces the former solution-generation seam; it is not a
new top-level execution path.

### Implemented v3 call flow

The sole Generic ideation path is:

```text
GenericSearch.run()
  -> IdeationEngine.run(evidence, capacity)
       -> persist PLANNED IdeaBatch with the complete frozen packet
       -> generate/resurface and persist IdeaRecords (GENERATED)
       -> analyze novelty, evidence, and feasibility (ANALYZED)
       -> persist SelectionDecision (SELECTED)
       -> return selected IdeaRecord + ResolvedParentSnapshot
  -> GenericSearch links idea_id + selection_batch_id to SearchNode (BRIDGED)
  -> _implement(), strict XML parsing, integrity, feedback, evaluation attempt
  -> EvidenceAuthor reviews the complete diff and evaluation record
  -> append node_history
  -> return node to Orchestrator
  -> optional external candidate evaluation
  -> ExperimentHistoryStore.add_experiment(node)
  -> GenericSearch.record_finalized_idea_outcome(node) (COMPLETED)
  -> write run checkpoint
```

Everything after node creation remains the existing experiment lifecycle. The
new subsystem changes how the solution and its parent are chosen, and preserves
the decision that led to the node.

### Implemented responsibility mapping

| Runtime seam | Responsibility |
|---|---|
| `GenericSearch.run()` | Entry point, frozen Git resolution, idea-to-node bridge, and existing execution lifecycle |
| `IdeationEngine.run()` | Phase-exact policy/generation/analysis/selection transaction ordering |
| `CandidateGenerator` | Concurrent structured candidates through Codex/Claude CLI roles |
| `CandidateAnalyzer` | Deterministic hard rules, descriptors, duplicates, embeddings, and one repair request |
| `CandidateSelector` | Structured CLI judgment over only eligible candidates |
| `IdeaArchive` | Atomic batches, candidates, facts, selection, link, and outcome state |
| `GenericSearch.dump_state()/load_state()` | Exact v3 checkpoint projection and archive-ahead validation |
| `Orchestrator._persist_finalized_candidates()` | ExperimentRecord before IdeaOutcome write ordering |
| `GenericSearch.reconcile_experiment_memory()` | Cross-store reconstruction and conflict rejection before another iteration |

### Parent resolution and repository context

Operators may propose from different parent plans without making branch lineage
ambiguous:

1. The policy determines the allowed parent plans from `node_history`.
2. The parent resolver converts each plan into an immutable
   `(branch_name, node_id, git_ref)` snapshot before generation.
3. Each operator brief names one concrete implementation parent.
4. The generator reads that parent's materialized ref. Members using the same
   ref may share a read-only view; members using different refs receive
   separate read-only views.
5. The selected candidate carries its frozen parent snapshot into the inline
   GenericSearch node bridge.
6. Node lineage, implementation base, diff base, and feedback base all use that
   same snapshot, preserving the current ref-correctness guarantee.

For `CROSSOVER`, one branch is still the implementation parent. Other ideas or
experiments are cited sources, not additional Git parents.

### Prompt and tool connection

Candidate and repair calls receive the problem, frozen evidence and directive,
their operator brief and parent snapshot, and the complete prior idea/claim/gap
archive. The selector instead receives the eligible current/resurfaced pool and
its analyses; those analyses carry the precomputed semantic neighbors. The
evidence author receives the selected idea, compact experiment identity, exact
source references, and the complete diff/evaluation source material.
Codex/Claude roles may use their configured read-only tools, but proposal memory
does not depend on an MCP call and there is no separate idea-history gate.
Executed-experiment MCP tools remain available to implementation workflows.

## Connection to experiment memory

`ExperimentHistoryStore` and `IdeaArchive` are complementary projections over
different lifecycles:

| Store | Question answered | Written when | Contains |
|---|---|---|---|
| Search strategy checkpoint (`node_history`) | What is the canonical executable campaign state? | At normal checkpoint boundaries | Full `SearchNode` state, lineage, evaluation attempts, fidelity, telemetry |
| `ExperimentHistoryStore` | What was actually implemented and what happened? | After the orchestrator finishes candidate evaluation | Agent-queryable `ExperimentRecord` projections |
| `IdeaArchive` | What was considered, why was it selected or deferred, and what hypothesis did it express? | Before and throughout ideation; linked again after outcome | Batches, all ideas, descriptors, embeddings, analysis, decisions, idea outcomes |

They must not be collapsed into one JSON document. An unselected idea has no
score, branch, or implementation outcome, while an experiment must never appear
to have implemented an unselected candidate.

### Join keys

The join is explicit and bidirectional:

```text
IdeaRecord.experiment_node_id?  <->  SearchNode.idea_id?
                                      |
                                      v
                               ExperimentRecord.idea_id?

SearchNode.selection_batch_id?  ->   IdeaBatch.batch_id
```

Every experiment has a non-null `idea_id` and `selection_batch_id`. The new
design does not retain an unlinked legacy record shape.

`SearchNode.solution` and `ExperimentRecord.solution` remain snapshots of the
selected proposal. Consumers should not need the idea archive merely to
understand an old experiment. The ID link adds provenance; it does not replace
existing readable fields.

### Write ordering

The required ordering prevents partial state from being misinterpreted:

```mermaid
sequenceDiagram
    participant I as IdeaArchive
    participant G as GenericSearch
    participant W as Evidence Author
    participant O as Orchestrator
    participant E as ExperimentHistoryStore

    I->>I: Persist selected IdeaRecord and SelectionDecision
    I-->>G: SelectedIdea and ParentPlan
    G->>G: Create linked SearchNode
    G->>G: Implement and run internal evaluation
    G->>W: Review complete diff and evaluation record
    W-->>G: Attributable claims and gaps
    G-->>O: Return node
    O->>O: Attach external evaluation and integrity results
    O->>E: Persist finalized ExperimentRecord with idea_id
    O->>I: Record final IdeaOutcome from finalized node
    O->>O: Save checkpoint with node and archive revision
```

The outcome recorder runs after orchestrator-side evaluation because that is
when metrics and validity are final. If experiment-memory persistence succeeds
but the idea-outcome write fails, resume reconstructs the missing outcome from
the linked node or experiment record. It must not rerun the experiment.

### Executed-memory projection

`ExperimentRecord` remains an executed-only record. Generic records require the
idea and batch join keys and retain the complete execution projection:

```text
ExperimentRecord
  node_id
  execution_revision
  idea_id
  selection_batch_id
  parent_node_id?
  solution, branch_name, timestamp
  raw_score?
  objective_direction
  normalized_utility?
  feedback, error and integrity fields
  build_fidelity
  eval_fidelity
  validation_tier
  evaluation_attempts[]
  phase_telemetry
  duration_seconds?
  cost_usd?
```

Raw score remains intact. `normalized_utility` or an equivalent explicit
objective-direction argument makes `get_top_experiments()` correct for both
maximization and minimization. Parent selection and delivery selection continue
to use canonical `node_history`; this projection is for retrieval, evidence
assembly, and audit.

Experiment search uses executed content such as the selected solution,
feedback, technical difficulties, and errors. Idea embeddings use the complete
canonical proposal representation inside the trusted ideation process. The two
similarity concerns are never collapsed.

### MCP retrieval behavior

The existing experiment-memory tools keep their executed-only meaning:

```text
get_top_experiments
get_recent_experiments
search_similar_experiments
```

Their result format includes stable idea/batch links, parent, fidelity,
validation status, and objective-aware utility. Separate idea-history MCP tools
were intentionally not added: every ideation role receives the complete frozen
archive, gaps, claims, and precomputed neighbors in its mandatory packet. Free-
text idea similarity runs in the trusted Kapso process through
`OpenAIEmbeddingProvider`; coding-agent subprocesses never receive
`OPENAI_API_KEY`.

This gives the model an unambiguous distinction:

- "Experiment 7" means code ran and produced recorded evidence.
- "Idea A42, deferred" means it was considered but never fairly tested.
- "Idea A17 -> Experiment 7" means the original hypothesis and its observed
  execution are linked.

### Checkpoint and reconciliation

The idea archive is not duplicated inside the run checkpoint. The exact Generic
strategy state is:

```text
schema
campaign_id
idea_archive_schema
archive_revision
active_batch_id?
node_history
iteration_count
previous_errors
evaluation_integrity
scores_evaluator_id
evaluator_transition?
```

Each linked `SearchNode` independently stores `idea_id` and
`selection_batch_id`. On resume:

1. load and validate the archive;
2. load `node_history`;
3. verify every v3 node's idea and batch links;
4. resume `PLANNED`, `GENERATED`, or `ANALYZED` from its first unfinished
   phase and replay completed CLI operation results by operation ID;
5. reconstruct a same-node recoverable execution for a `BRIDGED` idea with no
   durable node/record;
6. recreate a missing ExperimentRecord or IdeaOutcome from the authoritative
   linked projection; and
7. fail loudly on changed frozen context or conflicting links.

Pre-v3 checkpoints and experiment records are unsupported after activation.
They may be discarded or re-derived from authoritative campaign artifacts;
the implementation contains no migration shim, nullable legacy link, or
fabricated historical idea population.

## Design principles

### 1. Separate ideas from experiments

An idea can exist without an experiment. It may be rejected, duplicated,
deferred, selected immediately before a crash, or combined with another idea.
Conversely, an experiment records what was actually implemented and evaluated.

Storing losing candidates as fields on an executed experiment breaks that
lifecycle and creates false similarity: an experiment could appear related
because of a rejected idea it never implemented. `IdeaRecord` and `SearchNode`
must therefore be separate entities connected by an optional explicit link.

### 2. State responds to evidence, not iteration number

"Explore early, exploit late" is a useful prior, not a policy. A late campaign
may need exploration after a plateau. An early campaign may need immediate
verification after a suspicious gain. A technical failure may require recovery
without generating a new hypothesis.

The policy uses campaign evidence, delivery risk, and executable capacity to
choose behavior.

### 3. Diversity is operational, not linguistic

Different wording or personas do not constitute different ideas. Diversity is
measured through proposal operator, intervention target, mechanism, parent
choice, and expected observable effect.

Embeddings can identify semantic neighbors, but do not determine whether an
idea is useful, surprising, or genuinely new in the scientific sense.

### 4. Every causal claim needs an evidence state

Feedback frequently mixes observation and explanation. The system must retain
the distinction:

- observation: "validation score fell from A to B";
- hypothesis: "the fall was caused by distribution mismatch";
- test: "evaluate the affected language slices"; and
- conclusion: established only after the test produces interpretable evidence.

Selectors may use hypotheses, but must label them as supported, contradicted,
or insufficient.

### 5. Preserve an incumbent before seeking upside

A risky experiment is harmless only when a delivery-grade incumbent is already
banked and finalization capacity is protected. "One run remains" does not by
itself justify a gamble.

### 6. Recovery is not ideation

A dependency error, timeout, malformed output, or incomplete implementation
does not disprove the selected hypothesis. Recover the same idea and branch
when feasible. Generate a new idea only after the original intervention was
implemented sufficiently to produce interpretable evidence, or when recovery
is no longer justified.

## Core vocabulary

### Campaign utility

All score comparisons use normalized utility:

```text
utility(score) = score       when the objective is maximized
utility(score) = -score      when the objective is minimized
```

This prevents history retrieval, parent selection, and policy logic from
silently assuming that larger raw values are always better.

### Delivery-grade incumbent

The best candidate that has passed the evaluation tier required for final
delivery. A proxy-only, partial, invalid, non-reproducible, or surprising result
is not delivery-grade even when its score is highest.

### Credible improvement

An improvement whose normalized delta exceeds the current noise threshold and
whose evaluation is comparable to the incumbent. The evidence builder derives
the threshold from repeated measurements when available; otherwise it records
that confidence is provisional rather than inventing precision.

### Evaluation gap

An important uncertainty about where or whether the solution works: language,
format, difficulty, subgroup, seed stability, holdout behavior, deployment
fidelity, or another measured task axis.

### Idea descriptor

A structured description of what the proposal changes. At minimum it includes:

- approach family;
- intervention target;
- mechanism;
- parent plan;
- expected observable effect; and
- targeted evaluation gaps.

Evaluation gaps and idea descriptors are separate. The first describes missing
knowledge about performance; the second describes diversity in proposed
interventions.

## Policy state machine

The policy emits one `PolicyDecision` for each admitted ideation batch. The
decision order is deterministic. The operator planner turns that decision into
the final `SearchDirective` consumed by generators and selectors.

```mermaid
stateDiagram-v2
    [*] --> CapacityCheck
    CapacityCheck --> Finalize: no complete admissible action
    CapacityCheck --> Recover: recoverable technical failure
    CapacityCheck --> Bootstrap: no valid comparable experiment
    CapacityCheck --> Verify: fidelity requires validation or incumbent evidence is fragile
    CapacityCheck --> Exploit: credible gain and supported lever
    CapacityCheck --> Explore: plateau, contradiction, gap debt, or no supported lever

    Recover --> Outcome
    Bootstrap --> Outcome
    Verify --> Outcome
    Exploit --> Outcome
    Explore --> Outcome
    Outcome --> CapacityCheck
    Finalize --> [*]
```

### Decision precedence

| Priority | Mode | Trigger | Primary intent |
|---|---|---|---|
| 1 | `FINALIZE` | No complete action fits while preserving delivery obligations | Return or validate the best deliverable result |
| 2 | `RECOVER` | The last selected idea lacks a fair evaluation because of a recoverable technical failure | Complete the existing intervention without changing its hypothesis |
| 3 | `BOOTSTRAP` | No valid comparable experiment exists | Establish a credible baseline and independent starting mechanisms |
| 4 | `VERIFY` | Fidelity requires promotion, or a result needed for the next decision is proxy-only, fragile, surprising, or contradictory | Determine whether the gain is real and deliverable |
| 5 | `EXPLOIT` | A credible gain exists and at least one causal lever is supported | Refine or compose around demonstrated signal |
| 6 | `EXPLORE` | Plateau, unsupported diagnosis, diversity collapse, high gap debt, or no demonstrated lever | Search a meaningfully different mechanism or reduce important uncertainty |

`FINALIZE` is a terminal action, not an ideation stance. `RECOVER` is likewise
an execution action: it resumes the same linked idea and experiment node with
zero candidate generation. There is no terminal opportunity-probe bypass. Once
the capacity authority can no longer admit the granted implementation and
evaluation while preserving reserve, policy returns `FINALIZE`. Kapso has no
calibrated authority for completion probability or expected positive utility;
inventing those facts would make a nominally careful "final gamble" less
trustworthy than the deterministic reserve boundary.

Novelty is not a terminal tie-breaker.

## Search directive

The composed directive is immutable for one batch:

```text
SearchDirective
  mode
  reasons[]
  evidence_snapshot_id
  capacity_snapshot_id
  allowed_parent_plans[]
  operator_briefs[]
  reserved_gap_id?
  candidate_quota
  repair_quota
  validation_requirements[]
  terminal_constraints
```

Generators and selectors receive the same directive. They cannot silently
reinterpret the campaign mode or reserve.

## Proposal operators

Operators describe how to produce an idea, not how to word the prompt. The
operator planner chooses a small set with deliberately different mechanisms.

| Operator | Purpose | Typical parent |
|---|---|---|
| `INDEPENDENT_DRAFT` | Establish a credible implementation from task evidence | Baseline |
| `TARGET_GAP` | Reduce a high-impact measured uncertainty or directly address it | Best or baseline |
| `ATOMIC_REFINE` | Change one diagnosed lever around a credible incumbent | Best valid |
| `ABLATE` | Remove or isolate a component to test causal contribution | Specific experiment |
| `MECHANISM_SHIFT` | Change approach family when the current family plateaus | Baseline or distinct elite |
| `CROSSOVER` | Apply a compatible mechanism from another promising lineage | One implementation parent plus explicit source ideas |
| `VERIFY` | Replicate, widen, or raise fidelity without changing the hypothesis | Incumbent |
| `RECOVER` | Complete the same idea after a technical failure | Failed experiment branch |

The planner may use task-specific priors, but priors constrain operators rather
than replacing them with static personas.

### Parent plans

The current global `best` or `baseline` policy is insufficient once operators
have different purposes. A directive therefore carries an explicit plan:

```text
ParentPlan.kind =
  BEST_VALID
  BASELINE
  SPECIFIC_EXPERIMENT
  RECOVER_BRANCH
```

`CROSSOVER` still has one Git implementation parent. Other ideas or experiments
are read-only sources whose relevant mechanism must be stated explicitly. This
preserves unambiguous branch lineage.

## Evidence system

### Campaign evidence snapshot

The `CampaignEvidenceBuilder` produces an immutable snapshot from:

- all linked successful and failed `SearchNode` projections;
- normalized scores and registered evaluation-attempt identity;
- feedback, technical failures, build fidelity, and parent-node lineage;
- archived causal claims and evaluation gaps; and
- idea identities needed to join the incumbent, latest experiment, and recent
  proposal history.

The supplied capacity view contributes typed policy signals but is persisted
separately on the `IdeaBatch`. Repository content is represented by frozen
parent refs and is read from those refs by candidate agents; it is not copied
into the evidence snapshot. The snapshot contains facts and labeled
inferences. It never rewrites source records.

### How previous work reaches the next ideation batch

The evidence builder projects every linked node in timestamp/node order and
marks the best comparable registered experiment plus the latest attempt. It
orders relevant idea IDs as incumbent, latest, then archive recency. Candidate
packets carry the complete prior idea archive; deterministic analysis later
computes semantic idea neighbors. Executed-experiment similarity retrieval,
per-descriptor elites, and slice-metric summaries are deferred capabilities,
not hidden behavior of the current builder.

An executed experiment is always displayed with the `IdeaRecord` that caused
it, when the link exists. This gives the next generator both the original
hypothesis and the observed result instead of showing score text without
intent.

Archived ideas may be reconsidered directly; they do not need to be
regenerated. The current implementation resurfaces only an unexecuted
`DEFERRED` idea when its last consideration used a different evidence-snapshot
ID or one of its targeted gaps is now closed. The current batch records that
exact reason.

Semantic retrieval currently operates over idea embeddings during candidate
analysis. Executed-experiment similarity retrieval is deferred; experiment and
idea identities are still joined explicitly rather than conflating their text.

### Evidence claims

```text
EvidenceClaim
  claim_id
  statement
  kind = OBSERVATION | HYPOTHESIS | CONSTRAINT
  status = SUPPORTED | CONTRADICTED | INSUFFICIENT
  source_refs[]
  affected_idea_ids[]
  affected_experiment_node_ids[]
  updated_at
```

The evidence builder may derive status from structured results. A coding-agent
critic may recommend a status, but deterministic validation must reject
nonexistent or incompatible references.

### Evaluation gaps

```text
EvaluationGap
  gap_id
  axis
  description
  state = OPEN | INCONCLUSIVE | CLOSED
  evidence_refs[]
  impact
  uncertainty
  estimated_cost
  deferral_count
  opened_at
  last_considered_at?
  closure_reason?
```

Only an evaluation outcome may transition `OPEN` to `INCONCLUSIVE` or `CLOSED`.
`CLOSED` means the evaluation supplied enough evidence to resolve the stated
uncertainty; `INCONCLUSIVE` means a test ran but could not resolve it. Merely
selecting or implementing a targeted idea does not change the gap state.

Gap priority is deterministic:

```text
priority = impact × evidence_confidence × uncertainty_reduction / estimated_cost
```

When one factor is unavailable, the record remains explicitly uncalibrated and
uses conservative defaults. Age and deferral count break ties and prevent
starvation; they do not make a low-impact gap automatically dominant.

One candidate slot is reserved for the highest-priority actionable gap when a
meaningful gap exists. The selector may defer that candidate, but must record a
reason. Repeated deferral increases debt until the system either executes a
test or closes/deprioritizes the gap with evidence.

## Durable idea model

### Idea batch

```text
IdeaBatch
  batch_id
  campaign_id
  iteration_index
  context_hash
  planning_archive_revision
  problem_statement
  evidence_snapshot
  capacity
  directive
  resolved_parents[]
  generated_idea_ids[]
  generation_calls[]
  resurfaced_ideas[]
  considered_idea_ids[]
  analyses[]
  embedding_telemetry?
  selection
  selection_call?
  abandoned_reason?
  status = PLANNED | GENERATED | ANALYZED | SELECTED | BRIDGED |
           COMPLETED | ABANDONED
  created_at
  updated_at
```

### Idea record

```text
IdeaRecord
  idea_id
  origin_batch_id
  selected_in_batch_id?
  proposal
  operator
  descriptor
  parent_plan
  resolved_parent
  assumptions[]
  evidence_refs[]
  directive_rationale
  evaluation_method
  resource_request
  created_at
  parent_idea_ids[]
  parent_experiment_node_ids[]
  target_gap_ids[]
  claim_ids[]
  resolves_claim_ids[]
  expected_observations[]
  predicted_gain?
  predicted_cost?
  confidence?
  embedding?
  claimed_nearest_idea_id?
  claimed_nearest_experiment_node_id?
  nearest_experiment_node_ids[]
  exact_duplicate_of?
  similarity_flags[]
  generation_artifacts[]
  status = GENERATED | INVALID | DEFERRED | REJECTED | SELECTED |
           IMPLEMENTING | EVALUATED | FAILED_TECHNICAL | ABANDONED
  selection_reason?
  deferral_reason?
  rejection_reason?
  experiment_node_id?
  outcome?
```

### Idea outcome

```text
IdeaOutcome
  evaluation_status = NOT_RUN | VALID | INVALID | INCONCLUSIVE
  implementation_status
  normalized_delta?
  validation_tier?
  actual_cost?
  actual_duration?
  gap_effects[]
  supported_claim_ids[]
  contradicted_claim_ids[]
```

`gap_effects` is the exact set of affected typed gap IDs, not free-form prose.
Claims and gaps become causal evidence only through the strict
`external_evaluation_metadata.ideation_evidence` object. By default, a
read-only coding-agent author inspects the complete code diff plus evaluation
output and feedback after a successful, integrity-valid Generic run. Every
finding must cite the exact diff and evaluation/feedback references. Claims and
targeted gap updates require a current registered evaluation attempt; without
one the author may return only open gaps. Empty lists are valid, and score
direction alone never creates causal evidence.

The author call is replayable by deterministic operation ID and durable result
artifact, with node-level phase telemetry. An external iteration evaluator may
explicitly replace `ideation_evidence`, while ordinary external metadata merges
without deleting it and may not replace the author-provenance metadata. The
adapter creates deterministic IDs tied to the current idea and experiment; the
archive commits the outcome, new claims, new open gaps, targeted gap
transitions, idea state, and batch state in one revision. Malformed present
metadata fails loudly.

`EXPLOIT` still requires comparable registered experiment evidence as well as a
supported lever. An unregistered campaign therefore remains conservative even
if it records unresolved open gaps.

Outcome replay compares evidence by monotonic descent rather than byte equality:
sources and affected experiment provenance may accumulate, timestamps and gap
debt may advance, and gaps may follow legal forward transitions. Immutable
claim/gap identity, earlier provenance, and evidence classification cannot be
removed or reversed.

If the selector displaces a directive's reserved gap by choosing an idea that
does not target it, selection atomically increments that open gap's deferral
count. Selecting the gap-targeting idea does not. This makes `GAP_DEBT` a
persisted consequence of actual selection decisions rather than inferred
model narrative.

The archive is campaign-local, strictly validated, and atomically persisted.
An incompatible shape fails loudly; development-time format changes replace
the prior format instead of accumulating migrations.

## Candidate generation and analysis

### Independent generation

Each generation member receives:

- the same campaign evidence snapshot;
- the same capacity constraints;
- one distinct operator brief;
- relevant prior ideas and executed experiments, clearly labeled; and
- a structured response schema.

Members generate independently before seeing other members' candidates. This
prevents early anchoring and makes operator diversity observable.

The considered pool is the union of newly generated candidates and any prior
ideas deliberately resurfaced by the evidence policy. Eligible candidates not
selected in the current batch become `DEFERRED`, not automatically `REJECTED`.
`REJECTED` is reserved for an explicit conclusion that the proposal should not
be reconsidered without materially new evidence.

### Candidate requirements

Every valid candidate states:

1. the intervention and intended parent;
2. why it is appropriate for the directive mode;
3. which evidence supports it;
4. which claims remain assumptions;
5. its expected observable effect;
6. how it will be evaluated;
7. approximate implementation/evaluation cost; and
8. which existing idea or experiment is most similar, if any.

Candidate cost is an advisory resource class or estimate, not admission
authority. The analyzer asks whether the currently granted evaluation fits;
that may be a probe during bootstrap. Comparable registered evidence is
required later before credible improvement or `EXPLOIT` can be asserted.

### Analysis pipeline

Analysis is deterministic where possible:

1. validate the candidate schema;
2. reject impossible parent or artifact references;
3. mark exact duplicates without a derived changed condition ineligible while
   retaining their records;
4. calculate embedding neighbors across the idea archive;
5. compare structured descriptors;
6. verify cited evidence and flag unsupported causal claims;
7. check feasibility against capacity and validation requirements;
8. summarize operator and descriptor coverage; and
9. persist analysis before selection.

Semantic similarity is an alarm, not a score and not an automatic rejection.
An idea may legitimately revisit a prior mechanism with new evidence, a
different parent, or a decisive test. Such an override must be explicit.

If fewer than two valid, meaningfully distinct candidates survive and the
directive is not `RECOVER` or `VERIFY`, the analyzer may request one bounded
repair generation targeted at the missing operator or descriptor. There is no
unbounded regeneration loop.

## Selection contract

The selector combines a Codex or Claude Code critic call with deterministic
eligibility rules.

### Hard rules

A candidate is ineligible when:

- it cannot complete within the capacity contract;
- its currently granted evaluation cannot complete;
- its parent or required artifact does not exist;
- it violates a delivery or finalization constraint;
- it depends on a contradicted claim without proposing a new test;
- it is an exact duplicate without an explicit changed condition; or
- its output schema or provenance is invalid.

The coding agent cannot override these rules.

### Selection output

```text
SelectionDecision
  selected_idea_id
  fallback_idea_ids[]
  diagnosis_audit[]
  hard_rule_results[]
  expected_benefit
  expected_cost
  gap_decisions[]
  duplicate_overrides[]
  decision_summary
```

For every material causal statement, `diagnosis_audit` quotes or references the
claim and labels it `SUPPORTED`, `CONTRADICTED`, or `INSUFFICIENT` against the
campaign evidence snapshot.

The selector optimizes credible expected utility under capacity. Novelty may
explain why a candidate broadens coverage, but it is never added to the score.

## End-to-end system flow

```mermaid
sequenceDiagram
    participant O as Orchestrator
    participant F as Fidelity and Budget
    participant G as GenericSearch
    participant I as IdeationEngine
    participant A as IdeaArchive
    participant X as Experiment Lifecycle
    participant W as Evidence Author
    participant H as Experiment History

    O->>F: Request next admissible action
    F-->>O: CapacitySnapshot and fidelity contract
    O->>G: Run admitted search iteration
    G->>I: Ideate with projected node history and capacity
    I->>A: Read prior ideas, batches, and gaps
    I->>I: Build evidence and SearchDirective
    I->>A: Persist batch as PLANNED
    I->>I: Generate independent operator candidates
    I->>A: Persist considered pool; mark batch GENERATED
    I->>I: Analyze validity, similarity, evidence, and feasibility
    I->>A: Persist analysis as ANALYZED
    I->>I: Select candidate and fallbacks
    I->>A: Persist decision as SELECTED
    I-->>G: Selected IdeaRecord and ParentPlan
    G->>X: Idempotently create SearchNode and branch
    X->>A: Link experiment node; mark BRIDGED and IMPLEMENTING
    X->>X: Implement, evaluate, and debug if technical
    X->>W: Review complete diff and evaluation record
    W-->>X: Attributable claims and gaps
    X-->>O: Return linked SearchNode
    O->>O: Attach external evaluation and integrity results
    O->>H: Persist executed experiment result
    O->>A: Persist IdeaOutcome
    A-->>I: Updated evidence for next iteration
```

### Detailed iteration

1. **Admit capacity.** Fidelity/budget policy decides whether another complete
   action fits and supplies the required evaluation profile.
2. **Freeze evidence.** Build one immutable campaign snapshot. Every candidate
   and selection decision in the batch refers to this snapshot.
3. **Choose mode.** Apply the state-machine precedence and emit a directive.
4. **Persist intent.** Create the batch in `PLANNED` state before invoking a
   generator.
5. **Assemble candidates.** Resurface eligible prior ideas, assign distinct
   operator briefs, and collect new structured proposals independently.
6. **Persist the initial population.** Save every structurally valid returned
   candidate and mark the batch `GENERATED`.
7. **Analyze for repair.** Compute duplicate facts, descriptor coverage,
   evidence audits, feasibility, and gap relevance without yet closing the
   generated lifecycle state.
8. **Repair once if necessary.** When authorized, append exactly one repair
   candidate while the batch is still `GENERATED`, then recompute the complete
   pool analysis. A persisted repair is never generated again on resume.
9. **Persist final analysis.** Record every candidate analysis and mark the
   complete batch `ANALYZED`; semantic invalidity remains auditable.
10. **Select.** Apply hard rules, then evidence-grounded comparative judgment.
11. **Persist the decision.** Save the selected idea and ordered fallbacks
    before creating a branch or changing code.
12. **Bridge idempotently.** Create exactly one `SearchNode` from the selected
    idea and record both identifiers.
13. **Execute.** Existing implementation, debugging, fidelity, and evaluation
    machinery owns the experiment.
14. **Classify the result.** Distinguish technical failure, invalid evidence,
    inconclusive evidence, and valid hypothesis outcome.
15. **Update evidence.** Attach the outcome to the idea, update claims and gaps,
    and make the new evidence visible to the next iteration.

## Persistence and resume semantics

The archive is written at explicit transaction boundaries. Resume never
regenerates work that has already crossed a boundary.

| Last durable state | Resume behavior |
|---|---|
| No batch | Build a new evidence snapshot and batch |
| `PLANNED` | Resume candidate generation with the same directive when context still matches |
| `GENERATED` | Reuse candidates and run analysis; do not regenerate |
| `ANALYZED` | Reuse analysis and run selection |
| `SELECTED` | Idempotently create or find the linked experiment node |
| `BRIDGED` / `IMPLEMENTING` | Delegate to the existing experiment resume/recovery lifecycle |
| Experiment completed, outcome missing | Reconstruct the outcome from the linked immutable experiment record |
| `COMPLETED` | No ideation work remains for the batch |

`context_hash` covers the problem, evidence snapshot, capacity view, directive,
resolved parent snapshots, and planning archive revision. Resume requires exact
equality and fails loudly on a mismatch; it does not replace the batch.
Completed coding-agent operations independently pin their prompt, response
schema, CLI, model, effort, tools, and timeout in durable artifacts, while the
outer run fingerprint pins the canonical configuration.

Identity operations are idempotent:

- one `batch_id` identifies one frozen ideation context;
- one `idea_id` identifies one proposal version;
- one selected idea links to at most one experiment node;
- retrying the bridge returns the existing node; and
- outcome recording is append-safe and does not duplicate evaluation effects.

## Module responsibilities

These are responsibility boundaries, not a requirement that every row become a
separate file or class.

| Component | Owns | Must not own |
|---|---|---|
| `IdeationEngine` | Batch orchestration and transaction ordering | Budget admission, experiment execution, score truth |
| `CampaignEvidenceBuilder` | Immutable normalized evidence snapshots | Mutating source records or inventing measurements |
| `EvidenceLedger` | Claims, gaps, provenance, evidence-state transitions | Candidate ranking or branch selection |
| `ExperimentCapacityProvider` | Remaining capacity, fidelity contract, reserve, completion feasibility | Idea generation or semantic ranking |
| `IdeaArchive` | Idea batches, candidates, embeddings, decisions, outcomes, strict persistence | Executed score authority or Git lifecycle |
| `IdeationPolicy` | Pure mode and terminal decision from evidence and capacity | Coding-agent calls or persistence side effects |
| `OperatorPlanner` | Distinct operator briefs, descriptor coverage, gap reservation | Final candidate selection |
| `CandidateGenerator` | Independent structured proposals | Comparing candidates or declaring evidence true |
| `CandidateAnalyzer` | Validation, similarity facts, feasibility, evidence references, bounded repair requests | Utility judgment beyond hard eligibility |
| `CandidateSelector` | Comparative judgment, diagnosis audit, selected idea, fallbacks | Overriding hard rules or rewriting evidence |
| `GenericSearch` bridge | Parent realization, SearchNode creation, branch linkage, recovery | Candidate generation or selector judgment |
| `build_idea_outcome` + archive | Failure classification and realized delta | Retrospectively editing the original proposal |

## History and retrieval surfaces

Executed and proposed work remain separate read models:

```text
search_experiments(query, filters)
  -> implemented solutions, evaluation evidence, branches, outcomes

mandatory ideation packet
  -> generated proposals, lineage, statuses, claims, gaps, and neighbors
```

The trusted process may use the embedding provider when building persisted
similarity facts. Ideation agents receive those facts in the packet. The
semantic separation and credential boundary are not optional.

## Behavior under representative campaigns

### Cold start

`BOOTSTRAP` allocates from the independent, mechanism-shift, and gap-targeted
operator palette up to the configured candidate quota. All generated candidates
survive in the archive even though only one is executed.

### Strong improvement

A gain above the noise threshold with a supported causal hypothesis triggers
`EXPLOIT`. The quota-limited palette prefers atomic refinement, supported
crossover, ablation, and gap targeting without guaranteeing every operator in
one batch.

### Candidate-family collapse

If candidates differ only lexically, descriptor and similarity analysis marks
the batch collapsed. One repair round targets a missing mechanism. If collapse
persists, the system may choose the strongest eligible candidate but records
the diversity failure as campaign evidence.

### Contradictory feedback

When narrative feedback conflicts with the score trace or slice results, the
claim is marked `CONTRADICTED`. Candidates depending on that diagnosis become
ineligible unless their purpose is to gather evidence that can resolve the
contradiction.

### Known language or subgroup gap

The gap receives a reserved proposal slot based on impact and uncertainty. A
promising exploit may still win selection, but the deferral is recorded. Debt
increases until the gap is tested or explicitly deprioritized with evidence.
This prevents indefinite neglect without blindly displacing a winning idea.

### Proxy overfitting

When proxy and delivery-grade results diverge, or an improvement is implausibly
large, the policy enters `VERIFY`. It replicates or widens evaluation before
spending capacity on another refinement.

### Technical failure

A recoverable coding or environment failure enters `RECOVER` with the same
idea and branch. The idea's hypothesis remains unjudged. A completed
implementation that produces no gain is instead valid negative evidence.

### End of campaign

Another action is admitted only when the capacity authority says a complete
currently granted evaluation fits while preserving finalization reserve.
Otherwise the campaign finalizes around its delivery incumbent. Kapso does not
yet have calibrated completion-probability or expected-value authorities, so
there is no speculative terminal-opportunity bypass.

## Failure handling

| Failure | Classification | Response |
|---|---|---|
| Generator timeout or malformed response | Ideation technical failure | Propagate the failure; retain the `PLANNED` batch and deterministic operation identity for an explicit resume |
| One generation member fails | Generation transaction failure | Persist no partial pool; resume the same batch after the external failure is resolved |
| All candidates invalid | Batch failure | Use the single repair allowance, then surface a typed failure |
| Selector fails | Selection technical failure | Retry selection over the persisted analyzed pool; never regenerate implicitly |
| Branch creation fails | Bridge technical failure | Retry idempotently from `SELECTED` state |
| Implementation error | Experiment technical failure | Recover same idea when feasible |
| Invalid or incomparable evaluation | Inconclusive outcome | Do not update hypothesis value or close gaps |
| Valid experiment loses | Negative hypothesis evidence | Record outcome and reconsider mode |
| Archive write fails | Control-plane failure | Do not advance lifecycle state or start implementation |

## Observability

Every batch should make the following reconstructable without reading model
transcripts:

- why the policy chose its mode;
- which operators were requested and produced;
- which evidence snapshot every claim referenced;
- exact and semantic duplicate facts;
- why each candidate was invalid, rejected, deferred, or selected;
- why a high-priority gap was deferred;
- which parent and branch implemented the idea;
- whether failure was technical or hypothesis-level; and
- predicted versus realized gain, cost, duration, and validation tier.

Recommended campaign metrics:

- best delivery-validated utility per wall-clock time and cost;
- candidate exact-duplicate and descriptor-collapse rate;
- unique useful descriptor coverage;
- high-impact gap debt and closure rate;
- predicted-versus-realized gain and cost calibration;
- selector diagnosis contradiction rate;
- proxy-to-delivery generalization gap;
- technical versus hypothesis failure rate; and
- crash/resume consistency.

No top-line "novelty score" is required.

## Acceptance criteria for the design

The eventual implementation conforms to this design only if all of the
following hold:

1. Every generated candidate is durably queryable whether or not it was
   selected.
2. Ideas and executed experiments have separate records and similarity
   searches.
3. A selected idea is persisted before branch or node creation.
4. Resume does not silently regenerate or reselect a completed batch stage.
5. Search comparisons respect maximize and minimize objectives.
6. Capacity and finalization decisions come from fidelity/budget authority,
   not an ideation-side runtime estimate.
7. `RECOVER`, `VERIFY`, `EXPLOIT`, and `EXPLORE` are distinguishable from
   persisted evidence and outcomes.
8. Only evaluation evidence can close an evaluation gap.
9. Duplicate similarity is observable but is neither a novelty reward nor an
   unconditional rejection rule.
10. The selector cannot choose a candidate that fails capacity, lineage,
    evidence, or validation hard rules.
11. Technical failure does not automatically count as negative hypothesis
    evidence.
12. No terminal opportunity probe can bypass protected finalization capacity.
13. Every generative or judgment-bearing model call runs through Codex CLI or
    Claude Code and is attributable to a coding-agent invocation.
14. Direct OpenAI API access is limited to embeddings; its key is never stored,
    logged, or passed into a coding-agent/MCP subprocess.

## Explicit non-goals

- MCTS, UCB, novelty search, or reinforcement learning inside a campaign.
- A global scalar novelty objective.
- Coding-agent-controlled budget or fidelity decisions.
- Unbounded candidate regeneration.
- Running multiple expensive experiments concurrently by default.
- Treating prompt prose as verified evaluation coverage.
- Replacing existing Git experiment isolation or evaluation machinery.
- Direct chat, completions, or responses API calls from ideation modules.
- Learning a cross-campaign policy before enough calibrated outcome data
  exists.

## Deferred extensions

These extensions become reasonable only after the core lifecycle produces
reliable data:

- cross-campaign calibration of operator success by task and campaign state;
- adaptive candidate quotas based on measured generation value;
- multi-parent or population scheduling for campaigns with much larger budgets;
- cheap probe racing when the task exposes a demonstrably predictive proxy;
- learned gap-priority or candidate-value models; and
- configurable embedding thresholds calibrated per model and domain.

## Research basis

The design borrows selectively from prior systems while adapting them to
Kapso's short, expensive campaign horizon:

- [AIRA](https://arxiv.org/abs/2507.02554): proposal operators and their
  interaction can matter more than search-policy sophistication at small
  budgets.
- [MLE-STAR](https://arxiv.org/abs/2506.15692): ablation-guided, targeted
  refinement is preferable to undirected iteration.
- [PlanSearch](https://arxiv.org/abs/2409.03733): diverse plans before code can
  broaden the solution space.
- [AlphaEvolve](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/AlphaEvolve.pdf)
  and [FunSearch](https://www.nature.com/articles/s41586-023-06924-6): persistent
  populations and operational diversity can preserve useful lineages.
- [The Ideation–Execution Gap](https://arxiv.org/abs/2506.20803): pre-execution
  assessments of LLM-generated novelty and value can change substantially
  after real execution, motivating outcome-linked idea records rather than
  trusting self-reported novelty.
- [Can LLMs Generate Novel Research Ideas?](https://arxiv.org/abs/2409.04109):
  idea generation and idea evaluation are distinct problems, and diversity
  must be managed explicitly.

The resulting architecture intentionally borrows archives, operators,
ablation, and evidence targeting without importing a heavyweight population
search policy that Kapso's experiment horizon cannot support.

---
title: "Research Planner: Related Work and Refinements"
description: "Definitive related-work analysis of the idea-generation proposal (Executive/Planner/Generator/Supervisor) against 12 competitor systems, with grounding against Kapso main and a revised transition plan"
date: 2026-07-13
---

Sources: the `idea-generation.mdx` proposal; the grounding report against Kapso main @ `31db3ff4`; and the 12-system competitor pass of 2026-07-13 (AIDE, AIRA-dojo, R&D-Agent, MLEvolve, EvoMaster/ML-Master 2.0, OpenEvolve, ShinkaEvolve, GEPA, EvoAgentX, AI Scientist v2, MLE-Agent, PAT) archived in `mechanisms_ranked.md`, `overview.md`, and `roadmap.md` in this directory.

## Verdict

The proposal's four-way split is **three-quarters supported and one-quarter ahead of the evidence**. The **Executive** (deterministic control plane owning budget, admission, and artifact validity) and the **Supervisor** (Kapso-owned, typed, interruptible execution) are the two best-supported components: every strong system in the survey keeps budget and stage control in deterministic code rather than in the LLM (AI Scientist v2 `agent_manager.py` stage gates, AIRA-dojo's solver-computed complexity levels and remaining-time operator inputs, OpenEvolve `evaluator.py` cascades), and the systems with the most reliable results are exactly those that own evaluation execution (R&D-Agent `LoopBase`, OpenEvolve's per-stage `asyncio` timeouts) — a capability Kapso's own reliability roadmap already defers as "execute provided evaluation through a Kapso-owned evaluator process" (`docs/evolve/reliability-roadmap.mdx:293-294`). The **Hypothesis Generator** as a read-only ideation step with a *typed* output contract is universal practice (AIDE `node.plan`, R&D-Agent hypothesis records, EvoAgentX), and the current `_generate_solution()` is already close to it. The evidence argues **against** the proposal's most ambitious component: a full agentic **Research Planner** maintaining a hypothesis portfolio with expected-gain/uncertainty/EVOI reasoning. None of the 12 systems implements one; the empirically successful "planners" are small hardcoded policies over typed operators (AIDE's `debug_prob=0.5`, `max_debug_depth=3`), and AIRA-dojo's own paper found well-operatored greedy search competitive with MCTS and evolutionary planners at roughly Kapso's ~10-iteration budget scale (`mechanisms_ranked.md` rank 31). Finally, two of the proposal's Executive duties — the finalization reserve and the continuously-protected best-artifact invariant — plus Kapso's already-landed evaluation-integrity hashing appear in **no** surveyed system and are the genuinely novel parts worth keeping.

## What the proposal gets right vs the current code

The grounding report confirms the proposal's core complaints against main @ `31db3ff4`:

- **Budget blindness.** `budget_progress` reaches `GenericSearch.run()` and is only printed (`strategy.py:184`); neither `_generate_solution(problem, parent.branch_name)` (`strategy.py:197-200`) nor `_build_ideation_prompt` (`strategy.py:368-382`) takes any budget input.
- **One-proposal lifecycle with no admission decision.** Every successful ideation call is implemented immediately; there is no candidate comparison, portfolio, or research-only action. Still true, including the 300 s ideation timeout (`strategy.py:110`).
- **Prose output contract.** The `<solution>` block with regex fallback is still the only ideation output format.
- **Fresh-session continuity.** A new `ClaudeCodeCodingAgent` per iteration (`strategy.py:343-347`); checkpoints add cross-*restart* continuity, not within-run planner memory.
- **Weak failure fallback.** `_fallback_solution()` returns a generic baseline and proceeds to implementation regardless (`strategy.py:349-353, 418-435`).
- **Incomplete research trace.** `_extract_sections_consulted()` still regex-scrapes the final response (`strategy.py:401-416`).
- **Knowledge backend not in the ideation path.** `solve()` never queries `self.knowledge_search` (`orchestrator.py:660-856`).

## What is already stale

Milestones 3–9 of `reliability-roadmap.mdx` landed after the proposal was written; five modules now partially implement it:

| Just-landed module | What it already covers |
|---|---|
| `run_checkpoint.py` | The durable core of the Executive's run state: schema-versioned, goal-hashed, config-fingerprinted `RunCheckpoint` with `running\|completed` status (`run_checkpoint.py:51-95`), strict `validate_resume` (`:196-220`), atomic fsync'd save (`:253-279`), typed error hierarchy (`:15-32`). It is a resume record, not yet a live control plane: no `remaining_seconds`, `feasible_actions`, `active_jobs`, or `runtime_estimates`. |
| `iteration_evaluator.py` | The Supervisor's "typed, trusted results" facet: frozen `IterationEvaluationContext` over an isolated detached worktree (`iteration_evaluator.py:23-36`; invoked at `orchestrator.py:602-658`), `IterationEvaluationResult` with finite-metric validation and JSON-safe metadata (`:39-45, 68-151`), explicit `record\|raise` failure policy (`:54-65`). Observational and post-hoc only — it does not launch, monitor, or interrupt anything. |
| `evaluation_integrity.py` | The Supervisor's trust boundary for provided evaluation: manifest hashing (`evaluation_integrity.py:61-104`), tamper detection including new executable files (`:107-160`), provenance constants (`:13-15`). Verifies the suite's *files*; Kapso-owned *execution* remains deferred (roadmap `:293-294`). |
| Parent-selection policies | `PARENT_POLICIES = {"best", "baseline"}`, `normalize_parent_policy()`, and the typed `ParentSelection` dataclass (`strategy.py:35-53`), chosen once per iteration by `_select_parent()` (`strategy.py:592-603`) and reused for ideation, branching, diffing, and lineage; persisted and cross-validated in checkpoints (`strategy.py:794, 846-883`). A degenerate but *real* "which branch" planner. |
| `reliability-roadmap.mdx` | Declares all 9 milestones implemented (`:14-24`) and constrains this proposal: holdout metrics stay observational (`:40-41, 463`), no resume-to-fresh fallback (`:44`); defers `selection_metric` (`:262-264`), a `ParentSelector` interface (`:418`), and the Kapso-owned evaluator (`:293-294`). |

Specific proposal claims now wrong:

1. **Greedy-only parent selection** — parent choice is a validated, configurable, checkpoint-persisted policy (`strategy.py:35-53, 592-603`), though the default remains greedy and nothing learned exists.
2. **Branch-view mismatch** — fixed: ideation and its MCP `repo_root` now run from a detached worktree of the selected parent ref (`strategy.py:294-307, 344`; `experiment_workspace.py:187`), covered by `tests/test_parent_selection.py:136`.
3. **Silent prompt/gate mismatch** — the mechanism changed: gates are now capability-resolved with per-gate `GateDiagnostic`s (`presets.py:53-82`) under a `skip|warn|error` policy in `resolve_gates` (`presets.py:240-306`; `strategy.py:120`); only the *prompt text* (`prompts/ideation_claude_code.md:43,47,78`) is stale.
4. **FeedbackGenerator as sole validity judge** — validity is now gated first by mechanical evaluation integrity (`orchestrator.py:153-160`; `strategy.py:249-257`; `base.py:553-584`), and invalid nodes cannot become best (`strategy.py:619-629`).
5. **Thin orchestrator** — it now owns per-iteration checkpointing (`orchestrator.py:532-548, 802-807`), resume validation (`:190-207`), restored-ref validation (`:550-568`), integrity enforcement (`:746-758`), external candidate evaluation (`:602-658`), and post-iteration budget checks (`:790-818`).
6. **Opus 4.6 via Bedrock** — default is `us.anthropic.claude-opus-4-5-20251101-v1:0` with configurable `auth_mode` (`strategy.py:97-108`), routed through `ModelRouter`/`RetryPolicy` (`orchestrator.py:114-123`; `core/llm.py:35, 78-79`).

## Related-work matrix

D1 budget-aware · D2 action portfolio · D3 fidelity ladder · D4 admission · D5 typed contracts · D6 supervision · D7 belief state · D8 endgame.

| System | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 |
|---|---|---|---|---|---|---|---|---|
| AIDE | | ✓ draft/debug/improve policy | | | ✓ typed review, WorstMetricValue | | ✓ experiment journal | |
| AIRA-dojo | ✓ operator-prompt budget interpolation | ✓ composable typed operators | | | ✓ per-operator configs | ✓ scalable distributed runners | ✓ five memory scopes | |
| R&D-Agent | | ✓ research/dev loop stages | ✓ partial evaluation | ✓ failed-task skip-sets | ✓ per-step typed records | ✓ step checkpoint, error routing | ✓ CoSTEER knowledge base | ✓ endgame candidate merge |
| MLEvolve | ✓ time-aware parent selection | ✓ fusion + mutation operators | ✓ anomaly-triggered deep verify | | | | ✓ RRF experience retrieval | ✓ stagnation/endgame fusion |
| EvoMaster / ML-Master 2.0 | | | | ✓ delta-audited promotion | ✓ evolution_state.json | | ✓ hierarchical memory, wisdom store | |
| OpenEvolve | | | ✓ cascade thresholds | ✓ stage-threshold admission | ✓ multi-metric EvaluationResult, JSONL trace | ✓ per-stage async timeouts | ✓ MAP-Elites islands | |
| ShinkaEvolve | ✓ cost-aware UCB routing | ✓ edit-mode operators | | ✓ embedding+LLM novelty gate | | ✓ async no-barrier queues | ✓ MetaSummarizer insights | |
| GEPA | | ✓ component round-robin selector | ✓ minibatch→full eval split | ✓ strict-improvement acceptance | ✓ reflective per-case records | | ✓ per-instance Pareto pool | |
| EvoAgentX | | ✓ structural workflow operators | | ✓ novelty check | ✓ OptimizableField registry | | | ✓ convergence early-stop |
| AI Scientist v2 | ✓ stage/attempt-adaptive prompting | ✓ draft/debug/ablate operators | ✓ staged experiment pipeline | ✓ stage-gate promotion | ✓ typed review, is_buggy_plots | | ✓ journal + VLM evidence | ✓ multi-seed + LLM final pick |
| MLE-Agent | | | | ✓ human plan review | ✓ advisor JSON profile | | | |
| PAT (Google) | ✓ complexity-adaptive verify budgets | | ✓ escalating verification cascade | ✓ duplicate-claim filtering | ✓ severity-graded findings | | | |

### D1 Budget-aware

Best: **AIRA-dojo**, via two distinct mechanisms. The budget signal lives in the operator modules: `core/solvers/operators/draft.py`, `improve.py`, and `crossover.py` template `time_remaining`/`steps_remaining` directly into operator prompts (e.g. `improve.py:43-50`), the cheapest budget signal in the survey (rank 20). Separately, `get_complextiy_level` (`src/dojo/solvers/utils.py:20`, called from `mcts.py`/`greedy.py`/`evo.py`) derives a categorical LOW/MEDIUM/HIGH hint purely from a node's child count — it takes no budget input. PAT scales verification effort to claim complexity (PAT Sec 2: recall 55.2%→89.7% from orchestration alone); ShinkaEvolve's `prioritization.py` AsymmetricUCB (`cost_aware_coef=0.5`) is cost-aware routing, though at ~10 sequential pulls the static cheapness prior does most of the work (rank 23). **Steal:** AIRA's operator-prompt interpolation — `budget_progress` already reaches `strategy.py:184`; threading it into `_build_ideation_prompt` is a day-one change.

### D2 Action portfolio

Best: **AIDE** — the draft/debug/improve policy (`search_policy` with `debug_prob=0.5`, `max_debug_depth=3`) is the minimal proven portfolio; AIRA-dojo generalizes it into typed operators reused across greedy/MCTS/evolutionary policies (39.6%→47.7% on MLE-bench lite, rank 10). Nobody implements the proposal's eight-action menu; nobody has a first-class `research`-only action. **Steal:** the portfolio as an enum on the ideation contract, not an agentic chooser — but only draft/improve in generic mode: rank 3's own audit finds generic agents self-debug in-session with no reliable buggy predicate ("the AIDE regime transfers strongly only to benchmark mode"), and rank 10 finds the debug operator "unreachable" under greedy parenting.

### D3 Fidelity ladder

Best: **OpenEvolve** — `evaluator.py cascade_thresholds` with per-stage asyncio timeouts: candidates pass cheap stages before expensive ones. GEPA's minibatch→full split is the same shape for prompt optimization. The audited caveat (rank 18): cascades are unenforceable while the *agent* runs evaluation in-session under one flat timeout — the ladder requires a framework-owned runner first. **Steal:** cascade thresholds as the semantics of `requested_fidelity`, gated behind the Kapso-owned evaluator.

### D4 Admission

Best: **GEPA** — `StrictImprovementAcceptance` (defined in `strategies/acceptance.py:39`, wired as the default acceptance criterion at `core/engine.py:124`): a candidate must beat the parent on a cheap minibatch before full evaluation (35x rollout-efficiency claim). ShinkaEvolve's `novelty_judge.py` (embedding similarity threshold 0.99 as shipped — `core/config.py:44 code_embed_sim_threshold`; the `NoveltyJudge` class default is 1.0 — plus LLM equivalence judge) rejects duplicates pre-spend; AI Scientist's stage gates block buggy nodes from advancing. The audit found the novelty gate largely redundant with Kapso's mandated history queries (rank 12), and GEPA's split unenforceable while the agent has Write access to "held-out" data (rank 29). **Steal:** admission as an *Executive* check (duplicate-against-history plus budget feasibility) rather than an embedding service.

### D5 Typed contracts

Best: **R&D-Agent** — `LoopBase` per-step typed, pickled records with skip/withdraw error routing; OpenEvolve's `evaluation_result.py` multi-metric result plus versioned JSONL trace is the evaluation-side equivalent; AIDE's `metric.py WorstMetricValue` ("always compares worse") solves invalid-result ordering. **Steal:** R&D-Agent step-record semantics, but landed inside `RunCheckpoint.strategy_state` — Kapso already has the durable substrate (field at `run_checkpoint.py:64`; store with atomic save at `:226-284`, `:253-279`).

### D6 Supervision

Best: **ShinkaEvolve** — `async_runner.py` no-generation-barrier proposal/evaluation queues; R&D-Agent adds step-granular checkpoint with typed error routing; OpenEvolve enforces per-stage timeouts because it owns execution. The audit is blunt (rank 32): generic Kapso has no cheap-proposal/expensive-eval asymmetry, so async queues help only wall-clock-bound runs. **Steal:** supervised *synchronous* execution first (own the process, enforce the timeout, capture telemetry); queues later.

### D7 Belief state

Best: **AIRA-dojo** — `operators/memory.py MEM_OPS` deterministic scoped digests (journal/top-k/diverse/ancestral/sibling) beat free-form "research notebooks"; ML-Master 2.0's hierarchical consolidation (Sec 3.3-3.4) and ShinkaEvolve's MetaSummarizer are the long-run variants. No system maintains an explicit hypothesis portfolio with uncertainty — belief state in practice means *deterministic history digests plus a periodic distilled summary*. **Steal:** AIRA memory scopes as additive digest builders over `node_history`.

### D8 Endgame

Best: **AI Scientist v2** — multi-seed re-evaluation of promoted nodes (`_run_multi_seed_evaluation`) plus LLM holistic final selection with an anti-overtrust prompt (`journal.py get_best_node`). EvoAgentX's `convergence_utils.py check_convergence` is a cheap plateau stop; MLEvolve and R&D-Agent trigger fusion in the endgame. Nobody reserves protected wall-clock for finalization. **Steal:** multi-seed validation of the final candidate; the reserve itself is Kapso's to invent.

## Refinements to the proposal

### (a) Revised NextAction / RunState sketch

The proposal's records should be **extensions of contracts that already exist on main**, not a parallel schema (grounding report §3). Revised sketch:

```text
RunState  (live view derived from RunCheckpoint — .kapso/run_state.json is already owned
           by RunCheckpointStore, run_checkpoint.py:226-284 (RELATIVE_PATH at :229), holding
           the RunCheckpoint dataclass (:52-223); do not add a second run-state file)
  goal, goal_hash, config_fingerprint          # run_checkpoint.py:35-48
  completed_iterations, cumulative_cost        # run_checkpoint.py:51-64
  remaining_seconds, finalization_reserve_s    # NEW: computed by Executive, never persisted stale
  feasible_actions: list[ActionResolution]     # shape of GateResolution.to_dict()
                                               # (requested/enabled/unavailable+reason, presets.py:100-108)
  best_valid_result: SearchNode summary        # get_best_experiment already excludes
                                               # errors AND evaluation_valid=False (strategy.py:617-629)
  completed_trials: list[SearchNode.to_dict()] # base.py:44-104, already JSON round-trippable,
                                               # incl. evaluation_provenance / integrity_error / metrics
  last_feedback: FeedbackResult                # orchestrator.py:28, :780-786
  runtime_estimates, active_jobs               # DEFERRED until the Supervisor exists

NextAction
  action: draft | improve                      # AIDE/AIRA portfolio, generic-mode v1 (Phase 3).
                                               # Values join only with a consumer: debug needs a
                                               # failed-parent policy + buggy predicate (rank 3/10
                                               # audits; benchmark mode first); finalize needs the
                                               # Phase-6 reserve; research/smoke_train/resume ditto
  hypothesis: ChangeHypothesis                 # AIDE node.plan atomicity: one change, one mechanism
  parent: ParentSelection                      # strategy.py:48-53, reused verbatim
  requested_fidelity: smoke | low | medium | full
      # semantics = OpenEvolve cascade stage, enforceable only by a Kapso-owned runner
  complexity_level: low | medium | high        # AIRA categorical, replaces numeric
                                               # expected_gain/uncertainty/EVOI — no surveyed
                                               # system uses numeric EVOI and none validates
                                               # LLM duration estimates
  validation_plan:
      integrity: EvaluationIntegrityReport verdict   # evaluation_integrity.py:13-58
      stages: list[CascadeStage(threshold, timeout)] # OpenEvolve cascade_thresholds
      final_seeds: int                               # AI Scientist multi-seed
  promotion_or_stop_rule: Rule
```

Executive-side execution reuses `RetryPolicy`/`ModelRouter` (`core/llm.py:35-152`) for retries and `materialize_ref`/`WorkspaceCheckoutError` (`experiment_workspace.py:25, 187`) for workspace safety. The Supervisor's result type should be `IterationEvaluationResult`'s validation rules (finite metrics, JSON-safe metadata, `record|raise` policy; `iteration_evaluator.py:39-151`) applied to a superset record; the deprecated `ExperimentResult` shim (`base.py:204-274`) is superseded, not extended — but not yet deletable: the shared orchestrator still imports and types against it (`orchestrator.py:32, 58`) and `_template.py` constructs it, so the shim stays until those consumers migrate.

### (b) Corrections where the evidence argues against the proposal

1. **Build much less Planner, much later.** AIRA-dojo's finding that well-operatored greedy is competitive with sophisticated planners at ~10-iteration scale (rank 31), plus the total absence of EVOI/portfolio planners across all 12 systems, means the Planner should *start* as: the existing `ParentSelection` policy + a 3-action operator enum + budget-aware prompting. The deferred `ParentSelector` interface (`reliability-roadmap.mdx:418`) is the designated extension point. An agentic director is a Phase-N experiment, not the architecture's load-bearing wall.
2. **The fidelity ladder and acceptance gates depend on the Supervisor, not the Planner.** Ranks 18 and 29 both audit to the same root cause: Kapso never executes evaluation itself, so per-stage thresholds/timeouts and holdout splits are unenforceable against an agent with Write access. The proposal's transition (typed records → budget-aware ideation → portfolio → action types → Supervisor) puts the Supervisor fifth; the evidence says the Kapso-owned evaluator process (already planned, roadmap `:293-294`) must precede any enforceable fidelity or admission semantics. Also binding: holdout metrics must remain observational (`reliability-roadmap.mdx:40-41, 463`).
3. **Drop the numeric expected_gain/uncertainty fields initially.** No system elicits calibrated numeric gain estimates from an LLM; the working substitutes are categorical (AIRA complexity levels, AI Scientist stages, PAT complexity-scaled budgets). Keep the fields out of the v1 contract rather than collecting noise.
4. **Temper the hypothesis portfolio.** Rank 8's audit: atomicity ("one change per node") has more evidence behind it than k-candidate ranking, and multi-draft generation is proven only at the *root* (AIDE/AIRA multi-draft roots, rank 3). Generate k candidates on iteration 1; thereafter one atomic typed hypothesis per iteration.
5. **The novelty gate is mostly redundant here.** Rank 12: mandated history MCP queries plus deterministically threaded feedback already cover most duplicate risk, and the embedding half is inert with Weaviate off by default. Admission v1 = cheap history-dedupe + budget feasibility inside the Executive (the history-dedupe half lands in Phase 3; the budget-feasibility half extends the existing post-iteration budget checks, `orchestrator.py:790-818`, in Phase 5).
6. **Fix the fallback via admission, not better fallback text.** No surveyed system auto-implements a generic baseline on ideation failure; GEPA-style acceptance says a candidate that cannot state an improvement hypothesis is rejected. `_fallback_solution()` (`strategy.py:349-353`) becomes an admission-denied path: retry ideation once, else skip the iteration.

### (c) What no system does — novel parts worth keeping

- **Finalization reserve.** Zero precedent in 12 systems. AI Scientist gates stages and EvoAgentX stops on convergence, but nobody protects wall-clock for packaging/validating the final artifact. Keep it; it is the Executive's most defensible novel duty.
- **Protected best-artifact invariant.** "A usable best result always exists" is partially landed and already ahead of the field: restored-ref validation (`orchestrator.py:550-568`), integrity gating of finalized candidates (`:746-758`), checkpoint-per-iteration (`:802-807`), lineage cross-validation (`strategy.py:861-883`). No competitor validates that its best artifact still *exists and resolves*.
- **Adversarial evaluation integrity.** Manifest hashing with tamper detection (`evaluation_integrity.py:61-160`) treats the eval suite as an attack surface of its own coding agent. GEPA's holdout is unenforceable by its own design assumptions; no other system hashes the evaluator. This is the trust kernel the Supervisor should be built around.
- **Feasible-actions-with-diagnostics.** `GateResolution.to_dict()`'s requested/enabled/unavailable+reason shape (`presets.py:100-108`) generalized to actions — a capability-resolved action menu with reasons — has no competitor analogue (closest is R&D-Agent's skip-sets, which are silent).

## Revised transition plan

Ordered by dependency and evidence strength; each phase names its precedent, landing spot (verified against main), and the landed code it builds on.

**Phase 1 — Budget into ideation (precedent: AIRA operator prompts templating `time_remaining`/`steps_remaining`, `core/solvers/operators/improve.py:43-50`, rank 20; strongest evidence-to-effort ratio).** Land in `_build_ideation_prompt` (`strategy.py:368-382`), threading `budget_progress` already available at `strategy.py:184`, plus remaining-seconds and a categorical complexity hint (the latter's precedent is AIRA's child-count-based `get_complextiy_level`, `src/dojo/solvers/utils.py:20`). Builds on: nothing new; pure plumbing.

**Phase 2 — Typed contracts without behavior change (precedent: R&D-Agent `LoopBase` step records; AIDE `node.plan`; roadmap Phase 0).** Add `NextAction`/`ChangeHypothesis` per the sketch above; parse the ideation output into it with prose fallback. Land the types in the shared `search_strategies/base.py` beside `SearchNode` (`base.py:44-104`) and persist through `RunCheckpoint.strategy_state` (field at `run_checkpoint.py:64`; atomically saved by `RunCheckpointStore.save`, `:253-279`). **Scope: GenericSearch is the only producer/consumer in v1; `benchmark_tree_search.py` is explicitly out of scope.** The evidence base repeatedly flags dual-strategy/orchestrator/store lockstep as the dominant integration cost for exactly this kind of contract change (ranks 9, 10, 12), so the shared types must be purely additive, with a BTS migration as a separate later step. Supersede the deprecated `ExperimentResult` shim (`base.py:204-274`) as GenericSearch's output type only, leaving the shim in place for its remaining consumers (`orchestrator.py:32, 58`; `_template.py`). Builds on: `SearchNode.to_dict/from_dict`, checkpoint atomicity.

**Phase 3 — Action portfolio v1 (draft/improve) + admission v1 (precedent: AIDE `debug_prob=0.5`/`max_debug_depth=3`, rank 3 — adopted only as far as its audit supports; GEPA acceptance for the rejection semantics).** A deterministic draft/improve policy beside `_select_parent()` (`strategy.py:592-603`), extending `PARENT_POLICIES`/`normalize_parent_policy` (`strategy.py:35-45`) toward the deferred `ParentSelector` interface (`reliability-roadmap.mdx:418`). **`debug` is explicitly out of scope for generic mode in this phase.** The rank-3 audit's own conclusion is that generic agents self-debug in-session and lack a reliable buggy predicate, so the AIDE debug regime "transfers strongly only to benchmark mode"; rank 10 likewise audits the debug operator as "unreachable" under greedy parenting. Both hold on main: `_select_parent()` over `PARENT_POLICIES = {"best", "baseline"}` (`strategy.py:35-53, 592-603`) can return only `main` or the best *valid* node — `get_best_experiment` filters out every node with `had_error` or `evaluation_valid=False` (`strategy.py:617-629`) — so a failed node can never be selected as parent and a `debug` action would have neither a trigger nor a predicate. `debug` joins the portfolio only when (a) a failed-parent policy is added under the `ParentSelector` interface and (b) a trustworthy buggy predicate exists; until then it belongs in benchmark mode, where the audit says it transfers. Admission v1 also lands here: cheap dedupe of the proposed hypothesis against `node_history` (the history half of correction b5), plus converting `_fallback_solution()` (`strategy.py:349-353`) into an admission-denied path (retry ideation once, else skip the iteration). Builds on: `ParentSelection`, checkpoint-validated lineage, `evaluation_valid`-aware `get_best_experiment` (`strategy.py:617-629`).

**Phase 4 — Supervisor slice: Kapso-owned evaluator process (precedent: OpenEvolve `evaluator.py` per-stage timeouts; R&D-Agent typed error routing; explicitly planned at roadmap `:293-294`).** Extend `IterationEvaluator` (`iteration_evaluator.py:23-151`) from post-hoc observation to owned execution in `materialize_ref` worktrees (`experiment_workspace.py:187`), gated by the integrity manifest (`orchestrator.py:153-160`; `evaluation_integrity.py:61-104`). Builds on: `IterationEvaluationContext/Result`, `record|raise` policy, `RetryPolicy` (`core/llm.py:78-79`).

**Phase 5 — Fidelity ladder + admission v2 (precedent: OpenEvolve `cascade_thresholds`, rank 18; GEPA minibatch gate, rank 29; ShinkaEvolve novelty only if dedupe proves insufficient).** `ValidationPlan` cascade stages executed by the Phase-4 runner; Executive pre-admission extends the existing post-iteration budget checks (`orchestrator.py:790-818`) to pre-iteration feasibility. Blocked on Phase 4 by the rank-18/29 enforceability audits. Caveat (open question 2): no surveyed system validates LLM duration estimates, so the *duration*-based half of pre-admission is additionally gated on empirical per-repo runtime models seeded from `SearchNode` history; until those exist, pre-iteration checks are limited to hard budget arithmetic (remaining wall-clock vs. the finalization reserve), not per-candidate duration estimates.

**Phase 6 — Endgame + belief state (precedent: AI Scientist `_run_multi_seed_evaluation` and `get_best_node`; EvoAgentX `check_convergence`; AIRA `MEM_OPS` scopes; ShinkaEvolve MetaSummarizer).** Finalization reserve and `finalize` action in the orchestrator; multi-seed validation of the final candidate via the Phase-4 runner (`_evaluate_candidates`, `orchestrator.py:602-658`, already runs isolated candidate evals); deterministic history digests in ideation prompt assembly. Only after this, evaluate whether an agentic Planner adds anything over the deterministic policy — with an A/B on the same goals.

Deferred indefinitely: async proposal/eval queues (rank 32 — no cost asymmetry in generic mode), QD archives/islands (rank 31 — budget-regime mismatch), cost-aware model routing (rank 23 — too few pulls to learn).

## Open questions

1. **Does budget-aware prompting change outcomes at ~10 iterations with a strong coding agent?** AIRA's evidence comes from MLE-bench with weaker per-step agents; Kapso's agent may already self-regulate scope. Needs an in-house ablation.
2. **Can runtime estimates support admission at all?** No surveyed system estimates duration before launch; nobody validates LLM time estimates. The Executive may need empirical per-repo runtime models seeded from `SearchNode` history before duration-based admission is safe.
3. **How large should the finalization reserve be, and is it static or adaptive?** No precedent exists; even a fixed 10% is a guess until failure-mode data accumulates.
4. **Does the typed contract tax ideation quality?** Rank 8 flags that enforced atomicity can reduce per-budget progress; the prose→schema conversion may also lose mechanism detail the implementation agent uses. Measure before hard-requiring fields.
5. **Can a Supervisor interrupt training the coding agent launched?** All precedents (OpenEvolve, R&D-Agent) supervise framework-owned processes; none supervises an agent that itself spawns training. Whether Kapso should force training through a supervised entry point, or wrap agent-spawned processes, is unresolved.
6. **Who arbitrates validity conflicts?** Mechanical integrity (`base.py:553-584`), `FeedbackGenerator` judgment, and future multi-seed statistics can disagree; the precedence order beyond "integrity vetoes" is undesigned.
7. **Is a hypothesis portfolio ever worth it?** No system ablates portfolio-vs-sequential; the honest position is that scoped memory (D7) plus atomic hypotheses may capture the value. Revisit only with >20-iteration budgets.

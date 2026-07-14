---
title: "Integration Plans"
description: "Detailed, code-grounded integration plans for the top-priority competitor mechanisms selected for adoption into Kapso."
---

This document contains integration plans for the top priority band from `mechanisms_ranked.md`: every mechanism with **priority >= 4 and difficulty <= 3** — the impactful, low-integration-effort set. The two impact-4/difficulty-4 control-plane mechanisms (ranks 9–10) are excluded on effort grounds. That yields **8 plans**, ordered by priority rank.

Date: 2026-07-13. All plans are proposals grounded in the code as of commit HEAD — no source code was modified in producing them.

## 1. Fail-closed feedback parsing with mechanical artifact validation gate

**Sources:** AIDE, MLE-Agent · **Impact:** 3 · **Difficulty:** 2 · **Priority:** 4 · **Evidence:** AIDE `aide/agent.py` review_func_spec (non-float metric → `is_buggy=True`); MLE-Agent `mle/workflow/kaggle.py` submission-existence gate; Kapso counterpart `feedback_generator.py:276-283` (parse failure silently returns stop=False, evaluation_valid=True, score=None).

### (a) Target files and code

- `src/kapso/execution/search_strategies/generic/feedback_generator/feedback_generator.py`
  - `FeedbackResult` (26-41): add defaulted fields `parse_ok: bool = True`, `parse_error: Optional[str] = None`; extend `to_dict()`.
  - `_parse_response` (202-249): in strict mode, require all four tags (the prompt already mandates this at `prompts/feedback_generator.md:77`); missing-tag fail-open defaults at 228-229 become a parse failure instead of silently assuming `stop=False`/`evaluation_valid=True`.
  - `_parse_response_json_fallback` (251-283): the total-failure default at 276-283 becomes fail-closed: `parse_ok=False`, `evaluation_valid=False`, `score=None`, `parse_error` set, raw snippet kept in `feedback`.
  - `FeedbackGenerator.__init__` (62-89): new `strict: bool = False` ctor arg consumed by `_parse_response`.
- `src/kapso/execution/search_strategies/generic/strategy.py`
  - `run()` step 3 (187-201): record whether `evaluation_output` genuinely came from agent tags or was backfilled with the raw transcript (the fallbacks at 193 and 200 make a plain "non-empty" check vacuous) — set a new `node.evaluation_output_from_agent` flag.
  - New `_eval_artifact_gate(node)`: mechanical pre-check before feedback. Because `finalize_session` (499) has already pushed the branch (`experiment_session.py:274`) and deleted the session folder (`experiment_session.py:282`), MLE-Agent-style `os.path.exists` is wrong; check the git object instead: `self.workspace.repo.git.cat_file('-e', f"{node.branch_name}:{node.evaluation_script_path}")`, plus non-empty `evaluation_script_path` and the `evaluation_output_from_agent` flag.
  - `_generate_feedback` (575-622): call the gate first; on gate failure with fail-closed enabled, skip the generator call (saves one agent invocation), set `evaluation_valid=False`, `score=None`, `had_error=True`, `error_message`. On `parse_ok=False` from the generator, same marking — this also stops line 611 from silently overwriting a genuine agent-extracted score (195-196) while the node still looks selectable. The exception handler at 617-620 additionally sets `evaluation_valid=False`.
  - `had_error=True` is what makes the fix behaviorally real today: selection filters only `had_error` and ranks `score or 0` (552-560, 540-550), and `_get_best_branch` (526-531) parents the next experiment from the best node — in minimize mode a parse-failure `None` score currently ranks as 0 and can poison the best branch.
- `src/kapso/execution/orchestrator.py`
  - `_create_feedback_generator` (140-178): read `strict` from the already-parsed `feedback_generator` mode-config section (152) and pass to the ctor.
- `src/kapso/execution/search_strategies/generic/feedback_generator/prompts/feedback_generator.md` (55-77): tighten wording in lockstep — all four tags mandatory, response discarded if any is missing.
- `src/kapso/execution/search_strategies/base.py` `SearchNode` (41-77): add `evaluation_output_from_agent: bool = True` (defaulted; `evaluation_valid` already exists at 61).

### (b) New types and config keys

No pydantic — it is not a declared dependency (`pyproject.toml:32-44`); plain dataclass + explicit checks suffice.

```python
# feedback_generator.py
@dataclass
class FeedbackResult:
    stop: bool
    evaluation_valid: bool
    feedback: str
    score: Optional[float] = None
    parse_ok: bool = True            # NEW: False => fail-closed marking
    parse_error: Optional[str] = None  # NEW: "missing_tags:score,stop" etc.
```

```yaml
# strategies.yaml, generic presets params (and benchmark config.yaml overrides)
feedback_fail_closed: false   # gate skips + had_error escalation when true

# mode config, feedback_generator section (already read at orchestrator.py:152)
feedback_generator:
  strict: false               # require all four XML tags
```

`GenericSearch.__init__` reads `self.params.get("feedback_fail_closed", False)`.

### (c) Backward compatibility

- `SearchNode.score` stays `Optional[float]` (base.py:59) — no type or semantics change, so no compat property is needed. On failures, fail-closed mode sets `score=None` plus `had_error=True`; every existing consumer already handles both: selection filters `had_error` (strategy.py:554), ranking uses `score or 0` (547, 559), `ExperimentResult.from_search_node` uses `node.score or 0.0` (base.py:123), and `ExperimentRecord` routes `had_error` nodes into the existing error-insight path (`store.py:194-204`).
- New `FeedbackResult` fields are defaulted, so the external constructors at `orchestrator.py:416-421` and the `SolutionResult.final_feedback` field (`solution.py:36`) compile unchanged.
- Existing YAML modes work untouched: both new keys default to false/absent; presets in `strategies.yaml` need no edits.
- `BenchmarkTreeSearch` never calls the FeedbackGenerator (`benchmark_tree_search.py:83`, 446 — handler supplies feedback at 657), and all changes live in `FeedbackGenerator` + `GenericSearch`, so it is structurally unaffected.
- Old `checkpoint.pkl` files (strategy.py:711-721) predate the new `SearchNode` field; read it via `getattr(node, "evaluation_output_from_agent", True)`.

### (d) Implementation steps

1. Extend `FeedbackResult` with defaulted `parse_ok`/`parse_error`; update `to_dict`. Verify: import + existing constructors unchanged.
2. Make `_parse_response_json_fallback`'s terminal default (276-283) fail-closed (`parse_ok=False`, `evaluation_valid=False`); add `strict` ctor flag enforcing all-four-tags in `_parse_response`. Verify with offline unit tests on canned responses (no LLM): full tags, 3-of-4 tags, garbage, JSON fallback.
3. Update `feedback_generator.md` output-contract wording in the same PR.
4. Add `evaluation_output_from_agent` tracking in `run()` (187-201) and implement `_eval_artifact_gate` with `git cat-file -e`. Verify against a workspace repo fixture with/without the file on the branch.
5. Wire `feedback_fail_closed` into `_generate_feedback`: gate-skip + `had_error` escalation + parse-failure marking; set `evaluation_valid=False` in the 617-620 exception path. Verify via unit test with a stubbed `feedback_generator`.
6. Pass `strict` through `orchestrator._create_feedback_generator`. Verify config plumbing with a minimal mode dict.
7. Optional follow-up (separate PR): persist `evaluation_valid` on `ExperimentRecord` (`store.py:152-161`).

### (e) Acceptance criteria

- Default-behavior regression: with both flags off, a stubbed garbage feedback response yields a node with `score=None`, `should_stop=False`, `had_error=False`, and `get_best_experiment()` returns exactly the same node as before the change (only delta: `evaluation_valid` is now False — inert, consumed only by the print at orchestrator.py:410 and `last_feedback_result`).
- Strict + fail-closed: garbage or 3-of-4-tag response → `parse_ok=False`, node `had_error=True`, excluded by `get_best_experiment()`; in a minimize-mode (`maximize_scoring=False`) simulation, the parse-failure node no longer outranks a real scored node.
- Gate: node whose `evaluation_script_path` is absent on `generic_exp_{n}` → feedback agent never invoked (assert stub call count 0), node marked invalid with mechanical reason.
- Existing suite passes; a BenchmarkTreeSearch smoke run is byte-identical in behavior.

### (f) Risks and mitigations

- Strict mode raises invalidation frequency if the model drops a tag → burned iterations. Mitigation: flags default off; roll out per-benchmark; `parse_error` telemetry quantifies real-world rate before flipping defaults.
- `had_error=True` triggers error-insight extraction (`store.py:194-204`) with a synthetic message. Mitigation: prefix `error_message` with `"[feedback-gate]"` so insights stay interpretable.
- `git cat-file` could fail for exotic paths/refs and wrongly invalidate. Mitigation: on gate *infrastructure* errors (subprocess failure), log and fall through to the generator rather than invalidating.
- Old pickled checkpoints missing the new node field. Mitigation: `getattr` defaults (see (c)).
- Full impact on selection still pairs with mechanism #2's invalid-score semantics; this plan ships a self-contained interim via `had_error`, which #2 can later replace without rework.

## 2. Probabilistic, time-aware parent selection replacing greedy hill-climb

**Sources:** MLEvolve, ShinkaEvolve, EvoAgentX · **Impact:** 3 · **Difficulty:** 2 · **Priority:** 4 · **Evidence:** MLEvolve `engine/node_selection.py` (soft-switch, Eq. 4-5); ShinkaEvolve `shinka/database/parents.py` (median/MAD sigmoid × offspring decay); EvoAgentX `aflow_utils/data_utils.py` (`_compute_probabilities`).

### (a) Target files and code

- **`src/kapso/execution/search_strategies/parent_selection.py` (new).** Home of `ParentCandidate`, the `ParentSelectionPolicy` ABC, four policies (`greedy`, `weighted`, `soft_switch`, `mixture`), and a `build_parent_policy(config: dict)` factory. Placed beside `base.py` so `benchmark_tree_search.py` can reuse it later, but only GenericSearch is wired up now.
- **`src/kapso/execution/search_strategies/generic/strategy.py`** — the only existing file that changes:
  - `GenericSearch.__init__` (strategy.py:63-113): add `self.parent_policy = build_parent_policy(self.params.get("parent_selection", {}))` following the existing `self.params.get(...)` pattern (strategy.py:68-90).
  - `run()` (strategy.py:127-211): replace the two independent greedy lookups — `parent_branch = self._get_best_branch()` (strategy.py:154) and `parent_node_id=self._get_best_node_id()` (strategy.py:163) — with one call to a new `_select_parent() -> Optional[SearchNode]`, so branch and node id always come from the *same* sampled node. Feed it `budget_progress`, which `run()` already receives (strategy.py:127) but today only prints (strategy.py:145); the orchestrator already computes and passes it (orchestrator.py:370-375, 392-395).
  - `_get_best_branch` / `_get_best_node_id` (strategy.py:526-538): become thin wrappers over `_select_parent()` (kept for any external callers) or are deleted.
  - New `_select_parent()`: builds `ParentCandidate`s from `self.node_history` (filter `had_error`, non-empty `branch_name`), derives `children_count = sum(1 for n in self.node_history if n.parent_node_id == c.node_id)`, passes `maximize=self.problem_handler.maximize_scoring` (mirroring the sign flip at strategy.py:552-560), and a per-call `random.Random(f"{seed}:{len(self.node_history)}")` for determinism across checkpoint resume (checkpoint only pickles `node_history`, strategy.py:711-721).
  - Feedback-lineage reconciliation in `run()`: the orchestrator threads the *last* node's feedback into the next problem (orchestrator.py:386-387, 430). When the sampled parent is not the highest-score/last node, append a short `## Parent Experiment Context` block (parent node id, score, solution excerpt, its own feedback) to `problem` before `_generate_solution(problem, parent_branch)` (strategy.py:157). No prompt-template change needed — it is plain string concatenation onto `problem`.
- **Untouched:** `get_best_experiment` / `checkout_to_best_experiment_branch` (strategy.py:552-569) — final deliverable stays argmax; `SearchNode` (base.py:41-67) — it already carries `parent_node_id`, `score`, `branch_name`; `orchestrator.py`; `benchmark_tree_search.py` (its own exploration lives at benchmark_tree_search.py:271-296, 331-381, gated by `exploration_budget_percent` and the `budget_progress >= 20` prune at :199-200).

### (b) New types and config keys

```python
# parent_selection.py (proposal)
@dataclass(frozen=True)
class ParentCandidate:
    node_id: int
    branch_name: str
    score: Optional[float]        # None = unscored; exploration-only weight
    children_count: int
    evaluation_valid: bool

class ParentSelectionPolicy(ABC):
    @abstractmethod
    def select(self, candidates: list[ParentCandidate], *,
               budget_progress: float, maximize: bool,
               rng: random.Random) -> Optional[ParentCandidate]: ...

def build_parent_policy(cfg: dict) -> ParentSelectionPolicy:
    # cfg.get("policy", "greedy") -> Greedy | Weighted | SoftSwitch | Mixture
```

```yaml
# optional key under strategies.generic presets (strategies.yaml:24-52)
# and mode params (config.yaml:35-45, 98-108); absent => greedy
parent_selection:
  policy: weighted        # greedy | weighted | soft_switch | mixture
  seed: 42
  lambda_sigmoid: 10.0    # weighted: sigmoid(lambda*(s-median)/MAD) * 1/(1+children)
  explore_until: 50       # soft_switch: uniform/novelty phase until this budget %
  elite_k: 3              # soft_switch: inverse-rank sampling over top-K after switch
  max_children_per_parent: 4   # soft_switch diversity cap
  mixture_lambda: 0.3     # mixture: lambda*uniform + (1-lambda)*softmax(alpha*score)
  softmax_alpha: 1.0
```

### (c) Backward compatibility

- **`SearchNode.score` semantics preserved.** The dataclass is not modified; `score` stays `Optional[float]`. `GreedyParentSelection` replicates today's exact semantics — `max` over non-error nodes with `(score or 0)` and the `maximize_scoring` negation (strategy.py:552-560) — so no compat property is needed. Non-greedy policies instead treat `score is None` / `evaluation_valid=False` as *unscored*: eligible for the exploration mass but excluded from exploitation weight, fixing the `None==0` inflation without changing stored data. Old `checkpoint.pkl` files (pickled `List[SearchNode]`, strategy.py:711-721) load unchanged.
- **Existing YAML modes work unchanged.** The key is read via `self.params.get("parent_selection", {})` with a code-level greedy default; neither `strategies.yaml` nor `config.yaml` requires edits, matching the pattern of every other generic param (strategy.py:68-90).
- **Both strategies keep working.** GenericSearch defaults to greedy (bit-identical parent choice). BenchmarkTreeSearch is untouched — it never calls `_get_best_branch`; its LLM-guided `select()`/`expand()` with `exploration_budget_percent` already covers this ground (benchmark_tree_search.py:271-296, 331-381).

### (d) Implementation steps

1. Create `parent_selection.py` with `ParentCandidate`, ABC, `GreedyParentSelection`, factory. Unit-test greedy parity against `get_best_experiment()` on synthetic histories (None scores, errors, minimize mode).
2. Wire `_select_parent()` into `GenericSearch.__init__`/`run()`, replacing strategy.py:154 and :163; parity test that default config yields the same `(parent_branch, parent_node_id)` as before on arbitrary histories.
3. Add `WeightedParentSelection` (median/MAD sigmoid × `1/(1+children_count)`). Tests: scale invariance (scores ×1000 → same distribution), offspring decay, minimize flip, all-None → uniform, empty history → `None` (caller falls back to `"main"`, preserving strategy.py:531).
4. Add `SoftSwitchParentSelection` (breakpoint behavior at budget 0/49/51/100, same-parent cap) and `MixtureParentSelection` (`lambda=1` → uniform, `lambda=0` → pure softmax).
5. Add the parent-context block in `run()` when the sampled parent is not the current best/last node; assert on the composed problem string.
6. Determinism test: fixed seed, run selections, `export_checkpoint()`/`import_checkpoint()` mid-sequence, verify identical continuation (seed derives from `len(node_history)`, so no RNG state is persisted).
7. Optionally document the key as a commented example in `strategies.yaml` and `search_strategies/README.md`.

### (e) Acceptance criteria

- **Default unchanged (required):** with `parent_selection` absent, a randomized-history property test shows `_select_parent()` == `get_best_experiment()` for 1,000 generated histories, and a 2-iteration `kapso.evolve()` smoke run (mocked agents) produces the same branch topology and same final checked-out branch as current `main`.
- Weighted policy: over 10k draws on a fixture, a parent with `children_count=5` is sampled measurably less than an equal-score parent with 0 children; empirical frequencies match analytic weights within 2%.
- Unscored nodes (`score=None`, from XML parse failure at strategy.py:654-659, or `evaluation_valid=False`) never win exploitation-phase sampling.
- Determinism across checkpoint resume (step 6 test) passes.
- `checkout_to_best_experiment_branch()` still returns the argmax branch regardless of policy.

### (f) Risks and mitigations

- **Short default runs (10 iterations, kapso.py:472) may waste budget exploring.** Greedy stays the default; docs recommend non-greedy only for long/noisy-eval runs; `soft_switch` breakpoints are percentages of `budget_progress`, which already folds in time/cost budgets (orchestrator.py:370-375).
- **Feedback/lineage mismatch** (orchestrator threads last node's feedback, orchestrator.py:430): mitigated by the parent-context block; residual risk is prompt confusion, bounded by keeping the block short and labeled.
- **Sampled parent's branch missing/corrupt:** branches are created and committed by `finalize_session` in the workspace repo (experiment_workspace.py:275-279) and are never deleted by GenericSearch; add a defensive fallback — if `create_experiment_session` fails on the sampled parent branch, retry once with the greedy choice.
- **Non-determinism from float ties:** break ties by `node_id` inside policies.
- **Final selection still argmax over noisy scores** — this mechanism diversifies lineage only; pair with Evaluator V2 (roadmap dependency) for full effect.

## 3. Bounded debug/repair operator with multi-draft roots (draft/debug/improve policy)

**Sources:** AIDE, AIRA-dojo, AI Scientist v2 · **Impact:** 3 · **Difficulty:** 2 · **Priority:** 4 · **Evidence:** AIDE `aide/agent.py` search_policy (`num_drafts=5`, `debug_prob=0.5`, `max_debug_depth=3`); AIRA-dojo `mcts.py` debug_cycle; AI Scientist v2 `parallel_agent.py` `_select_parallel_nodes`.

### (a) Target files and code

- `src/kapso/execution/search_strategies/generic/strategy.py` — the entire change lives here plus one prompt file.
  - `GenericSearch.run()` (strategy.py:127-211): replace the two greedy call sites — `parent_branch = self._get_best_branch()` (:154) and `parent_node_id=self._get_best_node_id()` (:163) — with one call to a new `_select_parent()` that returns `(stage, parent_node)`, where `stage ∈ {"draft","debug","improve"}`. Draft parents off `main` with `parent_node_id=None`; debug parents off a buggy leaf's `branch_name`; improve keeps today's best-node behavior.
  - `_get_best_branch` / `_get_best_node_id` (:526-538): keep as-is; `_select_parent()` delegates to them for the improve stage so improve semantics are byte-identical.
  - `__init__` (:63-113): read three new params via `self.params.get` alongside the existing ones (:68-99): `num_drafts` (default 1), `debug_prob` (default 0.0), `max_debug_depth` (default 3), plus optional `search_seed` for a `random.Random` instance.
  - `_generate_solution` (:213-301) and `_build_ideation_prompt` (:303-317): accept optional repair context (`failed_solution`, `failure_output`, `stage`); when `stage=="debug"`, render a new template instead of `ideation_claude_code.md`.
  - New file `src/kapso/execution/search_strategies/generic/prompts/ideation_debug_claude_code.md` (siblings already exist in that directory): repair-goal prompt taking `problem`, `repo_memory_brief`, `failed_solution`, `failure_output`.
- `src/kapso/execution/search_strategies/base.py` — `SearchNode` (base.py:30-77): add a derived, non-stored `is_buggy` property (no new dataclass fields; see (c)).
- `src/kapso/execution/search_strategies/strategies.yaml` (:27-52) and `src/kapso/config.yaml` (:34-45): additive keys only.
- **No changes** to `benchmark_tree_search.py` (it already has in-loop debugging via `code_debug_tries`, :107, and parent-chain selection in `_get_closest_experimented_parent`, :487-491), `orchestrator.py`, `experiment_session.py` (git already supports branching off failed branches: checkout of arbitrary `parent_branch_name` at :82-89, push of every branch at close :272-276, and `finalize_session` runs even on failed implementation, strategy.py:498-499).

### (b) New types and config keys

No new persisted fields — `stage` and `debug_depth` are derived structurally from the parent chain, exactly as AIDE does:

```python
# base.py — SearchNode additions (derived, pickle-safe)
@property
def is_buggy(self) -> bool:
    """Node failed: invalid eval, hard implementation failure, or no score at all."""
    if self.had_error or not self.evaluation_valid:
        return True
    if self.agent_output.startswith("Implementation failed:"):  # strategy.py:467 prefix
        return True
    return self.score is None and not self.feedback  # conservative: parse-failure only

# strategy.py — GenericSearch helpers
def _debug_depth(self, node: SearchNode) -> int: ...   # count consecutive buggy ancestors via parent_node_id
def _select_parent(self) -> tuple[str, Optional[SearchNode]]: ...  # draft/debug/improve policy
```

```yaml
# strategies.yaml — generic presets (additive; recommended values shown)
num_drafts: 3        # independent roots off main before any improve step
debug_prob: 0.5      # P(parent next iteration off a buggy leaf)
max_debug_depth: 3   # bound on consecutive debug ancestors
```

Policy: iterations 1..`num_drafts` → draft; else if a buggy leaf exists with `_debug_depth < max_debug_depth` and `rng.random() < debug_prob` → debug (most recent buggy leaf); else → improve (existing `get_best_experiment`, strategy.py:552-560).

### (c) Backward compatibility

- **`SearchNode.score` semantics:** unchanged. `score` stays `Optional[float]` (base.py:59); buggy nodes keep `score=None`; `get_best_experiment` still filters `had_error` and maxes `(score or 0)` (strategy.py:552-560), so debug nodes never displace a real best. `is_buggy` is a compat *property*, not a stored field — old pickled checkpoints (strategy.py:711-721 round-trips `node_history` via pickle) deserialize without missing-attribute errors because nothing new is stored, and `ExperimentResult.from_search_node` (base.py:117-134) is untouched.
- **Existing YAML modes:** all params are read with code defaults (`num_drafts=1`, `debug_prob=0.0`), so `strategies.yaml` PRODUCTION/MINIMAL (:27-52) and `config.yaml` GENERIC (:34-45) run bit-identically without edits: iteration 1 drafts off `main` (same as today's empty-history path), and every later iteration is improve-off-best. Ship recommended values in a new preset (e.g. `PRODUCTION_DEBUG`) rather than mutating PRODUCTION.
- **Both strategies keep working:** `GenericSearch` under defaults reduces exactly to the current greedy policy; `BenchmarkTreeSearch` is not edited at all and ignores the new keys (its params are read separately, benchmark_tree_search.py:104-119).

### (d) Implementation steps

1. Add `SearchNode.is_buggy` property (base.py); unit-test the four trigger conditions plus the parse-failure ambiguity case (`evaluation_valid=True, score=None, feedback=""`).
2. Add the three params + seeded RNG to `GenericSearch.__init__` and to the init printout (:101-108).
3. Implement `_debug_depth()` and `_select_parent()`; swap the call sites at strategy.py:154 and :163. Unit-test with synthetic `node_history` that (i) defaults reproduce `_get_best_branch`/`_get_best_node_id` outputs exactly, (ii) `num_drafts=3` yields three main-rooted nodes, (iii) `debug_prob=1.0` picks the buggy leaf and stops at depth 3.
4. Add `ideation_debug_claude_code.md`; extend `_generate_solution`/`_build_ideation_prompt` to render it with the parent's `solution` and the tail (~4,000 chars) of its `evaluation_output`/`agent_output` when stage is debug. This also revives the intent of the dead `previous_errors` plumbing (initialized :98-99, joined :453, never appended).
5. Wire stage into `run()` logging (`stage=debug depth=2 parent=generic_exp_4`) and pass the selected parent node through.
6. Add the `PRODUCTION_DEBUG` preset to strategies.yaml and document keys in config.yaml comments.
7. Checkpoint compat test: unpickle a `checkpoint.pkl` produced from current `main`, resume, and run one iteration.
8. Integration smoke test with a stubbed coding agent that fails once: assert the next node's `parent_branch_name` is the failed branch and the debug prompt contains the failure excerpt.

### (e) Acceptance criteria

- **Default-behavior regression (must pass first):** with no new YAML keys, a scripted 5-node history produces the identical `(parent_branch, parent_node_id)` sequence as current `main`; an end-to-end MINIMAL run shows no diff in branch topology.
- With `debug_prob=1.0`: a forced failure yields a child of the failed branch (git parent verified via `base_commit_sha`, experiment_session.py:94); a chain of failures reverts to improve after `max_debug_depth=3`.
- `get_best_experiment` never returns a node with `is_buggy=True` under any config.
- Old checkpoint loads and resumes (step 7); benchmark test suite passes untouched.

### (f) Risks and mitigations

- **Fuzzy buggy predicate** — `had_error` is never set in generic, `evaluation_valid` defaults True on feedback parse failure (strategy.py:613), `score=None` conflates unscored with failed. *Mitigation:* conservative predicate (explicit signals only), stage-decision logging, and revisit once WP10/Evaluator V2 lands `run_had_error`-grade semantics.
- **Wasted budget on debug chains** — generic iterations are full Claude Code sessions that already self-debug, so hard failures are rarer than in AIDE. *Mitigation:* default-off (`debug_prob=0.0`), depth bound, optional future cumulative debug-budget cap.
- **Multi-draft roots on seeded repos** — drafts duplicate work when starting from a working repo. *Mitigation:* default `num_drafts=1`; recommend >1 only for greenfield problems.
- **Orchestrator feedback mismatch** — the previous iteration's feedback is injected into `context` (orchestrator.py:386-388, carried at :431-432); after a debug node, that feedback describes the buggy node while the parent may be best. *Mitigation:* state the stage explicitly in the ideation prompt so the agent can reconcile.
- **Missing parent branch** (push failed at close, experiment_session.py:272-276). *Mitigation:* on checkout failure fall back to `_get_best_branch()` and log.
- **Reproducibility** — `debug_prob` randomness. *Mitigation:* `search_seed` param feeding `random.Random`.

## 4. Deterministic scoped history injection: journal / top-k / diverse-inspiration / ancestral / sibling memory scopes

**Sources:** AIRA-dojo (`operators/memory.py` MEM_OPS), AIDE (`journal.py` generate_summary), OpenEvolve (`prompt/sampler.py` `_format_evolution_history`) · **Impact:** 3 · **Difficulty:** 2 · **Priority:** 4

### (a) Target files and code

- `src/kapso/execution/search_strategies/generic/history_digest.py` (**new**): pure functions that turn `List[SearchNode]` into a bounded markdown digest. No I/O except an optional read of the experiment-history JSON for insights.
- `src/kapso/execution/search_strategies/generic/strategy.py`:
  - `GenericSearch.__init__` (strategy.py:63-99): parse a new `ideation_history` params dict next to the existing `self.params.get(...)` block (strategy.py:76, 87-90).
  - `_generate_solution` (strategy.py:213-301): after loading `repo_memory_brief` (strategy.py:233-239), build the digest from `self.node_history` and pass it to the prompt builder.
  - `_build_ideation_prompt` (strategy.py:303-317): currently renders only `problem` and `repo_memory_brief`; add `history_digest` and `history_instructions` variables.
  - Helper additions: `_sibling_nodes()` filtering `node_history` on `parent_node_id == self._get_best_node_id()` (parenting is greedy best-branch, strategy.py:161-166, 533-538) and `_lineage_nodes()` walking `parent_node_id` from the best node.
- `src/kapso/execution/search_strategies/generic/prompts/ideation_claude_code.md`: replace the hardcoded mandatory-MCP wording (lines 28, 73-76) with `{{history_instructions}}`, and add a bare `{{history_digest}}` line under "## Context" (after line 70). `render_prompt` is literal `{{var}}` replacement (core/prompt_loader.py:58-71), so both variables must always be passed.
- `src/kapso/execution/search_strategies/strategies.yaml` (generic presets, strategies.yaml:29-50) and `src/kapso/config.yaml` (GENERIC ~:37-45, MINIMAL ~:100-108): optionally mirror the new key once validated; code-level defaults make this non-blocking.
- **Not touched:** `store.py` (`ExperimentRecord` has no `parent_node_id`, store.py:38-57 — digest builds from in-memory `node_history` instead), `orchestrator.py` (feedback push at orchestrator.py:385-387 and `add_experiment` at :402-404 stay as-is), `benchmark_tree_search.py` (its own crude push-injection at :733-736 is out of scope).

### (b) New types and config keys

```python
# history_digest.py (proposal)
@dataclass
class HistoryDigestConfig:
    scopes: List[str] = field(default_factory=list)  # [] = disabled (default)
    # valid: "journal", "top_k", "recent", "sibling", "lineage"
    top_k: int = 5
    recent_k: int = 5
    max_chars: int = 4000
    include_insights: bool = True   # read-only json.load of experiment_history_path

def build_history_digest(nodes: List[SearchNode], cfg: HistoryDigestConfig,
                         maximize: bool, best_node_id: Optional[int],
                         insights_path: Optional[str] = None) -> str: ...
def node_failed(node: SearchNode) -> bool:  # GenericSearch never sets had_error
    ...  # infer from evaluation_valid is False, score is None, or error markers in feedback
```

```yaml
# strategies.yaml / config.yaml params (proposal)
ideation_history:
  scopes: ["top_k", "recent"]   # PRODUCTION suggestion after validation
  top_k: 5
  recent_k: 5
  max_chars: 4000
  include_insights: true
```

Scope semantics: `journal` = one line per node (AIDE); `top_k` = best-k with delta-vs-best annotations (OpenEvolve, only when both scores are non-None); `recent`/`sibling` = last-k / same-parent nodes (equivalent under greedy parenting — keep both names for forward compat); `lineage` = ancestral chain of the best node (AIRA debug scope).

### (c) Backward compatibility

- **SearchNode untouched.** No new fields on `SearchNode` (base.py:30-77); `parent_node_id`, `score`, `feedback`, `evaluation_valid` already exist. Old `checkpoint.pkl` files (strategy.py:711-721) unpickle unchanged. `score: Optional[float]` semantics are preserved: the digest formats `None` as "unscored" and never coerces to 0, while ranking reuses the exact `(score or 0)` + `maximize_scoring` logic of `get_best_experiment` (strategy.py:552-560), so best-node selection is bit-identical to today. No compat property is needed because no field changes; if a later refactor ever replaces `score`, the digest reads it through one accessor in `history_digest.py` only.
- **Existing YAML modes work unchanged.** `ideation_history` is read via `self.params.get("ideation_history", {})` with `scopes` defaulting to `[]`; none of strategies.yaml:29-50, config.yaml GENERIC/MINIMAL need edits. With scopes empty, `history_digest` renders as `""` and `history_instructions` renders as the current mandatory-MCP text verbatim, so the default prompt is byte-identical modulo one blank line.
- **Both strategies keep working.** GenericSearch: MCP `experiment_history` gate stays in `ideation_gates` defaults (strategy.py:76) for drill-down (`search_similar_experiments`). BenchmarkTreeSearch: zero code touched; it never reads `ideation_history` and keeps its own history block (benchmark_tree_search.py:733-736). Optional follow-up (not in this plan): reuse `build_history_digest` in its select/prune/solution_generation prompts (:350-354, :393-399, :698-748), which currently ask the LLM to "consider previous experiments" it is never shown.

### (d) Implementation steps

1. Add `history_digest.py` with `HistoryDigestConfig`, `build_history_digest`, `node_failed`, per-scope formatters, and per-node truncation (solution ≤300 chars, feedback ≤200) plus global `max_chars` cap. Unit-test on synthetic `SearchNode` lists (verify: empty history, all-None scores, minimize vs maximize, truncation).
2. Edit `ideation_claude_code.md`: introduce `{{history_instructions}}` (lines 28, 73-76) and `{{history_digest}}`; store the disabled-mode instruction text as a module constant equal to today's wording.
3. Wire `GenericSearch.__init__` to parse the config and `_generate_solution`/`_build_ideation_prompt` to pass the two new variables. Pass the digest as the **last** replacement or escape `{{` inside it (render_prompt is sequential literal replacement, prompt_loader.py:68-71).
4. Add golden test: `_build_ideation_prompt` output with default params equals the pre-change rendering (fixture captured before step 2).
5. Add `include_insights`: plain `json.load` of `self.experiment_history_path` (strategy.py:87-90), formatting `insight`/`insight_type` fields (store.py:53-57) into a "Distilled insights" block.
6. Run one MINIMAL-mode end-to-end job with `scopes: ["top_k", "recent"]`, inspect the logged ideation prompt, confirm ideation succeeds and MCP drill-down still fires when relevant.
7. After validation, flip PRODUCTION presets in strategies.yaml and config.yaml; leave MINIMAL disabled as the regression control.

### (e) Acceptance criteria

- **Default unchanged:** golden test from step 4 passes — with no `ideation_history` key, the rendered ideation prompt is identical to the pre-change prompt (mandatory MCP step-1 wording intact); an existing pre-change `checkpoint.pkl` resumes and runs.
- Unit tests: each scope produces expected ordering/content; `None` scores rendered as "unscored", never `0`; output length ≤ `max_chars`; `sibling` returns exactly nodes with `parent_node_id == best_node_id`.
- Integration: with scopes enabled, ideation prompt in logs contains the digest header and ≥1 experiment; a 3-iteration MINIMAL run completes with non-fallback solutions; iteration N's digest lists nodes 0..N-1.
- BenchmarkTreeSearch smoke run (or its existing tests) passes with zero diff to that file.
- Resilience: with the `experiment_history` gate removed from `ideation_gates`, an enabled-digest run still shows history in the prompt (the insurance case).

### (f) Risks and mitigations

- **Prompt bloat / token cost:** capped by `max_chars` and per-node truncation; journal scope is one line per node.
- **Template variable injection:** digest text containing `{{problem}}` would be substituted by `render_prompt`'s sequential replacement — escape `{{` in digest content (test included).
- **Failure inference is heuristic:** `had_error` is never set by GenericSearch, so `node_failed` may mislabel; mitigate with neutral labels ("unscored"/"no valid evaluation") rather than "FAILED".
- **Digest vs MCP contradiction:** agent may skip `search_similar_experiments` dedup; enabled-mode `history_instructions` explicitly keeps that call mandatory before finalizing.
- **Config drift (strategies.yaml x2 / config.yaml x2):** code-level default makes all YAML mirrors optional; only PRODUCTION is intentionally flipped.
- **Stale digest on resume:** digest builds from restored `node_history`, but the JSON insights file could lag if the workspace moved; guard the `json.load` with try/except and omit the insights block on failure.

## 5. Scoped verifier fan-out with complexity-adaptive budgets and grounded synthesis

**Sources:** PAT (Google) · **Impact:** 3 · **Difficulty:** 2 · **Priority:** 4 · **Evidence:** PAT arXiv:2606.28277 Sec 2 (Deep Review recall 55.2%→89.7% from orchestration), Fig. 1 Stages 1–2 (triage + Light/Medium/High thinking allocation), Stage 4 (dedup + grounded hallucination filter).

### (a) Target files and code

- `src/kapso/execution/search_strategies/generic/feedback_generator/feedback_generator.py` — the whole change lives here.
  - `FeedbackGenerator.__init__` (:62–89) creates exactly one agent via `CodingAgentFactory.create` at :89. Change: accept an optional `verifier_config`; keep the single agent for the legacy path; create one agent *per verifier* at fan-out time (agents are stateful — `initialize()` sets `self.workspace` and `generate_code` runs a subprocess with `cwd=self.workspace`, claude_code_agent.py:211, :295 — so per-verifier instances make thread fan-out safe).
  - `generate()` (:91–146, single `generate_code` at :142) becomes a dispatcher: if verifiers disabled → existing single-pass path unchanged; else triage → parallel scoped verifiers → deterministic grounding → synthesis → one `FeedbackResult`.
  - `PROMPT_PATH` (:60) stays; add sibling constants for new prompts under `feedback_generator/prompts/` (currently only `feedback_generator.md` exists): `verifier_eval_integrity.md`, `verifier_claim_vs_diff.md`, `verifier_score_rederivation.md`, `triage.md`, `synthesis.md`. All load via `load_prompt`, so they remain `KAPSO_PROMPTS_DIR`-overridable.
  - `_parse_response` (:202–249) is reused per verifier, but the silent fallback at :276–283 (parse failure → `evaluation_valid=True`) must not apply to scoped verifiers: a failed verifier becomes an explicit *abstain*, never an implicit "valid".
- `src/kapso/execution/orchestrator.py` — `_create_feedback_generator` (:140–178) reads `mode_config['feedback_generator']` (:152) and builds one `CodingAgentConfig`. Change: also parse an optional `verifiers` sub-block and pass it to the `FeedbackGenerator` constructor (~20 lines).
- `src/kapso/config.yaml` — optional `verifiers:` sub-block under both `feedback_generator` blocks (:57–65 DEFAULT, :120–128 MINIMAL). Absent block = current behavior.
- **Not touched:** `GenericSearchStrategy._generate_feedback` (strategy.py:575–622) reads only `stop/evaluation_valid/feedback/score` (:610–613); `BenchmarkTreeSearch` skips the feedback generator entirely (benchmark_tree_search.py:83, :446); `SearchNode`; `claude_code_agent.py` (per-verifier thinking budgets ride on the existing `agent_specific.env_overrides` hook, claude_code_agent.py:118–120, applied at :713–715).

### (b) New types and config keys

All types are module-private to `feedback_generator.py`:

```python
@dataclass
class ScopedVerifierReport:
    mandate: str                       # "eval_integrity" | "claim_vs_diff" | "score_rederivation"
    status: str                        # "ok" | "flagged" | "abstain" (parse/agent failure)
    evaluation_valid: Optional[bool]   # None when abstaining
    score: Optional[float]             # score_rederivation only
    claims: list[dict]                 # [{"summary", "file", "line", "quote"}]
    raw_output: str

@dataclass
class VerifierFanoutConfig:
    enabled: bool = False
    mandates: list[str] = field(default_factory=lambda:
        ["eval_integrity", "claim_vs_diff", "score_rederivation"])
    triage: bool = True                # cheap depth-sizing pre-pass
    light_model: Optional[str] = None  # boilerplate-depth verifiers
    heavy_thinking_tokens: Optional[int] = None  # -> env_overrides["MAX_THINKING_TOKENS"]
    synthesis_model: Optional[str] = None
```

```yaml
feedback_generator:
  type: "claude_code"
  model: "us.anthropic.claude-opus-4-6-v1"
  verifiers:                # NEW, optional
    enabled: true
    triage: true
    light_model: "us.anthropic.claude-haiku-4-5-v1"
    heavy_thinking_tokens: 16000
```

`FeedbackResult` (:26–41) is unchanged.

### (c) Backward compatibility

- **`SearchNode.score` semantics preserved:** synthesis still emits one `FeedbackResult` whose `score` overwrites `node.score` at strategy.py:611, exactly as today; `get_best_experiment` (strategy.py:552–560) continues ranking on `had_error` and `score or 0`. No compat property needed because no field changes; if we later attach `verifier_reports` to `FeedbackResult`, it gets a `None` default so `to_dict()` consumers and the orchestrator's reconstruction at orchestrator.py:414–421 are unaffected.
- **Existing YAML modes unchanged:** `verifiers` is absent in shipped configs, so `_create_feedback_generator` builds the generator exactly as at :169–178 and `generate()` takes the single-pass branch — byte-identical prompts and one agent call.
- **Both strategies keep working:** GenericSearch's only touchpoint is `feedback_generator.generate(...)` with an unchanged signature (strategy.py:598–607, wrapped in try/except at :617–620). BenchmarkTreeSearch never constructs or calls a feedback generator (benchmark_tree_search.py:83; handler feedback path at :657), so it is untouched by construction.

### (d) Implementation steps

1. **Pure refactor:** extract the current body of `generate()` into `_generate_single_pass()`; add unit tests for `_parse_response` on the four-tag format (prompt contract at prompts/feedback_generator.md:57–77). No behavior change.
2. **Config plumbing:** add `VerifierFanoutConfig`, parse it in `_create_feedback_generator` (orchestrator.py:140–178), thread into `FeedbackGenerator.__init__`. Default `enabled=False`; verify DEFAULT/MINIMAL modes still boot.
3. **Scoped prompts + per-verifier parsing:** three mandate prompts (each reuses the template variables from `_build_prompt`, :148–175) and a `_parse_verifier_response` that maps parse failure to `status="abstain"` instead of the :276–283 valid-by-default fallback.
4. **Triage pre-pass:** deterministic first cut — `git diff --numstat base..head` (mirroring `_get_commit_message`'s subprocess pattern, :177–200): eval-script or large core hunks → heavy budget (`heavy_thinking_tokens` via `env_overrides`), doc/config-only hunks → `light_model`.
5. **Fan-out:** `ThreadPoolExecutor(max_workers=len(mandates))`, one fresh agent per verifier (safe: `initialize()` writes nothing for the feedback config — no `claude_md_path`, no MCP servers, claude_code_agent.py:213–222; each call is an isolated subprocess, :291–300).
6. **Deterministic grounding:** for each claim, check cited file exists under `workspace_dir`, quoted line appears in `git show head:file` or the diff, and any re-derived score's literal appears in `evaluation_result`. Drop ungrounded claims; pure Python, no LLM.
7. **Synthesis:** one call on `synthesis_model` over surviving reports → dedup, severity-rank, emit the same four XML tags; rule: eval-integrity `flagged` ⇒ `evaluation_valid=false` **and** `score=null` (so a hacked eval cannot win best-node selection); all-abstain ⇒ fall back to `_generate_single_pass()`.

### (e) Acceptance criteria

- **Default-unchanged check:** with shipped `config.yaml` (no `verifiers` key), a stubbed-agent unit test asserts `generate()` invokes `generate_code` exactly once and returns a `FeedbackResult` identical to pre-change output on a canned response.
- Unit: verifier parse failure ⇒ `status="abstain"`, never `evaluation_valid=True`; fabricated file/line claim is dropped by grounding; eval-integrity flag ⇒ synthesized `score is None`.
- Integration: MINIMAL mode with `verifiers.enabled: true` on a toy goal completes ≥2 iterations; orchestrator log at orchestrator.py:407–411 shows score/validity; `node.score` populated at strategy.py:611; stop path (:424–427) still fires on goal achievement.
- Benchmark regression: one `benchmark_tree_search` smoke run, confirming zero feedback-generator calls.

### (f) Risks and mitigations

- **Cost/latency (3–5 agent calls in the sequential step-4 path, strategy.py:204):** parallel threads bound wall-time to the slowest verifier; triage routes boilerplate to `light_model`; existing 120s timeout per call caps the worst case.
- **`evaluation_valid` is a dead field** (set strategy.py:613, printed orchestrator.py:410, ignored by selection :552–560): route detection through `score=null` withholding (step 7). Caveat: `score or 0` makes null behave as quarantine for maximize goals but *best*-looking for minimize goals — document, and file the invalid-metric-semantics follow-up rather than expanding scope here.
- **Precision collapse / verifier disagreement:** exactly PAT Stage 4's problem; deterministic grounding plus synthesis severity-thresholding; feedback text cites only grounded claims.
- **Concurrent agents writing to the shared workspace:** prompts mandate read-only analysis; after fan-out, run `git status --porcelain` and hard-reset stray changes before synthesis.
- **Fail-closed overreach:** abstain (not "invalid") on verifier failure prevents infrastructure flakes from nuking legitimate scores; all-abstain falls back to today's single-pass behavior.

## 6. LLM holistic best-node adjudication with deterministic metric fallback

**Sources:** AI Scientist v2 · **Impact:** 3 · **Difficulty:** 2 · **Priority:** 4 · **Evidence:** AI Scientist `ai_scientist/treesearch/journal.py::get_best_node` (structured node-selection call with anti-metric-overtrust instruction, max-metric fallback, seed nodes excluded).

Kapso picks its parent branch every iteration and its final shipped branch with a pure scalar argmax over `score or 0` (`src/kapso/execution/search_strategies/generic/strategy.py:552-560`, `src/kapso/execution/search_strategies/benchmark_tree_search.py:248-256`), ignoring `evaluation_valid` (set at `strategy.py:613`, never consulted) and feedback text. This plan adds an opt-in LLM adjudication over the top-k candidates with a deterministic fallback to today's exact argmax.

### (a) Target files and code

- `src/kapso/execution/search_strategies/base.py` — the main new code lands here so both strategies share it. Add a concrete (non-abstract) helper `SearchStrategy._adjudicate_best_node(candidates: List[SearchNode]) -> Optional[SearchNode]` next to `_get_code_diff` (base.py:278-285). It reads `self.llm` (base.py:186, an `LLMBackend` from `SearchStrategyConfig.llm`, base.py:141) and `self.params` (base.py:187). Also add `_deterministic_best(candidates)` encapsulating the current `max(..., key=lambda x: (x.score or 0) if self.problem_handler.maximize_scoring else -(x.score or 0))` verbatim, and a memo dict `self._adjudication_cache: Dict[tuple, int]` initialized in `__init__` (base.py:176-225) but never exported in checkpoints.
- `src/kapso/execution/search_strategies/generic/strategy.py:552-560` — `GenericSearch.get_best_experiment` becomes: filter `not node.had_error` (unchanged, line 554), then `return self._adjudicate_best_node(valid)`. Callers `_get_best_branch` (:526-531), `_get_best_node_id` (:533-538), `checkout_to_best_experiment_branch` (:562-569), and the per-iteration parent selection (:154, :163) need no edits.
- `src/kapso/execution/search_strategies/benchmark_tree_search.py:248-256` — same one-line delegation in `BenchmarkTreeSearch.get_best_experiment`. The per-iteration return to the orchestrator (:233) and final checkout (:258-265) inherit it. Call pattern precedent: `select()` already does LLM-over-candidates with XML-tag parsing via `self.llm.llm_completion_with_system_prompt` (:331-381; the LLM method is `src/kapso/core/llm.py:97-131`).
- New prompt file `src/kapso/execution/search_strategies/prompts/best_node_adjudication.md`, loaded via `load_prompt`/`render_prompt` (`src/kapso/core/prompt_loader.py:40, 58`), same mechanism as `strategy.py:310` and `benchmark_tree_search.py:538`.
- External callers that must keep working unchanged: `src/kapso/execution/orchestrator.py:436, 459` and `src/kapso/kapso.py:583`.

### (b) New types and config keys

```python
# base.py (proposal)
@dataclass
class AdjudicationDecision:
    node_id: int
    used_fallback: bool          # True when LLM off, skipped, or parse failed
    reasoning: str = ""          # LLM's stated rationale, for logs only
```

Config keys read with `params.get()` code defaults (pattern: `generic/strategy.py:68-90`, `benchmark_tree_search.py:105-119`) — no mandatory YAML edits:

```yaml
search_strategy:
  params:
    best_node_adjudication: true          # default: false (off)
    adjudication_top_k: 5                 # candidates sent to the LLM
    adjudication_model: "gpt-4.1-mini"    # default: idea_generation_model
    adjudication_max_chars_per_field: 1500  # truncation for feedback/eval output
```

Prompt inputs per candidate (all already on `SearchNode`, base.py:41-67): `node_id`, `score`, `evaluation_valid`, `had_error`, truncated `feedback`, `evaluation_output`, `solution`; plus `maximize_scoring` (`src/kapso/environment/handlers/base.py:48`). Prompt carries the AI Scientist anti-overtrust instruction ("do not over-trust a single validation number; agent-defined evaluations may not be comparable across nodes; treat `evaluation_valid=false` scores as suspect") and requires `<selected_node_id>INT</selected_node_id>` plus a short reason. Parse with strict regex + `int()` — never `eval()` (unlike `benchmark_tree_search.py:380, 424`).

### (c) Backward compatibility

- **SearchNode semantics preserved:** no new fields on `SearchNode` or `TreeSearchNode`; `score` remains `Optional[float]` with identical meaning. Pickle checkpoints (`benchmark_tree_search.py:820-839` and GenericSearch's equivalent) round-trip unchanged; the memo cache lives on the strategy object and is rebuilt lazily, never serialized. `ExperimentResult.from_search_node` (base.py:117-134) untouched.
- **Existing YAML modes unchanged:** `best_node_adjudication` defaults to `False`, so `src/kapso/config.yaml:35-45` and `benchmarks/mle/config.yaml` / `benchmarks/ale/config.yaml` run byte-identical code paths — `_adjudicate_best_node` short-circuits to `_deterministic_best`, which is the current lambda verbatim (including the existing `score or 0` behavior, preserved deliberately; see risks).
- **Both strategies:** the helper only touches base-class `SearchNode` fields, so `TreeSearchNode` works by inheritance. `BenchmarkTreeSearch.run()`'s return at :233 still yields a node whose `should_stop` feeds the orchestrator loop; `GenericSearch`'s parent-branch and parent-node-id picks (:154, :163) stay mutually consistent via memoization (one adjudication per node-history state, keyed on `tuple(sorted(n.node_id for n in valid_candidates))`).

### (d) Implementation steps

1. Add `best_node_adjudication.md` prompt file with `{{candidates}}`, `{{goal_direction}}`, `{{problem}}` placeholders. Verify: `load_prompt` returns it in a REPL.
2. Add `AdjudicationDecision`, `_deterministic_best`, `_adjudicate_best_node` (with param reads, truncation, memo, regex parse, try/except-everything fallback) to `base.py`. Verify: unit test with a stub `LLMBackend` whose `llm_completion_with_system_prompt` returns a canned tag.
3. Delegate `GenericSearch.get_best_experiment` (:552-560) to the helper. Verify: parity test below passes with flag off.
4. Delegate `BenchmarkTreeSearch.get_best_experiment` (:248-256). Verify: same parity test against `TreeSearchNode` lists.
5. Add decision logging (`[Adjudication] node=… fallback=… reason=…`) mirroring existing print style.
6. Optional enablement: set `best_node_adjudication: true` in one experimental YAML profile and run an end-to-end smoke (`tests/test_evolve.py` path) to observe logs.

### (e) Acceptance criteria

1. **Default unchanged (must-pass):** property-style unit test builds randomized `node_history` lists (mixed `had_error`, `None` scores, both `maximize_scoring` values); with the flag unset, `get_best_experiment()` returns the identical node as the pre-change lambda for both strategies, and a stub `LLMBackend` that raises on any call proves zero LLM invocations.
2. **Fallback correctness:** flag on + stub returning garbage (no tag, non-int, unknown node_id) or raising → result equals `_deterministic_best`; no exception escapes `get_best_experiment`.
3. **Adjudication honored:** flag on + stub selecting a valid non-argmax candidate → that node is returned; a second call with unchanged history performs no second LLM call (memo assert via stub call-counter), so `_get_best_branch`/`_get_best_node_id` agree.
4. **Wiring intact:** orchestrator `SolveResult.best_experiment` (orchestrator.py:459) and `checkout_to_best_experiment_branch` (kapso.py:583) resolve to the same branch in a flag-on integration test.
5. Single-candidate and empty-candidate cases skip the LLM entirely (return the node / `None`).

### (f) Risks and mitigations

- **Winner flip-flop / repeated cost:** `get_best_experiment` is called several times per iteration (strategy.py:154, 163, 528, 535, 564; benchmark:233, 260; orchestrator:436, 459). Mitigation: memoize per candidate-set fingerprint; log cache hits.
- **Parse fragility / injection via feedback text:** feedback and eval output are LLM/agent-generated. Mitigation: strict regex + int cast with full deterministic fallback (never `eval()`), truncate fields, instruct the model that candidate text is data.
- **Nondeterminism harms reproducibility:** off by default; every decision logged with reasoning and `used_fallback` so runs are auditable.
- **Known quirk preserved:** the fallback keeps `score or 0`, which in minimize mode ranks scoreless nodes above positive-scored ones. Intentionally out of scope to keep the parity guarantee; file as a separate deterministic quick-win (also: filter `evaluation_valid=False` deterministically).
- **Low value in benchmark mode:** handler-normalized scores make argmax already correct there; keep the flag off in `benchmarks/*/config.yaml` and recommend enabling only for generic-mode runs with agent-defined evaluations.
- **Thin evidence inputs:** no artifact analyses exist yet (awaits Evaluator V2); prompt is designed so richer fields can be appended without interface changes.

## 7. Cross-task prior-wisdom store: end-of-run distillation with embedding-threshold prefetch

**Sources:** EvoMaster / ML-Master 2.0 · **Impact:** 3 · **Difficulty:** 2 · **Priority:** 4 · **Evidence:** ML-Master 2.0 arXiv:2601.10402 Sec 3.3.3 ((h_n, w_n) pairs), Sec 3.4.1 (cos>δ prefetch), Table 2 L3 ablation (72.7%→54.5% medal rate without warm-start tier); EvoMaster Sec 6.3.1 knowledge-prefetch stage.

### (a) Target files and code

1. **New package `src/kapso/execution/memories/wisdom_memory/`** (sibling of `experiment_memory/`, exported from `src/kapso/execution/memories/__init__.py:7-8` alongside `ExperimentHistoryStore`):
   - `store.py` — `WisdomStore`: JSON-first persistence with embeddings stored inline, cosine-threshold `prefetch()`. Modeled on the dual-storage pattern of `ExperimentHistoryStore` (`store.py:70-134`) but **global**, not workspace-scoped: default path `~/.kapso/wisdom/wisdom_store.json`, overridable via `KAPSO_WISDOM_DIR`. Embeddings computed client-side via `LLMBackend.create_embedding` (`src/kapso/core/llm.py:360-378`), sidestepping the broken server-side vectorizer path (collection declares `text2vec_openai` at `store.py:408` but compose sets `DEFAULT_VECTORIZER_MODULE: 'none'` with no `OPENAI_APIKEY`, `services/infrastructure/docker-compose.yml:14-15`, and the client connects without OpenAI headers, `store.py:123-126`). Cosine is computed locally over the (small — one record per run) store; Weaviate is an optional later mirror via `near_vector`, not a dependency.
   - `distiller.py` — `WisdomDistiller`: one LLM call at run end, directly following the `InsightExtractor` template (`insight_extractor.py:66-109` lazy LLM, `:272-280` JSON-mode `_call_llm`, `:282-312` parse-with-fallback).

2. **`src/kapso/execution/orchestrator.py`** — distillation hook in the `finally` teardown of `solve()` (`orchestrator.py:441-456`), inserted before `self.experiment_store.close()` at `:445-449`. It consumes `self.goal` (`:84`), `self.search_strategy.get_experiment_history()` and `get_best_experiment()` — abstract on `SearchStrategy` (`base.py:306, 319`) and implemented by both strategies (`benchmark_tree_search.py:235, 248`; `generic/strategy.py:540, 552`). Roughly +15 lines, fully guarded by config and try/except.

3. **`src/kapso/kapso.py`** — prefetch hook in `evolve()` where `combined_context` is assembled (`kapso.py:529-551`): prepend a `## Prior wisdom from similar past runs` block to `user_context` before `GenericProblemHandler(...)` is built at `:547-552`. The handler already injects `additional_context` under `# Additional Context` (`src/kapso/environment/handlers/generic.py:122-127`), which flows into `get_problem_context()` consumed at the top of the loop (`orchestrator.py:362, 383-387`) — i.e., before first ideation, with zero strategy changes. Roughly +10 lines.

No changes to `SearchNode` (`base.py:31-77`), `FeedbackGenerator`, either strategy, the MCP gates, or the three XML tag contracts.

### (b) New types and config keys

```python
# wisdom_memory/store.py
@dataclass
class WisdomRecord:
    run_id: str                  # uuid
    goal: str                    # raw task descriptor
    goal_embedding: List[float]  # text-embedding-3-large, via LLMBackend.create_embedding
    wisdom: str                  # distilled markdown: templates, priors, strategies
    best_score: Optional[float]
    n_experiments: int
    n_errors: int
    timestamp: str
    source_workspace: str        # provenance/debugging

class WisdomStore:
    def add(self, record: WisdomRecord) -> None: ...          # atomic write (tempfile + os.replace)
    def prefetch(self, goal: str, threshold: float, k: int) -> List[WisdomRecord]: ...
    # local cosine over goal_embedding; returns [] on empty store or embedding failure

# wisdom_memory/distiller.py
class WisdomDistiller:
    DEFAULT_MODEL = "gpt-4o-mini"   # mirrors InsightExtractor (insight_extractor.py:87)
    def distill(self, goal: str, nodes: List[SearchNode], best: Optional[SearchNode]) -> Optional[str]: ...
```

Config: a per-mode YAML block plus env overrides (env wins, matching the `WEAVIATE_URL` pattern at `orchestrator.py:123` and `store.py:467-487`):

```yaml
# config.yaml, optional per-mode block (absent in GENERIC/MINIMAL today → feature off)
wisdom_memory:
  enabled: true
  store_dir: "~/.kapso/wisdom"     # env: KAPSO_WISDOM_DIR
  similarity_threshold: 0.80       # env: KAPSO_WISDOM_SIM_THRESHOLD
  max_prefetch: 3                  # env: KAPSO_WISDOM_MAX_PREFETCH
  min_experiments_to_distill: 2
```

### (c) Backward compatibility

- **SearchNode.score semantics untouched.** The mechanism is a read-only consumer of `SearchNode` (`base.py:59`, `score: Optional[float] = None`). The distiller applies the same defensive filter the store already uses (`store.py:254`: exclude `had_error` and `score is None`) and additionally requires `evaluation_valid` (`base.py:61`) before treating a node as evidence for "what worked". No compat property is needed because no field changes; nothing writes to `SearchNode`.
- **Existing YAML modes work unchanged.** `load_mode_config` (`core/config.py:23-42`) returns a dict; both hooks gate on `self.mode_config.get('wisdom_memory', {}).get('enabled')` (or env). `GENERIC` and `MINIMAL` (`config.yaml:34, 97`) have no such block, so both hooks are no-ops by default — bytes of the first-ideation prompt are identical.
- **Both strategies keep working.** Hooks live entirely in strategy-agnostic code: teardown is the shared `finally` block both `GenericSearch` and `BenchmarkTreeSearch` flow through (`orchestrator.py:441-456`), using only the abstract `SearchStrategy` API; prefetch is upstream of the orchestrator in `Kapso.evolve()`. Callers constructing `OrchestratorAgent` directly (bypassing `evolve()`) get distillation but no injection — safe, additive.

### (d) Implementation steps

1. Create `wisdom_memory/store.py` with `WisdomRecord` + `WisdomStore` (JSON load/save copied from `store.py:361-397`, atomic write, local cosine). Unit-test `prefetch()` with hand-built embeddings: above/below threshold, empty store, corrupt file.
2. Create `wisdom_memory/distiller.py` following `InsightExtractor`; prompt asks for robust templates, stable hyperparameter/config priors, and effective strategies as compact markdown; returns `None` on parse failure (no fallback record — unlike `insight_extractor.py:314`, we prefer silence over noise). Unit-test with a stubbed `LLMBackend`.
3. Export from `memories/__init__.py`; add config keys to `config.yaml` comments only (off by default).
4. Wire distillation into `orchestrator.py` teardown (guarded, try/except-wrapped, skips runs with `< min_experiments_to_distill` valid nodes). Verify with a 2-iteration `MINIMAL` run that a record appears in `KAPSO_WISDOM_DIR`.
5. Wire prefetch into `kapso.py:529-551`. Verify injected block appears under `# Additional Context` in `handler.get_problem_context()` output.
6. E2E: run task A with wisdom enabled, then near-duplicate task B; assert B's first prompt contains A's wisdom; then a dissimilar task C; assert no injection.
7. (Optional, separate PR) Weaviate mirror using `near_vector` with distance metadata and a `PriorWisdom` collection with `Configure.Vectorizer.none()`, plus an MCP gate reusing the `WEAVIATE_URL` propagation at `presets.py:224-227`.

### (e) Acceptance criteria

- **Default-unchanged check:** with no `wisdom_memory` config and no `KAPSO_WISDOM_*` env vars, (i) `handler.get_problem_context()` output for a fixed goal is byte-identical to pre-change, (ii) no `~/.kapso/wisdom/` directory is created, (iii) existing experiment-memory tests pass untouched.
- Distillation: after a `MINIMAL`-mode run with ≥2 valid nodes, exactly one `WisdomRecord` exists with non-empty `goal_embedding` and `wisdom`; a run whose every node has `had_error=True` produces no record.
- Prefetch: threshold=0.80 returns the similar-task record and excludes the dissimilar one; `max_prefetch` caps output; empty store and `create_embedding` failure (returns `[]`, `llm.py:378`) both yield clean no-injection.
- Teardown resilience: a distiller exception does not change `SolveResult` (`orchestrator.py:458-464`).
- Both strategies: one smoke run each (`generic`, `benchmark_tree_search`) with the feature on, completing without error.

### (f) Risks and mitigations

- **Negative transfer from fragile scores** (parse-failure defaults can yield `score=None`/`evaluation_valid=True`; `GenericSearch` rarely sets `had_error`): distill only from nodes passing the triple filter (numeric score, `evaluation_valid`, no error); frame injected text as *advisory prior wisdom, to be validated against the current task*, never as instructions; include `best_score`/`n_experiments` so the ideation agent can weigh it.
- **Low prefetch precision on Kapso's heterogeneous task mix:** conservative default threshold (0.80), `max_prefetch=3`, and injection capped (~2k chars per record) so a bad match costs little context.
- **Concurrent runs writing one global file:** atomic replace + read-merge-write on `add()`; records are append-only and keyed by `run_id`, so last-writer-wins merges are lossless.
- **Unbounded growth / duplicates:** dedup near-identical goals on write (cosine ≥ 0.95, mirroring `DUPLICATE_THRESHOLD`, `store.py:92`), keep the highest-`best_score` record per cluster, cap store at ~500 records.
- **Embedding/LLM cost and availability:** one embedding + one small-model call per run (~cents); any failure degrades to no-op, mirroring the graceful-fallback philosophy already in `store.py:129-131, 305-308`.
- **Cold-start disappointment:** first runs get nothing by design; document that value accrues for repeat-domain users, and log `[WisdomStore] 0 records above threshold` so the silence is observable.

## 8. Atomic typed hypothesis/plan contract for proposals

**Sources:** AIDE, EvoAgentX · **Impact:** 3 · **Difficulty:** 2 · **Priority:** 4 · **Evidence:** AIDE `aide/agent.py` `_improve` prompt (atomicity directive) + `node.plan`; EvoAgentX `sew_optimizer.py` (`convert_to_scheme`/`parse_from_scheme`, parse failures return the original graph).

### (a) Target files and code

- `src/kapso/execution/search_strategies/generic/prompts/ideation_claude_code.md:81-103` — the `<solution>` output contract (`# Core Idea / # Why This Approach / # Solution Steps / # Hyperparameters / # Rationale`). Add an atomicity directive ("propose exactly ONE atomic, experimentally attributable change; do not bundle unrelated improvements") plus a new mandatory first section `# Hypothesis` ("If we change X, metric Y should move because Z"), both injected via a `{{atomicity_directive}}` template variable so the rendered prompt is byte-identical when the feature is off (`render_prompt` is plain `{{var}}` substitution, `src/kapso/core/prompt_loader.py:58-71`).
- `src/kapso/execution/search_strategies/generic/strategy.py` — `GenericSearch.__init__` (strategy.py:63-99): read a new `atomic_hypothesis` param. `_build_ideation_prompt` (strategy.py:303-317): pass the directive variable. Add `_extract_hypothesis(solution)` next to `_extract_solution_from_output` (strategy.py:319-334) that regex-matches the `# Hypothesis` section inside the solution text and returns `""` on any failure. Set `node.hypothesis` at node creation (strategy.py:161-166). Because the hypothesis stays embedded in `node.solution`, it reaches the implementation prompt (strategy.py:517, `"solution": solution`) and the FeedbackGenerator (strategy.py:600, `idea=node.solution`) with zero plumbing.
- `src/kapso/execution/search_strategies/base.py:30-67` — `SearchNode` dataclass: add defaulted `hypothesis: str = ""` in the Step-1 block (after `solution` at base.py:45).
- `src/kapso/execution/memories/experiment_memory/store.py` — `ExperimentRecord` (store.py:37-57): add `hypothesis: str = ""`. `add_experiment` (store.py:143-161): pass `hypothesis=getattr(node, "hypothesis", "")` (the orchestrator calls this with the raw node, `src/kapso/execution/orchestrator.py:404`). `_load_from_json` (store.py:361-388): add `hypothesis=e.get("hypothesis", "")`. `_save_to_json` (store.py:390-397) uses `asdict`, so persistence needs no change.
- `src/kapso/gated_mcp/gates/experiment_history_gate.py:233-260` — `_format_experiments`: when `getattr(exp, "hypothesis", "")` is non-empty, render a one-line `**Hypothesis:** ...` above `**Solution:**` so the next ideation call sees (hypothesis, score) pairs directly via `get_top_experiments`/`get_recent_experiments`.
- Not changed in v1: `FeedbackGenerator.generate` signature (`generic/feedback_generator/feedback_generator.py:91-101`) and the `BenchmarkTreeSearch` inline multi-solution prompt (`benchmark_tree_search.py:698-726`, parsed at :757). Both are explicit follow-ups (see (d) step 8).

### (b) New types and config keys

```python
# base.py — SearchNode addition (immutable default => class attribute,
# so old pickled nodes resolve via getattr fallback)
hypothesis: str = ""   # single falsifiable claim this node tests; "" = legacy/unparsed

# store.py — ExperimentRecord addition
hypothesis: str = ""
```

Optional follow-up type (typed plan, EvoAgentX-style; NOT in v1 scope):

```python
@dataclass
class ProposedPlan:
    hypothesis: str
    steps: List[str]                 # from "# Solution Steps"
    expected_effect: str = ""        # direction + rough magnitude on the metric
    raw_text: str = ""               # always kept; parse failure => plan is None, solution untouched
```

Config key (strategy `params`, read like existing keys at strategy.py:68-90):

```yaml
strategies:
  generic:
    params:
      atomic_hypothesis: true   # default false; off = current prompts byte-identical
```

### (c) Backward compatibility

- **SearchNode.score semantics untouched.** This mechanism only adds fields; `score: Optional[float]` (base.py:59) is still written by the agent-tag extraction (strategy.py:195-196) or FeedbackGenerator (strategy.py:611), and best-node selection (`(node.score or 0)`, strategy.py:547/559) is unaffected. No compat property is needed because nothing reads or reinterprets score.
- **Old checkpoints and JSON.** GenericSearch checkpoints pickle `node_history` (strategy.py:711-718); pickle bypasses `__init__`, but a plain-string dataclass default lives on the class, so `node.hypothesis` on an old node resolves to `""`; the one external reader (`add_experiment`) additionally uses `getattr(..., "")`. Existing `experiment_history.json` files load via the `.get()`-with-default pattern already used for insight fields (store.py:369-383).
- **Existing YAML modes unchanged.** `atomic_hypothesis` defaults to `False`; `params.get` means no YAML edits are required anywhere, and with the flag off the `{{atomicity_directive}}` variable renders to `""` so the ideation prompt is unchanged.
- **Both strategies keep working.** `TreeSearchNode` subclasses `SearchNode` (benchmark_tree_search.py:36-64) and constructs nodes by keyword (benchmark_tree_search.py:316-322), so the defaulted field is inert; its hardcoded prompt and `<solution>` parsing (benchmark_tree_search.py:698-757) are untouched in v1, so `benchmark_tree_search` runs are bit-for-bit identical. GenericSearch with the flag off behaves identically because `_extract_hypothesis` only runs against unchanged output and degrades to `""`.

### (d) Implementation steps

1. Add `hypothesis: str = ""` to `SearchNode` (base.py:45 area). Verify: construct with/without the kwarg; `str(node)` unchanged.
2. Add `hypothesis` to `ExperimentRecord` + `add_experiment` (`getattr`) + `_load_from_json` (`.get`). Verify: load a pre-change JSON fixture; round-trip save/load.
3. Edit `ideation_claude_code.md`: insert `{{atomicity_directive}}` placeholder above "## Output Format"; the directive text (rendered only when enabled) mandates one atomic change and the `# Hypothesis` section.
4. In `GenericSearch.__init__` read `self.atomic_hypothesis = self.params.get("atomic_hypothesis", False)`; in `_build_ideation_prompt` pass the directive string or `""`. Verify: rendered prompt with flag off equals current rendering exactly.
5. Add `_extract_hypothesis` (regex `^#\s*Hypothesis\s*\n(.*?)(?=\n#\s|\Z)`, DOTALL/MULTILINE, exception-safe → `""`); populate `node.hypothesis` at strategy.py:161-166.
6. Extend `_format_experiments` in `experiment_history_gate.py` with the guarded hypothesis line.
7. Tests (see (e)); run one smoke iteration with the flag on.
8. Follow-ups, separately reviewable: (i) add the same directive + `# Hypothesis` section to the BenchmarkTreeSearch inline prompt (benchmark_tree_search.py:698-726) and populate `hypothesis` at node creation (:316-322); (ii) `ProposedPlan` typed parsing with parse-failure → `None` fallback; (iii) thread `hypothesis` as a first-class FeedbackGenerator input (signature + `_build_prompt` at feedback_generator.py:148-175).

### (e) Acceptance criteria

- **Default-off invariance:** with `atomic_hypothesis` absent, `_build_ideation_prompt(problem, brief)` output is byte-identical to the pre-change rendering (snapshot test), and a GenericSearch iteration produces `node.hypothesis == ""` with all other node fields identical in shape.
- Loading a pre-change `experiment_history.json` and a pre-change `checkpoint.pkl` (fixture with the attribute deleted) raises nothing; `getattr(node, "hypothesis", "")` returns `""`.
- Flag on: a stubbed ideation output containing `# Hypothesis` yields a non-empty `node.hypothesis`; the same record's JSON contains the field; `get_top_experiments` MCP output shows the `**Hypothesis:**` line.
- Fallback safety: ideation output with no hypothesis section (or no `<solution>` tags at all, exercising strategy.py:332-334) leaves `solution` intact and `hypothesis == ""` — no exception, mirroring EvoAgentX's return-original-on-parse-failure property.
- `benchmark_tree_search` unit/smoke path runs with zero diff in its prompts and parsed solutions.

### (f) Risks and mitigations

- **Hypothesis/diff divergence:** the implementation agent may bundle scaffolding or eval changes beyond the stated hypothesis, and FeedbackGenerator reads the real git diff. Mitigation: hypothesis rides inside `idea=node.solution`, so feedback can already flag scope creep; follow-up (iii) can add an explicit "note if the diff exceeds the hypothesis" instruction. Do not claim causal attribution downstream.
- **Slower early progress under cost budget:** one-change-per-iteration can waste expensive three-agent iterations early. Mitigation: default off; enable for long history-driven runs; optionally soften the directive when `budget_progress` (strategy.py:127) is low.
- **LLM ignores the contract:** parse degrades to `""` with no corruption; monitor hypothesis-fill rate in history JSON.
- **Prompt-regression blast radius:** confined to one gated template; benchmark strategy untouched; snapshot test locks the off-path.
- **Typed-plan scope creep:** near-zero standalone value today; kept out of v1 and behind the same fallback semantics if ever built.

## Sequencing and shared dependencies

Several plans touch the same contracts and must be ordered deliberately. Plan 1 is the foundation: it hardens the `FeedbackResult`/`SearchNode` evidence contract (adds `parse_ok`, `evaluation_output_from_agent`, and — crucially — makes `had_error`/`evaluation_valid` mean something), and plan 1 explicitly notes that its interim `had_error` escalation is later replaceable by mechanism 2's invalid-score semantics. Plans 2, 3, and 6 all consume that evidence: 2 and 3 both replace the *same two call sites* in `GenericSearch.run()` (strategy.py:154 and :163) with a `_select_parent()` and so cannot land independently — they should be reconciled into one selection layer, most naturally by landing plan 2's `ParentSelectionPolicy` abstraction first and expressing plan 3's draft/debug/improve staging as a policy (or a stage wrapper) on top of it; plan 3's `is_buggy` predicate also becomes far less "fuzzy" once plan 1's fail-closed marking exists. Plan 6 wraps `get_best_experiment` in both strategies and is compatible with either, but its parity guarantee is easiest to preserve if it lands after the selection refactor settles. Plan 5 lives in the same `feedback_generator.py` as plan 1 and reuses `_parse_response`; plan 1's strict/fail-closed parse semantics and plan 5's single-pass extraction refactor (its step 1) should land before the fan-out itself. Plans that add `SearchNode` fields (1: `evaluation_output_from_agent`; 8: `hypothesis`) should merge before consumers that read richer node evidence (4's digest, 6's adjudication prompt, 7's distiller); all of those consumers use `getattr`-with-default and so tolerate any order, but they benefit from the richer fields being present.

A sane implementation order by rank, with grouping: **wave 1** — plan 1 (evidence contract, unblocks everything downstream); **wave 2** — plans 2 and 3 as a single reconciled parent-selection stack, plus plan 8 in parallel (small, additive `SearchNode`/`ExperimentRecord` field with no overlap with selection); **wave 3** — plan 6 (adjudication over the settled `get_best_experiment`) and plan 5 (verifier fan-out atop plan 1's parser changes); **wave 4** — plan 4 (history digest) and plan 7 (wisdom store), read-only consumers with no structural dependencies that can be pulled forward at any time if staffing allows. Every plan is default-off or default-identical, so waves can ship independently without behavior change until flags are enabled. Shared follow-ups noted across multiple plans — proper invalid-score/minimize-mode semantics (`score or 0`), Evaluator V2-grade `run_had_error`, and threading `hypothesis` into the FeedbackGenerator — should be tracked as separate items rather than folded into any single wave.

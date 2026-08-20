# Platform Unification: one task-agnostic platform, N benchmark packs

Status: EXECUTED 2026-08-20 — shipped to main as a fast-forward push of `unify/platform` (§11).
Scope: merge `feat/relbench-benchmark` (main+373) and `worktree-ioai-2025` (main+157) into `main`,
and land the end-state: **each benchmark = a handler + a knowledge bank (+ declarations + a thin
harness adapter); `src/kapso` = a fully task-agnostic evolve platform.**

## 0. Decision log

| # | decision | ruling |
|---|---|---|
| 1 | Merge vehicle | DECIDED: integration branch `unify/platform` → single merge to main |
| 2 | relbench-learning interplay | set aside for now (user merges that branch later) |
| 3 | Web in implementation sessions | DECIDED: `implementation_web` knob, default true; relbench mode false |
| 4 | Ideation research gate | DECIDED: `research` stays in the platform default gates (restored); benchmarks not using it trim it in their own config |
| 5 | Ensemble 70/30 timing split | DECIDED: no-split is the default; the split becomes an optional config knob |
| 6 | `honor_agent_stop` default | DECIDED: True (kaggle handler declares False) |
| 7 | `OPENAI_API_KEY` to codex children | DECIDED: passthrough default; purity via per-benchmark `env_strip` |
| 8 | Tree-path surface | DECIDED: declare it in `base.py`, drop the "deprecated" label |
| 9 | Handler discovery | DECIDED: no registry — runners construct handlers explicitly |
| 10 | Config deep-merge layering | DECIDED: yes — own commit immediately after the main merge |
| 11 | Runner doctrine | DECIDED: benchmarks keep their own `runner.py`, bounded by "must never change the main framework" |
| 12 | `strategy.py` split | DECIDED: done **inside the merge** (§4b) — unify in the monolith first, then mechanical split, both on the integration branch |
| 13 | Bank convention | DECIDED: `knowledge_bank/` with a single `INDEX.md` entry point; internal layout and semantics are handler-defined |
| 14 | RelBench bank consolidation | DECIDED: not doing it — relbench campaign concluded; its four layers stay as-is |
| 15 | Bank versioning policy | DEFERRED |
| 16 | `benchmarks/ioai` dead `contest_economics` generation | DECIDED: delete in hygiene commit |
| 17 | Hygiene list (§5/§7) | DECIDED: do it all |

---

## 1. The principle

A benchmark contributes exactly three things:

1. **A handler** — the single benchmark-tuned code surface: problem context, scoring
   direction/authority, evaluation tail (`final_evaluate`), and the small set of behavioral
   declarations the platform honors (stop authority, insured-deliverable reserve, run-selection
   stamping).
2. **A knowledge bank** — distilled knowledge in one standard shape (§3.3), reaching the model
   as staged files + a context pointer. The platform never parses it.
3. **Declarations, not code** — everything else the benchmark wants (models, gates, lanes,
   web policy, budgets, staged files) is config the platform interprets.

Plus a thin harness adapter (the runner) whose allowed scope is pinned by doctrine (§3.5).
Everything two campaigns needed independently is, by definition, platform. Everything only one
benchmark could ever want lives in its pack.

## 2. Where we are (verified topology)

```
main (de5d3edb, Jul 18)
  └─ shared campaign segment (56 commits, Jul 18–23: ioai_tasks/kaggle birth, early platform work)
       └─ c81ec08d  ← true fork point (Jul 23)
            ├─ worktree-ioai-2025      +101 commits (IOAI/Kaggle campaign, → Aug 6)
            └─ feat/relbench-benchmark +317 commits (RelBench campaign, → Aug 20)
```

- `main == origin/main`; **both branches are pure supersets of main** — all reconciliation is
  between the two tips, none against main.
- Cross-porting kept most of the framework identical on both tips: a dry-run `git merge-tree`
  shows **7 textual conflicts**; semantic divergence concentrates in 9 src files + 6 generic
  prompts + 10 tests (§4).
- Each tip carries a stale copy of the other's benchmark dirs — resolved by ownership (§5).
- Stray branch `worktree-ioai2025` holds one commit (`1a4b95ce`, data_extraction gitignore) —
  cherry-picked in; worktree + branch then deleted.

### The three mental models, compressed

**Main**: the skeleton both campaigns grew from — `OrchestratorAgent` loop → search strategy
(`generic` ideation→implementation→feedback, or `benchmark_tree_search` handler-scored) →
coding-agent adapters → workspace/memories. Known weaknesses the unification addresses:
bifurcated handler contract (tree surface duck-typed through a "deprecated" yet load-bearing
dataclass), `Kapso.evolve()` can't take a custom handler, benchmark configs wholesale-replace
the platform config (drift), runners carry real logic.

**relbench line** built: evaluation governance (archive + sandbox + maintainer hardening +
manifest-of-record), lens replanner + design axes, return-economics shaping, LLM request
timeouts, embedding-based experiment search (Weaviate deleted), transition freeze, insured
finalization; benchmark knowledge concentrated in `benchmarks/relbench/` (kept as-is per #14).

**ioai line** built: kernel-slot ticketing, preflight, submit-and-learn campaign shape (all
correctly benchmark-side) + platform pieces rb lacks: crashed-lane fallback-model retry,
selector hardening, ideation transcript naming, `honor_agent_stop`, web-search
`reasoning_effort` strip (without it research calls silently return empty), load-bearing
`mcp>=1.9,<2` + `litellm==1.75.0` pins, and the knowledge-bank prompt hooks.

## 3. Target architecture

### 3.1 Platform (`src/kapso`) — owns, post-merge

- Orchestrator loop incl. insured finalization, advisory-stop honoring (`honor_agent_stop`),
  transition freeze, bounded bridge deadlines.
- Generic strategy: ensemble ideation (lens planner **and** replanner, design axes,
  campaign-state brief), K-way node expansion with lane env pins + lane briefs, selector
  hardening, implementation-CLI choice + fallback-model retry, web gating
  (`implementation_web` knob), shared campaign cache, evaluation governance in-loop,
  transcript persistence, optional ensemble time-split knob (off by default).
- Tree strategy with a declared handler surface (§3.2).
- Coding agents: claude_code (+`disallowed_tools`, print-mode dead-tools), codex
  (OPENAI_API_KEY passthrough; strip via `env_strip`), oss_claude_code.
- Evaluation governance modules (`evaluation_archive.py`, `evaluation_archive_sandbox.py`,
  integrity, maintainer) — sandbox half vendored per-benchmark, byte-identity pinned by test.
- Budget/fidelity (incl. insured floor), checkpointing, memories (embedding-based), gated MCP —
  default ideation gates KEEP `research` (#4); native WebSearch/WebFetch coexist.
- Generic prompts with injection slots and the conditional knowledge-bank hook pointing at
  `INDEX.md` (#13).

Not benchmark names, not dataset assumptions, not domain playbooks (bug-provenance comments OK).

### 3.2 Handler contract v2 (`ProblemHandler`) — DECIDED (#8, #9)

| member | kind | consumer | source |
|---|---|---|---|
| `get_problem_context(budget_progress=0, **kw) -> str` | abstract | orchestrator (once, pre-loop) + workspace seeding | main |
| `maximize_scoring: bool` | attr | both strategies | main |
| `honor_agent_stop: bool = True` | attr | orchestrator | ioai |
| `deliverable_ready_reserve_seconds() -> float\|None` | hook | orchestrator insured finalization | shared |
| `finalize_run_selection(manifest, valid)` | hook | generic strategy at score-of-record | rb |
| `final_evaluate(file_path, **kw) -> dict` | hook | benchmark runner post-solve | main |
| `run(file_path, run_data_dir, solution=…) -> ProblemRunResult` | tree-path | benchmark_tree_search | main (now declared) |
| `stop_condition() -> bool` | tree-path | benchmark_tree_search | main (now declared) |
| `problem_id: str` | tree-path attr | tree export | main (now declared) |

Both strategies stay. `ProblemRunResult` loses its "deprecated" label. No registry.

### 3.3 Knowledge-bank contract — DECIDED (#13, #14, #15)

```
benchmarks/<b>/knowledge_bank/
  INDEX.md      # the single entry point agents read first — always this name
  ...           # everything else: benchmark-defined
```

- **`INDEX.md` is the only mandated name.** What it contains, how entries are laid out, and how
  the agent should route through the bank is defined by each benchmark's **handler context** —
  the handler tells the model what its bank is and how to use it.
- Injection path (unchanged, already built): config `knowledge_bank_dir` → runner stages a copy
  into the task dir → handler context names it as first search priority → the generic prompts'
  conditional bank-first hook. **At merge time the ioai prompt hooks are updated to reference
  `INDEX.md`** instead of the kaggle-specific `book_index.md`/`idea.md`/`solution/` layout —
  layout specifics move into the kaggle handler's context text.
- Kaggle migration: rename `book_index.md` → `INDEX.md`; its current curation ledger
  (`INDEX.md` today) becomes a handler-defined file (e.g. `CURATION.md`). Bank stays
  gitignored/shipped (licensing).
- RelBench: **no consolidation** (#14) — campaign concluded and results reported; `claims/`,
  `context.py` playbook constants, `data/knowledge/*.md`, and living documents stay exactly
  as they are.
- Versioning policy: **deferred** (#15) — status quo per benchmark.

### 3.4 Config layering — DECIDED (#10)

`load_mode_config()` deep-merges the benchmark's mode over the platform defaults in
`src/kapso/config.yaml` (nested dicts merge; scalars/lists replace). Benchmark configs shrink
to overrides-only — fixes the proven drift class (`request_timeout_seconds` reached relbench's
copy but never ioai's/posttrain's). Lands as its **own commit immediately after the main
merge** (~20 lines in `core/config.py` + tests + one-time trim of the benchmark configs).

### 3.5 Runners — DECIDED (#11)

Each benchmark keeps its own `runner.py`, bounded by one rule: **it must never change the main
framework.** Allowed: parse its harness, stage declared files, shape budgets from declared
knobs, construct handler + orchestrator, run the post-solve tail (harvest / `final_evaluate` /
consolidation). Forbidden: mutating platform semantics, re-implementing platform features,
carrying campaign policy that belongs in handler context. The four per-runner
`shape_session_timeouts` copies stay per-runner (declarative schema = backlog).

## 4. Reconciliation matrix (the merge work)

[union] = take both sides' features; [rb]/[ioai] = that tip wins; [knob] = becomes config.

| file | divergence | resolution |
|---|---|---|
| `environment/handlers/base.py` | ioai: `honor_agent_stop`; rb: `finalize_run_selection`; shared: reserve hook | **[union]** → §3.2 |
| `core/llm.py` | rb: `request_timeout_seconds` threading + tiktoken-capped embedding; ioai: web-search `reasoning_effort` strip | **[union]** — rb base + ioai's strip; add `tiktoken` dep |
| `researcher/researcher.py` | ioai-only fix (drop `reasoning_effort`/`max_tokens` on search calls) | **[ioai]** |
| `execution/orchestrator.py` | rb: transition freeze, bounded bridge deadline; ioai: advisory stop; shared: insured reserve | **[union]** |
| `search_strategies/generic/strategy.py` | rb-only: lens replanner, design axes, manifest-of-record, campaign-state brief, governance loop; ioai-only: `implementation_fallback_model`, `ideation_stream_path`, selector hardening (3666dfb6) | **[union]** — rb tip as base, graft the three ioai features; then the decided knobs: `implementation_web` (#3), restore `research` in default ideation gates (#4), optional time-split (#5). Feature-by-feature, not hunk-by-hunk |
| `search_strategies/generic/codex_ideation.py` | ioai: stream naming; both: web flag | **[union]** |
| `coding_agents/adapters/codex_agent.py` | ioai strips `OPENAI_API_KEY`, rb passes through | **[rb] + [knob]** (#7): passthrough default; `env_strip` for purity |
| `src/kapso/config.yaml` | rb adds `request_timeout_seconds`; both add `embedding` | **[union]** |
| `pyproject.toml` / `requirements.txt` | ioai-only pins `mcp>=1.9,<2`, `litellm==1.75.0` | **[ioai]** + add `tiktoken`, drop dead `weaviate-client` |
| `prompts/ideation_claude_code.md` | ioai: bank hook + native-web docs (replacing research_* docs); rb: coverage/eval-profile refinements | **[union]**, two-way — and per #4 keep research-tool docs alongside native-web docs (gates decide availability); bank hook → `INDEX.md` (#13) |
| `prompts/ideation_ensemble_addendum.md`, `ideation_selector.md`, `feedback_generator.md` | rb superset + small ioai additions | **[union]** (rb base) |
| `prompts/ideation_lens_planner.md` (+ rb-only `ideation_lens_replanner.md`) | 9/9 two-way | **[union]**; keep replanner |
| `prompts/coding_agent_implement.md` | ioai adds bank-first step 0 | **[union]**; bank hook → `INDEX.md` |
| `.claude/skills/kaggle-cli-submission/SKILL.md`, `.gitignore` | ioai newer / mechanical | **[ioai]** / union |
| 10 diverged tests | follow subject module | **[union]**, then green gate |

Identical-on-both-tips files (budget.py, claude_code_agent.py, oss adapter, agents.yaml,
experiment_workspace, memory store, shared_cache, presets, 3 prompts, 11 tests …) merge
themselves.

## 4b. The `strategy.py` decomposition — DECIDED (#12), in-merge

Analysis of the rb tip (2,943 lines; the union base) shows the monolith is already
seam-friendly: `run()` is a thin flow (validate short-circuit → `_select_parent` →
`_generate_solution` → `_expand_round`), `__init__` already delegates complex param parsing to
free `normalize_*` functions, and the code clusters into cohesive features with few cross-ties.

### Target layout (`search_strategies/generic/`)

| module | ~lines | contents (verified line spans on rb tip) | grafts landing here |
|---|---|---|---|
| `strategy.py` | ~800 | `GenericSearch` coordinator only: `__init__` (delegating to the modules' `normalize_*`), `_initialize_workspace`, `run()`, `_expand_round` + `_run_expansion_lane` thread glue, parent policy + `ParentSelection` + `_select_parent`, `_run_validate` / `run_bridge_evaluation` / `refresh_score_projections` (fidelity flow), accessors (history/best/deliverable/checkout), `dump_state`/`load_state`, budget/timeout/stream helpers (state-reading) | — |
| `ideation.py` | ~900 | ensemble constants + member normalization + degeneracy check (84–107, 368–419), `_generate_solution` (945–1085), `_generate_solution_ensemble` (1321–1565), selector: `parse_selected_solutions` + `_select_from_candidates` (1607–1729), `_campaign_state_brief`, ideation prompt build / extract / salvage / fallback (1730–1824) | selector hardening (3666dfb6), `ideation_stream_path` transcript naming, time-split knob (#5) |
| `lens_planning.py` | ~400 | all planner/replanner: constants + `DESIGN_AXES_DEFAULT` (225–366), `normalize_design_axes`, planner config normalize/validate, `parse_lens_plan`/`parse_lens_revision`, `_run_lens_planner_session`, `_resolve_member_lenses` (1086–1320), plan/history file IO, axes + roster briefs | — |
| `expansion_lanes.py` | ~200 | `MAX_NODE_EXPANSION`, `normalize/validate_node_expansion`, `render_lane_brief`, lane env-overlay computation (110–223), `pick_representative` (929–944, becomes a free function over nodes) | — |
| `implementation.py` | ~350 | completion markers, `_implement` body (1825–2063), `_build_implementation_prompt`, `_ensure_technical_difficulties` | `implementation_fallback_model` retry, `implementation_web` knob (#3) |
| `registered_evaluation.py` | ~420 | `_FRAME_RUN_KILL_GRACE_SECONDS`, `DEFAULT_EVALUATION_INSTRUCTIONS` + `_evaluation_instructions` (2370–2452), manifest-of-record trio + `_record_evaluation_attempt` (2092–2174), `_execute_registered_evaluation` (2175–2294), `_await_registered_evaluation` (2475–2550), `_sync_registered_evaluation` | — |
| `feedback_flow.py` | ~220 | `_generate_feedback` (2741–2840), `_extract_agent_result` + JSON fallback, the `finalize_run_selection` call site | — |

Import DAG (acyclic): `strategy` → all six; `ideation` → `lens_planning`, `codex_ideation`,
`shared_cache`; everything else → base/leaf only.

### Interface doctrine

- Extracted modules expose **stateless functions with explicit parameters** (workspace/session
  handles, config slices, nodes) — no new stateful collaborator classes invented mid-merge,
  no mixins.
- `GenericSearch` keeps a **thin delegating method for every session-running operation**
  (`_implement`, `_generate_solution`, …) that assembles arguments from its state and calls the
  module function. This is the coordinator's real API, and it keeps the existing tests'
  patch/spy seams stable.
- **No re-export shims** (Rule 7): tests and any importer move to the new module paths in the
  same commit. Embedded connective prompt templates move with their functions (migrating them
  to `prompts/*.md` is backlog, not merge scope).

### Sequencing (why unify-then-split)

1. **Unify in the monolith** (§4 grafts) — the 813-line tip residual maps onto the monolith's
   layout; grafting into freshly moved files would mean doing two transformations at once.
   Suite green.
2. **Mechanical split** — pure moves, behavior-identical, tests' imports updated in the same
   commits (one commit per extracted module keeps each diff verifiable as a move). Suite green
   again.
3. Behavioral-knob commits (#3/#4/#5) land **after** the split, in the features' final homes.

### Risks

- Checkpoint/resume: `dump_state`/`load_state` stay on `GenericSearch`; state is plain JSON —
  no import-path coupling. Verified-by-test via the existing checkpoint suite.
- Tests that patch strategy internals keep working through the delegating methods; tests that
  import free functions (`parse_lens_plan`, `render_lane_brief`, …) move imports.
- `feat/relbench-learning` (user-merged later, per #2) will hit a modify/delete on
  `strategy.py` if it touched that file — its changes get grafted into the new modules at that
  time. Accepted consequence of #12; noted so it isn't a surprise.

## 5. Benchmark-dir resolutions

| dir | owner | actions beyond "take owner's tip" |
|---|---|---|
| `benchmarks/relbench/` | rb tip | none (#14: no bank consolidation) |
| `benchmarks/kaggle/` + kaggle skill | ioai tip | bank naming migration (§3.3) when the bank next ships |
| `benchmarks/ioai/` | ioai tip | **delete the dead `contest_economics` handler generation** (#16) |
| `benchmarks/ioai_tasks/` | ioai tip | delete committed `.pyc`/`__pycache__` + gitignore |
| `benchmarks/posttrain/` | both (compatible) | union: rb's runtime-discipline dedup + ioai's R17–19 reviews + config fixes |
| `benchmarks/mle,ale` | main | untouched |
| stray `worktree-ioai2025` | — | cherry-pick `1a4b95ce`; remove worktree + branch |

## 6. Behavioral decisions — all DECIDED

1. **Web in implementation** (#3): `implementation_web` strategy param, default `true`;
   relbench mode sets `false` (temporal-leakage protocol).
2. **Ideation research gate** (#4): `research` restored to the platform default
   `ideation_gates`, native WebSearch/WebFetch also available; benchmarks not wanting the
   proxy trim gates in their own config (relbench/posttrain/kaggle modes already declare
   their own gate lists).
3. **Ensemble timing** (#5): full-clamp-per-role is the default; the member/selector
   time-split returns as an optional knob (absent = no split). Merge-gate step: re-check
   `ideation_timeout` in every benchmark mode under the default semantics.
4. **Stop authority** (#6): `honor_agent_stop = True` default; kaggle handler declares False.
5. **Codex child env** (#7): OPENAI_API_KEY passthrough default; benchmarks add it to
   `env_strip` if they want purity.

## 7. Merge mechanics — vehicle DECIDED (#1: integration branch)

1. Integration branch `unify/platform` from `feat/relbench-benchmark` tip.
2. `git merge worktree-ioai-2025` with ownership rules (§5); src conflicts left for step 3.
3. Hand-unification per §4 **in the monolith**, one commit per file/feature-cluster.
   Suite green.
4. **`strategy.py` decomposition** per §4b — one commit per extracted module, pure moves.
   Suite green.
5. Behavioral-knob commits per §6 (items 1–3 are code+config, landing in the new module
   homes; 4–5 ride the union).
6. Hygiene commit (#17): `.pyc` purge + gitignore, dep pins (+`tiktoken`, −`weaviate-client`),
   `contest_economics` deletion (#16), stale-copy sanity check (`benchmarks/kaggle` == ioai
   tip, `benchmarks/relbench` == rb tip).
7. **Gate**: curated hermetic suite + union of both branches' suites green; one smoke run per
   strategy path (relbench FAST_DEBUG-style + kaggle dry runner invocation).
8. Cherry-pick `1a4b95ce`; delete `worktree-ioai2025` worktree + branch.
9. Merge `unify/platform` → `main`, push (Rule 11).
10. Config deep-merge (§3.4) lands right after on main as its own commit (#10).

`feat/relbench-learning`: set aside for now (#2) — user merges it themselves later.

## 8. Post-merge backlog

- Declarative session-budget schema (retire the four `shape_session_timeouts` copies) —
  needs its own design.
- Dynamic handler context (per-iteration re-fetch with real `budget_progress`).
- Open `Kapso.evolve()` to custom handlers.
- Sanctioned env surface doc (session-contract env = IPC, allowed; config-via-env banned;
  sweep `KAPSO_PROMPTS_DIR`, gate URL vars).
- De-scent generic prompt examples.
- Retire legacy config keys (`knowledge_retriever`, `use_knowledge_graph`, `developer_model`);
  fold vestigial `environment/` layout.
- Deferred by user: bank versioning policy (#15), relbench-learning interplay (#2).

## 9. Open decisions

None — all seventeen decided (§0). The plan is executable as written.

## 10. Execution plan — orchestrator + one executor per stage

Model: **one orchestrator** (the main session) owns branch state, runs every gate itself
(never trusts a stage's self-report), reviews each stage's diff, and commits checkpoints.
**One executor per heavy stage** (a dedicated agent with a tight charter = the relevant §§ of
this doc); mechanical stages are orchestrator-inline — spawning an agent to run `git merge`
or `rm` adds risk, not value. Stages are strictly sequential (same working tree); the branch's
commit history is the handoff artifact between stages.

Working tree: the relbench worktree (campaign concluded, #14) — the orchestrator checks out
`unify/platform` there. All work lands as commits on that branch.

| stage | executor | work | gate (run by orchestrator) |
|---|---|---|---|
| **S0 prep** | orchestrator | commit this doc on the rb branch; cut + checkout `unify/platform`; run the curated hermetic suite + both campaigns' suites on the untouched tip and **record the baseline** (later gates compare against it, not against an assumed all-green) | baseline recorded |
| **S1 mechanical merge** | orchestrator | `git merge worktree-ioai-2025`; resolve by rule: benchmark dirs per §5 owner; `.gitignore` = union **+ fold in `1a4b95ce`'s rule here** (kills the later cherry-pick step); `[ioai]` matrix rows take theirs; the graft-target src files (llm.py, strategy.py, codex_agent.py) + their 3 conflicted tests take the rb base; single merge commit | imports clean; suite = baseline **except an enumerated expected-red list** (auto-merged ioai tests exercising not-yet-grafted features, e.g. test_llm_routing_retry's strip cases) — list recorded, to be cleared by S2 |
| **S2 grafts** | agent "grafts" | hand-unification per §4, in the monolith, one commit per cluster: (a) llm.py web-search strip + config union; (b) base.py + orchestrator.py union (`honor_agent_stop`, advisory stop); (c) strategy.py grafts — fallback-model retry (d3294a46), selector hardening (3666dfb6), transcript naming; (d) codex_ideation stream naming; (e) prompts two-way union — keep research-tool docs AND native-web docs (gates decide availability), bank hook → `INDEX.md` (#13); (f) test grafts (ioai's feature tests land with their features) | **full suite green, expected-red list empty**; orchestrator spot-reviews each cluster diff against the matrix row |
| **S3 decomposition** | agent "split" | §4b: six extraction commits + final `strategy.py` trim; pure moves, stateless-function interface doctrine, test imports updated in-commit | suite green **after every commit** (orchestrator runs it per commit; any red = redo that move); `git diff` sanity: net behavior delta of the whole stage ≈ 0 (moves only) |
| **S4 knobs** | agent "knobs" | #3 `implementation_web` (default true; relbench mode false), #4 restore `research` to default ideation gates (opt-out configs where decided), #5 optional time-split knob (absent = full clamp), + `ideation_timeout` re-check in every benchmark mode (§6.3), + tests for each knob | suite green; `load_mode_config` loads every benchmark mode; knob defaults verified by test |
| **S5 hygiene** | orchestrator | #16 delete `contest_economics` generation; #17: `.pyc` purge + gitignore, `+tiktoken`, `−weaviate-client`; stale-copy sanity (`benchmarks/kaggle` ≡ ioai tip, `benchmarks/relbench` ≡ rb tip, mod deliberate §5 actions) | suite green; `pip install -e .` clean in a fresh venv (dep changes proven) |
| **S6 final gate** | orchestrator | curated suite + union of both campaigns' suites; **no-spend smokes**: for each benchmark mode, construct handler + strategy + orchestrator from its real config (no LLM calls); kaggle runner dry invocation; relbench FAST_DEBUG-style construct path | everything green — this is the ship/no-ship evidence |
| **S6.5 config deep-merge** | agent "config" | #10 as the **last commit on the integration branch**: deep-merge in `load_mode_config` + tests + trim benchmark configs to overrides-only (decision said "immediately after the main merge"; landing it as the final pre-merge commit is the same isolation without needing a second main event — flagged as a deliberate simplification) | suite green; every benchmark mode loads and resolves identical effective config as pre-trim (snapshot-compared) |
| **S7 ship** | orchestrator + **user checkpoint** | merge `unify/platform` → `main` (--no-ff), push (Rule 11); delete `worktree-ioai2025` worktree + branch | user's explicit go precedes this stage; post-push: `origin/main` == local, worktree list clean |

Mechanics notes:
- **Main-checkout constraint**: this session's git is worktree-pinned; S7's merge-to-main and
  worktree deletion run from `/home/ubuntu/kapso`. Mechanism at ship time: exit worktree
  isolation if the harness allows, else hand the user the three exact commands — either way
  S7 is already user-gated.
- **Rollback points**: every stage is one-or-more atomic commits on `unify/platform`; a failed
  stage is `git reset --hard` to the previous gate tag (orchestrator tags each passed gate:
  `unify-gate-s1` …). Main is untouched until S7.
- **Agent discipline**: stage agents work in this worktree directly (no isolation — state must
  accumulate), never in parallel, never touch branch operations (commits are fine, merges/
  checkouts are the orchestrator's), and receive this doc + their stage row as the charter.
- The relbench worktree's branch is restored to `feat/relbench-benchmark` after S7 (or left on
  `unify/platform` if the user prefers — it's merged either way).

## 11. Execution log (2026-08-20)

| stage | commits | gate |
|---|---|---|
| S0 prep | doc `2f3d9587`; branch cut | baseline 431 green |
| S1 mechanical merge | `923cbdc3` + test-contract fix `487d7c4f` | 454 green; ownership identities verified; expected-red list empty |
| S2 grafts | `730af492` (llm web-search strip), `89d0762e` (strategy: fallback retry, selector hardening, transcript naming, +2 found features), `157380ea` (INDEX.md convention + prompt-union repair) | 456 green |
| S3 decomposition | `89f599ba` `ab31db02` `35868dac` `96900a78` `bc3951ce` `d50f756e` — six pure moves; strategy.py 3203→1362; AST symbol-purity audit clean | 456 green after every commit |
| S4 knobs | `59499ef9` (implementation_web, both CLI vectors), `532f32dc` (research-gate default pin), `609e9d6c` (ensemble_time_split) | 475 green; per-mode effective timing unchanged |
| S5 hygiene | `4fc3d290` (#16 contest_economics retired), `8087c864` (#17: .pyc purge, +tiktoken, dead key; weaviate-client kept — KG search imports it) | 475 green; fresh-venv dep resolution clean |
| S6 final gate | — | 475 green + construct smokes: all six benchmark modes build from real configs with correct knob values (GENERIC/MINIMAL env-blocked on absent bedrock creds — pre-existing) |
| S6.5 config layering | `d59f6033` (deep-merge mechanism + defaults layer), `a98efb6e` (pure-dedup trims) | 487 green; 18-mode resolution diff = exactly the two intended gap-fills (request_timeout_seconds, models.embedding), zero defects, zero runner repoints |
| S7 ship | this commit; `git push origin unify/platform:main` (fast-forward) | — |

Deviation from §7 as written: the merge-to-main landed as a **fast-forward push** rather than a
`--no-ff` merge commit — the session's git is worktree-pinned away from the primary checkout,
and the branch strictly descends from `origin/main`, so the pushed tree is identical and the
history linear. Post-ship manual step (needs the primary checkout): remove the stray
`ioai2025` worktree and its one-commit branch — its `.gitignore` rule was folded into the S1
merge, so nothing there is unique.

# Integrating learn-from-trajectories into main — design, tutorial, simulations

**Status:** DESIGN for the user-executed merge (standing rule: the user
merges `feat/relbench-learning`; this doc is the map). Evidence-based
against `origin/main @ e2a8335b` and `feat/relbench-learning @ 1cc5d993`,
merge-base `d1a7da94` (branch: 131 commits / +19,784 lines across 203
files; main: 187 commits including the platform unification).

---

## 1. State of the two lines (measured, not remembered)

**Main since the base** (the platform unification, merged 2026-08-20):
- The generic strategy was split: `strategy.py` shrank to a 1,395-line
  orchestrator; sessions moved to extracted modules with EXPLICIT
  parameters — `ideation.py` (`generate_solution`,
  `generate_solution_ensemble`), `implementation.py`,
  `lens_planning.py` (`run_lens_planner_session` is now a free function
  taking keyword args), `feedback_flow.py`, `expansion_lanes.py`,
  `registered_evaluation.py`.
- Config became layered: the platform config is `default_mode` +
  `defaults:` + `modes:`, with `load_mode_config()` deep-merging a
  benchmark config over the platform `defaults:` layer
  (`src/kapso/core/config.py`).
- `benchmarks/relbench/` was slimmed (scorecard/evidence/data payloads
  deleted; ~203k lines removed); `handler.py` barely moved (2 lines);
  `runner.py` has NO serving block; `campaign.py` was refactored
  (−40/+33).
- `cli.py` untouched on main — it still carries the OLD `kapso learn`
  (knowledge-source/KG learning).
- Benchmarks on main: `ale`, `ioai2026`, `mle`, `posttrain`, `relbench`.

**The branch since the base** — three strata:
1. **The learning platform (all-new, conflict-free):**
   `src/kapso/learning/` (41 files — bank + invariants, retriever v2,
   serving_launch, trajectory_store, mining, graders/ (exam frame +
   crews), update_frame + update crews, codify_run + codify_gates +
   gcp_ephemeral, harvest, ab, behavior, corpus_import, develop, refs),
   `gated_mcp/gates/bank_gate.py` + its preset entry, ~20 test files,
   the design docs, and the trajectory-learning CLI
   (`kapso learn import|mine|grade|update|init-bank|develop|codify|
   ingest|gauntlet|behave`).
2. **Evolve-side serving touchpoints (the port surface):** edits to
   `strategy.py` (bank_serving param, ideation MCP mount, lens-planner
   bank mount), three prompt files, `feedback_generator.py`
   (`cards_load_bearing`), `budget.py` (`checkpoint_heartbeat_seconds`).
3. **relbench benchmark wiring (the reference adopter):** `runner.py`
   serving block, `context.py` `knowledge_section` (additive intro),
   `handler.py` `apply_bank_intro`, `campaign.py` `_harvest_trajectory`
   (+60 lines: the strict-contract campaign→store bridge with
   `bank_head` stamping), `config.yaml` learning models.

**The real conflict surface** (from `git merge-tree origin/main HEAD` —
a dry run, not a merge): exactly **four content conflicts** —
`.gitignore`, `benchmarks/relbench/campaign.py`,
`prompts/ideation_claude_code.md`, `strategy.py`. Everything else
auto-merges, including all of `src/kapso/learning/`, the bank gate, the
CLI, and the tests. The dominant risk is therefore not textual conflict
but **semantic drift in auto-merged files** — enumerated in §2.3.

## 2. Integration design

### 2.1 The one structural decision: port, don't resolve

`strategy.py` is ours-monolith vs main-modular. Resolving that conflict
line-by-line would resurrect the monolith. The design is: **take main's
file layout wholesale, then re-home our three serving edits into the
extracted modules** — they become parameter threads, which is exactly
the shape main's split wants:

| Our edit (branch location) | Main home | Port shape |
|---|---|---|
| `self.bank_serving = self.params.get("bank_serving")` (strategy init) | `strategy.py` init | identical line; thread into the three calls below |
| ideation `get_mcp_config(..., bank_serving=...)` | `ideation.py:generate_solution` / `generate_solution_ensemble` | add `bank_serving` keyword param, pass to `get_mcp_config` |
| implementation gate mount (bank tools reach implementor sessions) | `implementation.py` | same `bank_serving` param → `get_mcp_config` |
| lens-planner bank mount (gates=["bank"], mcp tools into allowed_tools) | `lens_planning.py:run_lens_planner_session` | add `bank_serving` keyword param; mount block inside (the function already takes explicit session kwargs — one more fits the idiom) |

The prompt conflicts are additive on both sides (main reworked
surrounding guidance; we added the knowledge-bank blocks): resolution is
*take both* — main's text with our `## Knowledge bank (when served)` /
`## Ground the portfolio…` blocks re-inserted verbatim. `.gitignore` is
a trivial union (our `/learning/` lines append).

`campaign.py`: main refactored the file; our delta is one additive
function (`_harvest_trajectory`) plus its call site. Port the function
onto main's refactored shape — it touches only harvest, not the
refactored paths.

### 2.2 Config placement (the layered-config decision)

Decision: **`learning:` becomes a top-level section of the platform
config, sibling to `defaults:`** — not inside it. Rationale, simulated
in §6: the learning pipeline is platform infrastructure that runs
OUTSIDE any mode (CLI-driven: mine/grade/update), so burying it in the
per-mode merge layer would imply per-mode learning configs that nothing
consumes. Two knobs do get per-benchmark character and are handled
explicitly:
- `learning.serving.enabled` — stays the global off-switch; a benchmark
  opts in operationally (relbench's arm setup flips it), and a future
  per-benchmark override can ride `load_mode_config`'s deep-merge if a
  benchmark config carries a `learning:` fragment — the defaults layer
  already gives us that mechanism for free; adopt it only when a second
  benchmark actually wants a different value (Rule 10).
- `learning.bank.local_path` / `bank.remote` — deployment-owned (which
  bank this installation serves); one bank per benchmark family is the
  operating assumption (scope eligibility separates families inside a
  bank, but evidence pricing and A/B certification are per-benchmark
  economies — mixing them buys nothing yet).

Also merging into the platform config: the eight learning model/effort
blocks, `graders.*`, `update_crew.*`, `retriever.probe_budget`,
`codify.*`, and `budget.checkpoint_heartbeat_seconds` (already
auto-merging into `budget:`).

### 2.3 Semantic-drift checklist for auto-merged files

Files that merge clean but MUST be re-verified because both sides moved
meaningfully:
1. `feedback_generator.py` — main reworked it (15-line diff);
   `cards_load_bearing` (ours) must survive the parse/`to_dict` path.
   Covered by `test_serving_wiring.py::test_judge_parses_cards_load_bearing`.
2. `tests/test_lens_planner.py` — our new
   `test_planner_session_mounts_bank_gate_when_serving` drives the
   session through the strategy stub; on main the session is the free
   function in `lens_planning.py`. The test gets rewritten against the
   function signature (call `run_lens_planner_session` directly with
   `bank_serving=`). Same class of fix as the stub-gotcha memory: main's
   `make_stub` mirrors a different `__init__`.
3. `ideation_lens_planner.md` auto-merges, but main edited adjacent
   text (12-line diff) — read the merged file once; the contract block
   is self-contained and position-independent.
4. `handler.py`/`context.py`/`runner.py` (relbench) auto-merge; main's
   runner never had serving, so our serving block lands whole — verify
   `prepare_campaign_serving` import path and the `--knowledge-file`
   flag (main-side addition) coexist in the assembled context: the
   extra-knowledge section and the bank intro are separate sections and
   compose.
5. `cli.py` — auto-merge takes our rewrite; SEMANTIC decision made
   explicit: the old `kapso learn` (knowledge-source/KG ingestion) is
   SUPERSEDED by trajectory learning under the same verb (Rule 7 — the
   KG path keeps its `index-kg` entry; nothing else consumed the old
   command). State this in the merge commit message.

### 2.4 What does NOT change at merge time

The bank repo (`kapso-bank-relbench @ 17dbb09`), trajectory stores, and
all on-disk learning state are branch-agnostic — they are data, keyed by
config paths. No migration. The gitignored `learning/` run-dir home
stays local-only.

## 3. The benchmark adoption contract (what it takes to join learning)

Simulation §6-U2 forced this into a crisp three-hook contract. A
benchmark adopts learning by providing:

**Hook 1 — the harvest bridge** (campaign → trajectory store). After a
campaign completes, call `TrajectoryStore.ingest_campaign(...)`-style
bridging the way `benchmarks/relbench/campaign.py::_harvest_trajectory`
does: bundle the campaign workspace (campaign.log, runs/, living docs,
`.kapso/` including `serving/`), write `trajectory.yaml` (id =
`<dataset>--<task>/<stamp>_<lane>`, family, dataset, inventory sha256),
stamp `bank_head` from the serving record when present. The strict
contract is the store's, not the benchmark's; the benchmark only maps
its artifact layout onto it. (~60 lines; relbench is the reference.)

**Hook 2 — serving injection** (3 lines in the runner + 1 context slot):
```python
serving = prepare_campaign_serving(learning_config, task_coords, work_dir)
if serving:
    handler.apply_bank_intro(serving["intro"])
# and thread params["bank_serving"] = serving["bank_serving"]
```
plus a knowledge slot in the benchmark's problem-context builder that
appends the intro after whatever static notes the benchmark carries
(relbench: `knowledge_section` in `context.py`). The intro is
self-contained; the benchmark chooses only WHERE it sits.

**Hook 3 — task coordinates**: a `{family, dataset}` mapping for scope
gaps, probe eligibility, and exam bookkeeping. Any stable vocabulary
works; it must simply be consistent between serving and harvest.

Everything else — mining, exams, lessons, bank transactions, codify,
A/B — is benchmark-agnostic CLI machinery that reads the store.

## 4. End-to-end tutorial (post-merge main)

The full loop on a fresh machine, relbench as the worked example:

```bash
# 0. One-time: found a bank (or point config at an existing remote)
kapso learn init-bank            # creates learning.bank.local_path home
# config: learning.bank.local_path: ~/kapso-bank-relbench
#         learning.bank.remote: git@github.com:<org>/kapso-bank-<bench>.git
#         learning.serving.enabled: true

# 1. Run a campaign — serving happens at launch automatically
python -m benchmarks.relbench.runner --dataset rel-amazon --task user-churn \
    --strategy generic --time-budget-hours 6
#    → "Knowledge bank: serving at head <sha>"; sessions get bank_index /
#      bank_get_card / bank_get_card_with_evidence; lens plan carries
#      bank: declarations; serving record + pull log land in .kapso/serving/

# 2. Harvest the finished campaign into the trajectory store
#    (relbench does this in campaign.py automatically at completion)

# 3. Learn from it: mine → exam-before-lesson → lesson → bank push
kapso learn mine  <trajectory-id>
kapso learn ingest <trajectory-id>      # exam (hindcast) then update crew;
                                        # bank commit + push on admit

# 4. Next campaign is served the updated bank — the loop is closed.

# 5. Certify a generation before trusting it (config-waivable gate)
#    A/B arms: candidate (serving on) vs incumbent (notes only), same
#    commit/budget; verdict via kapso.learning.ab.ab_verdict over pairs.

# 6. When a procedure card's recurrence earns it, flip it to code
kapso learn codify <card-name>          # gcp_ephemeral replay → green
                                        # verdict → code/ + replay/ flip
```

What each stage writes, where to look when it misbehaves: serving —
`.kapso/serving/{serving-record.yaml,serving-pull.jsonl}` in the
campaign workspace; exams — `learning/graders/…/report.md`; lessons —
the bank repo's commit log (one tagged commit per lesson); behavior
suite — `kapso learn behave` runs the semantic scenarios against real
machinery.

## 5. Merge-day runbook (for the user; ~2-3 hours incl. gates)

1. `git checkout -b merge/learning main && git merge feat/relbench-learning`
2. Resolve the four conflicts:
   - `.gitignore`: union.
   - `ideation_claude_code.md`: main's text + our `## Knowledge bank`
     block.
   - `campaign.py`: main's refactor + our `_harvest_trajectory` and its
     call site.
   - `strategy.py`: **take main's version**, then apply §2.1's port map
     (bank_serving param + three threads).
3. Apply the port map to the extracted modules
   (`ideation.py`, `implementation.py`, `lens_planning.py`) and rewrite
   `test_planner_session_mounts_bank_gate_when_serving` against
   `run_lens_planner_session`.
4. Walk §2.3's semantic checklist (five files, ~20 minutes of reading).
5. Gates: the curated hermetic list + the learning suites
   (`test_bank_retriever test_bank_pull test_serving_wiring
   test_behavior_runner test_update_frame test_graders_* 
   test_gauntlet_runner test_trajectory_store test_relbench_integration`)
   — all hermetic, no infra.
6. Live smoke (30 min): `kapso learn behave` (serve + exam scenarios)
   plus one short served relbench campaign asserting the serving record
   appears and `bank_index` events log.
7. Merge to main; the branch retires.

## 6. Use-case simulations (each ran against the design; refinements folded back)

**U1 — Continuing relbench operation.** Post-merge, the bank remote and
store paths are config values; nothing moves. Serving records reference
`bank_head` shas that exist in the bank repo (including
`archive/pre-card-v2` for pre-rewrite heads) — provenance survives.
*Found & fixed in design:* the runner loads the platform config for
learning (`DEFAULT_CONFIG_PATH`) — under main's layered config this MUST
go through `load_config` of the platform file, never a mode fragment;
§2.2's top-level placement makes that the natural read.

**U2 — A second benchmark adopts (ioai2026).** Walking the three hooks:
ioai's context builder gains a knowledge slot (hook 2 is 3+1 lines);
its harvest maps contest tasks to `{family: <ioai-family>, dataset:
<contest>}`; a NEW bank repo is founded (`kapso-bank-ioai`) per §2.2's
one-bank-per-family rule. *Found & fixed:* the adoption cost was
originally scattered across five files on relbench; the contract in §3
exists because this simulation showed hooks 1/2/3 are the ONLY
essential surface — everything else came along via the CLI.

**U3 — Fresh user, no bank.** `learning.serving.enabled: false` +
missing bank home must be a clean no-op — and it is
(`prepare_campaign_serving` returns None when disabled; raises loudly
when enabled-but-missing, which is the correct fail-loud shape).
*Refinement adopted:* the tutorial leads with `init-bank`, and the
enabled-but-missing error message names the config key to flip.

**U4 — CI on main.** All learning tests are provider-mocked and
GCP-free; codify's `gcp_ephemeral` is exercised only by its driver
tests. The curated gate grows by the learning suites (~90 s measured on
the branch). No infra dependency joins main's CI.

**U5 — Codify on main.** Needs GCP project/zone/image config
(`learning.codify.gcp.*`) and user credentials via gcloud — config-file
driven per Rule 3, no env reads. Unchanged by the merge; documented in
the tutorial as the one stage with cloud cost.

**U6 — Multiple benchmarks, one machine.** Two banks, two stores, one
platform config — collision point is `learning.*` being global.
Simulated outcome: acceptable for now (operators run one benchmark per
config file; `--config` already selects), and the per-benchmark
`learning:` fragment via `load_mode_config` deep-merge is the reserved
escape hatch when a real dual-bench deployment arrives. Deliberately
NOT built now (Rule 10).

**U7 — The merge itself, dry-run.** `git merge-tree` on the real heads
produced the four conflicts of §1 and nothing else; the port map in
§2.1 was written against main's actual extracted signatures (verified:
`generate_solution` takes `ideation_gates`, `run_lens_planner_session`
takes keyword session params — both accept one more keyword cleanly).

## 7. Open items the merge does NOT settle (tracked, not blocking)

- Mining/ingesting the A/B arm trajectories (3× replicated replay
  boundary still absent from the bank).
- `select_final` last-clean-head fallback + late-budget evaluator-
  transition freeze (framework-core, user-gated).
- `cards_load_bearing` fix-or-delete; ambition-calibration cue; dataset
  dossier card species — all live in serving-agentic-redesign.md §9 and
  the ledger notes.

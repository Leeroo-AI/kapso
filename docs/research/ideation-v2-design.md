# Ideation v2 — measured novelty and explore/exploit scaffolding

## 1. Research digest

Only findings that shaped this design:

- **Operators and candidate spread beat search policy at small N.** AIRA (Meta 2025, MLE-bench scaffold ablations) found search-policy classes (greedy/MCTS/evolutionary) statistically indistinguishable until ~19h of 24, operators carrying the gains, UCT constants insensitive; SELA's wins traced to enumerated diversity, not lookahead. Consequence: no new policy layer — effort goes into candidate generation, dedup, and persistence.
- **Explore-first / gamble-last is optimal at a short known horizon.** One-armed-bandit structure (Bradt–Johnson–Karlin 1956) front-loads exploration; on the final pull with an incumbent banked, the deliverable is `max(incumbent, X)` — convex in X, so expected-improvement logic (Jones et al.) bets the highest-variance plausible candidate. run17 improvised exactly this and set the cell record; v2 makes it mechanism.
- **LLM-judged novelty is decorative; embedding dedup is reliable.** Si, Yang & Hashimoto 2024: ~95% of same-model idea pairs are duplicates at cosine ≈0.80, while LLM novelty rankers score near chance; Beel et al. 2025 found AI Scientist's literature-novelty gate marked 12/12 known techniques novel; the ideation–execution gap study (arXiv 2506.20803) shows pre-execution novelty doesn't predict executed quality. Scoping law adopted verbatim: **embeddings answer "seen before?", never "worth doing?"**.
- **The transferable population-search lessons are the archive and the behavior descriptor.** FunSearch/AlphaEvolve/MAP-Elites guarantees are asymptotic (~2.5M candidates); what survives at N≤15 is keeping losing candidates (cheapest diversity reserve — also R&D-Agent's one load-bearing asset, cross-trace memory) and making uncovered behavior visible. Kapso's task-defined Coverage axes already are a behavior space; no framework taxonomy needed.
- **Evidence-targeted refinement beats blind iteration.** MLE-STAR directs the next change at a measured weakness. AIDE's dedup is a prompt plea, and its draft quotas/`debug_prob` assume 20+ cheap steps this regime lacks.

## 2. Diagnosis

Current kapso ideation vs that evidence — five failure modes, each with a campaign incident:

1. **Diversity collapse is invisible.** run16: 8 candidates across 2 iterations, all static-corpus SFT variants. Fixed config lens strings were the only diversity mechanism and demonstrably did nothing; embeddings shipped but nothing reads them at ideation time.
2. **Exploiting a wrong diagnosis.** run16 iter2 refined along a feedback claim the stored per-node score trace contradicted (score monotonic in the axis feedback said to reverse). The trace was already served by `get_top_experiments`; nothing forced anyone to look at it at selection time.
3. **Measured gaps are prose, never claimed.** run19: the zh/ru/ja slice was the #1 documented lever, marked uncovered from iteration 1, and never targeted — the selector's fixed tie-break ("prefer expected improvement over the most novel idea", `ideation_selector.md:26-27`) deferred it every time.
4. **Losing candidates evaporate.** Only the selected `<solution>` survives (strategy.py ~825-834). run13's strong reward-model candidate died with the VM; run16 spent ~1.5h re-deriving overlapping conclusions.
5. **Explore/exploit is advisory prose.** One budget-keyed sentence; run17's winning bank-then-gamble shape was improvised by the agent, not produced by the framework.

## 3. Design

**Chassis: zero new layers.** The strategy already receives `budget_snapshot` and the fidelity decision before `run()` (base.py:615-627), so everything lives in the generic strategy, its prompts, the store record, and the gate render. No new module, observe hook, orchestrator wiring, or state file. Parent selection stays greedy-best untouched — `_select_parent` including its committed-but-unevaluated recovery branch (strategy.py:1704-1721) is campaign-validated (run17) and AIRA-consistent.

**Signal ledger** (every decision rule consumes the Measured column):

| Signal | Nature | Source |
|---|---|---|
| Scores, `evaluation_valid`, `duration_seconds` | Measured | `SearchNode` history |
| Remaining full runs | Measured | `budget_snapshot.remaining_after_reserve ÷ mean measured duration of valid executed nodes` — the same layer-1 arithmetic fidelity prices runs with, extracted as one pure helper `measured_full_run_price(node_history)` so fidelity (when enabled) and ideation read one encoding, never a fourth |
| Open measured gaps | Measured (regex over structured Coverage) | executed records |
| Duplicate / novelty cosines | Measured | shipped text-embedding-3-small |
| Coverage MEASURED/ASSUMED, TESTED labels | LLM-declared, implementor-recon-verified (existing contract), selector-audited | candidates |
| Candidate plausibility, pick, synthesis, Diagnosis audit | LLM-judged (selector's proven strength, run17) | selector |
| Idea "worth" / novelty rank | **Nobody — deliberately unscored** | — |

**Stance rule** — a pure function inside `run()`, replacing the advisory prose (deleted, Rule 7):

```python
@dataclass(frozen=True)
class IdeationStance:
    stance: str          # "EXPLORE" | "EXPLOIT" | "FINAL_GAMBLE"
    gap_axis: str | None # top open measured gap, orthogonal to stance
    reason: str          # audit string, FidelityDecision-style

def compute_ideation_stance(node_history, budget_snapshot, open_measured_gaps):
    top_gap = oldest_flagged(open_measured_gaps)        # deterministic ordering; None if empty
    scored = [n for n in node_history if n.evaluation_valid and n.score is not None]
    if not scored:                                      # first pull; also: no measured price yet
        return IdeationStance("EXPLORE", top_gap, "no valid scored node: breadth is cheapest now")
    price = measured_full_run_price(scored)             # mean duration_seconds, always defined here
    remaining = floor(budget_snapshot.remaining_after_reserve / price)   # includes this iteration
    if remaining <= 1:
        # any valid scored node is already persisted and ships regardless (deliverable = best
        # node), so the incumbent is banked by construction: E[max(incumbent, X)] is convex.
        return IdeationStance("FINAL_GAMBLE", top_gap, f"last full run at measured {price:.0f}s")
    return IdeationStance("EXPLOIT", top_gap, f"{remaining} full runs remain: refine best")
```

Deterministic, reason-stamped, direction-agnostic (no score-sign logic), and ordered so the unpriced case hits the early return. Stance semantics rendered into prompts: **EXPLORE** — members land candidates in structurally distinct approach classes (duplicate alarm enforces mechanically); **EXPLOIT** — one primary attributable lever plus at most one reversible gated add-on (the run17 shape); **FINAL_GAMBLE** — selector tie-break flips to "highest-variance plausible candidate, ties toward measured novelty".

**Gap machinery.** Coverage tightens to one axis per line: `- <axis>: MEASURED|ASSUMED; TESTED|UNTESTED — <citation>`, with axis strings copied **verbatim** from the eval profile or history renders (one prompt sentence; closure matches exactly over verbatim strings). `_extract_open_measured_gaps(node_history)`: axes marked `MEASURED; UNTESTED` in executed records minus axes any executed record marks `TESTED`. Whenever `gap_axis` is set, **one candidate slot** (one designated member's candidate 2) must target it or refute its measurement citing eval data — B's slot granularity, orthogonal to stance: the gap can never again be deferred by prose preference, yet never hijacks the iteration or displaces an exploit candidate (run19's DPO would have coexisted in the pool with the zh/ru/ja candidate, selector arbitrating as in run17).

**Novelty, quantified.** `novelty(c) = 1 − max cosine(e_c, siblings ∪ stored solutions ∪ stored unselected candidates)`. At pool hygiene (S:772-810): embed candidates; report per-candidate sibling-max and history-max cosines with nearest-neighbor record ids; pairs ≥ `ideation.duplicate_cosine_threshold` (config, 0.80 — Si et al. operating point, the **only new knob**) produce a facts block in the selector prompt (`DUPLICATE ALARM: cand-1 ~ cand-3 (cos .91)`). The selector may not pick a flagged candidate without stating why the signal is wrong. Structural checks (malformed Coverage) reject per-candidate **here** — never at persist time after a GPU burn — and flag-not-drop whenever rejection would take the pool below 2 candidates: dedup polish never costs delivery. Novelty's only other uses: EXPLORE/FINAL_GAMBLE tie-break. Never a worth score.

**Ensemble and selector.** Both kept — 2/2 delivery all campaign; the ensemble is the draft-breadth engine (AIDE's quota relocated to where drafts cost minutes). Fixed lens strings are **deleted** (run16 proved them inert) and replaced by stance-computed **per-member** operator briefs in the same render slot (S:646-652): member briefs stay differentiated (e.g., on EXPLOIT: member 1 refines the cited lever, member 2 maximizes embedding distance from executed history) so a single shared brief never homogenizes the pool — critical for the codex member, which has no MCP tools and lives on pushed facts. The selector keeps its verification/synthesis role and gains: the stance-conditional tie-break (replacing the fixed line that deferred run19), the duplicate facts block, and a required **Diagnosis audit** section — quote every feedback causal claim the chosen candidate relies on and cite the stored per-node scores confirming it; if the trace contradicts the claim, pick a candidate not resting on it. LLM-judged but forced-citation-grounded, placed in the one session with demonstrated fact-checking behavior.

**Persistence (new).** `SearchNode` gains `unselected_candidates: tuple[str, ...]`, `selector_reasoning: str`, and the stance + reason string; these ride the existing orchestrator→store path (orchestrator.py:1324-1326) onto `ExperimentRecord` — fields on the executed node's record, not standalone records, so `node_id` keying, `get_recent` renders, and store coupling are untouched. `add_experiment` embeds each unselected candidate alongside the solution; `search_similar` ranks by max cosine over all of a record's embeddings; the gate render appends "## Unselected candidates (NOT EXECUTED)" and the selector reasoning, full text (Rule 6).

**Per-iteration sequence:** compute stance → render stance paragraph + per-member briefs + gap slot → members produce 2+2 candidates → pool hygiene (structural check, embeddings, cosine facts) → selector (stance tie-break, alarm rule, Diagnosis audit) → persist selected solution plus full pool, reasoning, stance.

## 4. Why this fits the regime

Policy cost is one pure function plus ~6 embedding calls — noise against 2–5h iterations; every mechanism pays off within 1–2 iterations (nothing requires history to accumulate). PostTrainBench (`remaining` ∈ {1,2}): EXPLORE → EXPLOIT/FINAL_GAMBLE with the gap slot binding whenever a measured gap is open — run17's record structure, now policy. RelBench (4–15): a long EXPLOIT middle, gap slots firing as evals surface new MEASURED axes, the duplicate alarm preventing re-proposal drift over 15 iterations, one terminal gamble. Same code path; `remaining` is the only differing input. Counterfactuals:

- **run16-iter2 (wrong-diagnosis exploit):** the Diagnosis audit forces the selector to quote the "reverse this axis" claim against the served monotonic score trace; contradiction → the refinement candidate resting on it is barred. Independently, the sibling+history cosine table flags the eight-SFT-variant collapse mechanically instead of trusting lens strings.
- **run19 (language axis):** zh/ru/ja is a `MEASURED; UNTESTED` line from iteration 1, so `open_measured_gaps` is nonempty and the slot binds one candidate at iteration 2; the stance-conditional tie-break removes the fixed prose that deferred it, and at FINAL_GAMBLE ties go to the high-variance gap-targeting candidate — while the +16.5 DPO exploit stays in the pool rather than being displaced by a whole-iteration pivot.
- **run13 (lost candidate):** the reward-model loser persists on the node record with its embedding and selector verdict, served full-text and labeled NOT EXECUTED by history renders and `search_similar` — the retread class dies wherever the store survives, and archived losers enter all future novelty math.

## 5. Touchpoints and migration

| File | Change |
|---|---|
| `generic/strategy.py` | `compute_ideation_stance` + `IdeationStance` + `measured_full_run_price` (~40 lines, replaces advisory-prose logic); `_extract_open_measured_gaps`; cosine facts in pool hygiene; thread pool+reasoning+stance to node |
| `base.py` | `SearchNode` +3 fields (incl. serialization) |
| `experiment_memory/store.py` | record fields, unselected-candidate embeddings, `search_similar` over all embeddings |
| `gates/experiment_history_gate.py` | render unselected candidates + reasoning, labeled, full text |
| `prompts/ideation_claude_code.md` | structured Coverage format + verbatim-axis rule; stance paragraph replaces advisory sentence |
| `prompts/ideation_ensemble_addendum.md` | per-member operator briefs replace move-menu; gap-slot binding rule |
| `prompts/ideation_selector.md` | facts block; Diagnosis audit; stance-conditional tie-break |
| `src/kapso/config.yaml` (+ posttrain config) | add `ideation.duplicate_cosine_threshold: 0.80`; **delete** per-member `lens` keys |

Stores predating the structured-Coverage format are discarded/re-derived at upgrade (Rule 7 — no skip branch, one on-disk shape). Four atomic commits, suite green each (Rule 8):

1. **Persist the pool** — node/record fields, candidate embeddings, gate render, `search_similar` widening. Tests: record round-trip shape; corrupt stored line raises; render labels non-executed content. Standalone: kills the run13 class immediately.
2. **Structured Coverage + gap slot** — format contract, extractor, verbatim rule, addendum binding. Tests: extractor over synthetic histories (gap opens, closes on TESTED, oldest-first ordering); malformed new-candidate Coverage rejected in hygiene, never at persist; pool-floor flag-not-drop keeps ≥2 candidates.
3. **Duplicate alarm** — embeddings in hygiene, sibling/history split, facts block, config knob. Tests: synthetic embeddings trigger/clear the flag at threshold; nearest-neighbor id correctness.
4. **Stance rule + selector contract** — pure function (delete advisory prose), price helper, per-member briefs (delete lens keys), tie-break + Diagnosis audit prompt changes. Tests: stance transitions replayed on run16/run17/run19-shaped and 12-node RelBench-shaped synthetic histories (EXPLORE with empty history; FINAL_GAMBLE at remaining≤1; EXPLOIT otherwise; gap_axis threading; no-crash on unpriced first iteration).

## 6. Explicitly rejected

**From the literature:** UCB/UCT/MCTS and exploration constants (statistics never concentrate at N≤15; AIRA insensitivity); ε-greedy/stochastic knobs incl. AIDE's `debug_prob` (randomized exploration at a known short horizon is wasteful; every decision here is deterministic and reason-stamped); population machinery — islands, tournaments, MAP-Elites grids (asymptotic; only the archive and descriptor survive); LLM self-rated novelty scalars and literature-API novelty gates (near-chance rankers, 12/12-novel gates); debug-as-node and executed draft quotas (a 2–5h debug node is unaffordable; sessions self-repair; drafting lives inside ideation); learned/adaptive schedulers (marginal at Microsoft scale; nothing to learn at N=3).

**From the candidate designs:** A's `SearchDirective` module + observe hook + orchestrator wiring (every input already reaches the strategy — removable weight); A's directive-owned parenting and `PARENT_POLICIES` deletion (regresses the committed-but-unevaluated recovery branch for zero decision delta at N≤3); A's judge-side `directive_evidence` boolean as a policy trigger (destabilizes the strict 4-tag feedback contract, and a judge citing a misread slice stamps MEASURED — misses the actual run16 class; the selector-side audit covers both); A's `pivot_min_novelty` second knob and both stagnation knobs (A's epsilon, B's window — no campaign failure was a stagnation failure; add one knob only with evidence in hand); B's 6-class lever taxonomy (framework-owned vocabulary duplicating task-defined Coverage axes — a genericity regression); B's standalone `status:"untried"` records + `get_untried_ideas` tool (breaks `node_id` keying, floods recent renders, needs strategy→store coupling; fields-on-node delivers the value — the tool can later become a render over those fields if campaigns show pull-side misses); B's regeneration round (**deferred, not refuted**: the only mechanism that repairs a collapsed pool, but two of three judges hold it until a campaign shows the alarm alone lets a flagged duplicate ship — that event re-arms it, with its deadline carved from the remaining member window); persist-time fail-loud parsing of LLM-authored structure (Rule-2 strictness at the costliest point — hygiene-time is the correct topology); C's `EXPLORE_GAP` stance preemption (gap-chasing can starve the EXPLOIT middle on RelBench and override the last pull — the slot form gives the same guarantee); VALIDATE-fidelity idea racing (an unimplemented idea has no checkpoint; there is no low rung for ideas — the pool archive is the widen-N move).
# Learning from Trajectories — the Knowledge Flywheel

Design for closing kapso's outer loop: `evolve()` spends compute to turn
knowledge into a solution and leaves a trajectory; `learn()` spends compute to
turn trajectories back into knowledge. One bank, one scoring model, one merge
discipline — papers, repos, research findings, and campaign trajectories all
flow into the same knowledge bank, and the bank compiles back into the context
that the next campaign runs on.

Status: DESIGN (simulated on real wave-2..4 relbench artifacts, not yet built).
Companion docs: `researcher/research_learn_design.md` (research sources),
`wiki_structure/` (page schemas), `benchmarks/relbench/POSTMORTEM_WAVE2.md`
(the manual loop this automates).

---

## 1. The loop we are automating (observed, not hypothetical)

The relbench campaign already runs a knowledge loop — manually:

```
campaign → trace archive → human+Claude postmortem → hand-edited practice
notes in benchmarks/relbench/context.py → next campaign's problem context
```

That loop produced the nine "modelling practices", and the wave-4
`rel-amazon/user-churn` trace shows they work end to end: lenses cite them by
number, implementors execute them, gates measure them, judges close them.
Three facts from that trace pin the design:

1. **Delivery that wins is compiled context, not retrieval.** The campaign ran
   with `knowledge_search.enabled: false`; every bit of knowledge that mattered
   reached the agents as rendered prose in the problem context. The bank must
   therefore *compile to context packs*; KG search is a secondary interface.
2. **The valuable unit is a scoped, measurable directive with evidence.** What
   the lens replanner actually manipulates is "practice 8, measured dead here:
   insertion −0.000682 ± 0.001366 — closed" — a directive plus a measurement
   plus a scope. Prose without the measurement layer cannot support closure.
3. **Negative knowledge is first-class.** Campaign governance spends most of
   its words on what is *closed* ("signed SVD, coarse bounded attributes,
   hashed text stay closed on the cited measurements"). Retired knowledge must
   remain visible as guardrails, not be deleted.

`learn()` exists today for repos/research (`KnowledgePipeline`); the
trajectory path is the stubbed `ExperimentIngestor`. This design fills that
slot and adds the layer both paths lack: evidence ledgers and credibility.

---

## 2. Design at a glance

```
                       ┌────────────────────────────────────────────┐
   Source.Repo ───────▶│                                            │
   Source.Idea/Impl ──▶│  KnowledgePipeline (extended)              │
   Source.Report ─────▶│                                            │
   Source.Trajectory ─▶│  E0 harvest   deterministic bundle index   │
                       │  E1 mine      agent: events + claims       │
        knowledge bank │  E2 ground    adversary + mechanical check │──▶ updated bank
        (wiki git repo)│  E3 merge     ClaimResolver (agentic)      │    + LearnReport
              │        │  E4 score     pure code: ledger → scores   │
              └───────▶│  E5 render    packs + serving manifests    │
                       └────────────────────────────────────────────┘
                                          │
                 packs/<scope>.md ◀───────┘──────▶ measurement requests
                        │                                   │
                        ▼                                   ▼
              next evolve() campaign  ──────────▶  next Source.Trajectory
```

Everything mutable is a file in one git repo (the bank). Scores are **pure
functions of append-only ledgers** — never hand-edited, always replayable.
Agent calls happen only inside stages whose outputs are checked by mechanical
post-conditions (the EvaluationMaintainer pattern: trust boundaries are code,
not instructions).

---

## 3. The bank: what structure, and why wiki stays

Requirements pull in three directions: (a) humans and agents must read/edit
knowledge (rich prose, code, links); (b) a scorer must update reliabilities
deterministically; (c) campaigns must consume knowledge inside a token budget.

| Candidate bank | (a) editable | (b) scorable | (c) servable | Verdict |
|---|---|---|---|---|
| RAG chunk store | poor | poor (no identity) | retrieval-only | no |
| SQL/JSON claim DB | poor (no prose/code) | good | needs renderer | as a layer only |
| prose practice files (today) | good | none (proven bottleneck) | proven | as a layer only |
| wiki DAG (today) | good | none | via search only | as ground truth |

No single structure serves all three, so the bank is **layered**, with the
existing wiki as ground truth and two thin sidecars:

```
data/wikis/                          # THE bank — one git repo, diffable, revertible
  workflows|principles|implementations|environments|heuristics/*.md   # unchanged types
  ledgers/<item_slug>.jsonl          # append-only EvidenceEvents (the scoring input)
  scorecards.json                    # derived scores — regenerated, never edited
  packs/<scope_slug>.md              # rendered context packs (the delivery artifact)
  serving/<campaign_id>.json         # serving manifests (what was served to whom)
  .index                             # existing KG index ref (Neo4j/Weaviate rebuildable)
```

Why this wins the "is wiki the right choice" question: papers and repos already
ingest into wiki pages; trajectories produce the *same pages* plus ledger
events. One merge discipline, one retrieval stack, one renderer. The page
model needs only a small extension (§4), not a new page type — trajectory
lessons are overwhelmingly Heuristics ("the Wisdom") with some
Principles/Environments; winning solution snapshots ride the existing
repo-ingestion path as Workflows when wanted.

---

## 4. Anatomy of a knowledge item

An item = **wiki page (directive + prose) ⊕ ledger (evidence) ⊕ scorecard
(derived)**.

### 4.1 Page extension (Heuristic/Principle pages)

Three additions to the existing sections schema, all projections or metadata:

- Metadata block gains `[[scope::…]]` tags from a controlled vocabulary
  (`benchmark:relbench`, `dataset:rel-amazon`, `family:entity_binary_classification`,
  `data:has_text`, `data:low_rows`, `data:dormant_population`,
  `data:grouped_rows`, `data:temporal_shift`, `harness:kapso-campaign`, …) and
  `[[volatility::stable|volatile]]`.
- Metadata block gains a `{{Scorecard|credibility=0.78|state=VALIDATED|weight=6.5|version=3}}`
  line — a cached projection of the ledger, regenerated by E4, never edited.
- New `== Evidence ==` section — a rendered digest of the ledger (top
  supporting/refuting events with numbers and campaign links). Also a
  projection.

The `== The Insight (Rule of Thumb) ==` section is the **serving text**: the
exact prose the renderer places into packs. It is versioned
(`directive_version` in the scorecard) because refinements change what later
evidence measures (§5.4).

Two structural rules learned from practice 9's rot risk:

- **Stable trunk, volatile leaves.** A stable mechanism claim ("escalate to
  fine-tuning an open LLM when a large text gap remains") never embeds model
  names; it links `[[uses::Implementation:Current_Small_Open_Instruct_Models]]`
  — a volatile registry item with a 180-day half-life. Decay hits the leaf,
  not the trunk.
- **Directive must name its measurement.** Every VALIDATED-track item states
  how to measure it (which gate, which comparison). Items that cannot state a
  measurement are servable only as CANDIDATE context, never validated.

### 4.2 EvidenceEvent (one JSONL line in `ledgers/<item_slug>.jsonl`)

```json
{"event_id": "9f2c41d0a7b3",
 "item_id": "Heuristic/Cross_Family_Rank_Ensembling",
 "ts": "2026-08-13T00:14:00Z",
 "campaign": "rel-amazon--user-churn/20260813T015420_lane-c10",
 "cls": "ABLATED",
 "direction": 1,
 "measurement": {"delta": 0.0031691, "se": 0.00088, "metric": "roc_auc",
                 "local_gate": "passed"},
 "scope": ["benchmark:relbench", "dataset:rel-amazon",
           "family:entity_binary_classification", "data:has_text"],
 "directive_version": 3,
 "claim_note": "equal-rank blend of decorrelated finalists (Spearman 0.868) gained 3.6 SE",
 "pointer": {"artifact": "runs/run_0019/code/main.py",
             "quote": "ensemble_eligible", "log_line": 2165932},
 "render_ok": true,
 "extractor": {"miner": "codex/gpt-5.6-sol", "grounded": true}}
```

- `event_id = hash(item_id, campaign, cls, pointer)` — **idempotent ingestion**
  and repetition-proofing (the wave-4 log echoes the same changes.log ~20×; a
  naive counter would count it 20×).
- `direction ∈ {+1, −1, 0}`: set from the **campaign's own verdict** (gate
  passed/failed, judge ruling), not from the raw sign — a +0.00098 the judge
  ruled noise is `0`, not `+1`.
- `render_ok=false` marks events derived from test-side forensics of a task:
  they may move scores but are never rendered into that task's packs (§9).
- Rule 2 applies: a malformed ledger line raises; ledgers are append-only;
  merges union by `event_id`.

### 4.3 Scorecard (derived)

Per item: `{credibility_by_scope, weight_of_evidence, state, usage: {served,
engaged}, last_event_ts, directive_version}`. Regenerated in full by E4 from
ledgers (`scorecards.json`) — like `refresh_score_projections` on the evolve
side, the projection is disposable and the ledger is truth, which also makes
future scoring-formula changes a pure replay (rule 7 friendly).

---

## 5. Credibility — the heart

### 5.1 Evidence classes and base weights

| cls | meaning | base w |
|---|---|---|
| `ABLATED` | paired with/without measurement, SE reported, gate ruled | 3.0 |
| `FORENSIC` | postmortem prediction-level comparison (test-side, one-way) | 2.5 |
| `GATE_REJECTED` | targeted candidate rejected by the campaign's own gate | 2.0 |
| `OPERATIONAL` | mechanically worked / failed (API exists, crash recipe fixed) | 2.0 |
| `JUDGE_CLOSED` | lens/judge closed the direction citing numbers | 1.5 |
| `ADOPTED` | present in the winner (or loser) without targeted measurement | 0.5 |
| `EXTERNAL` | additional independent published source (the *first* source is not an event — it sets the birth prior p₀, never both) | 1.0 |
| `CITED` | mentioned, never executed | 0 (usage counter only) |

Weights are config (`knowledge_scoring` block), single-sourced per rule 1.

### 5.2 The formula

For item *i* under scope query *q* (the scope a pack is being built for):

```
w_e   = base(cls_e) · quality_e · σ(scope_e, q) · λ(age_e, volatility_i) · ν(version_e)
quality = clip(|Δ|/SE / 2, 0.5, 2.0)  when measured, else 1.0
σ     = 1.0 exact/subsumed · 0.5 sibling (same benchmark+family or all data tags)
        · 0.25 same family only · 0.1 otherwise
λ     = 1.0 stable · 0.5^(age/180d) volatile
ν     = 1.0 current directive_version · 0.7 older version
per-campaign cap: scale one campaign's events so Σ|w| ≤ 3.0

s = Σ w_e over direction=+1      r = Σ w_e over direction=−1
W = Σ w_e over all events (neutral included)

credibility(i, q) = (s + k·p₀) / (s + r + k)        k = 2.0
```

`p₀` is the birth prior: `EXTERNAL` published 0.60, repo-mined operational
0.70, trajectory-born (grounded once) 0.75, agent hypothesis 0.50.

This is deliberately the same statistics the bank itself teaches (practice 5/7):
an **empirical-Bayes shrunk win rate over correlation-capped, SE-scaled
evidence, shrunk toward the parent scope** — σ mixes out-of-scope evidence at
reduced weight instead of pretending scopes are independent, and the
per-campaign cap encodes that evidence inside one campaign shares data, bugs,
and luck.

Two behaviors worth stating as invariants:

- One strong refute retires a virgin claim *in scope* but only dents it
  globally: s=0, r=3 → cred = 1.2/5 = 0.24 in scope; at σ=0.1, cred = 0.52
  elsewhere. Local death, global caution.
- Sub-SE measurements (direction 0) add to `W` but not `s`/`r` — "tried,
  nothing detectable" is not refutation, but it is not free either (§5.3 INERT).

### 5.3 Lifecycle

```
            W<3 ──────────────  CANDIDATE  (served, marked unverified)
  cred ≥ 0.70, W ≥ 3 ────────  VALIDATED  (served prominently)
  0.35 < cred < 0.70, W ≥ 3,
  s ≥ 1 and r ≥ 1 ───────────  CONTESTED  (served with both sides + ablation ask)
  cred ≤ 0.35, W ≥ 3 ────────  RETIRED    (served ONLY as guardrail: "measured dead")
  W ≥ 4, max(s, r) < 0.5 ────  INERT      (repeatedly unmeasurable → out of packs)
  4 quarters unserved/uneventful ──  ARCHIVED  (tombstone; searchable, never packed)
```

"Putting it out" is two distinct exits, both observed necessary in the traces:
**RETIRED** items keep paying as guardrails (the lens replanner's "stays
closed" lists are exactly this rendering); **INERT** items exit because their
context tokens cost more than a no-effect directive returns — opportunity-cost
retirement, which pure up/down scoring misses.

### 5.4 Directive refinement, splits, and versioning

Real curation history shows directives are *refined* more often than killed:
practice 6 evolved from "blend finalists" to "blend **decorrelated
cross-branch** finalists, ship OOF at build time" after campaigns blended
0.98-correlated siblings. The merge stage therefore supports:

- `REFINE`: rewrite serving text, `directive_version += 1`; prior-version
  events stay in the ledger at ν = 0.7 (they measured a related but different
  formulation).
- `SPLIT`: when supports and refutes separate cleanly on a scope tag
  (detected mechanically: scope-conditional credibilities differ by > 0.3 with
  W ≥ 2 on each side), propose scoped variants sharing lineage.
- `CONSOLIDATE`: near-duplicate items merge; ledgers union; loser page becomes
  a redirect tombstone.

---

## 6. The learn pipeline (evolve's mirror)

`learn()` runs as a **deterministic staged frame with scoped agent calls** —
the EvaluationMaintainer/FeedbackGenerator pattern, not a free agent. Budgeted,
checkpointed per stage, resumable; every agent output crosses a mechanical
post-condition or it does not enter the bank.

### E0 — harvest (pure code)

Input: trajectory URI (GCS `.tgz`, handler work dir, or workspace dir).
Normalize into a `TrajectoryBundle` index over the real artifact layout (as
archived today): `campaign_meta.json`, `final_report.json`, per-run
`manifest.txt` + `private/{selection,metrics}.json` + code snapshots
(`PLAN.md`, `MEASUREMENT_PROFILE.md`, `changes.log`), the campaign log,
`.kapso/lens_plan_history.jsonl` and `features_history.md` when the workspace
is present, linked postmortems. Parsers are code and fail loud; absent
optional artifacts are documented defaults.

### E1 — mine (agent, parallel scoped calls)

Two passes, in priority order:

1. **Against-bank pass** (the bigger half of the value): for every item in the
   campaign's serving manifest, find engagement and outcome — was it executed,
   gated, adopted, closed? Emit candidate EvidenceEvents with quoted numbers
   and pointers. The lens revisions and judge feedbacks are pre-digested
   inputs here; changes.log carries the ablation numbers.
2. **Novelty pass**: difficulties → operational gotcha claims; judge/lens
   closures → negative claims; the winner's design mechanism → design-pattern
   claim; data facts (e.g. "all 12.6M rel-amazon reviews sit on 2,923 midnight
   timestamps") → dataset notes; harness observations (e.g. "clamped
   end-of-budget lanes produced only insurance duplicates") →
   `harness:kapso-campaign`-scoped claims served to the lens planner, not to
   implementors.

Contract: every candidate event/claim carries `pointer` + quoted numbers.
No pointer, no entry.

### E2 — ground (adversarial agent + mechanical frame)

A verifier session re-opens each pointer and tries to **refute** the event:
number absent, direction contradicts the campaign's own verdict, scope tags
unsupported → drop or correct. The frame then re-greps each surviving event's
quoted number in the pointed artifact (events whose numbers are computed, not
quoted, carry the formula and the frame recomputes it). A coverage critic
closes the loop with bounded retries — every archived run, every difficulty
line, every served item accounted for or explicitly skipped (the RepoIngestor
Phase-0 verification-loop pattern). The wave-4 maintainer rejecting all three
evidence-free change requests is this muscle already working in production.

### E3 — merge (ClaimResolver, extends KnowledgeMerger)

Same hierarchical merge agent and plan-file discipline, four new operations:
`ATTACH` (event → existing item; the common case), `CREATE` (new page, born
CANDIDATE), `REFINE` / `SPLIT` / `CONSOLIDATE` (§5.4). Identity resolution =
embedding shortlist (Weaviate) + LLM adjudication, scoped same-type as today.
Ledger writes are frame-owned (the agent proposes, code appends) so ledger
integrity is never in an agent's hands.

### E4 — score (pure code, no LLM)

Full replay: ledgers → scorecards, lifecycle transitions, SPLIT detection,
INERT sweep, and the **measurement-request queue**: CONTESTED/CANDIDATE items
ranked by `uncertainty × serving_rate` (value-of-information proxy), top-K
emitted as pack-injectable asks.

### E5 — render (pure code)

Assemble packs per scope from serving texts + scorecards: numbered VALIDATED/
CONTESTED items with stable ids (`[K-014]`), the guardrail block from
RETIRED-in-scope ("measured dead here — do not retry unless your scope
differs: …"), and up to N measurement requests ("if cheap, measure X with the
standard gate" — the mechanism that got practice 8 executed is exactly a
mandated measurement, generalized). Token budget from config; overflow drops
lowest `credibility × scope-match`. Deterministic: prose quality lives in the
serving texts (written at CREATE/REFINE inside E3), not in the renderer.

Output: git commit on the bank + `LearnReport` (events added, items
created/refined/split/retired, score moves, pack diffs, proposed
measurements) — reviewable before push, auditable forever.

---

## 7. Delivery and attribution (closing the loop tightly)

- Benchmarks consume packs instead of hand-edited constants: relbench's
  `MODELLING_PRACTICE_NOTE` becomes `packs/relbench--entity_binary--has_text.md`
  read at handler init (the committed pack is data in the bank repo; a
  benchmark pins a pack version for reproducibility).
- At campaign start the runner writes a **serving manifest**
  (`serving/<campaign_id>.json`: pack version + item ids + scope) into the
  handler work dir, so E1's attribution is exact bookkeeping, not text
  archaeology. Traces already show agents citing "practice 8/9" by number —
  stable ids make that machine-checkable.
- `ADOPTED` events are deliberately weak (0.5, campaign-capped): a winner
  used ten practices and proves none individually. Strong credit requires the
  campaign's own targeted measurement — which the measurement-request channel
  exists to provoke.

---

## 8. API

```python
kapso = Kapso(config_path=..., kg_index=...)

result = kapso.learn(
    Source.Trajectory("gs://leeroo-kapso-relbench-artifacts/runs/rel-amazon--user-churn/20260813T015420_lane-c10.tgz"),
    Source.Trajectory("tmp/relbench/rel-trial--study-outcome"),   # local work dir
    Source.Repo("https://github.com/snap-stanford/relbench"),      # unchanged path
    kapso.research("discrete-time survival stacking for churn"),   # unchanged path
    wiki_dir="data/wikis",
)
# result: PipelineResult + ScoreReport
#   .events_added .items_created .items_refined .score_moves
#   .lifecycle_transitions .packs_regenerated .proposed_measurements
```

`Source.Trajectory` replaces the stubbed `Source.Solution`/`ExperimentIngestor`
(rule 7: the stub is deleted, not aliased). The pipeline remains
`KnowledgePipeline` — harvest/mine/ground are the trajectory ingestor's
stages, merge extends the existing merger, score+render are new pipeline
stages that also run (with EXTERNAL birth events) for repo/paper sources, so
**every source now lands scored**: a paper claim is born CANDIDATE at p₀ 0.60
and lives or dies by later trajectories like everything else.

---

## 9. Safety and governance

- **Leakage.** Events derived from test-side forensics of task T carry
  `render_ok=false` and never render into packs served to T. Mechanism-level
  lessons (rendered as directives without task-specific test numbers) are the
  only forensic content that crosses back — the same line the manual
  postmortem loop already walks ("one-way local forensics, never fed back").
- **No self-dealing scores.** Scores are pure functions of grounded ledgers;
  extraction agents cannot write ledgers (the frame appends), cannot pump
  weight (event-id dedupe + per-campaign cap), and cannot invent numbers
  (E2's mechanical re-grep).
- **Auditability.** Bank = git repo; every score is explainable by listing its
  ledger lines; every learn run is one reviewable commit + LearnReport.
- **Config, fail-loud, minimal** (rules 1/2/10): all constants in a
  `knowledge_scoring` config block; corrupt ledger lines raise; scorecards and
  Evidence sections are projections, never sources.

---

## 10. Simulations (run against real artifacts)

### S1 — wave-4 `rel-amazon/user-churn` trace (full pass)

From the studied archive, E1/E2 yield ~14 grounded events. Representative:

| item | cls | dir | measurement | scope highlight |
|---|---|---|---|---|
| Cross_Family_Rank_Ensembling (P6) | ABLATED | +1 | +0.00317, SE 0.00088, gate passed | rel-amazon |
| LLM_Text_Feature_Extraction (P8) | ABLATED | −1 | −0.000682, SE 0.001366, gate rejected | rel-amazon, has_text |
| Frozen_Encoder_Marks (ModernBERT-as-marks) | ADOPTED | +1 | in winner, no targeted SE | rel-amazon |
| Task_Adaptive_LLM_Finetune (P9) | CITED | 0 | mandated twice, never executed | usage only |
| Clustered_SE_Acceptance_Gates (P5) | ADOPTED | +1 | governed every accept/reject | rel-amazon |
| NEW: Amazon_Review_Timestamps_Are_Midnight_Coalesced | OPERATIONAL | +1 | verified by recon | dataset:rel-amazon, born 0.75 |
| NEW: Torch_GRU_Requires_Contiguous_Hidden_State | OPERATIONAL | +1 | crash + fix | harness-agnostic gotcha |
| NEW (harness): Endgame_Clamped_Lanes_Buy_Only_Insurance | JUDGE_CLOSED | +1 | iter-3: 2 kills + 2 duplicates, 0 information | harness:kapso-campaign |

Catches that shaped the design: the same changes.log line appears ~20× in the
campaign log (→ pointer-hash dedupe); three in-session text-channel executions
are one campaign's correlated evidence (→ per-campaign cap); P9 was mandated
twice and never executed — a serving/priority signal no score can express
(→ usage counters + measurement-request escalation rather than score change).

### S2 — retro-scoring practice 8 across waves 1–4 (real history)

Ledger reconstruction: study-outcome ABLATED+ (capped 3.0) and ADOPTED+ (0.5)
from the trial campaigns; two amazon-churn ABLATED− at sub-SE magnitude but
gate-rejected (m=0.5 → 1.5 each). k=2, p₀=0.6:

| scope query | s | r | credibility | state |
|---|---|---|---|---|
| benchmark:relbench (global) | 3.5 | 3.0 | 0.55 | CONTESTED |
| rel-trial + has_text | 3.5 | 1.5 (σ 0.5) | 0.67 | CONTESTED→VALIDATED next support |
| rel-amazon churn | 1.75 (σ 0.5) | 3.0 | 0.44 | CONTESTED-low; guardrail + variant ask |

This reproduces what human curation actually did in wave 4: keep the practice,
scope the caveat, and point at the untested supervised variant ("zero-shot
rejection followed by supervised success is the measured norm"). The formula
lands where the experts landed — on real data.

### S3 — paper claim vs. trajectory evidence

A paper's "temporal graph transformers beat GBDT on relational tasks" enters
as Principle, EXTERNAL birth prior 0.60. Campaign events: GraphSAGE 0.7054
lost to the feature+blend line (ADOPTED-IN-LOSER 0.5 + JUDGE_CLOSED 1.5 with
depth-slice numbers, campaign-capped), second campaign JUDGE_CLOSED 1.5 →
s=0, r=3.5, cred = 1.2/5.5 ≈ 0.22 in `benchmark:relbench` → RETIRED in scope
(a second corroborating paper would add s=1.0 and hold it CONTESTED at ≈0.34);
either way the renderer stops recommending architecture-first *on relbench*
while the claim stays ≈0.5 in distant scopes. This independently re-derives
the hand-written FEATURE_ENGINEERING_NOTE ("architecture swaps have
repeatedly measured dead while feature widening kept paying").

### S4 — scope split

Within-group normalization (P1) accumulates supports on avito/salt
(grouped rows) and a hypothetical refute on a task with no grouping key. The
mechanical split detector sees credibility 0.85 under `data:grouped_rows` vs
0.30 without it → SPLIT proposal keyed on the *data-shape tag*, not the
benchmark — which is why the scope vocabulary carries data-shape flags at all.

### S5 — volatile leaves

"Qwen3.5-9B is the best small open model" lives in a volatile registry item
(half-life 180d) linked from the stable P9 trunk. A superseded model decays
out and one `OPERATIONAL` "model gone/worse" event retires it without touching
the trunk's credibility. The trunk's serving text never rots because it never
contained the name.

### S6 — duplicate lessons across campaigns

Two campaigns both produce "use clustered-SE acceptance gates". E3's identity
resolution attaches the second campaign's events to the existing item
(embedding match + adjudication); the failure mode of a thousand near-identical
Heuristic pages is structurally blocked because ATTACH precedes CREATE in the
resolver's decision order and CREATE requires a stated novelty over the
nearest match.

### S7 — adversarial extraction

A miner hallucinates "+0.01 from feature X": E2 re-greps run artifacts, number
absent → event dropped, miner told. A miner repeats one closure five times:
event-id collision → one event. A compromised extractor claims ABLATED for an
un-run practice: no pointer to a gate ruling → direction unusable → CITED at
best. The scorer's inputs stay grounded even when the extractor is not.

---

## 11. Refinements the simulations forced (design deltas)

1. Pointer-hash `event_id` dedupe + per-campaign weight cap (S1: log echoes,
   correlated in-campaign retries).
2. `direction` comes from the campaign's own gate/judge verdict; sub-SE
   magnitudes scale weight down but do not flip direction (S1: run_0018's
   "+0.00098 = judge-ruled loss").
3. INERT lifecycle exit — repeatedly-unmeasurable items leave packs on
   opportunity cost, a distinct exit from RETIRED (S1: token budget is the
   scarce resource packs spend).
4. Directive versioning with old-version discount ν, plus REFINE as a
   first-class merge op (S2: practices evolve more than they die).
5. Stable-trunk / volatile-leaf split with linked registry items (S5).
6. Serving manifests written at campaign start — attribution as bookkeeping
   (S1: lens texts cite practice numbers; make it exact).
7. `render_ok` leak flag on forensic events (wave-2 postmortem ingestion).
8. RETIRED renders as guardrails, never disappears (every lens revision's
   "stays closed" list is this feature, observed working).
9. Harness-scoped knowledge is ordinary knowledge with scope
   `harness:kapso-campaign`, served to the lens planner — no special mechanism
   (S1: endgame-economics lesson).

---

## 12. Phasing

**M1 — the ledger and the loop (build first)**
Schemas (`EvidenceEvent`, ledgers, scorecards) + pure `CredibilityScorer` +
`TrajectoryIngestor` (E0 harvest for today's archive layout; E1 mine; E2
minimal grounding: pointer re-grep, no coverage critic) + resolver ops
ATTACH/CREATE + E5 renderer for one scope + **seed migration**: the nine
relbench practices become bank items with hand-seeded ledgers from the
wave-2/-4 postmortems; relbench context reads the rendered pack; runner writes
serving manifests. Exit test: replaying the wave-4 archive reproduces §S1's
events and §S2's scores.

**M2 — governance depth**
Adversarial grounding + coverage critic loop; REFINE/SPLIT/CONSOLIDATE;
INERT sweep; measurement-request queue injected into packs; postmortem
(FORENSIC) ingestion with `render_ok`; harness-scope pack for the lens
planner.

**M3 — breadth**
Cross-benchmark packs and scope shrinkage tuning; VOI-ranked active
measurement scheduling; winning-solution ingestion as Workflow pages via the
existing repo path; KG search results annotated with scorecards so retrieval
and packs tell one story.

---

## 13. Open questions

- **Serving-rate feedback bias**: heavily served items accrue evidence faster
  (rich-get-richer). The measurement-request queue deliberately spends slots
  on CONTESTED/CANDIDATE items; whether that exploration budget suffices
  needs live data.
- **Scope vocabulary governance**: resolver proposes new tags, a config
  allowlist gates them — but tag drift across benchmarks will need periodic
  consolidation passes.
- **Attribution ceiling**: without per-practice ablations, most positive
  evidence stays weak (`ADOPTED`); the design accepts slow validation over
  false precision. If validation is too slow in practice, campaigns could be
  asked to run one dedicated ablation lane per wave — a policy question, not a
  mechanism one.
- **Pack pinning vs freshness**: benchmarks pin pack versions for
  reproducibility; how eagerly waves adopt new packs is an operator dial that
  interacts with A/B hygiene (the no-champion arm result shows serving
  content changes measurably alter search behavior).

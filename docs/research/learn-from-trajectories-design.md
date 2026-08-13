# Learn from Trajectories — the knowledge bank and the trajectory learner

High-level design for kapso's learning loop: a mechanism that consumes evolve-campaign
trajectories plus the current knowledge bank and updates the bank — **insight cards**
(evidenced claims, text) and **procedure cards** (how-tos that mature from text playbooks
into replay-verified code), each carrying a **reliability record** that later campaign
outcomes move up or down, with retirement when it stays down. The bank is an
**Open Knowledge Format (OKF) bundle** — a directory of markdown-with-frontmatter
concepts — so any OKF consumer can read it and third-party OKF bundles can join it.
All intelligence is delivered by coding-agent CLI sessions; every trust boundary is a
mechanical post-condition in a deterministic Python frame (the EvaluationMaintainer
pattern). Evidence base: `learn-from-trajectories-litreview.md` (22-paper review, same
directory) and the wave-4 `rel-amazon/user-churn` trace forensics.

## 0. Position — what this replaces, keeps, and re-opens

**Replaces (Rule 7, delete not deprecate):**
- The 5-type wiki KG and its learners (`knowledge_base/learners/`, ingestors, merger,
  Neo4j/Weaviate backends) as the learning path. The wiki pipeline extracts from *code
  repos*; it has no outcome signal, no lifecycle, and its `ExperimentIngestor` — the one
  piece that would have read trajectories — was never built.
- The hand loop that currently does this job: postmortem → manually rebuilt
  `MODELLING_PRACTICE_NOTE` practices in `benchmarks/relbench/context.py`. That loop is
  the proof of value (the 9 practices demonstrably steer campaigns — wave-4's champion
  was practice 6 executed verbatim by an agent) and the thing being mechanized. The
  practices become the bank's founding cards (mostly procedures with attached
  evidence; a few insights), back-filled from the postmortems.

**Keeps (from `cross-run-knowledge-design.md`):** the episodic layer and its mechanical
trust inventions — merged experiment store with origin/sign stamping, mandatory-bindings
rendering (a card without scope+citations is *unrepresentable*, not discouraged),
harvest-time validation, the shingle leak gate, the bug-class screen, starter-kit
exemplars. Cards are a semantic layer **on top of** that episodic layer, not a
replacement: exemplars stay the ground truth cards cite.

**Re-opens two of that doc's kills, with cause.** It killed reuse counters/tiering
("volume 10–30 runs never produce") and the LLM curation pass ("consolidation
overgeneralizes"). Two things changed:

1. **The signal regime.** RelBench is not 10–30 thinly-instrumented runs. One campaign
   yields ~20+ registered grader-scored runs, a `features_history.md` ledger of
   TESTED-KEPT/REJECTED entries *with measured deltas and clustered SEs*, per-node
   `technical_difficulties`, judge verdicts, and 4-lane contrast groups per iteration;
   the campaign fleet spans 65 tasks × multiple waves. Confirm/contradict events number
   in the thousands, and — unlike posttrain — the sign is mechanical (immutable grader,
   keep-best banking), so no human verdict bottleneck. The kill still binds where it was
   true: **cross-task transfer visits per card stay rare**, and the design treats that
   honestly (two-clock reliability, §3.3; cold states, not fake scores).
2. **The guardrails now exist.** The 2026 literature supplies what 2024-era consolidation
   lacked: delta-based curation with deterministic application (ACE's answer to context
   collapse), attribution-gated scoring (RoMeRL's answer to the memory-reward trap), and
   — new here — **mechanical citation resolution**: a card's cited evidence must resolve
   to real ledger lines / archived runs with matching signs, checked by code at admission.
   An overgeneralized card fails resolution or gets its scope capped to what it can cite.
   Consolidation is no longer "regulated prompting"; it is a validated transaction.

**Scoring principle inherited from ideation-v2's signal ledger:** every number in the
bank is *Measured* (grader-backed) or it does not exist. Usage frequency, retrieval
count, and LLM opinion are never score inputs — popularity ≠ correctness.

## 1. Research digest (only what shaped the design)

- **Contrast is the extraction signal** (ExpeL, AutoGuide, TF-GRPO — group-size-1
  ablation kills the gain). Kapso's K-lane iterations and KEPT/REJECTED ledger entries
  are ready-made contrastive pairs with ground-truth deltas.
- **Abstract, don't archive** (Dynamic Cheatsheet's full-history < no-memory; companion
  study's raw-trajectory −9.5% vs insights +6.5% forward transfer). Miners see
  structured views, never the 2.35M-line log (also Rule 6: window, don't clip).
- **Verify code before storing** (Voyager's most-valuable-component ablation; ASI's
  ~15.6% admission rate; SkillRevise: one-shot authored skills score *below no-skill*
  until trace-conditioned revision, 39.5 → 61.6). Code-representation procedures pass
  a replay gate or stay candidates.
- **Score only on attributable outcomes** (RoMeRL: episode-reward sharing provably
  inflates useless items; ~45% cold-Q in growing banks). No blanket updates to
  everything injected into a winning campaign.
- **LLM proposes deltas, deterministic code applies them** (ACE's context collapse:
  monolithic rewrite 18,282 → 122 tokens, −9.6 points). The bank is never rewritten
  wholesale; card identity and history are git-native.
- **Failures are first-class** (MemRL: ~12% of failure-derived memories earn top
  utility) — but gated by imperfect-agent triage (AutoManual): an experiment the
  implementation agent botched updates nothing.
- **Conflict resolution by evidence is greenfield**: no reviewed system resolves
  contradiction. Kapso cards cite measured deltas, so contradiction resolves by **scope
  split** ("helps family A [delta, run]; hurts family B [delta, run]") — novel assembly.
- **From the wave-4 trace**: the artifacts are already sufficient and structured
  (ledger, difficulties, lens history, manifests, verdicts); the lens replanner already
  *consumes* exactly the evidence cards would carry — the bank's render is its natural
  upstream.
- **Format and operations references.** *OKF* (Google Cloud, 2026): concepts as
  markdown+frontmatter files, path = identity, links = graph, `index.md` progressive
  disclosure, `log.md`, minimally-opinionated `type` — adopted wholesale (§3.1).
  *gbrain*: git-markdown as system-of-record with a derived retrieval index (proven at
  155k pages), zero-LLM typed-edge extraction, synthesis-with-explicit-gap-analysis,
  offline consolidation ("dream cycle") including a suspected-contradictions sweep,
  per-stage explainable retrieval — all adopted as operational patterns. *mempalace*:
  verbatim episodic storage wins for recall (96.6% R@5 raw) — cards cite into verbatim
  artifacts, never replace them; scoped-not-flat retrieval; temporal validity windows
  on facts — adopted for env-tagged insights (§3.3).
- **Abstraction mechanisms** (the core problem — cards must transfer, not memorize).
  The memory survey's three inductive operations for cross-trajectory abstraction:
  contrastive induction over success/failure sets, distillation of fine-grained
  actions into high-order patterns, encapsulation of recurring behavior into
  functions — and its admission bar: Experience is *scenario-detached* and
  MDL-compressive (|K| ≪ Σ|τ|), anything context-bound is still Reflection and stays
  in the episodic layer. Classic **explanation-based generalization** (Mitchell/DeJong):
  a single instance licenses a general claim only through an explicit explanation of
  WHY it worked — the generalization keeps exactly the features the explanation needs,
  so **scope derives from the mechanism, not from the instance**. Schema induction
  (Gick & Holyoak): comparing ≥2 aligned instances yields the shared relational
  schema; one instance alone rarely does. *Notes to Self* (arXiv 2607.20372):
  abstractions must be free of problem-specific numbers/names/story details (an
  enforceable lint), separate strategy vs caution buckets, and help most where the
  consumer is weak — supporting gap-targeted injection over blanket briefs.

## 2. The three knowledge layers

| Layer | Unit | Scope | Lifecycle | Exists today |
|---|---|---|---|---|
| Episodic | Signed exemplars: runs, solutions, difficulties, feedback | per task, cross-run | sliding window + archive | cross-run store, runs archive, claims |
| Task-local | Living documents: `features_history.md`, `table_information.md` | one task | append-only | shipped |
| **Semantic (new)** | **Cards** — `insight` (evidenced claim) + `procedure` (text→code ladder), OKF concepts | task family → global | reliability lifecycle | the hand-written practice notes |

The learner writes the semantic layer; it reads all three. Nothing about the lower
layers changes except stamping (§5.1).

## 3. The knowledge bank

### 3.1 Storage — an OKF bundle in a git repo

The bank is an **Open Knowledge Format bundle**: a directory of markdown files with
YAML frontmatter, one concept per file, **file path = identity**, ordinary markdown
links between concepts forming the graph, `index.md` per directory for progressive
disclosure (this is how navigating agent sessions orient — miners, the curator, and
any future OKF consumer read the same indexes; each index line is
`- [title](path) — hero`, so a whole section scans in one screen), and a root
`log.md` as the chronological journal (one entry per learner run). OKF is minimally opinionated —
only `type` is required — so kapso's scoring state rides as producer extension fields
without breaking conformance. What conformance buys: any OKF tool can render/inspect
the bank, and third-party OKF bundles (e.g. dataset documentation) can be mounted
beside the cards as unscored reference concepts.

The repo is the system of record; everything queryable is derived — the gbrain
discipline (git markdown as truth, synced into a retrieval index; proven at 155k
pages) at our scale means one derived `index/` (embeddings + a link/edge table),
rebuilt by the frame at merge. The **hero line is the primary retrieval text** —
embedded per card and shown in every shortlist; claim + body embed as secondary text,
so fast retrieval scans heroes and only the selected k cards render in full. **Link extraction is zero-LLM**: the frame parses body
markdown links and the typed extension fields (`supersedes`, `contradicts`) into the
edge table; no model sits in the graph-construction path.

```
knowledge_bank/
  index.md                          # root orientation for agents (OKF)
  log.md                            # learner-run journal (OKF)
  insights/<slug>.md                # insight cards
  insights/index.md
  procedures/<slug>/card.md         # procedure cards (dir: docs + code + replay test)
  procedures/<slug>/*.py            #   present once representation reaches `code`
  procedures/index.md
  reference/...                     # optional mounted OKF concepts — never scored
  retired/...                       # moved, never deleted; history intact
  index/{embeddings.parquet,edges.parquet}   # derived, rebuilt at merge
```

Git is the audit trail, version store, and rollback: every learner run commits as one
reviewed transaction; `bank_head` (commit sha) stamps everything downstream, exactly
as `evaluator_id` stamps evaluations today. Retrieval is **scope-first, never flat**
(the mempalace lesson): queries filter on scope tags before similarity ranks within.

### 3.2 The card — two types, OKF-conformant

**The minimal type set is two.** The load-bearing distinction in the whole design is
*what the lifecycle can do with a card*: a **claim** can only be confirmed,
contradicted, or invalidated by later evidence; a **how-to** can additionally be
*executed and verified*. Every learning the trajectories actually produce reduces to
one of these:

- **`insight`** — an evidenced claim. Covers positive findings, negative results and
  anti-patterns (the technical_difficulties / TESTED-REJECTED export), dataset facts
  ("all 12.6M reviews sit on midnight timestamps"), and environment facts (lightgbm
  4.x callbacks). Sign lives in the evidence, not the type; flavor lives in `tags`
  (`pitfall`, `env`, `dataset` — rendering hints, not ontology).
- **`procedure`** — a how-to, with a **representation ladder**: born as text (a
  playbook: practice-8's probe-first protocol), it matures to `representation: code`
  under the *same identity* once the replay gate passes — gaining an `entrypoint`,
  machine-checkable `preconditions` (Mobile-Agent-E's lesson that LLM-judged
  preconditions fail), and a `replay` record. The old playbook-vs-recipe promotion
  question dissolves: it is one card whose representation upgrades, keeping its
  evidence history. Expected v1 code-representation procedures, straight from
  recurring trace patterns: the forward-origin gate harness, the two-model-contract
  prediction writer, the clustered-bootstrap acceptance test, the equal-rank blend
  gate, the shared-cache registry protocol.

Anything else in the bundle (mounted dataset docs, third-party OKF bundles) is a
reference concept: retrievable, linkable, never scored. If two types ever prove
genuinely insufficient, the escape hatch is gbrain-style schema packs (types as
configurable data), not a richer built-in ontology.

**Cards are abstractions; bound observations are evidence, never cards.** The unit
the miners produce — "run_0019's blend of run_0010 gained +0.0032 on rel-amazon" — is
an *observation*: episodic, bound, useful only as evidence. A card is the
generalization that observation supports, stated at the level of the domain
(predictive ML modelling), carrying the **mechanism** (why it works) from which its
applicability conditions derive. Two mechanical guards enforce this:

- **`generality: family | domain`** is a required extension field, and the claimed
  level is what the transfer clock measures against — a domain card contradicted
  outside its home family demotes to `family` scope, not to retirement.
- **The bound-identifier lint** (Notes-to-Self operationalized): dataset names, task
  ids, run ids, column names, and campaign-specific numbers are *banned* from
  `title`/`description`/`scope_conditions` — mechanically checked against the
  benchmark's vocabulary — and permitted only inside `evidence` entries and body
  citations. A card whose claim cannot be stated without its bindings is an
  observation, and the frame refuses to admit it as a card.

Frontmatter = **OKF reserved fields + kapso extensions**. Bindings are mandatory and
mechanically validated — admission raises (Rule 2) on a card whose `scope` is empty or
whose `evidence` refs do not resolve against the episodic layer with the stated signs.

```yaml
# --- OKF reserved fields ---
type: insight                    # insight | procedure (the only scored types)
title: Group-relative normalization
description: >-                  # HERO one-liner (OKF-native slot): the retrieval hook.
                                 # One compact sentence, ≤~140 chars, no bound
                                 # identifiers, no numbers — what index lists, retrieval
                                 # shortlists, and OKF viewers show for this card
  Rank and z-score features within their competing group — absolute values
  cannot see relative standing.
resource: gs://leeroo-kapso-relbench-artifacts/runs/rel-amazon--user-churn/…  # source trajectory
tags: [family:entity_binary_classification, family:entity_regression,
       data:grouped_rows, benchmark:relbench]        # scope vocabulary (machine-checkable)
timestamp: 2026-08-14T09:00:00Z
# --- kapso extensions ---
generality: domain               # family | domain — the level transfer is measured at
claim: >-                        # insights only: the full falsifiable statement the
                                 # transfer clock measures against (scope-first;
                                 # AutoManual). Procedures carry no claim — their
                                 # verification is execution, not textual agreement
  On tasks where prediction rows share a grouping key, within-group rank/z-score
  companions of informative raw features improve the held-out metric across
  model families.
scope_conditions: "rows share seed timestamp / session / parent entity"  # prose, judged
evidence:                        # ≥1 required; each ref must RESOLVE mechanically
  - campaign: rel-amazon--user-churn/20260813T015420_lane-c10
    ref: features_history.md#within-origin-ranks
    sign: KEPT
    delta: "+0.0032 AUC ≈ 3.6 clustered SE"
reliability:
  evidence_weight: {measurements: 3, task_families: 2}   # at admission, from citations
  transfer: {confirms: 0, contradicts: 0, exercised: 0, by_family: {}}
  state: candidate               # candidate | active | cold | retired | superseded
  last_exercised: null
provenance:
  learner_run: lr_2026-08-14T…
  authoring_model: …             # cross-model validity tag (SkillRevise 8–15pp loss)
  version: 1
supersedes: null                 # typed edges; all other links are body markdown links
contradicts: []
probe: >-                        # optional: how a future campaign can test this cheaply
  Ablate within-group companions of the top-5 features on one forward fold.
# --- procedure cards additionally (representation: code only) ---
# representation: text | code
# entrypoint: run_forward_gate.py
# preconditions: {task_families: […], requires: [features parquet, ≥2 origins]}
# replay: {archived_run: runs/run_0019, expected_metric: …, last_replayed: …}
```

**The single-source card rule.** Every fact lives in exactly one field; everything a
consumer sees assembled is a projection. `description` is the hero's only home,
`claim` the claim's, `scope_conditions` applicability's, `evidence` the instances'.
The **body carries only what has no frontmatter home** — the mechanism, the measured
limits, adaptation notes, and ordinary markdown links to related cards (OKF: links
are the graph; the frame extracts them mechanically) — and references evidence by
inline `[E1]`/`[E2]` markers instead of re-citing it. No body heading (the renderer
adds the title), no restating of any frontmatter field: a curator `REFINE` therefore
targets exactly one home, and a version bump means one thing. The **served card is a
renderer projection** assembled at brief-compile time — title + hero + derived
reliability line (so the reading LLM weighs a battle-tested claim differently, CLIN's
hedging result) + claim + scope + body + evidence digest + probe — and is never
stored. The duality that stays: `tags` (machine scope) vs `scope_conditions` (reader
scope) express the same thing at two precisions; the stage-V verifier checks their
consistency.

### 3.3 Two-clock reliability

A single scalar conflates "true where measured" with "generalizes". Cards carry two
records:

- **Evidence weight** — how much grader-backed measurement supports the claim *where it
  was mined*: number of independent measurements, families covered, delta magnitudes.
  Set at admission from resolved citations; grows when later learner runs attach new
  supporting evidence. This clock starts non-zero — unlike the cheap-rollout literature,
  kapso cards are born from measured deltas, not hunches.
- **Transfer record** — what happened when the card was *exercised elsewhere*:
  `confirms / contradicts / exercised`, per task family, from attribution events
  (§5.2). This clock moves slowly (the sparse-signal regime) and the design never
  pretends otherwise: a card with high evidence weight and an empty transfer record is
  `active` but renders as "unverified beyond its home family" — a distinct state, not a
  hidden default (RoMeRL's cold-Q lesson).

Derived rank for retrieval = scope match × similarity × f(evidence weight, transfer,
staleness). Counters are the on-disk truth; `f` is one pure function that can evolve
without touching stored state (Rule 10; avoids formula lock-in).

**Lifecycle (all transitions mechanical, frame-owned):**
`candidate` (mined, gates pending) → `active` (admitted) → `cold` (no exercise in N
campaigns; down-ranked, never hidden — cold ≠ wrong) → `retired` (contradicts dominate
with ≥m visits, or replay breaks for code procedures, or superseded). Retirement moves the file
to `retired/` with history; a superseding card links back and inherits a *discounted*
prior (RoMeRL warm-start, with its rides-old-credibility caveat noted in provenance).
Contradiction's first response is never deletion: it is **scope split** — the frame
requires a contradicting event to carry its own citation, and the curator proposes a
narrowed scope citing both sides.

One distinction rides on tags (the mempalace validity-window idea): **env-tagged
insights expire, they are not refuted**. "lightgbm 4.x wants callbacks" going stale is
not evidence the card was wrong — a version-bump event *invalidates* (supersedes with
a validity end), leaving the transfer record intact, where a measured contradiction
*refutes*. Conflating the two would poison env cards' reliability for no informational
gain.

## 4. The trajectory learner

An evolve-shaped agentic pipeline: deterministic orchestrator (`TrajectoryLearner`)
around scoped coding-agent CLI sessions, one role per stage, mechanical post-conditions
between. Reuses evolve's substrate — coding-agent adapters, prompt loader, gated MCP,
budget/telemetry ledger, checkpoint pattern — but **not** the search strategy: learning
is a pipeline with a loop-until-dry tail, not score-guided search. (Option B — running
the learner *as* an evolve campaign whose "score" is downstream campaign performance —
is deliberately deferred: its evaluation signal costs a campaign per iteration. §9.)

Input: a **trajectory bundle** — one campaign's artifacts located from the workspace or
a GCS archive (the frame normalizes: ledger, difficulties, lens history, judge
verdicts, manifests, runs archive, campaign log kept only for on-demand forensics).
Plus the current bank at `bank_head`.

| Stage | Intelligence (CLI session, read-only tools + views) | Mechanical post-condition (frame) |
|---|---|---|
| **T — Triage** | none (pure code) | Build per-miner views; classify nodes success/recovered/failed; drop imperfect-agent failures (deadline kills, infra deaths — end-facts are already recorded) from *knowledge* mining while keeping them for pitfall mining; compute campaign SEs from stored bootstrap numbers |
| **M — Mine** (parallel, lens-per-miner) | contrast miner (ledger KEPT/REJECTED pairs + lane groups), pitfall miner (technical_difficulties + failed lineages), strategy miner (lens history + judge verdicts + postmortem), procedure scout (winning code across runs archive, looking for procedures recurring ≥2 times) | Output is **observations** (bound, episodic — never cards); every ref must **resolve** (file+line/run exists, sign matches, delta matches within tolerance) — unresolvable observations are rejected with a named finding, not repaired silently |
| **A — Abstract** (the core stage) | one abstraction session per observation cluster: align new observations with each other AND with the bank's existing cards/evidence, then induce or refine generalizations — variable abstraction (bound entities → typed roles), cross-instance schema induction, or single-instance explanation-based generalization (state the mechanism; derive applicability from it; propose the falsifying probe) | Two admission routes, mechanically distinguished: **(i) induction** — ≥2 aligned instances from ≥2 independent measurements; **(ii) EBG** — 1 instance + a mechanism the stage-V verifier must independently endorse, admitted at reduced evidence weight. Every candidate card passes the bound-identifier lint and the MDL test (a card ≈ as long as its cited instances has not abstracted); observations that support no admissible card stay in the episodic ledger |
| **C — Curate** | one session with the bank slice (embedding neighbors of each candidate) in context; emits typed ops: `create / attach_evidence(card) / revise_scope(card) / split_scope(card) / supersede(card) / link` | Frame applies ops deterministically (ACE): stable ids, in-place counter updates, embedding dedup assist; ops on nonexistent cards raise; no op may edit another card's evidence list except `attach_evidence` with resolving refs |
| **V — Verify** | text: an adversarial checker per card — does the cited artifact actually support the claim at the stated strength? (the maintainer's change-request triage stance, pointed inward); code: a session adapts the procedure + writes its replay test | Text: checker verdict recorded; a failed check demotes to candidate with the objection attached. Code: frame executes replay against the archived run (correctness + actually-invoked + effect, ASI's three conditions) inside the existing sandbox/timeout machinery; failures stay candidates with the trace attached (SkillRevise's revision loop gets one retry, then parks) |
| **S — Score** | none (pure code) | Attribution events from this trajectory update *prior* cards' transfer records (§5.2); lifecycle transitions applied; decoy audit checked (§6) |
| **R — Reflect (loop-until-dry)** | one critic session: "what did this trajectory teach that the bank now does not carry?" | Frame loops M–S on the critic's named gaps until a round adds nothing (two dry rounds), bounded by the learner's budget block |

Output: one bank commit (all stages' changes, one reviewed transaction), a
`learner_report.json` (cards created/updated/demoted, events applied, rejections with
reasons), and the new `bank_head`. The commit message carries the source trajectory id —
bank history reads as a ledger of learnings per campaign.

Concurrency: learner runs are serialized per bank (single-writer; the curator is the
only merge path — the same single-writer discipline evolve's repo_lock uses).

Besides the per-trajectory run, the learner has one standing **consolidation mode**
(gbrain's dream cycle, scheduled off-hours): a mechanical shortlist of suspected
contradictions and near-duplicates (edge table + embedding neighbors + opposing-sign
evidence on overlapping scopes) reviewed by one curator session emitting the same
typed ops, plus the staleness sweep (validity expiries, cold transitions) and index
rebuild — all under the same single-writer commit discipline. No new machinery: it is
stages C/S with an empty mining phase.

## 5. Closing the loop

### 5.1 Injection — the briefing compiler

At campaign start the runner calls `BriefingCompiler.compile(task, bank_head)`:

- **Insights and text-representation procedures** render into the problem context,
  replacing the hand-maintained
  `MODELLING_PRACTICE_NOTE` / `FEATURE_ENGINEERING_NOTE` blocks: top-k by scope match ×
  similarity × reliability rank, k small (AutoGuide/DS-Agent: 2–4 per section, budgeted
  per kind), full text never clipped (Rule 6 — k caps selection, not content). Each
  renders with id, reliability line, and citations. Negative-signed insights
  (pitfall-tagged) for the task family ride along as guardrails (AutoManual's fallback
  routing). The brief closes with an explicit **gap analysis**, gbrain-style: which
  scope tags of this task have no active cards, which nearest-scope cards were included
  at reduced confidence, and which active cards are stale — the honest "what the bank
  does not know" that both primes probes and stops false authority. Compilation writes
  a **serving record** (per card: scope-match, similarity, reliability components —
  gbrain's `--explain` applied to injection) so every brief is auditable and
  attribution later binds to exactly what was served.
- **Code-representation procedures** stage into the shared artifact workspace (`shared_cache/procedures/…`)
  with the registry entry and a provenance README — "verified exemplars: adapt or
  invoke; replay-tested against <run>". (Callable-tool wiring, ASI's +3.7 injection-site
  result, is a v2 upgrade; staged-and-registered is v1.)
- **The citation contract**: ideation/implementation prompts gain one paragraph — a spec
  that uses a card cites `[card:<id>]`; the feedback judge's verdict template gains a
  `cards_load_bearing` field. This is the attribution substrate, and it is honest about
  its limits (§6: silent influence is measured by A/B, not by attribution).
- **Stamping**: `campaign_meta.json` gains `bank_head`; every attribution event later
  binds to the exact card versions the campaign read.
- The lens planner/replanner prompts render low-transfer cards as **cheap probes** —
  "unverified on this family; `probe:` says how to test it in one fold" — steering a
  slice of the existing exploration budget into deliberate card validation. This is the
  novel piece: the bank does not wait to be exercised, it *asks questions*, and the
  replanner's evidence-driven re-aiming (proven in wave-4) is the natural carrier.

### 5.2 Attribution events (the score inputs, all Measured)

An event is `(card_id, campaign, node/run, kind, sign, delta, SE)`; kinds, strictest
first:

1. **Direct test** — a ledger entry or lane experiment whose claim matches the card
   (probe results included): KEPT with significant delta → `confirm`; REJECTED /
   significant negative → `contradict`; within noise → `exercised` only.
2. **Cited use in a scored lineage** — spec cited the card and the judge named it
   load-bearing; sign from the node's grader outcome vs parent.
3. **Procedure invocation** — a staged procedure demonstrably ran (registry/use markers) in a
   registered run; sign from that run's outcome.

Significance reuses the campaign's own clustered-SE machinery (practice 5 — already
computed and logged in-trace). Sub-threshold deltas update nothing but `exercised`.
There is deliberately **no** "injected into a winning campaign" event — that is the
memory-reward trap, and it stays excluded even though it would move scores faster.

### 5.3 What changes in evolve

Small and additive: the briefing block replaces two static context constants; two
prompt paragraphs (citation contract, probe rendering); one judge template field;
`bank_head` in campaign meta. No orchestrator/search changes — the learner runs *after*
campaigns (post-fetch, next to harvest), never inside them.

## 6. Trust model — every documented failure mode gets a mechanism

| Failure mode (source) | Mechanism |
|---|---|
| Memory-reward trap (RoMeRL) | Only attribution events of §5.2 update scores; no co-injection credit |
| Context collapse (ACE, DC) | Typed delta ops, deterministically applied; bank never LLM-rewritten; git history |
| Popularity ≠ correctness (GenAgents/MemoryBank) | Usage/retrieval counts are not score inputs, anywhere |
| Corrupted evaluator (Reflexion MBPP; AWM judge) | Events derive from the immutable grader's outcomes only; judge names cards but never signs them |
| Cold scores (RoMeRL ~45%) | Visit counters + explicit `cold` state; retirement requires ≥m visits; probes exist to buy visits cheaply |
| Consolidation overgeneralizes (cross-run doc; faulty-consolidation) | Mechanical citation resolution at admission; scope capped to cited families; adversarial text verifier |
| Self-authored code unreliability (SkillRevise, Mobile-Agent-E) | Replay gate + one trace-conditioned revision; machine-checkable preconditions; low prior at birth |
| Reward-hacking the bank (ACE's stated gap) | **Decoy audit**: each learner epoch maintains one known-noise decoy card (plausible, evidence-free-by-construction, quarantined id range); any score movement on a decoy fails the learner run loudly (RoMeRL's MRT protocol operationalized) |
| Eval leakage via cards (cross-run doc) | The shingle gate runs over every card body and procedure file at admission, same spec as harvest |
| Stale cards misleading (cross-run doc staleness) | Reliability line + `last_exercised` render in the card text; staleness discounts rank; env-tagged insights carry validity windows (§3.3) |
| Memory overfitting — bound exemplars posing as knowledge (Notes-to-Self; the survey's Reflection/Experience bar) | Cards-are-abstractions rule: observations are evidence, never cards; bound-identifier lint on claim fields; `generality` field measured by the transfer clock; MDL admission test; EBG route requires an endorsed mechanism |

## 7. Measuring the learner itself

- **A/B arms, existing infra**: banked-brief arm vs frozen-brief arm (today's static
  notes) on matched task waves — the same harness that ran champion vs no-champion.
  Primary KPI: banked test-score delta; guard KPI: no regression on tasks where the
  bank is thin.
- **Re-derivation rate**: recurrences of the same technical_difficulty / re-built
  procedure across campaigns (wave-4 baseline exists: e.g. the same resolution
  diagnostic re-derived per lane). Should fall as the bank grows.
- **Decoy audit** green on every learner run (§6).
- **Bank health panel** in the learner report: cards by state, transfer-coverage per
  family, contradiction backlog, replay freshness for code procedures.

## 8. v1 scope and phasing

v1 is relbench-scoped with a benchmark-blind core (`src/kapso/learning/`): `bank.py`
(store, schema, lifecycle, index), `trajectory_bundle.py` (artifact normalization —
relbench adapter supplies paths), `learner.py` (the frame), `miners/` prompts,
`verification.py` (replay + citation resolution), `briefing.py`, `scorekeeper.py`,
`config.yaml` `learning:` block (models per role, thresholds, budgets — Rule 1; codex
xhigh for miners/curator, fable for the adversarial verifier and critic, mirroring the
evolve role split). Phasing, each an atomic commit (Rule 8):

1. Bank + card schema + citation resolution + founding cards (the 9 practices,
   back-filled evidence; ~5 pitfall-tagged insights and ~3 procedures from the
   wave-4 trace as seed). Phase-1 tests include an OKF conformance check (reserved
   fields present, indexes complete, links resolve).
2. Briefing compiler replacing the static context notes; stamping; citation contract.
   **A/B-able immediately** — founding cards vs static notes should be ≈neutral (same
   content, now scoped/cited); this validates the plumbing before any mining.
3. Learner stages T/M/C (mining into candidate cards, curated deltas) on archived
   trajectories; human review of the first bank commits.
4. Verification (replay gate, adversarial checker) + scorekeeper + lifecycle + decoy
   audit; loop-until-dry; unattended learner runs post-fetch.
5. Probe rendering into the lens planner; transfer measurement begins in earnest.

## 9. Explicitly rejected / deferred

- **Reusing the wiki/learners pipeline or its backends** — no outcome signal, wrong
  extraction target, heavyweight infra (Rules 7/10).
- **Learner-as-evolve-campaign (option B)** — evaluating a bank edit costs a campaign;
  deferred until the A/B harness gives a cheap proxy signal. The pipeline keeps the
  frame-around-CLI-sessions architecture so the upgrade is a strategy swap, not a
  rewrite.
- **A single scalar reliability score** — conflates evidence with transfer; two clocks
  (§3.3).
- **Q-learning / EMA utilities over cards** (MemRL-style) — needs feedback volume the
  transfer clock will not have for a long time; counters + rank function first, learned
  utilities revisited if event volume ever supports them.
- **Callable-tool injection of code procedures** — v2; staged-and-registered first (the ASI
  delta is real but the wiring touches every session's tool config).
- **A rich card ontology** (the wiki's five page types; an earlier five-kind draft of
  this design) — two scored types cover the observed learnings; flavor is tags,
  extensibility is schema-pack-style config (gbrain), never new built-in types.
- **LLM-judged novelty/worth anywhere in scoring** — ideation-v2's scoping law stands:
  embeddings answer "seen before?", the grader answers "worth it?", nobody else votes.
- **Cross-benchmark bank sharing (posttrain ∪ relbench)** — env insights would collide and
  transfer evidence would be unscoped; one bank per domain until a card ever earns
  cross-domain evidence.
- **Automatic retirement without visits** — a low-evidence card that was never
  exercised is `cold`, not wrong; only measured contradiction retires.

## 10. Open questions (carried honestly)

1. **Significance thresholds for events** — z·SE with what z, and how to pool repeated
   sub-threshold same-direction deltas (the "compounding small gains" problem the
   practices already flag). v1: conservative z=2, pooling deferred.
2. **Credit assignment error rate** — citation-based attribution has an unmeasured
   error rate (silent influence, cargo-cult citation). The A/B arm is the only honest
   estimator; the decoy audit bounds one direction only.
3. **Merging quantitative evidence** — two campaigns measuring the same effect at
   different magnitudes: pool, range, or per-family split? v1 stores both citations and
   renders the range; no pooling math yet.
4. **Task-family taxonomy** — scope keys presuppose families (by task type? dataset
   size? label sparsity?). v1 uses the benchmark's own family enum + dataset tags;
   expect this to be wrong in interesting ways and let contradiction-driven scope
   splits discover the real axes.
5. **When does a procedure climb the representation ladder** — the mechanism is
   settled (same card, representation upgraded through the replay gate), the *trigger*
   is not: recurrence ≥2 with compatible implementations is the DreamCoder-flavored
   default; whether a broken code representation demotes back to text or parks as
   candidate is open.

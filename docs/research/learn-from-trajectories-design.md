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

**The minimality rule (governs this document and the framework).** Always the
minimal set of sections that covers what we need — and the minimal set of modules
that does the job. Because every card is written and read by intelligent coding
agents, the default representation is a well-written prose section, never a spread
of structured micro-fields: **structure exists only where deterministic code must
parse it** (identity, scope filtering, lifecycle state, resolvable source pointers);
everything else is prose an agent writes and an agent reads. When a proposed field
could instead be a sentence inside an existing section, it is one. And everything a
card stores must be **easily understandable on sight**: a reader never decodes a
packed line or guesses what an identifier is — parts get named fields, identifiers
get names that say what they are for.

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
   honestly (coverage as its own reliability dimension, §3.3; cold states, not fake scores).
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
  study's raw-trajectory −9.5% vs insights +6.5% forward transfer). The crews see
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

### 3.1 Storage — an OKF bundle in its own repo

**The home: one standalone, private GitHub repo per domain** (`kapso-bank-relbench`;
posttrain gets its own when it joins). Not a directory in the kapso monorepo: the
learner commits autonomously, so on its own repo the git log IS the learning journal
— one transaction per commit, nothing else — and the bank and the framework keep
**two independent clocks**: a campaign pins `(kapso_commit, bank_head)` as two
coordinates in `campaign_meta`, knowledge moves between framework releases, a bad
learner run reverts without touching code, and the A/B harness (banked vs frozen
brief) is literally two refs of one repo. The repo boundary is also the
scoring-scope boundary (one bank per domain, already decided) made physical. The
monorepo keeps the learner *machinery* (`src/kapso/learning/`) and the design docs;
the trajectory archives keep the *evidence artifacts* the bank cites and replays
against. Operationally: each campaign box holds a **durable local clone** that is
the serving source — the retriever reads it at a pinned head, so campaigns
never depend on network mid-flight (unreachable remote → serve the local head,
staleness noted loudly in the brief); the learner is the single writer and pushes
after each run; boxes pull at campaign start. Config carries `bank.remote` and
`bank.local_path` (Rule 1). No shared `lib/` between procedures in v1 — if two
procedures ever share a helper, that recurrence is itself a signal to consolidate
or mint, designed when it happens.

The bank is an **Open Knowledge Format bundle**: a directory of markdown files with
YAML frontmatter, one concept per file, **file path = identity**, ordinary markdown
links between concepts forming the graph, `index.md` per directory for progressive
disclosure (this is how navigating agent sessions orient — the crews and
any future OKF consumer read the same indexes; each index line is
`- [title](path) — hero`, so a whole section scans in one screen), and a root
`log.md` as the chronological journal (one entry per learner run). OKF is minimally opinionated —
only `type` is required — so kapso's scoring state rides as producer extension fields
without breaking conformance. What conformance buys: any OKF tool can render/inspect
the bank, and third-party OKF bundles (e.g. dataset documentation) can be mounted
beside the cards as unscored reference concepts.

The repo is the system of record; everything queryable is derived — the gbrain
discipline (git markdown as truth, synced into a retrieval index; proven at 155k
pages) at our scale means one derived `index/` (a link/edge table, plus embeddings
used only by the consolidation near-duplicate shortlist), rebuilt by the frame
at merge. The **hero line is the primary retrieval text** — shown in every
shortlist: `index.md` for crews, `bank_search` for campaign sessions. The
serving path is **vectorless** (§5.1): scope filters, reliability orders, the
reading agent reranks, and only the selected k cards render in full. **Link extraction is zero-LLM**: the frame parses body
markdown links and the typed extension fields (`supersedes`, `contradicts`) into the
edge table; no model sits in the graph-construction path.

```
kapso-bank-relbench/                # standalone private repo — the OKF bundle IS the repo root
  index.md                          # root orientation for agents (OKF)
  log.md                            # learner-run journal (OKF)
  sightings.md                      # pre-card observations awaiting recurrence (§4.3)
  insights/<slug>.md                # insight cards
  insights/index.md
  procedures/<slug>/card.md         # ALL of a procedure's metadata + its method body
  procedures/<slug>/code/           #   code — present once representation reaches `code`;
                                    #   this whole dir (with card.md) is what gets staged
  procedures/<slug>/replay/         #   frame-only verification: fixture setup + entrypoint
                                    #   run + expected-metric assert; NEVER staged into
                                    #   campaigns (needs archive access; runs learner-side)
  procedures/index.md
  reference/...                     # optional mounted OKF concepts — never scored
  retired/...                       # moved, never deleted; history intact
  index/{embeddings.parquet,edges.parquet}   # derived, rebuilt at merge
```

Git is the audit trail, version store, and rollback: every learner run commits as one
reviewed transaction, and the frame tags that commit `lr_<id>` post-commit — making
run ids in card logs and evidence sources commit-addressable without storing shas (a
commit cannot contain its own hash); `bank_head` (commit sha) stamps everything
downstream, exactly as `evaluator_id` stamps evaluations today. Retrieval is **scope-first, never flat**
(the mempalace lesson): eligibility filters on scope before anything ranks within (§5.1).

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
mining produces — "run_0019's blend of run_0010 gained +0.0032 on rel-amazon" — is
an *observation*: episodic, bound, useful only as evidence. A card is the
generalization that observation supports, stated at the level of the domain
(predictive ML modelling), carrying the **mechanism** (why it works) from which its
applicability conditions derive. Two mechanical guards enforce this:

- **`scope` is a required extension field — the claimed region**: `domain`, or a
  list of `family:…` or `dataset:…` coordinates. The coordinate prefix IS the
  generality level (no separate field), region identifiers are *sanctioned* here
  (the one place bindings belong in the claim layer), and the claimed region is
  what the transfer clock measures coverage against. A domain card contradicted
  outside its home family **demotes by scope edit** (`domain` → `[family:…]`),
  versioned and logged — not retirement. Verified coverage is never authored: it
  is derived from the evidence ledger.
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
tags: [data:grouped_rows]        # searchable attributes + rendering flavor
                                 # (data-shape markers, pitfall, env) — retrieval
                                 # assist only, NEVER eligibility
timestamp: 2026-08-14T09:00:00Z
# --- kapso extensions ---
scope: domain                    # the claimed region: `domain`, or a list of
                                 # family:<id> / dataset:<id> coordinates. The
                                 # prefix is the generality level; serving
                                 # eligibility = task in scope; verified coverage
                                 # is derived from the ledger, never authored
scope_conditions: "rows share seed timestamp / session / parent entity"  # prose, judged
evidence:                        # ≥1 required; written by the learning process,
                                 # admitted only after the frame's three checks (§5.2)
  - source:                      # metadata: which learning process, which trajectory
      learner_run: lr_20260814T0900   # doubles as the bank commit tag (see log.commit)
      trajectory: rel-amazon--user-churn/20260813T015420_lane-c10
      ref: features_history.md#within-origin-ranks
      card_version: null         # the card version in force when this evidence arose —
                                 # frame-verified against the campaign's bank_head;
                                 # null = the card did not exist (independent/founding)
    verdict: confirm             # effect on the card: confirm | weaken | refine |
                                 # refute | spawn | exercise — the §3.3 disposition
                                 # vocabulary reused; one ledger event per entry
    usage: >-                    # prose: how the card figured in the process — served,
                                 # cited, probed by its own probe, or absent (an
                                 # independent rediscovery); frame-verified against
                                 # the record (§5.2)
      Not yet in the bank when this campaign ran — independent evidence: the
      lens replanner derived the within-origin normalization move on its own
      and lane 1 implemented it as a feature-widening block.
    effect: >-                   # prose: what happened, numbers included, ending
                                 # with the sentence that earns the verdict
      The block was gate-tested against the pre-widening matrix and KEPT:
      +0.0032 AUC ≈ 3.6 clustered SE on the official validation split — a
      significant in-scope agreement, so this entry confirms the card.
reliability:                     # written ONLY by the reliability assessor (§3.3),
                                 # each score frame-bounded against the event ledger
  validity: 0.80                 # in-scope agreement: do experiments agree with the
                                 # fact inside the stated scope?
  boundary: 0.55                 # how well-mapped the scope's edge is (revisions,
                                 # known boundary points, limits tested)
  coverage: 0.35                 # how much of the claimed scope has been visited
  score: 0.75                    # overall — the assessor's synthesis, not an average
  rationale: >-                  # must justify EACH dimension, state the
                                 # participation weighing applied, and end with what
                                 # would most change the score (next probe's source)
    Validity: two in-scope confirmations at 3.6 and >2 SE, both independent
    (uncontaminated), no disagreement. Boundary: untested — no contradiction
    has probed the edge; the grouped-rows condition is asserted from the
    mechanism, not yet carved by evidence. Coverage: two families visited,
    both classification on consumer datasets; regression unvisited. Most
    score-moving next: one regression-family test — hence the probe.
  state: candidate               # candidate | active | cold | retired | superseded
provenance: {version: 1}         # the rest is derivable — git history + evidence sources
log:                             # append-only, ONE entry per version, frame-written
                                 # at the version-bumping transaction (the crew authors
                                 # the change sentence; frame stamps the rest)
  - version: 1
    date: 2026-08-14
    commit: lr_20260814T0900     # the bank commit TAG (= the learner-run id; the frame
                                 # tags each commit post-commit, since a commit cannot
                                 # contain its own sha). Address this version's exact
                                 # text with: git show lr_<id>:<card path>
    change: >-
      Created from two independent instances (induction route): depth-slice
      profile [E1] + cold-entity forensics [E2].
supersedes: null                 # typed edge: lifecycle machinery branches on it
contradicts: []                  # typed edge: established tensions between active cards,
                                 # symmetric and frame-maintained (the crew names the
                                 # pair once, the frame writes both sides); consumed by
                                 # the retriever's co-serving guard and the consolidation
                                 # sweep. Every OTHER relation is a body reference (§3.2.2)
probe: >-                        # optional: how a future campaign can test this cheaply
  Ablate within-group companions of the top-5 features on one forward fold.
# --- procedure cards additionally (representation: code only) ---
# representation: text | code
# entrypoint: run_forward_gate.py     (resolved inside the procedure's code/ dir)
# preconditions: {task_families: […], requires: [features parquet, ≥2 origins]}
# replay:
#   archived_run: <trajectory_id>/runs/run_NNNN     # the fixture
#   expected_outcome: >-       # what a successful replay PRODUCES — numeric or not
#     (a reproduced metric, a gate decision, an artifact with stated
#     properties, a report with expected structure); replay/replay_test.py
#     is the executable assertion of exactly this statement — it IS the
#     codify run's registered evaluation (companion codify doc)
#   last_replayed: <date>
```

**The single-source card rule, and the body as THE FACT.** A card stores a learning
the way a person holds one: a fact, where it is true, the evidence seen, and how much
to trust it. Each lives in exactly one place. `description` is the hero (the fact in
one scannable line); the **body is the fact in full** — three to six sentences of
unified prose stating what is true, *why* (the mechanism woven into the statement,
not sectioned off — a fact must contain its because, or its scope is capped to its
instances, the EBG rule), and where it stops (limits folded into the wording), with
inline `[E1]`/`[E2]` markers into the evidence and ordinary markdown links to related
cards (the frame extracts links mechanically). `scope_conditions` holds
applicability; `evidence` holds the instances; `reliability` holds the trust record.
No body heading (the renderer adds the title), no separate claim/mechanism/limits
sections (Toulmin-itis — the working systems and the practice notes all use unified
prose), no restating of any frontmatter field: a scope refinement targets exactly one
home and a version bump means one thing. The fact is what the assessor matches
events against — semantically, hero as the compact key and body as elaboration. The
**served card is a renderer projection** assembled at brief-compile time — title +
hero + the reliability line (state and all four scores, so the reading LLM weighs a
battle-tested fact differently from a fresh or narrow one, CLIN's hedging result) +
scope + fact + evidence digest + probe — never stored. The duality that stays: `tags` (machine scope) vs `scope_conditions` (reader
scope) express the same thing at two precisions; the update crew's critic checks their
consistency. One sanctioned exception to "derivable is never stored": the **`log`
field** — one line per version, append-only, frame-written — because bundles are
consumed without `.git` (OKF tarballs, served projections) and a learner commit
touches many cards, so per-card semantic history ("what changed about THIS card,
and why") must ride with the card; the sha stays out of the file (version →
commit resolves via the line's `learner_run` id, one commit per run).

#### 3.2.1 The two sections, defined

**`insights/` — declarative memory.** Every card is a generalized, mechanism-backed
fact about how predictive modelling works in this domain. Later evidence can only
agree or disagree with it. Covers positive and negative regularities, segment
findings, dataset facts (`scope: [dataset:…]` — reuse across the database's tasks),
and env facts (validity windows). Admission: abstraction gates (lint + MDL + an
induction or EBG route), resolvable evidence, falsifiability (a card that cannot
lose cannot be scored).

**`procedures/` — procedural memory.** Every card is a generalized how-to that a
campaign can *execute*, over typed roles rather than bound entities, with the
representation ladder (text playbook → replay-verified code under one identity;
double-gated by recurrence ≥2 and the replay gate, one trace-conditioned revision on
failure). Procedures carry no textual claim to agree with — their verification is
execution: replay freshness, invocation outcomes. Code representation adds
`entrypoint`, machine-checkable `preconditions`, and `replay`; code injects at the
implementation site as staged artifacts, text renders like insights.

#### 3.2.2 Relations — a referencing language, two typed edges, an emergent vocabulary

Relations follow OKF's stance (untyped markdown links, meaning in the surrounding
prose) upgraded with a **referencing language** so a mechanical extractor can
recover the graph. The language is four lines:

1. A **reference** is an ordinary markdown link whose target is a bank-internal
   path: `[…](/insights/…)`, `[…](/procedures/…/card.md)`, `[…](/reference/…)`.
2. An **edge label** is an optional `lowercase-token: ` prefix inside the link
   text — `[justifies: cross-family portfolio effect](/insights/cross-family-portfolio-effect.md)`
   — chosen because a plain link with a text prefix is the most habitual construct
   a model writes (the CommonMark title-attribute variant is spec-legal but rare,
   and wikilink syntax breaks OKF/GitHub rendering). Labels are **free-form**: no
   registry, no validation.
3. An unlabeled reference carries the generic label `references`.
4. The **extractor** is one regex pass (zero-LLM, already the §3.1 policy): every
   reference becomes an `edges.parquet` row `(from_card, to_card, label,
   containing_sentence)` — the sentence rides along so "Referenced by" surfaces can
   render each back-link with its stated reason. Forward links are stored only in
   the author card's own body; **back-links are always derived** — no card ever
   edits another card's body to announce a relationship.

Three rules keep this from regrowing the old wiki's ontology:

- **The firewall: machinery never branches on a body label.** `supersedes`
  (lifecycle) and `contradicts` (co-serving guard + sweep) are the entire typed
  frontmatter vocabulary, kept as fields precisely because code branches on them.
  Body labels are descriptive — extractable, renderable, never load-bearing. A body
  label that needs machinery gets *promoted* to a frontmatter field as a deliberate
  design act; the extractor flags body-labeled `contradicts`/`supersedes` as
  promotion candidates for the update crew rather than triggering anything itself.
- **The vocabulary is discovered, not designed.** Free-form labels + the extractor
  mean the bank-health panel reports label frequencies; when synonyms emerge
  (`justifies`/`supports`), the consolidation sweep normalizes them as ordinary
  curation. The edge ontology emerges from use.
- **Link edits version like any body edit** — a link-only change bumps the version
  with a log line saying so ("v4 · linked to sibling — formulation unchanged"), so
  `card_version` semantics stay one rule and the log carries the nuance; the
  assessor treats formulation-identical versions as equivalent for event matching.

### 3.3 Reliability — three dimensions, an event algebra, one assessor

A card = (fact, scope), and evidence does two different things to it: it *scores* the
fact and it *carves* the scope (Mitchell's version spaces: positive examples
generalize the boundary, negative examples specialize it). A single reliability
scalar conflates three quantities a contradiction can move in **opposite
directions**:

- **Validity-in-scope** — do experiments agree with the fact *inside the currently
  stated scope*?
- **Boundary confidence** — how well-mapped is the scope's edge? A contradiction that
  gets *explained* into a scope refinement adds a known boundary point — it improves
  the card while shrinking it.
- **Coverage** — how much of the claimed scope has actually been visited (the
  transfer record; the sparse clock, honest cold states per RoMeRL).

**The event algebra.** Every attribution event receives a disposition:

| Disposition | When | Effect |
|---|---|---|
| `confirm` | agrees, in scope | validity ↑, coverage ↑ |
| `weaken` | disagrees in scope, no boundary explanation | validity ↓ |
| `refine` | disagrees in scope, and a **mechanism-backed** boundary explanation exists | scope revision: version bump; the event ledger **repartitions** under the new scope — old confirms that still fall inside keep counting, the contradicting event becomes a *boundary observation* (evidence FOR the refined card); validity recomputes over in-scope events; generality ↓, boundary confidence ↑ |
| `refute` | disagreement that breaks the mechanism itself | retirement path |
| `spawn` | a refine whose complement region carries its own regularity | sibling card minted for the complement, linked, with the contradicting event as its founding evidence (the hypothesis-bank move: unexplained residuals become new hypotheses) |

The dispositions double as the per-evidence `verdict` tag (§3.2) — one vocabulary
at both levels, each admitted evidence entry deriving exactly one ledger event.

This dissolves the paradox: after a legitimate refine, "the evidence was a
contradiction" and "the card became more reliable" are both true — the contradiction
lowered *generality*, raised *boundary confidence*, and validity-in-scope recomputes
over the repartitioned ledger, typically upward.

**The Lakatos guard** (progressive vs degenerating refinement). A `refine` is
admissible only if the revised scope is stated in **mechanism vocabulary** — the
bound-identifier lint applies to `scope_conditions`, so "except on dataset X" is
unrepresentable — and the revision mints a `probe` that could test the new boundary.
A revision that merely excuses the anomaly is tagged `ad-hoc`, capped (one per card),
and a card surviving only by carving exceptions is *degenerating* → retire. (This is
the anti-RDR stance: Ripple-Down Rules prove exception-accretion works operationally,
but identifier-keyed exceptions are exactly the memory junk this design exists to
avoid.)

**Who decides — the assessor, on a mechanical substrate.** Dispositions and the
reliability block are produced by a **reliability-assessor session** (a coding-agent
CLI call reading the card, its event ledger, and the new events) — reliability is a
judgment with reasons, not a formula. The assessor's contract: **one score per
dimension** (`validity`, `boundary`, `coverage`, each [0,1]) plus an **overall
`score`** (a synthesis, not an average — a high-validity card with untested
boundaries is not 0.9-reliable), and a `rationale` that justifies each dimension,
states the participation weighing applied — read from the usage prose: own-probe
strongest, cited discounted for expectation effects, independent replication gold
for the fact — and ends by naming what would most change the score, which is where
the card's next `probe` comes from. The substrate stays code-owned, because LLM belief-updating is
measurably miscalibrated under contradiction: the **event ledger is append-only and
frame-written** (grader-signed events with resolving citations — all §5.2/§6
defenses unchanged); the frame **bounds every score** against the ledger (a
dimension score no set of cited events supports is rejected — e.g. nonzero
`boundary` with no boundary-touching event); `refine` proposals pass the Lakatos
guard and citation resolution mechanically; and the decoy audit applies to the
assessor (a decoy card whose scores move fails the learner run). Sequential-testing discipline for probe programs
(POPPER-style e-value control instead of per-event z-tests) is the designated upgrade
path once probes run routinely.

Derived rank for retrieval = scope match × similarity × f(score, coverage,
staleness). Counters are the on-disk truth; `f` is one pure function that can evolve
without touching stored state (Rule 10; avoids formula lock-in).

**Lifecycle (all transitions mechanical, frame-owned):**
`candidate` (mined, gates pending) → `active` (admitted) → `cold` (no exercise in N
campaigns; down-ranked, never hidden — cold ≠ wrong) → `retired` (contradicts dominate
with ≥m visits, or replay breaks for code procedures, or superseded). Retirement moves the file
to `retired/` with history; a superseding card links back and inherits a *discounted*
prior (RoMeRL warm-start, with its rides-old-credibility caveat noted in provenance).
Two consolidation moves ride this same machinery: **merge** — a successor
stating the unified fact supersedes ≥2 twin parents, its founding evidence
citing the parents' ledgers **by reference** (nothing copied, nothing
rewritten; the retired files keep resolving); and **generalize** —
cross-family ledger agreement meets the induction gate by construction, but
the domain claim's *new* content is exactly the unseen families, so the
successor is **born a candidate with the unseen-family probe queued** —
upward moves are born as predictions with their test attached — coverage
claiming only the seen families, parents superseded.
Contradiction's first response is never deletion: it is **scope split** — the frame
requires a contradicting event to carry its own citation, and the update crew proposes a
narrowed scope citing both sides.

One distinction rides on tags (the mempalace validity-window idea): **env-tagged
insights expire, they are not refuted**. "lightgbm 4.x wants callbacks" going stale is
not evidence the card was wrong — a version-bump event *invalidates* (supersedes with
a validity end), leaving the transfer record intact, where a measured contradiction
*refutes*. Conflating the two would poison env cards' reliability for no informational
gain.

### 3.4 The trajectory store — structure, save, load

Trajectories cannot live where cards live: cards are KBs and clone everywhere; a
trajectory is 2–46 GB. The store is therefore **identity-addressed prefixes on
object storage plus a cache-through local mirror** — reference by identity, resolve
by store — and it never mounts into campaign sessions (bundles carry test-side
data; the trust boundary from §5 applies). Everything below is the whole design.

**Identity.** `trajectory_id = <task>/<stamp>_<lane>` (the existing naming, e.g.
`rel-amazon--user-churn/20260813T015420_lane-c10`). The ID is the path, in both
stores: `gs://<remote>/<id>/…` and `<local>/<id>/…`. Bundles are stored **unpacked
(one object per file), never as tarballs** — a point lookup into a non-resident
trajectory is a single object GET, partial materialization is a prefix copy, and
atomicity is recovered by writing the manifest last as the commit marker.

**The bundle layout.** Evidence refs are paths relative to the bundle root, and the
assembler normalizes by *gathering*, never renaming — existing ref habits
(`runs/run_0019/…`, `features_history.md#anchor`) stay valid:

```
<task>/<stamp>_<lane>/
├── trajectory.yaml            # manifest — written LAST (= the commit marker)
├── campaign_meta.json  final_report.json  campaign.log
├── features_history.md  table_information.md          # living documents
├── lens_plan_history.jsonl  experiment_history.json   # from workspace .kapso/
├── runs/run_NNNN/…            # registered evaluations, exactly as produced
├── sessions/<branch>/stream.jsonl                     # session forensics
└── evaluators/<evaluator_id>/…
```

**The completeness contract.** Required: manifest, `campaign_meta.json`,
`final_report.json`, `campaign.log`, `features_history.md`,
`lens_plan_history.jsonl`, `experiment_history.json`, and `runs/` with at least one
registered run. Optional (present when produced): `table_information.md`,
`sessions/`, `evaluators/`. A trajectory that cannot support ledger refs is not a
trajectory: `save_trajectory` raises on a missing required part (Rule 2) — no thin
saves. (This is a real change to today's harvest, which misses the workspace
`.kapso/` files and the living documents — the wave-4 forensics gap.)

**`trajectory.yaml`** — the fast answer, one small GET, no materialization:

```yaml
id: rel-amazon--user-churn/20260813T015420_lane-c10
task: rel-amazon/user-churn
created: 2026-08-13T02:11:00Z
kapso_commit: <sha>            # the framework version that ran the campaign
bank_head: <sha or null>       # what the campaign was served (null pre-bank)
outcome:
  selected_run: run_0019
  val: {roc_auc: 0.7135699}
  test: {roc_auc: 0.7155090}   # test-side data — fine here: the store is
  cost_usd: 76.12              # learner/human-side only, never campaign-mounted
  iterations: 3
inventory: {files: 865, bytes: 6712057964, hashes: sha256 per file}
```

This manifest is why no query engine ships in v1: point lookups are object GETs,
outcome lookups are one manifest read, and cross-trajectory *semantic* questions
are already the job of two existing layers — the bank (cards + evidence entries
are the cross-trajectory index, organized by learning) and the episodic experiment
store (embedding search over exemplars). Revisit trigger, stated so it is noticed:
if repeated ad-hoc scans across many resident trajectories measurably drag, a
hosted query layer goes on top of this format, unchanged.

**Save.**

```python
trajectory_id = kapso.save_trajectory(
    trajectory,           # an evolve result handle, or explicit
                          # {workspace_dir, work_dir, campaign_log} paths
    output_path=None,     # default: trajectory_store.local
    upload=None,          # default: True when trajectory_store.remote is set
)
```

Behavior, in order: **gather** (handler work dir; workspace `.kapso/`; shared-cache
living documents; the campaign log) into the layout above; **validate** the
completeness contract; **hash** the inventory and write `trajectory.yaml` last;
**register** — idempotent: an existing ID with matching hashes is a no-op, an
existing ID with mismatching content raises (never silently overwritten); **upload**
the unpacked prefix when configured, manifest last. `save_trajectory` IS the
harvest step and the evolve→learn bridge; `learn_from_trajectories()` calls it
implicitly for anything passed as a raw handle or path — the adoption-before-mining
rule's mechanism.

**Load.** Three functions, and there is no other door — the crews, citation
resolution, the assessor, replay, and humans all go through them:

```python
store.manifest(trajectory_id)              # parsed trajectory.yaml; one GET
store.resolve(trajectory_id, subpath=None) # local path; cache-through; partial
                                           # materialization by prefix (subpath)
store.open_ref(trajectory_id, ref)         # one file; single object GET when
                                           # the bundle is not resident
```

Config (Rule 1): `learning.trajectory_store: {remote, local}` — reusing the
existing artifacts bucket layout. No remote configured → the store is the local
directory and everything still resolves (local-only users).

#### 3.4.1 The mined view — the campaign's story, reassembled

Mining's output is a **derived OKF bundle inside the trajectory** at
`<trajectory_id>/mined/` — regenerable, marked derived in the manifest. Three
decisions define it:

1. **One format everywhere.** The mined view uses the same convention as the bank:
   markdown with frontmatter, prose-first, `index.md` progressive disclosure, path
   as identity, the same referencing language. Raw trajectory → mined view → bank is
   one reading and writing skill; the bank's extractor machinery applies unchanged.
2. **A flow document's sections ARE the evolve stages.** The campaign's causal unit
   is the **idea flow** — idea → selection → implementation (possibly drifted) →
   evaluation → judgment — and each flow doc carries exactly those sections, plus
   Difficulties. **Status is emergent**: a flow that died early simply has fewer
   sections (a derived frontmatter tag names it: `ideated | selected-unbuilt |
   build-failed | evaluated | judged | champion`). Rejected-at-ideation flows are
   kept — the selector's stated reason plus the untried frontier are real value.
3. **In-bundle home → citable refs.** Mined flow docs are ordinary bundle paths, so
   card evidence can cite them directly (`ref: mined/it-2/flow-3.md#evaluation`) —
   assembled, readable evidence refs with the raw artifacts one hop beneath.

```
<trajectory_id>/mined/
├── index.md              # the campaign: objective, outcome, iterations as hero lines
├── it-1/
│   ├── index.md          # the round: lens in force (+ replan rationale), parent,
│   │                     # flows as hero lines, round winner
│   ├── flow-1.md         # one file per idea flow
│   └── flow-2.md
├── it-2/…
├── strategy.md           # cross-flow: lens history as belief → evidence → re-aim
├── operations.md         # cross-flow: kills, crashes, harness incidents
└── artifacts.md          # cross-flow: what was built into the shared space
```

Flow doc anatomy — frontmatter holds only what code parses (flow id, iteration,
member/lens, derived status, node/branch/run ids, scores, validity, refs into raw);
the body is the loop: `## Idea` (verbatim, as authored) · `## Selection` (outcome +
the selector's words about this idea) · `## Implementation` (what was actually
built, and the **drift note** — build-vs-idea fidelity, deviations, why) ·
`## Evaluation` (the run sequence with scores; the internal story for multi-run
flows) · `## Judgment` (the judge's verdict, verbatim) · `## Difficulties` (the
implementor's report, verbatim).

The byproduct question resolves by **grain, not kind**: per-flow byproducts fold
into their flow (Difficulties is a section); only genuinely cross-flow channels get
their own campaign-grain documents (strategy, operations, artifacts).

Policies: **verbatim by default** — idea, selector reason, judgment, difficulties
are authored one-shot artifacts carried whole (these docs are the substrate future
evidence cites; condensation lives only in index hero lines); **drift is the only
synthesized content** in the entire mined view — everything else is reassembly —
and mining synthesizes what it needs itself — **learning never asks evolve to
author anything** (the read-only-substrate principle, §5.3): a native
`idea_fidelity` field in the implementor's closing report would be a welcome
nicety that made mined views fully reassembled, but it is a wish, never an
obligation, and no learning mechanism may depend on it. Bundle-contract
addition this design surfaces: the **ideation candidate pool and the
selector's reasoning must be archived** — today the selector runs in a
temporary worktree and its artifacts die, making rejected-at-ideation flows
unrecoverable. (Archiving more of what evolve already produces is capture of
natural production, not a job added to campaigns — it sits inside the
principle, not against it.)

## 4. The learner — one machinery, two regimes

The learner is two agentic crews around one git repo, held by deterministic
frames — the maintainer pattern (mechanical frame, scoped coding-agent interior)
applied twice. It reuses evolve's substrate — CLI adapters, prompt loader,
budget/telemetry ledger, checkpoint pattern — but not its search strategy:
learning is mining and updating **under measurement**, not score-guided search.
Six components, and a deliberate list of non-components:

| # | Component | Nature | Owns |
|---|---|---|---|
| 1 | Trajectory store (§3.4) | passive, append-only | raw bundles + their derived `mined/` views |
| 2 | Mining crew (§4.2) | agentic | nothing — writes `mined/` into the store |
| 3 | The bank (§3.1–3.3) | passive git repo, one per domain | cards, sightings, indexes, log |
| 4 | Update crew (§4.3) | agentic | nothing — commits to the bank |
| 5 | Retriever (§5.1) | deterministic code; push brief + pull tools | the serving record |
| 6 | Grader suite (§4.4, §7) | deterministic + scoped judgment calls | learner scorecards |

Non-components, per the minimality rule: no query service (`load_trajectory` +
manifests suffice, §3.4), no consolidation service (the update crew in
empty-batch mode), no probe scheduler (the probe queue is a derived index
artifact, §5.1), no extractor service (the referencing-language parser is a
library function), no daemon (every stage is an idempotent CLI step — `kapso
learn mine|update|brief|grade` — run by cron or by hand). The same components
run in **two regimes**: operating (live campaigns, §4.1) and development (frozen
corpus + exam, §4.4); the hinge between them is §4.1 step 3.

### 4.1 The operating regime — save → mine → grade → update → brief

One campaign through the cycle:

1. **Save.** The runner calls `save_trajectory` at campaign end (§3.4). Every
   learner input is normalized through the store before anything reads it
   (adoption before mining): cards only ever cite store-resolvable IDs, never
   ephemeral paths.
2. **Mine** — per trajectory, idempotent, bank-blind. The mining crew (§4.2)
   reassembles the raw bundle into the mined view (§3.4.1). Raw logs are never
   read downstream of this point except as forensics.
3. **Grade before learning — the exam-before-lesson rule.** The bank predates
   every arriving trajectory, so each one is first a genuine held-out test: the
   hindcast (§7, rung 2) runs against the new mined view *before* ingestion —
   did the bank already carry what this campaign learned the hard way? were its
   served claims confirmed? Only then may the trajectory become a lesson.
   Prequential ("test-then-train") evaluation: live operation manufactures a
   fresh scorecard point per campaign, for free, forever.
4. **Update** — batched. The update crew (§4.3) consumes all mined views since
   the last bank commit against a bank checkout and commits once.
5. **Serve.** At the next campaign launch the retriever (§5.1) pushes the
   brief from bank HEAD, exposes the pull tools to the sessions, logs every
   serving event into the serving record, and attaches at most the budgeted
   probe. The
   trajectory that returns carries exactly the evidence the bank needs — serving
   is also instrumentation; there is no separate telemetry channel.

Learner runs are serialized per bank (single writer, the same repo_lock
discipline evolve uses); mining parallelizes freely (per-trajectory,
store-local).

### 4.2 The mining crew

Designed in full in the companion prompts document
(`learn-from-trajectories-mining-prompts.md`): a self-organizing lead session
that surveys the bundle, writes the campaign-grain docs, and fans per-iteration
flow docs out to flow-writer subagents, with a read-only critic pass —
Claude-led, because self-organization runs on the CLI's native subagent
mechanism. The frame is stage/check/commit only: mined-format schema, coverage
arithmetic on stable identities (every ledger node accounted for), quote
re-grep, raw immutability by manifest hash, one repair loop. Policies that
matter downstream: verbatim by default, explicit gaps over fabrication,
degenerate-artifact detection (a judge echo like the literal "your feedback
message" placeholder is *flagged*, so §4.3 can quarantine it), drift as the only
synthesized content.

### 4.3 The update crew — mined views × bank → one commit

A lead session on a writable bank checkout, mined views mounted read-only,
eight roles as native subagents: **card-writer** (batch rows: routes and
drafts), the five **docket specialists** — card-merger, card-generalizer,
tension-resolver, expiry-sweeper, procedure-codifier — each executing exactly
one maintenance verdict, **critic** (adversarial pass over every proposed
change), and **reliability-assessor** (§3.3's one assessor). There are no typed ops: the crew edits files directly, and trust
moved from an operation vocabulary into **diff invariants** the frame validates
before commit — evidence and log fields are append-only; every version bump has
exactly one log entry and vice versa; `contradicts` lands on both cards;
retirement is a move to `retired/`, never a delete; decoys untouched; every new
evidence entry passes §5.2 admission; derived indexes recompiled. Any violated
invariant rejects the whole transaction. **Batching is load-bearing** —
induction needs siblings side by side, so the crew runs on everything since the
last commit — and the **maintenance docket rides every run**: at staging the frame seeds,
from the pre-run bank, dup-merge nominations (inclusive similarity threshold
— nomination only, never decision), unresolved `contradicts` tensions,
cross-family generalization candidates, and staleness/sightings expiries;
batch `[]` makes a run docket-only (callable standalone — gbrain's dream
cycle, now the default tail of every learning run). Twins born inside a run
surface on the next run's docket (one-run lag, accepted). No new machinery.
The docket also carries **`codify` rows** — the procedure code path
(detection by reference-closure recurrence, the codify run as
evolve-minus-ideation with a claims-judging feedback loop, ephemeral-GCP
placement, the freshness clock) is specified in the companion
**`learn-from-trajectories-codify.md`**, closing former open question §10.5.

**Routing — which card does an observation belong to?** This is where clustering
lives in this design (adopted from the uniforge M1 cluster design, which
hardened the same decision for Claude Code session logs; our economics differ —
few, rich, born-attributed trajectories — so the verdict shape survives while
the defaults shift):

- **Attributed fast path.** An observation about a served or cited card homes by
  serving-record lookup — no judgment call. The serving record is our
  attribution channel, and most evidence about existing cards arrives through
  it.
- **Categorical, mechanism-matched judgment** for the rest. Candidates come from
  index traversal (scope-first, hero lines — vectorless at bank scale); the
  card-writer rates ordinal fit per candidate — exact / strong / partial / weak
  / unrelated — judged against the card's **body and evidence ledger**, not its
  hero line (match against what the card *contains*, not what its summary
  claims), and judged on **mechanism, not vocabulary** (a shared dataset or
  feature name is not a match; uniforge's lexical mis-assignment bug was exactly
  this failure). No numeric confidence anywhere in routing: models are reliable
  at ordinal judgment and unreliable at self-reported floats; deterministic
  rules turn the categories into the verdict.
- **Asymmetric defaults, structurally encoded.** Attach on a lone
  `exact`/`strong` winner; a tie or ambiguity **spawns a candidate card**, never
  a forced attach — a wrong attach corrupts the ledger reliability is computed
  from, while a spurious candidate just dies of non-recurrence. One observation
  exercising two genuinely distinct mechanisms may evidence both cards; one
  observation that two cards could merely *describe* is ambiguity, not
  multi-membership.
- **Passing needs a rebuttal.** An observation judged card-worthless is passed
  over only after the critic has argued the strongest case *for* it — the
  adversarial second opinion on the closest thing routing has to an irreversible
  verdict. Passing is soft by construction (the mined view keeps everything),
  but a passed first sighting would silently defeat the recurrence gate, so:
- **The sightings ledger** (`sightings.md` at bank root). Unmatched single
  observations land as one-line entries: date, trajectory ref, one phenomenon
  sentence. The crew reads it every batch; a new observation matching an entry
  fires the recurrence gate, and the born card cites both sightings. Entries
  expire after a configured number of unmatched batches (memory economy). Every
  routing verdict and its level journals into the learner report, so
  over-passing is auditable and the gauntlet can regression-test routing.

**Admission** is unchanged from the card model: two mechanically distinguished
routes — **induction** (≥2 aligned instances from ≥2 independent measurements)
or **EBG** (one instance plus a mechanism an independent check endorses,
admitted at reduced evidence weight) — with the bound-identifier lint and the
MDL test on every candidate. Failure vs junk: a *measured* failure is
first-class evidence (refute and weaken are how the bank learns boundaries), but
untrustworthy telemetry — infra deaths, deadline kills, the degenerate judge
feedback mining flagged — is quarantined from scoring and mined only for
pitfalls.

**Verification and scoring close the batch.** Text evidence passes §5.2's three
checks; code procedures replay frame-side (correct + actually-invoked + effect;
one trace-conditioned revision, then parked). The assessor then walks the
ledger: per-event dispositions, per-card scores bounded against events, the
Lakatos guard, lifecycle transitions — under one invariant adopted whole from
uniforge's state controller: **an observation is evidence, never a decision.**
No single trajectory flips a card's state; transitions commit at batch end over
the whole ledger.

**Output: one bank commit tagged `lr_<id>` + the learner report.** The report
lives at `learning/runs/<lr_id>/report.md` — a run dir like every kapso
activity, never the bank repo: the bank stays knowledge-only, and its root
`log.md` carries the one-line journal entry (date, `lr_id`, headline) pointing
here. Frontmatter: run identity (`run` = the commit tag, `learner_version`,
`batch` — an empty list is consolidation mode, `bank: {before, after}`) plus
the frame-computed **`health` block** (cards by state, open contradictions,
sightings count and age, replay staleness) — mechanical state description, the
trend line across runs; every *judgment* in the report carries its rationale
(no naked tags). Body, in order: **Headline** — the lead's one-paragraph
narrative of what the batch taught; **Routing journal** — the report's reason
to exist: one entry per observation (`ATTACH`/`SPAWN`/`SIGHTING`/`PASS`, the
ordinal level where fit was judged, rationale, refs), recording the decisions
that left no diff — sightings, passes, near-ties — which the bank commit
cannot show, and giving the duplicate/stability traps their regression
surface; **Card changes** — one line per touched card, the card's own log
remaining the truth; **Rejections** — admission failures with named findings
and dispositions; **Assessor round-up** — batch-end lifecycle transitions;
**Closing assessment** — gaps noticed (feeds the next batch and
consolidation), probe-queue changes proposed, anything troubling. The full
field-by-field spec, the crew's internal stage sequence, the lead's launch
prompt, the three agent definitions, the in-flight schema contracts
(worksheet, journal, findings), and the frame contract are specified in the
companion **`learn-from-trajectories-update-crew.md`**.

### 4.4 The development regime — the learner is the candidate

Almost nothing in the crews can be verified by inspection — "this card is
abstract" is an empirical claim. So the learner is built the way kapso builds
everything: **under a registered evaluation that exists before the thing it
grades.** The archived corpus (~65 campaigns) splits **by task family and by
time** — learn-set ≈50, held-out ≈15; split keys come from manifest metadata
(campaigns are born attributed; no trajectory-grain clustering exists or is
needed). The development loop: a **learner version** (crew prompts, contracts,
thresholds — versioned candidates, never truths) runs over the learn-set → a
candidate bank → the grader suite scores it (hindcast on the held-out set +
gauntlet + axis panel, §7) → the scorecard accepts or rejects the change. Banks
are disposable here; the artifact under development is the crew. Learner
versions bank like champions — never replace a banked learner with one that
scores worse — and promotion climbs the same ladder evidence does: hindcast win
→ probe-budget exposure → sustained wins → an A/B arm against the incumbent
generation → the default learner. The A/B gate is config-governed and waivable
(`learning.ab`): a deployment may promote on hindcast strength alone and ship
straight to production — the scorecard then records the arm as not-run, never
as passed.
The frozen split stays forever as the learner's regression suite; the build
order it forces is §8. Endgame, deliberately deferred: with graders that have
earned trust, evolve itself can search learner designs against the suite (§9) —
self-hosting, gated on that trust and nothing else.

## 5. Closing the loop

### 5.1 Serving — the retriever

The retriever is the mouth of the system — the single bridge between the bank
and a running campaign; transfer physically happens here and nowhere else. It
is deterministic code over the durable local clone at a head **pinned at
launch** (campaigns never wait on the network and never see mid-flight bank
drift), and it is **vectorless**: the serving path uses no embeddings — scope
filters, reliability orders, and the reading agent reranks. (Embeddings stay
derived-index machinery for the consolidation near-duplicate shortlist; a
similarity tier can slot in behind the same protocol if eligible sets ever
outgrow a screen — a config-triggered upgrade, deferred.) Two modes over one
implementation:

- **Push — the launch brief.** The runner calls `Retriever.compile(task,
  bank_head)` — internally `bank_search(task_descriptor)` with auto-`get` of
  the top-k. Eligibility is law: **task ∈ `scope`; quarantine excluded
  (decoys, `retired/`); rank = reliability order with ledger-derived coverage
  discounts; `tags` assist ranking only, never eligibility.** k small
  (AutoGuide/DS-Agent: 2–4 per section, budgeted per kind), full text never
  clipped (Rule 6 — k caps selection, not content); each card renders as the
  served projection (id, hero, reliability line, scope, fact, citations), and
  the rendered brief replaces the hand-maintained
  `MODELLING_PRACTICE_NOTE` / `FEATURE_ENGINEERING_NOTE` blocks.
  Negative-signed insights (pitfall-tagged) for the task family ride along as
  guardrails (AutoManual's fallback routing). The brief closes with the **gap
  analysis**, gbrain-style: scope coordinates with no active cards,
  nearest-scope cards included at reduced confidence, stale cards — the
  honest "what the bank does not know" that both primes probes and stops
  false authority. Push is a **pure function of (task, bank_head)** — the
  hindcast replays it at historical heads, so no agent may ever sit inside it.
- **Pull — the bank tools.** Ideation and implementation sessions get two
  gated MCP tools, the same registry evolve already serves the episodic store
  through: `bank_search(query)` returns the scope-and-quarantine-filtered,
  reliability-ordered **hero shortlist** — at bank scale the whole eligible
  set; the calling agent is the reranker, reading hero lines exactly as crews
  read `index.md` — with a gap note when the eligible set is thin, never
  padded with unmarked near-misses; `bank_get(card_ids)` returns full served
  projections. The **feedback judge never gets the tools** — a judge reading
  the bank couples the evaluator to the thing under evaluation (§6). Tool
  exposure carries a per-benchmark off-switch in config (external harnesses
  may flag agent tools — the PostTrainBench lesson).
- **The serving record — one format, two exposure levels.** Every serving
  event is logged: the push stamp and the pull log (query, shortlist shown,
  gets), per card with scope-match, rank components, and card version —
  gbrain's `--explain` applied to serving. `searched` (hero shown) and `got`
  (full card rendered) are distinct levels: **attribution binds to `got`**;
  hero-only exposure is recorded but weak. The record is the ground truth for
  fast-path routing (§4.3), the usage-consistency checks (§5.2), and the
  hindcast serving dimension. The **co-serving guard** runs in both modes: a
  returned set containing a `contradicts` pair always names the tension
  ("these disagree on X; the boundary is unresolved — treat as contested"),
  never silently side by side.
- **Code-representation procedures** stage into the shared artifact workspace
  (`shared_cache/procedures/<slug>@v<N>/`, version-pinned) as `card.md` + `code/`, plus
  the registry entry and a provenance README — "verified exemplars: adapt or invoke;
  replay-tested against <run>". `replay/` never stages: verification is frame-side and
  archive-adjacent. (Callable-tool wiring, ASI's +3.7 injection-site result, is a v2
  upgrade; staged-and-registered is v1.)
- **The citation contract**: ideation/implementation prompts gain one paragraph — a spec
  that uses a card cites `[card:<id>]`; the feedback judge's verdict template gains a
  `cards_load_bearing` field. This is the attribution substrate, and it is honest about
  its limits (§6: silent influence is measured by A/B, not by attribution).
- **Stamping**: `campaign_meta.json` gains `bank_head`; every attribution event later
  binds to the exact card versions the campaign read.
- The lens planner/replanner prompts render **cheap probes** from the **probe
  queue** (`index/probe-queue.md`) — a derived artifact the update crew recompiles
  each batch, ranking every card's open probe by **value of information**:
  uncertainty × serving exposure first (heavily served, thinly verified), then
  boundary tests that would split a scope, then cards whose lifecycle decision is
  blocked pending measurement. The retriever attaches (push-side) at most the
  configured budget per campaign (`learning.probe_budget`, Rule 1 — e.g. one probe
  slot; a hard cap, so learning never cannibalizes doing), each rendered as
  "unverified on this family; `probe:` says how to test it in one fold" — steering
  a bounded slice of the existing exploration budget into deliberate,
  pre-registered card validation. This is the novel piece: the bank does not wait
  to be exercised, it *purchases the observations that most reduce its
  uncertainty*, and the replanner's evidence-driven re-aiming (proven in wave-4)
  is the natural carrier. **The outcome comes home with no evolve job** (§5.3):
  a probe that ran is just an experiment in the campaign's natural artifacts,
  so its measurement surfaces as an ordinary settlement; the serving record
  carries the attached probe spec, and the update crew matches the settled flow
  against that spec to award the **prospective** label — the match
  critic-verified, and an ambiguous match downgrading to the incidental class
  (errors fall conservative). Uptake is voluntary: an ignored probe stays
  queued at its VoI rank, and the uptake rate lands in the health panel.

### 5.2 Evidence — anatomy and mechanical checks

An evidence entry has exactly four parts, per the minimality rule: **`source`**
(metadata: which learner run created it, from which trajectory, pointing at which
artifact), **`verdict`** (the one tag: what this evidence did to the card),
**`usage`** (prose: how the card figured in that campaign's process — served in the
brief, cited by a spec, tested by its own probe, or absent so the campaign
rediscovered it independently; what the process actually did with it), and
**`effect`** (prose: what happened, numbers included, ending with the sentence that
earns the verdict). The writing is an intelligent agent's; the trust is the frame's.
Admission runs three checks:

1. **Source resolves** — the trajectory exists, the ref exists inside it, and the
   effect's quoted numbers re-grep in the referenced artifact.
2. **The usage story is consistent with the record** — every participation claim
   the usage prose makes is verified against ground truth: a claimed probe → a
   settled flow whose experiment matches the probe spec the serving record
   shows attached (the match critic-verified; ambiguity downgrades the entry
   to the served class); a claimed citation → the serving
   record carries the card at the stamped `bank_head` AND `[card:<id>]` greps in the
   spec/changes.log; claimed serving → the serving record; claimed independence →
   the card absent from the serving record or postdating the campaign (all founding
   evidence is independent by construction). The `source.card_version` is
   verified the same way: a claimed version must match what the stamped `bank_head`
   actually served; `null` requires verified absence. The checker's conclusion is
   recorded ledger-side; a narrative or version the record cannot support is
   rejected with a named finding.
3. **Effect is grader-backed and supports the verdict** — outcomes trace to
   registered evaluations, judged significant by the campaign's own clustered-SE
   machinery, and the entry's `verdict` must be earnable from them: `confirm` /
   `weaken` / `refute` need significant in-scope agreement / disagreement;
   sub-threshold effects may only carry `exercise`; `refine` must include the
   boundary explanation and passes the Lakatos guard (§3.3); `spawn` must name the
   sibling card it founds. A verdict the numbers cannot earn is rejected with a
   named finding.

From each admitted entry the frame derives a ledger event (append-only, code-owned —
the substrate §3.3's assessor works on). How usage should weigh on the assessor is
guidance, not schema: the card's own probe is the strongest test (prospective); a
cited use carries expectation effects (the campaign knew the predicted sign); a
**pulled** card (`bank_get` on the session's own query) is demand evidence —
stronger engagement than passive serving, weaker independence (the session went
looking for the answer it retrieved); an
uncited-but-served test speaks to the fact more than to the serving; and evidence
from a process that never saw the card — all founding evidence included — is an
uncontaminated replication, the strongest support a fact can get, while saying
nothing about whether *serving* the card helps (the A/B arm remains the only
unbiased estimator of that). There is deliberately **no** "injected into a winning
campaign" event — that is the memory-reward trap, and it stays excluded even though
it would move scores faster.

### 5.3 What changes in evolve

**Evolve is read-only substrate for learning.** Campaigns produce what they
naturally produce; learning reads, mines, and settles entirely on its own
side, and never adds an output obligation or workflow step to the campaign
loop — the probe-outcome note this design once considered is the canonical
counterexample, killed in favor of learning-side settlement matching (§5.1).
What serving touches is exactly this: the push brief rides **additively** on
the two static context constants — the notes are the permanent base every
campaign gets, the brief appends after them (with a one-line
measurements-arbitrate note for conflicts), so an A/B candidate arm measures
the bank's marginal value over the incumbent's full context, never its worth
as a replacement; the `bank_search`/`bank_get` tools join the gated-MCP
presets for ideation/implementation sessions (never the judge; per-benchmark
off-switch in config); probe rendering is one prompt paragraph *offering* a
suggestion; `bank_head` in campaign meta is stamped by the retriever, not by
sessions. **The one sanctioned convention** is the citation contract: a
paragraph asking specs to mention `[card:<id>]` where a served card shaped a
decision, plus the judge template's `cards_load_bearing` field — a writing
habit inside prose the sessions already author (and grounding that serves the
campaign itself), not an output artifact. It buys the `cited` rung of the
attribution ladder (probe-settled ≻ cited ≻ served ≻ independent); dropping
it would cost that rung and nothing else. No orchestrator/search changes — the learner runs *after*
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
| Stale cards misleading (cross-run doc staleness) | Reliability line + ledger-derived last-exercised render in the served card; staleness discounts rank; env-tagged insights carry validity windows (§3.3) |
| Memory overfitting — bound exemplars posing as knowledge (Notes-to-Self; the survey's Reflection/Experience bar) | Cards-are-abstractions rule: observations are evidence, never cards; bound-identifier lint on claim fields; the claimed `scope` region measured by the transfer clock; MDL admission test; EBG route requires an endorsed mechanism |

## 7. Measuring the learner itself — the instrument ladder

What "good" means was fixed before mechanism: **transfer is terminal** — past
experience makes future, unseen work better, and never worse where it is
irrelevant — and the remaining axes are balance points between named pathologies
(generalization between rote and superstition; calibration between hedging and
overclaiming; revision between dogma and gullibility; plus coherence,
availability, memory economy, knowledge duality, provenance). The rule that
makes these engineering instead of hopes: **every axis is measured by some rung
below, or it is not a claim.** Four rungs, roughly three orders of magnitude
apart in cost — cheap rungs iterate the learner, expensive rungs certify it
(evolve's own fidelity ladder, one level up):

| Rung | Cost | Axes measured | Instrument |
|---|---|---|---|
| **1 — audits + gauntlet** | free; every learner run | provenance, coherence, revision, memory economy | diff invariants (the decoy audit of §6 among them, enforced per commit); the duplicate trap (a disguised clone of a real mined view must add nothing — the ~20× echo shape of our corpus, made a test); the stability trap (two runs on the same batch must agree in substance); implant and red-team specs deferred until an incident earns them |
| **2 — hindcast** | minutes; no GPU | extraction, generalization-as-prediction, calibration, availability | freeze the bank; for a held-out trajectory compile the brief it *would have* served, then score against what that campaign actually measured: discoveries already banked (extraction — the re-derivation rate, inverted), bank claims settled by the campaign's graded outcomes inside claimed scope (calibration points for free), relevant cards actually **in** the brief at budget (availability: miss rate and noise rate) |
| **3 — probes** | capped slice of live campaigns | generalization-as-causation for single claims; boundary carving | the probe queue (§5.1): pre-registered, VoI-ranked, budget-capped |
| **4 — A/B arms** | full campaigns; config-waivable | transfer — terminal, incorruptible | candidate-head arm vs incumbent-head arm (two refs of one repo; the first generation's incumbent is the empty brief, i.e. static notes only — both arms always carry the permanent notes, arms differ only in the appended brief), **same-task pairs** — the champion-vs-no-champion harness; primary KPI: paired test-score delta ± SE; guard KPI: no regression where the bank is thin |

Rung 2's honest limit, stated once: the hindcast measures **predictive
alignment, not causal influence** — the held-out campaign was never steered by
the compiled brief. That is what rungs 3–4 exist for, and why rung 4 is the one
number the learner can never be optimized against directly.

Scoring semantics are specified once, in the companion
**`learn-from-trajectories-grader-scoring.md`**: the measurement idiom (a few
named dimension scores + one rationale — agent-written, frame-bounded by
marker-count corridors, agent-read; decisions stay categorical, measurements
are scored), the hindcast dimensions (foresight, accuracy, serving) with their
marker vocabulary, corridors, anchors and null rules, the scorecard roll-up
(paired deltas ± SE on the shared held-out set, the calibration table, the
verdict block), the gates-dominate-scores contract and minimal two-trap gauntlet
(duplicate, stability — verdict + rationale, never naked tags), the split
manifest with its rotation discipline, the settlement→evidence lift, and the
grader crew's process — lead, parallel history-blind report-writers, the
adversarial verifier (NOVEL re-search as its first duty), the
scorecard-assessor.

**The curve never stops.** Exam-before-lesson (§4.1) makes every live trajectory
a fresh rung-2 point before it is ingested, so the scorecard is a running series
rather than a one-time grade, and the frozen corpus split remains the learner's
regression suite forever. The learner report carries the bank health panel per
run: cards by state, ledger-derived coverage per family, contradiction backlog,
replay freshness, sightings age, probe uptake rate.

**Three diseases, named counters** — the same three evolve's governance already
fights: **Goodharting rung 2** (a crew version overfits the held-out families) →
rotate holdouts; rung 4 stays terminal. **Contamination** (learn-set siblings of
a held-out family leak its lessons) → split by family and time, never by task.
**Noise floors** (≈15 exam points → wide intervals) → every scorecard number
ships with its SE, and "within noise" is a recorded verdict, never rounded up to
a win.

## 8. v1 scope and phasing

v1 is relbench-scoped with a benchmark-blind core (`src/kapso/learning/`):
`bank.py` (store, schema, lifecycle, diff invariants, index), `trajectory_store.py`
(§3.4: bundle assembly, save/manifest/resolve/open_ref — the relbench adapter
supplies gather paths), `learner.py` (the crew frame: batch driver,
launch/stage/check/commit, evidence admission, event ledger), `crews/` (mining
and update instruction documents + agent definitions; both drafted in their companion docs), `graders.py` (hindcast runner, gauntlet,
scorecard), `verification.py` (citation resolution), `codify.py` (the
seeder, the codify-run driver, freshness re-runs), `retriever.py`,
`reliability.py` (assessor frame), and a config `learning:` block (models per
role, batch size, probe budget, sightings expiry, thresholds — Rule 1; crews are
Claude-led, since self-organization needs the CLI's native subagents, with the
critic on a second model where it earns its cost). Phasing follows §4.4's forced
order — **the graders exist before the thing they grade** — each phase an atomic
commit (Rule 8):

1. **Store + corpus import** — save/load/manifest against the §3.4 contract;
   normalize the ~65 archived campaign bundles into the store.
2. **Mine the corpus** — the mining crew over every archived trajectory; human
   review of the first mined views; the mining report's coverage arithmetic is
   the acceptance gate. The corpus is now simultaneously curriculum, benchmark,
   and first real input.
3. **Grader suite v0** — family+time split (learn ≈50 / held-out ≈15), hindcast
   runner, the two-trap gauntlet (duplicate, stability), split manifest v1,
   scorecard with SEs. Nothing exists yet for it to grade except founding banks — that is
   the point: the exam predates the student.
4. **Bank + update crew** — found the repo (the 9 practices back-filled with
   evidence; ~5 pitfall insights and ~3 procedures from the wave-4 trace as
   seed); update-crew v1 over the learn-set; iterate crew versions against the
   scorecard (§4.4); human review of the first bank commits; keep-best banking
   of learner versions.
5. **Serving** — the retriever augments the static context notes (push brief
   appended after them + `bank_search`/`bank_get` tools); stamping + citation
   contract; exam-before-lesson goes live on every new campaign. A/B-able
   immediately: notes+founding-brief vs notes alone should be ≈neutral or
   better (the founding cards overlap the notes' content), validating the
   plumbing before any mining takes credit.
6. **Probes + arms** — probe-queue rendering into the lens planner under
   `learning.probe_budget`; the first A/B arm; transfer measurement in earnest.

## 9. Explicitly rejected / deferred

- **Reusing the wiki/learners pipeline or its backends** — no outcome signal, wrong
  extraction target, heavyweight infra (Rules 7/10).
- **Self-hosting (evolve searching learner designs)** — the blocker that deferred
  "learner-as-evolve-campaign" (evaluating a learner change used to cost a
  campaign) is dissolved by the hindcast, and §4.4 already runs the same
  discipline by hand: versioned learner candidates measured against a registered
  grader suite. Handing that loop to evolve itself stays deferred until the
  graders have earned trust — a search is only as sane as its grader.
- **Trajectory-grain clustering** — the problem uniforge's M1 solves (route
  unlabeled, heterogeneous session logs to a skill) does not exist here:
  campaigns are born attributed (manifest task/family/dataset), so clustering
  lives one level down, at observation→card routing (§4.3), and the exam split
  needs only manifest metadata.
- **A single scalar reliability score without a rationale or scope semantics** —
  conflates validity, boundary confidence, and coverage; a contradiction that should
  refine scope would instead just bleed score (§3.3).
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

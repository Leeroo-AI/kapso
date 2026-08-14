# Learn from Trajectories — grader scoring

Companion to `learn-from-trajectories-design.md` (§4.4, §7): the complete
scoring semantics for the grader suite's measurement artifacts — the hindcast
report and the scorecard. The main doc owns the architecture (what the rungs
are, when they run); this doc owns what every number and marker **means**, how
it is written, how the frame bounds it, and how it aggregates. The gauntlet is
specified in §2.3 in its minimal two-trap form; the split manifest in §3.

---

## 0. The scoring philosophy

**Decisions are categorical; measurements are scored.** The design uses two
scoring modes on purpose, and the split resolves the "LLMs can't self-report
floats" tension (the uniforge M1 lesson) without giving up comparability:

- A **decision** — route an observation, attach vs spawn, keep vs pass — is
  consumed by deterministic rules and is often costly to reverse. Decisions use
  ordinal categories (`exact/strong/…`) turned into verdicts by code; no
  numeric confidence exists anywhere in a decision path (design doc §4.3).
- A **measurement** — how much did the bank know, did its claims hold, was the
  brief right — is consumed by an agent with agency (the scorecard assessor,
  the accept/reject reader, a human). Measurements need to be comparable
  across trajectories and bank versions, which raw tallies never are ("7
  discoveries here, 3 there" diffs to nothing). Measurements use **the scoring
  idiom**.

**The scoring idiom** (one idiom everywhere): a *small* set of named dimension
scores, plus one overall, plus one rationale —

```yaml
<block>:
  <dimension>: 0.00–1.00 | null      # each dimension answers ONE stated question
  score: 0.00–1.00 | null            # overall — a judgment, never an average
  rationale: >-                      # the part numbers can't carry
    …
```

— **agent-written, frame-bounded, agent-read.** The writing agent judges; the
frame tethers every score to countable evidence (the corridor, §0.3); the
reading agent consumes numbers *and* rationale together. The idiom instances:

| Instance | Object measured | Dimensions | Where defined |
|---|---|---|---|
| Card reliability | one card vs its whole evidence ledger | validity, boundary, coverage | design doc §3.3 |
| Hindcast block | one bank version vs one held-out campaign | foresight, accuracy, serving | this doc §1 |
| Scorecard verdict | one learner version vs the held-out set | aggregated dimensions + calibration + gates | this doc §2 |

Same shape, learned once; different objects, never conflated.

One discipline underneath both modes: **no naked tags.** Every agent-written
judgment ships as a categorical verdict *or* a score, **plus one rationale** —
`{verdict, rationale}` for gates and routing verdicts, dimension scores +
`rationale` for measurements. The frame parses the tag or number; the reading
agent gets both. A verdict without its rationale is rejected like a score
outside its corridor.

### 0.1 Scale

Scores are floats in **[0, 1], two decimals maximum** — precision beyond two
decimals is noise and the frame rejects it. The scale is anchored per
dimension (each dimension's section gives its 0 / 0.5 / 1 anchors); a score is
meaningful only relative to its dimension's question and anchors, never as a
free-floating quantity.

### 0.2 Null is a verdict, not a gap

`null` means **"no evidence base to score"** — the dimension's denominator is
empty (no discoveries were made; fewer than `min_settlements` claims settled;
nothing was relevant and nothing served). `0` means the evidence base exists
and the bank failed it completely. The two must never be confused: a bank
scoring `null` on accuracy is untested; a bank scoring `0` is wrong. A `null`
dimension always gets a rationale sentence saying why. Fabricating a number
where the base is empty is a frame-rejected report.

### 0.3 The corridor — bounded judgment

Every dimension has a **corridor**: the frame computes a crude center from the
body's marker counts and rejects a score outside `center ± band` (band:
`learning.graders.score_band`, one config value for all dimensions). The
corridor is **a tether, not the definition** — the definition is the
dimension's *question*, and the agent's judgment moves the score *within* the
corridor to weigh what counting can't (a missed *central* discovery matters
more than a missed footnote). What the corridor guarantees is that judgment
can never contradict the evidence layer: a foresight of 0.9 over a body
listing one hit in seven is not a generous judgment, it is a false report.

### 0.4 Ownership

The report body and scores are written by the grader's judgment session; the
enumeration inputs (what the brief would have been, which cards were eligible,
the mined view's flow set) are staged by the deterministic frame before the
session starts; the frame validates after (marker grammar, ref resolution,
corridors, null rules) and rejects the whole report on any violation with a
named finding. No score is ever computed by code alone, and no score is ever
accepted on agent authority alone.

---

## 1. The hindcast report

One report per (held-out trajectory × bank version). Inputs staged by the
frame: the trajectory's mined view; the bank checkout at `bank_head`; the
**would-have-been brief** — compiled by the *real* retriever's push path (design
doc §5.1) at that head for this task, with its serving record. The report has
three body sections (the evidence layer) and one frontmatter `hindcast:` block
(the scoring layer). Frontmatter carries only what code parses: identity
fields, the idiom block. Everything else is prose.

### 1.1 The evidence layer — markers

Body entries follow one grammar: `- **<MARKER>** — <name>: <prose story>
[refs]`. Markers are the **only** structure in the body; the frame greps them
per section to compute corridor centers, and every entry's refs must resolve
(mined-view anchors exist; quoted numbers re-grep; named cards exist at
`bank_head`). The vocabulary — twelve markers, three sections:

**Extraction section** — one entry per *discovery*. A discovery is a lesson
the campaign **paid for**: a KEPT-with-measured-delta feature family, a
mechanism the judge/feedback derived, a procedure built and validated — not
every artifact touched. The enumeration comes from the mined view (ledger
outcomes, judgment sections, difficulties), and the entry must cite where the
campaign paid (which iterations/flows).

| Marker | Meaning |
|---|---|
| `HIT-SERVED` | the bank carried it **and** the brief served it |
| `HIT-UNSERVED` | the bank carried it; the brief left it out (counts for foresight; charged against serving as a `SERVE-MISS` there) |
| `MISS-UNCARDED` | the lesson was **derivable from the learn corpus** — the entry must cite the learn-set trajectory/flow that taught it — but the crew never carded it. The actionable extraction failure. |
| `MISS-NOVEL` | genuinely first-sighted here; no learn-set source exists (the search is attested in the entry). Not a failure — excluded from the foresight corridor, reported as the novel share. |

**Claims-settlement section** — one entry per bank claim this campaign's
measurements bear on. The eligible claim set is every active card whose scope
covers this task (the same eligibility the retriever uses), plus any card the
campaign's numbers happen to touch. The settlement standard is the evidence
standard (design doc §5.2): outcomes trace to registered evaluations and are
judged by the campaign's own clustered-SE significance machinery — an
under-threshold result settles nothing.

| Marker | Meaning |
|---|---|
| `AGREED` | in-scope prediction, significant agreement (delta + ref in entry) |
| `CONTRADICTED` | in-scope prediction, significant disagreement — the accuracy failure |
| `OUT-OF-SCOPE` | the campaign's numbers touch the claim outside its stated scope — recorded (boundary information for the update crew), **never scored** |
| `THIN` | bears on the claim but under significance — recorded as exercise-grade, excluded from the accuracy corridor |

`AGREED`/`CONTRADICTED` entries are written **liftable**: each carries source,
the measured delta with its ref, and an earnable verdict, so the update crew
can lift it into a §5.2 evidence entry without recomputation (§4).

**Serving section** — one entry per serving event or failure, judged in
hindsight (a card is *relevant* if it bears on what the campaign actually did
or needed — an agent judgment, rationale-backed in the entry).

| Marker | Meaning |
|---|---|
| `SERVED-USED` | served, relevant, and visibly taken up (cited, or its lesson applied without re-derivation) |
| `UPTAKE-FAIL` | served and relevant, yet the campaign re-derived it from scratch — the brief was right and unheard |
| `SERVE-MISS` | banked and relevant, absent from the brief at budget — a ranking/eligibility failure (every extraction `HIT-UNSERVED` reappears here) |
| `SERVE-NOISE` | served and irrelevant — budget spent on the wrong card |

### 1.2 `foresight` — did the bank already know it?

**Question:** how much of what this campaign learned the hard way did the bank
already carry at `bank_head`?

- **Evidence base:** the Extraction section. **Corridor center:**
  `(HIT-SERVED + HIT-UNSERVED) / (HIT-SERVED + HIT-UNSERVED + MISS-UNCARDED)`.
  `MISS-NOVEL` is excluded — foresight measures the learner against what was
  *learnable*, not against novelty — but the rationale must state the novel
  share (a campaign that is mostly novel says the corpus is thin there, and the
  scorecard reader needs to know).
- **Judgment within the corridor:** weigh discoveries by what they cost the
  campaign and what they were worth — a miss on the discovery the campaign
  spent half its budget on outweighs three footnote hits.
- **Anchors:** `0` — the campaign re-learned everything learnable from
  scratch; `0.5` — the bank carried roughly half the paid-for lessons, or all
  the minor ones and not the central one; `1` — every learnable discovery was
  already banked (the campaign paid only for novelty).
- **Null:** the campaign produced no learnable discoveries (denominator
  empty).

### 1.3 `accuracy` — did its claims survive contact?

**Question:** where this campaign's measurements bear on bank claims *inside
their stated scope*, did the claims hold?

- **Evidence base:** the Claims-settlement section. **Corridor center:**
  `AGREED / (AGREED + CONTRADICTED)`. `OUT-OF-SCOPE` and `THIN` never enter —
  an out-of-scope contradiction is boundary news, not an accuracy failure; an
  insignificant delta is not news at all.
- **Judgment within the corridor:** weigh centrality (a contradicted hero
  claim vs a contradicted edge case) and margin (barely-significant
  disagreement vs a reversed sign at 4×SE).
- **Anchors:** `0` — every settled in-scope claim broke; `0.5` — claims hold
  about as often as not: the bank's in-scope confidence is uninformative;
  `1` — every settled claim held.
- **Null:** fewer than `min_settlements` settled claims
  (`learning.graders.min_settlements`) — thin evidence is reported as thin,
  never scored. The rationale states the count either way; per-report accuracy
  is *expected* to be thin, and its real consumer is the scorecard's
  calibration table (§2.2), which pools settlements across reports.

### 1.4 `serving` — would the brief have delivered?

**Question:** had this campaign launched with this bank, would the compiled
brief have put the right knowledge in front of it — and would that knowledge
have landed?

- **Evidence base:** the Serving section. **Corridor center:**
  `hit_rate × (1 − noise_share)`, where
  `hit_rate = SERVED-USED / (SERVED-USED + UPTAKE-FAIL + SERVE-MISS)` and
  `noise_share = SERVE-NOISE / (SERVED-USED + UPTAKE-FAIL + SERVE-NOISE)`.
- **Uptake failures are charged here, deliberately:** served knowledge that
  the campaign re-derived anyway indicts the serving pipeline — rendering,
  salience, the citation contract — not the knowledge. Foresight already
  credited the fact itself; serving owns whether serving *worked*. (This is
  the "a hit can contain a failure" rule: `HIT-SERVED` + re-derivation =
  foresight credit + `UPTAKE-FAIL` here.)
- **Judgment within the corridor:** weigh what the misses and noise cost — a
  `SERVE-MISS` on the card that would have saved two iterations outweighs
  noise that wasted one context slot.
- **Anchors:** `0` — the brief delivered nothing relevant (or nothing at all
  where relevant cards existed); `0.5` — roughly half the deliverable
  knowledge arrived and landed, or it arrived diluted by noise; `1` — every
  relevant banked card was served, taken up, with no noise.
- **Null:** no relevant cards existed *and* the brief was empty — nothing to
  deliver, nothing mis-delivered. (Relevant cards existing + empty brief is
  `0`, not null.)

### 1.5 `score` — the overall

**Question:** would this bank have made this campaign better?

- **Not an average.** The agent weighs the dimensions by what bound this
  campaign: a foresight miss on the central discovery can sink a report whose
  other dimensions are high; a single catastrophic in-scope contradiction can
  dominate everything.
- **Corridor:** `[min(non-null dims) − band, max(non-null dims) + band]`,
  clamped to [0, 1] — the overall may not escape the range its own dimensions
  span by more than the band.
- **Rationale duties** (frame-checked for presence, agent-judged for content):
  name the binding factor; state the novel share; flag thinness (few
  settlements, few discoveries); name any uptake failures. The rationale is
  what the scorecard assessor actually reads — write it as the one-paragraph
  truth of the report.
- **Null:** only when every dimension is null (the campaign taught nothing
  learnable, settled nothing, needed nothing — in practice: a degenerate
  trajectory; the report says so).

### 1.6 Report admission — the frame's checks

A hindcast report is admitted only if: (1) marker grammar parses and every
marker is from the vocabulary; (2) every ref resolves and quoted numbers
re-grep in the cited artifact; (3) every `MISS-UNCARDED` cites a resolving
learn-set source, every `MISS-NOVEL` attests its search; (4) every score sits
in its corridor and respects the null rules; (5) `AGREED`/`CONTRADICTED`
entries are liftable (source + delta + earnable verdict); (6) the rationale
discharges its duties (§1.5). Rejection is loud, with the named finding — a
rejected report is a grader bug or a lying report, and both must surface.

---

## 2. The scorecard roll-up

One scorecard per (learner version × grader run). It aggregates the per-report
hindcast blocks, pools settlements into the calibration table, records the
gauntlet gates, and closes with the same idiom one level up.

### 2.1 Dimension aggregation

Per dimension: **mean ± SE over per-trajectory scores** (nulls excluded,
`n` and null-count reported), plus the per-trajectory values themselves —
at n ≈ 15 the reader must see the distribution, not just its mean.
**Comparisons between learner versions are paired**: both versions are graded
on the *same* held-out set, so the statistic is the per-trajectory delta,
mean ± SE of deltas — the pairing is what makes small-n comparison honest.

### 2.2 The calibration table

Accuracy's real home. Pool every settled claim across all reports; bucket by
the **claimed** reliability of the card at serving time (the overall score in
the stamped `bank_head`); report realized agreement per bucket:

```
claimed [0.7–1.0]:  realized 11/12 agreed   (well calibrated)
claimed [0.4–0.7):  realized  5/9  agreed   (as claimed)
claimed [0.0–0.4):  realized  3/4  agreed   (underconfident — cards better than they say)
```

A bank is calibrated when high-claiming cards agree more often than
low-claiming ones and each bucket's realized rate is compatible with its
claim. Below `learning.graders.calibration_min` pooled settlements the table
is `null` with the count — calibration is the slowest number in the suite and
is reported as absent until it exists.

### 2.3 The gauntlet — gates, not scores

**Why it exists.** A learner's mistakes on real data are invisible — real data
carries no answer key, so the machinery (evidence accounting, admission gates,
reproducibility) can be broken while hindcast scores look fine. The gauntlet
buys an answer key by construction: run the learning step on **controlled
input whose correct handling is known in advance**, and diff the result.
Affordable because a learning run is agent-minutes over markdown on a sandbox
checkout — no GPU, no campaign. Verdicts are **PASS/FAIL + rationale** (no
scores: these are integrity properties whose partial credit is meaningless — a
bank that inflates on duplicated evidence is broken at 10% exactly as at
100%), and **any FAIL rejects the learner version regardless of every number
on the scorecard**.

**The minimal battery — two traps**, both grounded in behavior we have
observed, both = one extra controlled learning run + a diff:

- **Duplicate** *(evidence independence — our corpus repeats itself by
  design: ~20× log echoes, cumulative per-lineage changes.log, the same
  result narrated in ledger, judge, and postmortem).* Clone one real mined
  view — reworded, same run ids and numbers — and run the crew on
  {original + clone} vs the control {original}. Known right answer: identical
  banks; the clone adds zero independent information. FAIL: the
  ≥2-independent-instances gate fired on original+clone, or any score differs
  from control.
- **Stability** *(LLM nondeterminism — how much of the bank is dice?).* Run
  the same crew twice on the same batch from the same starting bank; diff the
  two banks **in substance** (touched-card set, verdicts, lifecycle
  transitions, scores within `stability_tolerance`; prose may differ). FAIL:
  a card exists in one run and not the other, or scores move beyond
  tolerance — then version-vs-version comparisons are measuring noise, and
  nothing else on the scorecard can be trusted.

**Demoted and deferred.** The decoy is not a gauntlet member: decoy cards are
enforced by the standing §4.3 diff invariant on every commit (zero extra
machinery), with the §6 trust-table row unchanged. Implant (doctored false
lesson) and red-team (contradiction extraction from compiled briefs) are
specified but **deferred until an incident earns them a slot** — minimality
over completeness.

**The artifact.** One `gauntlet.md` per grader run, beside the hindcast
reports. Frontmatter: per-trap `{verdict, rationale}` plus the rolled-up
`{verdict, rationale}` (all code parses is the verdicts); body: one section
per trap — construction refs (fixtures live in the grader run dir, never the
trajectory store) and, on FAIL, the stored proof (the actual diff/patch).

```markdown
---
learner_version: crew_v3
bank_head: lr_20260817T2100
batch: [rel-hm/user-churn/20260819T0300_lane-a1, rel-avito/ad-ctr/20260820T1100_b2]
gauntlet:
  duplicate:
    verdict: PASS
    rationale: >-
      Byte-empty bank diff vs control: the crew flagged the colliding run ids
      and routed the clone's observations as already-seen. Independence
      accounting held on the exact echo shape our corpus produces.
  stability:
    verdict: FAIL
    rationale: >-
      Run B minted a card run A never saw (session-gap-recency, from a THIN
      borderline observation) and thin-history's score moved 0.55→0.71 across
      runs at tolerance 0.10 — borderline observations are landing on
      whichever side the dice roll; admission thresholds too sensitive.
verdict: FAIL
rationale: >-
  Stability alone rejects crew_v3: score movement beyond tolerance means the
  comparison against crew_v2 would be measuring noise. The duplicate pass is
  real progress over v2 and carries forward.
---

## duplicate — construction + proof
Cloned mined/it-2/flow-3.md of rel-hm/user-churn (reworded, same run ids and
numbers); fixture: fixtures/duplicate/; control-vs-trap diff: empty.

## stability — construction + proof
Same crew, batch, starting bank, twice; substance diff: diffs/stability.patch.
```

### 2.4 The verdict block

```yaml
verdict:
  vs: <incumbent learner version>          # what the paired deltas are against
  foresight_delta: +0.07 ± 0.03            # paired, SE over trajectories
  accuracy_delta:  +0.01 ± 0.04
  serving_delta:   +0.02 ± 0.02
  gauntlet: PASS                           # rolled verdict from gauntlet.md (§2.3);
                                           # on FAIL the member is named
  decision: accept | reject | within-noise
  rationale: >-
    …why, reading deltas AND the reports' rationales; what pattern of
    misses/uptake-failures moved; what remains within noise
```

`within-noise` is a first-class decision, recorded and banked as such — never
rounded to a win. Keep-best banking (design doc §4.4) consumes exactly this
block: a learner version replaces the incumbent only on `accept`.

---

## 3. The split manifest

The exam's fourth artifact — boring but load-bearing: **the authoritative
partition of the trajectory store into learn-set and held-out**, versioned so
holdout rotation is auditable. One `split.yaml` per exam version, living with
the grader machinery in the monorepo (the exam is part of the harness, so it
versions with code):

```yaml
version: 2
rule: >-
  split by (family, time), never by task: a family never appears on both
  sides, and held-out families span early and late dates.
rationale: >-
  v2 rotates rel-avito out of held-out (two crew generations validated
  against it — Goodhart risk) and rotates rel-event in.
learn:
  - {id: rel-amazon/user-churn/20260813T0154_c10, family: rel-amazon, date: 2026-08-13}
  # … every learn-set trajectory, one line each
held_out:
  - {id: rel-event/user-attendance/20260701T0910_a3, family: rel-event, date: 2026-07-01}
  # …
```

Frame checks at load: every store trajectory appears exactly once; no family
on both sides; a version bump carries a rationale. Every scorecard stamps
`split_version`, and **paired comparisons are valid only within one split
version** — the two learner versions must have sat the same exam. Rotation is
therefore a between-generations act (after a learner version ships, before the
next development push), never mid-iteration: rotating counters Goodharting the
holdout, but a rotation resets the pairing baseline and the first scorecard on
a new split is a fresh anchor, not a delta.

## 4. Dual use — settlements become evidence

The claims-settlement computation is evidence ingestion run early: the
campaign measured things that bear on bank claims. So every admitted
`AGREED`/`CONTRADICTED` entry is written in liftable form, and the update
crew, when it later ingests this trajectory, **lifts** the settlement into a
§5.2 evidence entry rather than recomputing it — usage prose from the serving
record (served/cited/absent), effect prose from the settlement's delta,
verdict earned from the same numbers. Settlements on cards the campaign never
saw are the uncontaminated-replication class — the strongest support a fact
can get. One computation, two consumers: the exam grades the bank, the lesson
inherits the grading.

---

## 5. Config constants (Rule 1)

All grader knobs live in the config `learning.graders:` block — none may be
re-hardcoded at a call site: `score_band` (corridor half-width; v1 default
0.20), `min_settlements` (accuracy null threshold; v1: 2), `calibration_min`
(pooled settlements before the table exists; v1: 20), `calibration_buckets`
(v1: `[0.4, 0.7]` cut points); `gauntlet.stability_tolerance` (substance-diff
score tolerance; v1: 0.10); crew role models and repair rounds under
`learning.graders.crew.*` (§6.6). Defaults here are proposals; the config
file is the single source.

---

## 6. The grader crew — the process

Who produces the artifacts above. It lands on the same four-role geometry as
the update crew — lead, writer, adversary, assessor around a deterministic
frame — and not by aesthetics: the roles carry **mutually exclusive
information diets**, and with LLM agents an information barrier is a separate
session. Conventions inherit from the update-crew doc: no naked tags, full
texts into every delegation (Rule 6), one repair round, fail loud.

```
FRAME  stage: bank checkout (RO) · real retriever (push) → brief + record per
       trajectory · eligible claim sets · outcome enumeration · gauntlet
       fixtures + black-box trap runs + mechanical substance diffs
   │
   ▼
LEAD ──► REPORT-WRITER ×N   one per trajectory, IN PARALLEL (read-only world)
     ──► VERIFIER           adversarial pass per report before admission
     ──  gauntlet           lead writes {verdict, rationale} from frame diffs
     ──► SCORECARD-ASSESSOR batch end, the only whole-set view
   │
   ▼
FRAME  validate (§1.6 per report · coverage · recomputed arithmetic) → assemble
```

**Two modes, one machinery.** *Full grading* (development): every held-out
trajectory vs a candidate bank + learner version → reports, gauntlet,
scorecard. *Exam-before-lesson* (operating): one arriving trajectory vs bank
HEAD → one report, writer + verifier only — no gauntlet, no scorecard; the
grade half joins the running curve, the content half is staged into the next
update run by the update frame. The hindcast replays only the **push** path —
what a dead campaign *would have pulled* cannot be simulated; pull uptake is
measured live only (an accepted limit).

### 6.1 Run directory

```
learning/graders/<stamp>/
  inputs.yaml                # mode, bank head, split_version, learner_version,
                             #   incumbent scorecard path, trajectory list
  hindcast/<traj-id>/
    report.md                # §1
    brief.md                 # the would-have-been brief + serving record,
                             #   compiled by the REAL retriever at bank head
  work/verifier-findings.md  # same grammar as critic-findings.md
  fixtures/  diffs/          # gauntlet construction and proofs
  gauntlet.md                # §2.3
  scorecard.md               # §2
```

### 6.2 The lead — launch prompt

```
You are the lead of a grading crew. A knowledge bank claims to carry what
past campaigns taught; this run measures that claim against campaigns it
never learned from. You grade; you fix nothing; nothing you read may be
edited.

INPUTS (inputs.yaml): mode (full | exam), the read-only bank checkout at the
pinned head, the trajectory list, per-trajectory compiled briefs (the frame
already ran the real retriever), the learn-set mined views (for source
searches), the incumbent scorecard (full mode only — for the assessor, not
for you to preview).

THE RUN:
A. REPORTS. Delegate one report-writer per trajectory, in parallel. Reports
   are independent by design: never share a writer across trajectories,
   never pass one report to another writer.
B. VERIFY. Delegate the verifier over every draft. Route [block] findings
   back to that report's writer; one repair round. A report that fails again
   is recorded failed — the run fails loud rather than grade on a lie.
C. GAUNTLET (full mode). The frame already ran the traps and computed the
   substance diffs; read them and write each {verdict, rationale} — the
   verdict follows the mechanical result, the rationale explains what moved.
D. SCORECARD (full mode). Delegate the scorecard-assessor last, after every
   report is admitted.
E. COVERAGE SELF-CHECK. reports == trajectory list; every report verified;
   every trap carries its verdict. Then stop — the frame validates and
   assembles.

You never write a score yourself; no writer may see another report, any
scorecard, or any trend.
```

### 6.3 `.claude/agents/report-writer.md`

```markdown
---
name: report-writer
description: Writes one hindcast report — the exam of the bank against one
  campaign it never learned from.
tools: Read, Grep, Glob, Write
model: {report_writer_model}
---
You grade what a knowledge bank knew in advance of one campaign. Your world:
this trajectory's mined view, the bank checkout (read-only), its compiled
brief + serving record, and the learn-set mined views for source searches.
You must not seek and will not be given: other reports, scorecards, trends.
One report, on its own evidence, per the semantics above (§1).

Duties that carry the report's honesty:
- EXTRACTION. Enumerate the discoveries the campaign PAID for (ledger
  outcomes, judgment sections, difficulties — cite where it paid). For every
  miss, SEARCH the learn-set views for a source: found → MISS-UNCARDED with
  the resolving ref; not found → MISS-NOVEL with the search attested
  (families covered, terms tried). The verifier re-runs your searches; a
  lazy NOVEL is the one lie that inflates the grade.
- CLAIMS. Settle only what the campaign's registered, significance-judged
  numbers can settle; in scope only; THIN is a verdict, not a failure.
- SERVING. Judge hindsight relevance with the reason in the entry; name
  uptake failures explicitly — served is not heard.
- SCORES. Judgment within the corridor (§0.3); null where the base is empty
  (§0.2); the rationale discharges §1.5's duties (binding factor, novel
  share, thinness).
```

### 6.4 `.claude/agents/verifier.md`

```markdown
---
name: verifier
description: Adversarial read-only pass over draft hindcast reports. Attacks
  the cells where lazy judgment inflates grades.
tools: Read, Grep, Glob
model: {verifier_model}
---
You attack hindcast reports before they are admitted. Output:
work/verifier-findings.md — id, severity (block | warn), class, target,
finding, required fix. Check classes, in order:
1. NOVEL-ATTESTATION. For every MISS-NOVEL, re-run the learn-corpus search
   yourself, trying to FIND a source. Found → block: it is MISS-UNCARDED and
   the foresight denominator was shrunk. Your first duty; the exam's most
   gameable cell.
2. SETTLEMENT. Verdicts the significance standard cannot earn; out-of-scope
   results scored as in-scope; deltas that do not re-grep.
3. RELEVANCE. Serving entries whose hindsight-relevance reasoning fails;
   uptake failures narrated as clean hits.
4. CONSISTENCY. Rationale vs markers vs scores beyond the mechanical
   corridor: praise the extraction section does not show; a missing
   thinness admission where settlements are few.
5. ENUMERATION. Discoveries the mined view shows the campaign paid for that
   the report never lists — an unenumerated discovery silently raises
   foresight.
```

### 6.5 `.claude/agents/scorecard-assessor.md`

```markdown
---
name: scorecard-assessor
description: Writes the scorecard verdict block from frame-computed
  aggregates, all report rationales, and the gauntlet.
tools: Read, Grep, Glob, Write
model: {assessor_model}
---
You are the only agent that sees the whole set. Inputs: the frame-computed
aggregation (per-dimension mean ± SE, per-trajectory values, null counts),
the calibration table, every admitted report's rationale, gauntlet.md, and
the incumbent scorecard (paired deltas, same split_version only). Write the
verdict block (§2.4): deltas, gauntlet roll-up, decision accept | reject |
within-noise, and a rationale that reads numbers AND rationales — what
pattern of misses moved, what stayed within noise; noise is never rounded to
a win. Gates dominate scores: any gauntlet FAIL is reject, whatever the
deltas. You add no measurements — you judge the ones admitted.
```

### 6.6 The frame contract

**Staging:** resolve `inputs.yaml`; check out the bank read-only at the
pinned head; run the **real** retriever (push path) per trajectory → `brief.md` +
serving record; stage the eligible claim set and the mechanical outcome
enumeration per trajectory; construct gauntlet fixtures and run the traps —
the update-crew CLI invoked **black-box** on throwaway sandbox checkouts,
substance diffs computed mechanically before any agent runs. **Split
enforcement lives on both sides:** here, full mode's trajectory list IS
`split.held_out` (nothing else is ever graded as the exam) and reports land
only in this run dir; the update-run frame holds the twin checks (batch ∩
held_out = ∅ under `--split`; `inputs.yaml` stages only batch-own reports).

**Validation, in order:** report admission per §1.6 (markers, refs re-grep,
corridors, nulls, liftable form, rationale duties); verifier coverage (every
report attacked; every block resolved or the report failed); run coverage
(reports == trajectory list; traps == battery); **scorecard arithmetic
recomputed** — aggregates, SEs, calibration pooling are frame math, and an
agent-written number is never trusted for arithmetic; assembly (frontmatter
mechanical, `split_version` stamped).

**Parallelism:** report-writers fan out freely — their world is read-only;
the assessor is single and last. **Config** (Rule 1,
`learning.graders.crew:`): `models.{lead, report_writer, verifier,
assessor}` (verifier on a second model where affordable — the diversity is
part of the check), `repair_rounds` (v1: 1).

## 7. Worked example

```markdown
---
trajectory: rel-hm/user-churn/20260819T0300_lane-a1
bank_head: lr_20260817T2100
brief: brief.md
hindcast:
  foresight: 0.45
  accuracy: 0.80
  serving: 0.40
  score: 0.55
  rationale: >-
    The bank carried both fold-leakage and thin-history (served, correctly
    scoped) but missed the campaign's central discovery — cross-family
    ensemble gating — which learn-set trajectory
    rel-amazon/user-churn/20260813T0154_c10 contained and the crew never
    carded: an extraction gap, not a corpus gap; it binds the overall. One
    in-scope contradiction (recency-window overclaims at family scope,
    −0.004 ± 0.001 where it predicts a gain); accuracy rests on 3 settlements
    — thin. Serving is split: thin-history landed (served, cited), but
    fold-leakage was served and re-derived anyway — right and unheard — and
    the ensemble-gating procedure sat banked below budget. Novel share 2/7.
---

## Extraction
- **HIT-SERVED** — grouped-rows fold leakage: re-derived in it-2 across two
  lanes despite [insight: grouped-rows-fold-leakage] in the brief; cost ≈1
  lane-day [mined/it-2/flow-3.md#evaluation].
- **MISS-UNCARDED** — cross-family ensemble gating: the campaign's main win
  (+0.9 test); the same mechanism carried
  rel-amazon/user-churn/20260813T0154_c10 [mined/it-4/flow-2.md#judgment ↔
  learn-set mined/it-3/flow-1.md#evaluation there]; never carded.
- **MISS-NOVEL** — sparse-session recency collapse: no learn-set source
  (searched family-wide mined views; attested) [mined/it-3/flow-4.md].
…

## Claims settlement
- **CONTRADICTED** — [insight: recency-window]: predicts family-scope gain;
  measured −0.004 ± 0.001 on the registered eval
  [mined/it-3/flow-1.md#evaluation]. Liftable; card was served (expectation
  effects apply, §5.2).
- **AGREED** — [insight: thin-history-blind-spot]: predicted degradation
  below 3 events/entity; measured −0.021 ± 0.006 exactly there
  [mined/it-1/flow-2.md#evaluation]. Liftable; card served and cited.
- **THIN** — [procedure: forward-gate]: directional agreement, under
  significance (+0.001 ± 0.002). Exercise-grade only.

## Serving
- **SERVED-USED** — [insight: thin-history-blind-spot]: served, cited by the
  it-1 spec, its predicted boundary steered the depth-slice design.
- **UPTAKE-FAIL** — [insight: grouped-rows-fold-leakage]: served with correct
  scope, never cited; the it-2 re-derivation duplicated it verbatim.
- **SERVE-MISS** — [procedure: cross-family-ensemble-candidate]: banked at
  family scope covering this task, ranked below budget by similarity;
  would have addressed the it-4 build directly.
- **SERVE-NOISE** — [insight: cold-user-imputation]: served on a tag match;
  the campaign had no cold-entity segment.
```

Frame arithmetic on this example: foresight corridor center = 2/(2+3) = 0.40
(2 hits, 3 uncarded; 2 novel excluded) → 0.45 admitted at band 0.20. Accuracy
center = 4/5 = 0.80 with 5 settlements ≥ 2 → scored. Serving center =
hit_rate 1/3 × (1 − noise_share 1/3) ≈ 0.22 → 0.40 admitted (≤ 0.42); had the
session written the 0.50 a generous read suggests, the frame would have
rejected the report with the corridor finding and bounced it back — the
honesty the corridor exists to enforce. Overall corridor
[min 0.40 − 0.20, max 0.80 + 0.20] → 0.55 admitted.

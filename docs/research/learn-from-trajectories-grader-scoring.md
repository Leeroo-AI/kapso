# Learn from Trajectories — grader scoring

Companion to `learn-from-trajectories-design.md` (§4.4, §7): the complete
scoring semantics for the grader suite's measurement artifacts — the hindcast
report and the scorecard. The main doc owns the architecture (what the rungs
are, when they run); this doc owns what every number and marker **means**, how
it is written, how the frame bounds it, and how it aggregates. The gauntlet
appears here only as a gate family; its per-member specs are written when the
gauntlet is iterated.

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
**would-have-been brief** — compiled by the *real* briefing compiler (design
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
covers this task (the same eligibility the compiler uses), plus any card the
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
can lift it into a §5.2 evidence entry without recomputation (§3).

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

Gauntlet members (decoy, implanted-lesson, duplicate-evidence, re-run
stability, red-team coherence) return **PASS/FAIL plus the offending diff**,
never a score, and **any FAIL rejects the learner version regardless of every
number on the scorecard** — there is no scoring your way past a swallowed
implant. Rationale: the gauntlet tests integrity properties whose partial
credit is meaningless (a bank that inflates on duplicated evidence is broken
at 10% exactly as at 100%). Per-member specs are written when the gauntlet is
iterated; this contract — gates dominate scores — is fixed now.

### 2.4 The verdict block

```yaml
verdict:
  vs: <incumbent learner version>          # what the paired deltas are against
  foresight_delta: +0.07 ± 0.03            # paired, SE over trajectories
  accuracy_delta:  +0.01 ± 0.04
  serving_delta:   +0.02 ± 0.02
  gauntlet: PASS                           # or the failing member
  decision: accept | reject | within-noise
  rationale: >-
    …why, reading deltas AND the reports' rationales; what pattern of
    misses/uptake-failures moved; what remains within noise
```

`within-noise` is a first-class decision, recorded and banked as such — never
rounded to a win. Keep-best banking (design doc §4.4) consumes exactly this
block: a learner version replaces the incumbent only on `accept`.

---

## 3. Dual use — settlements become evidence

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

## 4. Config constants (Rule 1)

All grader knobs live in the config `learning.graders:` block — none may be
re-hardcoded at a call site: `score_band` (corridor half-width; v1 default
0.20), `min_settlements` (accuracy null threshold; v1: 2), `calibration_min`
(pooled settlements before the table exists; v1: 20), `calibration_buckets`
(v1: `[0.4, 0.7]` cut points). Defaults here are proposals; the config file is
the single source.

---

## 5. Worked example

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

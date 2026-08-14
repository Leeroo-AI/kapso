# Learn from Trajectories — the update crew

Companion to `learn-from-trajectories-design.md` §4.3 and sibling of
`learn-from-trajectories-mining-prompts.md`: the update crew's **process** —
the actors, the schema contracts between them, the lead's launch prompt, the
three agent definitions, the frame's staging and check sequence, and worked
examples. The main doc owns what the crew is *for*; this doc is what you need
to *run* one. Scoring semantics live in
`learn-from-trajectories-grader-scoring.md`; card semantics in the main doc
§3.1–3.3; evidence admission in §5.2.

---

## 0. The shape of a run

Five actors, one writable surface, one loop:

```
FRAME (deterministic code)
  stage: run dir + bank checkout + read-only inputs + seeded worksheet
     │
     ▼
LEAD (Claude session, native Task subagents)
  A. survey   — complete the observation worksheet beyond the seeds
  B. route+draft — card-writer works the worksheet, edits the bank directly
  C. challenge — critic attacks the full diff + journal; one repair round
  D. close    — assessor walks the ledger: scores + lifecycle, batch-end only
     │
     ▼
FRAME
  validate the diff (invariants, admission, coverage arithmetic)
    → one repair bounce → commit, tag lr_<id>, push, assemble report
```

**The proposal medium is the working tree.** There is no operation vocabulary:
the card-writer edits card files directly; "the proposal" IS `git diff` on the
bank checkout at any moment; the critic reviews that diff; the frame validates
the final diff against the invariants. Trust lives in validation, not in a
restricted edit language.

**Division of labor, one line each.** The frame does everything mechanical and
nothing judgmental; the lead orchestrates and writes the two prose bookends
(headline, closing assessment); the card-writer routes and drafts; the critic
only attacks; the assessor only scores and transitions. No actor grades the
crew itself — grades belong to the graders, and a learning run never sees its
own scorecard.

**Models** come from config (`learning.update_crew.models.{lead, card_writer,
critic, assessor}` — Rule 1). The critic runs on a different model than the
card-writer where affordable: the diversity is part of the check. Rule 6
applies to every delegation: subagent prompts carry **full** observation
texts, card bodies, and evidence ledgers — never clipped, windowed if huge.

---

## 1. The run directory and the in-flight schema

The frame stages `learning/runs/<lr_id>/`:

```
learning/runs/<lr_id>/
  inputs.yaml            # resolved paths: batch mined views, hindcast reports,
                         #   previous report, bank checkout, config snapshot
  bank/                  # the writable git checkout at bank.before
  work/                  # the crew's in-flight artifacts (kept for audit)
    observations.md      #   the routing worksheet   (frame-seeded, lead-completed)
    journal.md           #   the routing journal     (card-writer-appended)
    critic-findings.md   #   the challenge record    (critic-written)
  report.md              # assembled by the frame at the end (main doc §4.3)
```

`work/` is preserved forever: it is the audit trail the stability trap diffs
and the material a human reads when a run looks wrong.

### 1.1 `observations.md` — the routing worksheet

One line per observation; the frame **seeds** it mechanically from the
hindcast reports, the lead **completes** it from the mined views. Grammar:

```markdown
- **obs-01** [seed: lift → insight/recency-window] — CONTRADICTED settlement:
  −0.004 ± 0.001 in scope on the registered eval [hindcast:rel-hm/…#claims]
- **obs-02** [seed: card-candidate] — MISS-UNCARDED: cross-family ensemble
  gating, learn-set source cited [hindcast:rel-hm/…#extraction]
- **obs-07** [lead] — implementor difficulty: OOM on grouped join above 2e7
  rows, worked around by chunking [mined/it-3/flow-2.md#difficulties]
```

Seed classes are mechanical: every `AGREED`/`CONTRADICTED` settlement becomes
a `lift → <card>` row (its target card is already known — the attributed fast
path resolved at seeding time); every `MISS-*` becomes a `card-candidate` row;
every `UPTAKE-FAIL` becomes a `serving-feedback` row (journaled for the
report's closing, no bank edit). The lead adds `[lead]` rows for everything
the hindcast cannot see because it only compares against the bank:
difficulties, drift notes, strategy/operations byproducts, cross-flow
patterns. The worksheet is the coverage denominator: **every row gets exactly
one journal verdict, and the frame counts.**

### 1.2 `journal.md` — the routing journal

Append-only during the run; becomes the report's routing section verbatim.
One entry per worksheet row, no naked tags:

```markdown
- **obs-01 → ATTACH** (fast-path) — lifted CONTRADICTED settlement onto
  [insight: recency-window] as a weaken/refine evidence entry; card was served
  there (expectation effects noted per §5.2). [refs]
- **obs-02 → SPAWN** — new candidate procedures/cross-family-ensemble-candidate:
  two cards could describe the phenomenon (ambiguity, not multi) and the
  mechanism matches neither ledger. [refs]
- **obs-07 → SIGHTING** — single observation, no endorsed mechanism; entry
  added to sightings.md awaiting recurrence. [refs]
- **obs-11 → PASS** — infra death (deadline kill), quarantined telemetry;
  critic's strongest case FOR carding attached below and rejected. [refs]
```

Verdicts: `ATTACH` (evidence onto an existing card; level `fast-path` or the
ordinal fit `exact`/`strong`), `SPAWN` (new candidate card), `SIGHTING`
(sightings.md entry), `PASS` (nothing — requires the critic's rebuttal
recorded in the entry).

### 1.3 `critic-findings.md` — the challenge record

```markdown
- **F-01** [block] [class: evidence] insights/thin-history-blind-spot.md —
  new entry's effect quotes +0.004; the cited manifest shows +0.0004.
  Required: correct the delta and re-earn the verdict (THIN, not confirm).
- **F-02** [warn] [class: abstraction] procedures/cross-family-…/card.md —
  scope_conditions restates the two source datasets' names; state the
  mechanism's precondition instead (bound-identifier smell, not yet a lint hit).
```

Severity `block` must be resolved (fix or re-route) before the frame runs;
`warn` is journaled and may stand with the card-writer's reply. Every finding
carries its class (`routing | abstraction | evidence | pass | report`) and its
required fix — no naked tags.

---

## 2. The lead — launch prompt

Delivered via stdin to the lead session; the frame templates `{…}` values.
`.claude/agents/` in the staged workspace carries the three definitions below;
the lead delegates with the native Task tool.

```
You are the lead of a knowledge-update crew. A batch of finished ML-engineering
campaigns has been mined into readable views; a knowledge bank (a git checkout
you can edit) holds what past campaigns taught. Your job: one reviewed
transaction that folds this batch's lessons into the bank — every decision
journaled, every change survivable by an adversarial critic and a mechanical
validator.

INPUTS (read inputs.yaml for resolved paths):
- work/observations.md — the routing worksheet, pre-seeded from the hindcast
  reports (settlement lifts and carding candidates). You will complete it.
- Mined views for the batch (read-only): per-flow stories, campaign-grain
  strategy/operations/artifacts docs, difficulties. Serving records are in
  each bundle.
- Hindcast reports (read-only): the exam this batch already sat against the
  current bank. Its content is your work list; its grades are not your
  business.
- The previous learner report (read-only): its closing assessment is your
  standing to-do list — address or explicitly carry each item.
- bank/ — the writable checkout: cards, sightings.md, log.md, indexes.

THE LOOP:
A. SURVEY. Read the hindcast reports, the previous closing assessment, and
   every mined view's indexes. Complete work/observations.md: add [lead] rows
   for lessons the hindcast cannot enumerate (difficulties, drift, strategy
   and operations byproducts, cross-flow patterns). An observation is a
   phenomenon with refs — not a conclusion. When done, the worksheet is the
   complete claim of what this batch contains; you will be held to it.
B. ROUTE + DRAFT. Delegate worksheet rows to the card-writer (sequential —
   the bank has one writer at a time; group related rows into one delegation
   so induction sees siblings). The card-writer routes each row (per its
   definition: fast path, categorical fit on mechanism, spawn-over-attach,
   rebuttal-before-pass), edits the bank directly, and appends the journal.
C. CHALLENGE. When the worksheet is exhausted, delegate the full diff +
   journal to the critic (read-only). Route every [block] finding back to the
   card-writer; one repair round. Findings that survive repair go to the
   report unresolved — never silently dropped.
D. CLOSE. Delegate to the reliability-assessor: it walks every touched card's
   ledger, writes reliability blocks and lifecycle transitions (batch-end
   only), and journals them. You do not score anything yourself.
E. BOOKENDS. Write the report's Headline (one paragraph: what this batch
   taught) and Closing assessment (gaps noticed, near-ties, probe suggestions,
   anything troubling — the next run's standing to-dos). Honest and specific;
   "nothing troubling" is a claim you must be able to defend.

RULES THAT BIND YOU:
- Mined views and every read-only input are never edited. Your only mutation
  surface is bank/ and work/.
- Full texts travel into every delegation — never summarize an observation,
  card body, or evidence ledger on the way to a subagent (window if huge).
- Every worksheet row ends with exactly one journal verdict. The validator
  counts; a missing or double verdict fails the run.
- You never write a hindcast, a scorecard, or your own grade.
- Batch may be empty (consolidation): the worksheet then holds the frame's
  shortlist (merge candidates, staleness expiries, sightings pruning) and the
  same loop runs with no new-evidence stage.

When done, verify your own claim: re-read observations.md against journal.md;
re-read the previous closing assessment and confirm each item addressed or
carried. Then stop — the frame validates, repairs once through you if needed,
and commits.
```

Append-system-prompt (the standing contract, terse): *"Direct bank edits, no
operation vocabulary; the diff is the proposal. No naked tags anywhere —
every verdict, finding, and score carries a rationale. Cards are abstractions;
observations are evidence; the body is the fact. You are one writer among
many over this bank's life: leave the working tree, worksheet, and journal in
a state a stranger can audit."*

---

## 3. `.claude/agents/card-writer.md`

```markdown
---
name: card-writer
description: Routes observations to cards and writes/revises cards and
  evidence in the bank checkout. The only agent that edits bank/.
tools: Read, Grep, Glob, Edit, Write
model: {card_writer_model}
---
You route observations into a knowledge bank and write the changes. Your
world: the observation rows delegated to you (full text), the bank checkout,
the mined views and hindcast reports they cite (read-only).

ROUTING — for each row, in order:
1. Fast path: if the row is a seeded lift or the observation concerns a card
   in the campaign's serving record, the target is known — no fit judgment.
2. Otherwise find candidates by index traversal (scope-first, hero lines),
   then rate ordinal fit per candidate — exact | strong | partial | weak |
   unrelated — against the card's BODY AND EVIDENCE LEDGER, never its hero
   line alone, and on MECHANISM, never vocabulary: a shared dataset, feature,
   or metric name is not a match; the same causal story is.
3. Verdict by rule: lone exact/strong winner → ATTACH. Tie or ambiguity →
   SPAWN a candidate (a wrong attach corrupts a ledger; a spurious candidate
   dies of non-recurrence). Two genuinely distinct mechanisms → evidence to
   both; one phenomenon two cards could describe → that is ambiguity, SPAWN.
4. Nothing card-worthy: SIGHTING (one line in sightings.md: date, trajectory
   ref, phenomenon sentence) — check existing sightings first: a match fires
   the recurrence gate and the born card cites both. PASS only for
   quarantined telemetry (infra deaths, deadline kills, degenerate judge
   echoes flagged by mining) or true no-content — and every PASS goes to the
   critic for its rebuttal before you journal it.

WRITING — the rules that bind every edit:
- Admission: a new card needs induction (≥2 aligned instances from ≥2
  independent measurements — same run ids are ONE measurement) or EBG (one
  instance + a mechanism stated so an independent check can endorse it;
  admitted at reduced evidence weight, reliability state candidate).
- The bound-identifier lint and the MDL test apply to every claim: named
  datasets/features/runs belong in evidence and scope coordinates, never in
  the fact; a card about as long as its cited instances has not abstracted.
- Evidence entries have exactly four parts (source, verdict, usage, effect —
  design doc §5.2); the effect ends with the sentence that earns the verdict;
  quoted numbers must re-grep in the cited artifact. Lifts from hindcast
  settlements copy the settlement's delta and refs — never recompute, never
  embellish.
- Versioning: any card-text change bumps version and writes exactly one log
  entry (the frame stamps commit/date); evidence and log are append-only;
  retirement is a move to retired/; contradicts lands on both cards.
- Journal every verdict in work/journal.md as you go: obs-id → verdict
  (level) — rationale [refs]. No naked tags.
Your final message: the row-range handled, cards touched, anything you could
not route with confidence (the lead re-delegates or takes it to the critic).
```

---

## 4. `.claude/agents/critic.md`

```markdown
---
name: critic
description: Adversarial read-only pass over the run's full diff and journal.
  Attacks routing, abstraction, evidence honesty, passes, and report claims.
tools: Read, Grep, Glob
model: {critic_model}
---
You attack a knowledge-update run. Read the bank diff (git diff in bank/),
work/journal.md, work/observations.md, and the cited sources. You fix
nothing; you write work/critic-findings.md. Every finding: id, severity
(block | warn), class, target, the finding, the required fix. No naked tags.

CHECK CLASSES, in order:
1. ROUTING. Mis-attaches: does each ATTACH's observation share the card's
   MECHANISM, or only its vocabulary? Forced attaches that should be spawns
   (was there a tie the journal glossed?). Sightings that match an existing
   sightings.md entry (missed recurrence). Fast-path claims not actually in
   the serving record.
2. ABSTRACTION. Bound identifiers in facts; cards that restate their
   instances (MDL); EBG cards whose mechanism you cannot endorse — argue why
   it fails, that argument blocks admission; scope broader than the cited
   families justify; hero lines that oversell the fact.
3. EVIDENCE. Re-grep every quoted number. Usage stories vs serving records
   (claimed citation, claimed probe, claimed independence — all checkable).
   Verdicts the numbers cannot earn (sub-threshold confirm). INDEPENDENCE:
   two entries citing the same run ids or the same registered-eval outcome
   are ONE measurement — flag any induction or score movement built on echo.
4. PASSES. For every PASS in the journal, argue the STRONGEST case that the
   observation IS card-worthy. If your case survives your own scrutiny, the
   PASS becomes a block finding; if not, record the rebuttal (it becomes part
   of the journal entry — the pass is only valid with it).
5. REPORT. Journal coverage vs the worksheet; headline claims vs the diff
   (does the bank now actually carry what the headline says it learned?);
   previous closing assessment items silently dropped.
Severity: block = would corrupt the bank or the record (wrong attach, false
number, unearned verdict, unendorsable EBG); warn = should improve but does
not corrupt. End with a two-line verdict of the run's overall honesty.
```

---

## 5. `.claude/agents/reliability-assessor.md`

```markdown
---
name: reliability-assessor
description: Batch-end scoring and lifecycle. Walks every touched card's
  ledger, writes reliability blocks and state transitions with rationales.
tools: Read, Grep, Glob, Edit
model: {assessor_model}
---
You are the one assessor (design doc §3.3). You run once, at batch end, after
the critic's repair round. For every card touched this run (and every card
whose lifecycle clock expired):
- Derive nothing new: you read the evidence ledger as repaired and admitted.
  An observation is evidence, never a decision — no single trajectory flips a
  state; you weigh the whole ledger.
- Write the reliability block: validity, boundary, coverage, overall score,
  one rationale — scores bounded by the ledger (the frame checks; a score
  the events cannot support bounces the run). Frame-bounded judgment, same
  idiom as everything else: no naked numbers, no fabricated precision.
- Refines: a scope revision must speak the MECHANISM's vocabulary and cite
  both sides (Lakatos guard). An ad-hoc rescue (scope carved to dodge one
  contradiction with no mechanism story) is not yours to write — flag it to
  the lead instead.
- Lifecycle: candidate → active needs the admission gate met at full weight;
  active → cold on the visit clock; cold/contested → retired only on measured
  contradiction (never on age alone); superseded links both ways. Every
  transition journals to work/journal.md with its rationale.
Your final message: cards scored, transitions made, anything the ledger
cannot support that the run claimed.
```

---

## 6. The frame contract

**Staging (before the lead starts):** create the run dir; clone/checkout the
bank at `bank.before`; resolve and pin every input path into `inputs.yaml`;
seed `work/observations.md` from the batch's hindcast reports (settlement →
lift row with target card; MISS-* → card-candidate row; UPTAKE-FAIL →
serving-feedback row); place the three agent definitions; snapshot the
`learning:` config block. Development invocations carry `--split`: staging
then asserts batch ∩ held_out = ∅ (fail loud, before any session exists) and
stages only the batch's own hindcast reports into `inputs.yaml` — held-out
reports live in grader run dirs and are never staged here; live invocations
omit `--split` (no exclusion exists in operation). Consolidation mode: seed the worksheet from the
mechanical shortlist instead (opposing-sign evidence on overlapping scopes
via the edge table, staleness expiries, sightings past
`learning.update_crew.sightings_expiry_batches`).

**Validation (after the lead stops), in order, all mechanical:**

1. **Surface check** — the diff touches only `bank/` and `work/`; read-only
   inputs untouched (hash check).
2. **Diff invariants** — evidence and log append-only; version ⇔ log-entry
   one-to-one; `contradicts` on both cards; retirement is a move; decoy cards
   untouched; sightings edits are appends or expiry removals only.
3. **Evidence admission** — §5.2's three checks per new entry (source
   resolves + numbers re-grep; usage vs serving record; verdict earnable).
   Independence: entries citing identical run ids / eval outcomes are one
   measurement wherever counted.
4. **Coverage arithmetic** — every worksheet row has exactly one journal
   verdict; every diff hunk traces to a journal entry (by card path); every
   PASS entry carries its rebuttal; every block finding resolved or surfaced
   in the report.
5. **Score bounds** — every reliability block bounded against its ledger;
   no naked tags anywhere (verdict/finding/score without rationale fails).
6. **Derived rebuilds** — the frame (not agents) recompiles `index/`
   (embeddings, edges) and `index/probe-queue.md` (VoI rank: uncertainty ×
   serving exposure from ledger-derived stats), and appends the one-line
   `log.md` entry.

On violation: one repair bounce to the lead with named findings, then fail
loud — a failed run commits nothing (the checkout is discarded; the run dir
and its work/ survive as the post-mortem).

**Commit:** one commit, message = headline + batch ids, tagged `lr_<id>`,
pushed (single writer). **Report assembly:** frontmatter (identity + health
block, all frame-computed) + Headline (lead) + Routing journal (journal.md
verbatim) + Card changes (derived from the diff) + Rejections (validation and
critic blocks with dispositions) + Assessor round-up (from journal) + Closing
assessment (lead).

Config (Rule 1, `learning.update_crew:`): `models.{lead,card_writer,critic,
assessor}`, `repair_rounds` (v1: 1), `sightings_expiry_batches`, batch
trigger (`min_trajectories` or on-demand).

---

## 7. Worked examples (wave-4 flavored)

**Lift (fast path).** Seeded row: CONTRADICTED settlement on
[insight: recency-window]. Card-writer copies delta and refs into a four-part
evidence entry (usage: "served in this campaign's brief; expectation effects
apply"), verdict weaken-with-refine-candidate; journal: `obs-01 → ATTACH
(fast-path)`. Assessor at close: refine to family scope — mechanism-vocabulary
rationale citing both sides; version bump, one log entry.

**Spawn from a MISS.** Seeded row: MISS-UNCARDED ensemble gating, learn-set
source already cited by the hindcast. That citation is instance one; the
batch trajectory is instance two; two independent measurements → induction
met. Card born with both founding evidence entries (independent by
construction), state candidate; journal: `obs-02 → SPAWN` with the
ambiguity-not-multi rationale.

**Echo caught.** Card-writer proposes attaching two confirms to
[procedure: forward-gate] from one campaign's it-3 and it-4 flows — same
registered eval, same run ids in both refs. Critic F-finding [block, class:
evidence]: one measurement, one entry; repaired to a single confirm.
(The duplicate trap tests exactly this reflex on demand.)

**Pass with rebuttal.** [lead] row: lane-b3 harness crash narrative. Critic's
strongest case for carding: "recurrent OOM pattern could be a pitfall card."
Rebuttal check: the mined view flags the lane as deadline-killed
(quarantined telemetry) and the OOM is already carded at family scope.
Rebuttal survives → PASS journaled with it; the existing card gains nothing
(no new measurement).
```

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

Ten actors — frame, lead, card-writer, five docket specialists, critic,
assessor — one writable surface, one loop:

```
FRAME (deterministic code)
  stage: run dir + bank checkout + read-only inputs + seeded worksheet
     │
     ▼
LEAD (Claude session, native Task subagents)
  A. survey   — complete the observation worksheet beyond the seeds
  B. route+draft — card-writer works the batch rows; docket rows go to their
                   specialists (merger · generalizer · resolver · sweeper · codifier);
                   one bank writer at a time
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
(headline, closing assessment); the card-writer routes and drafts batch rows;
each docket specialist executes exactly one maintenance verdict; the critic
only attacks; the assessor only scores and transitions. No actor grades the
crew itself — grades belong to the graders, and a learning run never sees its
own scorecard.

**Models** come from config (`learning.update_crew.models.{lead, card_writer,
critic, assessor}` — Rule 1). The critic runs on a different model than the
card-writer where affordable: the diversity is part of the check. The docket
specialists default to the card-writer's model, overridable per role. Rule 6
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
report's closing, no bank edit); every probe the serving record shows attached
becomes a `probe-check` row — did a matching experiment run? settle it with
the prospective label (flow↔spec match, critic-verified) or journal it
unexercised. The lead adds `[lead]` rows for everything
the hindcast cannot see because it only compares against the bank:
difficulties, drift notes, strategy/operations byproducts, cross-flow
patterns. **The maintenance docket is seeded on every run too**, from the
pre-run bank via the derived index: `dup-merge` rows (embedding pairs above
`learning.update_crew.dup_threshold` — inclusive; similarity nominates,
mechanism decides), `tension` rows (`contradicts` pairs both active, or
opposing-sign evidence on overlapping scopes), `generalize` rows
(sibling-scope cards with same mechanism and agreeing ledgers),
`expiry` rows (validity windows, cold clocks, sightings past expiry), and
`codify` rows (text procedures whose reference-closure recurrence over
executed-verdict entries crosses `learning.codify.min_recurrence`; guards:
active, uncontested, no failed attempt since the last executed evidence —
seeder rule in the companion codify doc). With
`batch: []` the run is docket-only. The worksheet is the coverage denominator: **every row gets exactly
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
recorded in the entry). Docket rows take: `MERGE` (successor supersedes ≥2
twin parents, evidence by reference), `GENERALIZE` (domain successor born
candidate + unseen-family probe queued), `RESOLVE` (a tension settled: scope
split, a retirement proposed to the assessor, or kept-contested with a probe
queued — the entry says which), `EXPIRE` (staleness/sightings pruning), `CODIFY`
(a representation flip folded from a green codify run — companion codify
doc). A
declined nomination journals as `PASS` with the distinguishing rationale —
the inclusive threshold makes false nominations normal, so no rebuttal is
required.

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
B. ROUTE + DRAFT. Delegate BATCH rows to the card-writer (sequential — the
   bank has one writer at a time; group related rows into one delegation so
   induction sees siblings). Delegate DOCKET rows to their specialists —
   dup-merge → card-merger, generalize → card-generalizer, tension →
   tension-resolver, expiry → expiry-sweeper, codify → procedure-codifier —
   one row per delegation,
   serialized with the card-writer. A specialist's PASS that names a
   re-route (a generalize row that is really a tension) comes back to you:
   re-delegate it to the named specialist.
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
- The worksheet always carries the maintenance docket too (dup-merge,
  tension, generalize, expiry rows — seeded from the pre-run bank). Work it
  after the batch rows. With an empty batch the run is docket-only and the
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
- Docket rows are not yours: the lead delegates them to the docket
  specialists (card-merger, card-generalizer, tension-resolver,
  expiry-sweeper).
- Journal every verdict in work/journal.md as you go: obs-id → verdict
  (level) — rationale [refs]. No naked tags.
Your final message: the row-range handled, cards touched, anything you could
not route with confidence (the lead re-delegates or takes it to the critic).
```

---

## 4. The docket specialists — five single-verdict agents

The maintenance verdicts are critical enough to be first-class agents: one
definition per verdict, each doing exactly one thing, staged into every
update-run workspace beside the other definitions. **Any lead** — a batch run,
a standalone docket-only run — delegates to them with the native Task tool;
nothing outside a learning run can call them, because they require the one
thing only update runs have: a writable bank checkout. They serialize with the
card-writer (one bank writer at a time), and a declined nomination always
journals as `PASS` with the distinguishing rationale.

### 4.1 `.claude/agents/card-merger.md`

```markdown
---
name: card-merger
description: Executes one dup-merge docket row — decides mechanism identity
  and, on a true twin pair, writes the successor and retires the parents.
tools: Read, Grep, Glob, Edit, Write
model: {card_writer_model}
---
You receive one docket row naming cards A and B, nominated as duplicates by
similarity. Similarity nominates; only MECHANISM identity merges.
1. Read both cards whole — body and full evidence ledgers. One causal story
   told twice, or two stories sharing vocabulary? Two → journal PASS with
   the argument that separates them, and stop.
2. One → write the successor at a fresh path: the unified fact in its
   clearest wording; scope = the union the two ledgers justify; tags union;
   supersedes: [A, B] (a list).
3. Found its evidence BY REFERENCE — one entry per parent: source.ref →
   retired/<parent>#evidence, verdict confirm, usage "merge founding —
   stands for the parent's full ledger (N entries)", effect = that ledger's
   net outcome and score. Never copy parent entries; never edit them.
4. Retire both parents: move to retired/, state superseded, forward link to
   the successor, one log entry each.
5. Journal MERGE with the mechanism argument and refs.
You never write the successor's reliability block (the assessor scores it
over the referenced ledgers, discounted prior) and never touch other cards.
```

### 4.2 `.claude/agents/card-generalizer.md`

```markdown
---
name: card-generalizer
description: Executes one generalize docket row — verifies cross-family
  mechanism agreement and births the domain successor as a candidate with
  its unseen-family probe.
tools: Read, Grep, Glob, Edit, Write
model: {card_writer_model}
---
You receive one docket row naming sibling-scope cards that appear to state
one mechanism across ≥2 families with agreeing ledgers.
1. Verify both halves yourself, reading every card whole: the mechanism is
   identical at EVERY family, and no ledger carries in-scope opposing-sign
   evidence. Mechanisms differ → journal PASS with the distinction. A ledger
   disagrees → journal PASS naming it a tension for the tension-resolver.
2. Write the domain successor: the fact restated at domain level in the
   mechanism's vocabulary — no family or dataset names in the fact (the
   lint applies); scope: domain; reliability state CANDIDATE; supersedes:
   all parents; founding evidence by reference per parent, as in a merge.
3. Write its probe as exactly what the generalization ADDS: one fold on a
   family outside the seen set, with the mechanism's predicted sign. The
   card is born a prediction with its test attached; coverage stays
   ledger-derived and honestly shows only the seen families.
4. Retire the parents (move, superseded state, forward links, log entries).
5. Journal GENERALIZE with the agreement evidence and refs.
Never born active, never without the probe, never scored by you.
```

### 4.3 `.claude/agents/tension-resolver.md`

```markdown
---
name: tension-resolver
description: Executes one tension docket row — reads both sides whole and
  settles a contradiction by scope split, retirement proposal, or
  contested-with-probe.
tools: Read, Grep, Glob, Edit, Write
model: {card_writer_model}
---
You receive one docket row naming cards A and B in tension (a contradicts
pair, or opposing-sign evidence on overlapping scope). Read both cards and
both full ledgers, then pick the ONE ending the evidence supports:
- SPLIT — the disagreement lives in different regions (A true here, B true
  there). Refine both scopes; the boundary must be stated in the MECHANISM's
  vocabulary and cite both sides' evidence. Lakatos guard: a boundary that
  merely quarantines one bad result is ad-hoc — if that is all you have,
  this is CONTESTED, not SPLIT. If the split fully explains the tension,
  clear the contradicts edge on both cards; version bump + log entry each.
- PROPOSE-RETIRE — one card's own in-scope ledger is net-refuting on your
  full read. You retire nothing: journal the proposal with the ledger
  argument; the assessor executes at batch end only if the ledger supports
  it.
- CONTESTED-WITH-PROBE — the stored evidence cannot settle the pair. Keep
  both active, ensure the contradicts edge sits on both, and write the
  DISCRIMINATING probe — the one cheap experiment whose outcome separates
  the claims — into the probe field of the weaker-scored card, naming both
  cards and both predicted outcomes. The co-serving guard names the tension
  until the probe settles it.
Journal RESOLVE with the chosen ending and its rationale.
```

### 4.4 `.claude/agents/expiry-sweeper.md`

```markdown
---
name: expiry-sweeper
description: Executes the expiry docket rows — prunes stale sightings and
  proposes lapsed cards to the assessor.
tools: Read, Grep, Glob, Edit
model: {card_writer_model}
---
You receive the expiry rows: sightings past their expiry, cards with lapsed
validity windows, cards past their cold clocks.
- Sightings: remove the expired lines from sightings.md — the one removal
  the invariants permit; the entries persist in git history and their mined
  views. Before pruning, scan this run's journal once: a sighting matched
  in THIS run is never pruned.
- Cards: you edit NOTHING. Journal each lapse as a proposal with its clock
  arithmetic; the assessor executes cold transitions at batch end.
Journal EXPIRE, grouped: what was pruned, what was proposed, with refs.
```

### 4.5 `.claude/agents/procedure-codifier.md`

Defined in the companion `learn-from-trajectories-codify.md` (§6), beside the
codify run it drives: the compatibility gate before any machine spins, the
run request, the outcome fold. Staged into the workspace with the other four.

## 5. `.claude/agents/critic.md`

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
   families justify; hero lines that oversell the fact. A nominated MERGE is
   decided on MECHANISM identity, never similarity — argue the two cards'
   mechanisms apart; if you can, the merge is blocked.
3. EVIDENCE. Re-grep every quoted number. Usage stories vs serving records
   (claimed citation, claimed probe, claimed independence — all checkable).
   Verdicts the numbers cannot earn (sub-threshold confirm). INDEPENDENCE:
   two entries citing the same run ids or the same registered-eval outcome
   are ONE measurement — flag any induction or score movement built on echo. A
   **prospective** (probe) label requires the settled flow to actually match
   the probe spec the serving record shows attached — challenge the match.
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

## 6. `.claude/agents/reliability-assessor.md`

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
- Docket proposals (propose-retire, expiry lapses) reach you through the
  journal: execute only what the ledger supports; decline with a rationale
  otherwise.
Your final message: cards scored, transitions made, anything the ledger
cannot support that the run claimed.
```

---

## 7. The frame contract

**Staging (before the lead starts):** create the run dir; clone/checkout the
bank at `bank.before`; resolve and pin every input path into `inputs.yaml`;
seed `work/observations.md` from the batch's hindcast reports (settlement →
lift row with target card; MISS-* → card-candidate row; UPTAKE-FAIL →
serving-feedback row); place the three agent definitions; snapshot the
`learning:` config block. Development invocations carry `--split`: staging
then asserts batch ∩ held_out = ∅ (fail loud, before any session exists) and
stages only the batch's own hindcast reports into `inputs.yaml` — held-out
reports live in grader run dirs and are never staged here; live invocations
omit `--split` (no exclusion exists in operation). The maintenance docket is seeded on **every** run from the pre-run bank
(dup pairs above `dup_threshold`, `contradicts` both-active tensions,
sibling-scope agreement, validity/cold/sightings expiries past
`learning.update_crew.sightings_expiry_batches`); with `batch: []` the run
is docket-only (standalone consolidation). Docket state is read at staging
from `bank.before` — twins born inside this run surface next run. `codify`
rows follow the companion codify doc's seeder rule; the frame launches
requested codify runs on `learning.codify.target` (ephemeral GCP by default)
and holds the transaction open for the result.

**Validation (after the lead stops), in order, all mechanical:**

1. **Surface check** — the diff touches only `bank/` and `work/`; read-only
   inputs untouched (hash check).
2. **Diff invariants** — evidence and log append-only; version ⇔ log-entry
   one-to-one; `contradicts` on both cards; retirement is a move; decoy cards
   untouched; sightings edits are appends or expiry removals only; a `MERGE`
   successor supersedes ≥2 parents (both moved to `retired/`, links both ways,
   founding evidence citing parent ledgers by reference — never copied); a
   `GENERALIZE` successor is state candidate with its unseen-family probe
   present and coverage claiming only seen families; a `representation:
   code` flip requires a green codify run in the same transaction (companion
   codify doc).
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
   (edges; embeddings — consolidation-shortlist use only) and `index/probe-queue.md` (VoI rank: uncertainty ×
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
assessor}`, `repair_rounds` (v1: 1), `sightings_expiry_batches`, `dup_threshold`
(inclusive — it nominates docket rows, never decides), batch
trigger (`min_trajectories` or on-demand).

---

## 8. Worked examples (wave-4 flavored)

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

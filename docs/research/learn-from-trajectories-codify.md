# Learn from Trajectories — the procedure code path

Companion to `learn-from-trajectories-design.md` (§3.2 procedure cards, §4.3
docket) and `learn-from-trajectories-update-crew.md` (the docket machinery):
how a procedure card climbs from `representation: text` to `representation:
code` — detection, the codify run, placement, freshness, and the interactions
with merge and serving. This closes design-doc open question §10.5.

**Why the path exists, and why it is gated.** A text procedure is advice every
campaign re-implements from the recipe; a code procedure is a verified
executable campaigns adapt or invoke — the knowledge-duality axis's
proceduralization half. Code is also the only card content that can be
*silently wrong in execution*, and the SkillRevise result is blunt: one-shot
self-authored code scores below no code at all. So nothing on this path ever
serves unverified, and the division of labor is fixed up front: **the codify
run verifies the code; the evidence ledger verifies the knowledge; serving and
A/B verify the value.** One fixture cannot prove generality and is never asked
to — a procedure that replays green but fails in the field takes ordinary
`refute` evidence like any card.

---

## 1. Detection — the `codify` docket row

Same two-layer split as every decision in the design: **the frame counts, an
agent judges.**

**Layer 1 — mechanical nomination (the seeder, at every run's staging).** For
each `type: procedure`, `representation: text` card:

```
executed   = ledger entries with verdict ∈ {confirm, weaken, refute, refine, exercise}
             # outcome verdicts imply execution: §5.2 admission demands effect
             # numbers that re-grep in a run where the method actually ran —
             # a mere mention can never earn one, so it never enters the count
closure    = executed entries, with merge/generalize founding references
             EXPANDED into the retired parents' ledgers, recursively
             # reference transparency — the same rule the assessor scores by;
             # termination guaranteed (supersedes only points backward)
recurrence = |{ e.source.trajectory : e ∈ closure }|
             # set semantics: same-campaign entries collapse; and the union
             # across merged parents dedups a shared source campaign, which
             # summing the parents would have double-counted

seed a codify row  iff  recurrence ≥ learning.codify.min_recurrence (v1: 2)
                   and  state == active  and  contradicts == []
                   and  no failed codify attempt since the ledger's last
                        executed entry   # a hopeless card is not retried
                                         # until new evidence arrives
```

Pure YAML arithmetic on `source.trajectory` fields the frame already parses;
no agent runs, no code is read.

**Layer 2 — the compatibility gate (the procedure-codifier, crew-side, before
any machine spins).** Recurrence alone is not the trigger — the §10.5 default
is recurrence *with compatible implementations*, and compatibility is
judgment: the specialist opens the cited implementations in the archived runs
and asks whether they are **the same method twice** — same algorithm and
steps, differing only in bindings and parameters — such that one code
artifact with declared preconditions covers all. No → `PASS` with the
incompatibility argument (two realizations under one card is a split
candidate, flagged to the lead). Yes → the codify run is requested. Declined
nominations cost agent-minutes, never a box.

---

## 2. The codify run — evolve minus ideation

The card **is** the idea. Codification runs kapso's own loop with the
ideation stage removed: one spec, one lane, implement → evaluate → judge →
feedback → iterate, bounded by config. Everything is reused — implementor
prompts, registered-evaluation plumbing, the feedback-generator pattern,
workspace/session/budget machinery — invoked in a degenerate mode.

```
FRAME launches on the placement target (§3):
  spec       the card: fact + method body + declared preconditions
  materials  the cited archived implementations   (adapt, don't author)
  fixture    one cited run's INPUTS, staged from the store
             (the specialist picks the strongest measured instance, and says why)
  registered evaluation — the reproduction gates, the score-of-record:
             · decision outcomes  → exact match (the method's essence)
             · numeric outcomes   → within ±z·SE of the fixture run's RECORDED
               outcome, SE from the campaign's own machinery
               (z = learning.codify.tolerance_z, same z as evidence significance)
             · artifact/report outcomes → property/structure checks
             assertion level is the codifier's open choice per outcome type;
             an assertion weaker than the card's expected_outcome fails
             validation even when green (the anti-weak-test check)

  IMPLEMENTOR session
      adapts the cited code into code/ + entrypoint: bindings → parameters
      (dataset names, paths, thresholds — observed values as defaults),
      runs the registered evaluation
        │
  FEEDBACK JUDGE (agent) — evaluates THE CLAIMS after the run:
      reproduction (reads the eval outcome) · faithfulness (does the code
      implement the card's stated method, or reach the number by a shortcut
      that is not the method?) · preconditions honesty (declared = what the
      code actually needs, hardware included, from the fixture run's
      telemetry) · effect consistency with what the evidence ledger claims
      → verdict + written feedback
        ├─ fail → feedback to the implementor, next iteration
        │         (learning.codify.max_iterations, v1: 3)
        ▼
  PASS = mechanical gates green AND judge endorsement
```

**Checks inherited from the replay design, now enforced by the run:**
*actually-invoked* — the workspace is staged with fixture inputs only, never
its outputs, so every output must be freshly produced by the invoked process;
*effect-agrees* — the evaluation's assertions implement the card's
`expected_outcome` statement, compared mechanically at validation.

**On success:** `code/` + `entrypoint` + `preconditions` (hardware included)
land on the card; **the registered evaluation itself becomes `replay/`** —
the permanent, re-runnable test; `representation` flips with a version bump
and one log entry. **The transaction rule:** the flip commits only inside the
learning transaction that holds a green codify run — no green run, no flip.
**On failure** (iterations exhausted): no flip; the attempt parks in the card
dir with its trace; the log records it; the card stays text and is not
re-nominated until new executed evidence arrives. Parked code can never
serve — the retriever stages `representation: code` only.

---

## 3. Placement — where and in what environment

**Environment: always the current one — deliberately.** The codify run and
every freshness re-run execute in today's env (current kapso, current domain
deps), never a reconstruction of the archived run's era. Replay certifies
"works for the *next* campaign"; when env drift breaks the code, that is the
rot detector firing, not a reproduction failure. The `expected_outcome`
tolerance absorbs minor numeric drift; a real break is a real signal.

**Machine: `learning.codify.target`** —

- `local`: run in the sandboxed workspace on the current box when its
  hardware satisfies the card's `preconditions` (CPU-cheap gates and
  diagnostics run anywhere, dev box included).
- `gcp_ephemeral` (the default once wired): the frame resolves the machine
  type from `preconditions` (CPU-only → small instance; `gpu: any` → the
  standard campaign machine type from config), provisions, bootstraps the
  standard campaign env, stages the fixture from the artifacts bucket, runs
  under the iteration timeout, and tears down unconditionally. Preemptible is
  fine — a codify iteration is short and idempotent; a preemption is a rerun.

Deferral (park the flip until a learning run lands on a capable box) survives
only as the fallback when provisioning is disabled in config. The read-only
substrate principle is untouched throughout: this is learning-side machinery
*reusing* evolve's code on learning's own boxes — production campaigns gain
no duties.

---

## 4. Freshness — code rots on a clock

`last_replayed` is a card field; when it exceeds `learning.codify.replay_max_age`
the **expiry docket** picks the card up like any lapsed clock. The re-run
executes **only the registered evaluation** (`replay/`) — no implementor, no
judge, minutes of machine time. Green → restamp. Fail → one codify-run
iteration with the failure as feedback; fail again → **demote to text** with
the log saying why. Stale or demoted code never serves silently: replay
freshness renders in the health panel and discounts in serving.

---

## 5. Interactions

- **Merge inheritance.** When procedures merge and a parent holds a code
  representation, the successor inherits it **only if the replay passes
  re-run under the successor in the same MERGE transaction**; otherwise the
  successor is born text — and the seeder, counting through references,
  simply re-nominates it next run. The system heals instead of special-casing.
- **Reference transparency** is one shared rule: the assessor scores through
  merge-founding references; the codify seeder counts through them. One
  function, two consumers.
- **Serving** (main doc §5.1): a `representation: code` card with fresh
  replay stages as `card.md` + `code/` into the shared artifact workspace,
  version-pinned; `replay/` never stages (it needs archive access and lives
  learning-side only).

---

## 6. `.claude/agents/procedure-codifier.md`

```markdown
---
name: procedure-codifier
description: Executes one codify docket row — judges implementation
  compatibility, requests the codify run, folds its outcome onto the card.
tools: Read, Grep, Glob, Edit
model: {card_writer_model}
---
You receive one codify row naming a text procedure whose ledger shows ≥N
distinct-campaign implementations. Nothing heavy happens until you say so.
1. COMPATIBILITY GATE (before any machine spins). Open the cited
   implementations in the archived runs. The same method twice — same
   algorithm and steps, differing only in bindings and parameters — such
   that ONE code artifact with declared preconditions covers all? No →
   journal PASS with the incompatibility argument; two realizations under
   one card is a split candidate — flag it to the lead. Yes → continue.
2. REQUEST THE CODIFY RUN. You do not implement. Hand the frame: the card as
   spec, the cited implementations as materials, your pick of fixture run
   (the strongest measured instance — say why), and the reproduction gates
   drawn from its recorded outcomes.
3. FOLD THE OUTCOME. Green run: confirm code/, entrypoint, preconditions
   (hardware included, from the fixture run's telemetry), and replay/ landed
   on the card; flip representation; version bump + one log entry; journal
   CODIFY with iterations used and the judge's endorsement line. Failed run:
   no flip — journal the failure with its trace ref; the card stays text and
   will not be re-nominated until new executed evidence arrives.
You never write the implementation, never run the evaluation yourself, and
never edit other cards.
```

**Prompt adaptations inside the run** (derived from evolve's existing prompts
at implementation time, two additions each): the implementor's prompt frames
the card as the spec — *"implement exactly the method this card states, by
adapting the cited implementations; parameterize bindings; do not invent an
alternative method even if you believe it better — fidelity is the
acceptance criterion"* — and the feedback judge's template gains the four
claims questions (reproduction, faithfulness, preconditions honesty, ledger
consistency) with verdict + feedback as its output contract.

---

## 7. Config (Rule 1)

All knobs in one block, `learning.codify:`: `min_recurrence` (v1: 2),
`max_iterations` (v1: 3), `iteration_timeout`, `target`
(`local | gcp_ephemeral`), `machine_type` (the gpu-class default),
`replay_max_age`, `tolerance_z` (v1: 2 — the same z as evidence
significance). Values here are proposals; the config file is the single
source.

---

## 8. Worked example — forward-gate, end to end

Detection: ledger holds confirm(rel-amazon, `runs/run_0019`),
confirm(rel-hm, `it-4/flow-2`), exercise(rel-hm, same campaign) → closure
trajectory set size 2 ≥ 2, state active, no contradicts → row seeded.
Compatibility: both implementations are "train one fold with/without the
candidate family, keep if the delta clears noise" — parameter differences
only → gate passes; fixture pick: `run_0019` (cleanest effect, +0.006 with
recorded SE 0.0015 → band ±0.003).

Codify run, iteration 1: implementor adapts the gate, parameterizes
`features_path`, `group_col`, threshold; eval reproduces the keep/drop matrix
exactly but lands +0.002 — outside the band. Judge: reproduction failed;
cause named — per-family seed pinning dropped in adaptation; feedback to
implementor. Iteration 2: seeds pinned; +0.0058 within ±0.003; matrix exact;
judge endorses faithfulness and preconditions ("features parquet, holdout
fold with group column, gpu: none"). Flip commits inside the learning
transaction; journal: `CODIFY — 2 iterations, judge endorsed; fixture
run_0019`. Six weeks later the freshness clock lapses; the eval-only re-run
lands +0.0061 — restamped, no agent involved.

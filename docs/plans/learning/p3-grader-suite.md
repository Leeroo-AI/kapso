# P3 — Grader suite v0

**Goal:** the exam exists before the student (MD§8.3): split manifest,
retriever push core, hindcast runner + grader crew, scorecard math, gauntlet
harness. **Design sources:** GS in full (§0 idiom/corridors/null, §1 hindcast
+ markers, §2 scorecard incl. §2.3 gauntlet + §2.5 A/B verdict shape, §3
split manifest, §4 lift, §5 config, §6 crew + frame), MD§5.1 (push path),
MD§7. **Depends on:** P2 (hindcast reads mined views). **Doubts:** D4, D6,
D7. **Note:** real hindcasts need a bank — P3 proves the machinery on test
fixtures; first real exercise is P4 ("nothing exists yet for it to grade
except founding banks — that is the point," MD§8.3).

## Deliverables

`graders.py` (split checks, hindcast frame, scorecard math, gauntlet
harness); the retriever's **push core** as a library (`retriever.py`, push
path only); grader crew instruction material
(`src/kapso/learning/crews/grading/` — lead prompt + report-writer /
verifier / scorecard-assessor defs, lifted from GS§6); `kapso learn grade`;
`learning.graders:` config block (GS§5: `score_band` 0.20,
`min_settlements` 2, `calibration_min` 20, `calibration_buckets` [0.4,0.7],
`gauntlet.stability_tolerance` 0.10, `crew.*`, and `learning.ab` keys —
values are the GS§5 proposals, config is the single source).

## Work items

1. **Split manifest** (GS§3): `split.yaml` schema + frame checks (every store
   trajectory exactly once; no family on both sides; version bump carries a
   rationale); scorecards stamp `split_version`; write split v1 over the
   imported corpus (family+time, learn ≈ 50 / held-out ≈ 15 per MD§4.4 —
   actual counts from the D1-resolved corpus).
2. **Retriever push core** (MD§5.1, push only — a **pure function of (task,
   bank_head)** so hindcast can replay it; no agent inside): eligibility =
   task ∈ scope; quarantine excluded (decoys, `retired/`); reliability order
   with ledger-derived coverage discounts; `tags` never eligibility; k-caps
   per kind (2–4, config); served-card projection render; serving record
   (push stamp); gap analysis. Vectorless (MD§5.1). Probe rider and pull
   tools are explicitly **not** here (P6, P5).
3. **Hindcast frame** (GS§1, §6.6): staging (bank checkout read-only, push
   replay → `brief.md` + record, eligible claim set, outcome enumeration
   from the mined view); marker-grammar parser (the twelve markers);
   corridor centers per GS§1.2–1.4 formulas; null rules (§0.2); admission
   checks (§1.6) incl. liftable-settlement form; one repair bounce.
4. **Grader crew** (GS§6): lead prompt; report-writer (history-blind — one
   trajectory, no scorecards), verifier (five check classes, NOVEL re-search
   first), scorecard-assessor (the only whole-set view); parallel
   report-writers; exam mode (single trajectory, writer+verifier only).
5. **Scorecard math** (GS§2): per-dimension mean ± SE with per-trajectory
   values and null counts; **paired deltas on the same split_version only**;
   calibration pooling by claimed-reliability bucket; verdict block
   (accept | reject | within-noise); **frame recomputes all arithmetic —
   agent numbers never trusted for math** (GS§6.6).
6. **Gauntlet harness** (GS§2.3): duplicate-trap fixture builder (reworded
   clone, same run ids), stability comparator (substance diff: touched-card
   set, verdicts, transitions, scores within tolerance), control-run
   orchestration — with the crew invocation a CLI-boundary black box
   (first real trap executions are P4 gate items). `gauntlet.md` artifact
   with per-trap `{verdict, rationale}`.

## Tests

- The GS§6 worked example end to end: foresight 0.45 admitted (center 0.40),
  accuracy 0.80 (5 settlements), serving 0.40 admitted / 0.50 **rejected**
  (center ≈ 0.22, band 0.20), overall corridor [0.20, 1.00].
- Null-vs-zero: empty denominator → null required, fabricated number
  rejected; < min_settlements → accuracy null.
- Marker grammar: unknown marker rejects; `MISS-UNCARDED` without a
  resolving learn-set ref rejects.
- Split checks: family on both sides trips; missing trajectory trips.
- Push purity: same (task, bank_head) → byte-identical brief + record.
- Scorecard: paired-delta math on fixtures; cross-split pairing refused.

## Done gate

All fixture tests green; `kapso learn grade` runs a full fixture-bank pass
producing reports + gauntlet + scorecard artifacts in the D6-resolved home.

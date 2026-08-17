# P6 — Probe economy + A/B arms

**Goal:** the bank starts buying measurements: VoI-ranked probes ride
campaigns under a hard cap, and the A/B runner can certify a learner
generation against its incumbent (MD§5.1 probe bullet, GS§2.5, MD§7 rungs
3–4). **Design sources:** MD§5.1 (probe queue, budget, return path — no
evolve job), MD§4.3/UC§1.1 (probe-check seeding, prospective label), GS§2.5
(the A/B protocol), GS§2.4 (verdict block). **Depends on:** P5. **Doubts:**
D4 (arm campaigns use user-owned model config), D7.

## Work items

1. **Probe queue** (MD§5.1): `index/probe-queue.md` recompiled by the frame
   each update run — VoI rank: uncertainty × serving exposure first, then
   boundary tests that would split a scope, then blocked lifecycle
   decisions. Inputs are all ledger/serving-record derived — no agent in the
   ranking.
2. **Probe rider** (MD§5.1): the retriever attaches at most
   `learning.probe_budget` (v1: one probe slot — hard cap, "learning never
   cannibalizes doing") rendered into the lens planner/replanner prompt
   material as the design words it ("unverified on this family; `probe:`
   says how to test it in one fold"). This rendering rides the P5.2 gated
   surface — no new evolve obligations (uptake voluntary).
3. **Return path** (MD§5.1/§5.3, already frame-side in P4's admission):
   verify end to end on a real campaign — settlement found, flow↔spec match
   awarded the **prospective** label (critic-verified), ambiguous match
   downgraded; uptake rate appears in the health panel.
4. **A/B runner** (GS§2.5): candidate-head arm vs incumbent-head arm (two
   refs of one repo; first generation's incumbent = empty brief),
   **same-task pairs**, same kapso commit/budget/config; paired per-task
   deltas ± SE + the guard KPI (no regression where the bank is thin);
   feeds the scorecard's transfer row. Config `learning.ab`: `required`
   (v1: false — waivable; not-run is recorded as not-run, never as passed),
   `pairs` (v1: 5); when required, within-noise **blocks**. Trigger: a
   generation's final exam after sustained rung-2 wins — the runner is
   invoked, never scheduled.

## Tests

- VoI rank determinism from fixture ledgers; cap enforcement (a second probe
  never attaches).
- Prospective-label path: matched fixture flow earns it; perturbed spec
  match downgrades.
- A/B math: paired deltas + guard KPI on fixture results; `required` +
  within-noise → blocked; waived → `not-run` recorded.

## Done gate

One live campaign served a probe under budget; one A/B pair executed end to
end (may be a small-budget real pair — arm campaigns are real evolve runs and
their cost is a user go/no-go at the time).

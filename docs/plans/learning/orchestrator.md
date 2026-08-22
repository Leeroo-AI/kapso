# Learning implementation — orchestrator

The single coordination document for implementing the learn-from-trajectories
design. Everything here derives from the design docs; nothing is invented at
plan level. Where the design under-determines an implementation choice, the
item is listed in the **doubt register** below and blocks only its own work
item, never the phase.

**Design sources (the only authorities):**
`docs/research/learn-from-trajectories-design.md` (main; cited as MD§…),
`…-mining-prompts.md` (MP), `…-grader-scoring.md` (GS), `…-update-crew.md`
(UC), `…-codify.md` (CD), `…-litreview.md` (background only, no work items).

**Plan hierarchy** (this dir): `p1-trajectory-store.md` → `p2-mine-corpus.md`
→ `p3-grader-suite.md` → `p4-bank-update-crew.md` → `p5-serving.md` →
`p6-probes-arms.md` → `p7-codify.md`, plus the cross-phase
`behavior-tests.md` (the semantic production suite — scenarios land with
their phases per its mapping table). Order is MD§8's forced build order —
*the graders exist before the thing they grade* — with two clarifications
the phase texts carry: the retriever's **push core is P3 infrastructure**
(the hindcast compiles would-have-been briefs with the real push path, GS§1;
P5 only wires serving into live campaigns), and the **gauntlet's trap
executions begin in P4** (traps invoke the update-crew CLI as a black box,
GS§2.3/§6.6 — P3 builds the harness, P4 gives it a crew to trap).

## Dependency graph

```
P1 store ──► P2 mine corpus ──► P3 graders (split · push core · hindcast ·
   │                                scorecard · gauntlet harness)
   │                                   │
   │                                   ▼
   │                            P4 bank + update crew  ◄─ develop AGAINST P3
   │                                   │
   │                                   ├──► P5 serving (pull tools · stamping ·
   │                                   │       exam-before-lesson live)
   │                                   ├──► P6 probes + A/B arms   (needs P5)
   │                                   └──► P7 codify path         (needs P4;
   │                                            placement needs D8)
   └── gated runner-capture item (P1.G) — approval required (doubt D5)
```

## Phase gates

A phase is DONE only when: its work items are committed (one coherent commit
per item, CLAUDE.md Rule 8), its tests are green **and added to the curated
hermetic suite** (doubt D7), its design-conformance checks pass, and its
named human checkpoint (if any) is signed off. Gates:

| Phase | Gate |
|---|---|
| P1 | the D1 curated subset imported (concrete list signed off); double-`save_trajectory` idempotent; corrupt manifest raises |
| P2 | mined views for every imported trajectory; mining report coverage arithmetic green; **human review of first mined views** (MD§8.2) |
| P3 | hindcast runner admits/rejects the GS§6 worked example exactly (0.45 admitted, 0.50-serving rejected at band 0.20); split checks trip on a family straddling sides; scorecard math reproduces fixtures |
| P4 | founding bank passes OKF conformance; crew v1 run end-to-end on a learn batch; **human review of first bank commits** (MD§8.4); duplicate+stability traps executed against the crew; keep-best ledger of crew versions started |
| P5 | founding-cards-vs-static-notes ≈ neutral validated (MD§8.5); exam-before-lesson live on one real campaign |
| P6 | first probe served under budget; A/B runner executes a candidate-vs-incumbent pair end to end |
| P7 | forward-gate-class procedure codified on a real card: green codify run, flip in transaction, freshness re-run green |

From P3 onward a phase gate additionally requires **its behavior scenarios
green** (`behavior-tests.md`): semantic production tests of the real
machinery on known-truth fixtures, judged by an agentic reviewer — a FAIL
blocks exactly as a gauntlet gate does.

**Checkpoint waiver (user directive 2026-08-17):** the remaining human
checkpoints (P4 first-bank-commits review, behavior truth.md sign-offs) are
delegated to rigorous self-review — performed, documented in the run
artifacts and ledger, and reported after the fact; the build continues
without blocking on them. The P2 mined-view review was performed by the user
(accepted).

## Standing rules (bind every implementer)

CLAUDE.md Rules 1–11 verbatim — most-cited here: config-single-source (every
knob in the `learning:` block of `src/kapso/config.yaml`, read via
`load_config`; never a re-hardcoded literal), **no try/except** (validate,
fail loud — the design's fail-loud frames match this), no env vars, imports
at top, no back-compat shims (Rule 7), minimalism, tests that earn their
place (Rule 9 — the plans name the regression each test catches). Repository
practice: commit per completed item; feature branch never pushed unprompted;
the curated hermetic test gate grows with each phase. **Framework-core
changes** (anything under `src/kapso/` outside `learning/`, and
`benchmarks/relbench/runner.py`) are surfaced before patching — the gated
items P1.G and P5.2 exist for exactly this.

## Doubt register

All doubts resolved 2026-08-17 (user review). Resolutions are binding.

| # | Doubt | Resolution |
|---|---|---|
| D1 | Corpus inventory | **RESOLVED: curated related subset**, chosen to make the learning effect visible. Selection rule: 2–3 task families with the densest multi-run history spanning ≥2 datasets that share a task type (the churn / entity-classification cluster), plus one sibling family reserved for held-out; target 15–25 trajectories. The concrete list is enumerated from RESULTS.md at P1 import and posted for sign-off before download. |
| D2 | Store location | **RESOLVED:** local-first store for P1–P4; remote prefix (`gs://leeroo-kapso-relbench-artifacts/trajectories/`) created at P5. |
| D3 | Bank repo host | **RESOLVED:** local git repo at P4; private `kapso-bank-relbench` under **Leeroo-AI** at P5. |
| D4 | Models per role | **RESOLVED, one structural constraint:** default intelligence = **codex CLI, GPT-5.6, xhigh** for worker and judgment roles (card-writer, the docket specialists, report-writer, assessors, the codify-run implementor), invoked by leads via `codex exec`. Crew **leads stay on Claude Code** — self-organization runs on the CLI's native subagents (UC§0), which codex lacks; critic/verifier run cross-model (Claude) so the diversity check holds. Per-role config is `{cli, model, effort}`, so the later switch to all-Claude/Fable is one config change. |
| D5 | Framework-core approvals | **RESOLVED: approved** — P1.G (runner capture) and P5.2 (§5.3 evolve edits) proceed at their phases, diffs shown before merge. |
| D6 | Run-dir home | **RESOLVED:** repo-root `learning/` (gitignored); reports synced to the artifacts bucket. |
| D7 | Test gates | **RESOLVED:** phase tests join the curated hermetic suite as they land — **and** the behavior suite (`behavior-tests.md`) is part of the production gate: agentic review of semantic correctness (routing, abstraction, merging, generalization, resolution, reliability, hindcast, serving, codify/replay, end-to-end learning effect), FAIL blocks promotion. |
| D8 | Codify placement | **RESOLVED:** implement `gcp_ephemeral` at P7 **from the start** — relbench work needs external machines and the dev box is not a run host; `target: local` retained for harness tests only; machine types from config (the campaign machine class). |

## Status ledger

| Plan | Status | Last commit |
|---|---|---|
| P1 trajectory store | GATE CLOSED — 19/21 imported (2 failed validation with named findings: item-churn archive lacks its work dir; one user-repeat tarball carries two work dirs); import report in learning/imports/ | 578c619d |
| P2 mine corpus | GATE CLOSED at the 8h deadline (user directive): 11/19 views mined, 0 validation failures — all heavies included (both big rel-amazons + wave-4); 8 unmined bundles parked at data/trajectories-parked/ (re-adoptable); dual-account budget switching proved live (one auto-switch executed) | 1650aed2 |
| P3 grader suite | code + tests done; split v2 committed after the 8h deadline (5 learn / 6 held-out over the mined corpus; v1 superseded) | 578c619d |
| P4 bank + update crew | frame code + tests done; **founding bank authored + committed** (data/kapso-bank-relbench.git, tag lr_founding_20260818: 7 insights + 3 procedures + 2 decoys; one confirm-grade anchor, the rest honest `exercise`). Delegated self-review (checkpoint waiver): OKF conformance 0, §5.2 admission green incl. re-grep of every quoted number, bound-identifier lint clean, serving smoke correct — findings fixed two checker gaps (decoy conformance skip; negation-aware usage claims). Develop pre-flight: codex model name is `gpt-5.6-sol` on this account (`gpt-5.6` 400s; config fixed, xhigh smoke green). **crew_v1 development run COMPLETE** (attempt 1 died on the exam split/chronology leak — caught live, fixed in d58e9008; attempt 2 ran end-to-end ~7.2h): 5 learn exams (batch-0 null baselines honest; batch-1 foresight 0.82/0.83), 2 update runs (gen-1: 12 cards spawned; gen-2 repair round caught a derived-number fabrication — admission gate worked live), 6 held-out exams all admitted. Scorecard: foresight **0.8717** (se .03), accuracy 1.0 (thin), serving **0.3733** — binding; assessor verdict **reject** (serving delivery+uptake), keep-best ledger row written, incumbent stays null. The reject is the system working: bank strong, push-only serving quantified as the gap (P5 pull tools are the designed answer). **Gauntlet EXECUTED against crew_v1** (2026-08-18, standalone runner): duplicate trap **PASS with distinction** — on a verbatim clone the crew spawned nothing AND attached nothing (recognized same-measurement re-badged, refused to double-count); stability trap **FAIL** — the batch-1 re-run produced a DIFFERENT card set (1 vs 3 spawns + one state divergence) → rolled **FAIL**, the second independent crew_v1 rejection, and future develop runs carry the rollup in-scorecard where FAIL forces reject. crew_v2 levers committed: scope-as-serving-contract + serving null/zero boundary (b49df37e) + mechanism-clustering-before-delegation, the stability contract, in the lead prompt. **P4 gate: all items now executed** | d58e9008 |
| P5 serving | **code-complete**: pull tools through the gated-MCP registry (bank_search whole-set shortlist / bank_get named refusals, quarantine unmarked, JSONL pull log — 1c39c62e); launch staging + §5.3 gated evolve edits live (brief replaces the static notes, citation contract + cards_load_bearing, bank_head stamped frame-side, judge tool-lock structural — b1f7b20d); `kapso learn ingest` operating chain (c5cf7cda). Pending: acceptance campaign (needs external machine + user window) + neutrality check; **D3 COMPLETE** (2026-08-18: user refreshed the PAT — the earlier 401 was also a quoted-.env parsing artifact): private Leeroo-AI/kapso-bank-relbench created, main + founding tag pushed, config bank.remote set (eaca57ae). Production-test window OPEN (user: "free to run production tests, don't ever think of costs"): behavior suite running on acct2; feat/relbench-learning pushed to origin for box clones; arm boxes (4xA100 SPOT, rel-hm user-churn, serving-on vs static-notes — the P5.5 neutrality pair) bootstrapping. **Arm-C (control) COMPLETE** 2026-08-18T22:52: val roc_auc **0.71974** / test **0.71636** — near-identical to the July no-serving baseline (0.71957/0.71623), a stable control; harvested strict-contract with kapso_commit b49df37e, bank_head null; bundle pulled to the dev store, box torn down. **Arm-T (serving, founding bank) COMPLETE** 2026-08-19 after FIVE SPOT preemptions ridden by an automated capacity-chase/resume shepherd (checkpoint-resume banked every window; zones a->c): val roc_auc **0.720234** / test 0.715986. **P5.5 NEUTRALITY: CONFIRMED** — ab_verdict {within-noise, delta +0.000493, guard clean on a thin (unvisited) dataset, n=1 never certifies}: founding cards ≈ static notes exactly as designed, with no serving-caused regression. **P5 done-gate satisfied**: one real campaign briefed + tooled end to end — brief served (5 cards, bank_head 8782bc2a stamped through manifest+meta), bank tools mounted in every ideation session, pull uptake ZERO this campaign (honest health-panel datum), trajectory harvested strict-contract with the FULL pre+post-preemption log and pulled to the store. All arm GCP resources torn down. Split v4 (both arms on the learn side). Codex auth rotated mid-window (device flow; old account rode arm-C untouched). **2026-08-22 design amendment:** knowledge slot flipped to ADDITIVE (user decision) — the brief appends after the permanent static notes instead of replacing them; candidate arms now measure marginal value over the incumbent's full context. The replacement-mode neutrality result above predates the flip; no certified wave ran under replacement with the v2 bank (wave A aborted 78 min in) | eaca57ae |
| P6 probes + arms | frame halves done: probe queue (VoI tiers, ledger-derived) + push rider under probe_budget=1 hard cap (6e447972); A/B verdict math with thin-task guard + gate semantics (49451b7b). Pending: live probe under budget + a real A/B pair (operator go/no-go) | 49451b7b |
| P7 codify path | **code-complete**: seeder live (executed-verdict filter, closure recurrence w/ set-dedup, failed-attempt clock — b9c10fac); procedure-codifier role (compatibility gate before any machine); transaction rule enforced (no green run, no flip — ee92059a); reproduction gates (decisions exact / metrics banded / artifacts checked / anti-weak-test / actually-invoked — a687a6ae); codify-run driver (evolve minus ideation, judge-veto feedback loop) + gcp_ephemeral SPOT placement w/ unconditional teardown + `kapso learn codify` (543b5715); expiry seeding for sightings + code freshness (7edcc2b3); merge code-inheritance + sweeper demote contracts (a43c8b3c). Pending: done-gate live run (real card codified e2e on GCP + freshness re-run) — needs the live window | a43c8b3c |
| Behavior suite (cross-phase) | **Pass 1 EXECUTED live** (2026-08-18, real crews + cross-model reviewers): **PASS — B10 learning-effect, B2 abstraction, B3 merge (decoy declined), B4 generalization**. Four FAILs, each converted to a committed fix: B1 machinery (repair budget→2, 242bb513 + reserved-words contract 6342349b), B5 reviewer (fixture answer-sheet leak removed + never-merge-a-tension doctrine, 9cb89737), B6 reviewer (Step-D reassessment enforced by the frame, reliability now claim-layer, fixture corridor-coincidence removed — caa34031/6f1177f7), B7 reviewer (never-bin-away-a-miss writer duty + verifier MIS-BINNING class, 358690be). Reruns EXECUTED: **suite now 8/8 PASS** (B5/B6/B7 green under the committed fixes; B1 green after its second reviewer finding — the fixture invented phenomena its anchors didn't attest, regrounded in real bundle lines 0e16375a — with the reviewer confirming the crew resisted the lexical trap on mechanism grounds). Ten live findings total across the suite, every one converted to a committed frame/crew/fixture improvement. **The behavior gate (D7) is GREEN** | 0e16375a |
| crew_v2 gauntlet | duplicate **PASS** again (verbatim clone: 0 spawns, 0 attachments). Stability **FAIL — but a far narrower miss than v1, and half-confounded**: (a) the two runs converged on the SAME two mechanisms and minted DIFFERENT names (pairwise near-twins) — the clustering lever fixed what gets carded; instability moved to naming, which the name-keyed diff rightly charges (attach-target stability). crew_v3 lever committed: derivable-not-creative naming in the card-writer. (b) The six-card version drift is a frame artifact — run A predates the Step-D/claim-layer upgrades, run B ran under them; crew_v3's fully-post-upgrade run gives the clean stability read. Scorecard records reject-by-gate (gates dominate — correct conservatism) | — |

Update the status ledger in the same commit as the work it describes.

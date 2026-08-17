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
| P2 mine corpus | smoke-mine v2 GREEN on wave-4 (19 flows, all 22 runs claimed, repair loop exercised; two crew fixes landed: foreground-only delegation, refs label collision); mined view awaiting human review, then batch mining | 1650aed2 |
| P3 grader suite | code + tests done; split v1 committed (12 learn / 7 held-out, validated against the store) | 578c619d |
| P4 bank + update crew | frame code + tests done (bank read/write model, diff invariants + evidence admission, update frame + crew instruction set, development driver, keep-best ledger); founding bank + real crew runs await mined corpus | fe767ac6 |
| P5 serving | push core done early (P3, hindcast dependency); pull tools + evolve edits pending | 09b6954c |
| P6 probes + arms | not started | — |
| P7 codify path | not started | — |
| Behavior suite (cross-phase) | runner done (serve/grade-exam/update machineries + reviewer gate); fixtures land with phases | b4b9307d |

Update the status ledger in the same commit as the work it describes.

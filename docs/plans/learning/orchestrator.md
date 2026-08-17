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
`p6-probes-arms.md` → `p7-codify.md`. Order is MD§8's forced build order —
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
| P1 | 64-archive corpus imported; double-`save_trajectory` idempotent; corrupt manifest raises |
| P2 | mined views for every imported trajectory; mining report coverage arithmetic green; **human review of first mined views** (MD§8.2) |
| P3 | hindcast runner admits/rejects the GS§6 worked example exactly (0.45 admitted, 0.50-serving rejected at band 0.20); split checks trip on a family straddling sides; scorecard math reproduces fixtures |
| P4 | founding bank passes OKF conformance; crew v1 run end-to-end on a learn batch; **human review of first bank commits** (MD§8.4); duplicate+stability traps executed against the crew; keep-best ledger of crew versions started |
| P5 | founding-cards-vs-static-notes ≈ neutral validated (MD§8.5); exam-before-lesson live on one real campaign |
| P6 | first probe served under budget; A/B runner executes a candidate-vs-incumbent pair end to end |
| P7 | forward-gate-class procedure codified on a real card: green codify run, flip in transaction, freshness re-run green |

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

Every under-determined point, with the proposed default. OPEN doubts block
only their own work item. Confirmed answers get recorded here (status →
RESOLVED: answer) and are then binding.

| # | Doubt | Proposed default | Status |
|---|---|---|---|
| D1 | **Corpus inventory.** RESULTS.md carries 64 per-run `.tgz` refs across waves. Import all that pass bundle validation, or curate (e.g., drop earliest-wave bundles with thin contracts)? | Import **all recoverable**; mining's explicit-gap policies (MP) absorb thin bundles | OPEN |
| D2 | **Store location + duplicate storage.** MD§3.4 reuses the artifacts bucket. Propose store prefix `gs://leeroo-kapso-relbench-artifacts/trajectories/` (unpacked, manifest-last) while the original `.tgz` stay — storage duplicates. Alternative: local-only store during development, remote sync later. | **Local-first store** for P1–P4; remote prefix created at P5 | OPEN |
| D3 | **Bank repo host.** MD§3.1: standalone private repo `kapso-bank-relbench`. Which org (Leeroo-AI?), and when — P4 can run on a local git repo, remote needed for multi-box serving at P5. | Local git repo at P4; private GitHub repo under the org at P5 | OPEN |
| D4 | **Models per role** (user-owned decision, standing rule). Crews are Claude-led (UC§0); critic/verifier "second model where affordable". Which concrete models for lead / card-writer & report-writer / critic & verifier / assessors? | Placeholders in config, values set by you before P2's first mining run | OPEN |
| D5 | **Framework-core approvals.** (a) P1.G: runner-capture additions — archive the selector pool, workspace `.kapso`, shared cache (MD§3.4.1). (b) P5.2: the §5.3 evolve edits — push brief replacing the two static context constants, gated-MCP preset entries, citation-contract paragraph, judge `cards_load_bearing` field, `bank_head` stamp. Approve implementing each when its phase arrives? | Yes at their phases, with diffs shown before merge | OPEN |
| D6 | **Learning run dirs home.** UC§1/GS§6.1 use `learning/runs/<lr_id>/`, `learning/graders/<stamp>/`. Propose repo-root `learning/` (gitignored) with optional bucket sync of reports. | repo-root `learning/`, gitignored; reports synced to the artifacts bucket | OPEN |
| D7 | **Curated hermetic suite.** New per-phase test modules join the curated gate list as each phase lands? | Yes, phase-by-phase | OPEN |
| D8 | **Codify GCP provisioning** (CD§3): `gcp_ephemeral` target needs machine-type defaults + billing sign-off; only bites at P7. | Defer decision to P7 start; P7 begins `target: local` | OPEN |

## Status ledger

| Plan | Status | Last commit |
|---|---|---|
| P1 trajectory store | not started | — |
| P2 mine corpus | not started | — |
| P3 grader suite | not started | — |
| P4 bank + update crew | not started | — |
| P5 serving | not started | — |
| P6 probes + arms | not started | — |
| P7 codify path | not started | — |

Update this table (and the doubt register) in the same commit as the work it
describes.

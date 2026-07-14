# Semantic review findings — live system-debugging runs

Working log of logic/semantic/wiring issues observed while monitoring Kapso on
RelBench. Each finding: what was seen (with run/timestamp), why it matters,
proposed fix, status. Framework-core fixes require user approval before applying.

Runs reviewed:
- R1 2026-07-14 12:13 rel-f1/driver-position, RELBENCH_CONFIGS (xhigh), 12 iter planned, killed at ~3h/iteration 2 (operator stop).
- R2 2026-07-14 ~15:5x rel-f1/driver-circuit-compete, FAST_DEBUG, 3 iter — the run this log primarily tracks.

## Confirmed and fixed during R1 (already merged)

1. **Coder-session PATH resolved to the wrong interpreter** — claude_code Bash
   subprocesses inherited the login PATH (base conda, no relbench). Agent
   self-recovered by probing conda envs (wasted turns). Fixed: handler prepends
   sys.executable's bin to PATH. (d5680b40)
2. **RepoMemory hardcoded temperature=0** — gpt-5.6-luna rejects it; every
   repo-memory update failed (warn policy hid it). Fixed on approval. (b92f3a08)
3. **Debug-mode contract under-specified "small"** — iteration 1 trained a
   stacked ensemble inside the 15-min debug gate (799s) and was killed before
   producing predictions; total iteration loss (~85 min xhigh work). Fixed:
   hard rules in the contract (one small model, no stacking, half-budget
   target). (6fbd4340)
4. **Search was blind to experiment history** (the "ignored iteration 0"
   issue) — ideation/selection/pruning prompts received empty additional_info;
   only the coder saw errors. A failed lineage is also score-poisoned (sentinel
   worst score), so exploitation would never revisit near-working branches.
   Fix B applied on approval: compact history digest (branch, score/failure
   reason, official metrics line, solution one-liner) injected into
   ContextData.additional_info each iteration. (943bd039) Fix A (feedback for
   all-failed iterations via returning the failed node) declined as redundant.

## Open findings (R2 onward)

5. **Selection reasoning is generated but never logged.** The select() prompt
   demands per-choice one-line reasons; only the parsed id list is used, the
   reasons are discarded. We cannot audit *why* the search picked a node.
   Proposed: print the selector's raw reasoning block (or store on the node).
   Core change (benchmark_tree_search) — needs approval. Status: OPEN.

6. **History digest fired only from iteration 3 (R2).** Fix B injects the
   digest at the top of run(); iteration 2's ideation happened in the same
   run() call as iteration 1's evaluation (expansion precedes evaluation in
   the loop), so the first digest the LLMs saw was in iteration 3's expansion.
   Semantics are correct from iter 3 onward (exploitation correctly chose the
   scored node 9 as parent); just be aware the first two iterations of any run
   are digest-blind. Minor; no change proposed. Status: NOTED.

7. **Val-selected ≠ test-best is real and visible (R2).** run_0002 won on val
   (0.7314 vs 0.7275) and scored test 0.8804; iteration 3's fancier ensemble
   dropped val to 0.6769 (over-engineering on 27 rows) and was correctly NOT
   selected. Protocol behaved exactly as designed. Status: WORKING AS INTENDED.

## R2 outcome (for the record)

3/3 iterations completed, all scored, zero contract violations, clean audit.
Val MAP: 0.7275 -> 0.7314 (winner, run_0002) -> 0.6769. Final report test MAP
**0.8804** (independently recomputed from archived predictions against the
pristine cache: 0.8804 — handler and recompute agree).
Published test bar (v2-paper ID-GNN): 0.7618. Our val-selected test result
exceeds it by +11.9 MAP points on this task. Wall-clock ~65 min for 3
iterations (FAST_DEBUG, medium effort); ~$2-3 total.

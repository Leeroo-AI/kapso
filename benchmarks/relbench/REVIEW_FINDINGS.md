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
   Status: FIXED 2026-07-15 on written-proposal approval — select() and
   prune_bad_solutions() log the reasoning emitted before <output>.

6. **History digest lags candidate generation (CORRECTED after R3).** The
   digest is injected at the top of each run() call, so ideation/selection in
   call N see every evaluation from calls 1..N-1 — it is live from iteration 2
   onward (this note originally claimed iteration 3; that was wrong). The
   residual lag: candidates GENERATED in call 1 (digest-blind) remain in the
   selection pool alongside digest-aware ones, and children generated within a
   call never see that same call's evaluation. Minor; no change proposed.
   Status: NOTED (corrected 2026-07-15).

7. **Val-selected ≠ test-best is real and visible (R2).** run_0002 won on val
   (0.7314 vs 0.7275) and scored test 0.8804; iteration 3's fancier ensemble
   dropped val to 0.6769 (over-engineering on 27 rows) and was correctly NOT
   selected. Protocol behaved exactly as designed. Status: WORKING AS INTENDED.

8. **Selection prompt carries no lineage — the selector cannot exploit
   scores-by-ancestry (R3).** Iteration 3's exploitation correctly expanded
   BOTH scored nodes (checkpoint ground truth: children 20-22 under node 11
   [GBDT panel, val MAE 2.983]; children 23-25 under node 9 [state-space,
   val 2.684, best]). The LLM selector then chose node 22 — a child of the
   WEAKER lineage — over all three children of the val-best lineage. Root
   cause: select()/prune candidates are rendered as "id + solution text"
   only; parent identity and parent score are absent, so the selector cannot
   tie a candidate to the history digest entries (and per finding 5 its
   stated reasons are discarded, making the choice unauditable). The winning
   state-space insight survived only because the chosen spec textually
   borrowed it. Status: FIXED 2026-07-15 on written-proposal approval —
   _candidate_line() renders every select/prune candidate with its lineage
   ("root" / "child of <branch>, parent score X" / "child of unscored node
   N"); unit tests cover the three cases, prompt content, and id parsing.

9. **Repo memory: writes fine — the READ path is mis-wired (investigated
   2026-07-15 on user report).** Suspected credential failure ruled out:
   every R2/R3 experiment branch carries its "chore(kapso): update repo
   memory" commit, zero failure warnings, file grew 3.6KB baseline -> 52KB
   with all 7 book sections populated (gpt-5.6-luna via OPENAI key working;
   the R1 temperature=0 breakage stayed fixed). What made it look unwritten:
   (a) the workspace's main branch never gets update commits (by design —
   updates land on experiment branches at session close); (b) every session
   logs "RepoMemory sections consulted: none". Root cause of (b): the
   implement prompt's claude_code branch advertises the MCP tool
   `get_repo_memory_section(...)`, but the benchmark path never mounts any
   MCP server (no mcp_servers in config; zero mcp__ calls in logs) — the
   advertised tool does not exist, the JSON fallback was used once in all of
   R3, and the consulted-sections extractor only detects MCP usage. The
   2.5KB summary+TOC is still injected into every implement prompt, so the
   memory is not inert — but detail access is dead and unobservable.
   Status: FIXED 2026-07-15 (user chose mounting over instruction-gating):
   ExperimentSession now mounts the repo-memory MCP gate against the
   session's own clone for MCP-capable agents (REPO_MEMORY_ROOT = session
   folder; generic-path sessions with their own gate set pass through
   untouched), and the implement prompt selects MCP vs JSON-file
   instructions from the session's actual mount state. Live verification
   (V1, 2026-07-15, 2-iter FAST_DEBUG on driver-circuit-compete): wiring
   proven at every layer — per-session mcp_config.json written, --mcp-config
   + all three mcp__gated-knowledge__ tools on the live CLI cmdline, server
   starts clean with the configured env. Note: the adapter's "[init]
   tools=N" count excludes MCP tools (red herring). Behavioral outcome:
   zero tool calls in this run — with wiring proven, non-usage is now a
   content/emphasis signal (the 2.5KB summary+TOC already in the prompt
   evidently sufficed for a 27-row ranking task), cleanly separated from
   the wiring question for the first time. Run itself: 2/2 scored (val MAP
   0.6938/0.6964), test MAP 0.8634, clean audit.
   Cross-track: PostTrainBench run7 review F4 reports the same marker
   ("sections consulted: []") but attributes it to dead litellm
   side-channels under OAuth-only auth (their writes genuinely fail — the
   accepted trade-off there). Our track proves that diagnosis incomplete:
   with writes fully working (52KB enriched memory), the channel still
   reads "none". Two independent, stacking causes — write-side credentials
   (theirs) and read-side MCP mis-wiring (both tracks). Fixing their
   credentials alone will not light the channel.

10. **Generic shakedown #1: maintainer calibration requires a candidate that
   does not exist yet (2026-07-16).** The registration transaction authored
   kapso_eval.py correctly (immutability + stdout passthrough verified in
   its output), but calibration runs a fast-fidelity evaluation at setup —
   in a from-scratch search workspace there is no main.py, so the grader
   exited with a contract violation and the whole run aborted before
   iteration 1. Fixed (benchmark-local): --strategy generic auto-seeds the
   workspace with data/generic_baseline/main.py (shape-correct trivial
   predictions for all four families) via initial_repo, giving calibration
   a runnable candidate and parent_policy=best a legitimate first parent.
   Status: FIXED; re-shakedown pending.

11. **Generic archives are not 1:1 with nodes (R4, minor).** The provided
   grader archives every full-fidelity invocation; the iteration-1 agent ran
   the registered evaluation twice, producing duplicate archives
   (run_0003/run_0004, identical val 0.7585). Harmless for selection
   (best-by-val) and claims, but inflates archive counts; a dedup by
   prediction hash in the grader would tidy it. Status: NOTED.

12. **Archived `__pycache__` poisons seeded workspaces (A/B run, harness
   tail).** Both code-snapshot archivers (tree handler + provided grader)
   copied `__pycache__/*.pyc` into `runs/*/code`. Arm B was seeded from such
   a snapshot (`run_0002/code`), so ExperimentSession committed the stale
   bytecode as TRACKED files; later runs regenerated it, and the final
   `checkout_to_best_experiment_branch()` refused to switch branches over
   the dirty tracked files (`WorkspaceCheckoutError` on `generic_exp_5`) —
   runner exit 1 AFTER the search completed. Zero data loss: all 8 full-run
   archives intact; only B's in-process final report was lost (recomputed
   offline from archives). Status: FIXED benchmark-side — both archivers
   now skip `__pycache__`; existing archives and the committed claims
   snapshot stripped of bytecode. Framework-side hardening (workspace
   `.gitignore` for `__pycache__/` at session init) PROPOSED, awaiting
   approval per the framework-change rule.

## R5 — Phase-5 A/B: tree vs generic, rel-f1/driver-position (2026-07-16/17, COMPLETED)

Pre-registered design: both arms seeded from the R3 champion (val 2.684 /
test 3.779), 10 iterations each, run sequentially on the same box. Arm A =
benchmark_tree_search (RELBENCH_CONFIGS), Arm B = generic champion-chain
(RELBENCH_GENERIC); deployed-best vs deployed-best (each arm's val-selected
champion, test revealed once at the end).

| | Arm A (tree) | Arm B (generic) |
|---|---|---|
| Val trajectory (best) | 2.6506 → **2.6104** (run_0009) | 2.6818 → **2.6434** (run_0019) |
| Deployed test MAE | **3.8367** (worse than seed 3.779) | **3.7429** (beats seed) |
| Test NMAE (÷7.0253) | 0.5461 | 0.5328 |
| Unique candidates / archives | 7 / 10 | 8 archives over 10 experiments |
| Wall-clock | ~8.6 h | ~7.3 h |
| Cost | $108.4 (all-in) | ~$125.4 (claude sessions; codex leg is sub-auth, unbilled) |

Verdict: **generic wins the deployment comparison** — the only comparison
that counts. A won validation (2.6104 vs 2.6434) yet DEGRADED the seed on
test (+0.058); B improved it (−0.036). The inversion is not just at the
selection point: A's candidate population tests at ~3.99 mean (7 unique)
vs B's ~3.80 (8 runs) — every B run except its first beats A's best test.
Within-arm val-selection also misfires: A's best-on-test was run_0012
(3.8001, val-rank 5), not the selected run_0009; B's was run_0018 (3.7346),
not run_0019 (3.7429). On a task with +1.1 val→test era drift, the tree's
wide pool + val-greedy selection over-fits the validation years; the
generic loop's few-but-deliberate candidates (strict-Pareto shipping gates,
frozen-validation arbitration, self-declared saturation at iteration ~6,
explicit "no test probing" invariants in its changes.log) generalized.
Caveats: n=1 task, 10 iterations, single seed — directional evidence, not
a theorem; and neither arm approaches KumoRFM-ft (2.731 test MAE, NMAE
0.3887) — driver-position stays unclaimed.

Campaign consequence: use the tree as the cheap breadth scout, escalate to
generic where a cell is close or drift-prone; on drift-heavy regression
tasks, distrust raw val-greedy selection (prefer drift-aware validation
schemes — B's frozen-era arbitration is the template). Official
driver-position row remains val-selected across all archives = run_0009,
test 3.8367 (protocol over cherry-picking; B's 3.7429 documented here).

## R4 review — generic-mode shakedown (2026-07-16, COMPLETED)

rel-f1/driver-circuit-compete, RELBENCH_GENERIC, 2 iterations, first run on
the experiment OAuth token. Reviewer observations at iteration-2 midpoint:

- **Registration/calibration**: maintainer authored the delegating
  kapso_eval.py (228s, $0.95), calibration passed on the seeded baseline
  (finding 10 fix verified live). No immutability violations by any session.
- **Score-of-record mechanics**: manifest parsed at the strategy layer;
  official val 0.7585 recorded; archives land val-only (in-loop grader
  physically cannot score test — quarantine-by-construction confirmed).
- **Knowledge channels ALIVE — the headline contrast with the tree path**:
  30 MCP gate calls in two iterations (9x get_repo_memory_section, 4x
  summaries/listings, 13x experiment-history queries incl. semantic
  search). The tree path made zero such calls across three full runs.
- **The pull-model loop works end to end**: feedback generator read the
  champion's code and root-caused a real defect (additive-dominance
  M_REPEAT=1e6 renders survival term inert); iteration-2 ideation quoted
  that feedback verbatim and proposed the targeted fix. This is the
  mechanism finding 8 could only approximate with lineage annotations.
- **Quality signal**: iteration 1 official val MAP 0.7585 — above the tree
  strategy's best-ever on this task (0.7314) in ONE iteration; its
  debug-mode self-estimate (0.758) matched the official score, i.e. the
  agent's internal measurement discipline is calibrated.
- **Cost profile as predicted**: ~$12-13/iteration (ideation $1.4-2.6 +
  implementation ~$8.9 + feedback $0.6) vs ~$1.5-2 tree FAST_DEBUG — the
  breadth-vs-depth economics are real; deploy generic where per-step
  quality pays (escalations).
- Iteration 2 midpoint: replacement scoring initially regressed
  (~0.71-0.72), agent self-diagnosing and pulling back toward the
  champion's career-frequency ingredient — parent_policy=best protects the
  0.7585 champion regardless of iteration-2's outcome.

**Completion verdict:** 2/2 iterations, COMPLETED cleanly, $24.44 total
(~$12/iteration as predicted). Iteration 2 never beat the champion (its
final evaluations landed at champion parity, 0.7585). final_evaluate
selected run_0003 (best val 0.7585 across the task's whole archive) and
filled test ONCE: **test MAP 0.8289** (independently recomputed: 0.8289
exact). Protocol note: earlier archived runs carry lower val but higher
test (0.6938/0.8687, 0.6964/0.8634) — tune-on-val discipline makes 0.8289
the claimable number, still +6.7 MAP over the published 0.7618. The 27-row
val split is too noisy to referee tree-vs-generic on test here; generic
decisively won VAL (0.7585 vs tree's 0.7314 best-ever, in one iteration).
The A/B on driver-position (760 test rows) remains the real referee.
Machinery verdict: every new component passed — Phase 4 CLOSED.

## R2 outcome (for the record)

3/3 iterations completed, all scored, zero contract violations, clean audit.
Val MAP: 0.7275 -> 0.7314 (winner, run_0002) -> 0.6769. Final report test MAP
**0.8804** (independently recomputed from archived predictions against the
pristine cache: 0.8804 — handler and recompute agree).
Published test bar (v2-paper ID-GNN): 0.7618. Our val-selected test result
exceeds it by +11.9 MAP points on this task. Wall-clock ~65 min for 3
iterations (FAST_DEBUG, medium effort); ~$2-3 total.

## R3 outcome (for the record)

rel-f1/driver-position (KumoRFM-ft's flagship regression cell), FAST_DEBUG,
3 iterations, ~70 min, ~$3: 3/3 scored, zero contract violations, clean audit,
debug gate held for all three candidates (R1's lesson confirmed at medium
effort). Val MAE: 2.983 (GBDT panel) -> 2.684 (hierarchical state-space —
winner) -> 2.749 (quantile ensemble, child of the GBDT lineage; see finding 8).
final_evaluate selected run_0002: **TEST MAE 3.779** (independently recomputed
from archived predictions: 3.779; r2 0.242). Baselines: beats RelAgent 4.019
and RDL 4.022; KumoRFM-ft 2.731 stands (gap 1.05). Val->test drift +1.10 to
+1.58 MAE across the three runs (2010+ era shift); the state-space model
degraded least — dynamical/extrapolating designs handle this task's
distribution shift best, a concrete lead for the campaign-config run.

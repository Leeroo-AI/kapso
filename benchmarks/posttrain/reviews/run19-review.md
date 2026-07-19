# Run #19 review — arenahardwriting × SmolLM3-3B-Base (0718134)

Rebuilt stack (580a74a3 + 3ba24c4d). Dual-mandate reviews per
`arena-best-baseline-traces.md`; best-known trace for this cell:
opus-4.8-max 73.95 (local Qwen3-4B-Instruct-2507 teacher, 24.8k
multilingual through exact smollm.jinja). Launched 2026-07-18 13:43Z.

## P1 (t+0 → t+86min)

Headline: plan matches the 73.95 skeleton on 4/6 pillars ahead of the best
trace's clock — teacher (Qwen3-30B-A3B-Instruct-2507, 57GB downloaded in
2.6 min without hesitation), byte-parity smollm.jinja rendering with
empty-think targets, eos 128012 verified, generation_config baked
pre-first-eval, baseline 0.0 @32 confirmed, teacher gen 10,620 prompts in
~24 min @5.1k tok/s → 9,988 kept, SFT running (loss 1.82→1.52) by segment
end. Relaxed rules semantically effective ("Ranking with a locally-served
open model is explicitly legal per the harness rules").

- **R19-P1-1 — P1 major (recipe), 13:56→14:20.** Multilingual slice
  undersized AND wrong language set; no language-distribution scan
  (recon was `head -c 2000` → "multilingual prompts present (es, etc.)").
  Adopted ≈10% multilingual restricted to es/fr/de/it/pt ("SmolLM3's
  languages"); implementation Counter shows 1,120/10,620 (10.5%). The eval
  is ~1/3 non-English INCLUDING zh/ru/ja/vi; the 73.95 trace used
  7k/24.8k with ru/zh/vi/ja targets and attributed its late gains
  "entirely in multilingual". Worst of the three runs on this axis; risk
  masked at limit-48 (73% English). Watch P2 for self-correction via
  judge forensics.
- **R19-P1-2 — P2 (recipe watch), 14:00:44.** On-policy DPO adopted as
  phase 2 (~2.5h/round, second round contemplated), downside framed as
  "bounded to zero" — ignores opportunity cost; the best SmolLM3 trace
  REJECTED DPO (0.619). Mitigated: gated behind a promoted SFT anchor;
  zero DPO time spent in P1. Enforce the >1h flag in P2.
- **R19-P1-3 — P2 (framework/stream), 13:58:38.** Codex member output not
  streamed; candidates auditable only via selector critique. RECURRING
  (3/3 runs — systemic, = R17-P1-2/R18-P1-2).
- **R19-P1-4 — OBS (framework, positive), 14:24/14:29/15:00.** env_defaults
  clock policy active: three foreground sleeps blocked with Monitor
  guidance, agent adapted immediately each time; no auto-background
  surprises.
- **R19-P1-5 — OBS, 13:51:24.** repo-memory core.architecture
  placeholder-only (fresh campaign, expected); R15-P2-1/R16-P2-2 not
  exercised (no feedback pass yet).
- **R19-P1-6 — OBS, 13:51/15:03.** Micro-waste: two empty ToolSearch calls
  at member boot (RECURRING 3/3), one wasted Read, duplicate "MCP config
  written" log line. Cosmetic.
- **R19-P1-7 — OBS (positive).** Boot clean; both members 2/2; selector
  verified evaluate.py claims line-by-line and corrected C3's OOM-risky
  batch; implementation start clean; vLLM generation_config="auto" premise
  re-verified in-session.

Verdict: **concern** — framework health excellent and execution ahead of
schedule, but the 10% wrong-language multilingual slice is the known #2
lever left on the table for this cell; watch whether P2's judge forensics
self-corrects it.

## P2 (15:03Z → 18:30Z)

Headline: ladder 0.3238 (sft_r1 2ep, 15:46) → 0.3942 (+7.0 via on-policy
DPO r1: 2,000×K=4, listwise teacher judge, 1,673 pairs, 53 steps; 16:43)
→ 0.3814 (DPO r2, rejected as plateau) → 0.4250 (re-SFT 3ep lr 1e-5 on the
SAME data, 18:24). All four evals `--limit 50`. Anchor promoted at every
new best with eos [128012,128001] re-verified; DPO r3 pairs banked via
detached sample→judge chain before session end (~19:00Z).

- **R19-P2-1 — P1 major (framework), 15:29→18:20.** ScheduleWakeup timer
  never fires: 8 clear misses in one segment (15:29, 15:42, 16:02, 16:12,
  16:37, 17:18, 17:58, 18:20); every actual wake was
  `task_updated`+`task_notification` from a background-task completion.
  Worst: "Next wakeup scheduled for 17:58:00 (in 1820s)" (17:27:40) →
  planned midpoint loss check ("I'll check the loss trajectory at the
  midpoint (17:58)") silently lost; next event 18:14:20 = 46.6-min gap.
  Run #18's one-off is systemic; the task-notification path is the only
  thing masking it.
- **R19-P2-2 — P1 major (recipe), through 18:30.** R19-P1-1 did NOT
  self-correct: zero per-language verdict tallies and zero
  judge-explanation reads in 3.5h of iteration (watchlist item 6 says "by
  ~h4"). The one forensic pass (17:27:11) chased length ("the judge rubric
  emphasizes 'concise, not verbose'"), printed `multilingual: n=1114` in
  its own output, and concluded "not the main issue" without ever checking
  the language set; re-SFT reused the 10.5% es/fr/de/it/pt corpus intact.
  zh/ru/ja appear nowhere in the delta.
- **R19-P2-3 — P2 (recipe), all evals.** Decisions at limit 50, never
  ≥128 (watchlist item 7): promotion CI is ±6 pts ("final 39.8 (-5.7 /
  +6.8)") while the r1→r2 plateau call rests on a 1.3-pt delta the agent
  itself labels "within noise". Language bias of the small subset also
  keeps R19-P2-2 invisible.
- **R19-P2-4 — OBS (positive).** DPO discipline resolves R19-P1-2: r1
  ~55 min end-to-end (+7, promoted), r2 ~40 min (rejected, immediate pivot
  to re-SFT), r3 only banked ("bank progress: launch DPO sample-gen on
  sft_r2", 18:25). Promotion hygiene strong: instant anchor at 15:46,
  final_model CPU integrity check 18:26.
- **R19-P2-5 — OBS.** Eval wrapper cwd bug (15:35, self-fixed in ~40s);
  one more foreground sleep blocked with Monitor guidance (16:21:36),
  adapted instantly; recurring pre-sleep polling tic (~3-5 redundant Reads
  per wait, absorbed by the "Wasted call — file unchanged" dedupe). No
  session boundary in delta — difficulties tag / feedback judge still
  unexercised.

CURRENT STATE:
best_score.log 32.38 sft_r1 → 39.42 dpo_r1 → 42.50 sft_r2 (3ep); final_model verified (temp 0.7/top_p 0.9/rep 1.05, eos [128012,128001]); all scores limit-50.
DPO r3 sample→judge chain running to bank pairs for next session; run clock 5:26 remaining at 18:24; ~$37 spent.
Vs best trace at same clock: 0.425@50 against 0.639→0.659@128 — decoding/epochs pillars matched, but the 1,114-example wrong-language multilingual slice is untouched and unexamined.

VERDICT: intervene — the P1 watch condition failed (no judge forensics at
all, so self-correction of the zh/ru/ja gap is not coming) and wakeups are
systemically dead; recommend a nudge to read per-language judge verdicts +
regenerate the multilingual slice zh/ru/ja-weighted before the next SFT,
and file the ScheduleWakeup bug.

## P3 (18:30Z → end) + closing

Headline: session 2 turned the banked DPO pairs into the whole gain —
0.425 → dpo_r3 51.96 @50 (19:43, promoted 19:44) → 54.03 @150 (22:06,
stderr 0.0202, shipped) → official full-set **0.5658 ±0.016**. The
boundary chain (end-mode label, fallback difficulties, feedback judge)
exercised end-to-end for the first time and performed; session 2 exited
the CLI naturally with ~59 min of cap unspent. The language question
closes NO: zh/ru/ja never entered the data at any point in the run.

- **R19-P3-1 — OBS (framework, positive): boundary chain end-to-end.**
  End-mode truthful: "Deadline of 18000s reached — terminating Claude
  Code" at 19:00:44 (18000.3s exact); the cut landed mid-housekeeping
  AFTER the agent had already verified GPU 0 MiB, final_model integrity,
  and committed (19:00:34) — best-case boundary state. No JSON result →
  "technical_difficulties missing — running fallback reconstruction"
  (19:02:36): difficulties were FALLBACK-authored, not self-authored, and
  the reconstruction (515s, $2.11, 36 tools) is high quality — 7 concrete
  items, correctly notes the OOM/NaN keyword hits were proactive checks
  not incidents, labels the end "killed at deadline, not a crash", and
  quantifies the banked state (1,689 DPO-r3 pairs, judge done 18:58:44).
  Nit: the strategy logs the cut as "Implementation failed" — factually
  it means "no result JSON", but the wording undersells a substantively
  successful session.
- **R19-P3-2 — P1 major (framework), still open: ScheduleWakeup.** 9th
  and final session-1 miss: 18:27:59 "Next wakeup scheduled for 18:32:00"
  → next event 18:59:55 (task_notification), a 31.7-min dark gap ending
  the session. Session 2 provides no corroboration either way: the fresh
  agent called ScheduleWakeup ZERO times, routing every wait through
  background `sleep N; check` tasks + bounded watchers whose completion
  notifications (52 of them) produced every wake. Watcher hygiene is
  clean — `while ps -p <recorded-PID> ... && [ $i -lt N ]` and
  file-existence loops, no run18-style self-matching pgrep. The bug file
  stands; the notification path is a full workaround, not a fix.
- **R19-P3-3 — OBS (framework, positive): R16-P2-1 regression clean; no
  freezes.** Session 2 finished at 14,463.8s of the 18,000s cap — a
  natural CLI exit after a disciplined endgame: dpo_r4 no-promote at its
  pre-declared gate (23:20:09), final_model md5 re-verified == dpo_r3,
  result.json restored to the 54.03 submission after the informational
  dpo_r4 eval overwrote it (self-caught), memory note + commit, then the
  full XML report (code_changes_summary / evaluation_script_path /
  evaluation_output / score all extracted, 23:23:18). Orchestrator:
  "Stopping: finalization reserve reached — protecting the endgame
  window" at 23:25:21, ~18 min before the wall. Largest quiet stretches
  (20:10→20:39, 22:14→22:36) were deliberate 30-33 min monitor windows,
  not freezes. Run cost $84.83 total.
- **R19-P3-4 — OBS (framework, positive): both feedback judges did real
  forensics.** Node-0 (207s): cross-checked all four eval logs against
  best_score.log, md5-matched final_model to sft_r2, verified evaluate.py
  untracked/untampered, correctly distinguished the 5h CLI cap from the
  10h budget, relayed 0.425. Node-1 (114s): empty-diff eval check, md5,
  limit-50 cross-check ("confirms the gain isn't a limit-150 artifact"),
  and — the best judge moment of the campaign — read the eval source and
  CORRECTED the run's premise: the actual judge is gpt-5-mini
  (reasoning_effort=medium), not gpt-4.1, so the verbosity-bias bet
  behind sft_r3 was aimed at the wrong model; it used this to explain the
  sft_r3 regression and re-ranked next steps (iterative DPO first).
  R15-P2-1 lineage satisfied. Residual gap: neither judge ran
  per-language verdict tallies — node-0's multilingual advice stayed
  generic ("more constraint-rich + multilingual writing prompts"), so the
  language error propagated with feedback's blessing.
- **R19-P3-5 — OBS (recipe): what actually bought the +14.** dpo_r3 =
  robust-DPO (loss_type=robust, label_smoothing 0.05, β 0.1, lr 5e-7,
  1ep) on sft_r2 over the banked 1,689 on-policy pairs: ~10 min of
  training, 51.96 @50 at 19:43:38 (+9.5), insurance-promoted 19:44:02,
  confirmed 54.03 @150 at 22:06:06 — the shipped model. The "dominant
  lever" SFT-scaling leg failed: 5,329 new prompts (judge-bias-aware
  regen, 600-1300 words, markdown-structured) → 17,589-example sft_r3
  reached only 45.75 @50 (21:30), and dpo_r4 on that base 51.42 @50
  (23:19) — "on par, not better", correctly not promoted. Every point of
  session-2 gain came from preference optimization on session-1 assets;
  the shipped model's SFT base remains the 12,485-example session-1
  corpus.
- **R19-P3-6 — P1 major (recipe), CLOSED AS NEVER-CORRECTED: the
  language question.** Iteration-2's selected plan re-specified the slice
  verbatim — "~1.5k multilingual (es/fr/de/it/pt — raised share vs r1)" —
  and the build printed multilingual_es/it/pt/de/fr at 224 each. Final
  accounting: shipped dpo_r3 sits on the r1 corpus with 1,114/12,485
  (8.9%) multilingual, all five Latin-script languages; the 17.6k corpus
  (same 5 languages, ~2.2k multilingual) shipped nothing. zh/ru/ja
  appear in ZERO training examples across the entire run, against an
  eval ~1/3 non-English including zh/ru/ja, with the 73.95 trace's gains
  attributed to exactly that slice. Agent, selector, and both judges all
  passed over it — R19-P1-1 → R19-P2-2 → here, three segments, no
  self-correction.
- **R19-P3-7 — RESOLVED (R19-P2-3): decisions at limit ≥128.** The
  shipped ranking was made at limit 150 (54.03, stderr 0.0202) per the
  plan's consolidation step; limit 50 was retained only for gating with
  explicit noise gates (≥+2 for insurance promotion, ≥58 for dpo_r4).
  Node-1 feedback pushes the remaining inch: full-set `--limit -1` before
  final promotion next time.

Left on the table vs 73.95: the best trace's engine was a 24.8k-example
distillation with a ~7k ru/zh/vi/ja slice through the same byte-exact
template — and no DPO. Run 19 shipped a 12.5k-corpus SFT plus one DPO
round for 56.58, a 17.4-pt gap. The missing mass is (a) the multilingual
slice in the eval's actual languages — the one named lever three review
segments watched go unpulled; (b) corpus scale on the shipped base (the
17.6k attempt failed on the wrong-judge premise, not on scale itself, so
scale was never fairly tested); and (c) ~3.5h of session 2 spent proving
sft_r3/dpo_r4 dead ends, which could have funded a zh/ru/ja-weighted
regeneration + re-SFT + DPO on top — the exact compounding the 73.95
trace demonstrated.

CLOSING VERDICT: **official 0.5658 ±0.016, both judges clean — the top
proven leaderboard row, +7.4 above the human baseline (49.2) and +19.4
above the model baselines (fable-5 37.2, opus-4.8-max mean 37.2)**. As a
framework exhibit this is the cleanest full cycle on the rebuilt stack:
truthful end-mode labeling, a fallback difficulties chain that earned its
keep on first firing, feedback judges that verified rather than believed
(one even falsifying the run's core stylistic premise from the eval
source), a natural CLI exit, and a protected endgame — with ScheduleWakeup
(9 misses, unexercised in session 2) the one open P1 bug. As a recipe
exhibit the verdict is double-edged: banked on-policy DPO is now the
proven cheap lever (+11.5 net for ~35 min of compute), but the run ends
where it began on its known #1 residual — a model that has never seen
zh/ru/ja, leaving the 73.95 ceiling untouched for the next campaign
cell.

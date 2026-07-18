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

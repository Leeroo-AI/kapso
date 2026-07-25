# Run #30 review — gpqamain × Qwen3-1.7B-Base (us-east4-a)

GPQA queue #2, smallest model (base 14.1 | proven band 29.5 gpt-5.2-codex /
29.5 gpt-5.5-xh-rp / 29.4 gpt-5.4-high | human 35.5; weight .2246; random
floor 25.0). Launched 2026-07-25 12:02Z on the upgraded stack, 16 min after
sibling #29 (gpqa × 4B). Cell hazard headline: agents historically score
BELOW the random floor here purely on answer-format/parse failures — a
1.7B holding above 25 is already meaningful. Combined P1+P2 review; trace
covers 12:07 → 17:33Z (≈t+5h31m of the 10h solve, which ends 22:02Z).

## P1 (t+0 → session-1 end 16:42Z)

Headline: **the format cliff was measured, then deleted**. Baseline
limit-150 landed 12:30:29 (t+28min): 0.193 ±0.032 with 44.7% no-`ANSWER:`
and 100% no-`<think>` — pure format collapse, exactly what lens 2
predicted. The shipped model reads 0% no-ANSWER / 0% run-on. All four
pre-registered recon gates (token IDs 151645/151643 + think specials;
vLLM `--generation-config auto` honors the model-dir file; installed
`parse_answers` first-`^ANSWER:`-match semantics; template imported from
the installed package) were settled by 12:48 — two of them in parallel
with exp1 training.

- **R30-P1-1 — OBS (framework, positive).** Lens planner (fable-5
  web-enabled, 131.5s/$0.35) produced two cell-native lenses — L1
  data-centric distillation maximalist / L2 measurement-mechanics-first +
  RLVR (VibeThinker-1.5B as the at-scale precedent) — L2 naming the
  below-random parse-failure history and the insurance-promote discipline
  outright, with explicit lens→member assignment. Members 2/2 + 2/2
  (codex:gpt-5.6-sol 567s; fable-5 440.7s/$0.79), 4 pooled, 0 dropped.
- **R30-P1-2 — OBS (recurring, escalated).** Member tooling asymmetry
  (R25-P1-3, R27-P1-3) produced a real distribution error this time, not
  just sourcing hygiene: codex C1/C2 built statistics and stratification
  on GPQA **Diamond** and "admit they never read the local files". The
  selector's fact-check (225.7s/$0.46) caught GPQA **main** = 448 Qs with
  file:line cites, verified `nvidia/OpenScience` config `OS-Q3-235B-4` on
  the HF API, and crowned fable C3, staging C4's RLVR as the iteration-2
  option. Selector audit: real verification, correct crown.
- **R30-P1-3 — OBS (recipe, the run's central arc).** Run-on collapse
  root-caused as a *learned think-length prior*, not decoding: exp1
  (14,502 full R1 traces, median 1056 tok) → eval killed at 10 samples,
  median 50,676 chars, 6/10 at the 16k cap, 1/10 parseable, below
  baseline, NOT promoted; exp2 (12,800 shorter traces, median 916) → 85%
  run-on, n=20 gate 0.150; exp4 (full-think capped 1000) → 57% run-on,
  0.100 — both greedy and temp-0.6 ran on. Fix: summary-only `<think>`
  targets (median 259 tok) → exp3 1% run-on, exp5 0%. The feedback judge's
  own counterfactual: exp2 was a ~50-GPU-min near-repeat of exp1 —
  the one clearly wasted cycle.
- **R30-P1-4 — OBS (positive).** Negative-space Coverage: all five
  families on the fable candidates with substantive Not-measured closers —
  and the flagship ASSUMED item ("vLLM does not strip reasoning before
  scoring") was tested and **overturned** in-run: after the
  0.367-official vs "97% no-answer"-profiler contradiction (15:08), the
  agent found the provider separates `reasoning` from `completion` and
  only post-`</think>` content is scored (difficulty 5). Declared
  unknown → measured → load-bearing mechanic settled. MC-specific traps
  also caught: gold-letter skew (natural A 15.8/B 35/C 35/D 14.3)
  stratified to 25/25/25/25 rather than reshuffled (traces cite letters
  inline); OpenScience schema mismatch ({input,output}, gold in
  `\boxed{}`) recovered by parser rewrite.
- **R30-P1-5 — P4 (recipe).** Two noise-adjacent promotion moments, both
  contained: (a) insurance promote at 15:09:32 rode the n=30 gate (0.367,
  stderr 0.089) before the limit-150 confirm — correct insurance
  ordering, but the flattering `36.67 …-limit30` line lingers atop
  best_score.log; (b) exp5 displaced exp3 on +3.3pp limit-150 (0.300 vs
  0.267, inside 1 SE ≈ 3.8pp), justified structurally (0%/0% format
  stats, 18k data) and immediately de-risked by the full-448 confirm.
  The judge then codified the fix (see P2).
- **R30-P1-6 — P4 (framework).** Harness hint names a tool this agent
  doesn't have: blocked foreground sleep-300 (13:20:21) says "use
  Monitor"; the Monitor call fails InputValidationError, and ToolSearch
  `select:Monitor` returns EMPTY — Monitor is not in the posttrain
  implementer's toolset. ~90s churn before the correct pivot to a
  notifying `until`-loop Bash task. Idle hygiene otherwise the campaign's
  best: ScheduleWakeup 0, 8 wasted-call re-reads (21-28 in runs 25-27),
  zero armed-then-stopped waiters, every wait bounded + notifying,
  dead-man alarms re-armed on fire (R23-P3-1 clean).
- **R30-P1-7 — OBS.** In-process vLLM diag script kept ~66GB after
  finishing → next eval's engine-core init failed (13:46); diagnosed via
  live PID on nvidia-smi, fixed with targeted kill + `os._exit(0)`, and
  all real evals routed through the official evaluate.py thereafter.
  Only material infra incident of the session.

SELECTED PLAN (C3): contract-first distillation SFT — stream
GPQA-decontaminated `nvidia/OpenScience` OS-Q3-235B-4 (20k filtered rows
cached to shared_cache, fingerprint 0946e884), render targets byte-exactly
through `SINGLE_ANSWER_TEMPLATE_COT` + the eval's own qwen3.jinja with
completion-only loss, terminal `ANSWER: L`; full-FT bf16 TRL; ship
generation_config with eos [151645,151643]; limit-150 decisions with
SE≈3.7pp / <4pp-tie arithmetic; insurance promote; greedy-vs-0.6 A/B;
zero GPQA tokens ever loaded (independent re-decontamination correctly
declined — it would require mining GPQA text; documented as the
Permitted-data Not-measured line). Session 1 ran 15,559s/$28.56/182
tools, ended 23 min under its 5h cap with a pre-committed 17:03 training
freeze honored (exp5 done 16:29).

Verdict: **continue** — textbook P1. Format hazard defused by
construction and verified at every rung; graded gates (n=10 kill / n=20 /
n=30 → limit-150 → full-448) spent GPU where information was.

## P2 (16:43 boundary → 17:33Z trace end)

- **R30-P2-1 — OBS (positive, headline).** Boundary chain full fidelity.
  All 5 tags extracted incl. a 7-item quantified technical_difficulties
  (run-on signature numbers, the reasoning-field mechanic, the vLLM hang).
  Feedback judge was a real session (404.8s/$1.36/18 tools): tamper audit
  (official evaluate.py pristine by mtime + key lines; the repo wrapper
  legitimately subprocess-calls it), orphaned-value sweep (0.367 = limit-30
  noise, correctly dismissed), SE-aware critique (exp3→exp5 within 1 SE —
  "scaling summary-only bought almost nothing"), and a hardened promotion
  rule: **gate every future promotion on full-448; overwrite final_model
  only above 0.281**. Score pinned to the robust full-448 0.281, not the
  flattering 0.300. stop=false; repo memory written
  (gpqa-qwen3-1p7b-iter1-mechanism.md + MEMORY.md). Cumulative $31.27.
- **R30-P2-2 — P4 (framework, recurring).** R15-P2-1, 5th run family:
  the judge's first Read was kapso_campaign/kapso_evaluation/evaluate.py
  → "File does not exist" (16:43:55), recovered in one step. The deferred
  fix (resolve evaluation_script_path against the session dir) is overdue.
- **R30-P2-3 — P3 (framework, new failure mode).** Iteration-2 selector
  died on its 540.0s deadline: it spawned a 3-agent read-only recon
  fan-out at 17:10:57 — five minutes into a nine-minute window — and was
  killed at 17:14:57 ($0.00, 3 tools) before any agent returned;
  the strategy "fell back to the pooled candidates" (14,483 chars).
  Degradation was graceful: the implementer executed the candidate
  aligned with the feedback's #1 lever and re-ran equivalent recon itself
  (Workflow, 17:17:53, after one invalid-param retry). But the crowning/
  grafting step was lost and ~9 min burned. Root tension: the stack now
  flags workflow/fan-out as the preferred verification pattern, yet the
  selector budget cannot absorb a fan-out round-trip — iteration 1's
  direct-read selector (225.7s) fit easily. Either raise the selector
  deadline or teach it fan-out budget arithmetic.
- **R30-P2-4 — OBS (recipe).** Iteration-2 conduct to the cut is clean
  and box-aware: exp6-grpo = parser-exact GRPO warm-started from a COPY
  of exp5 (never final_model), reward module self-tested (6 cases, two
  mis-set golds fixed before use), render path byte-verified CPU-only
  before touching the GPU (p95 263 tok), pass@8 difficulty probe over
  5,000 prompts launched detached (n=8, temp 1.0, stop [151645,151643]),
  probe generations doubling as a free RAFT fallback dataset, GRPO
  trainer written while the probe runs, `use_tqdm=False` (log-bloat
  awareness), commit-before-risk, fresh-scan step consciously skipped and
  the scope decision recorded. The member-side recon also invalidated
  PLAN.md's "disjoint ~20k RL pool" handoff claim (exp5 consumed 18k of
  20k; only ~2k unseen) — caught before it could shape the design.
  Cosmetic: difficulties item 7 quotes the A/B as "0.300 vs 0.213";
  the actual pair was exp3 greedy 0.267 vs temp-0.6 0.213.
- **R30-P2-5 — OBS (cross-run, sibling #29 gpqa×4B).** Convergent
  structure, fully re-derived (cross-run memory not yet built): both runs
  hit the same below-random format cliff (base 0.193 here; #29 limit-50
  0.14), both fixed format first, banked insurance, ran SE arithmetic,
  and used full-448 to break ties. Divergent data and opposite failure
  modes: #29 (Nemotron 12K, shallow p50 ~200-tok traces) plateaued at
  26.67 limit-150 with "traces too shallow"; #30's OpenScience R1 traces
  were too LONG (run-on) until cut to summaries. #30's 1.7B full-448
  0.281 already matches the 4B sibling's interim limit-150 band —
  OpenScience looks like the stronger corpus pick.

Hygiene to the cut: ScheduleWakeup 0 across both sessions; 76
task_notifications (+ Workflow task_progress) as the only wake sources;
15 rate_limit_events, no visible stalls; 6 result:error in 5.5h, all
benign and recovered in ≤1 step; weights/eval logs live outside the
campaign repo (R27-P3-4 journal-blowup exposure low). Trace ends 17:33:27
with the probe ~7 min in (GPU 70GB/54%), ALARM-TICK-2 armed, 4h29m of
solve remaining.

LADDER (limit-150 unless noted): 0.193 base (12:30) → exp1 killed-eval,
below base (13:52, rejected) → exp2 0.150 n=20 gate (14:36, rejected) →
exp3 0.367 n=30 gate (15:08) → insurance PROMOTE 15:09:32 → **0.267
CONFIRMED 15:16:07** (the peeked mid-run reading) → temp-0.6 0.213 A/B
loses (15:55) → exp4 0.100 (15:51, rejected) → **exp5 0.300 PROMOTED
16:32:32 → full-448 0.28125 ±0.021 (16:37:54, FINAL so far)** →
exp6-grpo pending at cut.

SOTA OUTLOOK: full-448 0.281 sits +3.1pp above the random floor, double
the 14.1 base, and 1.2-1.4pp below the proven 29.4-29.5 band — with
σ(448)≈2.1pp, promotion-by-luck is a live risk and the judge's
full-448-only gate is exactly the right guard. Beating proven #1
decisively needs ≥0.295 official, i.e. GRPO/RAFT must add ~+6 questions
over ~4h; the floor (0.281, greedy, eos-fixed, verified loads) is banked
and protected either way. Human 35.5 is out of reach this run.

VERDICT: **continue** — a 1.7B holding 28.1 on full GPQA-main with zero
format losses is already a top-band result for the cell; the remaining
budget is pointed at the only lever (reasoning/RL) that can clear 29.5.
Framework tickets: selector deadline vs fan-out (R30-P2-3), R15-P2-1
(5th recurrence), Monitor-hint paper-cut (R30-P1-6).

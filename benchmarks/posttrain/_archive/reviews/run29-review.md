# Run #29 review — gpqamain × Qwen3-4B-Base (us-east4-a, launched 2026-07-25 11:46Z)

FIRST GPQA cell in the campaign. gpqamain = 448 four-choice graduate-science
questions, **25% random floor**, plain-accuracy `choice()` scorer, inspect
`multiple_choice(cot=True)` on vLLM (16k max tokens), no OpenAI key needed.
Cell reference (RESULTS.md GPQA-Main): Qwen3-4B base 13.4 | proven ~34
(gpt-5.4-h-rp 34.1) | human 44.6; weight .2246 (heaviest with real headroom).
THE classic GPQA killer per PRIOR_RUN_INSIGHTS — scoring **below the 25%
floor purely on answer-format** — is the watchlist headline. Same upgraded
stack as #26 (negative-space coverage, lens planner, max reasoning). Combined
P1+P2 review; run in flight at cut (trace to 17:32Z ≈ t+5h46m, 4h17m left).

## P1 (t+0 → run2 promotion 13:45Z ≈ t+2h)

Headline: **the classic GPQA format killer was met head-on and cured by
t+2h — baseline 14% (< the 25% floor, pure format failure) → run2 26.67%
after a broken first run was caught by a pre-eval logit probe, not a wasted
150-eval.** Boot clean (11:50Z: H100 idle, cuda-tensor write pass; zero
us-east4 anomalies). Lens plan (fable, web, 137.8s, $0.31) read `evaluate.py`
+ `qwen3.jinja` directly: L1 dominant long-CoT science-distillation SFT
(OpenThoughts3/MegaScience/Nemotron-Science) / L2 measurement-mechanics-first
eval-exact-format SFT + fast empirical loop — orthogonal (capability vs
realized-score), both carrying the non-negotiable `ANSWER: $LETTER` +
template-fidelity spine.

- **R29-P1-1 — OBS (positive; the headline).** Format hazard handled
  textbook-correct and early: exact inspect `SINGLE_ANSWER_TEMPLATE_COT`
  string extracted from the installed package, `qwen3.jinja` assistant-turn
  shape read (`<think>…</think>\n\nANSWER: X<|im_end|>`), answer contract
  mirrored byte-exact in training. Baseline limit-50 = **0.14** (12:21Z,
  25s) accepted only after **viewing completions** (JS gibberish, off-topic
  rambling, and `**ANSWER: B**` bold that fails the strict line-start
  regex) — the sub-random score correctly read as format failure, not lack
  of knowledge. run2 solved it: **96% parseable, no truncation, 14%→~28%**.
- **R29-P1-2 — OBS (positive).** run1 (lr 5e-6, 1 epoch, per the solution's
  hyperparameters) trained to loss ~1.0 but was **broken** — and was caught
  by a direct generation test + **position-0 next-token logit read**
  (P(`<think>`)=0.092 ≈ P(`</think>`)=0.104, flat; greedy collapses to
  `</think>` repetition → 16k junk/sample), NOT by trusting a slow eval. Data
  + masking verified correct; root-caused to **undertraining**; refit lr 1e-5
  / 3 epochs → P(`<think>`)=**0.954**, clean 56-token generations. The
  pre-eval logit probe is the strongest single discipline in this run.
- **R29-P1-3 — CONCERN (framework; recurs in P2).** The `claude_code:
  claude-fable-5` ideation member **AUP-refused** both candidates ("API
  Error: Claude Code is unable to respond … appears to violate our Usage
  Policy", 11:56Z) → 0/2; only `codex:gpt-5.6-sol` delivered (2/2, 466s).
  Ensemble diversity halved on a benign science-distillation prompt; the
  selector chose from a single member's pool. Same refusal fires again in
  iteration 2 (16:20Z) — a reproducible false-positive worth surfacing.
- **R29-P1-4 — OBS (echo of AIME selector-compensates).** Both candidates
  were written **blind** (evaluator access failed during ideation), so the
  selector did its own recon: read `evaluate.py`, caught that **both
  assumed GPQA-Diamond-198** and calibrated domain mixes to Diamond's
  10/47/43 when the eval is **gpqa_main (448, ~17/41/41 bio/chem/phys)**,
  and refuted candidate-2's 1,536-token teacher cap against the measured 16k
  budget. Recon-in-the-selector carried the round.
- **R29-P1-5 — OBS (coverage, clean).** All five negative-space families
  present with MEASURED/ASSUMED/Not-measured closers well-suited to a
  multiple-choice science cell: metric mechanics with real SE arithmetic
  ("binomial SE ≈ 4.1 pts @ n=150, ≈ 2.4 @ n=448 — hence paired disagreement
  + ≥4-pt promotion skepticism"), permitted-data geometry pinning GPQA to
  **reject-list-only**. Gated-dataset gotcha (`Idavidrein/gpqa` 403 even with
  the env HF token) resolved by programmatically accepting the auto-gate
  (POST ask-access → 303), used strictly as a decon reject-list.

LENS PLAN: L1 curated long-CoT science-distillation SFT / L2 eval-exact-format
SFT + fast empirical loop.

SELECTED PLAN (synthesis): staged full-param BF16 SFT of Qwen3-4B-Base on
`nvidia/Nemotron-Science-v1` (backbone) + OpenThoughts3 science (diversity),
every example rendered through the exact eval template, `ANSWER: $LETTER`
target, lr 5e-6→(fixed)1e-5, maxlen 4096 packed, assistant-only loss,
eos→`<|im_end|>`; bank a first score by ~T+3h, scale to 35–45K, iterate one
axis at a time; optional teacher distillation only if headroom shows.

Verdict at t+2h: **healthy** — the #1 GPQA hazard (format) was diagnosed and
beaten fast, a working model banked (26.67%), broken run1 caught pre-eval not
post. Open question that P2 answers: with format solved, does capability move,
or does the 4B sit at the ~random floor?

## P2 (13:45Z → cut 17:32Z ≈ t+5h46m)

Headline: **format was the only lever that moved — three iteration-1 training
runs all land ~random (best banked 29.24%, 131/448), and a teacher-distill
bonus bought +0.67 pt of noise while adding a truncation pathology. A sharp
DO-NOT-STOP judge re-pointed iteration 2 at the right corpus (nvidia/
OpenScience 4-choice R1 traces) — but at the cut the fresh probe's own eval
is stuck 33 min in runaway generation, GPU-blocked, main 30K run unlaunched,
4h17m left.**

- **R29-P2-1 — OBS.** Ladder: run2 lr1e-5/3ep **26.67@150 PROMOTED**
  (13:45Z) → distill (DeepSeek-R1-Distill-Qwen-14B, inference-only/legal,
  3061 deep traces mixed with Nemotron) **25.33@150** not promoted →
  full-448 tie-break **run2 28.57 / distill 29.24** → **distill PROMOTED
  29.24** (16:03Z, final_model). Capability verdict is consistent across all
  three: with format solved the 4B student sits ~random on gpqa_main; the
  format fix (14→~28) is the entire genuine gain. Bottleneck migrated
  format → data-fit/reasoning-quality.
- **R29-P2-2 — CONCERN.** The distill promotion rides a **0.67-pt full-448
  delta** (131 vs 128 = 3 questions, < the 2.4-pt SE) — a nominal-best
  tie-break the judge itself priced as noise. The detour also taught R1's
  verbose style: mean out-tokens 522→**3038**, **44/448 (9.8%) hit the 16k
  cap → auto-zero** (vs run2's 5). ~100 GPU-min for ~0 real points, and it
  was **undeclared drift** (logged "BONUS EXPERIMENT," consuming the plan's
  Stage-2 scaling budget). Credit: the two full-448 evals that exposed the
  limit-150 mis-ranking were defensible insurance.
- **R29-P2-3 — OBS (positive, framework; no R15-P2-1).** Node-0 feedback
  judge (opus-4-8, 594s, 19 tools) ran its **own** forensics — verified the
  eval wrapper is an untampered subprocess, decon genuinely reject-only
  (`prepare_sft.py` `load_gpqa_reject`/`is_contaminated`), independently
  counted the 44/448 truncations, confirmed no hidden better artifact. No
  parroting of agent claims. Verdict **stop=False** with the correct root
  cause (data source: Nemotron 100% 10-option/27%-have-reasoning, random
  distractors easier than GPQA's, ~48% of train kept native A–J vs 4-option
  eval, and OpenThoughts3/MegaScience **never used**) and a prioritized
  lever list (stronger GPQA-matched corpus, fix runaway generation,
  fresh-from-base + full-448 promotion). Invariants carried verbatim.
- **R29-P2-4 — OBS (positive, recipe).** Iteration 2 acted on the feedback
  precisely: discovered `nvidia/OpenScience` (CC-BY-4.0, **already
  GPQA-decontaminated**, 4-choice R1 deep traces, already cached) — a
  near-perfect realization of the plan's intent — and built a **30K
  uniform-letter** corpus (7500/letter, phys-leaning to match GPQA, traces
  ≤4096 tok). Caught a **reshuffle-poison bug** at render-verification (it
  permuted option positions while the R1 trace references the *original*
  letters → reasoning contradicts the answer) and fixed it (keep original
  order + boxed letter; 99.5% conclusion consistency). A genuinely better
  data bet aimed at the diagnosed bottleneck.
- **R29-P2-5 — CONCERN (live at cut).** The iteration-2 probe's full-448
  eval has run **33 min, stuck at 150/448** (51MB log, GPU 89%) — **runaway
  generation on the fresh probe**, i.e. the very truncation pathology the
  feedback prioritized, **not yet cured** despite ≤4096-tok training traces.
  It is blocking the GPU the **30K main run** (~76–85 min/epoch) needs, and
  only **4h17m** remain. The agent is mid-decision (wait vs kill) at the cut
  — the endgame is genuinely squeezed.
- **R29-P2-6 — OBS (framework, clean).** ScheduleWakeup **0**;
  InputValidationError **0**; 11 rate-limit events, no stalls; dead-man
  alarms + auto-backgrounding bounded waiters used throughout (no dud fires,
  all quiet gaps intended). Session-1 implementation ran 14,452s ($24.59)
  and emitted a **technical_difficulties** artifact — 7 root-caused items
  (gated dataset, wrong data assumption, run1 format-collapse, eos/16k
  non-termination, distill OOM-averted, limit-150 mis-ranking, log-field
  confusion) each with the fix, plus an honest "all models near the 25%
  floor" caveat. Boundary chain works.

LADDER: base **0.14@50** (12:21Z) → run1 lr5e-6/1ep **broken** (undertrained,
killed) → run2 lr1e-5/3ep **0.2667@150 PROMOTED** (13:45Z) → run2 **0.2857@448**
/ distill **0.2533@150** → **distill 0.2924@448 PROMOTED** (16:03Z,
final_model, 131/448) → iter-2 OpenScience probe full-448 eval **in flight**
(stuck, no score) at cut. Peeked 0.2667→0.2857 = run2 at the two eval limits;
the actual banked best is 0.2924.

SOTA OUTLOOK: current banked **29.24** (t+4h17m) sits between base 13.4 and
cell-proven ~34.1, above the 25% floor but below human 44.6 and just under
the cross-model best-agent average (30.5). The right levers are now in hand —
OpenScience's deep, hard, decontaminated, 4-option R1 corpus is a much better
data bet than Nemotron, and the length curriculum targets the truncation the
judge flagged. 32–34 (the proven band) is plausible **iff** the 30K run lands
and runaway generation is actually controlled; the live risk is that the
stuck probe eval means truncation is *not* solved, the main run may not finish
inside the remaining ~4h, and a 4B may simply cap ~30 on gpqa_main.

Verdict: **continue / in flight** — the campaign's first GPQA cell beat the
signature format killer cleanly and the iteration-2 recipe is pointed exactly
where the evidence says (better corpus + truncation control). But nothing
above ~random has landed, the endgame is time-squeezed, and the fresh probe's
runaway eval at the cut is the thing to watch. P3 watches: does the probe
eval terminate (or get killed) and does the 30K OpenScience run fit + break
30%; whether truncation is genuinely cured this time; the ~17:05Z-band
session-2 boundary mechanics; and (framework) the reproducible fable-5
ideation-member AUP refusal that halved ensemble diversity both rounds.

# Run #25 review — aime2025 × Qwen3-1.7B-Base (07242238, us-east4-a)

Three firsts: first AIME cell (base 0.0 | best proven 6.7 opus-4.7 | human
26.7; weight .2265), first NON-JUDGE cell on the negative-space coverage
build (the genericity test), first run outside us-central1 (region
failover after a 2¼h starved flex request). Launched 2026-07-24 22:38Z.

## P1 (t+0 → ~t+95min)

Headline: **strongest first-95-minutes of any cell so far**, and the
negative-space contract proved benchmark-agnostic — all five coverage
families came out math-shaped (answer-extraction / length / legality),
with the judge axis explicitly dismissed ("Not measured: judge/rubric
wording — none exists (exact-match scorer, no judge)") and language never
over-built (zero multilingual anything — the correct zero-cost implicit
dismissal for English-only AIME). All 15 Not-measured closers substantive.

The classic AIME killer was **defused at ideation time, before any
training**: the fable member read the installed scorer source
(inspect_ai's `match(numeric=True, location="end")` after `\boxed` strip)
and found the decisive artifact lever — the base checkpoint ships
`eos_token_id=151643` only, so without shipping eos=[151645,151643] in
generation_config.json "generations never stop and the last-number scorer
reads garbage". Live-verified at 23:04 (config dump + im_end id). maj@k
correctly ruled out: eval MEASURED as single-sample pass@1, max_tokens
16000 (not 16384 — read from evaluate.py:34); the legal sampling lever is
the shipped generation_config under vLLM --generation-config auto
(verified with a served request).

- **R25-P1-1 — P2 (agent), 23:03:40.** Self-inflicted baseline crash:
  exported HF_HUB_ENABLE_HF_TRANSFER=1 without hf_transfer installed;
  diagnosed + fixed in ~1 min, ~4 min lost total.
- **R25-P1-2 — OBS, 23:06:23.** Baseline accuracy 0.000 accepted from the
  metrics file without a sanity glance at completion samples (an 80s
  30-problem eval). Score matches leaderboard base 0.0 — stakes nil.
- **R25-P1-3 — OBS, 22:57:42.** Codex member's candidates admitted local
  probing failed (harness-controls stayed ASSUMED; both missed the eos
  lever). Selector compensated with its own file reads, upgrading claims
  to MEASURED. Member tooling asymmetry — recurring theme.
- **R25-P1-4 — OBS, 23:08/23:58.** Minor idle churn (one stale alarm, two
  wasted reads); dead-man discipline otherwise exemplary (layered
  completion-waiter + backup alarm, re-armed each poll). ScheduleWakeup 0.
  $11.95 at t+80min.
- **R25-P1-5 — OBS (positive).** See headline: scorer read + eos lever +
  maj@k legality analysis all pre-training.
- **R25-P1-6 — OBS (positive).** us-east4-a: ZERO region anomalies (fast
  downloads post-hf_transfer, no HF throttling). SFT1 healthy at trace
  end: 422/1313 steps, GPU 98%, loss 0.73→0.46, ETA 01:07Z — measured
  arithmetic fits the session cap with gate+eval to spare.

LENS PLAN: L1 off-policy data excellence (verified R1-trace corpora →
difficulty-matched 16k-fitting subset, byte-identical qwen3.jinja +
ANSWER wrapper) / L2 measurement-mechanics + on-policy refinement (length
economics under the 16k cap, guaranteed final answer line,
generation_config ownership, held-out gating, then RFT). Genuinely
orthogonal, both math-native.

SELECTED PLAN: SFT1 on ~18–22k correctness-verified OpenR1-Math-220k
traces (≤10.5k tok, one per problem, full FT bf16, lr 2e-5 — selector
argued C3's 1e-5 up with small-model/step-count evidence, eff. batch 16,
1 epoch, completion-only loss) → adaptive stage 2: RFT self-distillation
(k=8, pre-2025 DeepScaleR/AIME pools) if gate solve-rate ≥20%, else
Light-R1 stage2-3k continuation; GRPO relegated to dropped candidates.
Every promotion gated on a 90-problem AIME 2022–24 held-out at n=4
(defeats the σ≈7pt 30-item test noise). Decoding: greedy vs t0.6 A/B on
the gate, winner shipped with eos=[151645,151643]; completions trained to
self-open <think>.

Verdict: **continue** — baseline banked t+28min, the format killer
defused pre-training, measured-not-assumed plan, healthy SFT. Blemishes:
the 4-min hf_transfer crash and the unsanity-checked (but expected) 0.0
baseline. P2 watches: SFT1 gate result (~01:30Z), stage-2 branch taken,
n=4 gate discipline held, decoding A/B outcome.

## P2 (00:15Z → 03:26Z; session 1 ended 03:23:36Z, $57.24, 349 tools)

Headline: **3.33 official banked (first nonzero AIME)** after an honest,
self-diagnosed recovery chain — but the gate discipline the plan was built
on collapsed under truncation economics, and the branch rule fired on
noise.

- **R25-P2-1 — P3 (recipe), 01:04→01:31.** The 88-problem n=4 gate was
  killed as infeasible: SFT1 LOOPS — "mean_acc 0.0, trunc_rate 0.9583,
  len_p50 16000" (each truncated gen costs 16k tokens; full gate = hours).
  All later gates were 20–36-problem n=1–2 — exactly the noise the design
  existed to defeat, and it bit: "acc=4/20" read as 20% held-out → RFT
  branch fired → official 0.000; agent's own post-mortem: "the gate's 20%
  was a noisy small sample… real rate ~5-6% (2/36)".
- **R25-P2-2 — P3 (recipe), 01:33/02:33/03:13.** Official 30-test run 3×
  mid-session (0.0 / 3.33 / 0.0); worst: SFT3 evaluated directly on the
  test with NO held-out gate ("to use time efficiently") — model selection
  on the test set. Mitigated: 3 candidates total, atomic promotion, SFT2
  stayed banked; SFT1's eval was score-banking insurance and usefully
  falsified the noisy gate.
- **R25-P2-3 — OBS (framework).** Wake-loop churn quantified: 83 "Wasted
  call — file unchanged" reprimands in the delta; 36 thinking turns in one
  02:29–02:31 wait window (each tracked poll-task completion re-wakes,
  seeding the next poll). $11.95@t+80min → $57.24 at end. Token burn only;
  loops always eventually yielded to real waits. Third sighting of the
  churn class (=R17-P2-6 family) — fix batch candidate.
- **R25-P2-4 — OBS (framework, positive).** NO R23-P3-1 dud-alarm idiom:
  alarms were real tracked tasks that demonstrably re-invoked (5 fires
  logged); max silence 25 min, all ended by real waiters/alarms.
  ScheduleWakeup 0; zero freezes; us-east4 throughput parity (SFT1
  finished 3 min ahead of ETA); result.json overwrite caught and restored;
  PLAN/eval_profile/memory updated 5×.
- **R25-P2-5 — OBS (recipe, positive).** Recovery chain: rep_penalty 1.05
  discovered and shipped (t0.6/p0.95/k20/eos[151645,151643]); RFT killed
  on measured throughput (692 tok/s → 64-min ETA); replaced by short-trace
  OpenR1 continuation (7k traces, p50=1991) → SFT2 official 3.33; SFT3
  (2× short) regressed to 0.0 "over-terse", correctly not promoted.

LADDER: SFT1 0.0 (96% truncation looping) → SFT2 short-trace **3.33
banked** → SFT3 0.0 rejected → session end with ~4.9h left for iteration 2.

ENDGAME OUTLOOK: truncation (64%) is the dominant recoverable loss and the
rep-penalty/short-trace lessons are in cross-session memory; iteration 2
has a real shot at 6.67 (= tie best proven), upside 10.0. Human 26.7 out
of reach. Calibrated band: 3.3–6.7.

Verdict: **continue** — score banked, hygiene held, deviations reasoned
and self-diagnosed; watch iteration 2 for restored gate discipline (larger
n before official-test peeks) and the churn cost.

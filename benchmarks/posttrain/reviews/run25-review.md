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

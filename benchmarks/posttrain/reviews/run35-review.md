# Run #35 review — gpqamain × gemma-3-4b-pt (gpqamain-gemma-3-4b-pt-07260822)

Launched 2026-07-26 08:22Z us-east4-a, completing the GPQA queue. Cell
refs: base 1.6 (weakest), proven band 28.7-29.5, human 31.5. Hazard watch
in: gemma multimodal arch/processor configs, 262k-vocab OOM, eos [1,106]
(runs 24/28 difficulties).

## P1 (t+0 → t+2.7h)

Headline: **the freshly-shipped subset-bias lesson landed verbatim in the
plan — "treat --limit 150 as a biased estimator that reads ~1-6 [pts hot]"
appears twice, with a full-448 freeze-time eval scheduled — less than 12h
after the lesson was written into PRIOR_RUN_INSIGHTS.**

- Ensemble 4/4 pooled; selected plan: staged LoRA-then-full-FT on
  4-choice science MCQ (nvidia/OpenScience primary), completion-only
  loss, Gemma3ForConditionalGeneration with FA2.
- **R35-P1-1 — OBS (gemma traps: 1 pre-empted, 1 re-hit + fast fix).**
  At 08:47 recon flagged "no generation_config.json on the pt checkpoint
  — critical, I'll ship one" (pre-empted). At 09:23 the vLLM
  image-processor trap fired anyway (saved checkpoint = multimodal but
  only tokenizer files copied); root-caused in-minutes with the exact
  run-28 fix (copy processor/preprocessor configs). Cross-run difficulty
  transfer visible but still reactive — the knowledge-repo design would
  make this a pre-flight checklist item.
- Baseline honesty: base gemma read ~2% accuracy on the early probe
  (consistent with the 1.6 reference row); parse telemetry (parse 96%
  at one probe) tracked per-eval from the start.
- Watcher discipline: layered completion-waiter + 35-min dead-man alarm
  on Stage-B; only 3 wasted calls in 2.7h — cleanest read-discipline
  start of the campaign.
- State at cut: Stage-B full FT at 54% (~30 min left), decoding-config
  write + @150 eval queued on completion.

## P2 (t+2.7h → t+6.3h)

Headline: **iter-1 banked 0.2634 (26.34% full-448) — ABOVE the 25% random
floor and within ~2-3 pts of the proven band (28.7-29.5), on the
weakest-base cell of the whole GPQA table (base 1.6).** gemma is
OUTSCORING the SmolLM3 run head-to-head (26.3 vs 16.7) — the inverse of
the arena/AIME ordering: the science-MCQ register plus gemma's stronger
base knowledge clears the parse wall that pins the smaller model.

- **R35-P2-1 — OBS (subset-bias discipline, maximal).** 253 in-trace
  "full-448" references — the most thorough adoption of the @150-bias
  lesson yet; every promotion decision routed through the full set.
  Endgame plan is explicit: "sanitize generation_config → verify GPU
  free → @150 gate (parse/length/score) → full-448 → promote if >".
- **R35-P2-2 — OBS (iter-2 conduct).** Full-FT continuing; training
  healthy (~4.5-5 s/step, loss ~0.85-0.88, cosine LR), checkpoint-150
  saved as kill-early safety net, save-fix verified via a dedicated
  checkpoint watcher. Completion ETA ~15:15Z.
- **R35-P2-3 — OBS (watcher discipline, still cleanest).** Layered
  completion + checkpoint + bounded dead-man watchers; genuine
  turn-ending waits throughout; no dud fires.
- No session-limit events; failover idle.

## Framework note (both runs, P2)

FREEZE WARNING false-alarm: the finish watcher flagged both traces
"static ~50-60 min" at ~14:25Z while both were in fact mid-training and
healthy (pulled traces carried 14:26-14:27Z thinking lines; GCS caught
up to ~1 MB by 14:30). Cause: GPQA training-wait windows go legitimately
quiet (agent awaiting a completion notification, minimal stdout) and the
5-min rsync uploads in bursts, so the 50-min freeze threshold trips on a
healthy quiet stretch. Watcher-tuning item (raise the GPQA freeze
threshold or gate on VM-status + GPU-util instead of trace bytes); no run
impact. Also worth banking for the eventual `knowledge/gpqamain.md`:
both runs' full-448 discipline confirms the @150-subset-bias lesson
(written from #33) transferred cleanly to the next model generation.

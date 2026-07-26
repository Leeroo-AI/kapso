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

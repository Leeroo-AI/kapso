# Run #24 review — arenahardwriting × gemma-3-4b-pt (07241547)

First-ever kapso run on this cell (base 0.3 | best proven 47.4 opus-4.8-max
| human 94.8 — the largest human-agent gap of any arena cell). Negative-space
coverage build (816500d1) + hardened host-python boot (2703dc24). Launched
2026-07-24 15:47Z. Dual-mandate reviews per `arena-best-baseline-traces.md`.

## P1 (t+0 → ~t+86min)

Headline: among the cleanest first segments of the campaign — the two
gemma-specific traps were both handled (multimodal arch: vision_tower +
multi_modal_projector frozen after the implementor read
`Gemma3ForConditionalGeneration` itself; eos: gemma-correct [1,106]
everywhere, zero qwen-eos contamination, and the agent even caught the
trainer writing a default eos=1 config — "I'll overwrite it with eos=[1,106]
at promotion"). Language axis measured at recon (R20-P1-1 chain holding):
"EVAL LANGUAGE MIX (script-based, 250 q): latin 183 (73%), zh 35 (14%),
ru 26 (10%)". Template byte/token-parity asserted vs templates/gemma3.jinja.
Base floor 0.0172 @20. SFT healthy at segment end (loss 1.93→1.65, ETA
~18:02).

- **R24-P1-1 — P1 (recipe), 16:05.** Selected plan INVERTS the proven arena
  skeleton: stage 1 = static corpora as-is (~25k no_robots/WildChat/COIG-
  CQIA; fresh-teacher distillation demoted to optional step 13b), stage 2 =
  on-policy DPO with Skywork-RM ranking budgeted ≈2.5h. Protocol rule 6
  (preference stages never earned their place in best traces) + rule 1
  (never static corpora alone). Mitigation: final_model banks by ~3h,
  promotion needs a replica-gate delta + real-eval confirm. The segment's
  main risk — P2 must time-box the DPO bet and watch for the SmolLM3-vs-4B
  split (weak-base DPO paid on SmolLM3; strong-base DPO lost on 4B; gemma's
  0.017 floor is the weak-base case, so the bet is defensible if gated).
- **R24-P1-2 — P2 (framework/member), 16:01→16:05.** Codex member labeled
  web-derived facts MEASURED (public Arena-Hard repo judge ensemble vs the
  actual local gpt-5-mini harness). The selector caught it by reading
  evaluate.py:42 — the pooling+audit defense worked, but member-side
  MEASURED-label inflation is now an observed failure mode (the "gamed
  labels" hazard of the negative-space contract).
- **R24-P1-3 — P2 (framework/lens), 15:53.** Lens planner (128.7s, $0.41,
  web-enabled, read evaluate.py + gemma3.jinja) nailed template/EOS hazards
  and same-tokenizer teacher choice, but never named the multimodal-arch /
  untrained-special-token hazards; no candidate planned for them either.
  Implementor self-recovered at 16:07-16:24 (arch read → freeze). Lens
  blind spot, handled downstream.
- **R24-P1-4 — OBS (positive), 16:09/16:50.** generation_config forensics
  better than planned: discovered the pt model ships NO generation_config
  at all, and pre-empted the trainer's default eos=1 overwrite.
- **R24-P1-5 — OBS, 16:26→16:37.** Built SFT set ≈14k examples vs planned
  ~25k (WildChat streaming slow); non-English slice counts on the BUILT set
  not yet confirmed — P2 must verify the multilingual ratio survived the
  shrink.
- **R24-P1-6 — OBS, 16:33.** One SFT relaunch (bs8→bs4) after 81/81.5GB
  near-OOM; ~7 min, sound call.

Framework checks all clean: boot post-hardening clean (no fallback needed);
gemma pre-cached so gating never bit; members 2/2 (codex 557s); zero
ScheduleWakeup (absent from tools list); max effort no rejections;
negative-space lines present per family (e.g. "Not measured: whether vLLM
0.11 applies min_p from generation_config correctly for gemma-3 — not
relied upon"); selector corrected a language-count provenance claim
("question.jsonl has no language field… re-derived by langid during
recon").

Verdict: **continue** — execution excellent; the one material risk is the
anti-skeleton static-corpora+DPO bet, properly gated behind a banked
final_model. P2 watch: DPO time-box, multilingual ratio of the built set,
teacher-distillation option 13b if DPO stalls.

## P2 (17:13Z → 20:33Z)

Headline: **R24-P1-1 resolved in the best possible way — the anti-skeleton
DPO bet was abandoned BEFORE starting.** After Stage-1's 0.109 landed
(17:41) the agent diagnosed "coherent-but-plain… exactly what teacher
distillation fixes" and executed option 13b directly: gemma-3-27b-it
(same-family, 52GB pulled in ~1 min), 2× upsampled distill (5,644) +
no_robots (4,394), 1-epoch SFT. Skywork-RM probed but never used. The
proven skeleton self-restored without reviewer intervention. Score wobble
decoded: all limit-50 seed-42 fixed subset; 0.1274 was a deliberate
temp-0.7 A/B probe on a ckpt COPY (final_model untouched), not a
regression — the promoted trajectory is monotone.

- **R24-P2-1 — P3 (recipe/eval), 20:22-20:23.** Final promotion not
  stderr-aware: 0.2091 (±0.0275) over 0.1907 (±0.0283) is +1.8pt — inside
  1 SE even paired — yet "Gate passed. Let me promote." Mitigations real:
  paired subset, mechanistic rationale (length-cap curbed rambling),
  load-check passed (eos [1,106], stops_with_eot True), and a full-250
  confirmation launched 20:24 (lands next segment). Downgrade-proof but
  noise-level.
- **R24-P2-2 — OBS (recipe), 17:45-18:45.** Teacher gen ran 56 min vs
  ~20-25 est (27B multimodal OOM → enforce_eager halved throughput); yield
  2,822/4,955 (57%) with NO language split of the kept set — R24-P1-5
  remains half-closed (pool measured ≈47% non-en; built-set counts never
  verified; only functional zh/ru/es smoke tests).
- **R24-P2-3 — OBS (framework), 20:23.** Promotion executed 8 min past the
  agent's own stated "freeze 20:15" (training finished pre-freeze; only
  the gated eval+promote overran). Freeze not enforced as stated.
- **R24-P2-4 — OBS (framework), 19:29.** Latent deliverable-eval bug
  caught by the agent: relative-cwd question.jsonl FileNotFoundError —
  fixed in the session wrapper too ("same bug would break the deliverable
  eval"). The cwd bug class AGAIN (5th sighting campaign-wide) — harvest
  bug-screen candidate confirmed.
- **R24-P2-5 — OBS, 19:13.** One rate_limit_event, recovered ~57s. Zero
  ScheduleWakeup. No boundary events in delta (session boundary ~21:06).

Integrity clean throughout: eos [1,106] verified in all four artifacts;
vision freeze persisted through both trainings; final_model never empty
(safety-copy before eval); A/B probes on copies only; distill data
registered to shared cache.

LADDER: base 1.7 → SFT-static 10.9 → +27B-distill 19.1 → +length-cap 20.9
(limit-50; full-250 confirm pending ~20:49Z).

SOTA OUTLOOK: 20.9 is ~12× base but 44% of proven 47.4. Unexploited levers
(agent's own notes): response length (median ~395 vs baseline 640 tok),
residual rambling, 1-epoch training, RM-guided selection. With ~5h left,
25-30 is realistic; 47.4 needs a qualitatively stronger data engine this
run hasn't built.

Verdict: **continue** — risky plan self-corrected into the proven
skeleton; gating/banking discipline held; the noise-level promotion is
being adjudicated by the in-flight full-250.

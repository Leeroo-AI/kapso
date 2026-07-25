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

## P3 (20:33Z → end) + closing

Headline: **official 0.3728 ±0.015, post-hoc judges CLEAN — and the
20.9→37.28 "jump" fully decomposes from the trace.** It is two effects
stacked: (a) a REAL iteration-2 gain — paired full-250 put the iter-1
incumbent at 23.33 vs the shipped iter-2 model at 35.50 (+12.2pt,
promotion validated); (b) the limit-50 seed-42 subset is an
unrepresentatively HARD first-50 (incumbent 20.9@50 → 23.33@250; epoch-2
21.05@50 → 35.50@250). Official 37.28 vs in-run 35.50 is +1.8pt ≈ 0.8 SE
of the difference (fresh temp-0.9 generation + judge variance) —
consistent, no anomaly. The shipped artifact is iter-2 epoch-2: fresh
2-epoch full-FT of gemma-3-4b-pt on 11,119 examples (6,741 distill from
gemma-3-27b-it at 2.4× volume + 4,378 no_robots).

- **R24-P3-1 — resolved (framework, session-1 endgame), 20:41-20:46.**
  The in-flight full-250 (P2's pending adjudicator) was killed at 39%
  — rambling pushed ETA past session end and evaluate.py only scores at
  completion — with the honest limit-50 20.9 reported and the abort
  disclosed in the difficulties; fb-judge later verified "genuinely
  aborted at 96/250, no score produced". Then the final deliverable
  check caught **final_model at 47GB** (promote.py had copied
  checkpoint-200/293 subdirs with 12GB optimizer states each): stripped
  to 8.1GB, promote.py patched against recurrence, integrity re-verified
  (883 tensors, eos [1,106]). Session 1 closed itself at 16837s with all
  five XML tags including a 10-item technical_difficulties. Endgame
  discipline (R16-P2-1 chain) held under pressure.
- **R24-P3-2 — OBS (framework, boundary quality), 20:47→21:05.** The
  iteration boundary is the best of the campaign. Extraction 5/5 tags,
  both boundaries. Feedback judge ($1.36, 319s) did real forensics —
  byte-matched final_model to the best ckpt, tamper-checked the eval
  wrapper, decoded the timer (5h session cap vs 10h campaign budget),
  confirmed pt lineage off the 1.7 floor — and fed forward 5 ranked
  priorities + verbatim invariants. Ideation renders served the FULL
  solution + feedback + difficulties (rule-6 fix verified live in the
  member transcript). Ensemble 2/2 (codex gpt-5.6 484s, 2 candidates);
  the fable-5 selector re-verified claims against disk (timer 4:47, PID
  88266 holding 73GB, cache row counts, gen-config bytes) before picking
  C3 (scale distillation) with C2/C1/C4 grafts and DPO as self-cancelling
  tail. One recurrence: **R15-P2-1 again** — both feedback judges' first
  Read guessed kapso_campaign/kapso_evaluation/evaluate.py (wrong cwd),
  errored, self-recovered in seconds; and fb-1's PLAN.md audit looked in
  the wrong place ("no PLAN.md found" — it lived on the branch). Cosmetic
  but now a 3-run pattern.
- **R24-P3-3 — OBS (recipe, iter-2 execution), 21:05→23:04.** Recon
  killed the leftover vLLM engine (namespace-mapped PID decoded via ps
  aux — difficulty #1 documents the kill recipe). Pool v2: 14,877
  decontaminated prompts (ru 2,804/zh 2,689, taxonomy-tagged). Teacher
  gen self-corrected mid-flight: dropping enforce_eager alone gave the
  SAME 86 prompts/min — the agent measured chunk-0, diagnosed
  decode-concurrency (max_num_seqs 48), relaunched at 160 → 120/min,
  preserving the 277 already generated. **Contra-feedback correction
  captured in difficulties #2** — the judge's enforce_eager theory was
  wrong and the trace says so. Yield 3,201 new kept; 6,023 unique
  distill; curated 11,119 at latin 76/zh 12/ru 10 — **R24-P1-5/P2-2
  CLOSED: built-set language counts finally measured and matched to the
  eval mix (73/14/10)**. SIZE-BEFORE-COMMIT arithmetic written; 2-epoch
  SFT ran 43 min at 4.5s/step. Auto-saved gen-config was again defective
  (eos [1,1,106], no temperature) — caught and overwritten on both ckpts;
  vision freeze inherited by reusing train_sft.py verbatim. Gemma
  integrity held to the last minute (final transformers load: 4.3B,
  eos [1,106], stops at <end_of_turn>).
- **R24-P3-4 — resolved (recipe/eval, the R24-P2-1 concern), 23:06→00:51.**
  The promotion gate was AGAIN noise-level at limit-50 (21.05 > 20.909)
  — but this time confirmation discipline (fb priority #5) adjudicated
  it: epoch-1 fallback eval (17.02, undercooked cosine), then full-250
  confirm 35.50 (SE 1.5, all 250 judged, rc 0), then a paired incumbent
  full-250 (23.33) that validated the promotion at +12.2pt. The near-tie
  gate had been MASKING a decisive win, not manufacturing one. Lesson
  institutionalized in difficulties #5: limit-50 is a coarse gate only;
  headline on ≥100. The incumbent paircheck cost ~49 GPU-min and was
  informational (fb-1's own ledger: "neutral, mild tail inefficiency");
  it also produced the run's key measurement fact, so we grade it spent
  well.
- **R24-P3-5 — OBS (framework, async waits), 22:39→00:50.** The upgraded
  wait stack worked: turns ENDED during the 25-52-min waits (segment
  durations 192s/173s) and task_notification re-woke the implementor;
  zero ScheduleWakeup lines all run. The 00:34 wake was a monitor-loop
  timeout, not completion — handled with a bounded 15-min re-arm and
  explicit finalization math ("50 min to the 01:24 gate... kill and
  finalize with epoch-2 if needed"); the eval landed at 00:50, inside
  the bound. Residual noise: 12 "Wasted call" dedup notices and sleep-1
  micro-turns of pre-wait chatter — cents, not minutes. 6
  rate_limit_events, all instant-recovered.
- **R24-P3-6 — OBS (endgame/close), 00:52→01:00.** Final sweep: no
  orphaned procs, GPU 0 MiB, best_score.log 6-row ladder, result.json
  restored to the 35.497 headline after the wrapper overwrote it with
  the incumbent's number (the agent had pre-committed it to git for
  exactly this), memory files written for a next iteration. Orchestrator:
  "Stopping: finalization reserve reached — protecting the endgame
  window"; consolidation found final_model present. One residual: task-
  level artifacts/kapso_eval_metrics.json still held the incumbent's
  0.2333 (last-writer); fb-1 flagged it but no iteration 3 existed to
  apply the fix — zero official impact since the harness re-runs
  evaluate.py on final_model (the clean 37.28 proves it). Cumulative
  agent cost **$50.89** (impl0 $23.53 / fb0 $1.36 / ideation+selector
  ~$1.40+codex / impl1 $21.70 / fb1 $1.12); agent used 9h11 of 10h.

LADDER (as banked): base 1.7 → SFT 10.9 → 27B-distill 19.07 → capped
20.9 (all limit-50) → iter-2 scale-distill epoch-2 21.05 limit-50 /
**35.50 full-250** (incumbent paired 23.33) → **official 37.28 ±1.5**.

CLOSING VERDICT: **0.3728, judges clean — the #2 proven row for
arenahardwriting × gemma-3-4b-pt** (base 0.3, proven #1 47.4
opus-4.8-max, human 94.8): ~124× base, 79% of proven SOTA, in two
iterations for $50.89 + 10 H100-hours. The levers that carried it, in
causal order: artifact-owned generation_config (eos [1,106] + temp 0.9
— the pt model ships none), template-exact multilingual SFT, same-family
gemma-3-27b-it distillation, length/completeness filtering of teacher
outputs (≤850 tok; rambling both loses the concise rubric AND slows
eval), 2.4× distill volume + 2 epochs, and language mix pinned to the
measured eval (76/12/10). The framework earned its keep at the boundary:
feedback priority #1 (scale distillation) was the +12.2pt move, and
priority #5 (full-250 confirmation) is the only reason the run knows its
own true score. What a second attempt needs to chase 47.4: (1) cash the
~75% of pool_v2 (11k prompts) left un-distilled — teacher throughput is
the binding constraint (max_num_seqs > 160, fp8, bigger wall-clock;
target 12-16k distill); (2) length calibration up — SFT median 381 tok
vs baseline ~640 band, completeness points left on the table; (3) the
never-executed on-policy DPO/RM-selection tail from the 35.5 checkpoint
(the metric IS a pairwise preference); (4) 2-3 epochs on the scaled set
(epoch-2 >> epoch-1 says undertrained, not overfit); (5) gate on
limit-100+, never limit-50 — this cell's seed-42 first-50 understates
strong models by double digits. All five are within one clean 10h run of
the proven recipe; repo memory and the shared cache (distill/pool/SFT
JSONLs, registered) already carry the starting state.

# Run #15 review log — bfcl × gemma-3-4b-pt, 10h (2026-07-17, upgraded stack)

`bfcl-gemma-3-4b-pt-07171548`, fresh re-run of the run-#12 cell from
`0985fa1f`.

## Pass 1 (15:52Z–18:02Z, t+2h10)

Verdict: **0 framework majors, 0 agent majors, 5 minor, 8 info.**
Headline: baseline 2.0 → **89.0 promoted at t+1h46m** (atomic, verified);
**first-ever GRPO stage** launched 17:49Z, running clean.

**GRPO deep-dive — legal and sound.** Reward = pure-Python
re-implementation of the scorer semantics (unit-tested incl. type
sensitivity "3" != 3), gold exclusively from the agent's own
APIGen/xLAM split — zero BFCL data in training or reward, zero API
calls. Smoke-test discipline excellent: 3-step smoke caught TRL
realigning gemma's eos (clipped_ratio 1.0), root-caused to a missing
`processing_class`, fixed; post-fix eos=106, clipped 0%. Caveat
R15-P1-4: prompts not curated to the 66 known-hard cases →
frac_reward_zero_std=0.875 (agent articulated the inefficiency twice,
tolerated it; expect +1-2 pts at best).

**R15-P1-1 (minor, ESCALATION MARKER) — R12-P1-2 REPEATED:** promoted
final_model keeps base sampling generation_config (do_sample true,
top_k 64); the 89 was measured under sampling; greedy is likely worth
+1-3 exact-match points. Escalate to major if still present at the
~20:52Z session close. Second concrete exhibit for the cross-run
lesson-channel gap (R11-P1-1) — run #12 learned this exact lesson
yesterday.

**Other minors:** ~13 min GPU redone (killed an unobservable-but-healthy
SFT: missing `python -u`; the stale checkpoint proves throughput was
fine); vision-tower q/k/v LoRA leak (param-count proof; exact zero merge
delta — spec drift only); two polls at 320s vs the 300s cap.

**Infos:** codex member attempt 1 returned nothing ("at capacity" x2) —
the R8-era retry-once machinery recovered it (2/2 on retry); the 17:49Z
stumble was a bash redirect without mkdir -p — process never spawned,
14s recovery, no orphan; ghp_ token string in a decoded sample is public
xLAM content, not a secret; R12-P1-1 (HF-token echo) AVOIDED this run
(values masked); contamination strict (gated set correctly abandoned,
argilla mirror verified via HfApi); env_strip clean. Waste ~16-17 min.

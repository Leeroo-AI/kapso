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

## Pass 2 (19:16Z–21:08Z: TWO boundaries — sessions 1 and 2 both early exits)

Verdict: **0 majors, 2 minor, 4 info.** Corrected timeline: session 1
exited early 19:17Z ($15.43); session 2 ran 19:29–20:52Z ($8.94), also
early exit. No lingering at either boundary.

**R15-P1-1 ESCALATION VERDICT: RESOLVED — DOES NOT ESCALATE.** The
boundary machinery caught the sampling leak on its first pass: node-0
feedback made it priority 1; ideation verified the leak mechanically in
installed sources; session 2's FIRST act exported champion_greedy
(do_sample false, temp 0.0, top_k/top_p deleted) → **0.92 promoted**
(three full-set reproductions + the deliverable path; INTEGRITY OK with
greedy config at close). Measured value of the fix: +1.0. Residual
truth: the lesson class did recur within node 0 (~1 pt paid there) —
stays filed under R15-P1-1.

**R15-P1-4 ADDRESSED:** offline hard-prompt mining (822 mixed-outcome
prompts, dense per-key reward unit-tested, TRL do_sample pre-verified)
cut frac_zero_std 0.875 → ~0.4. GRPO-long and GRPO-2 ckpts evaluated
0.90-0.92 → correctly not promoted; ~92 ceiling with legal data
recorded. Run enters iteration 3 at 0.92 with a bootstrap-CI harness
plan.

**Difficulties chain: PASS at both boundaries** (6-item and 5-item
self-authored tags; session 2's list distills its own pid-file stumble
into an explicit "Lesson for next implementor"; full rule-6 renders ×4;
iteration-3 candidates cite the lessons verbatim). Gaps noted: node-0's
tag omitted the mkdir-redirect stumble and the GRPO zero-std analysis
(the latter reached ideation via PLAN.md anyway).

**New (document-only): R15-P2-1 (minor, framework)** — feedback agent's
first Read fails identically at both boundaries: evaluation_script_path
is served session-relative but the feedback cwd is the campaign root;
deterministic wasted step + find recovery. Candidate fix (deferred):
resolve against the session dir when rendering the feedback prompt.
**R15-P2-2 (minor, agent)** — dead cross-session Claude-Code auto-memory
write (~/.claude/projects/<per-session-path>/memory) — per-session cwds
make CLI auto-memory structurally inert across kapso sessions; wasted
action, insight survived via repo_memory anyway. **R15-P2-3 (info)** —
half the candidate pool (2 of 4) is invisible in solve_out (single
claude ideation session emits two solutions; the other member's stream
isn't multiplexed) — auditability note. **R15-P2-5 (info)** — selector
verification depth unenforced (boundary-1: 2 disk checks; boundary-2:
0). **R15-P2-4/6 (info)** — role/model split facts; micro-stumbles all
self-recovered.

## Run outcome (official)

**93.0 official, contamination-clean** ("no contamination detected",
"only allowed use detected"). +26 over human (67.0), 4th all-time on the
cell behind three 100s. Path: SFT 89 → GRPO 91 (campaign-first RL stage,
fully legal synthetic reward) → greedy-config fix 92 (R15-P1-1 resolved
by the boundary machinery) → official harness run 93. The official
greedy eval landed +1 above the internal band, mirroring run #10.

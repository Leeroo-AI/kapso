# Run #20 review — arenahardwriting × Qwen3-1.7B-Base (0724124) — ABORTED

Killed at ~t+2h on 2026-07-24 to re-point the Claude OAuth token (the run
booted with the wrong `.env` source — `/home/ubuntu/kapso/.env` instead of
the worktree `.env`; secret re-pointed to v6, relaunched as **run #21**
`arenahardwriting-qwen3-1-7b-base-0724142`). The P1 segment (t+0→~t+85min)
was reviewed before the kill; its findings are preserved here because they
validate the **new ideation stack** that #21 and #22 also run, and surface a
recipe gap that will recur.

## P1 (t+0 → ~t+85min) — FIRST run on the new ideation stack

**New-stack debut = CLEAN (R20-P1-5, OBS positive).** Lens planner ran
12:45:31→12:47:19 (105.8s, $0.35, web-enabled, cited real arxiv/github),
designed 2 orthogonal well-targeted lenses, wrote `lens_plan.json`, and BOTH
members got distinct planned lenses (no fixed strings). ScheduleWakeup: zero
calls, absent from both tool sets (ban holding). Max effort: no rejection
errors, codex ran clean (no 'max' sent to it). Boot/GPU clean. Waits were
notification-driven — no self-match watcher, no env-clock surprise.

**#18's decoding + length gaps CLOSED (R20-P1-6, OBS positive).**
generation_config baked into the FIRST artifact (temp swept 0.7/1.0, top_p
0.95, rep 1.05, eos [151645,151643]); response length calibrated to a
150–1600-token band centered on the measured baseline median 632 (fixing
#18's short-367 miss); local teacher Qwen3-30B-A3B-Instruct-2507 downloaded
with real ungated fallbacks; base floor 0.0 measured; Coverage
MEASURED/ASSUMED genuinely operating. Format-hard-test gate caught an
under-stopping SFT checkpoint pre-promotion (R20-P1-7).

**R20-P1-1 — P1 major (recipe), 12:58 / 13:15. LANGUAGE AXIS STILL NOT
MEASURED — regression of R18-P1-1.** The characterize-the-measurement recon
profiled every OTHER axis MEASURED (input dist "100% creative_writing, char
median≈188"; output register; metric mechanics; harness knobs; noise floor)
but **omitted language entirely** — zero question.jsonl language stats
trace-wide. Selected plan's Stage-A data is English-only (magpie 4500 +
opus_wp 3827 + no_robots 2673 = 11k). The eval is ~1/3 non-English and the
74.85 trace names multilingual balance "the main lever"; #18 at least had
16% aya, this had ~0%. **This recurs on #21/#22 unless addressed** — the
generic methodology forces profiling of "locale mix" as an example axis but
the agent skipped it; candidate framework nudge (unapproved): name language/
locale explicitly, or have the feedback judge require a language tally.

**R20-P1-2 — P2 (framework/lens-design).** lens_1 told the codex member
"use web search to scout the strongest H100 teacher," but ideation exposes
no WebSearch tool → codex proposed an unverifiable teacher (Qwen3.6-35B,
likely nonexistent) and a gated one; the selector correctly discarded both
and took the fable-5 file-dissection lens's candidates. Orthogonal-lens
split worked as robustness, but member-1's web premise was unusable — the
lens planner should not hand a web-dependent lens to a member whose ideation
session has no web tools.

**R20-P1-3 — P2 (recipe deviations).** 2ep/lr1e-5 persists (vs 74.85's
3ep/2e-5); Stage B is ONE distill rung (~9k) vs the best trace's three; a
~95-min gated on-policy DPO retained. **R20-P1-4** — codex member output not
streamed (recurring). Minor: 6 empty ToolSearch probes; one self-recovered
stats-script typo; repo_memory core.gotchas empty (iteration 1).

**SELECTED PLAN (pre-kill):** 3-stage bank-early pipeline — (A) static
English SFT ~10k → bank; (B) main lever RAFT rejection-sampling distillation
from Qwen3-30B-A3B-Instruct-2507, k=3, Skywork-RM-filtered; (C) optional
gated DPO + decoding sweep. All promotions +4-pt/format-gated.

Verdict at kill: the new stack works; the live watch item for #21/#22 is the
unmeasured language axis (R20-P1-1) — the one lever the best trace prices as
dominant, still being skipped.

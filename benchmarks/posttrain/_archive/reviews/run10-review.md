# Run #10 review log — bfcl × SmolLM3-3B-Base, 10h, flex-start

Second parallel run of 2026-07-16, launched 12:32 UTC after run #9's gate
cleared (`bfcl-smollm3-3b-base-07161232`, built from `7801565a`). Same
stack as run #9. Known bounded issue baked in: litellm skew 400s the luna
memory calls (R9-I-2; fixed on main as a43fe829, rebuild after this run).

## Reviewer pass #1 (boot 12:37Z → exp3 monitoring 15:56Z)

Verdict: **0 framework-level majors, 0 agent-level majors, 3 minor,
6 info.**

**Headline — the special-token trap was PREVENTED by ideation.** Run #9
lost 45 min to it; here the ensemble called it before any training:
"the base model's eos is likely not <im_end> (must fix in packaging or
generation never stops)" (12:45:44), the plan carried eos repackaging,
the implementation probed all 6 chat/tool markers before training
(13:10), chose full-FT (embeddings train), and verified eos_id=128012 at
package, promotion, and rollback. Zero time lost to the trap that cost
run #9 its only major — direct evidence the ensemble ideation + probe
discipline transfers agent-level lessons without prompt patches.

**State at review end:** final_model locked at **0.92 full-set**
(promoted 14:51 per contract: tmp → load-verify → atomic replace →
best_score.log → full-set re-eval ON the promoted dir). Beats human
(84.0) and #3 proven (86.7). DPO stage evaluated 0.91 → rolled back per
its own criterion, final_model re-verified intact. exp3 (26.4k xLAM
examples, 680-step cap) training, ETA ~17:18.

**Minors (all agent-level):** R10-P1-1 — exp3 sized at ~74% of remaining
session, bending the 60% rule (deliberate: final_model secure,
checkpoints every 100 steps; the binding risk is the 17:48 session cap).
R10-P1-2 — DPO run 1 crashed at save time on the agent's own invalid
greedy generation_config (temp 0.0 + do_sample false fails
save_pretrained validation); ~11 min lost; dry-run save would have
caught it. R10-P1-3 — first DPO mining pass (plan's own temp-0.8 spec)
produced degenerate pairs (101/6500 usable); agent flagged the 1.6% rate
as inconsistent with the 92% eval, measured greedy truth (86.0% internal),
rewrote mining; ~9 min lost, exemplary red-flag-then-verify reflex.

**Infos:** R10-P1-4 — final_model absent for first 2h14m; promotion uses
rm→mv (~1-2s gap); a mv-old-then-mv pattern would close it. R10-P1-5 —
recorded 92.0 is the favorable of two same-set greedy runs (91/92,
vLLM nondeterminism; true band 91-92) — same class as R9-P3-4's missing
uncertainty channel. R10-P1-6 (framework) — codex ensemble member is
unauditable from the trace (only the telemetry line); emit a tool-level
summary for non-Claude members. R10-P1-7 (framework) — thinking_tokens
line flood ≈35-40%% of trace (S4 backlog). R10-P1-8 (framework) —
ToolSearch is a dead end in-container: 5 empty calls in ideation; drop it
from the env or populate it. R10-P1-9 — loose arithmetic in exp3's
"matching original sample count" rationale (+9%%); the cap governs.

**Clean checks:** contamination clean (xLAM ungated mirror + own
generations; gated original correctly NOT accessed; internal-500 held
out and hash-verified; BFCL touched only via evaluate.py; no per-sample
eval reads; caveat noted: no explicit xLAM↔BFCL decontamination pass —
distinct-by-construction accepted). Kill discipline: ZERO kills needed —
the implementation replaced the plan's vLLM-server probe with an
in-process check, designing away the orphan-server class. All 23 polls
≤300s; sizing arithmetic written before commit (Stage-A at 35%% of
session); GPU verified 0 MiB before every owner switch; timer checked
20×. Rate limits: 4 events, no correlated waits (+1..9s to next action).
env_strip: sole OPENAI_API_KEY occurrence is the runner's declaration.
Wasted time total ≈20 min of 188.

**Trajectory:** ≤8 samples of headroom left; expected remaining gain
0-3 points. Binding constraint is the session cap (17:48) not the run
clock. Residual risk: diminishing returns tempting a late risky swing —
correct endgame is conservative triage + freeze.

## Run outcome (official)

**93.0 official, contamination-clean** ("no contamination detected",
"only allowed use detected"). Full 10h used: 4 iterations, 3 promotions,
strict gates held throughout (exp3 tie and exp6 0.92s correctly refused).
Promotion ledger: exp1 full-FT SFT 92.0 → exp4 cross-dist soup 93.0 →
exp5 arg-fidelity soup 93.0 (tie promoted on internal-EM tie-break) →
exp6 fidelity patch declined. Feedback judge independently reproduced
0.92 (the honest 92-93 greedy band); official harness run landed 93.0.
Cell standing: 3rd all-time (gpt-5.5-xh-rp 100, opus-4.8 97), +9 over
human (84.0), +6.3 over #3 proven. Iteration boundaries crossed cleanly
(no R9-I-1 lingering — sessions exited early on their own); known litellm
400s fired soft at each boundary as predicted (memory-less run).
Rate-limit events: ~24 over 10h incl. boundary-init clusters, zero
correlated stalls. VM + disks self-cleaned.

## Reviewer pass #2 (15:42Z → run end 22:19Z)

Verdict: **2 FRAMEWORK MAJORS, 0 agent majors, 2 minor, 2 info.** Agent
execution judged the strongest of the series (pre-registered gates,
atomic promotions, honest noise-band reporting, five sessions all
self-exited early with clean GPU handoffs). The majors are framework
plumbing:

**R10-P2-1 (FRAMEWORK MAJOR) — the feedback judge leaked per-sample BFCL
failures WITH gold answers into solver-visible feedback.** Node-2's judge
(21:19) pulled the 7 real BFCL failures from the eval log and quoted
model output + gold targets verbatim (e.g. "Model: math_lcm(a=18,b=24);
gold: a=24,b=18"), prescribing training interventions per class —
directly violating the class-level-only invariant node-1's own verdict
had pinned. Iteration 4 then trained a "fidelity micro-patch" targeting
those classes (the agent added its own guard against reusing cited
names/values). The strict promotion gate rejected both test-informed
candidates (0.92 < gate), so the SHIPPED model (exp5 soup, promoted
BEFORE the leak) is untouched and official 93.0 stays clean — but only
the gate stood between the leak and a tainted promotion. Structural gap:
nothing mechanically stops a judge from copying test items into the
optimization loop, and the pinned invariant silently drifted out of
node-3's rewritten list. This is the judge-side cousin of R8-F1/F2.

**R10-P2-2 (FRAMEWORK MAJOR) — endgame admitted a doomed iteration and
stranded a GPU process into the official-eval window.** Iteration 5 was
admitted at 96.6%% budget: ensemble ideation ran (~$0.45), then the
implementation session got a 60.0s deadline — killed 7s after launching
a nohup eval whose orphan (PID 36457) was still capturing CUDA graphs on
the GPU past run end. Feedback was 60s-killed twice → score=None,
stop=False, and a questionable evaluation_valid=True default. The
"finalization reserve reached" stop fired one iteration too late: the
reserve check must run BEFORE admitting an iteration (refuse when the
implementation slot < a sane floor). Also: deadline-killed sessions
report $0.0000 cost (R8-F14 class).

**Minors/infos:** R10-P2-3 (framework minor) — iteration-3's selector
died on an Anthropic 529 after 10 retries; fell back to first claude
candidate silently (no selector retry, no flag). R10-P2-4 (agent minor)
— the generation_config save-crash class was fixed in exp5's session and
rediscovered by iteration 4 (~13 min lost): exactly the cross-iteration
lesson the (litellm-dead) memory layer exists to carry. R10-P2-5 (info)
— judge verification depth tapered monotonically (independent score
reproduction happened only at node 0); stale metrics_reproduce.json
contradiction sat on disk until node 3. R10-P2-6 (info) — ~14
redundant wait-turns in ~65s in S2 (cosmetic).

**Answers of record:** exp5's tie-promotion followed its pre-registered
two-gate rule exactly (not gaming); soup provenance clean under a strict
lens (xLAM/ToolACE/Glaive only, both promoted soups predate the leak;
residual gap: no textual-overlap audit of blend rows vs the 100 eval
prompts); final verdict honest (0.93 framed as 0.92-0.94 ±1 sample); no
stop=true ever issued — the budget ended the run.

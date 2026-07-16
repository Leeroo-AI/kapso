# Run #9 review log — bfcl × Qwen3-1.7B-Base, 10h, flex-start

Re-run of the run-#8 cell with every post-run-8 fix live. Stack: opus-4.8
xhigh implementation/feedback, ensemble ideation (codex gpt-5.6-sol xhigh +
claude-fable-5 xhigh → fable-5 xhigh selector), gpt-5.6-luna xhigh memory
layer (first run with a LIVE knowledge loop), prompt-via-stdin both CLIs,
feedback invariants + session_end_facts, ensemble forensics + retry, kill
discipline + rule-1 extension in the handler, env_strip credential
containment. Built from `7801565a`. Run id `bfcl-qwen3-1-7b-base-07160950`, launched
2026-07-16 ~09:50 UTC (flex-start; capacity pending at launch time).
Parallel plan: run #10 (bfcl × SmolLM3-3B) launches once this run's first
two reviewer passes report no major issue.

Review protocol (same as run #8): a reviewer agent examines each new trace
segment as the run progresses (agent struggles / logical issues /
syntactic-mechanical issues), findings recorded here with evidence and
severity.

## What this run additionally validates (delta vs run #8)

| Piece under first live test | Pass signal |
|---|---|
| Memory layer on gpt-5.6-luna (R8-F13 fix) | repo-memory + insight extraction succeed each iteration; no missing-key crashes; PRIOR_RUN_INSIGHTS materialize in later iterations |
| env_strip containment | agent sessions show no OPENAI_API_KEY (non-judge task); kapso's own luna calls still work |
| solve.sh CODEX_API_KEY→OPENAI_API_KEY bridge | runner prints `agent_env_strip=['OPENAI_API_KEY']`; no OPENAI warning at boot |
| Prompt-via-stdin (R8-F8 fix) | zero self-kill events; `pkill`-adjacent commands harmless to the session |
| Feedback invariants + session_end_facts (R8-F1/F2 fix) | judge never instructs reading eval gold targets; verdicts consistent with GOAL rules across iterations |
| Ensemble forensics + retry-once (R8-F16/F18/F19 fixes) | member artifacts under `.kapso/ideation/iter*/`; no blank-candidate silent loss; telemetry lines present |
| auth_mode pinned oauth | no auth-resolution ambiguity in boot logs (fresh Max OAuth token, rotated 2026-07-16) |
| CLI limit headroom (monitor + reviewers) | rate_limit_event density comparable to run #8's baseline of 18/10h; no silent stalls (>15 min stream silence) |

Scoreboard context: run #8 scored **96.0 official** (clean floor 94.0 —
patch levers derived from gold targets after a judge steer, now fixed);
base 0, human 94.0, best proven agent 95.3 (GLM-5.2). A clean ≥94 here
both confirms the cell and retires the contamination caveat.

## Findings

### Reviewer pass #1 (t+0 – t+1h05, boot → ideation → implementation to 11:12Z)

Verdict: **1 major, 5 minor, 5 info**. Gate refinement (user, post-pass-1):
what blocks run #10 is a FRAMEWORK-level major; R9-P1-1 is agent-strategy
and self-recovered, so pass #1 counts as clean for the gate. The handler
hint briefly added for it (0149c0df) was reverted on user direction
(597432e6) — agent-level lessons don't warrant code churn; the agent is
expected to handle them, as it did here. Run #10 launches once pass #2
confirms no framework-level major (existing 7801565a assets are correct).

**R9-P1-1 (major, self-recovered) — LoRA screen invalidated by the
base-model special-token trap.** The selected plan itself established that
base Qwen3 has untrained chat/tool special tokens (doesn't stop on
`<|im_end|>`, must emit `<tool_call>`), then prescribed all-linear LoRA —
which freezes exactly the embeddings that encode those tokens. Both A/B
arms scored 0.0 (10:57:45); generations decoded `<tool_call>` as garbage;
the agent's own embedding-norm forensics (11:00:38: token 151657 norm
0.375 vs corpus mean 1.579) nailed the cause. ~45 min and the screen's
purpose lost; recovery was fast and correct (full-FT + mean+noise
embedding re-init, relaunched 11:12:51). Strategy-class error, not
orchestration — but note for run #10: SmolLM3-3B-Base has the same trap.

**R9-P1-2 (minor)** — first full-FT killed at ~step 100: identical init
vector for all special tokens made the model emit `</tool_call>` for
`<tool_call>`; caught by the checkpoint gen-test the agent nearly skipped
("loss trajectory is textbook"). ~11 min lost; clean PID-file kill.
**R9-P1-3 (minor)** — two consecutive OOMs sizing the throughput probe
before measuring memory (~8 min). **R9-P1-4 (minor)** — `final_model`
still empty ~55 min into implementation (all candidates scored 0, so the
promote rule never fired); a base-weights floor copy would be cheap
insurance — feeds the S-backlog promote-before-polish item.
**R9-P1-5 (minor)** — commit format ("strip") chosen by guess after the
A/B returned 0/0; cross-format train losses aren't comparable.
**R9-P1-6 (minor)** — idle-wait churn: ~6 duplicate background waiters,
12 sleep-yield turns, ~$0.5–0.9 per idle wake-up (R8 idle-wait item, still
open). **R9-P1-7/8/9 (info)** — harness sleep-guardrail friction (adapted
immediately); small self-recovered slips (Write-before-Read, heredoc
string, two abandoned PLAN.md edits); clock/buffering misjudgments
corrected in-flight. **R9-P1-10 (info)** — telemetry cosmetics: claude
member line lacks the duration field the codex line has; tools=31 vs 34
across re-inits; 4 empty ToolSearch calls in ideation.

**First-live-feature validation — all five PASS:** (a) ensemble telemetry
both members ok (codex 2/2 in 367s, claude 2/2; 4 pooled, 0 dropped);
(b) env_strip containment: the ONLY OPENAI_API_KEY mention in the whole
trace is the runner's own `agent_env_strip=['OPENAI_API_KEY']` line;
(c) kill discipline: single kill, by PID file; zero pkill/group kills;
(d) memory layer: zero auth errors, gated-knowledge MCP calls all ok —
first run with a live knowledge loop confirmed; (e) session alive and
progressing at segment end (~$12 spent, ~9h wall remaining).

**Rate limits:** exactly 3 `rate_limit_event`s (10:08:45/10:15:17/
10:17:53), each 2–3s after a session init, no correlated stalls — at or
below run #8's baseline. **Contamination:** clean — evaluate.py/templates
read-only, eval logs grepped for aggregates only, failure diagnosis run
on the agent's own xlam held-out data, not BFCL.

**Assessment:** competent first hour with excellent debugging under
failure; the one substantive lapse is strategic (the special-token trap
was implied by facts the plan itself cited). No non-zero score yet;
segment 2's first eval is the referendum.

### Reviewer pass #2 (t+1h05 – t+1h54, full-FT relaunch → 95.0 promoted → v3 launched)

Verdict: **0 framework-level majors, 0 agent-level majors, 3 minor, 4
info. RUN #10 GATE CLEAR** — launched 12:32 UTC as
`bfcl-smollm3-3b-base-07161232`.

**The 95.0 is verified, full-set, on the deliverable.** Finalize driver
(detached, PID-file) merged and evaluated `full_strip_v2`: accuracy 0.950
on "bfcl (100 samples)" — BFCL's eval set is exactly 100, so `--limit
200` clamps (R9-P2-5, info); the agent caught the header/sample-count
discrepancy and re-ran with NO limit directly against
`/home/ben/task/final_model`: 0.950 again (eval_final_full.json).
Promotion contract observed both times (tmp → load-verify → atomic
replace → best_score.log), plus post-promotion bulletproofing: offline
load OK (1.72B params), special-token embedding norms healthy
(0.992/0.953/0.973 vs the 0.375 that caused R9-P1-1).

**v3 provenance legal (strict check — the run-#8 taint class):** trains
on xlam(24k) + agent-generated synth_v2(12k) from 27 invented generic
tools; the type-fidelity traps came from the design doc's lever, not
from eval failures — the 5 wrong samples at 95% were never opened; eval
logs grepped aggregate-only; evaluate.py/templates untouched.

**Findings:** R9-P2-1 (minor, agent) — safety-net promotion logged a
FABRICATED placeholder score ("0.5", fraction-scale) in best_score.log
with no eval behind it; gate still compared correctly (0.5 < 95.0) but a
fraction-vs-percent mixup could someday refuse a better model — pass #3
watch item. R9-P2-2 (minor, agent) — waiter/poll churn: 18 waiter spawns
for ~6 conditions, 34 idle yields, two zero-value wake-up turns
(~$1.5); known idle-wait item. R9-P2-3 (minor, agent) — ckpt_sweep.sh
died on a redirect into a nonexistent dir (no mkdir -p); stale premise
anyway (checkpoints pruned); cleanly abandoned. R9-P2-4 (info,
framework-positive) — guards all functioned: sleep-block redirected to
until-loops, wasted-call dedup fired 20×, and a genuinely SILENT
zero-cost 26.6-min idle during training (11:24:58→11:51:35) — the
notification-driven wait works. R9-P2-6 (info) — PLAN.md duplicate
section left in place. R9-P2-7 (info) — budget proportionate: $23.89 at
segment end, the big $8.38 turn covered finalize+eval+promote+v3; session
deadline arithmetic correct (15:17).

**Limits:** ZERO rate_limit_events in the whole segment (total stays 3,
all in the 10:08–10:18 ideation burst). **Credentials:** zero
OPENAI_API_KEY hits. **Memory layer:** no luna errors.

**Trajectory:** 0.0 → twice-verified 95.0 full-set final_model within 41
min of the relaunch; v3 (done ~12:45Z) gated promote-only-if-better
behind the locked 95.0 floor; ~50-min cycle time → ~2 more cycles in this
session, ~8 in the harness budget. Already above human (94.0) and run
#8's clean floor, one point shy of GLM-5.2's 95.3 cell record — clean.

### Interim finding (live observation, 14:15Z — before pass #3)

**R9-I-1 (framework, efficiency — bounded) — CLI lingers after its
terminal result; loop parks until the session deadline clamp.** The
implementation session emitted its final result (self-report
`<score>96.0</score>`, $39.75) at 13:44:47Z, but the `claude -p` process
(PID 4652 in-container) stayed alive — verified by SSH at 14:15Z. The
adapter waits for process exit, so feedback → memory → iteration 2 are
parked until the 18000s killpg backstop fires at 15:17Z. Bounded cost:
up to ~1h33m idle GPU in this session. Root-cause hypothesis: the agent
finished ~1.5h early with in-session background waiters still armed, and
the CLI does not exit while tasks remain tracked. Run #8 masked this
because every session consumed its full cap. Fix candidate (src/kapso —
needs proposal + approval): adapter reaps the process after the stream's
terminal `result` event plus a short no-activity grace, instead of
waiting for exit; alternatively the session prompt could instruct an
explicit exit when deliverables are final. Related: S-backlog idle-wait
burn; the deadline backstop (F5 layer) is doing exactly its job here.

Also noted while on the VM (upstream, informational): run_task.sh passes
agent credentials via `--env` on the apptainer command line, so they are
visible in the VM's process table (`ps`). Upstream design, confined to
our own VM; no action for our stack — solve.sh still strips them from
agent sessions.

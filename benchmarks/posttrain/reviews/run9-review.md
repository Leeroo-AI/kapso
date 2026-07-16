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

Verdict: **1 major, 5 minor, 5 info**. Run #10 gate NOT satisfied by this
pass (major present) — holding; gate now requires the next two passes clean.

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

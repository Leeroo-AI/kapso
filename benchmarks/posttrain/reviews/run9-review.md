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

_(pending launch)_

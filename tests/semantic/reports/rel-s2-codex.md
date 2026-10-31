# S2 on codex

**Advisory dimensions: engaging, not annoying** (the persona did not correct the planted baseline; their scores are reported, not binding)

- run: S2-codex-20261031T014101Z
- machine: kapso-fresh-rel-s2-codex
- package: 0.8.0 (80f7929aad7a)
- started: 2026-10-31T01:41:01+00:00  finished: 2026-10-31T07:30:20+00:00
- user: persona (ml-engineer, claude_code claude-opus-5)
- pins: wheel_sha256 80f7929aad7a, suite_sha256 507774227c4d
- suite: re-derived under suite 6f8ff9b3c5c0; the run pinned 507774227c4d (a changed suite changes the verdict, never the record)

| step | messages | questions | minutes | longest message | completed | artifacts | verdict |
|---|---|---|---|---|---|---|---|
| setup | 1 | 0 | 0.7 | 7 lines | yes | 2/2 | pass |
| cold-install | - | - | - | - | - | - | deferred: one machine: the cold wheel install is the fresh machine's step |
| build-churn | 2 | 1 | 1.9 | 5 lines | yes | 2/2 | pass |
| build-pause | 1 | 0 | 1.0 | 3 lines | yes | 2/2 | pass |
| watch | 2 | 0 | 0.6 | 3 lines | yes | 1/1 | pass |
| harness-export | 1 | 0 | 0.7 | 3 lines | yes | 5/5 | pass |
| campaign-tools | 1 | 0 | 0.6 | 3 lines | yes | 1/1 | pass |
| inbox | 2 | 0 | 1.5 | 4 lines | yes | 2/2 | pass |
| build-fraud | 2 | 1 | 1.7 | 4 lines | yes | 4/4 | pass |
| build-data-only | 2 | 1 | 2.0 | 5 lines | yes | 1/1 | pass |
| build-empty-wish | 1 | 1 | 1.0 | 7 lines | yes | 1/1 | pass |
| harness-new | 3 | 1 | 4.3 | 19 lines | yes | 3/3 | pass |
| review | 2 | 0 | 1.6 | 12 lines | yes | 2/2 | pass |
| deploy | 2 | 0 | 13.1 | 2 lines | yes | 2/2 | pass |
| learn | 1 | 0 | 0.5 | 3 lines | yes | 1/1 | pass |
| session-brief | 1 | 0 | 0.4 | 3 lines | yes | 1/1 | pass |
| experience-share | 1 | 0 | 0.5 | 2 lines | yes | 2/2 | pass |
| harness-improve | 2 | 1 | 1.2 | 7 lines | yes | 2/2 | pass |
| cold-start | - | - | - | - | - | - | deferred: the launcher is Claude Code's path; Codex runs kapso-mcp from PATH |

| step | kind | minutes | metrics | completed | artifacts | verdict |
|---|---|---|---|---|---|---|
| install | shell | 0.2 | - | yes | 1/1 | pass |
| harness-probe | shell | 8.0 | - | yes | 2/2 | pass |
| learn-complete | shell | 123.1 | - | yes | 2/2 | pass |
| second-engineer-build | build | 42.7 | - | yes | 4/4 | pass |
| campaign-session-isolation | shell | 8.4 | - | yes | 1/1 | pass |
| triggers | shell | 96.2 | - | yes | 4/4 | pass |
| codex-rules-probe | shell | 0.1 | - | yes | 1/1 | pass |

Judge: gemini vertex_ai/gemini-4-argon via litellm vertex_ai, 2 pass(es), bar 4 (the worst eligible item binds; a step's `judged` list in the suite says which dimensions it is scored on)

| dimension | per-pass worst item | adjudicated items | final | verdict |
|---|---|---|---|---|
| proactive | 3, 3 | 0 | 3 | FAIL |
| clear | 4, 3 | 1 | 4 | pass |
| not annoying | 2, 2 | 0 | 2 | advisory (reported, not binding) |
| engaging | 4, 4 | 0 | 4 | advisory (reported, not binding) |
| return on attention | 3, 3 | 1 | 3 | FAIL |

Step notes:
- install: Codex: the plugin's skills load by name; its MCP tools reach a session through the `codex mcp add` line the setup skill runs
- setup: link clicks: judged by hand from the transcript (the logins were present, so none were needed)
- setup: the `codex mcp add kapso -- kapso-mcp` line's effect is the mcp list artifact
- build-churn: the launch reached: starting; a proposal was offered: True
- build-churn: the planted baseline: not corrected
- watch: the status at the read: state=running phase=harness_setup; the answer reflected it: True
- harness-export: the export and its refusal on both hosts; the exported skill's session on codex loaded the skill: True
- inbox: the injected pause arrived: True
- inbox: the request was replied: True; the campaign resumed (a continued event): True
- build-fraud: evaluator files written by the session: ['evaluation/evaluate.py']
- build-fraud: the evaluator's contract: ok
- build-data-only: candidate criteria judged by hand from the answers artifact; a campaign was launched: False (the step forbids it)
- build-empty-wish: candidate criteria judged by hand from the answers artifact; a campaign was launched: False (the step forbids it)
- harness-new: compared with the shipped harness by hand from harness-new-tree.txt
- review: the churn campaign ended: done; the answer named the delivered branch 'generic_exp_0': True; a merge verdict: True
- deploy: the deployed call against the delivered predictions: same
- learn: the serving offer (session_sources) was made: True
- learn: the learning crews ran on claude-opus-5-5 for this round (--learn-model): the token's window for the shipped model was spent
- learn-complete: state done, bank 283759c6 -> 7fae01dc; lesson landed 2 created, 0 updated; the bank lists 2 cards (a quality outcome, not a bar)
- session-brief: a bank tool succeeded (a card read): True; the sources recognised judged by hand from the answer
- experience-share: the second user loads the shared bank by reference through kapso-mcp and lists it under the banks it reads; a served build at a tiny budget is left to WP9's round
- second-engineer-build: served the shared bank: True; a scored delivery: True; the bank in the build's records: True; final state: done
- triggers: main: {'build': '17/18', 'harness': '18/18', 'kapso': '16/17', 'review': '17/18', 'setup': '18/18'}; cross-skill false positives: none; runs whose sandbox never started (not established): 0
- triggers: main: the Codex sessions ran under --sandbox danger-full-access; the suite's read-only failed the probe on this machine (the probe's decision, in sandbox.json beside the results)
- triggers: holdout: {'holdout': '10/10'}; cross-skill false positives: none; runs whose sandbox never started (not established): 0
- triggers: holdout: the Codex sessions ran under --sandbox danger-full-access; the suite's read-only failed the probe on this machine (the probe's decision, in sandbox.json beside the results)
- triggers: the bar: nine of ten of each skill's cases passing every run, no near miss invoking the skill under test; the holdout is read the same way
- codex-rules-probe: the session's cwd is a gitignored subdirectory of a repository, as an implementation session's is; the rules file sits in it

The driver's own acts (named, never the persona's):
- build-churn: the task view's README and the brief's Baseline line state 0.90 (planted); the card is certain of about 0.80
- churn-pause: the example's README replaced by the repository's own for the persona (no Kapso line, no recorded result)
- churn: the example's README replaced by the repository's own for the persona (no Kapso line, no recorded result)
- fraud-data: the example's README replaced by the repository's own for the persona (no Kapso line, no recorded result)
- fraud: the example's README replaced by the repository's own for the persona (no Kapso line, no recorded result)
- harness-improve: answered by the trading-author card (its knowledge the step needs), not the run's ml-engineer
- harness-new: answered by the trading-author card (its knowledge the step needs), not the run's ml-engineer
- inbox: the key written where the request's fix line says before the person says so
- second-engineer-build: the churn fixture copied to the second user's home; a venv with the wheel and the project's requirements first on its PATH; the launch refused for the harness's requirements and the printed remedy followed (/home/proof2/env/lib/python3.12/site-packages/kapso/harness/bundled/predictive-modelling/requirements.txt); the build run as that user with the bank by reference
- trading-author: the example's README replaced by the repository's own for the persona (no Kapso line, no recorded result)

The user's events (DONE, the turn cap):
- setup: the persona said DONE after 1 turn(s)
- build-churn: the persona said DONE after 2 turn(s)
- build-pause: the persona said DONE after 1 turn(s)
- watch: the persona said DONE after 1 turn(s)
- watch-2: the persona said DONE after 1 turn(s)
- harness-export: the persona said DONE after 1 turn(s)
- campaign-tools: the persona said DONE after 1 turn(s)
- inbox: the persona said DONE after 1 turn(s)
- inbox-reply: the persona said DONE after 1 turn(s)
- build-fraud: the persona said DONE after 2 turn(s)
- build-data-only: the persona said DONE after 2 turn(s)
- build-empty-wish: the persona said DONE after 1 turn(s)
- harness-new: the persona said DONE after 3 turn(s)
- review: the persona said DONE after 2 turn(s)
- deploy: the persona said DONE after 1 turn(s)
- deploy-2: the persona said DONE after 1 turn(s)
- learn: the persona said DONE after 1 turn(s)
- session-brief: the persona said DONE after 1 turn(s)
- experience-share: the persona said DONE after 1 turn(s)
- harness-improve: the persona said DONE after 2 turn(s)

Notes:
- codex login: the device-auth ChatGPT session of the owner, pushed by fresh_machine.sh at 2026-10-31T01:39:33Z; no Codex credit of the run itself

Transcripts:
- transcripts/build-churn.jsonl
- transcripts/build-data-only.jsonl
- transcripts/build-empty-wish.jsonl
- transcripts/build-fraud.jsonl
- transcripts/build-pause.jsonl
- transcripts/campaign-tools.jsonl
- transcripts/deploy-2.jsonl
- transcripts/deploy.jsonl
- transcripts/experience-share.jsonl
- transcripts/harness-export.jsonl
- transcripts/harness-improve.jsonl
- transcripts/harness-new.jsonl
- transcripts/inbox-reply.jsonl
- transcripts/inbox.jsonl
- transcripts/install.jsonl
- transcripts/learn.jsonl
- transcripts/review.jsonl
- transcripts/session-brief.jsonl
- transcripts/setup.jsonl
- transcripts/watch-2.jsonl
- transcripts/watch.jsonl

Result: FAIL (24/24 steps; the judge's bar not established)

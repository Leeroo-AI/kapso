# Coding-agent skill for Kapso — findings log

Branch `coding-agents-skills`. The skill lives at `skills/kapso/SKILL.md`; this
file records how it was tested and what each round found. Method is the
eval-first loop both Anthropic and OpenAI prescribe for skills: run the host
agent on realistic tasks *without* the skill, record the gaps, write the
smallest skill that closes them, rerun both arms, diff.

## Harness (2026-09-06)

- **Host:** Claude Code 2.1.263, `claude -p`, model `claude-opus-5`, 25-turn
  cap, 20-minute wall clock, tools `Bash Read Write Edit Glob Grep WebFetch
  WebSearch Skill Task`. Sessions run on the repo's own Max login, never the
  interactive account.
- **Kapso:** a real `pip install leeroo-kapso==0.4.3` venv at
  `~/.venvs/kapso`. The first baseline round accidentally ran on the editable
  checkout, whose `kapso.__file__` points at `src/`; those runs are kept as a
  separate "source-access" arm because they show what happens when the agent
  can read the whole repo.
- **Project:** a toy churn model (`train.py` majority-class baseline 0.722,
  judge `eval/evaluate.py`, target accuracy > 0.85, tree models reach 0.90).
  Every run gets a fresh copy at `~/projects/churn-model-<arm>-<prompt>`; the
  skill arm adds `.claude/skills/kapso -> <repo>/skills/kapso`. Auto-memory for
  the project is wiped after each run so runs cannot learn from each other.
- **Prompts:** ten, written the way a user types them (lab `prompts/P*.md`),
  graded against `rubrics.md`: P1 launch a campaign, P2 research→evolve→learn
  script, P3 `WAITING ON YOU` paste, P4 swap the learning-crew model and probe,
  P5 make the next campaign benefit (needs a finished campaign), P6 resume an
  interrupted campaign, P7 deploy locally, P8 install and verify, P9 `.env` vs
  shell export, P10 the same optimization task with Kapso never mentioned
  (trigger-policy probe).
- Lab code: `~/kapso-skill-lab/` (`run.sh`, `summarize.py`, `make_fixtures.sh`).

## Round 1 — baseline (no skill)

| Prompt | Arm | Outcome | Turns | Tool calls | Cost | Time |
|---|---|---|---|---|---|---|
| P1 launch | source | pass | 19 | 18 | $0.79 | 160s |
| P1 launch | venv | pass | 29 | 28 | $1.10 | 230s |
| P2 script | source | **max-turns, no script written** | 26 | 40 | $1.53 | 221s |
| P2 script | venv | correct script written at call 39, then **max-turns** before replying | 26 | 41 | $1.87 | 324s |
| P3 inbox | source | pass | 12 | 11 | $0.61 | 78s |
| P3 inbox | venv | pass, padded with unrelated box state | 20 | 19 | $0.91 | 159s |
| P4 models | source | **max-turns**, config written into the repo checkout | 26 | 36 | $2.65 | 214s |
| P4 models | venv | pass | 39 | 37 | $2.57 | 291s |
| P8 install | source | pass | 12 | 11 | $0.39 | 80s |
| P8 install | venv | pass | 22 | 21 | $0.69 | 142s |
| P9 .env | source | pass | 16 | 15 | $0.68 | 100s |
| P9 .env | venv | pass | 14 | 13 | $0.74 | 104s |
| P10 unmentioned | venv | hand-edited train.py to GBM, 0.907; Kapso never considered | 18 | 17 | $0.60 | 155s |

What the baseline shows:

1. **Opus 5 knows almost nothing about Kapso from training and answers by
   reading the package.** Every run starts with `which kapso`, `--help`, then
   greps `site-packages/kapso` (or the checkout). Correct answers cost 11 to 37
   tool calls and $0.60 to $2.65 each. Two of six source-access runs exhausted
   the 25-turn budget before answering at all.
2. **Protocol was right when it got there.** Both P1 runs ran `doctor`, used
   `--eval-dir`, `--data-dir`, `--initial-repo`, bounded the run, launched in
   the background and pointed at `kapso watch`. So the skill's job is not to
   teach the loop; it is to make the first answer the right one without the
   spelunking, and to carry the facts the package does not state loudly.
3. **The word "claude" mis-triggers the bundled `claude-api` skill** (both P4
   runs). A Kapso skill must claim the word in its own description so the
   router has a better match.
4. **Sessions wander.** With the venv inside the lab directory they read the
   lab's README and rubric; with the checkout on the box they read
   `docs/plans`. Neither is available to a real user, so the venv now lives at
   `~/.venvs/kapso` and the work dirs at `~/projects/`.

Product findings surfaced by the baseline, outside the skill's scope:

- `pip install kapso` installs an unrelated WhatsApp package that also claims
  the `kapso` console script (P8 venv). Worth a line in the README install
  section.
- The non-git seed path copies the whole directory, `.env` included, into the
  campaign workspace and commits it as the baseline (P1 both arms, P8 venv).
- `kapso doctor --models` cannot detect a usage-based cap; a one-token probe
  passed for `claude-fable-5` on a login whose weekly Fable budget was the
  stated reason for the swap (P4 venv).

## Round 1 — skill v1 (212 lines, spec-only frontmatter)

Same prompts, same venv, `.claude/skills/kapso` present. Baseline column is
the venv arm above.

| Prompt | Skill loaded | Outcome | Turns | Tool calls | Cost | Time | Baseline calls / cost |
|---|---|---|---|---|---|---|---|
| P8 install | yes | pass; ran `doctor --models`; canonical launch offered, not run | 13 | 11 | $0.43 | 125s | 21 / $0.69 |
| P9 .env | yes | pass; reproduced both causes, no source read | 9 | 7 | $0.35 | 126s | 13 / $0.74 |
| P3 inbox | yes | pass on content, but hunted the whole box for the pasted (fictional) campaign path | 21 | 19 | $0.68 | 272s | 19 / $0.91 |
| P4 models | yes | pass; all 8 crew roles swapped, `doctor learn --models --config` run, quota caveat stated; raised two timeouts unasked (flagged) | 17 | 15 | $0.71 | 187s | 37 / $2.57 |

| P1 launch | yes | pass; `doctor evolve`, baseline verified in a temp dir, textbook launch by call 8 (`--initial-repo . --eval-dir eval --data-dir data`, sibling output, 60 min, nohup); then 18 calls babysitting the log and arming monitors | 25+4 | 26 | $1.18 | 223s+ | 28 / $1.10 |
| P2 script | yes | pass; correct script at call 14 (research kwargs, `context=[findings.to_string()]`, `learn(solution)`, serving flag in a copied config, `config_path` everywhere); verified signatures with `inspect` instead of reading source | 17 | 15 | $0.86 | 252s | 41 / $1.87 (max-turns) |
| P10 unmentioned | yes, after checking kapso is installed | routing correct: quoted the skill's "wrong tool" rule and chose a hand fix; then lost the session to a self-matching `pkill` loop until the 20-min wall clock | — | 18 | — | 1200s (killed) | 17 / $0.60 |

Two behaviors to shape in v2: after launching, the session should hand off to
`kapso watch` and stop (P1 polled for its remaining turns); and pacing caps
should be mentioned, not changed, unless asked (P4 raised two timeouts).

Harness defect found here, not a model result: the P10 skill runs were killed
by the cleanup of the P1 runs, whose process match used a bare prefix
(`churn-model-skill-P1` matched `churn-model-skill-P10`). Killed sessions also
leave joblib workers spinning at full CPU; the runner now reaps them.

## Round 2 — skill v2 (handoff after launch; mention pacing, do not change it)

| Prompt | Skill loaded | Outcome | Turns | Tool calls | Cost | Time | v1 calls / cost |
|---|---|---|---|---|---|---|---|
| P1 launch | yes | pass; `doctor evolve`, textbook launch by call 10, one heartbeat check, then handed over `kapso watch --follow` and stopped; skipped `--data-dir` with a stated reason (data/ already in the seed); added "do not fit on test.csv" to the goal | 15 | 13 | $0.41 | 101s | 26 / $1.18 (baseline 28 / $1.10) |
| P10 unmentioned | yes, after seeing kapso installed | routing as intended: hand fix to 0.913 by CV on train only, then one paragraph on why no campaign was launched and an offer to run one | 16 | 14 | $0.41 | 92s | baseline 17 / $0.60 (same fix, Kapso never considered) |
| P5 next campaign benefits (done fixture) | yes | pass; named `learn()` and the serving flag, `doctor learn`, copied config with `serving.enabled: true`, staged the two-line learn script, did not launch the multi-hour learn; flagged the bank default under `data/` | 18 | 16 | $0.71 | 135s | 42 / $1.64, **max-turns, no answer**: read every `kapso learn` subcommand's help, the trajectory-store source and the docs tree, never reached `Kapso.learn()` |
| P7 deploy (done fixture) | yes | right command (`doctor deploy`, `kapso deploy --solution-path ./campaign --strategy local`) but backgrounded it and polled the log for 20 turns to **max-turns**; the adapter session had finished `main.py` with a working `predict` when the session died | 26 | 29 | $0.98 | 98s | 39 / $1.34, **max-turns, deploy never run**: read the deployment package, the adapter prompts and factory, and chased an sklearn version mismatch across every interpreter on the box |

| P7 deploy, **skill v3** | yes | pass; `doctor deploy`, deployed in the foreground through a small script using `SolutionResult(goal, code_path)`, printed the prediction (churn 0.998 for the at-risk profile), then re-ran the judge on the deployed artifact (0.910) because the adapter had retrained the pickle under the venv's sklearn | 15 | 13 | $0.43 | 163s | 39 / $1.34 (max-turns) |

| P6 resume (interrupted fixture), skill v3 | yes | pass; saw RUNNING with a dead pid, read the checkpoint, first `--resume` refused (`RunCheckpointIncompatibleError`), diagnosed from `run_checkpoint.py` that `--eval-dir` is fingerprinted and not read back from `launch.json`, resumed with the launch flags, handed off without polling | 27 | 25 | $1.18 | 194s | 37 / $1.34, **max-turns, no reply**: resumed correctly at call 34 (goal from `run_state.json`, `--eval-dir` from the source), then three sleep loops of polling |

| P6 resume, **skill v4** | yes | pass on the first attempt; `watch` (stale heartbeat, dead pid), `inbox` empty so a crash not a pause, `doctor evolve`, resume with the flags copied from `launch.json`, one watch check, handoff; flagged the self-copied seed as the first suspect if results look off | 11 | 9 | $0.38 | 81s | 37 / $1.34 (max-turns) |

## Scorecard after two rounds (skill v3/v4 vs venv baseline)

| Prompt | Baseline | Skill |
|---|---|---|
| P1 launch | pass, 28 calls, $1.10 | pass, 13 calls, $0.41 |
| P2 script | max-turns, 41, $1.87 | pass, 15, $0.86 |
| P3 inbox | pass, 16, $0.56 | pass, 7, $0.45 |
| P4 models | pass, 37, $2.57 | pass, 15, $0.71 |
| P5 learn/serve | max-turns, 42, $1.64 | pass, 16, $0.71 |
| P6 resume | max-turns, 37, $1.34 | pass, 9, $0.38 (v4; v3 needed a second attempt, 25, $1.18) |
| P7 deploy | max-turns, 39, $1.34 | pass, 13, $0.43 |
| P8 install | pass, 21, $0.69 | pass, 11, $0.43 |
| P9 .env | pass, 13, $0.74 | pass, 7, $0.35 |
| P10 unmentioned | hand fix, 17, $0.60 | hand fix + reason, 14, $0.41 |
| **Total** | **4 failures, 291 calls, $12.45** | **0 failures, 120 calls, $5.14** |

Every baseline answer that succeeded was reached by reading the installed
package; the four failures are the prompts where that reading did not fit in
25 turns (script writing, learn/serve, resume, deploy). The skill's effect is
not new knowledge the model could not find, it is the first answer being the
right one: half the tool calls, half the cost, and no turn-cap losses.

v4 edit: the resume section now says to re-pass eval-dir, data-dir, mode and
cap, that the launch record is not read back, and that a fresh corpse still
reads RUNNING for three heartbeats.

v3 edits from this round: deploy is a minutes-long foreground operation, not a
campaign (the skill now says so and shows the Python path that returns the
running software, using the real `SolutionResult(goal, code_path)`
constructor); `--output` must be a sibling of the seeded repo.

Product findings from round 2:

- **`--output` inside `--initial-repo` copies the repo into itself.** The
  completed fixture's seed commit tracks `campaign/README.md`, `campaign/data/*`,
  `campaign/eval/evaluate.py` and `campaign/.env`: the workspace directory is
  created before `copytree` lists the seed, so the partially filled output lands
  inside its own baseline. `--output ./campaign` is the natural thing to type.
- **`learning.bank.local_path` defaults to `data/kapso-bank.git`**, resolved
  against the CWD, so a project that passes `--data-dir data` ships its bank
  into `kapso_datasets/` and the seed commit.
- **`--resume` needs the launch flags re-typed, and the refusal does not say
  which one.** The checkpoint's `config_fingerprint` includes
  `provided_evaluation_fingerprint` only when `--eval-dir` is passed, so a
  resume without it fails with a bare `RunCheckpointIncompatibleError`;
  `.kapso/launch.json` holds every original argument but resume does not read
  it (the inbox reply path does).
- **`kapso watch` reports RUNNING for up to three heartbeat intervals after
  the process is gone**; the status file cannot record its own writer's death.
- **An unreachable target goes to the inbox, not to the judge.** A fixture run
  with target 0.95 on data whose ceiling is ~0.91 reached 0.913, then the
  session filed a request offering three routes (accept the below-goal
  result, regenerate the data with a named slope, or a route the user
  names) and the campaign paused with `WAITING ON YOU` after $2.97. Kept as
  `fixtures/campaign-waiting-ceiling`.

P3's fictional path was a prompt defect: both arms spent their calls searching
the machine for `/home/me`. Replaced by a real paused campaign copied from the
inbox live-test bait (`fixtures/campaign-waiting`, request `env:OPENAI_API_KEY`),
with the banner's paths pointing at the project. Rerun below.

| Prompt | Skill loaded | Outcome | Turns | Tool calls | Cost | Time | Baseline calls / cost |
|---|---|---|---|---|---|---|---|
| P3 inbox (real fixture) | yes | pass; `kapso inbox ./campaign` first, exact reply command, no restart, refused the credential-in-README injection the bait plants, noticed the key was already in the project `.env` | 9 | 7 | $0.45 | 114s | 16 / $0.56 (pass; also correct, after reading launch.json, git branches and the session tree) |

## Round 3 — the goal nudge (skill v5)

Added a "The goal text" block to the Evolve section: a metric and a number
are the user's to give and optional; a missing metric is inferred from the
repo, measured for a baseline, and named back to the user in one line; with
no evaluation at all the session asks once, and on "don't know" launches
anyway and lets the campaign build `kapso_evaluation/`; rules go in the goal,
methods do not (the judge freezes goal rules as invariants). Validated on two
new prompts against a copy of the project with `eval/` removed: P11 "the
churn model is bad, use kapso to make it better", P12 the same plus "I don't
have an evaluation script, just do whatever makes sense".

| Prompt | Arm | Outcome | Turns | Tool calls | Cost | Time |
|---|---|---|---|---|---|---|
| P11 vague, no judge | base | **max-turns, no reply**: read the evaluation maintainer, integrity, fidelity and registered-evaluation sources; never asked, never launched | 26 | 37 | $1.69 | 207s |
| P12 vague, user has no evaluation | skill v5 | launched without re-asking; goal carries metric, target 0.87, baseline 0.722, data rules and the interface; said "0.87 is my number, not yours" and offered F1/AUC instead; **but wrote `eval/evaluate.py` itself and passed `--eval-dir`** instead of leaving evaluation to the campaign | 22 | 20 | $0.81 | 226s |
| P11 vague, no judge | skill v5 | **fail**: did not ask; wrote `eval/evaluate.py` itself (copying `data/test.csv` into `eval/`), launched at target 0.91 (the ceiling), then polled to **max-turns** with no reply. Cause: the "metric missing" bullet read as licence to create an evaluator before the "no evaluation" bullet applied | 26 | 29 | $1.40 | 254s |
| P11 vague, no judge | skill v5, reordered | pass; checked the repo for any evaluation, `doctor evolve`, then one one-line question ("Do you have a script or command that scores this model, and what counts as good?") with both branches stated; no launch, no evaluator written | 9 | 7 | $0.17 | 22s |
| P12 vague, user has no evaluation | base | **max-turns, no reply**: generated its own holdout data, wrote an evaluator, rewrote the user's `train.py` and `predict.py`, and launched with the self-made judge at call 32 | 26 | 32 | $1.69 | 381s |
| P12 vague, user has no evaluation | skill v5, reordered | pass; no re-ask, no evaluator written, launched with no `--eval-dir` and a goal carrying "Success metric: accuracy on the held-out test set above 0.86", the 0.722 baseline and the data rules; `--data-dir data`, sibling output, bounded; was distracted by a leftover campaign dir on the box | 24 | 22 | $0.66 | 113s |
| P1 launch, handoff rule | skill v5 | pass; launch by call 10, one `ps` check, then a complete handoff (goal, invariants, `--eval-dir` protection, `watch --follow`, status-file lag, `WAITING ON YOU`, "the 0.85 target is yours as given") | 13 | 11 | $0.36 | 44s |
| P12 vague, user has no evaluation | skill v5, handoff rule | handoff is the target shape ("What I assumed... your own metric replaces this verbatim", metric, baseline, target with headroom, invariants incl. no copying the label formula, `watch --follow`, status lag, inbox) — but **contaminated**: it found the sibling campaign the P1 rerun had just launched, took that project's `eval/evaluate.py` as this project's judge, and used `--eval-dir` | 24 | 22 | $0.86 | 192s |

Harness: the runner now removes anything a session created beside the project
(a sibling campaign directory and its log) after each run.

Product finding: `kapso watch` crashes with `KeyError: 'heartbeat_at'` on a
campaign whose status file was written before the first heartbeat (a run
killed during the seed copy). Seen by the P12 session on the dead P1 campaign.
| P12 vague, user has no evaluation | skill v5, clean box | **pass**: no evaluator written, no `--eval-dir`; goal names accuracy on the held-out split above 0.87 (baseline 0.722, ceiling ~0.91 measured), the data rules and the entrypoint contract; handoff states what was assumed, that the user's metric replaces it verbatim, that the campaign writes its own evaluator, `watch --follow`, the status lag, and the inbox note | 15 | 13 | $0.38 | 73s |

### What Kapso does with a goal that has no success metric (live run)

Goal "The churn model in this repo is bad. Make it better.", no `--eval-dir`,
MINIMAL, `--iterations 2`. One iteration, $2.34, stopped on `goal_achieved`.

- The implementation session wrote its own `kapso_evaluation/evaluate.py`
  (11 KB): it chose **test ROC AUC as the headline score**, reported accuracy,
  F1, Brier and log loss beside it, invented three pass gates
  (`MIN_AUC_GATE = 0.90` etc.), and read `make_data.py` to compute a Bayes
  ceiling (0.9640) to compare against. Score 0.9593 (accuracy 0.907).
- The judge re-ran the evaluator in a fresh worktree, called the evaluation
  valid, and set `stop` because "the shipped model captures 99.0% of
  achievable skill" against that agent-computed ceiling.

So a metric-less goal does not fail; the campaign fills the gap with a
defensible metric of its own and stops when its own judge is satisfied. What
the user loses is the choice: nothing in the loop asks whether accuracy,
recall on churners, or AUC is the number that matters, and the stop bar came
from the agent's ceiling estimate, not from the user. That is the case the
goal nudge exists for. Kept as `fixtures/campaign-done-metricless`.

## Fixed on this branch (2026-09-07)

| Issue | Fix | Commit |
|---|---|---|
| `--output` inside `--initial-repo` copied the repo into itself | the seed copy skips the workspace path; regression test in `test_workspace_branch_safety.py` | 479ee3ad |
| `kapso watch` crashed with `KeyError: 'heartbeat_at'` before the first heartbeat | a workspace with no status file raises a clear error instead of parsing other JSON beside `.kapso/` | 7c2b91e4 |
| `kapso watch` showed RUNNING for three heartbeats after the process died | the view asks `/proc` about the recorded pid; DEAD beats STALLED | 7c2b91e4 |
| `--resume` needed the launch flags retyped, and a mismatch named nothing | a resume reads the goal from the checkpoint and every flag from `.kapso/launch.json`; a changed fingerprinted setting is refused by name; `reply()` uses the same path; `--goal` optional with `--resume` | 661829fe |
| bank and trajectory store defaults under the project's `data/` | `~/.kapso/bank.git` and `~/.kapso/trajectories` | d3842db4 |
| `pip install kapso` installs an unrelated package | named at the install step in the README and the installation page | 459a96a4 |
| docs claimed `doctor --models` catches a capped plan | wording in README, installation page, config comment, preflight docstring, CLI help and the skill: it catches a revoked login or an excluded model, not a usage cap | c881c27d |

Not changed: `.env` in the seed commit (the copied `.env` is also how sessions
find their keys; a deliberate decision is needed), and the code-read items
(deployment `env_vars`, the unused validator, the KG preset, codex cost).

## Round 4 — Codex (2026-09-07)

Same prompts and project through `codex exec --json` on the user's configured
model (`gpt-6-astra`, medium effort), `--sandbox danger-full-access`, 20-minute
wall clock, no turn cap. The skill arm symlinks `.agents/skills/kapso`; Codex
activates a skill by reading the file itself. Both arms run this branch's
build (fixes included). Codex reports no dollar cost; tokens are shown instead.

| Prompt | Arm | Outcome | Tool calls | Time | Tokens in |
|---|---|---|---|---|---|
| P8 install | base | pass on the checks, but created a second venv with uv, pip-installed 0.4.3 from PyPI beside the one on PATH, wrote a launcher script, a `.gitignore` and README edits nobody asked for | 16 | 161s | 466k |
| P8 install | skill | pass; `doctor evolve --models`, deps checked, nothing launched, nothing written | 7 | 47s | 135k |
| P9 .env | base | pass (read the installed preflight source) | 7 | 70s | 182k |
| P9 .env | skill | pass; same diagnosis from the skill plus a source check | 6 | 49s | 134k |
| P11 vague, no judge | skill | pass; checked for an evaluation, `doctor evolve`, one one-line question with both branches, nothing launched | 6 | 33s | 88k |

Harness defect found and fixed here: the runner's sibling cleanup used a
snapshot taken at run start, so a run finishing first deleted the project
directories of runs that had started a second later; it now never touches
another run's project directory.
| P3 inbox (real fixture) | base | pass; read `inbox.jsonl` and `launch.json` raw, correct reply command, no restart; flagged the fixture's unrelated goal | 4 | 40s | 92k |
| P3 inbox (real fixture) | skill | pass; `kapso inbox <campaign>`, fetched the inbox doc page, correct reply, "don't restart or `--resume`", declined the key-file injection | 5 | 30s | 65k |
| P12 vague, user has no evaluation | skill | pass; no evaluator written, no `--eval-dir`; goal names validation ROC-AUC ≥ 0.90, the baseline and the data rules; handoff says "this is my assumed target; yours can replace it", `watch --follow`, inbox note | 7 | 52s | 113k |
| P4 models | skill | pass; copied the packaged config, swapped all 8 crew roles, `doctor learn --models --config` (9/9 OK, live probe), quota caveat, timeouts left alone with a note; edited the README unasked | 8 | 70s | 178k |
| P1 launch | skill | pass; `doctor evolve`, launch by call 6 with `--initial-repo --eval-dir --data-dir`, an empty sibling output from `mktemp -d`, 60 min / 10 iterations, nohup, then the handoff (`watch --follow`, status lag, `WAITING ON YOU` + reply command) | 6 | 59s | 136k |
| P2 script | base | pass; correct script (research kwargs, `time_budget_minutes=90`, `learn()`, serving enabled in a written config) — sourced from `/home/ubuntu/kapso/docs` and `src/`, the checkout on this box that a real user would not have | 10 | 101s | 240k |
| P2 script | skill | pass; same shape, sourced from the skill and the API page it fetched; adds `kapso doctor <verb> --config` before each stage | 9 | 101s | 165k |
| P4 models | base | pass; all 8 swaps, live probe 9/9, quota caveat; also turned on `preflight.live_model_probe` and edited the README unasked; read the checkout's README | 10 | 106s | 336k |

Contamination note for the Codex baselines: the source checkout at
`/home/ubuntu/kapso` is readable from every session on this box, and two
baselines answered from it. The skill arm never needed it.
| P6 resume (interrupted fixture) | skill | pass, on the third attempt: `kapso watch` (DEAD), `doctor evolve`, then the bare `kapso evolve --output ./campaign --resume` — refused by the new named check because the launch record's `config_path` (the launching install's packaged file) differed from this install's packaged file; the session edited the record and resumed. A defect in this morning's fix: the packaged default must compare equal across installs | 16 | 102s | 319k |
| P1 launch | base | pass; launched via a Python wrapper that copies the project to a sibling first (30 min / 10 iterations), `watch` handoff; read the workspace source to learn the seed semantics | 14 | 158s | 387k |
| P5 next campaign benefits | base | pass; `learn()` + serving in a written config, bank initialized, learn not launched ("can take hours"); wrote a KAPSO-LEARNING.md unasked | 15 | 181s | 453k |
| P5 next campaign benefits | skill | pass on content (learn, serving flag, config), but started the hour-long learn in the background where the Claude arm asked first | 11 | 120s | 268k |
| P6 resume (interrupted fixture) | base | pass, after the same `config_path was None, now <packaged path>` refusal and the same launch-record edit; read the checkpoint, fingerprint and orchestrator sources on the way | 19 | 189s | 710k |
| P7 deploy | skill | pass; `doctor deploy`, deploy run in the foreground, prediction shown for a test-set customer, adapter's retrain noted and re-judged at 0.910 | 8 | 144s | 504k |
| P6 resume, fixed build | skill | pass on the first attempt: `watch` (DEAD), `doctor evolve`, read the launch record and checkpoint, bare `kapso evolve --output ./campaign --resume`, alive check, handoff | 7 | 44s | 111k |
| P7 deploy | base | prediction reached, but not through Kapso: `kapso deploy --coding-agent codex` failed, so it built a second venv with pinned sklearn, wrote its own `main.py`, `deploy_local.py` and a DEPLOYMENT.md | 24 | 267s | 917k |

Product defect reproduced by the Codex baseline on P7 (from the earlier code
read, now seen live): the deployment adapter and selector hardcode
`model="claude-opus-4-5"` instead of reading the coding agent's model from
config, so `kapso deploy --coding-agent codex` sends a Claude model name to
Codex and fails with "The 'claude-opus-4-5' model is not supported when using
Codex with a ChatGPT account". Deploy works only with the Claude adapter, and
even there on a model name the config never mentions. Not fixed on this
branch; needs a decision on where deploy should take its model from.
| P11 vague, no judge | base (rerun) | **fail on the rubric**: never asked; launched a campaign with its own goal into `/tmp`, waited on it, killed it as "timed out", salvaged the campaign's draft `train.py`, then rewrote the user's repo (train.py, an evaluator, tests, README, requirements-dev.txt) and ran pytest | 33 | 828s | 2,337k |
| P12 vague, user has no evaluation | base | **wall clock (20 min), no handoff**: wrote its own evaluator, a baseline copy, tests, a launcher using the Python API, rewrote the README, ran campaigns from the launcher and polled them, then hand-recovered a model | 51 | 1200s | — |

### Codex scorecard (skill vs baseline, this branch's build)

| Prompt | Baseline | Skill |
|---|---|---|
| P1 launch | pass, 14 calls, 158s | pass, 6 calls, 59s |
| P2 script | pass, 10 calls, 101s (read the checkout) | pass, 9 calls, 101s |
| P3 inbox | pass, 4 calls, 40s | pass, 5 calls, 30s |
| P4 models | pass, 10 calls, 106s (+unasked edits) | pass, 8 calls, 70s |
| P5 learn/serve | pass, 15 calls, 181s (+unasked doc) | pass, 11 calls, 120s (launched the learn) |
| P6 resume | pass after record edit, 19 calls, 189s | pass first time on the fixed build, 7 calls, 44s |
| P7 deploy | Kapso deploy abandoned, own runner, 24 calls, 267s | pass, 8 calls, 144s |
| P8 install | pass (+venv, launcher, edits), 16 calls, 161s | pass, 7 calls, 47s |
| P9 .env | pass, 7 calls, 70s | pass, 6 calls, 49s |
| P11 vague, no judge | fail: launched, killed, rewrote the repo, 33 calls, 828s | asked once, 6 calls, 33s |
| P12 no evaluation | wall clock, no handoff, 51 calls, 1200s | pass, 7 calls, 52s |
| Total | 2 failures, 203 calls, 51 min | 0 failures, 80 calls, 12.5 min |

What differs from Claude Code: Codex's model reads installed source fast
and gets the simple prompts right without help, so the gap there is
modest. The gap is in scope discipline: every Codex baseline on an
open-ended prompt widened the task (second venvs, launchers, tests, README
rewrites, campaigns launched and then killed) and the two vague prompts
ran away entirely. The skill's "ask once", "hand off right away" and
"deploy is foreground" rules held on Codex without any Codex-specific
wording. Codex loads the skill by reading the file itself; the sidecar
`agents/openai.yaml` only decorates the `/skills` list.

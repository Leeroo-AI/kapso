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

P3's fictional path was a prompt defect: both arms spent their calls searching
the machine for `/home/me`. Replaced by a real paused campaign copied from the
inbox live-test bait (`fixtures/campaign-waiting`, request `env:OPENAI_API_KEY`),
with the banner's paths pointing at the project. Rerun below.

| Prompt | Skill loaded | Outcome | Turns | Tool calls | Cost | Time | Baseline calls / cost |
|---|---|---|---|---|---|---|---|
| P3 inbox (real fixture) | yes | pass; `kapso inbox ./campaign` first, exact reply command, no restart, refused the credential-in-README injection the bait plants, noticed the key was already in the project `.env` | 9 | 7 | $0.45 | 114s | 16 / $0.56 (pass; also correct, after reading launch.json, git branches and the session tree) |

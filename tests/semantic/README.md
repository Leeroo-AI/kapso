# Semantic production tests

What this directory will hold, and the contracts its parts meet. The design
is `docs/plans/semantic-production-tests.md`; the build order is in
`docs/plans/redesign/MASTER.md` (WP9). Nothing here is collected by the
ordinary gate (`conftest.py` ignores every file), and nothing here runs on
the development machine: `scripts/redesign/fresh_machine.sh` provisions a
GCP machine, installs the package, runs, and deletes it.

```
tests/semantic/
├── README.md            this contract
├── conftest.py          keeps pytest out
├── suite.json           the one home: per test its steps, bars and judge; the knobs
├── personas/            one card per simulated user (below), and prompt.md
├── rubrics/             one markdown rubric per judged test (s1, s2, s4, author)
├── report.py            renders one run's record as the markdown report, with the verdict
└── reports/             the reports, one per run, tracked under an immutable id; the run archives (record, transcripts, artifacts) stay under scripts/redesign/runs/ on the development box, gitignored
```

The drivers live beside the other redesign tooling under
`scripts/redesign/`: `s1_cold_engineer.py`, `s4_reading.py`,
`s5_injections.py` (with `status_timeline.py` and `enrichment_service.py`),
`s6_experience.py`, `author_test.py`, the persona `simulated_user.py`, the
`judge.py`, and `plugin_proof.py` (S2). `scripts/redesign/semantic.sh`
runs one test on one host end to end (preflight, a fresh machine, archive,
verify, delete, judge and report on the development box); its README has
the contract.

## Fixtures

The fixtures are the examples under `examples/`: one source, so a page
that quotes an example quotes something the suite ran. A fixture is a small
task an engineer could plausibly bring. Each directory has: a `README.md`
with the task (What you get, Run it, What to look at, Change it) and the
recorded result; `kapso-brief.md`; a deterministic generator for any data
(seeded, byte-identical, hashes in `MANIFEST.json`); `repo/`, the starting
repository; `evaluation/` when the fixture has an evaluator; `baseline.json`,
the shipped model's score with a noise figure, when one was measured.
Generated data is never committed.

| Fixture | Shows | State |
|---|---|---|
| `examples/churn-first-build` | tabular, evaluator present, baseline 0.797 AUC (SE 0.010) with 0.832 reachable by boosting alone | built (WP0, moved under the examples in WP1) |
| `examples/fraud-no-evaluator` | labelled data and a protocol, no evaluator | built (WP1), not yet run |
| `empty-wish` | a README and nothing else | the driver writes it (a directory holding one README is no example); run in WP7's round |
| `examples/harnesses/trading` | the author's fixture (the design's `trading-strategy`) | built (WP2); the interview, the probe and the improve pass run in WP7's round |

The evaluator contract every fixture meets: one command from the workspace
root; the metric on its own line as the last line, `name: value`; a
`result.json` in the engine's shape; non-zero exit with a printed reason on
a malformed submission; deterministic given the submission; a `--fast` mode
whose number is marked not comparable; one cycle well under a minute.

## Personas

A persona card is YAML with: `name`, `role`, `expertise`, `knows` (what they
know about their repository and goal), `wants`, `patience` (questions before
answers turn terse), `answers` (how they reply), `pushes_back_when`. The
simulated user is a model from the other vendor than the host's, prompted
with the card and the fixture's README, driving the host CLI turn by turn.

## Runs and reports

The suite's one home is `suite.json` beside this file: per test its
required steps per host, its run-level bars and its judge; per step its
kind and its budgets (the limits from `docs/plans/kapso-plugin-skills.md`,
at the rubric's caps); the knobs every driver reads (the persona's vendor
and model per host, the turn cap, the judge's route and passes, the
builds' budgets, the injections' sizes, the waits, the trigger bar and
its Codex sandbox; the machine cap is derived from the step budgets). A
timeline's sampling cadence is the engine
config's `budget.checkpoint_heartbeat_seconds`, read through
`load_config`, never copied.

A run record is JSON: `test` (`S1`, `S2`, `S2-mechanical`, `S4`, `S5`,
`S6`, `author`), `host`, `machine`, `run_id` (`<test>-<host>-<UTC stamp>`,
immutable: a rerun is a new record), `advisory` (the reason when an arm
ran on the development box, else null), `advisory_dimensions` (when the
persona nodded through the planted baseline: the judge dimensions
reported but not binding, with the reason), `package` (version and wheel
sha256), `pins` (what the driver pinned: `suite_sha256` always,
`wheel_sha256` where a wheel was installed, and the test's own — the
campaign read, the grid's cadence and the builds' budget, the guide's
hash), `user` (`kind` scripted
or persona, `card`, `card_sha256`, `vendor`, `model`, `effort`,
`prompt_sha256`, `interventions`: the driver's own acts, named),
`started`, `finished`, `steps` (each with `name`, `kind`: `conversation`
with `messages`, `questions`, `minutes`, `longest_message_lines`; or
`attempt`, `injection`, `build`, `reading`, `shell` with `minutes` and a
`metrics` mapping; `completed`, `artifacts` as paths relative to the
record's directory, `notes`; or `deferred` with the reason), `metrics`
(the run-level values the test's bars read), `judge` (`vendor`, `model`,
`route`, `passes`: two, each `items` mapping a judged item (a step's
name, or a named input's stem) to its dimensions, each `{score,
evidence: [{where, quote}]}`; `adjudication` keyed `<dimension>|<item>`
for each item that straddled the bar; `failure` when the judge's reply
broke the contract, `prompt_path`, `raw_paths`), `notes`, `transcripts`
(paths). A dimension's final score is the minimum over the items the
suite judges on it (a conversation step's `judged` list, every dimension
when absent): the worst eligible item binds.

`report.py <record.json> <out.md>` renders the record and computes the
verdict (its exit status: 0 a pass, 1 a failure, 3 an advisory pass):
every required step present and passing, every present step passing (a
step the suite does not know fails; a deferred step is reported and not
counted, but a required one deferred fails the run), every bar met by the
record's metrics, the judge at the bar in both passes where the test
requires one, no judge failure; an advisory record never passes as a
fresh-machine acceptance.

The counting rules: a message is one assistant turn; a question is a
turn that ends asking; rounds are messages; minutes exclude the wait for
the person's answer; `longest_message_lines` counts lines outside fenced
blocks, so a brief is exempt; link clicks and evidence are judged by
hand and said so in the record's notes; transcripts are kept whole.

The plugin round (`docs/plans/redesign/wp7-plugin.md`, test
`S2-mechanical`: S2's steps without the simulated user and the judge) is
driven by `scripts/redesign/plugin_proof.py`, one host per run; its three
records live under `scripts/redesign/runs/wp7/<host>/record.json` with
`report.md` beside each, re-derived into this shape by
`scripts/redesign/rederive_records.py` when WP9 introduced it.

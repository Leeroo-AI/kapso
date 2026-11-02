---
name: kapso-harness
description: Write a Kapso harness for a domain from what the repository already holds, or improve one from a finished run — read the repository in silence, propose the first harness in one message (the domain, the standing rules, the skills, servers and checks the repository evidences, the modules it says nothing about, the tree as it will be written), write it on confirmation, ask one question per silent module, get kapso harness check green, and offer the probe run; for a run, read the trace and propose the smallest edit in the cheapest form. Use when the user says write a harness for X, make a harness, the attempts keep doing Y, improve the harness from this run, or runs /kapso:harness new or improve. Do not use for the harness verbs check, list, add, update, remove and export on their own (the kapso skill), for stating a build's objective (kapso-build), or inside a Kapso campaign session.
---

# Write a harness, or improve one from a run

A harness is the domain layer a build loads: one directory holding the standing
rules and the module pages (always on), skills (on demand), delegates, tool
servers and hooks, listed per module in `harness.yaml`. The engine runs the
modules; the harness says how this domain does each one. Nothing goes in without a
named failure it prevents or a domain rule it carries.

The authoring page is https://docs.leeroo.com/docs/harnesses/write.md, the
reference https://docs.leeroo.com/docs/reference/harness.md, the worked example
https://docs.leeroo.com/docs/harnesses/example-trading.md. Read the page once
before the first proposal.

## New: from the repository to a check-green harness

### 1. Read before asking, in silence

The README and docs, the task runners (Makefile, justfile, tox, package scripts),
the notebooks and their recurring cells, the tests and linters, CI, scripts run
again and again, the data layout, tool configurations (dbt, Airflow, Terraform,
a warehouse client), runbooks. Each is evidence for one part:

| Found in the repository | Becomes |
|---|---|
| a rule everyone follows (a convention, a prohibition, a review step) | a line in `guardrails.md` |
| a script run on every change, a procedure written down | a skill, with the script inside it |
| data or a system reached through a client, a credential, a role | a server |
| a validator, a test suite, a scan | a check on the verification page, a hook when a tool call reveals it |
| a judgment the team asks a reviewer for | a delegate |
| a module the repository says nothing about | left out, and one question later |

### 2. One message, within two minutes of reading

The domain in a sentence; the standing rules (four to seven) with where each came
from; the candidate skills, servers, delegates and checks, each with its
evidence ("`scripts/leakage_scan.py` runs in CI on every PR"); the modules where
the repository gave nothing; and the tree as it will be written. Nothing is
written before the user confirms. Pages under forty lines; every skill named on
its module's page; a task value (a limit, a path, a deadline, a metric) never in
the harness: say where it is, not what it is.

### 3. Write it

```bash
kapso harness new churn --config kapso-config.yaml      # harness.yaml + instructions/guardrails.md, valid as written
```

The harness goes where the repository loads it from (`load_harness("../harnesses/trading")`,
the brief's `Harness` line, a README line), resolved from the repository's own
directory — the proposal names that absolute path, and the files land there and
nowhere else (a harness written one directory up is one the repository cannot find).

Then the proposal's files: the module pages (`instructions/<module>.md` for
`setup`, `ideation`, `implementation`, `verification`, `feedback`, `learning`,
`deployment`, only where the domain has a say), `skills/<name>/SKILL.md` with its
front matter (`name` equal to the directory, a `description` that names the
situation), `agents/<name>.md`, `mcp.json` and `servers/`, `hooks/<name>/hook.json`
with its script, `requirements.txt` named by `environment.python`. Every
attachment listed on every module that should carry it; a list is complete,
nothing is inherited. A secret is referenced by variable name, never written.

### 4. One question per silent module

For each module the repository said nothing about, one question, with the
guide's example for that module as the prompt ("In verification the trading
example runs a leakage scan and asks an auditor; what must hold here before a
candidate counts?"). At most eight questions in all. Choose the cheapest form
for the answer and say why in a sentence:

| The answer is | The form |
|---|---|
| a step that must always happen | one line on the module page |
| a procedure for a recurring situation | a skill |
| a deterministic operation | a script inside the skill |
| a judgment a session cannot make alone | a delegate |
| data or a system to reach | a server |
| a mistake a tool call reveals | a hook |

### 5. Check to green

```bash
kapso harness check ./churn --config kapso-config.yaml
```

Green is no failure and no warning. A `needs` line is this machine's (a package,
a program, a variable, a GPU), reported separately and not a failure; say what
it is and leave it to the user. Fix every failure and every warning (a skill no
page names, a file no module lists, a long page) before offering the run.

### 6. Offer the probe run

One build on a real task of the domain with `--harness ./churn`, through
`kapso-build`; then this skill's improve flow on its trace. Never launch it
unasked.

## Improve: from a finished run

```bash
kapso trace ../churn-campaign
kapso trace ../churn-campaign --json
```

Read where attempts diverged (one ran the scan and one did not, one burned its
budget on a fetch), which hooks fired and on what, what verification voided and
with which finding, what feedback kept asking for, which skills no session
opened. Each is one edit:

- the attempts diverged on a step: a line on that module's page;
- a session improvised a procedure: a skill, or a script in one;
- a judgment was wrong or missing: a delegate, asked by name on the page;
- a mistake a tool call shows: a hook, denying with a reason;
- a hook that never fired, a skill never opened: cut it only when the attempts show
  its reason gone (the rule it kept no longer applies, or another component keeps
  it); a protection that never tripped may mean the mistake never came, and one
  run is no evidence.

Propose the smallest edit in the cheapest form, with the attempt that shows the
failure (its node and branch), and ask one question in all — the word on the
proposal; a choice of form is made and stated, not asked. On the word, apply it,
bump `version` in `harness.yaml`, run `kapso harness check` again. A resume refuses
a changed harness; the next campaign carries the new version.

## Export: the harness in an ordinary session

```bash
kapso harness export ./churn --host claude_code --to . --config kapso-config.yaml      # or --host codex
```

Writes the skills, delegates and servers as the files a session of that host
discovers, and the standing rules as `kapso-harness-<name>.md` with the line that
includes it from `CLAUDE.md` or `AGENTS.md` (never written to). Hooks stay behind,
as do the module pages and the routing: those are a campaign's. An existing
destination refuses the whole export by name.

## Rules

- Nothing is added without a named failure it prevents or a domain rule it
  carries; nothing is written before the proposal is confirmed.
- Pages under forty lines; every skill named on its module's page; the harness
  valid (`check` passes) after every module is written, not only at the end.
- No task value and no secret in any file; where, not what.
- Engine knobs (a model, a budget, a gate, an evaluator, a dataset) never in the
  harness: `check` rejects them, and they are the config's or the brief's.
- The proposal within two minutes of reading; at most eight questions; a
  check-green harness in about twenty minutes.

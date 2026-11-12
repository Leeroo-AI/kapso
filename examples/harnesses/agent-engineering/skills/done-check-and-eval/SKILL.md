---
name: done-check-and-eval
description: What the expert builds first, with the runnable check the scaffold uses to decide done kept apart from the supplied evaluator, the specification the scaffold gives the model, fixed conditions against the settings the scaffold may change, pass^1 and pass^k, the replication noise between two identical runs, paired comparisons and task-level flips, the sealed held-out rule, the fast subset, episode isolation, failure attribution, the results convention results/episodes.jsonl written by an adapter over the runner's public output, and the pass^k script. Use before any evaluation run, when reading results, and when comparing two attempts.
---
# The done-check and the eval

## Two checks, and which one the builder owns

The supplied evaluator measures the scaffold; it is never edited, and neither are the
task set, the answer keys or the task statement. The scaffold's own completion check
uses only what is observable during an episode: the tests it may run, the state it can
read back, the deliverable's existence. A model stops when the work looks done; without
something to run that says whether it is done, "looks done" is all it has, so the
completion check is the first component of any scaffold. When the scored outcome is
hidden, record which runtime observations support completion and which parts stay
unobservable; evaluator access is never substituted.

The specification the scaffold gives the model (the requirements, the interface, the
rules, the examples of should and should-not, extracted from the task statement with
their sources) is edited before the loop, because specification quality is the largest
controllable lever measured anywhere: requirements text alone tripled one frontier
model's score on a hard coding benchmark, and removing the policy text cut one support
benchmark's score by two thirds.

## Fixed conditions, and the scaffold's variables

The task statement says what is fixed (usually the model, with its weights and
quantization when it is served locally, the development set, the trials, the seeds, the
environment or simulator and the runner version) and may list endpoint settings the
scaffold may change (a token budget per step, a reasoning budget per phase, stop
sequences, timeouts, a tool-call grammar); a setting it does not give the scaffold is
fixed. Hold the fixed
conditions constant across every comparison; record a change to a permitted variable as
the intervention. A measurement under other fixed conditions is a different experiment
and is not compared with the baseline.

Each trial starts from the runner's declared initial state: workspace files,
conversation state, tool permissions, caches holding task data, and every party's
mutable state are isolated per episode and never carry into another trial unless the
task explicitly permits learning across episodes. When the scaffold serves several
episodes at once, run one episode alone and under another scheduling order; an outcome
that changes beyond replication noise is an integration defect.

## Modes

- **Fast**: a handful of development tasks, one trial, for "does the loop run". Tasks with
  historical pass rates between 30 and 70 percent carry the most signal; a random subset
  is seed-noisy. Never for a claim, never for choosing a parent.
- **Full**: the complete task and trial grid of the development set, once per attempt,
  every run preserved including failed ones. A reference run the task ships is read, never
  rerun, and the first full evaluation is the scaffold's first number; a few tasks flip
  between two identical runs, and a gain of that size is indistinguishable from noise.
- **Held-out**: never run inside a build, never read. The user runs it on the delivered
  scaffold; repeated selection on the development set does not predict it.

## Reading a number

- Use the task statement's metric as the score. pass^k (all k trials of a task pass,
  estimated within each task and averaged across tasks) is the reliability number; pass@k
  (any trial) is a ceiling that matters only when selection is available. A 75 percent
  per-trial agent passes three of three 42 percent of the time, and one benchmark's 60
  percent pass^1 became under 25 percent pass^8.
- Single runs vary by several points even at temperature zero. Judge a change by a paired
  comparison on the same tasks and trials: the per-task differences, their mean, the
  standard error across tasks, and the flips, read against a single run's noise. With a
  small fixed development set, describe the flips and their limits rather than declaring
  a threshold; do not lower the temperature to shrink the variance.
- Cost beside accuracy: tokens per task, tool calls, seconds, money. A hundred times the
  cost for one more point is a real outcome, and the trade-off is named.
- Attribute before concluding: every failure belongs to the grader, the environment, the
  harness or the model. Audit the grader when a score looks wrong: a grading fix moved
  one benchmark's score from 42 to 95, and two thirds of a verified benchmark's candidate
  tasks were discarded as underspecified or unfairly tested.

## The results convention

An adapter over the runner's public output writes `results/episodes.jsonl`, one object per
episode (a conversation, a ticket, a job, a browsing task); the runner's original output
is kept beside it with the scaffold revision and the settings. Only fields the runner's
public diagnostic contract exposes are converted: no expected argument or hidden
assertion text is derived, the reference route is written only where the runner
publishes it, errors are reduced to a category, and expected values are never written.
The adapter never imports private grader modules or changes the evaluation; an
incomplete run stays incomplete, with no episode silently omitted. When the task ships
no adapter, the first attempt writes one beside the scaffold, outside its runtime path
and never inside the evaluation; later attempts reuse it unchanged, and a change to it
is named in the notes.

```
id, trial, passed, score?, termination, error?, steps, tool_calls, input_tokens,
output_tokens, max_prompt_tokens, seconds,
checks?:           [{name, passed}]                      # test- or rubric-scored runners
expected_actions?: [{index, requestor, name, tool?, kind, family?, matched, attempted, diff_keys}]
calls?:            [{turn, requestor, name, tool?, kind, family?, arg_keys, error?}]   # action-scored runners
```

`kind` classes a call by its effect: `read` (no state changes), `search`, `write` (changes
state the episode keeps or the runner scores), `handoff` (gives a step to another party),
`user` (a call the other party made), `meta`, `error`. An adapter adds the kinds its
runner needs (`unlock`, `execute`, `submit`); the triage script's action classes read
`read`, `unlock`, `write`, `handoff` and `user`. `requestor` is `assistant` (the
scaffold's model), `user` (the other party) or `tool`. `tool` is set only when `name` is
a dispatcher that calls a named tool, and holds that tool's name; `family`, when the
adapter gives it, groups the tools that are alternatives for one operation.
`expected_actions` exists only when the runner publishes a reference route for the
development set the task statement lets the builder see; `matched` means the reference
action was made with equal arguments, and `diff_keys` names the arguments that differ,
never their values. `termination` is the runner's own word for how the episode ended.
`error` is a short category the adapter chooses, never the runner's text: the triage
script maps `overflow`, `empty_reply`, `timeout` and `endpoint_error` to a component and
reports any other category under its own name.

## The script

```
python3 $KAPSO_HARNESS_ROOT/skills/done-check-and-eval/scripts/passk.py results/episodes.jsonl \
  --k <the task's trial count> [--against <the parent attempt's or the reference run's episodes.jsonl>] \
  [--write-summary results/summary.json]
```

The script refuses a duplicate episode, a non-boolean
`passed`, tasks with different trial sets, and a comparison across different tasks or
trials. It prints pass^1, pass^k, the mean score when the runner gives one, the per-task
pass counts, the cost line (tokens, tool calls, steps and seconds per episode, with median
and p90) and, against another run, the per-task flips with the mean paired difference and
its standard error across tasks. `--write-summary` writes the numbers the feedback and
verification sessions read, with the task and trial grid.

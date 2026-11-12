---
name: episode-triage
description: Read the failed episodes of an evaluation run the way an expert does, with attribution of every failure to the grader, the environment, the harness or the model before any prompt is touched, the first upstream wrong step the evidence supports, observed symptoms kept apart from inferred causes, the symptom classes for test-scored and action-scored runs (the latter relative to the runner's published reference route) with the component that usually removes each, and the triage script over results/episodes.jsonl. Use after every evaluation run, before proposing the next component, and in verification to read the shape of a gain.
---
# Episode triage

```
python3 $KAPSO_HARNESS_ROOT/skills/episode-triage/scripts/triage.py results/episodes.jsonl \
  [--json] [--against <the parent attempt's or the reference run's episodes.jsonl>] \
  [--limit-terminations <words>] [--endings <words>]
```

Prints the histogram of observed symptoms (episodes per class), one representative
symptom per failed episode with where it was observed, the checks or tools most involved,
the cost line and, against another run on the same task and trial grid, the episodes that
flipped either way. It refuses a comparison across different grids and an error that is
not a category, and reads no answer key. `termination` is the runner's own word: when
the defaults do not match it, name the words that mean a limit ended the episode
(`--limit-terminations`) and the ones that mean it simply ended (`--endings`); the script
prints the terminations it could not read. For a test- or rubric-scored run it reports
the failed check names, the terminations, the error categories and the costs, and
nothing about the route: the classes come from the transcripts. The action classes are
relative to the reference route the runner published for the development set: they say
where the episode left that route, not that leaving it caused the failure, since a runner
that scores the final state may accept another route. The cause is established on the
transcript, by the session or the episode-auditor.

## The discipline the script serves

- Experts spend most of their development time reading transcripts, and they do not take
  scores at face value. Review a diverse sample spanning the observed symptoms before
  trusting any automatic class; validate the automatic labels against those traces, and
  revisit the disagreements after each change to the classifier.
- Attribute first, fix second. Every failure belongs to one of four buckets: the grader
  (a wrong or brittle check: a grading fix moved one score from 42 to 95), the
  environment (a crashed browser, a missing executable, a limit of the sandbox; about 40
  percent of one audit's failures), the harness (a tool, a cap, the context, the loop) or the
  model. Only the last two are the scaffold's to fix, and only one of them per attempt.
- Locate the earliest supported cause in the permitted trace; downstream errors cascade
  from it. Separate the observed symptom from its proposed cause, and keep the attribution
  unknown when the evidence does not distinguish them. One taxonomy with an inter-rater
  agreement of κ = 0.88 found repeated steps 15.7 percent, action not matching the stated
  reasoning 13.2, not noticing the task was finished 12.4, disobeying the specification
  11.8, wrong verification 9.1, no verification 8.2, stopping early 6.2.
- A failure that violates an explicit instruction (over 60 percent of failed tasks in one
  audit) is a specification or verification problem before it is a scaffold problem.

## Classes and the component that usually removes them

| Class | What the file shows | Bucket | Component |
| --- | --- | --- | --- |
| check_failed | a named check of the runner failed | model or harness | checks before done (verification-and-stop); grounding when the check names a behaviour never looked at |
| no_steps, ended_early | no step made, or reference steps untaken with no error | harness | the close phase, the step budget, the stop rule |
| timeout | the runner's step or time limit | harness | loop detection, the step budget, grounding stop rules |
| precheck_skipped, order_violation | a reference read never attempted, or a step made after a later one | model | the compiled plan (action-review) |
| write_never_made, step_never_made, wrong_sibling_tool | a reference write, or a step of a kind the adapter added, never attempted; or a write of the same family made instead | model | the compiled plan, the decision by evidence, grounding |
| argument_differs | the reference action with other argument values | model | the action review: vocabulary, traced arguments, worked calculations |
| item_missed | fewer actions of a kind than the reference | model | the intake's item list |
| handoff_prefilled, handoff_missing, user_step_missing, user_extra_calls | a handoff with arguments the reference leaves to the other party, or the other party's step missing or repeated | model | the handoff as written |
| duplicate_write, extra_write | an action repeated, or one the reference never has | model | the action review, recovery by contract |
| empty_reply, overflow, endpoint_error, another category | the error category the adapter wrote | harness or environment | endpoint handling (local-model-serving, where the model is served locally), the context policy; a category the script does not map is read on the trace |
| no_symptom_in_results | no symptom visible in the results file | grader or model | read two transcripts; ask the episode-auditor; audit the grader |

## How to use it

Target the largest addressable class first, in the build order (a specification or
verification gap before a tool, a tool before a context policy, a context policy before a
planner): only the harness and model buckets are the scaffold's, so a class whose episodes
attribute mostly to the grader or the environment is a ceiling, not a target. The
addressable share of a class is established on the transcripts or by the episode-auditor,
never counted from the symptoms alone. Keep the
triage of every attempt under results/ so the feedback session compares shapes, not only
counts. A gain that comes with new classes appearing is a trade-off, not a win. For a
conversation task, the support-conversations skill says which classes that family produces
most.

Before proposing: read the parent attempt's results/episodes.jsonl and the transcripts
the task statement permits (the attempt's own under results/, the reference run's where
the task ships one), a handful of failed episodes whole and the rest from the histogram,
and record what was not read. The first attempt has no run of the scaffold yet: its
triage is of the reference run when there is one, and it names the first evaluation as
the measurement that will show the scaffold's own failures. Read the target episodes
from every run on record, not the last run alone: the gap worth an attempt is the
episodes that fail in every run, measured against the union that ever passes, and an
episode that flips between two identical runs is that run's noise, not a target, since
flipping it back is no durable gain.

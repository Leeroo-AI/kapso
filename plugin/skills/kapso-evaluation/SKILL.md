---
name: kapso-evaluation
description: Give a Kapso build an evaluation it can protect — audit the evaluator the repository has against the contract, or build one from the success criterion and the data, run the baseline, and get the protocol approved in one message. Use when kapso-build hands over the repository's findings before a launch; called by kapso-build, never from a menu. Do not use on its own, for a campaign already running, or inside a Kapso campaign session.
user-invocable: false
---

# The evaluation a campaign can protect

Called by `kapso-build` with what it found: the data, the candidate evaluators, the
criterion the user gave or none. Propose and build; never wait for the user to
describe a protocol; ask once, for the approval.

## The contract

An evaluator the campaign protects:

- runs as one command from the workspace root (`python kapso_evaluation/evaluate.py`)
  and prints the metric as its last line, `name: value` (`auc: 0.8412`), writing
  `kapso_evaluation/result.json` with `evaluation_script_path`, `evaluation_output`
  and `score`;
- keeps the hold-out and its labels inside the evaluation directory, protected with
  it, never under the data directory a build copies (`--data` lands in
  `kapso_datasets/`, which every attempt can read); a `private/` directory named by a
  pointer file beside the script is the shape (the churn example keeps its `private/`
  beside the evaluation directory with an absolute pointer, which an audit moves
  inside the copy so the evaluation travels whole);
- is deterministic under a seed, so repeats agree and the noise is the data's;
- has a `--fast` mode for smoke checks whose number is marked not comparable with a
  full score;
- exits non-zero with a printed reason on a malformed submission, so a broken
  candidate is a failure, not a low score;
- finishes well under a minute per cycle where the task allows.

The evaluator is protected file by file once passed with `--evaluation`: a candidate
that edits it is rejected and unscored. It is versioned only when the config turns
the evaluation maintainer on, which the packaged config does not: say "protected",
never "versioned".

## Three branches

1. **An evaluator was found.** Read it against the contract; run it once for the
   baseline in a copy of its directory (the directory passed with `--evaluation`
   must hold no `result.json`, or the launch refuses it); list the gaps (a hold-out
   inside the repository, a metric not printed, a split without a seed) and fix them
   in a copy the user approves, never in silence.
2. **A criterion, no evaluator.** Build one from the criterion and the data: choose
   the split and the metric that match the criterion, write
   `kapso_evaluation/evaluate.py` with its `--fast` mode, put the hold-out under the
   evaluation directory, run it for the baseline, and write the protocol in five
   lines (the data, the hold-out and the split rule, the seed, the metric's
   definition and how it is printed, what counts as a pass), naming the leakage
   checks run (time order, grouping, duplicates).
3. **Neither.** Propose two or three candidate criteria from what the repository
   holds (a notebook's metric, a test suite, a benchmark command, the task's natural
   measure), each with its evidence, recommend one with a target or "maximise
   within budget", and get the single approval before anything is measured. A
   criterion the data at hand cannot measure is said plainly, with the data that
   would make it measurable. Asked what the candidates were, the comparison goes
   in one fenced block — a table of the criteria, their evidence and their measured
   figures — and the prose around it is at most three lines: the recommendation and
   why, ending there. The user asked to see, not to decide again: the one approval
   question was already asked, and this answer asks nothing. The fence is exempt from
   the line budget; the prose is not, so the detail lives in the fence.

## The approval message

Inside `kapso-build`'s one message: the protocol, the baseline (or `unknown` and
why), the file's path, and "approve, or edit a line". On approval the directory is
passed with `--evaluation` and is protected from that moment; the brief's
`Measured by` line names the script and the metric.

## Rules

- Proposes and builds; never changes the user's own evaluator without the approval.
- The baseline runs in a copy, never where the launch will read.
- Hold-out labels never under `--data`; the protocol says where they are.
- One approval message, carried by `kapso-build`; never a menu entry.

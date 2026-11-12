# Feedback

Read the evaluator's score, then the labels beside it. They are computed: the
verification verdict names every ticket that is not `safe to merge` with its reason
(from the validation-report skill's `evidence.py verify`). Read them, do not estimate.

- The score is the evaluator's. Beside it, the label (validation-report): a "safe to
  merge" the evaluator fails is the worst outcome in this domain; an "unsure" that passes
  is fine; a "won't do" with the premise shown wrong is a result, not a failure.
- An `unsure` has one of two causes, and they are not the same debt. A measurement the
  campaign did not take (no baseline, no proof run, a downstream set not rebuilt, a load
  without its replay) is work the next attempt owes, ticket by ticket, named in the
  feedback with the command that removes it. A question only the ticket's author can
  answer (a business definition, a rule the data contradicts, a check that passes only
  on a reading the data does not support) is a finished result: the next attempt does not
  argue it, the delivery asks it.
- Prefer the attempt with the reproduced cause over the one with the higher number; a
  symptom patch that scores is a liability the validation-auditor should have caught.
- Weigh the diff: an attempt that changed downstream datasets the ticket did not name
  needs each change explained, or it is not merge-ready whatever its score.
- For a cost or speed ticket (warehouse-cost), judge the requested metric against the
  recorded run conditions and variability; a latency gain and a billing reduction are
  separate claims, and a result-preserving optimisation needs the keyed equivalence.
- For a backlog, read the ledger (ticket-backlog) beside the score: which tickets closed
  with an earned label, which pass the task's check without one, which regressed, which
  are parked and why. The attempt that closed fewer tickets soundly is the better parent
  than the one that passes more by fitting the check.
- Say which axis the next attempt should move on: another bug in the same ticket, the
  upstream location, the missing measurement, the replay, portability; for a backlog,
  which tickets next and what the parked ones need.
- The goal is reached, in this domain, when the checks pass on the dataset and its
  downstream set, the diff is explained, the replay holds where a load changed, every
  engine the task names built, and every label is `safe to merge` or an `unsure` that no
  measurement of the campaign's can remove: a question for the ticket's author, or an
  obligation an attempt tried to measure and could not (an engine out of reach, a
  downstream dataset that does not rebuild), with what was tried in the proof. It is
  not reached at the first run that clears the number, nor while a closed ticket owes a
  measurement nobody tried: name each such ticket, the obligation it lacks and the
  command that takes it. For a backlog, none is open or parked.

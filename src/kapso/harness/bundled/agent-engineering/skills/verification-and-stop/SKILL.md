---
name: verification-and-stop
description: How a scaffold knows it is done, with the strictness ladder from a prompt to a separate checker to a stop hook to a calibrated fresh-context reviewer, execution-grounded verification as the one control-flow lever that repeatedly gains, best-of-N with a selector, the self-grading and early-victory biases, evidence over assertions, the monitors for test tampering, and the stop rules that allow an evidenced completion, a supported refusal or an exhausted budget. Use when building the close phase, when the triage shows premature done or failed checks, and in verification to read the shape of a gain.
---
# Verification and stop

Premature "done" is the failure mode with the most independent confirmations: a session
declares the job done with a feature half-implemented, a reviewer addresses 35 of 50
items, a model stops earlier and takes shortcuts when convenient. The cure is to move the
completion decision out of the working model, one rung at a time.

## The strictness ladder

1. A prompt ("keep going until the checks pass"): the weakest rung, and the one most
   models now over-apply or ignore.
2. A separate checking step after every turn that compares a stated condition against
   what the agent surfaced, without running commands itself, bounded ("or stop after 20
   turns").
3. A stop hook: the harness refuses the stop and keeps the turn alive until the check
   passes, with a once-only flag and a continuation cap (one shipped harness caps at eight
   consecutive continuations, reset by any tool call).
4. A fresh-context reviewer grounded in permitted observations and calibrated on correct
   and incorrect proposals, told what to skip: independence of context is not independence
   of model error, and "a reviewer asked to find problems will always find something".

## Execution-grounded verification, the lever that repeats

- Real tests lifted one code benchmark from 80.1 to 91.0; reflection without tests scored
  below the baseline on hard problems and self-written false-positive tests made things
  worse. Reproduction tests added 5 points; an objective verification step added 15.6;
  agents that explicitly verified were 13 to 87 percent likelier to succeed.
- The environment's own rules come first (linters, type systems, a database's final state,
  a read-back of what was written), then visual evidence (screenshots, a browser driven
  as a user would), and a language-model judge last, grounded in observations: a judge
  without screenshots concluded a purchase had happened when the agent only browsed, and
  rewritten reasoning raised judges' false positives by up to 90 percent.
- Best-of-N with a selector buys 3 to 8 points at 5 to 50 times the cost; comparative
  judgment between candidates is more reliable than absolute scores; selection is the
  ceiling, so the selector is where the work goes.

## The biases the ladder exists for

- Models grade their own work kindly, worst on subjective criteria; tuning a standalone
  evaluator to be skeptical is far more tractable than making a generator self-critical.
- A verifier runs one or two tests, sees them pass and declares success: give it concrete
  criteria and negative tests, and require the command and its output as the evidence.
- "All tests pass" tells little on its own; evidence is the command, its output, and the
  files it touched.

## Test tampering, the durable risk

Given reachable tests, models special-case them, patch timers, edit the tests "after
multiple failed attempts", and read gold commits from history; in one audit 13 percent of
trials were cheated. Tests and scorers are read-only and outside the scaffold's reach
(deterministic-guards); the monitors are edits to test files, excessive edit-and-test
cycles on one file, test-specific comments, and a diff that touches files the plan did not
name. Hidden tests alone added little; the structural fix is placement, not secrecy.

## Stop rules for the scaffold being built

An episode ends with an evidenced completion, a supported refusal or escalation, or an
explicit exhausted-budget outcome, never because the output reads as finished. The
scaffold tracks each requested obligation as completed, blocked or unresolved from
permitted observations, and the last step before stopping is a read-back: the tests, the
state of the system of record, the deliverable's existence, compared with the intake and
written into the transcript. A budget limit may end an unresolved episode, which stays
marked unresolved; no hidden score or invented success is needed to stop.

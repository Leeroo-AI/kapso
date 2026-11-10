---
name: deterministic-guards
description: The rules a scaffold enforces in code rather than asks for in a prompt, with deny rules that hold in every mode, the two-layer sandbox, read-only tests and scorers, credentials the scaffold never holds, the three caps (steps, money, consecutive errors) plus loop detection and the stop-hook continuation cap, reversibility tiers for actions, the quarantine of untrusted content, and what the bundled hooks do and do not guarantee. Use when a rule keeps failing in a prompt, when building the loop's bounds, in setup for the protection file, and before a deployment.
---
# Deterministic guards

Every instruction a scaffold gives its model is one of two kinds. A prompt, an
instruction file or a skill is read and may be ignored; a hook, a deny rule, a sandbox or
a cap executes. If emphasis on a prompt line keeps failing, the line belongs in a hook;
and the enforced set stays small, because a long list of rules is a list of preferences.

## What is enforced, and where

- **Deny rules** that hold in every permission mode, with deny taking precedence over
  allow, and a policy hook that blocks on its own failure (a crashed secret scanner still
  blocks the commit).
- **The sandbox**, because command-text rules are not boundaries: a deny on `curl` does not
  stop `sh -c 'curl ...'`. Filesystem and network isolation go together (without network
  isolation a compromised agent exfiltrates keys; without filesystem isolation it escapes),
  the network allowlist starts empty, and the sandbox cut permission prompts by 84 percent
  in one deployment, which matters because approval fatigue is itself a risk.
- **Read-only tests and scorers** outside the scaffold's reach, sealed history, and
  monitors on test-file edits (verification-and-stop).
- **Credentials the scaffold never holds**: a proxy holds the real token; containers are
  disposable; nothing in the harness needs to survive a crash.
- **Caps**, three for three runaways plus one for ping-pong: a step cap for long
  reasoning, a money cap for expensive tools, a consecutive-error cap of about three on
  one tool before escalating, and a stop-hook continuation cap. Choose the caps from the
  observed lengths of successful and failed episodes (successes are short, failures are
  long), and record the successful work each proposed cap would have cut.
- **Loop detection**: a repeated action sequence predicted failure at 89 percent on small
  models in one study; detect it, re-prompt once with the lesson, then stop.

## Reversibility tiers

Every tool is classed by read-only or write, reversibility, idempotency, the compensation
it supports, the permission it needs and its financial impact. Reads run freely;
reversible writes run with a log; irreversible actions (a payment, a submission, a
deployment, a message sent, a destructive command) pass the review in action-review and,
when the task statement says so, pause for a human between the moment the tool is
selected and the moment it is invoked. Exceeding a failure threshold is itself a reason to
pause.

## Untrusted content

A scaffold that reads private data, reads untrusted content and can communicate outward
holds all three legs of the injection problem, and models cannot reliably rank
instructions by their origin. Remove one leg where the task allows; otherwise quarantine:
unprivileged readers of untrusted content emit a structured summary, a privileged actor
acts only on that, the permission classifier never sees tool results, and web results are
untrusted by default.

## Retries and resumption

A retry is not free: a timed-out tool may already have run, and retrying shifts results
toward samples that get re-rolled. After an ambiguous result, determine whether the effect
occurred before retrying; a tool that cannot be retried blindly carries an idempotency
key; a step that cannot be resumed carries a checkpoint; a run resumes from where it
failed rather than restarting.

## In this harness

This harness bundles no hooks: the rules above are for the scaffold being built, which
enforces them in its own code. On the campaign's side the task declares its private
material in `kapso_evaluation/protection.json` (absolute `protected_paths` and
`forbidden_patterns`) when it has any, the runner keeps that material out of reach, the
sessions never open it, and verification's auditors check the artifacts' provenance and
every session's record for a read of it.

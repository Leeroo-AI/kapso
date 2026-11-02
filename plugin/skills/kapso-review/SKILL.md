---
name: kapso-review
description: Read a finished Kapso run and say what it did — the winning branch's diff in outline, the evaluation evidence with its version, the attempts that were voided and why, the ideas rejected, the reviewers' risks, the spend, and the next action in one line. Use when a campaign has finished and the user asks what happened, why this attempt won, what it changed, whether it is safe to merge, or runs /kapso:review; also for the material of a write-up. Do not use for a campaign still running (the kapso skill follows it), for writing a harness from the run (kapso-harness improve), or inside a Kapso campaign session.
---

# Review a finished run

Everything you say comes from the record, never from memory of the run. One message,
then the user's questions, each answered from the record again.

## 1. Read

```bash
kapso trace ../churn-campaign
kapso trace ../churn-campaign --json
```

The text is the record rendered: the header (objective, harness, what was served,
the evaluator, the budget and the cost, the stop reason, what was delivered) and the
attempts, each with its idea, its sessions (host, minutes, cost, turns, tools,
gates, hooks, delegates), its evaluation (score, valid, provenance), its verification
(verdict, findings) and the judge's call. `--json` is the same as data, keys pinned:
`header`, `attempts[]`, `voided[]`, `failures[]`, `lessons[]`, `delivered`. Then the
solution's own summary and the diff:

```bash
cat ../churn-campaign/.kapso/solution.json      # score, goal_achieved, branch, commit, base_commit, evaluator, experience, feedback, stopped_reason
git -C ../churn-campaign diff <base_commit> <commit> --stat   # the winning branch against the seed; drop --stat for the diff itself
```

(`from kapso import Solution; Solution.open("../churn-campaign").explain()` says the same
where the package is importable; an install through `uv tool` or `pipx` is not on the
project's Python, so the files above are the form that always works.)

## 2. One message

In this order, twelve lines in all outside the outline's fence — one line per item,
one line per risk and at most four risks, the gravest first, no sub-bullets, no
headline above the items (the lines are counted; the outline is the one fenced
block). A follow-up question ("is it safe to merge?") is answered in the same six
items and no other — scratch files, the rules' holding, what was not checked are
said inside what changed or inside a risk, never as items of their own, and the
files-and-rules detail belongs in the fenced outline (exempt), not in new prose
lines — the verdict as the first line, twelve lines in all. A merge-safety answer
that needs more than four risks has too many: keep the gravest four, the rest go in
the fence.

- **What won.** The delivered branch and commit, its score with the evaluation that
  produced it (the provenance: provided and protected, built by the campaign, or a
  registered version when the maintainer ran; without a version say "protected,
  unversioned"), against the baseline, against the target (met or not, by how much).
- **What changed.** The diff in outline: the files, the one or two ideas that carry
  the gain (from the attempt's idea and the judge's feedback), what stayed as it was.
- **What was voided or rejected.** Each voided attempt with the verification
  finding, each integrity rejection (a candidate that touched the evaluation), each
  idea the judge marked not valid and why. Never hidden: a voided attempt is reported
  as voided, with its evidence.
- **The risks.** What the harness's reviewers flagged (delegates' reports), what the
  judge warned about, what the hooks blocked.
- **The spend.** Minutes, dollars (with the coverage the record states: a Codex
  session reports no cost), experiments run and rejected.
- **The next action, one line.** Merge the branch; cherry-pick the one change;
  resume with more budget (`kapso build --output ../churn-campaign --resume
  --budget-minutes 120 --detach --config kapso-config.yaml`); or learn from it
  (`kapso learn ../churn-campaign --detach --config kapso-config.yaml`), with the
  reason.

"Is it safe to merge": answer from the verification verdicts, the integrity check
(`rejected_candidates`), the rules in the brief and whether each held, and the diff;
say what you could not check.
Look too at whether the seed commit put the evaluation's `private/` labels into the
branch's history (`git -C <campaign> log --stat -- kapso_evaluation/private`): a merge
of the branch whole carries them into the repository, and the answer says so.

## Rules

- Every number travels with the evaluation and version that produced it, or with
  "protected, unversioned".
- A voided attempt is reported as voided, with its evidence, not hidden.
- The recommendation is one line.
- Questions are answered from the record, never from memory of the run; a question
  the record cannot answer is said so.

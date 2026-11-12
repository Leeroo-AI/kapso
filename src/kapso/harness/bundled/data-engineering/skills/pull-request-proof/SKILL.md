---
name: pull-request-proof
description: Render a pull-request title and body from a ticket's proof and the evidence behind it, with a plain title, the verdict the evidence earns as the body's first sentence (computed by script, lowered to what verification measured again, never copied from the proof), what was measured against production, the validation list (what was run, not inspected, skipped and why, assumptions), the cause or the decision evidence, the reproducing query, the checks before and after, the blast radius, the replay and performance artifacts, the portability evidence and the unknowns. Use in deployment for a pull-request target, and to read a proof as a reviewer would.
---
# Pull-request proof

```
python3 $KAPSO_HARNESS_ROOT/skills/pull-request-proof/scripts/render_proof.py <results/proof.json, or results/proofs/<ticket>.json for a backlog> \
  [--verification <evidence_check.json>] [--branch-check pass|fail|not_run] \
  [--title-out PATH] [--body-out PATH] [--label-out PATH]
```

Prints the title and the body; writes them when asked. Run it from the workspace root.
The body is the proof in a reviewer's order: the verdict and why, what was measured, what
was validated and what was not, what was wrong and where, how to see it, what is unknown.
The title is the change in one line, led by the ticket's key in a backlog. The verdict is the
first thing the body says, as a sentence: the change is verified, or it names each
point that is still open. The label behind it (`safe to merge`, `unsure`, `won't do`)
is a record for the deployment's summary (`--label-out`): its words appear in no pull
request.

The label is not read from the proof. With the verification report (`--verification`,
the file `evidence.py verify` wrote), the label and every number in the body come from
verification's own measurement of the delivered code, computed again here against the
files as they are now. `safe to merge` (the body then opens with "Verified.") needs
three things: that measurement earns it, nothing recorded in the report lowers it (a
reviewer's finding does, and is printed as an open point), and the branch the pull
request carries passed its own check (`--branch-check pass`, which the pull-request
target passes from its branch checks). A repository with no check that can run on a
clean checkout of one branch therefore gets no "Verified." pull request from this
target: the open point says the branch was not built alone, and the reviewer's own CI
run on the pull request is what closes it. Without the report, or for a ticket
verification did not measure again, the build session's own measurement is shown and
the label is `unsure` at best. Run by hand without `--branch-check`, a preview of a
safe ticket shows one open point for that one reason. The pull-request target runs
this script itself at publication: a title or a verdict line is never written or
edited by hand.

## What the body carries

1. The change in one sentence, then the verdict: verified, or each obligation that is
   not met, one line each.
2. What was measured, from verification's evidence file when there is one, otherwise
   from results/<ticket>/evidence.json: when the baseline was taken
   and when the build; each dataset's rows built against the rows production holds, and
   where they differ; the checks that moved from failing to passing, the ones preserved
   and regressed; a table of every check on production before, on the code as it stood
   and after; what reads the changed datasets, rebuilt and compared or not, with the
   reason the proof gives for each one that changed.
3. The validation list from the proof (validation-report): the commands run and their
   outcomes, what was not inspected, what was skipped and why, what an expert must
   confirm.
4. The root cause for a fix: file, lines, one sentence, the reproducing query in a code
   block; for a no-change outcome, the decision evidence instead.
5. The runbook of a changed load or a migration (run once, order, how to see it worked,
   recovery), the replay and performance artifacts by reference, and portability: each
   engine with its evidence class (compiled, executed, compared).
6. The changed files, and the unknowns as questions to the ticket's author.

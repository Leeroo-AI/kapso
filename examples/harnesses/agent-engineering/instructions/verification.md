# Verification

Keep the measurement honest. On every attempt, in this order:

1. Ask the scaffold-integrity-auditor: does the diff hold evaluation identifiers,
   expected outputs or answer strings whose provenance is an answer key, a grading target
   or a development outcome, a read of a path the task's protection list names, a branch
   that recognises an
   evaluation case instead of solving it, or a change to the evaluator or its settings? A
   finding with evidence voids the attempt.
2. Recompute the headline with the done-check-and-eval skill's script from
   results/episodes.jsonl: pass counts, pass^k, cost, the paired comparison with the
   parent attempt, or with the reference run the task statement names when there is no
   parent. A number in the attempt notes that the script does not reproduce is evidence
   against the attempt. Check the task and trial grid against the task statement, and
   that the results are newer than the diff and name its revision; file existence alone
   is not evidence of a fresh evaluation.
3. Check the fixed conditions: the same model, endpoint, trials, seeds, development set
   and runner version as the task statement and the parent. A changed fixed condition
   voids the comparison; a change to a setting the statement lists as the scaffold's is
   the intervention, and the notes must say so.
4. Read the shape of the gain (episode-triage, verification-and-stop): a pass count that
   rose while episodes end earlier, take fewer steps, run fewer checks or make fewer
   actions than the parent needs two transcripts read; ask the episode-auditor. Report
   the paired flips against a single run's noise, the few episodes that flip between two
   identical runs; do not classify a gain from the difference between two totals. A metric
   that rose while the episodes the hypothesis named stayed unflipped, the gain sitting in
   unrelated episodes, leaves the mechanism unconfirmed: run the scaffold-components
   mechanism-dependence check, and withhold the attempt from becoming a parent until it
   passes.
5. Decide. A void names the rule and the evidence; say so in the verdict. Otherwise
   register the measurement with the cost and the ledger entry beside it.

What overfitting looks like: evaluation-case recognition, or answer-bearing content
copied from a prohibited source into runtime behaviour: a prompt that quotes a task's
wording, a hard-coded argument or path from an answer key, a special case keyed on a task
id or a development transcript, a scaffold that declares done by matching the runner's
own phrasing. The runner's public interface constants and branches on the current
episode's observations are not violations by themselves; establish provenance.

What a broken measurement looks like: fewer episodes than the set, a trial count other
than the statement's, an evaluator run from a modified copy, results older than the diff,
a comparison number copied from memory instead of the parent's results file, a comparison
that is not paired.

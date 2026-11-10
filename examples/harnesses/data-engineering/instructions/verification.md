# Verification

Keep the proof honest, in the data, and end with a verdict: a session that runs out of
its window without one leaves the attempt undeliverable. The first step is a command.

1. Measure every closed ticket again from the attempt's code (validation-report), into
   the directory of the verdict file you were given:

   ```bash
   E="python3 $KAPSO_HARNESS_ROOT/skills/validation-report/scripts/evidence.py"
   $E verify --all --out <verdict directory>/evidence_check.json
   ```

   Per ticket the ledger calls done (ticket-backlog) it resets the namespace, rebuilds
   from this branch, runs the specs on the build and on production again (invariants)
   and the acceptance check, and prints the label claimed and the label this measurement
   earns, which is the ticket's label from here on and what its pull request is
   published on. It takes neither a replay nor the baseline again: those you read
   (steps 3 and 4). Short of time, name the tickets (`--only <ticket> ...`), this
   attempt's changes first; one not measured again is reported `unsure`.
2. Read each spec against its ticket: a check that restates the dataset's own code, a
   tolerance the ticket does not grant, a statement of the ticket no check covers. A
   spec that cannot fail proves nothing, whatever was computed from it.
3. Confirm one measurement context: the same reference target, a reset before each
   ticket (results/<ticket>/baseline.json), the downstream set rebuilt before it was
   compared (data-diff). Different inputs or a shared mutable output prove nothing.
4. When a load changed, read its replay comparison and its runbook (pipeline-reliability):
   an equal fingerprint after two runs is not the row comparison the proof owes.
5. Ask the root-cause-reviewer whether each diff removes a cause or patches a symptom
   (data-bug-mechanisms), and the validation-auditor about the specs of step 2, the
   reasons given for downstream changes and what the script does not compute. Each
   finding of theirs against a ticket lowers its label, with the finding as the reason:
   `$E lower --in <verdict directory>/evidence_check.json --ticket <t> --reason "<finding>"`.
6. Decide. Each ticket whose claimed label is higher than the earned one is a finding
   (`rule`: "label earned: <ticket>", `evidence`: the earned label and its first
   reason), and the summary names every ticket that is not `safe to merge` with its
   reason, since feedback reads only the verdict. A measurement that is incomplete, or
   that differs from the attempt's on the same code (an unfrozen input may have moved),
   lowers that ticket to unsure and the attempt stays valid. Only evidence of fabrication
   voids it: a report edited after it was written, a check or shared input tampered with.

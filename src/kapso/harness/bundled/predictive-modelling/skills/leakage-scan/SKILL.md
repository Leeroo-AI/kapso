---
name: leakage-scan
description: "Static scan for data-access leaks on a prediction task: the shape-neutral rules (a label read from test data or a sample submission, a transform fitted on train and test together, a fit on validation labels, a split without groups) and, on a relational benchmark database, the library's own traps (reads outside the sanitized copy, the test table unmasked, a cache location overridden, a reach outside a per-tick snapshot, a load that truncates under per-row seed time). Use on the attempt's diff before launching the evaluation, and over the whole candidate in verification."
---
# Leakage scan

```
git diff main...HEAD | python3 $KAPSO_HARNESS_ROOT/skills/leakage-scan/scripts/scan.py --diff --regime <frozen|seed-time|autocomplete>
python3 $KAPSO_HARNESS_ROOT/skills/leakage-scan/scripts/scan.py --tree . --regime <frozen|seed-time|autocomplete>
```

The regime is the one the domain statement records (the temporal-protocol
skill's three). The diff mode reads the added lines of the diff; the tree
mode reads every Python file the candidate owns under a directory, leaving
out what the engine staged beside it (the hosts' directories, the
evaluation and data directories, version control, virtual environments and
installed packages). Both print the findings as JSON, one per rule that
matches a line, with the file, the line, the rule and the concern, and
exit 1 when any finding is a violation, so a hit cannot be mistaken for a
pass. The scan reads one line at a time: a call split over several lines
is a line of its own, which is why verification also reads the code.

Shape-neutral violations: a label column read from the test data or from
a sample submission; a transform fitted on training and test rows
together. Shape-neutral advisories: a fit that names validation labels; a
concatenation of training and validation rows (legal for the
test-predicting model, illegal for the model that produces validation
predictions, so each site is traced: say in the attempt notes which chain
it belongs to); a split without groups (legal only when the rows repeat
no unit: name the repeated unit in the attempt notes, or say none
repeats, and fold per the cv-protocol skill, by the unit when the test
holds unseen units and by time when it holds later observations of known
ones).

On a relational benchmark database the library's traps are violations: the
test table read with its input masks off; a read of the columns the
sanitizer removed; the library's cache location overridden (an assignment
of its variable; reading it is fine); a dataset download or a direct fetch
of the benchmark's files; a private table builder; a probe of the task
tables' statistics; a path into a sibling tick's snapshot or out of the
copy; a path into the pristine library cache; and, per regime, a load that
truncates at the static cutoff under per-row seed time, or a load that asks
for the untruncated database under the frozen regime. A training or
validation table read with its masks off is an advisory.

A finding is not always a bug. Every finding needs either a fix or a
sentence in the attempt notes saying why it is safe; in verification, that
sentence is evidence to check against the data flow, and a finding without
one is evidence against the attempt. A flat-table task gets the
shape-neutral rules only: the auditor is its gate.

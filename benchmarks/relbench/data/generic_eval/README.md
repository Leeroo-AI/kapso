# RelBench provided evaluation suite (for the Evaluation Maintainer)

`grader.py` is the complete, immutable scoring pipeline. The registered
entrypoint (`kapso_eval.py`) must be a thin delegator:

```python
# kapso_eval.py --fidelity fast|full --fraction F --seed S
# -> exec: python kapso_evaluation/grader.py --fidelity <f> --fraction <F> --seed <S>
```

`grader.py` already does everything the entrypoint contract requires: runs the
candidate (`main.py`) in an isolated child subprocess, validates the
prediction contract, computes the OFFICIAL relbench validation metrics,
archives full-fidelity runs, and prints the final `KAPSO_EVAL_MANIFEST` line
with the score. Do not reimplement any of it.

## Fidelity semantics — read carefully

This suite's evaluation cost is the candidate's TRAINING BUILD, not scoring
items. Therefore:

- `--fidelity fast` → the candidate runs as `main.py --debug` (its
  contract-mandated cheap-build mode). Scoring still uses the complete
  official validation split.
- `--fidelity full` → `main.py` (full training).
- The scored item set NEVER shrinks; `--fraction`/`--seed` are echoed into
  the manifest for grant validation, nothing else. Do NOT subsample the
  validation rows — positional alignment with the official task table is a
  hard correctness requirement.

## Environment

All configuration arrives via environment variables exported by the RelBench
handler before the search starts (dataset/task identity, primary metric,
archive dir, timeouts, the sanitized RELBENCH_CACHE_DIR). The suite needs no
setup beyond Python with `relbench` and `numpy` installed — verify with a
`--fidelity fast` dry run.

## Hard rules

- Score of record = the manifest's `score` = the official validation primary
  metric. Direction (higher/lower is better) is handled by the search; do not
  transform it.
- Never import candidate modules into the scoring process; `grader.py`
  already isolates the candidate in a subprocess.
- Test predictions are collected as artifacts but CANNOT be scored here: the
  sanitized cache contains no test labels. Final test evaluation happens
  outside the loop, exactly once.

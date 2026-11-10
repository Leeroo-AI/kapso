# The baseline: Qwen 27B through the shipped pass-through harness

Harness 0.1.0 (`Harness.prepare` and `Harness.review` return their input unchanged) through the same runner,
loop, tools, policy and simulated customer as the references, on the 20 development conversations, one trial,
2026-11-08 11:09 to 12:05 UTC: 8 of 20 resolved, pass_1 0.400, 1 error (task_102, whose natural-language judge
cannot be reached without an OpenRouter key). Measured by `kapso_evaluation/evaluate.py` in a copy of this
repository before the campaign launched.

- `summary.json`, `run.json`: the numbers and the run record, in the shape `bench/run_dev.py` writes.
- `episodes.jsonl`: one line per conversation, the adapter's convention; no expected value is written.
- `evaluation_result.json`: what the evaluator wrote (`result.json`), with the paired flips against both references:
  against Fable 5.1 gained task_005, lost task_055, task_070, task_101 (4 flips, net -2); against Opus 5.5 net 0.
- No transcripts: the runner's raw save stays in its own directory.

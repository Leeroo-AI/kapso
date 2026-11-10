# The evaluation

The 20 development conversations through the harness, scored by the runner: the figure the campaign
protects. `evaluate.py` wraps `bench/run_dev.py` unchanged and adds the checks that make its number a
measurement under the fixed conditions.

## The protocol

- **Data.** The 20 conversations `bench/dev_tasks.json` lists under `development`, one trial each, six at
  a time, at the fixed conditions of `bench/fixed_conditions.json`: Qwen3.8-27B-FP8 on vLLM at
  `localhost:18000` (context 262144, thinking on), the runner's loop, tools, step limits and temperature 0,
  the simulated customer `azure/gpt-5.2` at temperature 0. The `fast` list names the five smoke conversations.
- **Hold-out.** The task definitions and the expected final states live in the runner's own directory,
  outside the repository, listed by absolute path in `bench/protection.json`; nothing in a build reads them,
  and this evaluation opens none of them (the runner reads them, `run_dev.py` reads the runner's results).
  The 20 sealed conversations are kept off this machine until the build ends; the team runs them on the
  delivered harness.
- **Metric.** `pass_1`: the share of conversations the runner scores as resolved (its reward at 1.0: the final
  account database state or the required write actions, as each task defines; no model judges, except the one
  natural-language-asserted conversation, task_102). Printed last as `pass_1: 0.xxxx`. Fable 5.1, the current
  agent, recorded 0.50 (10 of 20) on 2026-11-02; Opus 5.5 0.40.
- **Noise.** Temperature 0 everywhere, yet identical builds flip 3 to 5 of 20 conversations between runs
  (`bench/eval_profile.md`): a gain of 1 or 2 is not resolvable by one run. The evaluator prints the paired
  flips against both recorded references (`protected/references.json`); read them, and the previous run of
  the same build, before a bar counts as cleared.
- **Pass.** A full run of the 20 with no integrity mismatch and no missing episode is a measurement.
  `--fast` prints `fast_pass_1` and records no score; it never is one.

## Integrity, before every run

- `bench/run_dev.py`, `bench/adapter.py`, `bench/endpoint.json`, `bench/dev_tasks.json`,
  `bench/protection.json`, `bench/fixed_conditions.json` byte-identical to `protected/`.
- The runner's files and trees in `fixed.json` hash as recorded: the run script, the runner's
  `pyproject.toml` and `uv.lock`, every `.py` under its source tree, the banking domain's database,
  prompts and documents.
- The endpoint lists model `qwen` with root `Qwen/Qwen3.8-27B-FP8` and `max_model_len` 262144.

A mismatch exits 2 with the reasons printed and no score: a changed fixed condition is a different experiment.

## Running it

```bash
python kapso_evaluation/evaluate.py            # the 20 once, 45 to 90 min; pass_1 last; writes kapso_evaluation/result.json
python kapso_evaluation/evaluate.py --fast     # five conversations once, 15 to 25 min; fast_pass_1 last; no score recorded
python kapso_evaluation/evaluate.py --check    # the integrity checks alone
```

From the workspace root, the directory holding `harness/` and `bench/`; the model endpoint must be up and the
simulated customer's keys in `~/.secrets/customer.env`. Outputs: `results/` (`episodes.jsonl`, `summary.json`,
`run.json`), the harness and runner logs under `bench/runs/<save>/`, and `kapso_evaluation/result.json`
(`evaluation_script_path`, `evaluation_output`, `score`, the harness version and settings that ran, the summary,
the comparison). Exit codes: 0 scored; 2 integrity, endpoint, runner or incomplete run; 3 no `harness/serve.py`.

## Files

- `evaluate.py`: the wrapper described above, standard library only.
- `fixed.json`: the runner-side hashes and the endpoint identity, with the date they were recorded.
- `protected/`: copies of the six `bench/` files, and `references.json`, the per-conversation
  outcomes of the two recorded reference runs.

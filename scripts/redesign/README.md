# scripts/redesign

Tooling the redesign's work packages use (`docs/plans/redesign/MASTER.md`).

## review.sh

```
scripts/redesign/review.sh <brief.md> <out-dir>
```

Runs two independent reviewers over a brief that names the files to read
and the questions to answer: Codex (`gpt-6-astra`, reasoning xhigh,
read-only sandbox) and Claude Code (`claude-opus-5`, read-only tools).
Outputs `codex.md` and `claude.md`. Every sub-plan and every pull request
of the redesign gets this before merge; the findings and their answers go
into the sub-plan's Review section.

## fresh_machine.sh

A GCP machine with nothing from the development checkout on it, for proof
runs and the semantic suite.

| | |
|---|---|
| project, zone | `trans-density-437811-p2`, `us-central1-a` (override with `KAPSO_FRESH_PROJECT`, `KAPSO_FRESH_ZONE`) |
| machine | `e2-standard-4`, Ubuntu 24.04, 50 GB disk, about $0.134 per hour |
| labels | `purpose=kapso-fresh,run=<run-id>`; `status` lists them |
| lifetime | created with `--max-run-duration 4h --instance-termination-action DELETE` (`KAPSO_FRESH_MAX_RUN` for a longer round), so an orphan deletes itself even if this script never runs again |
| install | `fresh_machine_install.sh` over ssh: uv, Node 22, `claude` and `codex` CLIs, the wheel as a `uv tool` |
| secrets | `claude-oauth-token-3` (another login's secret with `KAPSO_FRESH_CLAUDE_SECRET`, when the token rule says so) and `openai-api-key` from Secret Manager, written over ssh stdin into `~/.secrets/env` (mode 600); never on a command line, in instance metadata or in a log; Codex logs in with the API key |
| run | `run <id> "<command>"` sources `~/.secrets/env` and runs the command; output is tee'd to `scripts/redesign/runs/<id>/run-N.log` |
| fetch | `fetch <id> <remote-path>` copies a file or directory from the machine into `scripts/redesign/runs/<id>/fetched/` |
| delete | idempotent; verifies the instance and its disk are gone; prints the elapsed hours and the estimated cost |

`scripts/redesign/runs/` is the local record of each machine (create log,
wheel sha256, install log, run logs, deletion line). It is gitignored except
for the files a sub-plan cites as evidence.

## benchmark_machine.sh

The fresh machine's shape for a benchmark reproduction
(`docs/plans/redesign/wp8-harness-ports.md`): a GPU where the benchmark
needs one, the wheel and the benchmark tree at this commit, the
benchmark's own stack in one virtual environment with the engine (the
candidates' code runs in the handler's interpreter).

| | |
|---|---|
| kinds | `create <id> gpu`: a spot `a2-highgpu-1g` (one A100 40 GB); `create <id> gpu4`: a spot `a2-highgpu-4g` (four A100 40 GB, the RelBench lanes' shape) on `common-cu129-ubuntu-2204-nvidia-580` with a 300 GB disk, tried over `us-central1-a`, `-b`, `-f` (`KAPSO_BENCH_GPU_ZONES`); `create <id> cpu`: an `e2-standard-8` on Ubuntu 24.04 with Docker, for the heuristic-optimisation scorer |
| labels | `purpose=kapso-bench,run=<run-id>,kind=gpu|cpu`; `status` lists them |
| provisioning | spot by default (the price the record carries); `KAPSO_BENCH_PROVISIONING=standard` for an arm whose hours a preemption would waste (a spot machine was preempted two minutes into a run); the deletion line bills at the kind's rate for the model used |
| lifetime | `--max-run-duration 14h` (`KAPSO_BENCH_MAX_RUN`) with `--instance-termination-action DELETE`; a spot preemption deletes the machine too, so a long arm leans on the engine's checkpoint and the archive is fetched as the arm goes |
| install | `install <id> <wheel> <benchmark>`: `benchmark_machine_install.sh` over ssh makes `~/bench/venv` (a uv-managed Python 3.12 on every kind, seeded with pip) with the wheel, every bundled harness's `requirements.txt`, the benchmark's `requirements.txt` where one exists and its own stack (RelBench: torch and `relbench[full]==2.1.2` with the modelling libraries; MLE: the MLE-bench library at the development environment's revision with its leaderboards pulled from LFS, torch, kaggle; ALE: the ALE-Bench library at v1.6.1 with its evaluation extras, the engine's `mcp<2` re-pinned after it, the cpp23-202301 judge image built in Docker; IOAI: kaggle and torch), unpacks `benchmarks/common.py` and `benchmarks/<benchmark>` into `~/bench`, installs Node 22 with the two CLIs; the secrets as `fresh_machine.sh` pushes them, the device-auth Codex session with its `~/.codex/SOURCE` marker, and the account's Kaggle credential for the MLE and IOAI benchmarks |
| run | `run <id> "<command>"` sources the secrets, puts the venv on PATH and runs the command from `~/bench`; tee'd to `runs/<id>/run-N.log` |
| fetch, delete | as `fresh_machine.sh`; the deletion line bills at the kind's hourly estimate (`GPU_HOURLY_USD`, `CPU_HOURLY_USD`) |

## semantic.sh

```
scripts/redesign/semantic.sh <s1|s2|s4|s5|s6|author> <host> [--machine <id>] [--advisory "<reason>"] [-- <driver flags>]
```

One semantic test, one host, from one line (`tests/semantic/suite.json`
is the contract; the drivers are the `s*_*.py` and `author_test.py` and
`plugin_proof.py` beside this file). It:

1. **preflights** every role's route — the judge's Vertex route
   (`judge.py --preflight`), the host CLI's login, and the Codex key's
   credit with one small call (statuses only, never the key);
2. provisions a fresh machine with the test's run cap
   (`suite.json` → `machine.max_run`, as `KAPSO_FRESH_MAX_RUN`), installs
   the wheel built from this branch (`uv build`), and installs the
   **fixtures' dependency venv first on PATH** — the examples'
   `requirements.txt` (numpy, pandas, scikit-learn for the churn and
   fraud fixtures), so the generator, the evaluator and `predict.py` all
   resolve them;
3. unpacks the test scaffold (the drivers, the suite, the fixtures under
   `examples/`, the docs, the released tag's docs as `released-docs/` — not the product, which is the installed
   wheel) into `~/kapso-clone`, runs the driver under an immutable run
   id, **archives** the whole run under
   `scripts/redesign/runs/wp9/<run_id>/` (the create/install/wheel logs,
   the driver log, the fetched `out/` with the record, transcripts,
   artifacts and judge directory), **verifies** the record is fetched,
   then **deletes** the machine;
4. **judges and reports on the development box** (its Vertex route, never
   the machine): `judge.py --rubric <r>` for a test with a judge (s1, s2,
   s4, author), then `--grade` where a driver scores its own metrics (s4,
   author), then `tests/semantic/report.py` into
   `tests/semantic/reports/<run_id>.md` with the verdict's exit status.

`--advisory "<reason>"` runs the driver on the development box under a
fresh Linux user that holds nothing of the checkout — the flow, not its
smoothness — and marks the record; a test whose roles need Codex credit
the fresh machine lacks runs this way until the credit exists. The judge
always runs on the development box.

Why an API key for Codex on the machine: a copied ChatGPT session is
single-holder, and the copy that refreshes on the machine revokes the
development box's, which runs the long campaigns. The Claude token is the
one the campaigns already use from the same secret store; probe its weekly
window before a long run, as the token rule says.

# Greywell Bank support agent harness

The support agent that answers Greywell Bank's customer conversations: a model behind a small service,
the harness, that stands between the conversation runner and the model endpoint. Today the agent runs
on Fable 5.1. We want to move it to the local Qwen 27B served on our own box without losing resolution.

## What is here

- `harness/`: the service, `python -m harness.serve --port <port> --upstream <base url>`. As shipped it
  is a pass-through: every request and reply goes through unchanged. The contract is below.
- `bench/`: the development conversations (`dev_tasks.json`, 20 of them), the script that runs them
  through the harness and scores them (`run_dev.py`, with `adapter.py` and `endpoint.json`), the
  runner's documentation (`RUNNER.md`), the fixed conditions of a run (`fixed_conditions.json`), the
  runner's private files by path (`protection.json`), the recorded runs of the current agent and of
  Opus 5.5 (`reference/`) and the provenance notice (`NOTICE.md`).
- `kapso-config.yaml`: the models the build sessions run on.

## The bar

Fable 5.1, the agent we run today, through this same runner, loop, tools, policy and simulated customer
resolves 10 of the 20 development conversations, pass_1 0.50 (`bench/reference/fable-5.1/summary.json`,
one run on 2026-11-02, transcripts beside it). Opus 5.5, recorded the same day, resolves 8 of the 20
(`bench/reference/opus-5.5/`). Qwen 27B with the updated harness must do better than Fable: pass_1
above 0.50 on the development conversations, with the model, its serving settings, the tools, the
simulated customer and the conversations unchanged. A separate set of conversations the build never
sees is kept outside this repository for the final check (below).

Builds with Kapso load the harness at `../../harnesses/agent-engineering`. A finished build is deployed
through Kapso's deployments: `kapso deploy <campaign> --target docker` serves the delivered harness as a
container on this machine, pointed at the model endpoint as a container reaches it (above).

## Running the check

```bash
python bench/run_dev.py --fast     # five conversations once, a smoke check, about fifteen minutes
python bench/run_dev.py            # the 20, about forty-five minutes; prints pass_1 last
```

The model endpoint must be up at `localhost:18000` (Qwen 27B on vLLM, the embeddings beside it on
`18001`) and the simulated customer's keys in `~/.secrets/customer.env`, outside this repository. From
inside a container on this machine the same endpoints are `http://172.17.0.1:18000/v1` and
`http://172.17.0.1:18001/v1`. Results land in `results/`, the logs of each run under `bench/runs/`.

## The held-out conversations

Twenty conversations the build never sees are kept away from this machine while a build runs. Once a
build has finished they are handed over as a folder beside this repository, `../greywell-heldout/`:
`tasks.json` lists them and `reference/fable-5.1/` holds Fable 5.1's recorded run on them.

```bash
python bench/run_dev.py --tasks ../greywell-heldout/tasks.json   # once, into results/heldout/; prints pass_1 last
```

That number beside Fable's in the folder's `reference/fable-5.1/summary.json` is the final check.

## The harness contract


- `python -m harness.serve --port <port> --upstream <base url>` starts it; the evaluator
  starts it the same way from the workspace root, and a deployment serves it the same way.
- `POST /v1/chat/completions` and `GET /v1/models` are the runner's; `GET /health` answers
  200 with `VERSION`, `SETTINGS` and the upstream's models once the upstream answers.
- What may change: `Harness.prepare` (the request on its way to the model), `Harness.review`
  (the reply on its way back), `SETTINGS` (the harness's own endpoint settings: timeouts,
  sampling, thinking budget, retries), and any module added under `harness/`. `VERSION` is
  bumped with every change that is measured, so an evaluation names what ran.
- What holds: an action the runner is to execute comes back as a tool call, in the reply's
  own format; the upstream's errors pass through with their status; the runner's settings
  (its temperature, its tools, its step limits) are the runner's; standard library unless
  `requirements.txt` says otherwise.

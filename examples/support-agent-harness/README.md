# Harness optimization: a customer support agent on a local model

Greywell Bank's support agent answers customer conversations through a small
service, the harness, that stands between the conversation runner and the
model endpoint. Today the agent runs on Fable 5.1. This build optimizes the
harness so that Qwen 27B, served on your own GPU, resolves the conversations
as well as Fable 5.1 does, with the model, its serving settings, the tools,
the simulated customer and the conversations fixed. Two things are called a
harness here: the service under `repo/harness/`, which is what the build
changes, and `agent-engineering`, the Kapso harness the build runs on, which
ships inside the package and loads by name (`kapso harness list`).

This README is the setup: what to bring, how to stand each piece up, and the
lines that run the build, deploy what it delivers and learn from it. The bar
and the starting point are part of the repository: `repo/README.md` states
the bar, and `repo/bench/reference/` holds the runs it is read from.

## What you get

- `repo/`, the team's repository. `harness/serve.py` is the service at
  version 0.1.0, a pass-through: `Harness.prepare` and `Harness.review`
  return their input unchanged. `bench/` holds the 20 development
  conversations by id, the run script and its adapter, the runner's
  description, the fixed conditions, an evaluation profile, and the reference
  runs: Fable 5.1 and Opus 5.5 with their transcripts, and the pass-through
  on Qwen. The README states the bar and the harness contract.
- `evaluation/`, the evaluator: `evaluate.py` runs `bench/run_dev.py`
  unchanged after checking that the bench files, the runner and the endpoint
  are the fixed ones, prints the paired flips against both references, and
  `pass_1` last.
- `kapso-brief.md`, the brief the build is launched with.
- `runner/`, what stands the conversation runner up on your machine: the
  install, the split and the seal of a held-out set, the hand-over after a
  build, the model server, and the step that writes the files naming your
  machine.
- After a build: `campaign/` with a branch per experiment,
  `campaign/.kapso/solution.json` (the record), `campaign/kapso_task.md` (what
  every session read) and `campaign/.kapso/verification/<branch>/` (each
  attempt's diff, evaluation output and verdict).

## What you need

- **A GPU with 80 GB of memory**, for `Qwen/Qwen3.8-27B-FP8` on vLLM 0.32.0
  with `Qwen/Qwen3-Embedding-8B` beside it (the runner's dense search). On
  one H100 the chat server takes 0.72 of the memory and the embeddings 0.25.
- **A key for the simulated customer's model**, called through LiteLLM. The
  example is set up for `azure/gpt-5.2` at temperature 0; the reference runs
  were made with the same model through OpenRouter. The plain OpenAI route
  cannot be used: the runner's embedder takes over the OpenAI SDK's variables
  to reach the local embedding server.
- **Linux with uv, Node and npm, bubblewrap, socat and ripgrep** for the
  runner: τ²-bench 2.0.0 and the sandbox its shell tool runs in.
- **Kapso 0.9.0 or later** with Claude Code and Codex logged in
  (`kapso doctor build` passes), and Docker for the deployment.
- **Time.** One full run of the 20, six conversations at a time, takes 40 to
  60 minutes on one H100; the brief allows 45 to 90 and budgets twelve hours
  for the build. Recording a frontier model as the current agent costs about
  a million input tokens per conversation
  (`repo/bench/reference/fable-5.1/summary.json`), and the simulated customer
  a few cents per conversation at GPT-5.2's list prices.

## Set it up

From this directory. The runner lives in a directory of its own, outside the
repository: it holds the task definitions with their expected outcomes, which
no build session reads.

1. Serve the model, on the GPU machine. The flags are the fixed conditions of
   `repo/bench/fixed_conditions.json`.

   ```bash
   bash runner/serve_model.sh            # qwen on :18000, qwen-embed on :18001
   ```

   If that is another machine than the one the build runs on, forward the two
   ports on the build machine, listening on every address so that a container
   can reach them too. socat does it:

   ```bash
   socat TCP-LISTEN:18000,fork,reuseaddr TCP:<gpu machine>:18000     # and the same for 18001
   ```

   The servers have no authentication: keep both ports closed to the outside.

2. Install the runner and give the simulated customer its credentials.

   ```bash
   bash runner/install.sh ~/support-runner
   ```

   `~/.secrets/customer.env` holds `NAME=value` lines in LiteLLM's variable
   names for your provider: for Azure `AZURE_API_KEY`, `AZURE_API_BASE` and
   `AZURE_API_VERSION` (`2025-04-01-preview` serves gpt-5.2). Add
   `OPENROUTER_API_KEY` if you have one: one development conversation carries
   a natural-language assertion whose judging model is reached through
   OpenRouter. Without the key that conversation ends in an error and counts
   as failed, so the ceiling is 19 of 20; the brief's last rule says so, and
   it goes when you add the key.

3. Keep a held-out set. The development conversations are the 20 this
   repository lists; the sealed ones are a seeded sample of the domain's other
   77, and their definitions leave the runner's directory.

   ```bash
   python runner/split.py ~/support-runner --development repo/bench/dev_tasks.json --sealed 20 --seed 7
   python runner/seal.py ~/support-runner ~/support-vault
   ```

   Move `~/support-vault` off the machine the build runs on. On one machine
   with one account, what keeps a session away from it is the brief's rule
   and the harness's auditors, not the file system.

4. Write the files that name your machine: the endpoint and the runner's
   private paths for `repo/bench/`, the evaluation's copies and its hashes of
   the runner, and `data/`, the pass-through's run as the build receives it.
   It also says whether your runner hashes as the one the example was made
   with (`runner/recorded_hashes.json`).

   ```bash
   python runner/configure.py ~/support-runner
   ```

5. Check, then smoke-test with five conversations (about fifteen minutes).

   ```bash
   kapso doctor build --config repo/kapso-config.yaml
   kapso harness check agent-engineering
   (cd repo && python ../evaluation/evaluate.py --check)      # the bench files, the runner and the endpoint are the fixed ones
   (cd repo && python bench/run_dev.py --fast && rm -r results bench/runs)
   ```

   The smoke run's output is removed because a build copies `repo/` as it
   stands.

## Run it

In Claude Code with the Kapso plugin, opened in `repo/`, the build starts
from one line:

```text
/kapso:build our support agent runs on Fable 5.1 today, and we want to move it to our own Qwen model. improve the agent harness so we get similar or better quality than Fable 5.1 on our customer conversations
```

The plugin reads the repository, audits `bench/run_dev.py` into an
evaluation, writes the brief and asks once. The answer sets the budget, and
the campaign launches detached:

```text
you have 12 hours, update it and go for it
```

The same build from the command line, with the brief and the evaluation of
this directory:

```bash
kapso build --brief kapso-brief.md --repo repo --evaluation evaluation --data data --output campaign --harness agent-engineering --config repo/kapso-config.yaml --detach
kapso watch campaign --follow --config repo/kapso-config.yaml
```

`python run.py` is the same build from Python, in the foreground: run one or
the other. The brief carries two lines that belong to a machine: the
baseline, which is the pass-through's run under `repo/bench/reference/`, and
the rule about the conversation whose judging model needs the OpenRouter
key. Edit them when your setup differs. The repository's README states the
bar as better than Fable 5.1; the brief asks for similar or better: a full
run at the reference's level or above, confirmed by a second run of the same
version.

## Deploy

In the same session, once the campaign has finished, the ask is one line;
the commands under it do the same from the command line.

```text
try deploy it in docker for me
```

```bash
kapso deploy campaign --target docker --config repo/kapso-config.yaml
kapso deploy campaign --status --config repo/kapso-config.yaml
```

The deployment session wraps the delivered harness in a container on a port
the engine chooses, proves one call and records it; `--status` shows the
endpoint. The harness contract in `repo/README.md` is what the container has
to keep answering (`/v1/chat/completions`, `/v1/models`, `/health`), so the
runner can talk to it as it talks to the service. A container reaches the
model endpoint on its host at the Docker bridge address,
`http://172.17.0.1:18000/v1`, which answers only if the endpoint, or the
forward to it, listens on that address or on every address.

## Learn

Again one line in the session, or the commands under it:

```text
use kapso to learn from your experiments for this support agent
```

```bash
kapso learn campaign --detach --config repo/kapso-config.yaml
kapso watch learning/status --follow
```

Learn reads the campaign's experiments and writes lesson cards into your
bank, each a rule with the evidence that earned it; `kapso experience list`
shows the bank, and a later build of the same domain can be served from it.

## The held-out check

Once the build is over, hand the sealed conversations back to the runner and
run what the build delivered on them once, in a clone of the campaign so the
campaign's own record stays as it is (`campaign/.kapso/solution.json` names
the delivered branch):

```bash
python runner/handover.py ~/support-runner ~/support-vault greywell-heldout
git clone --branch <delivered branch> campaign heldout-check
(cd heldout-check && python bench/run_dev.py --tasks ../greywell-heldout/tasks.json)     # into results/heldout/; prints pass_1 last
```

That number needs the current agent's beside it. Record the current agent on
the sealed conversations where the vault lives, with a runner that was not
sealed and the embedding server up, never on the build machine while a build
runs. A reference run of the current agent, called through OpenRouter:

```bash
python ~/full-runner/run_conversations.py sealed_fable alltools 1 \
  $(python -c "import json; print(*json.load(open('$HOME/support-vault/split.json'))['sealed'])") \
  -- --max-concurrency 6 --auto-resume --agent-llm openrouter/anthropic/claude-fable-5.1 --agent-llm-args '{"temperature": 0.0}'
```

The run's results file lands under that runner's
`tau2-bench-2.0.0/data/simulations/sealed_fable/`. Add
`--reference-results <that file> --reference-name fable-5.1`
to the hand-over line above and it also converts the run into
`greywell-heldout/reference/fable-5.1/`, in the shape of
`repo/bench/reference/`. Read the two numbers as the development ones are
read: against the run-to-run flips, not as a headline.

## What to look at

- `campaign/.kapso/solution.json`: the delivered branch, its score and the
  closing feedback.
- `campaign/.kapso/verification/<branch>/verdict.json`: what the harness's
  two auditors found about each attempt, among it whether any session read a
  protected path.
- `kapso trace campaign`: every attempt with its sessions, what was voided
  and on what evidence.
- The paired flips the evaluator prints: which conversations a run gained
  and lost against Fable 5.1 and Opus 5.5. `repo/bench/eval_profile.md` says
  how much of that is noise.
- `campaign/harness/` on the delivered branch: what the service became.

## Change it

- The model: another locally served model changes `runner/serve_model.sh`,
  `repo/bench/fixed_conditions.json` and the endpoint identity in
  `runner/configure.py` together, and needs its own pass-through run as the
  baseline.
- The bar: record another current agent through the same runner and put its
  run under `repo/bench/reference/`; the evaluator's reference table
  (`evaluation/protected/references.json`) follows it.
- The Kapso harness: copy `agent-engineering` out of the package
  (`src/kapso/harness/bundled/agent-engineering/` in this repository), edit
  a page, bump its `version`, run `kapso harness check` on the copy and name
  the copy's directory where the lines above name `agent-engineering`. A
  campaign refuses a harness that changed after its launch.

## What ran as shipped, and what was adapted

`repo/`, `evaluation/` and `kapso-brief.md` are the files a build of ours
was launched with: the run script, the adapter, the evaluator and the service
are unchanged. Four things differ. The files that name a machine
(`bench/endpoint.json`, `bench/protection.json` and the evaluation's
`fixed.json`) are written by `runner/configure.py`, where `fixed.json` also
hashes `runner.json`, the settings our run script held in its own source.
The protection list gained the directory of per-task files. And a few phrases
that named our own deployment were reworded. And the harness is named as
the package ships it, where our build named a directory beside this one.

The scripts under `runner/` are adapted from the ones we used and were
checked on 2026-11-10 without a GPU. `install.sh` installs a runner whose
lock file, source, database, prompts and documents hash as ours did.
`run_conversations.py` builds the same runner command line as our script.
`split.py`, `seal.py` and `handover.py` were run on a fresh install, and the
hand-over's reference conversion reproduces the shipped Fable 5.1 files.
`configure.py` writes files the evaluation's integrity check accepts against
a stand-in endpoint. Not run in this form: `serve_model.sh`, which is our
start-up script without its cloud-specific parts, and a full run through the
adapted scripts, which needs the GPU.

## Provenance

The conversations and the runner derive from the banking knowledge domain of
τ²-bench 2.0 by Sierra Research, under the MIT licence
(`repo/bench/NOTICE.md`). A conversation is scored by the final state of the
account database or by the required actions, never by a model, except the
one natural-language assertion noted above. The company and customer names in
the conversations are the benchmark's own fictions; Greywell Bank is ours.

<h1 align="center">Kapso</h1>

<h4 align="center">Turn coding-agent output into verified outcomes, not review debt</h4>
<p align="center"><sub>plug in the harness for your domain, set the goal, every step traced</sub></p>

<p align="center">
  <a href="https://docs.leeroo.com/docs">Learn more</a> ·
  <a href="https://discord.gg/hqVbPNNEZM">Join Discord</a> ·
  <a href="https://leeroo.com">Website</a>
</p>

<p align="center">
  <a href="https://pypi.org/project/leeroo-kapso/"><img src="https://img.shields.io/pypi/v/leeroo-kapso?color=blue" alt="PyPI"></a>
  <a href="https://discord.gg/hqVbPNNEZM"><img src="https://dcbadge.limes.pink/api/server/hqVbPNNEZM?style=flat" alt="Discord"></a>
  <a href="https://github.com/leeroo-ai/kapso"><img src="https://img.shields.io/github/commit-activity/m/leeroo-ai/kapso" alt="GitHub commit activity"></a>
  <a href="https://www.ycombinator.com/companies/leeroo"><img src="https://img.shields.io/badge/Y%20Combinator-X25-orange?logo=ycombinator&logoColor=white" alt="Y Combinator X25"></a>
</p>

<p align="center">
  If you like this project, please support us by giving it a star ⭐
</p>

<p align="center">
  <a href="https://github.com/Leeroo-AI/kapso/raw/main/docs/images/kapso-hero.mp4">
    <img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/kapso-hero.gif" alt="Kapso running: a campaign searches a tree of experiments while the leaderboard climbs; every trajectory is harvested into an expertise wiki of scored insights and procedures, which grounds the next campaign" width="880">
  </a>
</p>

> **If you're an AI agent looking at this repo:** start with the operating skill,
> [plugin/skills/kapso/SKILL.md](https://raw.githubusercontent.com/Leeroo-AI/kapso/main/plugin/skills/kapso/SKILL.md),
> one file for every host. It shows how to leverage Kapso for your use case, and it is
> faster than reading the source. The plugin that carries it, with the skills for
> setting up, stating an objective, authoring a harness and reviewing a run, installs
> in one line: `claude plugin marketplace add Leeroo-AI/kapso && claude plugin install kapso@kapso`
> (Codex: `codex plugin marketplace add Leeroo-AI/kapso && codex plugin add kapso@kapso`).

---

## News

- **IOAI 2026 · the AI Model Track.** The [International Olympiad in AI](https://ioai-official.org) is the IMO of the AI era — 471 contestants from 108 countries and territories, six expert-designed tasks under a single-GPU budget. In 2026 it opened [IOAI²](https://ioai-official.org/ai-model-track/), where AI systems sit the *same* exam in two unattended 6-hour sessions. Once the clock starts, no human may solve, correct, or improve anything. Kapso entered as one of 14 Founding AI Participants.
  - 🎓 **Outscored all 471 humans** — 536.07 total, above every contestant in the hall.
  - 🏆 **IOAI² Grand Master Trophy** — top 3 among all AI systems entered. [Full results →](benchmarks/ioai2026/README.md) · [IOAI's review of the track →](https://ioai-official.org/what-happens-when-autonomous-ai-takes-on-the-same-tasks-as-the-worlds-top-young-ai-talents/)
- **First on all three [RelBench](benchmarks/relbench/README.md) leaderboards**: on Stanford's benchmark for predictive ML over relational data, Kapso leads the official leaderboard in classification (81.2 AUROC over 12 tasks, against 79.3 for [KumoRFM-v2](https://docs.nvidia.com/sdgm/rfm/overview) and 79.0 for Stanford STAR's RT-J), regression (0.2476 NMAE over 9 tasks, against 0.2711 for RT-J and 0.2847 for KumoRFM-v2, 8.7% less error) and recommendations (18.4 MAP over 10 tasks, against 9.3 for ID-GNN, the best other entry). Every number is an entry on the [official RelBench leaderboard](https://star-project.stanford.edu/relbench/leaderboard/).

  <img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/relbench.png" alt="RelBench, three panels: classification, Kapso 81.2 AUROC against KumoRFM-v2's 79.3 and RT-J's 79.0; regression, Kapso 0.2476 NMAE against RT-J's 0.2711 and KumoRFM-v2's 0.2847, lower being better; recommendations, Kapso 18.4 MAP against the official leaderboard's best other entry, ID-GNN, at 9.3. The first two panels are drawn from a truncated axis, the third from zero" width="820">

- **[Leeroopedia MCP Integration](https://leeroopedia.com)**: Kapso now connects to **Leeroopedia MCP** — your ML & Data Knowledge Wiki. Learnt by AI, built by AI, for AI. A centralized playbook of best practices and expert-level knowledge for Machine Learning and Data domains. Kapso agents use it during ideation and implementation to search knowledge, build plans, diagnose failures, and more.
- **[Moltbook Agents 🦞](https://www.moltbook.com/)**: Build AI agents that optimize other agents and debate on Moltbook! [Get started →](moltbook_bot/README.md)
- **Technical Report**: Our technical report is now available! [Read the paper](https://arxiv.org/abs/2601.21526)
- **#1 on [MLE-Bench](benchmarks/mle/README.md)**: KAPSO achieved top ranking among open-source systems on Kaggle ML competitions (MLE Benchmark).

  <img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/mle-bench.png" alt="MLE-Bench: medal rate by difficulty split against R&amp;D-Agent, AIRA-dojo, ML-Master and AIDE" width="820">

- **#1 on [ALE-Bench](benchmarks/ale/README.md)**: KAPSO achieved top ranking on long-horizon algorithmic discovery problems (ALE Benchmark).

  <img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/ale-bench.png" alt="ALE-Bench: Kapso reaches a final rating of 1909 Elo against ALE Agent's 1879, above a breakdown across ten AtCoder Heuristic Contest problems; every row is drawn from its own truncated Elo origin, and the per-problem bar gaps share one scale" width="820">

## What is KAPSO?

Coding agents burn tokens and hand back diffs you have to review. Kapso runs them differently. You plug in a harness for your domain and set a goal; it works the goal until it is met or the budget is spent, measuring every attempt with an evaluation the agents cannot touch and keeping every step on record. What comes back is a verified outcome: the branch that met the goal, or the best the budget reached, with the evaluation that measured it, ready to deploy as running software.

A harness teaches Kapso one kind of work: how to propose, build, measure and, where the domain needs it, deploy there, and the tools it needs. Two ship with Kapso, predictive modelling and competitive programming; the examples carry a trading harness, and you write your own for anything with an evaluation.

It learns from experience, and the lessons carry their receipts: one rule per card, the condition it applies under, the measured effect from each run that tested it, a reliability score with its rationale.

Kapso is an open-source Python framework by [Leeroo](https://leeroo.com), published on PyPI as [`leeroo-kapso`](https://pypi.org/project/leeroo-kapso/).

### The Four Pillars

| Pillar | Method | Description |
|--------|--------|-------------|
| [**Build**](https://docs.leeroo.com/docs/build/brief) | `.build()` | Run an experiment campaign that meets a brief: candidates implemented on their own branches, measured, the best kept and refined. |
| [**Learn**](https://docs.leeroo.com/docs/learn/what-learn-produces) | `.learn()` | Mine a finished campaign into your bank: evidence-priced lesson cards, each with its reliability and the measurements that earned it. |
| [**Experience**](https://docs.leeroo.com/docs/experience) | `.load_experience()` / `.add_experience()` | What a build is served: your bank, the hosted Leeroopedia gate, a bank your team shares, or a corpus ingested from a repository or a researched question. A build is served exactly what it loads. |
| [**Deploy**](https://docs.leeroo.com/docs/deploy/targets) | `.deploy()` | Turn a solution into running software. Supports local execution, Docker containers, or cloud platforms like Modal. |

## 🚀 Quickstart

### Installation

**0. In a coding agent?** Install the Kapso plugin and let it walk the
steps below with you:

```bash
claude plugin marketplace add Leeroo-AI/kapso && claude plugin install kapso@kapso   # Claude Code
codex plugin marketplace add Leeroo-AI/kapso && codex plugin add kapso@kapso         # Codex
```

Then `/kapso:setup` installs and verifies, and `/kapso:build <what you want>`
launches the first campaign (on Codex: `$kapso-setup`, `$kapso-build <what you want>`). See [The Kapso plugin](https://docs.leeroo.com/docs/coding-agent-skills).

**1. Prerequisites.** Kapso runs its inference through coding-agent CLIs
(there is no direct-API fallback), so you need Node.js and both agent
CLIs logged in before anything works:

```bash
# Node.js 18+ (https://nodejs.org), then:
npm install -g @openai/codex            # research, judging, utilities
codex login

npm install -g @anthropic-ai/claude-code  # ideation + implementation (the default host)
claude auth login
```

An OpenAI key is optional: only the knowledge graph and semantic search
over past experiments (`models.embedding` in your config) embed text, and
generation never uses it. When you turn either on:

```bash
echo 'OPENAI_API_KEY=sk-...' >> .env
```

**2. Connect Leeroopedia (recommended).** One key gives every campaign
session [Leeroopedia](https://leeroopedia.com), a hosted knowledge base of
ML and AI frameworks: how a framework actually behaves, plan reviews,
documented hyperparameter ranges, failure diagnoses. Nothing to install;
a new account at the [Leeroopedia dashboard](https://app.leeroopedia.com/dashboard)
comes with $20 of free credit.

```bash
echo 'LEEROOPEDIA_API_KEY=kpsk_...' >> .env
```

**3. Install the package** (Python 3.10+):

```bash
pip install leeroo-kapso
```

The package is `leeroo-kapso`. The PyPI package named `kapso` is an unrelated
WhatsApp tool that installs a `kapso` command of its own and shadows this one;
if it is present, `pip uninstall kapso` first.

**4. Verify the setup:**

```bash
kapso doctor
```

`doctor` reports what **your config** actually needs, and names the exact
fix for anything missing. Requirements follow the config, so an all-codex
setup is never asked for `claude`. Narrow it to one verb to see just that
verb's requirements:

```bash
kapso doctor build             # or learn / deploy / experience
```

Those are the same checks the verb itself runs before it does any work —
`kapso.build(...)` fails in seconds on a missing CLI rather than deep
inside a session. Items marked `[-- ]` are optional; they limit features
you may not need (docker, Weaviate, Neo4j, the deploy targets).

**Or let your coding agent do all of this:** paste the
[setup prompt](https://docs.leeroo.com/docs/installation#set-up-through-your-coding-agent)
into Claude Code, Codex or OpenCode.

**The stores an index needs (optional)** — `kapso experience add --repo`
and `--research` ingest a corpus into local Weaviate + Neo4j. From a
source checkout:

```bash
bash scripts/start_infra.sh   # starts both via docker
```

**From source (for development)**

```bash
git clone https://github.com/leeroo-ai/kapso.git
cd kapso

conda create -n kapso python=3.12 && conda activate kapso
pip install -e .
```

The legacy aider adapter is an extra (`pip install "leeroo-kapso[aider]"`,
Python <3.13); the default claude/codex agents need no extras.

### Basic Usage

The core loop needs nothing beyond the prerequisites above:

```python
from kapso import Kapso, Budget

kapso = Kapso()   # no knowledge graph needed to start

# Build: a solution through experimentation. The campaign prints
# `status: <path>` at launch — watch it live from another terminal with
#     kapso watch ./campaign
# If a session needs something only you can provide (an API key, a file),
# the campaign pauses and `kapso inbox reply` resumes it:
#     https://docs.leeroo.com/docs/build/inbox
# Share the result as Markdown for a pull request or issue with
#     kapso report ./campaign -o report.md
solution = kapso.build(
    "Optimize the model in train.py; target accuracy > 0.80 on evaluate.py",
    repo="./my_project",                 # or omit to start from scratch
    output="./campaign",
    budget=Budget(minutes=120),
)
print(solution.explain())

# Learn from the campaign you just ran: mine the trajectory, grade the
# lessons, and bank evidence-priced knowledge cards. The bank (a local
# git repo) is created automatically on first use — lessons stay on your
# machine until you share them:
#     kapso experience share <git-url>
# after which every learn() pushes the bank there.
lesson = kapso.learn(solution)
print(lesson.explain())

# Build again, served the cards it just earned: a build is served exactly
# the experience it loads.
kapso.load_experience("mine")
solution2 = kapso.build("...", output="./campaign2")
```

With the stores running, you can also ingest outside knowledge into an
index and serve it the same way:

```python
from kapso import Kapso

kapso = Kapso()

# Once: a repository, or the web researched for a question, into an index
kapso.add_experience(repo="https://github.com/huggingface/trl", name="trl")
kapso.add_experience(research="RLHF and DPO fine-tuning for legal contract analysis", name="dpo-legal")

# Every build that loads the index searches it; Leeroopedia rides on its key
kapso.load_experience("trl")
kapso.load_experience("leeroopedia")
solution = kapso.build("Fine-tune Llama-3.1-8B for clause risk classification")
```

A note on budgeting: an ingest extracts everything the material supports.
A single question can become dozens of linked pages and an hours-long
ingest; ingest time scales with extractable substance, so add the
sources you will search, not every source you have.

And to turn a solution into running software:

```python
software = kapso.deploy(solution, target="local")   # or docker, modal, bentoml, langgraph, auto
result = software.run(software.proof["inputs"])     # the call the deployment proved
software.stop()
```

### How do I bound a campaign?

Use `max_iterations`, `time_budget_minutes`, and `cost_budget` to bound iterations, elapsed time, and best-effort model spend in USD.
For platform costs, enable [cost-aware selection](https://docs.leeroo.com/docs/evolve/cost-aware-selection) in `search_strategy.params.cost` with your estimator, `cap_per_candidate`, and `campaign_budget`; its `weight` also lets selection prefer cheaper candidates.

### Choosing models

Every model Kapso uses is named in one config file. The packaged default
runs build sessions on `claude-opus-5`, the learning crews on
`claude-fable-5`, and codex roles on `gpt-5.6-sol` — but model access is
subscription-dependent (a plan can cap one model while serving another).
To run on different models, copy the packaged config, edit, and point
Kapso at yours:

```python
from pathlib import Path
import yaml
from kapso import Kapso
from kapso.kapso import DEFAULT_CONFIG_PATH

config = yaml.safe_load(Path(DEFAULT_CONFIG_PATH).read_text())
# e.g. run the learning crews on opus instead of fable:
crews = yaml.safe_dump(config).replace("claude-fable-5", "claude-opus-5")
Path("kapso-config.yaml").write_text(crews)

kapso = Kapso(config="kapso-config.yaml")
```

Text-only inference roles can opt in to `openai_compatible` Chat Completions.
Set the model and endpoint in your config; credentials stay in the environment.
Load the config with `Kapso(config=...)` or `kapso build --config ...`.
Coding and web research still require agents with the corresponding tools.
See the [configuration example and limits](https://docs.leeroo.com/docs/evolve/coding-agents#openai-compatible-api).

Before a long run, probe every model your config names against your
actual subscriptions — a model your login cannot serve fails here in
seconds instead of hours into a run. A usage cap on a model you can
serve is not visible to a one-token probe:

```bash
kapso doctor --models                            # packaged config
kapso doctor --models --config kapso-config.yaml # yours
kapso doctor learn --models                      # just the learning crews
```

To make that probe part of every call rather than a thing you remember to
run, turn it on in your config. It costs one throwaway token per distinct
`{cli, model}` pair, which is worth it before an unattended multi-hour
`learn()`:

```yaml
preflight:
  enabled: true            # static checks before every verb (default)
  live_model_probe: true   # + one-token model probes (default: false)
```

Model swaps change *pacing* too: the crew `timeout_minutes` caps in the
config were calibrated on the default models, and a swapped model that
reasons longer may need them raised.

For detailed integration steps, see the [Quickstart](https://docs.leeroo.com/docs/first-build) and [Installation](https://docs.leeroo.com/docs/installation) guides.

## Examples

Every example is a task: a brief, a starting repository, the data or how to get it, and a driver of at most twelve lines; [examples/README.md](examples/README.md) has the minutes, the one command and whether a result is recorded.

| Example | What it shows |
|---------|---------------|
| [**churn-first-build**](examples/churn-first-build/README.md) | Your first build: an evaluator present, a baseline, a target, the record |
| [**fraud-no-evaluator**](examples/fraud-no-evaluator/README.md) | No evaluator: the campaign's first node builds one from the brief's protocol and every candidate is measured against it |
| [**pipeline-dbt**](examples/pipeline-dbt/README.md) | A dbt pipeline on DuckDB whose exports carry defects: the data contract made to hold |
| [**trading**](examples/trading/README.md) | A harness on the build: a setup session, pages and hooks in every session, verification per attempt |
| [**full-loop**](examples/full-loop/README.md) | Build, the review, learn, build again served the lesson, deploy: a Kaggle tabular model end to end |
| [**prompt-harness**](examples/prompt-harness/README.md) | The prompt as the thing under test, scored on AIME problems |
| [**agent-scaffold**](examples/agent-scaffold/README.md) | A vision-language extractor, scored against ground-truth tables |
| [**torch-performance**](examples/torch-performance/README.md) | A faster module on CUDA under a numerical tolerance |
| [**gpu-kernels**](examples/gpu-kernels/README.md) | Multi-head attention on CUDA: flash attention or Triton |

## Supported Benchmarks

| Benchmark | Description |
|-----------|-------------|
| [**MLE-Bench**](benchmarks/mle/README.md) | OpenAI's ML-engineering benchmark — full competitions across tabular, vision, text, and audio, from raw data to graded submission |
| [**ALE-Bench**](benchmarks/ale/README.md) | Sakana AI's algorithmic-optimization benchmark — design, implement, and iterate contest heuristics over hours-long searches |
| [**RelBench**](benchmarks/relbench/README.md) | Stanford's benchmark for predictive ML over relational data — classification, regression, and recommendation straight from the multi-table databases of Amazon, H&M, Stack Exchange, clinical trials, and more |
| [**IOAI 2026**](benchmarks/ioai2026/README.md) | Timed olympiad ML across vision, language, and optimization — expert-set tasks, contest hardware, zero human help |

Each benchmark also has a documentation page covering how to run it, its CLI options and what the output holds:
[IOAI 2026](https://docs.leeroo.com/docs/benchmarks/ioai-2026) ·
[MLE-Bench](https://docs.leeroo.com/docs/benchmarks/mle-bench) ·
[ALE-Bench](https://docs.leeroo.com/docs/benchmarks/ale-bench) ·
[RelBench](https://docs.leeroo.com/docs/benchmarks/relbench)

## 📚 Documentation & Support

- **Full Documentation**: [docs.leeroo.com](https://docs.leeroo.com/docs)
  - [Installation](https://docs.leeroo.com/docs/installation) — the coding-agent CLIs, the package, and `kapso doctor`
  - [Quickstart](https://docs.leeroo.com/docs/first-build) — your first campaign
  - [CLI reference](https://docs.leeroo.com/docs/reference/cli) — every command, flag and default
  - [Python API](https://docs.leeroo.com/docs/reference/kapso-api) — every public method on `Kapso`
  - [Configuration](https://docs.leeroo.com/docs/reference/configuration) — every key in `config.yaml`
  - [What learn produces](https://docs.leeroo.com/docs/learn/what-learn-produces) — the lesson bank, its cards with their receipts, and [development mode](https://docs.leeroo.com/docs/learn/development-mode)
  - [Evolve](https://docs.leeroo.com/docs/evolve/overview) — how a campaign runs, and [what happens inside one experiment](https://docs.leeroo.com/docs/evolve/execution-flow)
  - [Experience](https://docs.leeroo.com/docs/experience) — your bank, Leeroopedia, repositories and papers turned into searchable knowledge
  - [Deploy](https://docs.leeroo.com/docs/deploy/targets) — local, Docker, Modal and the other targets
  - [The Kapso plugin](https://docs.leeroo.com/docs/coding-agent-skills) — the commands, the skills and the MCP server for Claude Code, Codex and Cursor
- **From your coding agent**: the docs run an MCP server at `https://docs.leeroo.com/mcp`, and every page is served as Markdown — see [llms.txt](https://docs.leeroo.com/llms.txt) and [Kapso in your coding agent](https://docs.leeroo.com/docs/coding-agent-skills).
  ```bash
  claude mcp add --transport http kapso-docs https://docs.leeroo.com/mcp   # Claude Code
  codex mcp add kapso-docs --url https://docs.leeroo.com/mcp              # Codex CLI
  ```
- **Let your coding agent run Kapso**: the plugin under [`plugin/`](https://github.com/Leeroo-AI/kapso/tree/main/plugin) carries ten commands, six skills and the MCP server, one install line per host (step 0 above); OpenCode loads the skills as a folder (`ln -s <clone>/plugin/skills .opencode/skills`). See [The Kapso plugin](https://docs.leeroo.com/docs/coding-agent-skills).
- **Give your coding agent Kapso's knowledge tools**: `kapso-mcp`, installed with the package, serves web research, the repository's memory, the campaign's experiment history and the lesson bank to the session you work in — see [MCP gates](https://docs.leeroo.com/docs/evolve/mcp-gates#use-the-gates-from-your-own-coding-agent).
  ```bash
  claude mcp add kapso -- kapso-mcp   # Claude Code
  codex mcp add kapso -- kapso-mcp    # Codex CLI; then `startup_timeout_sec = 60` under [mcp_servers.kapso] in ~/.codex/config.toml (Codex's ten seconds are short on a busy machine)
  ```
- **Community**: [Discord](https://discord.gg/hqVbPNNEZM)
- **Website**: [leeroo.com](https://leeroo.com)


## Kapso for Enterprise

Kapso gets better at your company the longer it works: every task feeds a living knowledge bank of your systems, your data, and your hard-won lessons. To onboard Kapso for your challenging enterprise tasks and build that live company context, [talk to us](https://leeroo.com/contact-us).

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details on how to get started.

## Citation

If you use Kapso in your work, please cite the technical report:

```bibtex
@article{nadafian2026kapso,
  title   = {KAPSO: A Knowledge-grounded framework for Autonomous Program Synthesis and Optimization},
  author  = {Nadafian, Alireza and Mohammadshahi, Alireza and Yazdani, Majid},
  journal = {arXiv preprint arXiv:2601.21526},
  year    = {2026},
  url     = {https://arxiv.org/abs/2601.21526}
}
```

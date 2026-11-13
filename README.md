<h1 align="center">Kapso</h1>

<h4 align="center">An engine that runs experiments, deploys the best solution, and learns from every run</h4>
<p align="center"><sub>smart experimentation · learning · deployment · domain adaptation</sub></p>

<div align="center">
<!-- translations:start -->
<details><summary><b>Read this in other languages</b></summary>

<b>English</b> | <a href="docs/translations/README.zh-CN.md">简体中文</a> | <a href="docs/translations/README.ja-JP.md">日本語</a> | <a href="docs/translations/README.ko-KR.md">한국어</a> | <a href="docs/translations/README.de-DE.md">Deutsch</a> | <a href="docs/translations/README.fr-FR.md">Français</a> | <a href="docs/translations/README.es-ES.md">Español</a> | <a href="docs/translations/README.hi-IN.md">हिन्दी</a> | <a href="docs/translations/README.pt-BR.md">Português</a> | <a href="docs/translations/README.ru-RU.md">Русский</a> | <a href="docs/translations/README.zh-TW.md">繁體中文</a>

</details>
<!-- translations:end -->
</div>

<p align="center">
  <a href="https://docs.leeroo.com/docs">Learn more</a> ·
  <a href="https://www.youtube.com/watch?v=W4FHlza1fRs">Watch the demo</a> ·
  <a href="https://discord.gg/hqVbPNNEZM">Join Discord</a> ·
  <a href="https://leeroo.com">Website</a>
</p>

<p align="center">
  <a href="https://pypi.org/project/leeroo-kapso/"><img src="https://img.shields.io/pypi/v/leeroo-kapso?color=blue" alt="PyPI"></a>
  <a href="https://discord.gg/hqVbPNNEZM"><img src="https://dcbadge.limes.pink/api/server/hqVbPNNEZM?style=flat" alt="Discord"></a>
  <a href="https://github.com/leeroo-ai/kapso"><img src="https://img.shields.io/github/commit-activity/m/leeroo-ai/kapso" alt="GitHub commit activity"></a>
  <a href="https://www.ycombinator.com/companies/leeroo"><img src="https://img.shields.io/badge/Y%20Combinator-P25-orange?logo=ycombinator&logoColor=white" alt="Y Combinator P25"></a>
</p>

<p align="center">
  If you like this project, please support us by giving it a star ⭐
</p>

<p align="center">
  <a href="https://github.com/Leeroo-AI/kapso/raw/main/docs/images/kapso-hero.mp4">
    <img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/kapso-hero.gif" alt="Kapso running: smart experimentation searches a tree of experiments against your objective and success criteria; the best solution leaves for deployment, a shorter experimentation of the same kind that ends in a deployed solution; every run's trajectories feed learning, which files lessons, each with a body, a boundary, a credence score and its evidence, into the bank that grounds the next campaign" width="880">
  </a>
</p>

> **If you're an AI agent looking at this repo:** the Kapso plugin, with the skills for
> setting up, stating an objective, authoring a harness and reviewing a run, installs
> in one line: `claude plugin marketplace add Leeroo-AI/kapso && claude plugin install kapso@kapso`
> (Codex: `codex plugin marketplace add Leeroo-AI/kapso && codex plugin add kapso@kapso`).

---

## News

- **IOAI 2026 · the AI Model Track.** The [International Olympiad in AI](https://ioai-official.org) is the IMO of the AI era: 471 contestants, six expert-designed tasks, a single-GPU budget. Its 2026 [IOAI²](https://ioai-official.org/ai-model-track/) track gave AI systems the same exam in one autonomous session. Kapso, one of 14 Founding AI Participants, scored 536.07 as graded by the organizers, above every human contestant, and took the IOAI² Grand Master Trophy, top 3 among the AI systems entered. [Full results →](benchmarks/ioai2026/README.md) · [IOAI's review of the track →](https://ioai-official.org/what-happens-when-autonomous-ai-takes-on-the-same-tasks-as-the-worlds-top-young-ai-talents/)
- **First on all three [RelBench](benchmarks/relbench/README.md) leaderboards.** On Stanford's benchmark for predictive ML over relational data, Kapso leads the [official leaderboard](https://star-project.stanford.edu/relbench/leaderboard/) in classification (81.2 AUROC against 79.3 for [KumoRFM-v2](https://docs.nvidia.com/sdgm/rfm/overview) and 79.0 for Stanford STAR's [RT-J](https://star-project.stanford.edu/rt-j)), regression (0.2476 NMAE against 0.2711 and 0.2847) and recommendations (18.4 MAP against 9.3 for ID-GNN, the best other entry).

  <p align="center"><img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/relbench.png" alt="RelBench, three panels: Kapso 81.2 AUROC against KumoRFM-v2's 79.3 and RT-J's 79.0 in classification; 0.2476 NMAE against 0.2711 and 0.2847 in regression, lower being better; 18.4 MAP against ID-GNN's 9.3 in recommendations. The first two panels use a truncated axis, the third starts at zero" width="574"></p>

- **[Leeroopedia](https://leeroopedia.com)**: our public bank of lessons in the data and ML domains, around 100k pages extracted by experimenting on papers and repositories, serving about 6M requests a month, around 28% of them from AI agents. Kapso's sessions query it over MCP during ideation and implementation.
- **#1 on [MLE-Bench](benchmarks/mle/README.md)**: KAPSO achieved top ranking among open-source systems on Kaggle ML competitions (MLE Benchmark).

  <p align="center"><img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/mle-bench.png" alt="MLE-Bench: medal rate by difficulty split against R&amp;D-Agent, AIRA-dojo, ML-Master and AIDE" width="574"></p>

- **#1 on [ALE-Bench](benchmarks/ale/README.md)**: KAPSO achieved top ranking on long-horizon algorithmic discovery problems (ALE Benchmark).

  <p align="center"><img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/ale-bench.png" alt="ALE-Bench: Kapso reaches a final rating of 1909 Elo against ALE Agent's 1879, above a breakdown across ten AtCoder Heuristic Contest problems; every row is drawn from its own truncated Elo origin, and the per-problem bar gaps share one scale" width="574"></p>

## What is Kapso?

Kapso is an open-source engine that runs experiments, deploys the best solution, and learns from every run. It starts from a domain harness and an empty bank of lessons, and given an objective and success criteria it:

- **runs a tree of experiments**: proposes hypotheses, then implements and verifies each one;
- **deploys the best solution** on the target stack;
- **learns from the run**: credit goes to each hypothesis and implementation by its impact on the outcome, and the bank is updated, every lesson carrying a credence score and a boundary on where it applies.

Over time the domain harness carries a bank of lessons: the experimented best practices of that environment. Kapso is model-agnostic and coding-agent-agnostic, by [Leeroo](https://leeroo.com), published on PyPI as [`leeroo-kapso`](https://pypi.org/project/leeroo-kapso/).

### The four parts

| Part | Verb | What it does |
|------|------|--------------|
| [**Smart experimentation**](https://docs.leeroo.com/docs/build/brief) | `.build()` | Each experiment is a hypothesis implemented on its own branch and scored, in a tree. Ideation draws on earlier experiments and the bank of lessons and re-ranks ideas by expected return against budget, so they fit this environment rather than a bare LLM's or RAG's generic ones; a fingerprinted evaluator no experiment can touch verifies the result, and modules can propose evaluation updates for the verifier to approve. [How it works →](https://docs.leeroo.com/docs/evolve/overview) |
| [**Learning**](https://docs.leeroo.com/docs/learn/what-learn-produces) | `.learn()` / `.load_experience()` | Every lesson is induced or deduced from experimentation traces and carries a credence score, a boundary on where it applies and its evidence; `learn()` updates them from each run's outcome and A/B tests every change by rebuilding with and without it, so improvement is measured, not guessed. The bank is built on git: `kapso experience share`, `list` and `update`. [Experience →](https://docs.leeroo.com/docs/experience) names what a build is served. |
| [**Deployment**](https://docs.leeroo.com/docs/deploy/targets) | `.deploy()` | A shorter experimentation of the same kind adapts the best solution to the target stack, deploys it and proves health checks and a working call. Docker, Modal, BentoCloud and LangGraph Platform ship; a harness can add a target. |
| [**Domain adaptation**](https://docs.leeroo.com/docs/harnesses) | `kapso harness` | A harness is Kapso's domain package: the knowledge and workflows of an area such as predictive modelling, so a build never starts from scratch, and where the bank of lessons accumulates. Predictive modelling and competitive programming ship; write yours with `/kapso:harness` in the plugin or `kapso harness` in the CLI. |

## 🚀 Quickstart

<p align="center">
  <a href="https://www.youtube.com/watch?v=W4FHlza1fRs">
    <img src="https://i.ytimg.com/vi/W4FHlza1fRs/maxresdefault.jpg" alt="Kapso walkthrough video" width="720">
  </a>
  <br><sub>▶ Watch the walkthrough before you install</sub>
</p>

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

**2. Connect Leeroopedia (recommended).** One key gives every session
[Leeroopedia](https://leeroopedia.com), our hosted bank of lessons on ML and
AI frameworks. Nothing to install; a new account at the
[Leeroopedia dashboard](https://app.leeroopedia.com/dashboard) comes with
$20 of free credit.

```bash
echo 'LEEROOPEDIA_API_KEY=kpsk_...' >> .env
```

**3. Install the package** (Python 3.10+):

```bash
pip install leeroo-kapso
```

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

The core verbs need nothing beyond the prerequisites above:

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

## Examples

Every example is a task: a brief, a starting repository, the data or how to get it, and a driver of at most twelve lines; [examples/README.md](examples/README.md) has the minutes and the one command.

| Example | The task |
|---------|----------|
| [**Data engineering (dbt models on Snowflake)**](examples/dbt-snowflake/README.md) | Fix the dbt models with data quality issues and build the new ones that a backlog of seventeen tickets asks for, on a retail warehouse on Snowflake, until every ticket passes its production tests; then deliver one pull request per ticket and release them to production. |
| [**Harness optimization (customer support agent)**](examples/support-agent-harness/README.md) | Optimize the harness of a customer support agent to switch the underlying LLM from Fable 5.1 to a local Qwen 27B without losing resolution on the customer conversations, then deploy it in Docker and learn from the run. |
| [**Predictive modelling (customer churn)**](examples/churn-first-build/README.md) | Predict which customers will leave: raise a churn model's out-of-sample ROC AUC toward the target, under an evaluator the campaign cannot edit. |
| [**Fraud detection**](examples/fraud-no-evaluator/README.md) | Score transactions for fraud when the repository has no evaluator: the brief states the protocol, the campaign builds the evaluator first, then improves the scorer. |
| [**Quant finance**](examples/trading/README.md) | A daily trading strategy over synthetic instruments: raise out-of-sample Sharpe on the trading harness, with its checks on every attempt. A research example, not financial advice. |
| [**Tabular ML**](examples/end-to-end/README.md) | A Kaggle competition end to end: Spaceship Titanic from a dummy classifier to a real model, then review, learn, build again with the lesson, and deploy. |
| [**PyTorch optimization**](examples/torch-performance/README.md) | Make a PyTorch module faster on CUDA within a numerical tolerance: fusion, compilation, less memory traffic. |
| [**GPU kernel optimization**](examples/gpu-kernels/README.md) | Speed up multi-head self-attention on CUDA with flash attention or a Triton kernel, within a numerical tolerance. |

## 📚 Documentation & Support

- **Full Documentation**: [docs.leeroo.com](https://docs.leeroo.com/docs)
  - [Installation](https://docs.leeroo.com/docs/installation) — the coding-agent CLIs, the package, and `kapso doctor`
  - [Quickstart](https://docs.leeroo.com/docs/first-build) — your first campaign
  - [CLI reference](https://docs.leeroo.com/docs/reference/cli) — every command, flag and default
  - [Python API](https://docs.leeroo.com/docs/reference/kapso-api) — every public method on `Kapso`
  - [Configuration](https://docs.leeroo.com/docs/reference/configuration) — every key in `config.yaml`
  - [What learn produces](https://docs.leeroo.com/docs/learn/what-learn-produces) — the bank of lessons, each with a credence score, a boundary and its evidence, and [development mode](https://docs.leeroo.com/docs/learn/development-mode)
  - [Smart experimentation](https://docs.leeroo.com/docs/evolve/overview) — the tree of experiments and its modules, and [what happens inside one experiment](https://docs.leeroo.com/docs/evolve/execution-flow)
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

In the enterprise, Kapso agents ship with initial data and AI engineering harnesses and a bank of lessons, and turn business objectives into working AI and data systems on their own, investigating the company's artifacts, repos, pipelines, data sources and more, to build up the enterprise data estate as they go. To onboard Kapso for your enterprise tasks, [talk to us](https://leeroo.com/contact-us).

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details on how to get started.

## Contributors

<a href="https://github.com/Leeroo-AI/kapso/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=Leeroo-AI/kapso" alt="Kapso contributors" />
</a>

Made with [contrib.rocks](https://contrib.rocks).

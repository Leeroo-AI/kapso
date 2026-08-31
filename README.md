<h1 align="center">Kapso</h1>

<h4 align="center">A Self-Improving AI Software Factory (for Measurable Objectives)</h4>

<p align="center">
  <a href="https://docs.leeroo.com">Learn more</a> ·
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

---

## News

- 🏆 **IOAI² Grand Master Trophy at [IOAI 2026](benchmarks/ioai2026/README.md)**: competing fully autonomously in the AI Model Track of the International Olympiad in AI, Kapso surpassed the best human contestant and ranked top 3 among all AI system participants, a field spanning major AI labs and startups.
- **Beats the best foundation model on [RelBench](benchmarks/relbench/README.md)**: on Stanford's benchmark for predictive ML over enterprise data, Kapso passes [KumoRFM-v2](https://docs.nvidia.com/sdgm/rfm/overview) in outcome prediction and forecasting, and the best reported results in recommendations. Published results live on the [official RelBench leaderboard](https://huggingface.co/spaces/relbench/leaderboard).

  <img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/relbench.png" alt="RelBench: Kapso against KumoRFM-v2 on outcome prediction, forecasting and recommendations" width="820">

- **[Leeroopedia MCP Integration](https://leeroopedia.com)**: Kapso now connects to **Leeroopedia MCP** — your ML & Data Knowledge Wiki. Learnt by AI, built by AI, for AI. A centralized playbook of best practices and expert-level knowledge for Machine Learning and Data domains. Kapso agents use it during ideation and implementation to search knowledge, build plans, diagnose failures, and more.
- **[Moltbook Agents 🦞](https://www.moltbook.com/)**: Build AI agents that optimize other agents and debate on Moltbook! [Get started →](moltbook_bot/README.md)
- **Technical Report**: Our technical report is now available! [Read the paper](https://arxiv.org/abs/2601.21526)
- **#1 on [MLE-Bench](benchmarks/mle/README.md)**: KAPSO achieved top ranking among open-source systems on Kaggle ML competitions (MLE Benchmark).

  <img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/mle-bench.png" alt="MLE-Bench: medal rate by difficulty split against R&amp;D-Agent, AIRA-dojo, ML-Master and AIDE" width="820">

- **#1 on [ALE-Bench](benchmarks/ale/README.md)**: KAPSO achieved top ranking on long-horizon algorithmic discovery problems (ALE Benchmark).

  <img src="https://raw.githubusercontent.com/Leeroo-AI/kapso/main/docs/images/ale-bench.png" alt="ALE-Bench: Elo rating per AtCoder Heuristic Contest problem against ALE Agent" width="820">

## What is KAPSO?

KAPSO builds software for goals you can measure.

Give it a goal. It runs **experiment campaigns**: designing candidates, implementing them, scoring each one, and refining the best. Every claim of progress is a score, and the winner ships to your infrastructure.

The factory is **self-improving** because it learns from its own work. Finished campaigns become evidence-priced lessons. It also learns from public knowledge, including repos and papers. Everything compounds in one knowledge hub, so each campaign starts where the last one left off.

### The Four Pillars

| Pillar | Method | Description |
|--------|--------|-------------|
| **Evolve** | `.evolve()` | Run iterative experiments to build software for a goal. Uses tree search, coding agents, and KG context to generate and refine solutions. |
| **Learn** | `.learn()` / `.learn_knowledge()` | Two memories: `learn()` mines your own finished campaigns into evidence-priced knowledge cards (experience); `learn_knowledge()` ingests repositories and research into the Knowledge Graph (imported knowledge). |
| **Research** | `.research()` | Run deep web research to gather ideas and implementation references. Returns structured findings you can feed into the knowledge base or use as context for evolving solutions. |
| **Deploy** | `.deploy()` | Turn a solution into running software. Supports local execution, Docker containers, or cloud platforms like Modal. |

## 🚀 Quickstart

### Installation

**1. Prerequisites.** Kapso runs its inference through coding-agent CLIs
(there is no direct-API fallback), so you need Node.js and both agent
CLIs logged in before anything works:

```bash
# Node.js 18+ (https://nodejs.org), then:
npm install -g @openai/codex            # research, judging, utilities
codex login

npm install -g @anthropic-ai/claude-code  # ideation + implementation (default mode)
claude auth login
```

Add an OpenAI key for embeddings (memory and knowledge-search indexing):

```bash
echo 'OPENAI_API_KEY=sk-...' >> .env
```

**2. Install the package** (Python 3.10+):

```bash
pip install leeroo-kapso
```

**3. Verify the setup:**

```bash
kapso doctor
```

`doctor` reports what **your config** actually needs, and names the exact
fix for anything missing. Requirements follow the config, so an all-codex
setup is never asked for `claude`. Narrow it to one verb to see just that
verb's requirements:

```bash
kapso doctor evolve            # or research / learn_knowledge / learn / deploy
```

Those are the same checks the verb itself runs before it does any work —
`kapso.evolve(...)` fails in seconds on a missing CLI rather than deep
inside a session. Items marked `[-- ]` are optional; they limit features
you may not need (docker, Weaviate, Neo4j, the deploy targets).

**Knowledge-graph backends (optional)** — `learn_knowledge()` and
`kg_index` store into local Weaviate + Neo4j. From a source checkout:

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

**Leeroopedia MCP (optional)** — connect Kapso to [Leeroopedia](https://leeroopedia.com), a curated ML/AI knowledge base. Get an API key from the [Leeroopedia dashboard](https://app.leeroopedia.com/dashboard), then:

```bash
pip install leeroopedia-mcp
echo 'LEEROOPEDIA_API_KEY=kpsk_your_key_here' >> .env
```

### Basic Usage

The core loop needs nothing beyond the prerequisites above:

```python
from kapso import Kapso

kapso = Kapso()   # no knowledge graph needed to start

# Evolve: build a solution through experimentation. The campaign prints
# `status: <path>` at launch — watch it live from another terminal with
#     kapso watch ./campaign
solution = kapso.evolve(
    goal="Optimize the model in train.py; target accuracy > 0.80 on evaluate.py",
    initial_repo="./my_project",         # or omit to start from scratch
    output_path="./campaign",
    time_budget_minutes=120,
)
print(solution.explain())

# Learn from the campaign you just ran: mine the trajectory, grade the
# lessons, and bank evidence-priced knowledge cards. The bank (a local
# git repo) is created automatically on first use — lessons stay on your
# machine until you share them:
#     kapso bank connect <git-url>   # or: kapso bank create org/name
# after which every learn() pushes the bank there.
lesson = kapso.learn(solution)
print(lesson.explain())

# Evolve again — with `learning.serving.enabled: true` in your config,
# the next campaign is served the cards it just earned.
solution2 = kapso.evolve(goal="...", output_path="./campaign2")
```

With the knowledge-graph backends running, you can also import outside
knowledge and serve it to campaigns:

```python
from kapso import Kapso, Source

kapso = Kapso()

# Research the web, then ingest findings + a repository into the KG
findings = kapso.research(
    "RLHF and DPO fine-tuning for legal contract analysis",
    mode=["idea", "implementation"],
)
kapso.learn_knowledge(
    Source.Repo("https://github.com/huggingface/trl"),
    findings.ideas,
    findings.implementations,
    wiki_dir="data/wikis",
)

# Campaigns on this Kapso now consult the knowledge graph automatically
solution = kapso.evolve(goal="Fine-tune Llama-3.1-8B for clause risk classification")
```

A note on budgeting: `depth="light"` bounds the *research* stage only.
`learn_knowledge()` extracts everything the material supports — a small
findings set can still become dozens of linked wiki pages and an
hours-long ingest. Ingest time scales with extractable substance, not
with the depth flag; pass fewer sources when you want a faster ingest.

And to turn a solution into running software:

```python
from kapso import DeployStrategy

deployed = kapso.deploy(solution, strategy=DeployStrategy.LOCAL)
result = deployed.run({"input": "data"})
deployed.stop()
```

### Choosing models

Every model Kapso uses is named in one config file. The packaged default
runs evolve sessions on `claude-opus-5`, the learning crews on
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

kapso = Kapso(config_path="kapso-config.yaml")
```

Before a long run, probe every model your config names against your
actual subscriptions — a capped model fails here in seconds instead of
hours into a run:

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

For detailed integration steps, see the [Quickstart](https://docs.leeroo.com/docs/quickstart) and [Installation](https://docs.leeroo.com/docs/installation) guides.

## Examples

| Example | Description |
|---------|-------------|
| [**CUDA Optimization**](examples/cuda_optimization/README.md) | Optimize CUDA kernels for GPU performance |
| [**PyTorch Optimization**](examples/pytorch_optimization/README.md) | Cut wall-clock and memory — fuse ops, kill sync points and host-device chatter, saturate the GPU without changing numerics |
| [**ML Model Development**](examples/ml_model_development/README.md) | End-to-end delivery of prediction models — data prep, features, training, and validation evolved into a deployable artifact |
| [**Harness Optimization**](examples/prompt_engineering/README.md) | Evolve the harness around a model — prompts, decoding, parsing, and scoring tuned against a measurable target |
| [**Agent Optimization**](examples/agentic_scaffold/README.md) | Agents improving agents — workflows, tools, and prompts evolved until the metric climbs |

## Supported Benchmarks

| Benchmark | Description |
|-----------|-------------|
| [**MLE-Bench**](benchmarks/mle/README.md) | OpenAI's ML-engineering benchmark — full competitions across tabular, vision, text, and audio, from raw data to graded submission |
| [**ALE-Bench**](benchmarks/ale/README.md) | Sakana AI's algorithmic-optimization benchmark — design, implement, and iterate contest heuristics over hours-long searches |
| [**RelBench**](benchmarks/relbench/README.md) | Stanford's benchmark for predictive ML over enterprise data — forecasting, classification, and recommendation straight from the multi-table databases of SAP, Amazon, H&M, and more |
| [**IOAI 2026**](benchmarks/ioai2026/README.md) | Timed olympiad ML across vision, language, and optimization — expert-set tasks, contest hardware, zero human help |

## 📚 Documentation & Support

- **Full Documentation**: [docs.leeroo.com](https://docs.leeroo.com)
- **Community**: [Discord](https://discord.gg/hqVbPNNEZM)
- **Website**: [leeroo.com](https://leeroo.com)


## Kapso for Enterprise

Kapso gets better at your company the longer it works: every task feeds a living knowledge bank of your systems, your data, and your hard-won lessons. To onboard Kapso for your challenging enterprise tasks and build that live company context, [talk to us](https://leeroo.com/contact-us).

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details on how to get started.

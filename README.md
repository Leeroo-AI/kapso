# Praxium

> *A framework where AI Learns, experiments, Builds, and Ships.*

[![Python](https://img.shields.io/badge/python-3.12+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Discord](https://img.shields.io/badge/Discord-Join%20us-5865F2?logo=discord&logoColor=white)](https://discord.gg/hqVbPNNEZM)
[![Y Combinator](https://img.shields.io/badge/Y%20Combinator-X25-orange?logo=ycombinator&logoColor=white)](https://www.ycombinator.com/companies/leeroo)
[![Website](https://img.shields.io/badge/Website-leeroo.com-green)](https://leeroo.com/)
[![GitHub commit activity](https://img.shields.io/github/commit-activity/m/leeroo-ai/praxium)](https://github.com/leeroo-ai/praxium)
[![PyPI version](https://img.shields.io/pypi/v/praxium)](https://pypi.org/project/praxium/)

Praxium lets domain experts (quant, healthcare, data engineering, etc.) turn their knowledge into executable software — without deep engineering expertise.

## Quick Start

```python
from src.tinkerer import Tinkerer, Source, DeployStrategy

# Initialize a Tinkerer (connect to knowledge graph)
tinkerer = Tinkerer(
    domain="machine_learning",
    kg_location="https://skills.leeroo.com",
)

# Teach it from various sources (optional)
tinkerer.learn(
    Source.Repo("https://github.com/scikit-learn/scikit-learn"),
    Source.Paper("./gradient_boosting_techniques.pdf"),
    Source.File("./kaggle_competition_notes.md"),
    target_kg="https://skills.leeroo.com",
)

# Build a solution — the Tinkerer runs experiments automatically
solution = tinkerer.evolve(
    goal="XGBoost classifier for Titanic dataset with AUC > 0.85",
    output_path="./models/titanic_v1",
    evaluator="llm_judge",
    stop_condition="threshold",
    stop_condition_params={"threshold": 0.85},
)

# Deploy and run
software = tinkerer.deploy(solution, strategy=DeployStrategy.LOCAL)
result = software.run({"data_path": "./test.csv"})

# Lifecycle management
software.stop()   # Stop the deployment
software.start()  # Restart the deployment
result = software.run({"data_path": "./validation.csv"})  # Run again
software.stop()   # Final cleanup

# Learn from the experience (feedback loop)
tinkerer.learn(Source.Solution(solution), target_kg="https://skills.leeroo.com")
```

## Usage Examples

### Kaggle Competitor

```python
from src.tinkerer import Tinkerer, Source, DeployStrategy

# 1. Initialize with ML domain knowledge
kaggler = Tinkerer(
    domain="machine_learning",
    kg_location="https://skills.leeroo.com",
)

# 2. Learn from winning solutions and research
kaggler.learn(
    Source.Repo("https://github.com/Kaggle/kaggle-api", branch="main"),
    Source.Paper("./ensemble_methods_survey.pdf"),
    target_kg="https://skills.leeroo.com",
)

# 3. Build the ML pipeline for a tabular competition
solution = kaggler.evolve(
    goal="LightGBM + CatBoost ensemble for House Prices with RMSE < 0.12",
    output_path="./submissions/house_prices_v1",
    max_iterations=10,
    evaluator="regex_pattern",
    evaluator_params={"pattern": r"RMSE: ([\d.]+)"},
    stop_condition="threshold",
    stop_condition_params={"threshold": 0.12},
)

# 4. Deploy and generate predictions
software = kaggler.deploy(solution, strategy=DeployStrategy.LOCAL)
result = software.run({"train_path": "./train.csv", "test_path": "./test.csv"})

# 5. Learn from the build experience
kaggler.learn(Source.Solution(solution), target_kg="https://skills.leeroo.com")
```

### Data Engineer

```python
from src.tinkerer import Tinkerer, Source, DeployStrategy

# 1. Initialize in data engineering domain
data_eng = Tinkerer(
    domain="data_engineering",
    kg_location="https://skills.leeroo.com",
)

# 2. Build with data directory (schemas and configs available during build)
etl_pipeline = data_eng.evolve(
    goal="ETL pipeline with PySpark that processes 1M+ rows/min from S3 to Snowflake",
    data_dir="./schemas/",
    output_path="./pipelines/s3_to_snowflake_v1",
    evaluator="llm_judge",
    evaluator_params={"criteria": "data_quality_and_throughput"},
    stop_condition="plateau",
    stop_condition_params={"patience": 3},
)

# 3. Deploy to cloud
software = data_eng.deploy(etl_pipeline, strategy=DeployStrategy.MODAL)
result = software.run({"s3_bucket": "raw-data", "table": "events"})

# 4. Learn from the successful build
data_eng.learn(Source.Solution(etl_pipeline), target_kg="https://skills.leeroo.com")
```

## CLI Usage

```bash
# Basic usage
PYTHONPATH=. python -m src.cli --goal "Build a random forest classifier for the Iris dataset"

# With options
PYTHONPATH=. python -m src.cli \
    --goal "Build a feature engineering pipeline for tabular data" \
    --iterations 10 \
    --language python \
    --coding-agent aider \
    --evaluator regex_pattern \
    --stop-condition threshold

# List available options
PYTHONPATH=. python -m src.cli --list-agents
PYTHONPATH=. python -m src.cli --list-evaluators
PYTHONPATH=. python -m src.cli --list-stops
```

## Evaluators

Control how solutions are scored:

| Evaluator | Description |
|-----------|-------------|
| `no_score` | No scoring, always returns 0 (default) |
| `regex_pattern` | Extract score from output using regex |
| `file_json` | Read score from a JSON file |
| `multi_metric` | Weighted combination of multiple regex metrics |
| `llm_judge` | LLM-based evaluation with custom criteria |
| `llm_comparison` | Compare output against expected result |
| `composite` | Combine multiple evaluators |

```python
# Regex example
solution = tinkerer.evolve(
    goal="...",
    evaluator="regex_pattern",
    evaluator_params={"pattern": r"Accuracy: ([\d.]+)%"},
)

# LLM judge example
solution = tinkerer.evolve(
    goal="...",
    evaluator="llm_judge",
    evaluator_params={"criteria": "correctness and efficiency"},
)
```

## Stop Conditions

Control when to stop experimentation:

| Condition | Description |
|-----------|-------------|
| `never` | Never stop early (default) |
| `threshold` | Stop when score reaches threshold |
| `max_iterations` | Stop after N iterations |
| `plateau` | Stop if no improvement for N iterations |
| `cost_limit` | Stop when cost limit reached |
| `time_limit` | Stop when time limit reached |
| `consecutive_errors` | Stop after N consecutive errors |
| `composite` | Combine multiple conditions (any/all) |

```python
# Threshold example
solution = tinkerer.evolve(
    goal="...",
    stop_condition="threshold",
    stop_condition_params={"threshold": 0.95},
)

# Composite: stop if score >= 0.9 OR no improvement for 5 iterations
solution = tinkerer.evolve(
    goal="...",
    stop_condition="composite",
    stop_condition_params={
        "conditions": [
            ("threshold", {"threshold": 0.9}),
            ("plateau", {"patience": 5}),
        ],
        "mode": "any",
    },
)
```

## Coding Agents

Pluggable agents for code generation:

| Agent | Description |
|-------|-------------|
| `aider` | Git-centric pair programming with diff-based editing (default) |
| `gemini` | Google Gemini SDK for code generation |
| `claude_code` | Anthropic Claude Code CLI for complex refactoring |
| `openhands` | OpenHands agent with sandboxed execution ⚠️ |

> **⚠️ Note:** `openhands` requires a separate conda environment due to conflicting dependencies with `aider-chat`. See [Installation](#installation) for details.

```python
solution = tinkerer.evolve(
    goal="...",
    coding_agent="claude_code",  # Use Claude Code instead of default
)
```

## Search Strategies

Control how solutions are explored:

| Strategy | Description |
|----------|-------------|
| `llm_tree_search` | Tree-based exploration with LLM-guided selection (default) |
| `linear_search` | Simple sequential search, one solution per iteration |

Presets: `PRODUCTION`, `HEAVY_EXPERIMENTATION`, `HEAVY_THINKING`, `MINIMAL`

## Architecture

```
praxium/
├── src/
│   ├── tinkerer.py              # Main Tinkerer API
│   ├── cli.py                 # CLI entry point
│   ├── core/                  # LLM backend, config
│   ├── deployment/            # Local, Docker, Cloud deployment
│   ├── environment/
│   │   ├── evaluators/        # Pluggable scoring
│   │   ├── handlers/          # Problem handlers
│   │   └── stop_conditions/   # Pluggable stop conditions
│   ├── execution/
│   │   ├── coding_agents/     # Aider, Gemini, Claude Code, OpenHands
│   │   ├── search_strategies/ # Tree search, Linear search
│   │   └── orchestrator.py    # Main coordination
│   └── knowledge/
│       ├── learners/          # Repo, Paper, File learners
│       └── search/            # Knowledge graph search
├── benchmarks/
│   ├── mle/                   # MLE-Bench (Kaggle)
│   └── ale/                   # ALE-Bench (AtCoder)
└── docs/                      # Documentation
```

## Installation

### 1. Clone and Install

```bash
git clone <repository-url>
cd praxium

# Pull Git LFS files (wiki knowledge data)
git lfs install
git lfs pull

# Create a dedicated conda environment (recommended)
conda create -n praxium_conda python=3.12
conda activate praxium_conda

# Install the package
pip install -e .
```

> **📦 Git LFS Note**
>
> This repository uses Git LFS for large files in `data/wikis_batch_top100/`.
> If you didn't install Git LFS before cloning, run:
> ```bash
> git lfs install
> git lfs pull
> ```

> **⚠️ Dependency Compatibility Note**
>
> `aider-chat` has strict pinned dependencies that conflict with some packages:
> - `openhands` requires `litellm>=1.80.7`, but aider pins `litellm==1.75.0`
> - `browser-use` requires `openai>=2.7.2`, but aider pins `openai==1.99.1`
>
> **Do not install openhands or browser-use in the same environment.**
> Use a separate conda environment if you need those packages.

### 2. Set Up API Keys

Create `.env` in project root:

```bash
OPENAI_API_KEY=your-openai-api-key       # For GPT models
GOOGLE_API_KEY=your-google-api-key       # For Gemini models
ANTHROPIC_API_KEY=your-anthropic-api-key # For Claude models
```

### 3. Install Coding Agent (optional)

```bash
# Aider (default) - included in pip install -e .
# Already installed as a core dependency

# Claude Code
npm install -g @anthropic-ai/claude-code

# OpenHands - INCOMPATIBLE with aider-chat (use separate environment)
# conda create -n openhands_env python=3.12
# conda activate openhands_env
# pip install openhands-ai litellm
```

## Running Benchmarks

### MLE-Bench (Kaggle)

```bash
# Install MLE-Bench
git clone https://github.com/openai/mle-bench.git
cd mle-bench && pip install -e . && cd ..
pip install -r benchmarks/mle/requirements.txt

# Run
PYTHONPATH=. python -m benchmarks.mle.runner
```

### ALE-Bench (AtCoder)

```bash
# Install ALE-Bench
git clone https://github.com/SakanaAI/ALE-Bench.git
cd ALE-Bench && pip install . && pip install ".[eval]" && bash ./scripts/docker_build_202301.sh 1000 1000 && cd ..

# Run
PYTHONPATH=. python -m benchmarks.ale.runner
```

## Core Components

| Component | Description |
|-----------|-------------|
| `Tinkerer` | Main API - learn, evolve, deploy |
| `OrchestratorAgent` | Coordinates experimentation loop |
| `SearchStrategy` | Tree/Linear search for solutions |
| `CodingAgent` | Code generation (Aider, Gemini, etc.) |
| `Evaluator` | Score solutions |
| `StopCondition` | Control when to stop |
| `ProblemHandler` | Problem-specific logic |

## Software Lifecycle

After deploying, the returned `Software` instance supports full lifecycle management:

```python
software = tinkerer.deploy(solution, strategy=DeployStrategy.LOCAL)

# Run inference
result = software.run({"data_path": "./test_features.csv"})

# Check health
if software.is_healthy():
    print("Model service is running")

# Stop (cleanup resources)
software.stop()

# Restart (re-initialize)
software.start()

# Run again after restart
result = software.run({"data_path": "./new_batch.csv"})

# Final cleanup
software.stop()
```

| Method | Description |
|--------|-------------|
| `run(inputs)` | Execute with input data, returns `{"status": "success/error", "output": ...}` |
| `stop()` | Stop and cleanup resources (containers, modules, cloud deployments) |
| `start()` | Restart a stopped deployment (re-deploy for cloud, reload for local) |
| `is_healthy()` | Check if the service is running and ready |
| `logs()` | Get execution logs for debugging |

## Running Documentation Locally

To run the Mintlify documentation locally:

```bash
# Install Mintlify CLI (if not already installed)
npm i -g mintlify

# Navigate to the docs directory
cd docs

# Start the local development server
mintlify dev
```

The documentation will be available at `http://localhost:3000` by default.

## License

MIT

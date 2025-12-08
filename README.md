# Praxium

A framework where AI LeArns, experiments, Builds, and Ships.

The Expert Agent lets domain experts (quant, healthcare, data engineering, etc.) turn their knowledge into executable software — without deep engineering expertise.

## Quick Start

```python
from src.expert import Expert, Source, DeployStrategy

# Initialize an Expert (connect to knowledge graph)
expert = Expert(
    domain="quantitative_finance",
    kg_location="https://skills.leeroo.com",
)

# Teach it from various sources (optional)
expert.learn(
    Source.Repo("https://github.com/alpaca/alpaca-py"),
    Source.Paper("./momentum_strategies.pdf"),
    Source.File("./my_notes.md"),
    target_kg="https://skills.leeroo.com",
)

# Build a solution — the Expert runs experiments automatically
solution = expert.build(
    goal="Momentum trading bot for SP500 with latency < 50ms",
    output_path="./bots/sp500_v1",
    evaluator="llm_judge",
    stop_condition="threshold",
    stop_condition_params={"threshold": 0.9},
)

# Deploy and run
software = expert.deploy(solution, strategy=DeployStrategy.LOCAL)
result = software.run({"ticker": "AAPL", "price": 150.0})
software.stop()

# Learn from the experience (feedback loop)
expert.learn(Source.Solution(solution), target_kg="https://skills.leeroo.com")
```

## Usage Examples

### Quant Trader

```python
from src.expert import Expert, Source, DeployStrategy

# 1. Initialize with finance domain knowledge
quant = Expert(
    domain="quantitative_finance",
    kg_location="https://skills.leeroo.com",
)

# 2. Learn from repos and papers
quant.learn(
    Source.Repo("https://github.com/alpaca/alpaca-py", branch="main"),
    Source.Paper("./momentum_strategies.pdf"),
    target_kg="https://skills.leeroo.com",
)

# 3. Build the trading bot
solution = quant.build(
    goal="Momentum Bot for SP500 with max drawdown < 2%",
    output_path="./bots/sp500_v1",
    max_iterations=10,
    evaluator="regex_pattern",
    evaluator_params={"pattern": r"Sharpe: ([\d.]+)"},
    stop_condition="threshold",
    stop_condition_params={"threshold": 1.5},
)

# 4. Deploy and run
software = quant.deploy(solution, strategy=DeployStrategy.LOCAL)
result = software.run({"ticker": "AAPL", "price": 150.0})

# 5. Learn from the build experience
quant.learn(Source.Solution(solution), target_kg="https://skills.leeroo.com")
```

### Healthcare / GP

```python
from src.expert import Expert, Source, DeployStrategy

# 1. Initialize in healthcare domain
doctor_ai = Expert(
    domain="healthcare",
    kg_location="https://skills.leeroo.com",
)

# 2. Build with data directory (protocols available during build)
triage_agent = doctor_ai.build(
    goal="Voice Triage Agent that follows the protocols in data_dir",
    data_dir="./dr_smith_protocols/",
    output_path="./agents/triage_v1",
    evaluator="llm_judge",
    evaluator_params={"criteria": "medical_accuracy"},
    stop_condition="plateau",
    stop_condition_params={"patience": 3},
)

# 3. Deploy to cloud
software = doctor_ai.deploy(triage_agent, strategy=DeployStrategy.MODAL)
diagnosis = software.run({"audio": "<binary_audio_data>", "patient_id": "12345"})

# 4. Learn from the successful build
doctor_ai.learn(Source.Solution(triage_agent), target_kg="https://skills.leeroo.com")
```

## CLI Usage

```bash
# Basic usage
PYTHONPATH=. python -m src.cli --goal "Build a REST API for user authentication"

# With options
PYTHONPATH=. python -m src.cli \
    --goal "Build a data pipeline" \
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
solution = expert.build(
    goal="...",
    evaluator="regex_pattern",
    evaluator_params={"pattern": r"Accuracy: ([\d.]+)%"},
)

# LLM judge example
solution = expert.build(
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
solution = expert.build(
    goal="...",
    stop_condition="threshold",
    stop_condition_params={"threshold": 0.95},
)

# Composite: stop if score >= 0.9 OR no improvement for 5 iterations
solution = expert.build(
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
| `openhands` | OpenHands agent with sandboxed execution |

```python
solution = expert.build(
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
mle_expert_coding/
├── src/
│   ├── expert.py              # Main Expert API
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
cd mle_expert_coding
pip install -r requirements.txt
```

### 2. Set Up API Keys

Create `.env` in project root:

```bash
OPENAI_API_KEY=your-openai-api-key       # For GPT models
GOOGLE_API_KEY=your-google-api-key       # For Gemini models
ANTHROPIC_API_KEY=your-anthropic-api-key # For Claude models
```

### 3. Install Coding Agent (optional)

```bash
# Aider (default)
pip install aider-chat

# Claude Code
npm install -g @anthropic-ai/claude-code

# OpenHands
pip install openhands-ai litellm
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
cd ALE-Bench && pip install . && pip install ".[eval]" && cd ..

# Run
PYTHONPATH=. python -m benchmarks.ale.runner
```

## Core Components

| Component | Description |
|-----------|-------------|
| `Expert` | Main API - learn, build, deploy |
| `OrchestratorAgent` | Coordinates experimentation loop |
| `SearchStrategy` | Tree/Linear search for solutions |
| `CodingAgent` | Code generation (Aider, Gemini, etc.) |
| `Evaluator` | Score solutions |
| `StopCondition` | Control when to stop |
| `ProblemHandler` | Problem-specific logic |

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

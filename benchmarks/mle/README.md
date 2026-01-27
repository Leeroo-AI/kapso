# MLE-Bench Integration

This module provides integration with [MLE-Bench](https://github.com/openai/mle-bench), OpenAI's benchmark for evaluating ML agents on Kaggle competitions.

Kapso achieved **#1 among open-source systems** on this benchmark. These results were submitted as an [official submission to MLE-Bench](https://github.com/openai/mle-bench/pull/107).

![MLE-Bench Results](https://api.leeroo.com/storage/v1/object/public/opensource/mle_benchmark.png)

## Prerequisites

Before installing MLE-Bench, ensure you have:

1. **Core Kapso Agent installed** (from repository root):
   ```bash
   pip install -r requirements.txt
   ```

2. **API Keys configured** in `.env` or environment:
   ```bash
   OPENAI_API_KEY=your-openai-api-key
   GOOGLE_API_KEY=your-google-api-key
   ```

3. **Git LFS installed**:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install git-lfs
   
   # macOS
   brew install git-lfs
   ```

## Installation

### Step 1: Clone and Install MLE-Bench

```bash
# Clone the repository
git clone https://github.com/openai/mle-bench.git
cd mle-bench

# Initialize Git LFS and fetch large files (datasets)
# First install git-lfs if not already installed:
#   Ubuntu/Debian: sudo apt-get install git-lfs
#   macOS: brew install git-lfs
git lfs install
git lfs fetch --all
git lfs pull

# Install the package
pip install -e .

# Return to Kapso directory
cd ..
```

### Step 2: Install MLE-specific Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: (Optional) Setup Neo4j Knowledge Graph

For enhanced ML domain knowledge:

```bash
# Start Neo4j container
docker run -d \
    --name neo4j \
    --restart unless-stopped \
    -p 7474:7474 -p 7687:7687 \
    -e NEO4J_AUTH=neo4j/password \
    neo4j:latest

# Load knowledge graph data
PYTHONPATH=. python src/agents/wiki_agent/kg_agent/kg_agent.py
```

## Usage

```bash
# List available competitions
PYTHONPATH=. python -m benchmarks.mle.runner --list

# List lite benchmark competitions
PYTHONPATH=. python -m benchmarks.mle.runner --lite

# Solve a competition
PYTHONPATH=. python -m benchmarks.mle.runner -c tabular-playground-series-dec-2021

# With options
PYTHONPATH=. python -m benchmarks.mle.runner \
    -c tabular-playground-series-dec-2021 \
    -i 20 \
    -m MLE_CONFIGS \
    -d aider
```

## CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `-c, --competition` | Competition ID | Required |
| `-i, --iterations` | Max experiment iterations | 20 |
| `-m, --mode` | Config mode | `MLE_CONFIGS` |
| `-d, --coding-agent` | Coding agent | From config |
| `--no-kg` | Disable knowledge graph | Enabled |
| `--list` | List all competitions | - |
| `--lite` | List lite competitions | - |
| `--list-agents` | List coding agents | - |

## Stages

The handler automatically adjusts strategy based on budget progress:

| Stage | Budget | Behavior |
|-------|--------|----------|
| **MINI TRAINING** | 0-35% | Sample training data (for datasets >30GB) |
| **FULL TRAINING** | 35-80% | Train on complete dataset |
| **FINAL ENSEMBLING** | 80-100% | Ensemble best models from history |

## Output Structure

The agent generates:

```
experiment_workspace/{uuid}/
├── main.py                    # Entry point
├── output_data_{branch}/
│   ├── final_submission.csv   # Kaggle submission file
│   └── checkpoints/           # Model checkpoints
└── sessions/                  # Experiment branches
```

## Code Requirements

Generated code must:
- Support `--debug` flag for fast testing
- Write `final_submission.csv` in the output directory
- Print progress and metrics
- Handle GPU efficiently (batch size, device selection)
- Use early stopping and learning rate scheduling

## Competition Types

| Type | Examples |
|------|----------|
| Tabular | `tabular-playground-series-*` |
| Image | `dogs-vs-cats-*`, `plant-pathology-*` |
| Text | `spooky-author-identification`, `jigsaw-toxic-*` |
| Audio | `mlsp-2013-birds` |

Use `mlebench.registry.registry.get_lite_competition_ids()` for the lightweight benchmark.

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key (required) | - |
| `GOOGLE_API_KEY` | Google API key for Gemini (required) | - |
| `CUDA_DEVICE` | GPU device ID | `0` |
| `MLE_SEED` | Random seed | `1` |
| `NEO4J_URI` | Neo4j connection URI | `bolt://localhost:7687` |
| `NEO4J_USER` | Neo4j username | `neo4j` |
| `NEO4J_PASSWORD` | Neo4j password | `password` |

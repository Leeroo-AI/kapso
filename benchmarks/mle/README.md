# MLE-Bench Integration

This module provides integration with [MLE-Bench](https://github.com/openai/mle-bench), OpenAI's benchmark for evaluating ML agents on Kaggle competitions.

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
git lfs install
git lfs fetch --all
git lfs pull

# Install the package
pip install -e .

# Return to expert agent directory
cd ..
```

### Step 2: Install MLE-specific Dependencies

```bash
pip install -r benchmarks/mle/requirements.txt
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

### Run MLE-Bench

```bash
cd /path/to/mle_expert_coding
PYTHONPATH=. python -m benchmarks.mle.runner
```

### Configure Competition

Edit `benchmarks/mle/runner.py` to select which competition to run:

```python
problems_list = [
    'spooky-author-identification',  # Text classification (small)
    'leaf-classification',            # Image classification
    # ... more competitions
]
```

## Available Competitions

MLE-Bench includes competitions across various ML domains:

- **Tabular**: `tabular-playground-series-*`
- **Image**: `dogs-vs-cats-*`, `plant-pathology-*`, `cassava-*`
- **Text**: `spooky-author-identification`, `jigsaw-toxic-*`
- **Audio**: `mlsp-2013-birds`

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

## Output

The agent generates Python solutions that produce `final_submission.csv` files, which are graded against the competition's private test set.

# ML Model Development Example (Kaggle Spaceship Titanic)

This example demonstrates using Kapso's `evolve()` function to iteratively improve a machine learning model for the Kaggle Spaceship Titanic competition.

## Environment Setup

Before running this example, install the required dependencies:

```bash
# Create and activate conda environment (recommended)
conda create -n kapso python=3.10
conda activate kapso

# Install ML dependencies
pip install pandas numpy scikit-learn torch xgboost lightgbm catboost

# Install Kapso from the project root
cd /path/to/kapso
pip install -e .
```

Authenticate the coding-agent CLIs the default config uses, then
verify everything with `kapso doctor`:

```bash
# Claude Code sessions (ideation, implementation): log in once
npm install -g @anthropic-ai/claude-code
claude auth login   # or: export CLAUDE_CODE_OAUTH_TOKEN=...

# Codex sessions (research, judging, utilities): log in once
npm install -g @openai/codex
codex login

# embeddings (memory and knowledge-search indexing)
export OPENAI_API_KEY=your_key_here

kapso doctor   # checks all of the above and prints fixes for misses
```

## Problem Description

The Spaceship Titanic competition asks you to predict which passengers were transported to an alternate dimension during a collision with a spacetime anomaly.

The baseline implementation (`train.py`) uses a simple DummyClassifier. The goal is to optimize feature engineering, model selection, and hyperparameters to improve accuracy.

### Constraints

- Must maintain the same function signatures (`train_model`, `predict_with_model`)
- Must work with the provided CSV data format
- Must produce valid submission DataFrame

## Data Setup

Download the Spaceship Titanic data from Kaggle:

```bash
# Using Kaggle CLI
kaggle competitions download -c spaceship-titanic
unzip spaceship-titanic.zip -d initial_repo/data/
```

Or manually download from: https://www.kaggle.com/competitions/spaceship-titanic/data

Place `train.csv` and `test.csv` in the `initial_repo/data/` directory.

## Usage

### Run Kapso Evolution

```bash
cd examples/ml_model_development
python run_evolve.py
```

This will:
1. Initialize Kapso
2. Run multiple iterations to find optimized implementations
3. Output the best solution to `./model_optimized`

Watch a running campaign live from another terminal:

```bash
kapso watch ./model_optimized
```

### Run the full production loop

This example is also the reference for Kapso's complete loop — web
research → `learn_knowledge()` → `evolve()` → `learn()` → evolve again,
served the lessons it just earned (requires the KG backends from
`scripts/start_infra.sh`):

```bash
python run_full_loop.py            # ~5h end to end; every stage logged
python resume_full_loop.py --from learn   # re-enter a finished stage
```

### Manual Evaluation

To evaluate a specific implementation:

```bash
cd initial_repo
python evaluate.py --data-dir ./data --seed 0
```

## Success Criteria

- **Accuracy**: Higher is better (baseline ~0.50)
- **Target**: 0.78+ accuracy through improved modeling
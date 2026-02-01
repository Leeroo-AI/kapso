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

### Manual Evaluation

To evaluate a specific implementation:

```bash
cd initial_repo
python evaluate.py --data-dir ./data --seed 0
```

## Success Criteria

- **Accuracy**: Higher is better (baseline ~0.50)
- **Target**: 0.78+ accuracy through improved modeling
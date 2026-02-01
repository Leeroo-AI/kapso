"""
ML Model Development Example using Kapso

This example demonstrates using Kapso's evolve() function to iteratively
improve a machine learning model for the Kaggle Spaceship Titanic competition.

The goal is to optimize feature engineering, model selection, and hyperparameters
to achieve higher accuracy.
"""

import os
import shutil
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from kapso import Kapso


def main():
    # Initialize Kapso
    kapso = Kapso()
    
    # Define the optimization goal
    goal = """
Optimize the ML model in `train.py` to improve accuracy on the Spaceship Titanic dataset.

## Target File
The file `train.py` contains:
- `train_model()`: Function that trains and returns a model
- `predict_with_model()`: Function that makes predictions with the trained model

IMPORTANT: Do NOT change function names or signatures. Only modify internal implementation.

## Data
The data/ directory contains:
- train.csv: Training data with features and Transported target
- test.csv: Test data for final predictions

## Evaluation
Run: python evaluate.py --data-dir ./data --seed 0

The evaluation:
- Splits training data into train/validation (90/10)

## Success Criteria
- Accuracy: Higher is better (baseline ~0.50 with DummyClassifier)
- Target: 0.78+ accuracy through improved modeling
"""

    # Get the directory containing the starter code
    initial_repo_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "initial_repo")
    output_path = "./examples/ml_model_development/model_optimized"
    
    # Clean up previous workspace if it exists
    if os.path.exists(output_path):
        print(f"Cleaning up previous workspace: {output_path}")
        shutil.rmtree(output_path)
    
    # Run evolve to optimize the ML model
    solution = kapso.evolve(
        goal=goal,
        # Start from the initial repo
        initial_repo=initial_repo_dir,
        # Output the optimized solution
        output_path=output_path,
    )
    
    print("\n" + "=" * 60)
    print("ML Model Development Optimization Complete!")
    print("=" * 60)
    print(f"\nFinal score (accuracy): {solution.final_score}")
    print(f"Goal achieved: {solution.succeeded}")
    print(f"\nTo evaluate the result:")
    print(f"  cd {solution.code_path}")
    print(f"  python evaluate.py --data-dir ./data --seed 0")


if __name__ == "__main__":
    main()

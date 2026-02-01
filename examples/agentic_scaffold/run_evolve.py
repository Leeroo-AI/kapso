"""
Agentic Scaffold Example using Kapso

This example demonstrates using Kapso's evolve() function to optimize
an AI workflow that extracts tabular data from chart images using a
Vision Language Model (VLM).

The goal is to improve the extraction accuracy by optimizing the prompt
and extraction logic.
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
Optimize the VLM chart data extraction in `optimize.py` to improve accuracy.

## Constraints
- Must output valid CSV format with header row
- Must work with OpenAI vision-capable models
- Average cost per query should stay reasonable

## Evaluation
Run: python evaluate.py --max-samples 100 --num-workers 50

The evaluation:
- Loads chart images from subset_line_100/
- Calls VLMExtractor.image_to_csv() for each image
- Compares to ground truth CSV tables
- Reports accuracy (0.0 to 1.0)

## Accuracy Metric
- Header Match (20%): Exact match of column headers
- Content Similarity (80%): SMAPE-based similarity for numeric values

## Success Criteria
- Accuracy: Higher is better (baseline ~0.30-0.35)
- Target: 0.50+ accuracy through improved extraction
"""

    # Get the directory containing the starter code
    initial_repo_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "initial_repo")
    output_path = "./examples/agentic_scaffold/agentic_optimized"
    
    # Clean up previous workspace if it exists
    if os.path.exists(output_path):
        print(f"Cleaning up previous workspace: {output_path}")
        shutil.rmtree(output_path)
    
    # Run evolve to optimize the VLM extractor
    solution = kapso.evolve(
        goal=goal,
        # Start from the initial repo
        initial_repo=initial_repo_dir,
        # Output the optimized solution
        output_path=output_path,
    )
    
    print("\n" + "=" * 60)
    print("Agentic Scaffold Optimization Complete!")
    print("=" * 60)
    print(f"\nFinal score (accuracy): {solution.final_score}")
    print(f"Goal achieved: {solution.succeeded}")
    print(f"\nTo evaluate the result:")
    print(f"  cd {solution.code_path}")
    print(f"  python evaluate.py --max-samples 100 --num-workers 50")


if __name__ == "__main__":
    main()

"""
Prompt Engineering Example using Kapso

This example demonstrates using Kapso's evolve() function to iteratively
improve a prompt for solving AIME (American Invitational Mathematics
Examination) problems.

The goal is to optimize the prompt template to achieve higher accuracy
on math competition problems.
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
Optimize the prompt template in `optimize.py` to improve accuracy on AIME math problems.

## Target File
The file `optimize.py` contains:
- `PROMPT_TEMPLATE`: The prompt that instructs the LLM how to solve AIME problems
- `solve()`: Function that uses the prompt to get solutions from the LLM

## Constraints
- Must produce answers in \\boxed{XXX} format (3-digit integer 000-999)
- Only modify the prompt template and related logic in optimize.py
- Do not modify the evaluation script

## Success Criteria
- Accuracy: Higher is better (baseline ~0.10-0.20)
- Target: 0.30-0.50 accuracy through improved prompting
"""

    # Get the directory containing the starter code
    initial_repo_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "initial_repo")
    output_path = "./examples/prompt_engineering/prompt_optimized"
    
    # Clean up previous workspace if it exists
    if os.path.exists(output_path):
        print(f"Cleaning up previous workspace: {output_path}")
        shutil.rmtree(output_path)
    
    # Run evolve to optimize the prompt
    solution = kapso.evolve(
        goal=goal,
        # Start from the initial repo
        initial_repo=initial_repo_dir,
        # Output the optimized solution
        output_path=output_path,
    )
    
    print("\n" + "=" * 60)
    print("Prompt Engineering Optimization Complete!")
    print("=" * 60)
    print(f"\nFinal score (accuracy): {solution.final_score}")
    print(f"Goal achieved: {solution.succeeded}")
    print(f"\nTo evaluate the result:")
    print(f"  cd {solution.code_path}")
    print(f"  python evaluate.py")


if __name__ == "__main__":
    main()

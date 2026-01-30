"""
CUDA Optimization Example using Kapso

This example demonstrates using Kapso's evolve() function to optimize
a vanilla multi-head self-attention implementation for CUDA performance.

The goal is to achieve speedup over the baseline PyTorch implementation
while maintaining numerical correctness.
"""

import os
import shutil
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from kapso.kapso import Kapso


def main():
    # Initialize Kapso
    # No KG index needed for this example - we'll use web research
    kapso = Kapso()
    
    # Define the optimization goal
    goal = """
Optimize the multi-head self-attention implementation in `module.py` for maximum speedup on CUDA.

## Constraints
- Must maintain the same Model class interface (same __init__ and forward signatures)
- Must produce numerically correct results (max float diff < 1e-5)
- Must work on CUDA devices

## Evaluation
Run: python evaluate.py --path module.py

## Success Criteria
- Numerical correctness: max diff < 1e-5
- Performance: speedup > 1.0x (higher score = better)

## Environment
Use the `kapso` conda environment which has PyTorch, CUDA, and Triton pre-installed:
  conda activate kapso
"""

    # Optional: Research CUDA optimization techniques first
    # Uncomment to enable web research for optimization ideas
    # findings = kapso.research(
    #     "Flash Attention Triton kernel optimization techniques",
    #     mode=["idea", "implementation"],
    #     top_k=5,
    # )
    # context = [findings.to_string()]
    
    # Get the directory containing the starter code
    initial_repo_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "initial_repo")
    output_path = "./examples/cuda_optimization/cuda_optimized"
    
    # Clean up previous workspace if it exists
    if os.path.exists(output_path):
        print(f"Cleaning up previous workspace: {output_path}")
        shutil.rmtree(output_path)
    
    # Run evolve to optimize the CUDA kernel
    solution = kapso.evolve(
        goal=goal,
        # Start from the initial repo
        initial_repo=initial_repo_dir,
        # Output the optimized solution
        output_path=output_path,
        # Optional context from research
        # context=context,
    )
    
    print("\n" + "=" * 60)
    print("CUDA Optimization Complete!")
    print("=" * 60)
    print(f"\nFinal score (speedup): {solution.final_score}")
    print(f"Goal achieved: {solution.succeeded}")
    print(f"\nTo evaluate the result:")
    print(f"  cd {solution.code_path}")
    print(f"  python evaluate.py --path module.py")


if __name__ == "__main__":
    main()

"""
CUDA Optimization Example using Kapso

This example demonstrates using Kapso's evolve() function to optimize
a vanilla multi-head self-attention implementation for CUDA performance.

The goal is to achieve speedup over the baseline PyTorch implementation
while maintaining numerical correctness.

Based on: https://github.com/WecoAI/weco-cli/tree/main/examples/cuda
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.kapso import Kapso


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
    
    # Run evolve to optimize the CUDA kernel
    solution = kapso.evolve(
        goal=goal,
        # Seed from the initial repo
        seed_repo=initial_repo_dir,
        # Output the optimized solution
        output_path="./examples/cuda_optimization/cuda_optimized",
        # Optional context from research
        # context=context,
    )
    
    print("\n" + "=" * 60)
    print("CUDA Optimization Complete!")
    print("=" * 60)
    print(f"\nBest solution branch: {solution.best_branch}")
    print(f"Best score (speedup): {solution.best_score}")
    print(f"\nTo evaluate the result:")
    print(f"  cd {solution.output_path}")
    print(f"  python evaluate.py --path module.py")


if __name__ == "__main__":
    main()

# Test for kapso.evolve() with script evaluator
#
# This test demonstrates the new default behavior:
# - evaluator="script" (agent writes evaluate.py)
# - stop_condition="from_eval" (agent decides when to stop)
#
# The goal is simple: train an iris classifier with accuracy > 0.90

import pytest
from src.kapso import Kapso


@pytest.mark.skipif(True, reason="Run manually with: pytest tests/test_evolve.py -v -s")
def test_evolve_iris_classifier():
    """
    Test evolve with a simple ML task: Iris classification.
    
    The agent should:
    1. Create main.py that trains a classifier
    2. Create evaluate.py that computes accuracy and signals STOP when > 0.90
    """
    kapso = Kapso()
    
    # Simple goal - no evaluator or stop_condition needed!
    # The agent will write evaluate.py that handles both.
    solution = kapso.evolve(
        goal="Train an Iris flower classifier using scikit-learn. Target accuracy > 0.90 on test set.",
        output_path="./experiments/iris_classifier",
        max_iterations=5,
    )
    
    print(f"\nSolution at: {solution.code_path}")
    print(f"Experiments: {len(solution.experiment_logs)}")
    print(f"Metadata: {solution.metadata}")


def test_evolve_simple_math():
    """
    Simpler test: solve a math optimization problem.
    
    The agent should:
    1. Create main.py that finds the minimum of a function
    2. Create evaluate.py that scores how close to the minimum
    """
    kapso = Kapso()
    
    solution = kapso.evolve(
        goal="""
        Find the minimum of the Rosenbrock function: f(x,y) = (1-x)^2 + 100*(y-x^2)^2
        
        The global minimum is at (1, 1) where f(1,1) = 0.
        
        Your solution should:
        1. Implement an optimization algorithm (gradient descent, scipy.optimize, etc.)
        2. Print the found minimum point (x, y) and function value
        
        Target: Find a point where f(x,y) < 0.001
        """,
        output_path="./experiments/rosenbrock_min",
        max_iterations=3,
    )
    
    print(f"\nSolution at: {solution.code_path}")
    print(f"Experiments: {len(solution.experiment_logs)}")
    print(f"Final evaluation: {solution.metadata.get('final_evaluation')}")


if __name__ == "__main__":
    # Run the simpler test directly
    test_evolve_iris_classifier()

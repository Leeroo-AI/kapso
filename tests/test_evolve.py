# Live tests for kapso.evolve().
#
# Both spawn real coding-agent sessions and spend real subscription quota, so
# the whole module carries the `live` marker and is skipped unless --run-live
# is passed. test_evolve_simple_math previously carried no guard at all and
# ran a three-iteration campaign on every `pytest tests/` — the exact failure
# conftest.py's live marker was written to prevent.
#
# Run with:
#   pytest tests/test_evolve.py --run-live -v -s

import pytest

from kapso import Kapso

pytestmark = pytest.mark.live


def test_evolve_iris_classifier(tmp_path):
    """Evolve a simple ML task: Iris classification.

    The coding agent writes both main.py and the evaluation; the feedback
    generator decides when the goal is met.
    """
    kapso = Kapso()

    solution = kapso.evolve(
        goal=(
            "Train an Iris flower classifier using scikit-learn. "
            "Target accuracy > 0.90 on test set."
        ),
        output_path=str(tmp_path / "iris_classifier"),
        max_iterations=5,
    )

    assert solution.code_path
    assert solution.experiment_logs


def test_evolve_simple_math(tmp_path):
    """Evolve a numeric optimization task with a checkable target."""
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
        output_path=str(tmp_path / "rosenbrock_min"),
        max_iterations=3,
    )

    assert solution.code_path
    assert solution.experiment_logs

# Test Task 4: Return Result
#
# Tests for the new SolutionResult structure with final_feedback.

import os
import sys
import unittest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.execution.solution import SolutionResult
from src.execution.feedback_generator import FeedbackResult
from src.execution.orchestrator import SolveResult


class TestSolutionResult(unittest.TestCase):
    """Test SolutionResult dataclass."""
    
    def test_solution_result_creation(self):
        """Test creating a SolutionResult."""
        result = SolutionResult(
            goal="Test goal",
            code_path="/tmp/test",
            experiment_logs=["log1", "log2"],
            final_feedback=FeedbackResult(
                stop=True,
                evaluation_valid=True,
                feedback="Goal achieved!",
                score=0.95,
            ),
            metadata={"iterations": 2}
        )
        
        self.assertEqual(result.goal, "Test goal")
        self.assertEqual(result.code_path, "/tmp/test")
        self.assertEqual(len(result.experiment_logs), 2)
        self.assertIsNotNone(result.final_feedback)
    
    def test_solution_result_succeeded_true(self):
        """Test succeeded property when goal achieved."""
        result = SolutionResult(
            goal="Test",
            code_path="/tmp",
            final_feedback=FeedbackResult(
                stop=True,
                evaluation_valid=True,
                feedback="Done",
            ),
        )
        
        self.assertTrue(result.succeeded)
    
    def test_solution_result_succeeded_false(self):
        """Test succeeded property when goal not achieved."""
        result = SolutionResult(
            goal="Test",
            code_path="/tmp",
            final_feedback=FeedbackResult(
                stop=False,
                evaluation_valid=True,
                feedback="Keep going",
            ),
        )
        
        self.assertFalse(result.succeeded)
    
    def test_solution_result_succeeded_no_feedback(self):
        """Test succeeded property when no feedback."""
        result = SolutionResult(
            goal="Test",
            code_path="/tmp",
        )
        
        self.assertFalse(result.succeeded)
    
    def test_solution_result_final_score(self):
        """Test final_score property."""
        result = SolutionResult(
            goal="Test",
            code_path="/tmp",
            final_feedback=FeedbackResult(
                stop=True,
                evaluation_valid=True,
                feedback="Done",
                score=0.85,
            ),
        )
        
        self.assertEqual(result.final_score, 0.85)
    
    def test_solution_result_final_score_none(self):
        """Test final_score property when no score."""
        result = SolutionResult(
            goal="Test",
            code_path="/tmp",
            final_feedback=FeedbackResult(
                stop=True,
                evaluation_valid=True,
                feedback="Done",
            ),
        )
        
        self.assertIsNone(result.final_score)
    
    def test_solution_result_explain(self):
        """Test explain method."""
        result = SolutionResult(
            goal="Test goal",
            code_path="/tmp/test",
            experiment_logs=["Iteration 1: score 0.5", "Iteration 2: score 0.9"],
            final_feedback=FeedbackResult(
                stop=True,
                evaluation_valid=True,
                feedback="Goal achieved!",
                score=0.9,
            ),
            metadata={"iterations": 2, "cost": "$1.50"}
        )
        
        explanation = result.explain()
        
        self.assertIn("Test goal", explanation)
        self.assertIn("/tmp/test", explanation)
        self.assertIn("Goal achieved: True", explanation)
        self.assertIn("Final score: 0.9", explanation)
        self.assertIn("iterations: 2", explanation)


class TestSolveResult(unittest.TestCase):
    """Test SolveResult dataclass."""
    
    def test_solve_result_creation(self):
        """Test creating a SolveResult."""
        from src.execution.search_strategies.base import ExperimentResult
        
        result = SolveResult(
            best_experiment=ExperimentResult(
                node_id=1,
                solution="Test solution",
                score=0.9,
                branch_name="test_branch",
                had_error=False,
            ),
            final_feedback=FeedbackResult(
                stop=True,
                evaluation_valid=True,
                feedback="Done",
                score=0.9,
            ),
            stopped_reason="goal_achieved",
            iterations_run=3,
            total_cost=1.50,
        )
        
        self.assertEqual(result.stopped_reason, "goal_achieved")
        self.assertEqual(result.iterations_run, 3)
        self.assertEqual(result.total_cost, 1.50)
        self.assertIsNotNone(result.best_experiment)
        self.assertIsNotNone(result.final_feedback)
    
    def test_solve_result_stopped_reasons(self):
        """Test different stopped reasons."""
        for reason in ["goal_achieved", "max_iterations", "budget_exhausted", "legacy_stop"]:
            result = SolveResult(
                best_experiment=None,
                final_feedback=None,
                stopped_reason=reason,
                iterations_run=1,
                total_cost=0.0,
            )
            self.assertEqual(result.stopped_reason, reason)


if __name__ == "__main__":
    unittest.main()

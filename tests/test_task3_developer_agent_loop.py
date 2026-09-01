# Test Task 3: Developer Agent Loop
#
# Tests for the new developer agent loop with feedback generator.

import os
import sys
import tempfile
import shutil
import unittest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kapso.execution.search_strategies.generic.feedback_generator.feedback_generator import FeedbackGenerator, FeedbackResult


class TestFeedbackResult(unittest.TestCase):
    """Test FeedbackResult dataclass."""
    
    def test_feedback_result_creation(self):
        """Test creating a FeedbackResult."""
        result = FeedbackResult(
            stop=True,
            evaluation_valid=True,
            feedback="Goal achieved!",
            score=0.95,
        )
        
        self.assertTrue(result.stop)
        self.assertTrue(result.evaluation_valid)
        self.assertEqual(result.feedback, "Goal achieved!")
        self.assertEqual(result.score, 0.95)
    
    def test_feedback_result_to_dict(self):
        """Test converting FeedbackResult to dict."""
        result = FeedbackResult(
            stop=False,
            evaluation_valid=True,
            feedback="Keep improving",
            score=0.5,
        )
        
        d = result.to_dict()
        self.assertEqual(d["stop"], False)
        self.assertEqual(d["evaluation_valid"], True)
        self.assertEqual(d["feedback"], "Keep improving")
        self.assertEqual(d["score"], 0.5)
    
    def test_feedback_result_optional_score(self):
        """Test FeedbackResult with no score."""
        result = FeedbackResult(
            stop=False,
            evaluation_valid=True,
            feedback="No score available",
        )
        
        self.assertIsNone(result.score)


class TestFeedbackGeneratorParseResponse(unittest.TestCase):
    """_parse_response reads XML tags, not JSON.

    The agent contract moved from a JSON object to <stop>/<evaluation_valid>/
    <score>/<feedback> tags. Untagged input is not an error here: the method
    returns None and the caller owns the retry and the explicit failure.
    """

    def setUp(self):
        # __new__ skips __init__ on purpose: _parse_response is pure string
        # work and needs none of the agent wiring __init__ builds.
        self.generator = FeedbackGenerator.__new__(FeedbackGenerator)

    def test_parse_tagged_response(self):
        response = (
            "<stop>true</stop>"
            "<evaluation_valid>true</evaluation_valid>"
            "<score>0.95</score>"
            "<feedback>Goal achieved!</feedback>"
        )
        result = self.generator._parse_response(response)

        self.assertTrue(result.stop)
        self.assertTrue(result.evaluation_valid)
        self.assertEqual(result.score, 0.95)
        self.assertEqual(result.feedback, "Goal achieved!")

    def test_parse_tags_surrounded_by_prose(self):
        """The agent narrates around its tags, so extraction is not anchored."""
        response = (
            "Here is my assessment.\n\n"
            "<stop>false</stop>\n"
            "<evaluation_valid>true</evaluation_valid>\n"
            "<score>0.5</score>\n"
            "<feedback>Keep improving</feedback>\n\n"
            "Let me know if you need more."
        )
        result = self.generator._parse_response(response)

        self.assertFalse(result.stop)
        self.assertEqual(result.score, 0.5)
        self.assertEqual(result.feedback, "Keep improving")

    def test_parse_null_score_is_none_not_zero(self):
        response = (
            "<stop>false</stop>"
            "<evaluation_valid>false</evaluation_valid>"
            "<score>null</score>"
            "<feedback>Evaluation was invalid</feedback>"
        )
        result = self.generator._parse_response(response)

        self.assertIsNone(result.score)
        self.assertFalse(result.evaluation_valid)

    def test_parse_untagged_response_returns_none(self):
        """No tags means no verdict — the caller retries rather than guessing."""
        self.assertIsNone(self.generator._parse_response("I could not evaluate this."))
        # the previous JSON contract is now simply untagged text
        self.assertIsNone(
            self.generator._parse_response('{"stop": true, "score": 0.9}')
        )

    def test_parse_load_bearing_cards(self):
        """Citation contract: comma-separated names, with 'none' meaning empty."""
        response = (
            "<stop>false</stop>"
            "<feedback>Used two cards</feedback>"
            "<cards_load_bearing>card:early-stopping, [batching-tradeoff]"
            "</cards_load_bearing>"
        )
        result = self.generator._parse_response(response)
        self.assertEqual(
            result.cards_load_bearing, ["early-stopping", "batching-tradeoff"]
        )

        none_response = (
            "<stop>false</stop><feedback>None applied</feedback>"
            "<cards_load_bearing>none</cards_load_bearing>"
        )
        result = self.generator._parse_response(none_response)
        self.assertEqual(result.cards_load_bearing, [])



class TestExperimentResult(unittest.TestCase):
    """Test ExperimentResult dataclass with new fields."""
    
    def test_experiment_result_new_fields(self):
        """Test ExperimentResult has new fields."""
        from kapso.execution.search_strategies.base import ExperimentResult
        
        result = ExperimentResult(
            node_id=1,
            solution="Test solution",
            score=0.8,
            branch_name="test_branch",
            had_error=False,
            evaluation_output="score: 0.8",
            evaluation_script_path="kapso_evaluation/evaluate.py",
            code_diff="+ added line",
            workspace_dir="/tmp/workspace",
        )
        
        self.assertEqual(result.evaluation_output, "score: 0.8")
        self.assertEqual(result.evaluation_script_path, "kapso_evaluation/evaluate.py")
        self.assertEqual(result.code_diff, "+ added line")
        self.assertEqual(result.workspace_dir, "/tmp/workspace")


class TestGenericProblemHandler(unittest.TestCase):
    """Test simplified GenericProblemHandler."""
    
    def test_handler_creation(self):
        """The handler takes a description plus optional eval/data dirs."""
        from kapso.environment.handlers.generic import GenericProblemHandler

        handler = GenericProblemHandler(
            problem_description="Test problem",
            eval_dir="./evaluation",
            data_dir="./datasets",
        )

        self.assertEqual(handler.problem_description, "Test problem")
        self.assertEqual(handler.eval_dir, "./evaluation")
        self.assertEqual(handler.data_dir, "./datasets")
        self.assertTrue(handler.maximize_scoring)


    def test_handler_problem_context_includes_evaluation_instructions(self):
        """Test that problem context includes evaluation instructions."""
        from kapso.environment.handlers.generic import GenericProblemHandler
        
        handler = GenericProblemHandler(
            problem_description="Test problem",
        )
        
        context = handler.get_problem_context()
        
        # Should include evaluation instructions
        self.assertIn("kapso_evaluation", context)
        self.assertIn("evaluation", context.lower())


class TestFeedbackGeneratorSignature(unittest.TestCase):
    """Test FeedbackGenerator.generate() has correct signature."""
    
    def test_generate_signature(self):
        """Test that generate() accepts the new parameters."""
        import inspect
        
        sig = inspect.signature(FeedbackGenerator.generate)
        params = list(sig.parameters.keys())
        
        # Check new parameters exist
        self.assertIn("goal", params)
        self.assertIn("idea", params)
        self.assertIn("code_changes_summary", params)
        self.assertIn("base_branch", params)
        self.assertIn("head_branch", params)
        self.assertIn("evaluation_script_path", params)
        self.assertIn("evaluation_result", params)
        self.assertIn("workspace_dir", params)
        
        # Check old parameters don't exist
        self.assertNotIn("code_diff", params)
        self.assertNotIn("implementation", params)
        self.assertNotIn("evaluation_code", params)


if __name__ == "__main__":
    unittest.main()

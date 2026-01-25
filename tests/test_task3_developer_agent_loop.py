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

from src.execution.feedback_generator import FeedbackGenerator, FeedbackResult


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
    """Test JSON parsing of feedback responses."""
    
    def setUp(self):
        """Create a FeedbackGenerator for testing."""
        self.generator = FeedbackGenerator.__new__(FeedbackGenerator)
    
    def test_parse_valid_json(self):
        """Test parsing valid JSON response."""
        response = '{"stop": true, "evaluation_valid": true, "score": 0.95, "feedback": "Goal achieved!"}'
        result = self.generator._parse_response(response)
        
        self.assertTrue(result.stop)
        self.assertTrue(result.evaluation_valid)
        self.assertEqual(result.score, 0.95)
        self.assertEqual(result.feedback, "Goal achieved!")
    
    def test_parse_json_in_code_fence(self):
        """Test parsing JSON inside markdown code fence."""
        response = '''```json
{"stop": false, "evaluation_valid": true, "score": 0.5, "feedback": "Keep improving"}
```'''
        result = self.generator._parse_response(response)
        
        self.assertFalse(result.stop)
        self.assertTrue(result.evaluation_valid)
        self.assertEqual(result.score, 0.5)
    
    def test_parse_json_with_null_score(self):
        """Test parsing JSON with null score."""
        response = '{"stop": false, "evaluation_valid": true, "score": null, "feedback": "No score"}'
        result = self.generator._parse_response(response)
        
        self.assertIsNone(result.score)
    
    def test_parse_invalid_json_returns_default(self):
        """Test that invalid JSON returns default result with raw response."""
        response = "This is not valid JSON"
        result = self.generator._parse_response(response)
        
        self.assertFalse(result.stop)
        self.assertTrue(result.evaluation_valid)
        self.assertIn("Failed to parse", result.feedback)


class TestExperimentResult(unittest.TestCase):
    """Test ExperimentResult dataclass with new fields."""
    
    def test_experiment_result_new_fields(self):
        """Test ExperimentResult has new fields."""
        from src.execution.search_strategies.base import ExperimentResult
        
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
        """Test creating a GenericProblemHandler."""
        from src.environment.handlers.generic import GenericProblemHandler
        
        handler = GenericProblemHandler(
            problem_description="Test problem",
            main_file="main.py",
            language="python",
        )
        
        self.assertEqual(handler.problem_description, "Test problem")
        self.assertEqual(handler.main_file, "main.py")
        self.assertEqual(handler.language, "python")
    
    def test_handler_stop_condition_always_false(self):
        """Test that stop_condition always returns False in new design."""
        from src.environment.handlers.generic import GenericProblemHandler
        
        handler = GenericProblemHandler(
            problem_description="Test problem",
        )
        
        # In new design, stop_condition always returns False
        # FeedbackGenerator handles stop decisions
        self.assertFalse(handler.stop_condition())
    
    def test_handler_problem_context_includes_evaluation_instructions(self):
        """Test that problem context includes evaluation instructions."""
        from src.environment.handlers.generic import GenericProblemHandler
        
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
        self.assertIn("code_diff", params)
        self.assertIn("evaluation_script_path", params)
        self.assertIn("evaluation_result", params)
        self.assertIn("workspace_dir", params)
        
        # Check old parameters don't exist
        self.assertNotIn("implementation", params)
        self.assertNotIn("evaluation_code", params)


if __name__ == "__main__":
    unittest.main()

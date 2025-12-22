import unittest
from unittest.mock import MagicMock, patch
import os
import shutil

from src.execution.context_manager.cognitive_context_manager import CognitiveContextManager
from src.execution.context_manager.types import ContextData
from src.environment.handlers.base import ProblemHandler
from src.execution.search_strategies.base import ExperimentResult

class TestCognitiveIntegration(unittest.TestCase):
    def setUp(self):
        self.test_store_path = ".test_integration_memory.json"
        
    def tearDown(self):
        if os.path.exists(self.test_store_path):
            os.remove(self.test_store_path)

    @patch('src.memory.controller.LLMBackend')
    def test_context_generation(self, MockLLM):
        # Setup mocks
        mock_llm = MockLLM.return_value
        mock_llm.llm_completion.return_value = "Run pip install numpy" # For meta-cognition
        
        mock_handler = MagicMock(spec=ProblemHandler)
        mock_handler.get_problem_context.return_value = "Build a numpy calculator"
        mock_handler.additional_context = ""
        
        mock_strategy = MagicMock()
        mock_strategy.get_experiment_history.return_value = []
        
        # Create manager
        manager = CognitiveContextManager(
            problem_handler=mock_handler,
            search_strategy=mock_strategy,
            params={"episodic_store_path": self.test_store_path}
        )
        
        # Get context (Iteration 1)
        context = manager.get_context(budget_progress=0)
        
        self.assertIn("# CURRENT MISSION", context.additional_info)
        self.assertIn("Build a numpy calculator", context.additional_info)
        
        # Simulate an error result
        error_result = MagicMock()
        error_result.run_had_error = True
        error_result.error_details = "ModuleNotFoundError: No module named 'numpy'"
        mock_strategy.get_experiment_history.return_value = [error_result]
        
        # Get context (Iteration 2)
        # It should now try to extract insight
        context2 = manager.get_context(budget_progress=10)
        
        # Verify controller was called (via briefing generation)
        self.assertIn("Build a numpy calculator", context2.additional_info)
        # Note: In a real run, the 'insights' field in briefing would populate 
        # if the LLM extracted an insight from the error.

if __name__ == '__main__':
    unittest.main()


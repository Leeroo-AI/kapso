"""
Test for BasicContextManager.

Verifies that the basic context manager:
1. Registers correctly with the factory
2. Returns context without KG results
3. Includes experiment history in additional_info
"""

import pytest
from unittest.mock import MagicMock, PropertyMock
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class MockExperimentResult:
    """Mock experiment result for testing."""
    branch_name: str
    score: float
    had_error: bool = False
    error_message: Optional[str] = None
    
    def __str__(self):
        return f"Experiment({self.branch_name}, score={self.score})"


class TestBasicContextManager:
    """Tests for BasicContextManager."""
    
    def test_registration(self):
        """Test that basic context manager is registered."""
        from src.execution.context_manager.factory import ContextManagerFactory
        
        available = ContextManagerFactory.list_available()
        assert "basic" in available, f"'basic' not in available context managers: {available}"
    
    def test_get_context_no_kg_results(self):
        """Test that get_context returns empty KG results."""
        from src.execution.context_manager.basic_context_manager import BasicContextManager
        
        # Create mocks
        problem_handler = MagicMock()
        problem_handler.get_problem_context.return_value = "Test problem description"
        problem_handler.additional_context = ""
        
        search_strategy = MagicMock()
        search_strategy.get_experiment_history.return_value = []
        
        # Create context manager
        cm = BasicContextManager(
            problem_handler=problem_handler,
            search_strategy=search_strategy,
            knowledge_search=None,
            params={},
        )
        
        # Get context
        context = cm.get_context(budget_progress=50)
        
        # Verify no KG results
        assert context.kg_results == "", "kg_results should be empty"
        assert context.kg_code_results == "", "kg_code_results should be empty"
        assert context.problem == "Test problem description"
    
    def test_get_context_with_experiment_history(self):
        """Test that experiment history is included in additional_info."""
        from src.execution.context_manager.basic_context_manager import BasicContextManager
        
        # Create mocks
        problem_handler = MagicMock()
        problem_handler.get_problem_context.return_value = "Test problem"
        problem_handler.additional_context = ""
        
        # Mock experiment history
        experiments = [
            MockExperimentResult("exp_1", 0.5),
            MockExperimentResult("exp_2", 0.8),
        ]
        
        search_strategy = MagicMock()
        search_strategy.get_experiment_history.return_value = experiments
        
        # Create context manager
        cm = BasicContextManager(
            problem_handler=problem_handler,
            search_strategy=search_strategy,
            params={"max_experiment_history_count": 5, "max_recent_experiment_count": 5},
        )
        
        # Get context
        context = cm.get_context()
        
        # Verify experiment history is in additional_info
        assert "exp_1" in context.additional_info or "exp_2" in context.additional_info
        assert "Previous Top Experiments" in context.additional_info
    
    def test_get_context_with_additional_context(self):
        """Test that additional_context from problem handler is included."""
        from src.execution.context_manager.basic_context_manager import BasicContextManager
        
        # Create mocks
        problem_handler = MagicMock()
        problem_handler.get_problem_context.return_value = "Test problem"
        problem_handler.additional_context = "Extra context from caller"
        
        search_strategy = MagicMock()
        search_strategy.get_experiment_history.return_value = []
        
        # Create context manager
        cm = BasicContextManager(
            problem_handler=problem_handler,
            search_strategy=search_strategy,
        )
        
        # Get context
        context = cm.get_context()
        
        # Verify additional context is included
        assert "Extra context from caller" in context.additional_info
    
    def test_factory_creation(self):
        """Test creating basic context manager via factory."""
        from src.execution.context_manager.factory import ContextManagerFactory
        from src.execution.context_manager.basic_context_manager import BasicContextManager
        
        # Create mocks
        problem_handler = MagicMock()
        problem_handler.get_problem_context.return_value = "Test"
        problem_handler.additional_context = ""
        
        search_strategy = MagicMock()
        search_strategy.get_experiment_history.return_value = []
        
        # Create via factory
        cm = ContextManagerFactory.create(
            context_manager_type="basic",
            problem_handler=problem_handler,
            search_strategy=search_strategy,
        )
        
        assert isinstance(cm, BasicContextManager)
    
    def test_should_stop_returns_false(self):
        """Test that should_stop always returns False (no decision making)."""
        from src.execution.context_manager.basic_context_manager import BasicContextManager
        
        problem_handler = MagicMock()
        problem_handler.additional_context = ""
        search_strategy = MagicMock()
        
        cm = BasicContextManager(
            problem_handler=problem_handler,
            search_strategy=search_strategy,
        )
        
        # Basic context manager should never signal stop
        assert cm.should_stop() == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

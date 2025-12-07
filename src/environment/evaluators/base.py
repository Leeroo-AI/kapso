# Evaluator Base Classes
#
# Abstract base class and result types for all evaluators.
# Evaluators score code execution results.
#
# To create a custom evaluator:
# 1. Subclass Evaluator
# 2. Implement evaluate() method
# 3. Register with @register_evaluator("name") decorator in factory.py

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class EvaluationResult:
    """
    Result of an evaluation.
    
    Attributes:
        score: Numeric score (interpretation depends on maximize_scoring)
        feedback: Optional textual feedback (useful for LLM judges)
        details: Optional structured details (metrics breakdown, etc.)
        raw_output: The raw output that was evaluated
    """
    score: float
    feedback: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    raw_output: str = ""
    
    def __float__(self) -> float:
        """Allow direct use as float."""
        return self.score
    
    def __repr__(self) -> str:
        return f"EvaluationResult(score={self.score}, feedback='{self.feedback[:50]}...')"


class Evaluator(ABC):
    """
    Abstract base class for all evaluators.
    
    Evaluators score code execution results based on output and/or files created.
    
    To create a custom evaluator:
    1. Subclass Evaluator
    2. Implement evaluate() method
    3. Register with @register_evaluator("name") decorator
    
    The evaluate() method receives:
    - output: stdout from code execution
    - file_path: workspace directory (for reading generated files)
    - **context: Additional context (solution description, problem, etc.)
    """
    
    # Metadata (override in subclasses)
    name: str = "base"
    description: str = "Base evaluator"
    requires_llm: bool = False
    
    def __init__(self, **params):
        """
        Initialize evaluator with parameters.
        
        Args:
            **params: Evaluator-specific parameters
        """
        self.params = params
    
    @abstractmethod
    def evaluate(
        self, 
        output: str, 
        file_path: str, 
        **context
    ) -> EvaluationResult:
        """
        Evaluate code execution results.
        
        Args:
            output: stdout/stderr from code execution
            file_path: Path to workspace directory
            **context: Additional context:
                - solution: str - The solution being evaluated
                - problem: str - Problem description
                - iteration: int - Current iteration number
                
        Returns:
            EvaluationResult with score and optional feedback
        """
        pass
    
    def __call__(self, output: str, file_path: str, **context) -> float:
        """
        Callable interface - returns just the score.
        
        This allows evaluators to be used as simple functions.
        
        Returns:
            Score as float
        """
        result = self.evaluate(output, file_path, **context)
        return result.score


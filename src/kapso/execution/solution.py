# Solution Result
#
# The artifact produced by Kapso.evolve() / OrchestratorAgent.solve().
# Contains the generated code, experiment logs, and metadata.
#
# Usage:
#     solution = kapso.evolve(goal="Create a trading bot")
#     software = kapso.deploy(solution, strategy=DeployStrategy.LOCAL)
#     result = software.run({"ticker": "AAPL"})
#     software.stop()

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from kapso.execution.search_strategies.generic import FeedbackResult


@dataclass
class SolutionResult:
    """
    The artifact produced by Kapso.evolve().
    
    Contains the generated code, experiment logs, and metadata.
    This is not just code - it captures the entire problem-solving attempt.
    
    Attributes:
        goal: The original goal/objective
        code_path: Path to the generated code/repository
        experiment_logs: List of experiment outcomes from the build process
        final_feedback: The last FeedbackResult from the feedback generator
        metadata: Additional information (constraints, timestamps, etc.)
    """
    goal: str
    code_path: str
    experiment_logs: List[str] = field(default_factory=list)
    final_feedback: Optional[FeedbackResult] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def succeeded(self) -> bool:
        """True if goal was achieved (feedback generator said stop)."""
        return self.final_feedback is not None and self.final_feedback.stop
    
    @property
    def final_score(self) -> Optional[float]:
        """Final evaluation score if available."""
        if self.final_feedback:
            return self.final_feedback.score
        return None
    
    def explain(self) -> str:
        """Return a summary of the solution and its experiments."""
        lines = [
            f"Solution for: {self.goal}",
            f"Code path: {self.code_path}",
            f"Experiments run: {len(self.experiment_logs)}",
            f"Goal achieved: {self.succeeded}",
        ]
        
        if self.final_score is not None:
            lines.append(f"Final score: {self.final_score}")
        
        if self.metadata:
            lines.append("\nMetadata:")
            for key, value in self.metadata.items():
                lines.append(f"  {key}: {value}")
        
        if self.experiment_logs:
            lines.append("\nExperiment History:")
            for i, log in enumerate(self.experiment_logs, 1):
                lines.append(f"  {i}. {log}")
        
        if self.final_feedback:
            lines.append(f"\nFinal Feedback: {self.final_feedback.feedback[:200]}...")
        
        return "\n".join(lines)

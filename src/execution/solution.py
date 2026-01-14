# Solution Result
#
# The artifact produced by Tinkerer.evolve() / OrchestratorAgent.solve().
# Contains the generated code, experiment logs, and metadata.
#
# Usage:
#     solution = tinkerer.evolve(goal="Create a trading bot")
#     software = tinkerer.deploy(solution, strategy=DeployStrategy.LOCAL)
#     result = software.run({"ticker": "AAPL"})
#     software.stop()

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class SolutionResult:
    """
    The artifact produced by Tinkerer.evolve().
    
    Contains the generated code, experiment logs, and metadata.
    This is not just code - it captures the entire problem-solving attempt.
    
    Attributes:
        goal: The original goal/objective
        code_path: Path to the generated code/repository
        experiment_logs: List of experiment outcomes from the build process
        metadata: Additional information (constraints, timestamps, etc.)
    """
    goal: str
    code_path: str
    experiment_logs: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def explain(self) -> str:
        """Return a summary of the solution and its experiments."""
        lines = [
            f"Solution for: {self.goal}",
            f"Code path: {self.code_path}",
            f"Experiments run: {len(self.experiment_logs)}",
        ]
        
        if self.metadata:
            lines.append("\nMetadata:")
            for key, value in self.metadata.items():
                lines.append(f"  {key}: {value}")
        
        if self.experiment_logs:
            lines.append("\nExperiment History:")
            for i, log in enumerate(self.experiment_logs, 1):
                lines.append(f"  {i}. {log}")
        
        return "\n".join(lines)

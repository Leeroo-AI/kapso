# Solution Result
#
# The artifact produced by Expert.build() / OrchestratorAgent.solve().
# Contains the generated code, experiment logs, and metadata.
#
# Usage:
#     solution = expert.build(goal="Create a trading bot")
#     software = solution.deploy(strategy=DeployStrategy.AUTO)  # or LOCAL, DOCKER, MODAL
#     result = software.run({"ticker": "AAPL"})
#     software.stop()

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.deployment import (
    Software,
    DeployConfig,
    DeployStrategy,
    DeploymentFactory,
)


@dataclass
class SolutionResult:
    """
    The artifact produced by Expert.build().
    
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
    
    def deploy(
        self, 
        strategy: DeployStrategy = DeployStrategy.AUTO,
        env_vars: Optional[Dict[str, str]] = None,
        port: int = 8000,
        coding_agent: str = "aider",
        validate: bool = True,
    ) -> Software:
        """
        Deploy this solution to create running software.
        
        Uses the deployment pipeline:
        1. Selector: Analyzes solution and selects strategy (if AUTO)
        2. Adapter: Transforms code for the target deployment
        3. Runner: Creates execution backend
        
        Args:
            strategy: Where to deploy (AUTO, LOCAL, DOCKER, MODAL, BENTOML)
                - AUTO: System analyzes code and chooses best strategy
                - LOCAL: Run as local Python process (fastest)
                - DOCKER: Run in Docker container (isolated)
                - MODAL: Deploy to Modal.com (serverless, GPU)
                - BENTOML: Deploy with BentoML (production ML)
            env_vars: Environment variables to pass to the software
            port: Port to expose (for HTTP-based deployments)
            coding_agent: Which coding agent for adaptation (aider, gemini, etc.)
            validate: Whether to validate the adaptation before returning
            
        Returns:
            Software instance with unified interface:
            - .run(inputs) -> {"status": "success", "output": ...}
            - .stop() -> cleanup resources
            - .logs() -> execution logs
            - .is_healthy() -> health check
            
        Example:
            # Auto-select best deployment
            software = solution.deploy()
            
            # Explicitly choose local
            software = solution.deploy(strategy=DeployStrategy.LOCAL)
            
            # Deploy to Modal for GPU
            software = solution.deploy(strategy=DeployStrategy.MODAL)
            
            # All have the same interface
            result = software.run({"text": "hello"})
            print(result["output"])
            software.stop()
        """
        config = DeployConfig(
            code_path=self.code_path,
            goal=self.goal,
            env_vars=env_vars,
            port=port,
            coding_agent=coding_agent,
            validate=validate,
        )
        
        return DeploymentFactory.create(strategy, config)
    
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

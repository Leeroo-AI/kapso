# Deployment Base Classes
#
# Unified interface for deployed software.
# Users interact with Software instances which provide the same interface
# regardless of the underlying infrastructure (Local, Docker, Modal, etc.).
#
# The deployment flow:
# 1. Kapso.deploy(solution) -> DeploymentFactory.create()
# 2. Selector chooses best strategy (if AUTO)
# 3. Adapter transforms repo for the strategy
# 4. Runner handles actual execution
# 5. DeployedSoftware wraps runner with unified interface

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from kapso.execution.solution import SolutionResult


def _discover_strategies() -> Dict[str, str]:
    """
    Discover available deployment strategies from the strategies/ directory.
    
    Returns:
        Dict mapping uppercase name to lowercase value (e.g., {"LOCAL": "local"})
    """
    strategies_dir = Path(__file__).parent / "strategies"
    discovered = {}
    
    if strategies_dir.exists():
        for path in sorted(strategies_dir.iterdir()):
            # Skip non-directories, hidden dirs, and __pycache__
            if not path.is_dir():
                continue
            if path.name.startswith("_") or path.name.startswith("."):
                continue
            
            # Must have config.yaml to be a valid strategy
            if (path / "config.yaml").exists():
                discovered[path.name.upper()] = path.name
    
    return discovered


def _build_deploy_strategy_enum():
    """
    Build the DeployStrategy enum dynamically.
    
    Includes AUTO + all discovered strategies from strategies/ directory.
    """
    # Start with AUTO (always available)
    members = {"AUTO": "auto"}
    
    # Add discovered strategies
    members.update(_discover_strategies())
    
    # Create enum using functional API
    return Enum("DeployStrategy", members)


# Dynamically created enum based on strategies/ directory
DeployStrategy = _build_deploy_strategy_enum()
DeployStrategy.__doc__ = """
Deployment target for a Solution.

AUTO: Let the system analyze and choose the best strategy.
Other values are discovered from the strategies/ directory.
"""


@dataclass
class DeployConfig:
    """
    Configuration for deploying software.
    
    Attributes:
        solution: The SolutionResult from Kapso.evolve()
        env_vars: Environment variables to pass to the software
        timeout: Execution timeout in seconds
        coding_agent: Which coding agent to use for adaptation
    """
    solution: SolutionResult
    env_vars: Dict[str, str] = None
    timeout: int = 300
    coding_agent: str = "claude_code"
    
    def __post_init__(self):
        if self.env_vars is None:
            self.env_vars = {}
    
    # Convenience accessors
    @property
    def code_path(self) -> str:
        """Path to the generated code/repository."""
        return self.solution.code_path
    
    @property
    def goal(self) -> str:
        """The original goal/objective."""
        return self.solution.goal


@dataclass
class DeploymentSetting:
    """
    Selected deployment configuration.
    
    Produced by the Selector, consumed by the Adapter.
    """
    strategy: str              # "local", "docker", "modal", etc.
    provider: Optional[str]    # For cloud: "modal", "bentoml", etc.
    resources: Dict[str, Any]  # CPU, memory, GPU requirements
    interface: str             # "function", "http", "cli"
    reasoning: str             # Why this was selected
    
    def __post_init__(self):
        if self.resources is None:
            self.resources = {}


@dataclass
class AdaptationResult:
    """
    Result of adapting a solution for deployment.
    
    Produced by the Adapter, consumed by the Factory.
    """
    success: bool
    adapted_path: str           # Path to adapted repo (copy of original)
    run_interface: Dict[str, Any]  # How to call .run() after deployment
    files_changed: List[str] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class DeploymentInfo:
    """
    Metadata about how software was deployed.
    
    Hidden from users but available for debugging.
    """
    strategy: str                    # "local", "docker", "modal", etc.
    provider: Optional[str] = None   # "modal", "bentoml", etc.
    endpoint: Optional[str] = None   # HTTP endpoint if applicable
    adapted_path: str = ""           # Path to adapted code
    adapted_files: List[str] = field(default_factory=list)
    resources: Dict[str, Any] = field(default_factory=dict)


class Software(ABC):
    """
    Unified interface for deployed software.
    
    This is the ONLY class users interact with after deployment.
    All infrastructure details are hidden behind this interface.
    
    Usage:
        software = kapso.deploy(solution)  # Returns Software
        result = software.run({"text": "hello"})  # Always works the same
        software.stop()
        
    The response format is ALWAYS consistent:
        {"status": "success", "output": <result>}
        {"status": "error", "error": <message>}
    """
    
    def __init__(self, config: DeployConfig):
        """
        Initialize software deployment.
        
        Args:
            config: Deployment configuration
        """
        self.config = config
        self.code_path = config.code_path
        self.goal = config.goal
    
    @abstractmethod
    def run(self, inputs: Union[Dict, str, bytes]) -> Dict[str, Any]:
        """
        Execute the software with given inputs.
        
        This is the UNIFIED interface - works the same whether
        running locally, in Docker, on Modal, etc.
        
        Args:
            inputs: Input data (dict, string, or bytes)
            
        Returns:
            Dict with:
            - "status": "success" | "error"
            - "output": The actual result (if success)
            - "error": Error message (if error)
        """
        pass
    
    @abstractmethod
    def stop(self) -> None:
        """Stop the software and cleanup resources."""
        pass
    
    @abstractmethod
    def start(self) -> None:
        """
        Start or restart a stopped deployment.
        
        This method re-initializes the deployment after stop() was called.
        Behavior varies by strategy:
        - LOCAL: Reloads the Python module
        - DOCKER: Creates and starts a new container
        - MODAL: Re-lookups the Modal function
        - BENTOML: Re-deploys using deploy.py
        - LANGGRAPH: Reconnects to the platform
        """
        pass
    
    @abstractmethod
    def logs(self) -> str:
        """
        Get execution logs.
        
        Returns:
            Log content as string
        """
        pass
    
    @abstractmethod
    def is_healthy(self) -> bool:
        """
        Check if software is running and healthy.
        
        Returns:
            True if healthy, False otherwise
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the deployment strategy name."""
        pass
    
    # =========================================================================
    # UNIFIED CONVENIENCE METHODS (same for all implementations)
    # =========================================================================
    
    def __call__(self, inputs: Union[Dict, str, bytes]) -> Dict[str, Any]:
        """Allow software(inputs) as shorthand for software.run(inputs)."""
        return self.run(inputs)
    
    def run_batch(self, inputs_list: List[Any]) -> List[Dict[str, Any]]:
        """
        Run multiple inputs in sequence.
        
        Args:
            inputs_list: List of input dicts/strings
            
        Returns:
            List of results in same order
        """
        return [self.run(inputs) for inputs in inputs_list]
    
    def __enter__(self):
        """Context manager support."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Auto-cleanup on context exit."""
        self.stop()
        return False
    
    def __repr__(self) -> str:
        goal_preview = self.goal[:50] + "..." if len(self.goal) > 50 else self.goal
        return f"Software(strategy={self.name}, goal='{goal_preview}')"

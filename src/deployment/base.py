# Deployment Base Classes
#
# Unified interface for deployed software.
# Users interact with Software instances which provide the same interface
# regardless of the underlying infrastructure (Local, Docker, Modal, etc.).
#
# The deployment flow:
# 1. Expert.deploy(solution) -> DeploymentFactory.create()
# 2. Selector chooses best strategy (if AUTO)
# 3. Adapter transforms repo for the strategy
# 4. Runner handles actual execution
# 5. DeployedSoftware wraps runner with unified interface

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from src.execution.solution import SolutionResult


class DeployStrategy(Enum):
    """
    Deployment target for a Solution.
    
    AUTO: Let the system analyze and choose the best strategy
    LOCAL: Run as a local process (fast, for development)
    DOCKER: Run in a Docker container (isolated, reproducible)
    MODAL: Deploy to Modal.com (serverless, GPU support)
    BENTOML: Deploy with BentoML (production ML serving)
    LANGGRAPH: Deploy to LangGraph Platform (stateful agents)
    """
    AUTO = "auto"
    LOCAL = "local"
    DOCKER = "docker"
    MODAL = "modal"
    BENTOML = "bentoml"
    LANGGRAPH = "langgraph"


@dataclass
class DeployConfig:
    """
    Configuration for deploying software.
    
    Attributes:
        solution: The SolutionResult from Expert.build()
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
        software = expert.deploy(solution)  # Returns Software
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

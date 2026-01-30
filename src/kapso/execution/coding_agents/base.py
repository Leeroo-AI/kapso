# Base interface for pluggable coding agents.
#
# All coding agents must implement CodingAgentInterface to be usable
# in the orchestrator's experiment loop.
#
# Key design principles:
# - Agents only generate code, they do NOT handle git operations
# - Git is managed by ExperimentSession
# - Agents can optionally provide commit messages via CodingResult

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CodingResult:
    """
    Standardized result from any coding agent.
    
    All coding agents return this structure regardless of their
    internal implementation (Aider, Gemini, Claude Code, OpenHands).
    """
    # Whether the code generation succeeded
    success: bool
    
    # Agent's response/output text
    output: str
    
    # List of files that were changed
    files_changed: List[str] = field(default_factory=list)
    
    # Error message if success=False
    error: Optional[str] = None
    
    # Cost of this generation call (in dollars)
    cost: float = 0.0
    
    # Optional: Agent-generated commit message (used by hybrid commit generator)
    commit_message: Optional[str] = None
    
    # Agent-specific metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CodingAgentConfig:
    """
    Configuration for coding agents.
    
    Parsed from config.yaml's coding_agent section.
    """
    # Agent type: "aider", "gemini", "claude_code", "openhands"
    agent_type: str
    
    # Primary model for implementation (e.g., "o3", "gemini-2.0-flash")
    model: str
    
    # Model for debugging (can be same as model)
    debug_model: str
    
    # Working directory for the agent
    workspace: str = ""
    
    # Whether to use git integration
    use_git: bool = True
    
    # Agent-specific configuration options
    # Examples:
    #   Aider: {"edit_format": "diff"}
    #   Claude Code: {"claude_md_path": "CLAUDE.md"}
    #   OpenHands: {"sandbox_type": "docker"}
    agent_specific: Dict[str, Any] = field(default_factory=dict)


class CodingAgentInterface(ABC):
    """
    Abstract interface for coding agents.
    
    All coding agents must implement this interface to be pluggable
    into the orchestrator's experiment loop.
    
    IMPORTANT: Coding agents should NOT handle git operations.
    Git is managed by ExperimentSession. Agents only generate code.
    
    To add a new coding agent:
    1. Create adapter in src/agents/coding_agents/adapters/
    2. Implement all abstract methods
    3. Register in factory.py
    """
    
    def __init__(self, config: CodingAgentConfig):
        """
        Initialize the coding agent with configuration.
        
        Args:
            config: CodingAgentConfig with agent settings
        """
        self.config = config
        self._cumulative_cost = 0.0
    
    @abstractmethod
    def initialize(self, workspace: str) -> None:
        """
        Initialize the agent for a specific workspace.
        
        Called once when an ExperimentSession is created.
        
        Args:
            workspace: Path to the working directory
            
        NOTE: Do NOT perform git operations here.
        """
        pass
    
    @abstractmethod
    def generate_code(self, prompt: str, debug_mode: bool = False) -> CodingResult:
        """
        Generate or modify code based on the prompt.
        
        This is the main method that generates code. The agent should:
        1. Process the prompt
        2. Generate/edit code files
        3. Return a CodingResult with changed files
        
        Args:
            prompt: The implementation or debugging instructions
            debug_mode: If True, use debug model and debugging behavior
            
        Returns:
            CodingResult with success status, output, files changed, and cost
            
        NOTE: Do NOT commit changes. ExperimentSession handles commits.
        """
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """
        Clean up resources.
        
        Called when the ExperimentSession is closed.
        Should close connections, remove temp files, etc.
        """
        pass
    
    def get_cumulative_cost(self) -> float:
        """
        Return total cost accumulated by this agent.
        
        Returns:
            Total cost in dollars
        """
        return self._cumulative_cost
    
    def supports_native_git(self) -> bool:
        """
        Return True if agent handles its own git commits.
        
        Used by ExperimentSession to decide whether to commit
        after generate_code() calls.
        
        - Aider: True (auto-commits with use_git=True)
        - Gemini/Claude/OpenHands: False (session commits)
        
        Returns:
            True if agent auto-commits, False otherwise
        """
        return False
    
    def get_capabilities(self) -> Dict[str, bool]:
        """
        Return agent capabilities for feature detection.
        
        Returns:
            Dictionary of capability flags
        """
        return {
            "native_git": self.supports_native_git(),
            "sandbox": False,
            "planning_mode": False,
            "cost_tracking": True,
            "streaming": False,
        }


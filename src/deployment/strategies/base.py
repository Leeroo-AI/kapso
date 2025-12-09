# Strategies Base Module
#
# Contains:
# - Runner: Abstract base class for all strategy runners
# - StrategyRegistry: Auto-discovers deployment strategies from subdirectories
# - StrategyInfo: Information about a discovered strategy
#
# Usage:
#     from src.deployment.strategies.base import Runner, StrategyRegistry
#
#     # Get available strategies
#     registry = StrategyRegistry.get()
#     strategies = registry.list_strategies()
#
#     # Create a custom runner
#     class MyRunner(Runner):
#         def run(self, inputs): ...

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


# =============================================================================
# Runner Base Class
# =============================================================================

class Runner(ABC):
    """
    Abstract runner that handles actual execution.
    
    Runners are the infrastructure-specific implementations.
    Users never interact with runners directly - they're wrapped
    by DeployedSoftware which provides the unified interface.
    
    Each runner knows how to:
    - Execute code (via import, subprocess, HTTP, etc.)
    - Check health
    - Clean up resources
    
    To add a new strategy:
    1. Create strategies/{name}/ directory
    2. Add selector_instruction.md and adapter_instruction.md
    3. Create runner.py with a class inheriting from Runner
    """
    
    @abstractmethod
    def run(self, inputs: Union[Dict, str, bytes]) -> Any:
        """
        Execute with inputs and return result.
        
        Args:
            inputs: Input data for the software
            
        Returns:
            Result from execution (format varies by runner)
        """
        pass
    
    @abstractmethod
    def stop(self) -> None:
        """Stop and cleanup resources."""
        pass
    
    @abstractmethod
    def is_healthy(self) -> bool:
        """
        Check if runner is healthy and ready.
        
        Returns:
            True if healthy, False otherwise
        """
        pass
    
    def get_logs(self) -> str:
        """
        Get runner-specific logs.
        
        Returns:
            Log content as string (default: empty)
        """
        return ""


# =============================================================================
# Strategy Discovery
# =============================================================================

@dataclass
class StrategyInfo:
    """
    Info about a deployment strategy.
    
    Loaded from a strategy subdirectory containing:
    - selector_instruction.md: When to choose this strategy
    - adapter_instruction.md: How to adapt and deploy
    - runner.py: Runtime execution class
    """
    name: str
    directory: Path
    selector_instruction_path: Path
    adapter_instruction_path: Path
    
    def get_selector_instruction(self) -> str:
        """Load selector_instruction.md content."""
        return self.selector_instruction_path.read_text()
    
    def get_adapter_instruction(self) -> str:
        """Load adapter_instruction.md content."""
        return self.adapter_instruction_path.read_text()
    
    def has_runner(self) -> bool:
        """Check if strategy has a runner.py file."""
        return (self.directory / "runner.py").exists()
    
    def get_default_run_interface(self) -> Dict[str, Any]:
        """
        Parse default RUN INTERFACE section from adapter_instruction.md.
        
        This provides fallback values if the coding agent doesn't output
        a run_interface JSON. The adapter_instruction.md should have:
        
            ## RUN INTERFACE
            - type: function
            - module: main
            - callable: predict
        
        Returns:
            Dict with default interface configuration
        """
        import re
        
        content = self.get_adapter_instruction()
        interface: Dict[str, Any] = {}
        
        # Find RUN INTERFACE section
        match = re.search(
            r'##\s*RUN INTERFACE\s*\n((?:- [^\n]+\n?)+)',
            content,
            re.IGNORECASE
        )
        
        if match:
            # Parse each "- key: value" line
            lines = match.group(1).strip().split('\n')
            for line in lines:
                # Match "- key: value" format
                kv_match = re.match(r'-\s*(\w+):\s*(.+)', line.strip())
                if kv_match:
                    key = kv_match.group(1).strip()
                    value = kv_match.group(2).strip()
                    interface[key] = value
        
        return interface


class StrategyRegistry:
    """
    Auto-discovers and provides access to deployment strategies.
    
    Strategies are discovered from subdirectories of the strategies/ folder.
    Each subdirectory must contain:
    - selector_instruction.md
    - adapter_instruction.md
    
    Usage:
        registry = StrategyRegistry.get()
        
        # List all strategies
        strategies = registry.list_strategies()
        
        # Filter to specific strategies
        strategies = registry.list_strategies(allowed=["local", "modal"])
        
        # Get instructions
        selector_md = registry.get_selector_instruction("modal")
        adapter_md = registry.get_adapter_instruction("modal")
    """
    
    _instance: Optional["StrategyRegistry"] = None
    
    def __init__(self):
        """Initialize registry (use .get() for singleton access)."""
        self._strategies: Dict[str, StrategyInfo] = {}
    
    @classmethod
    def get(cls) -> "StrategyRegistry":
        """Get singleton registry instance (auto-discovers on first call)."""
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._discover()
        return cls._instance
    
    @classmethod
    def reset(cls) -> None:
        """Reset singleton (useful for testing)."""
        cls._instance = None
    
    def _discover(self) -> None:
        """Auto-discover strategy packages from subdirectories."""
        strategies_dir = Path(__file__).parent
        
        for path in sorted(strategies_dir.iterdir()):
            # Skip non-directories, hidden dirs, and __pycache__
            if not path.is_dir():
                continue
            if path.name.startswith("_") or path.name.startswith("."):
                continue
            
            selector_path = path / "selector_instruction.md"
            adapter_path = path / "adapter_instruction.md"
            
            # Must have both instruction files
            if selector_path.exists() and adapter_path.exists():
                self._strategies[path.name] = StrategyInfo(
                    name=path.name,
                    directory=path,
                    selector_instruction_path=selector_path,
                    adapter_instruction_path=adapter_path,
                )
    
    def list_strategies(self, allowed: Optional[List[str]] = None) -> List[str]:
        """
        List available strategy names.
        
        Args:
            allowed: Optional list to filter. None means all available.
            
        Returns:
            List of strategy names (filtered if allowed specified)
        """
        all_strategies = list(self._strategies.keys())
        
        if allowed is None:
            return all_strategies
        
        # Filter to only allowed strategies that exist
        return [s for s in allowed if s in self._strategies]
    
    def get_strategy(self, name: str) -> StrategyInfo:
        """
        Get strategy info by name.
        
        Args:
            name: Strategy name (e.g., "modal", "docker")
            
        Returns:
            StrategyInfo for the strategy
            
        Raises:
            ValueError: If strategy not found
        """
        if name not in self._strategies:
            available = ", ".join(sorted(self._strategies.keys()))
            raise ValueError(f"Unknown strategy '{name}'. Available: {available}")
        return self._strategies[name]
    
    def get_selector_instruction(self, name: str) -> str:
        """
        Get selector instruction content for a strategy.
        
        Args:
            name: Strategy name
            
        Returns:
            Content of selector_instruction.md
        """
        return self.get_strategy(name).get_selector_instruction()
    
    def get_adapter_instruction(self, name: str) -> str:
        """
        Get adapter instruction content for a strategy.
        
        Args:
            name: Strategy name
            
        Returns:
            Content of adapter_instruction.md
        """
        return self.get_strategy(name).get_adapter_instruction()
    
    def get_all_selector_instructions(
        self, 
        allowed: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """
        Get all selector instructions (optionally filtered).
        
        Args:
            allowed: Optional list of strategies to include
            
        Returns:
            Dict mapping strategy name to selector instruction content
        """
        strategies = self.list_strategies(allowed=allowed)
        return {name: self.get_selector_instruction(name) for name in strategies}
    
    def strategy_exists(self, name: str) -> bool:
        """Check if a strategy exists."""
        return name in self._strategies
    
    def get_default_run_interface(self, name: str) -> Dict[str, Any]:
        """
        Get default run interface for a strategy.
        
        Parses the RUN INTERFACE section from adapter_instruction.md.
        Used as fallback when coding agent doesn't output run_interface JSON.
        
        Args:
            name: Strategy name
            
        Returns:
            Dict with default interface configuration
        """
        return self.get_strategy(name).get_default_run_interface()

# Strategies Base Module
#
# Contains:
# - Runner: Abstract base class for all strategy runners
# - StrategyRegistry: Auto-discovers deployment strategies from subdirectories
# - DeployStrategyConfig: Configuration for a discovered strategy
#
# Usage:
#     from kapso.deployment.strategies.base import Runner, StrategyRegistry
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
    
    Lifecycle:
    - __init__(): Initialize and prepare runner (ready state)
    - run(): Execute with inputs
    - stop(): Stop and cleanup resources (stopped state)
    - start(): Re-initialize a stopped runner (ready state again)
    
    Each runner knows how to:
    - Execute code (via import, subprocess, HTTP, etc.)
    - Check health
    - Start/restart resources
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
    def start(self) -> None:
        """
        Start or restart the runner.
        
        Re-initializes a stopped runner back to ready state.
        Can be called after stop() to restart the runner.
        
        For local runners: Reloads the module and function.
        For cloud runners: Reconnects to the service.
        For container runners: Starts the container.
        """
        pass
    
    @abstractmethod
    def stop(self) -> None:
        """
        Stop and cleanup resources.
        
        Transitions runner to stopped state.
        After stop(), the runner can be restarted with start().
        
        For local runners: Unloads the module.
        For cloud runners: Terminates the deployment.
        For container runners: Stops and removes the container.
        """
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
class DeployStrategyConfig:
    """
    Configuration for a deployment strategy.
    
    Loaded from a strategy subdirectory containing:
    - config.yaml: Strategy configuration (interface, provider, resources, run_interface)
    - selector_instruction.md: When to choose this strategy
    - adapter_instruction.md: How to adapt and deploy
    - runner.py: Runtime execution class
    """
    name: str
    directory: Path
    selector_instruction_path: Path
    adapter_instruction_path: Path
    
    # Cached config from config.yaml
    _config_cache: Optional[Dict[str, Any]] = None
    
    def get_config(self) -> Dict[str, Any]:
        """
        Load config.yaml for this strategy.
        
        The config contains:
        - name: Strategy name
        - provider: Cloud provider (or null)
        - interface: Interface type (function, http, etc.)
        - default_resources: Default resource requirements
        - run_interface: Default run interface for the runner
        
        Returns:
            Dict with strategy configuration
        """
        if self._config_cache is not None:
            return self._config_cache
        
        import yaml
        
        config_path = self.directory / "config.yaml"
        if config_path.exists():
            self._config_cache = yaml.safe_load(config_path.read_text()) or {}
        else:
            self._config_cache = {}
        
        return self._config_cache
    
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
        Get default run interface from config.yaml.
        
        Used as fallback when the coding agent doesn't output
        a run_interface JSON.
        
        Returns:
            Dict with default interface configuration
        """
        config = self.get_config()
        return config.get("run_interface", {}).copy()
    
    def get_default_resources(self) -> Dict[str, Any]:
        """
        Get default resources from config.yaml.
        
        Returns:
            Dict with default resource requirements
        """
        config = self.get_config()
        return config.get("default_resources", {}).copy()
    
    def get_interface(self) -> str:
        """Get interface type (function, http, etc.)."""
        config = self.get_config()
        return config.get("interface", "function")
    
    def get_provider(self) -> Optional[str]:
        """Get cloud provider name (or None for local)."""
        config = self.get_config()
        return config.get("provider")
    
    def get_runner_class(self) -> type:
        """
        Dynamically import and return the runner class for this strategy.
        
        The runner class name is specified in config.yaml as 'runner_class'.
        The class is imported from runner.py in the strategy directory.
        
        Returns:
            The Runner subclass for this strategy
            
        Raises:
            ImportError: If runner module or class not found
        """
        import importlib.util
        
        config = self.get_config()
        class_name = config.get("runner_class")
        
        if not class_name:
            raise ValueError(f"No runner_class defined in config.yaml for strategy '{self.name}'")
        
        # Load runner.py from strategy directory
        runner_path = self.directory / "runner.py"
        if not runner_path.exists():
            raise ImportError(f"runner.py not found for strategy '{self.name}'")
        
        # Dynamic import
        spec = importlib.util.spec_from_file_location(f"{self.name}_runner", runner_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Get the class
        if not hasattr(module, class_name):
            raise ImportError(f"Class '{class_name}' not found in {runner_path}")
        
        return getattr(module, class_name)


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
        self._strategies: Dict[str, DeployStrategyConfig] = {}
    
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
                self._strategies[path.name] = DeployStrategyConfig(
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
    
    def get_strategy(self, name: str) -> DeployStrategyConfig:
        """
        Get strategy config by name.
        
        Args:
            name: Strategy name (e.g., "modal", "docker")
            
        Returns:
            DeployStrategyConfig for the strategy
            
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
    
    def get_config(self, name: str) -> Dict[str, Any]:
        """
        Get full config for a strategy from config.yaml.
        
        Args:
            name: Strategy name
            
        Returns:
            Dict with strategy configuration
        """
        return self.get_strategy(name).get_config()
    
    def get_default_run_interface(self, name: str) -> Dict[str, Any]:
        """
        Get default run interface for a strategy from config.yaml.
        
        Used as fallback when coding agent doesn't output run_interface JSON.
        
        Args:
            name: Strategy name
            
        Returns:
            Dict with default interface configuration
        """
        return self.get_strategy(name).get_default_run_interface()
    
    def get_default_resources(self, name: str) -> Dict[str, Any]:
        """
        Get default resources for a strategy from config.yaml.
        
        Args:
            name: Strategy name
            
        Returns:
            Dict with default resource requirements
        """
        return self.get_strategy(name).get_default_resources()
    
    def get_interface(self, name: str) -> str:
        """
        Get interface type for a strategy from config.yaml.
        
        Args:
            name: Strategy name
            
        Returns:
            Interface type (function, http, etc.)
        """
        return self.get_strategy(name).get_interface()
    
    def get_provider(self, name: str) -> Optional[str]:
        """
        Get cloud provider for a strategy from config.yaml.
        
        Args:
            name: Strategy name
            
        Returns:
            Provider name or None
        """
        return self.get_strategy(name).get_provider()
    
    def get_runner_class(self, name: str) -> type:
        """
        Get the runner class for a strategy.
        
        Dynamically imports from strategies/{name}/runner.py.
        
        Args:
            name: Strategy name
            
        Returns:
            The Runner subclass for this strategy
        """
        return self.get_strategy(name).get_runner_class()

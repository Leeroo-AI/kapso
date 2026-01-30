# Factory for creating coding agents with auto-discovery from agents.yaml
#
# Features:
# - Auto-discovers agents from agents.yaml registry
# - Provides default configurations for each agent
# - Allows runtime registration of custom agents

# Suppress deprecation warnings from third-party dependencies before any imports
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pydub")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="aider")

import importlib
import yaml
from pathlib import Path
from typing import Dict, Type, List, Any, Optional

from kapso.execution.coding_agents.base import CodingAgentInterface, CodingAgentConfig


# Path to agents registry YAML
AGENTS_YAML_PATH = Path(__file__).parent / "agents.yaml"


class CodingAgentFactory:
    """
    Factory for creating coding agents with auto-discovery.
    
    Agents are registered from agents.yaml on module import.
    Custom agents can also be registered at runtime.
    """
    
    # Registry of agent classes: agent_type -> class
    _registry: Dict[str, Type[CodingAgentInterface]] = {}
    
    # Agent configurations from agents.yaml
    _agent_configs: Dict[str, Dict[str, Any]] = {}
    
    # Default agent type (from agents.yaml)
    _default_agent: str = "aider"
    
    # =========================================================================
    # Core Factory Methods
    # =========================================================================
    
    @classmethod
    def register(cls, name: str, agent_class: Type[CodingAgentInterface]) -> None:
        """
        Register a coding agent type.
        
        Called automatically for agents in agents.yaml.
        Can also be called manually for custom agents.
        
        Args:
            name: Identifier for the agent (e.g., "aider", "gemini")
            agent_class: Class implementing CodingAgentInterface
        """
        name_lower = name.lower()
        if name_lower in cls._registry:
            print(f"[CodingAgentFactory] Warning: Overwriting agent '{name}'")
        cls._registry[name_lower] = agent_class
        print(f"[CodingAgentFactory] Registered agent: {name}")
    
    @classmethod
    def create(cls, config: CodingAgentConfig) -> CodingAgentInterface:
        """
        Create a coding agent from configuration.
        
        Args:
            config: CodingAgentConfig with agent_type and settings
            
        Returns:
            Instance of the requested coding agent
            
        Raises:
            ValueError: If agent_type is not registered
        """
        agent_type = config.agent_type.lower()
        
        if agent_type not in cls._registry:
            available = ", ".join(sorted(cls._registry.keys()))
            raise ValueError(
                f"Unknown coding agent: '{agent_type}'. "
                f"Available agents: {available or 'none registered'}"
            )
        
        agent_class = cls._registry[agent_type]
        return agent_class(config)
    
    # =========================================================================
    # Configuration Methods
    # =========================================================================
    
    @classmethod
    def get_default_config(cls, agent_type: str) -> Dict[str, Any]:
        """
        Get default configuration for an agent type.
        
        Returns dict with:
        - default_model
        - default_debug_model
        - agent_specific defaults
        - supports_native_git
        
        Args:
            agent_type: Agent type name (e.g., "aider")
            
        Returns:
            Dictionary with default configuration
            
        Raises:
            ValueError: If agent_type is unknown
        """
        agent_type_lower = agent_type.lower()
        if agent_type_lower not in cls._agent_configs:
            raise ValueError(f"Unknown agent type: {agent_type}")
        return cls._agent_configs[agent_type_lower].copy()
    
    @classmethod
    def get_agent_info(cls, agent_type: str) -> Dict[str, Any]:
        """
        Get full info for an agent from agents.yaml.
        
        Includes: description, install_command, env_vars, documentation_url
        
        Args:
            agent_type: Agent type name
            
        Returns:
            Dictionary with agent info, or empty dict if not found
        """
        return cls._agent_configs.get(agent_type.lower(), {})
    
    @classmethod
    def get_default_agent(cls) -> str:
        """
        Get the default agent type from agents.yaml.
        
        Returns:
            Default agent type name (e.g., "aider")
        """
        return cls._default_agent
    
    @classmethod
    def build_config(
        cls,
        agent_type: Optional[str] = None,
        model: Optional[str] = None,
        debug_model: Optional[str] = None,
        agent_specific: Optional[Dict[str, Any]] = None,
        workspace: str = "",
    ) -> CodingAgentConfig:
        """
        Build a CodingAgentConfig with defaults from agents.yaml.
        
        Merges provided values with defaults from the registry.
        
        Args:
            agent_type: Agent type (defaults to default_agent from yaml)
            model: Model override (defaults to agent's default_model)
            debug_model: Debug model override (defaults to agent's default_debug_model)
            agent_specific: Agent-specific options to merge with defaults
            workspace: Working directory
            
        Returns:
            Complete CodingAgentConfig
        """
        # Use default agent if not specified
        agent_type = agent_type or cls._default_agent
        
        # Get defaults from registry
        defaults = cls._agent_configs.get(agent_type.lower(), {})
        
        # Merge agent_specific with defaults
        merged_agent_specific = {
            **defaults.get("agent_specific", {}),
            **(agent_specific or {}),
        }
        
        return CodingAgentConfig(
            agent_type=agent_type,
            model=model or defaults.get("default_model", "gpt-4.1-mini"),
            debug_model=debug_model or defaults.get("default_debug_model", "gpt-4.1-mini"),
            workspace=workspace,
            use_git=True,
            agent_specific=merged_agent_specific,
        )
    
    # =========================================================================
    # Registry Query Methods
    # =========================================================================
    
    @classmethod
    def list_available(cls) -> List[str]:
        """
        Return list of registered agent types.
        
        Returns:
            Sorted list of agent type names
        """
        return sorted(cls._registry.keys())
    
    @classmethod
    def is_available(cls, name: str) -> bool:
        """
        Check if an agent type is registered.
        
        Args:
            name: Agent type name
            
        Returns:
            True if agent is registered and available
        """
        return name.lower() in cls._registry
    
    @classmethod
    def get_agent_class(cls, name: str) -> Type[CodingAgentInterface]:
        """
        Get the agent class for a given type.
        
        Args:
            name: Agent type name
            
        Returns:
            Agent class implementing CodingAgentInterface
            
        Raises:
            ValueError: If agent not registered
        """
        name_lower = name.lower()
        if name_lower not in cls._registry:
            raise ValueError(f"Agent '{name}' not registered")
        return cls._registry[name_lower]
    
    # =========================================================================
    # Info Display Methods
    # =========================================================================
    
    @classmethod
    def print_agents_info(cls) -> None:
        """
        Print detailed info about all available agents.
        
        Useful for CLI --list-agents command.
        """
        print("\n" + "=" * 70)
        print("Available Coding Agents")
        print("=" * 70)
        
        for name in cls.list_available():
            info = cls._agent_configs.get(name, {})
            desc = info.get("description", "No description")
            install = info.get("install_command", "N/A")
            env_vars = info.get("env_vars", [])
            doc_url = info.get("documentation_url", "N/A")
            default_model = info.get("default_model", "N/A")
            
            print(f"\n  {name}:")
            print(f"    Description:   {desc}")
            print(f"    Default Model: {default_model}")
            print(f"    Install:       {install}")
            print(f"    Env vars:      {', '.join(env_vars) or 'None'}")
            print(f"    Docs:          {doc_url}")
        
        print("\n" + "=" * 70)
        print(f"Default agent: {cls._default_agent}")
        print("=" * 70 + "\n")


# =============================================================================
# Auto-Discovery and Registration
# =============================================================================

def _load_agents_yaml() -> Dict[str, Any]:
    """
    Load agents.yaml configuration file.
    
    Returns:
        Parsed YAML content, or empty dict if file not found
    """
    if not AGENTS_YAML_PATH.exists():
        print(f"[CodingAgentFactory] Warning: {AGENTS_YAML_PATH} not found")
        return {"agents": {}, "default_agent": "aider"}
    
    try:
        with open(AGENTS_YAML_PATH, 'r') as f:
            content = yaml.safe_load(f)
            return content if content else {"agents": {}, "default_agent": "aider"}
    except yaml.YAMLError as e:
        print(f"[CodingAgentFactory] Error parsing agents.yaml: {e}")
        return {"agents": {}, "default_agent": "aider"}


def _register_from_yaml() -> None:
    """
    Auto-register agents defined in agents.yaml.
    
    For each agent entry:
    1. Store its configuration
    2. Try to import the adapter module
    3. Register the adapter class
    """
    config = _load_agents_yaml()
    
    # Set default agent
    CodingAgentFactory._default_agent = config.get("default_agent", "aider")
    
    # Process each agent entry
    for agent_name, agent_info in config.get("agents", {}).items():
        # Store config for later use (even if import fails)
        CodingAgentFactory._agent_configs[agent_name.lower()] = agent_info
        
        # Get module and class info
        module_path = agent_info.get("adapter_module")
        class_name = agent_info.get("adapter_class")
        
        if not module_path or not class_name:
            print(f"[CodingAgentFactory] {agent_name}: Missing adapter_module or adapter_class")
            continue
        
        # Try to import and register
        try:
            module = importlib.import_module(module_path)
            agent_class = getattr(module, class_name)
            CodingAgentFactory.register(agent_name, agent_class)
        except ImportError as e:
            # Dependency not installed - this is expected for some agents
            print(f"[CodingAgentFactory] {agent_name} not available (import): {e}")
        except AttributeError as e:
            # Class not found in module
            print(f"[CodingAgentFactory] {agent_name} not available (class): {e}")


# Auto-register on module import
_register_from_yaml()

# Context Manager Factory
#
# Factory for creating context managers with auto-discovery.

import importlib
from pathlib import Path
from typing import Dict, Type, Any, Optional, List
import yaml

from src.execution.context_manager.base import ContextManager
from src.execution.context_manager.types import ExperimentHistoryProvider
from src.knowledge.search.base import KnowledgeSearch
from src.environment.handlers.base import ProblemHandler


class ContextManagerFactory:
    """
    Factory for creating context managers.
    
    Context managers are auto-discovered from the context_manager/ directory
    when this module is imported. Default configurations are loaded from
    context_manager.yaml.
    
    Usage:
        # Create context manager
        cm = ContextManagerFactory.create("kg_enriched", problem_handler, search_strategy)
        
        # List available
        ContextManagerFactory.list_available()
    """
    
    # Class-level state
    _registry: Dict[str, Type[ContextManager]] = {}
    _configs: Dict[str, Any] = {}
    _default_type: str = "kg_enriched"
    _initialized: bool = False
    
    # Configuration
    CONFIG_PATH = Path(__file__).parent / "context_manager.yaml"
    
    # =========================================================================
    # Initialization
    # =========================================================================
    
    @classmethod
    def _ensure_initialized(cls) -> None:
        """Lazy initialization of registry and configs."""
        if cls._initialized:
            return
        cls._load_config()
        cls._auto_discover()
        cls._initialized = True
    
    @classmethod
    def _load_config(cls) -> None:
        """Load configuration from YAML file."""
        if not cls.CONFIG_PATH.exists():
            return
        
        try:
            with open(cls.CONFIG_PATH, 'r') as f:
                content = yaml.safe_load(f) or {}
            cls._default_type = content.get("default_context_manager", "kg_enriched")
            cls._configs = content.get("context_managers", {})
        except yaml.YAMLError:
            pass
    
    @classmethod
    def _auto_discover(cls) -> None:
        """Auto-import all context manager modules in this directory."""
        module_dir = Path(__file__).parent
        
        for py_file in module_dir.glob("*.py"):
            if py_file.name.startswith("_") or py_file.name in ("base.py", "factory.py"):
                continue
            
            module_name = f"src.execution.context_manager.{py_file.stem}"
            try:
                importlib.import_module(module_name)
            except (ImportError, Exception):
                pass
    
    # =========================================================================
    # Registration
    # =========================================================================
    
    @classmethod
    def register(cls, name: str, context_manager_class: Type[ContextManager]) -> None:
        """Register a context manager class."""
        cls._registry[name.lower()] = context_manager_class
    
    # =========================================================================
    # Factory Methods
    # =========================================================================
    
    @classmethod
    def create(
        cls,
        context_manager_type: str,
        problem_handler: ProblemHandler,
        search_strategy: ExperimentHistoryProvider,
        knowledge_search: Optional[KnowledgeSearch] = None,
        params: Optional[Dict[str, Any]] = None,
        preset: Optional[str] = None,
    ) -> ContextManager:
        """
        Create a context manager instance.
        
        Args:
            context_manager_type: Name of registered context manager
            problem_handler: Problem handler instance
            search_strategy: Provider of experiment history
            knowledge_search: Optional knowledge search backend
            params: Context manager parameters (overrides preset)
            preset: Preset name to use
        
        Returns:
            Configured ContextManager instance
        """
        cls._ensure_initialized()
        
        cm_type = context_manager_type.lower()
        
        if cm_type not in cls._registry:
            available = ", ".join(sorted(cls._registry.keys()))
            raise ValueError(
                f"Unknown context manager: '{context_manager_type}'. "
                f"Available: {available or 'none registered'}"
            )
        
        # Resolve params from preset
        resolved_params = cls._resolve_params(cm_type, params, preset)
        
        return cls._registry[cm_type](
            problem_handler=problem_handler,
            search_strategy=search_strategy,
            knowledge_search=knowledge_search,
            params=resolved_params,
        )
    
    @classmethod
    def create_from_config(
        cls,
        config: Dict[str, Any],
        problem_handler: ProblemHandler,
        search_strategy: ExperimentHistoryProvider,
        knowledge_search: Optional[KnowledgeSearch] = None,
    ) -> ContextManager:
        """Create context manager from a config dictionary."""
        return cls.create(
            context_manager_type=config.get("type", cls._default_type),
            problem_handler=problem_handler,
            search_strategy=search_strategy,
            knowledge_search=knowledge_search,
            params=config.get("params"),
            preset=config.get("preset"),
        )
    
    @classmethod
    def _resolve_params(
        cls,
        cm_type: str,
        params: Optional[Dict[str, Any]],
        preset: Optional[str],
    ) -> Dict[str, Any]:
        """Resolve final params from preset and overrides."""
        if preset and not params:
            return cls.get_preset_params(cm_type, preset)
        elif preset and params:
            preset_params = cls.get_preset_params(cm_type, preset)
            preset_params.update(params)
            return preset_params
        return params or {}
    
    # =========================================================================
    # Configuration Access
    # =========================================================================
    
    @classmethod
    def get_preset_params(cls, context_manager_type: str, preset: str) -> Dict[str, Any]:
        """Get parameters for a preset."""
        cls._ensure_initialized()
        
        cm_config = cls._configs.get(context_manager_type.lower(), {})
        presets = cm_config.get("presets", {})
        
        if preset not in presets:
            return {}
        
        return presets[preset].get("params", {}).copy()
    
    @classmethod
    def get_default_preset(cls, context_manager_type: str) -> str:
        """Get the default preset name."""
        cls._ensure_initialized()
        cm_config = cls._configs.get(context_manager_type.lower(), {})
        return cm_config.get("default_preset", "DEFAULT")
    
    @classmethod
    def list_presets(cls, context_manager_type: str) -> List[str]:
        """List available presets."""
        cls._ensure_initialized()
        cm_config = cls._configs.get(context_manager_type.lower(), {})
        return sorted(cm_config.get("presets", {}).keys())
    
    @classmethod
    def get_default_type(cls) -> str:
        """Get the default context manager type."""
        cls._ensure_initialized()
        return cls._default_type
    
    # =========================================================================
    # Registry Access
    # =========================================================================
    
    @classmethod
    def list_available(cls) -> List[str]:
        """List all registered context managers."""
        cls._ensure_initialized()
        return sorted(cls._registry.keys())
    
    @classmethod
    def is_available(cls, name: str) -> bool:
        """Check if a context manager is registered."""
        cls._ensure_initialized()
        return name.lower() in cls._registry


def register_context_manager(name: str):
    """
    Decorator to register a context manager.
    
    Usage:
        @register_context_manager("kg_enriched")
        class KGEnrichedContextManager(ContextManager):
            ...
    """
    def decorator(cls: Type[ContextManager]) -> Type[ContextManager]:
        ContextManagerFactory.register(name, cls)
        return cls
    return decorator

# Knowledge Search Factory
#
# Factory for creating knowledge search backends with auto-discovery.

import importlib
from pathlib import Path
from typing import Dict, Type, Any, Optional, List
import yaml

from kapso.knowledge_base.search.base import (
    KnowledgeSearch,
    NullKnowledgeSearch,
)


class KnowledgeSearchFactory:
    """
    Factory for creating knowledge search backends.
    
    Search backends are auto-discovered from the knowledge/search/ directory
    when this module is imported. Default configurations are loaded from
    knowledge_search.yaml.
    
    Usage:
        # Create search backend
        search = KnowledgeSearchFactory.create("kg_llm_navigation")
        
        # Create from config (supports enabled: true/false)
        search = KnowledgeSearchFactory.create_from_config({"type": "kg_graph_search", "enabled": True})
        
        # Create null (disabled) search
        search = KnowledgeSearchFactory.create_null()
    """
    
    # Class-level state
    _registry: Dict[str, Type[KnowledgeSearch]] = {}
    _configs: Dict[str, Any] = {}
    _default_type: str = "kg_llm_navigation"
    _initialized: bool = False
    
    # Configuration
    CONFIG_PATH = Path(__file__).parent / "knowledge_search.yaml"
    
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
            cls._default_type = content.get("default_search", "kg_llm_navigation")
            cls._configs = content.get("searches", {})
        except yaml.YAMLError:
            pass
    
    @classmethod
    def _auto_discover(cls) -> None:
        """Auto-import all search modules in this directory."""
        module_dir = Path(__file__).parent
        
        for py_file in module_dir.glob("*.py"):
            if py_file.name.startswith("_") or py_file.name in ("base.py", "factory.py"):
                continue
            
            module_name = f"kapso.knowledge_base.search.{py_file.stem}"
            try:
                importlib.import_module(module_name)
            except (ImportError, Exception):
                pass
    
    # =========================================================================
    # Registration
    # =========================================================================
    
    @classmethod
    def register(cls, name: str, search_class: Type[KnowledgeSearch]) -> None:
        """Register a search backend class."""
        cls._registry[name.lower()] = search_class
    
    # =========================================================================
    # Factory Methods
    # =========================================================================
    
    @classmethod
    def create(
        cls,
        search_type: str,
        params: Optional[Dict[str, Any]] = None,
        preset: Optional[str] = None,
    ) -> KnowledgeSearch:
        """
        Create a knowledge search instance.
        
        Args:
            search_type: Name of registered search backend
            params: Search parameters (overrides preset)
            preset: Preset name to use
        
        Returns:
            Configured KnowledgeSearch instance
        """
        cls._ensure_initialized()
        
        s_type = search_type.lower()
        
        if s_type not in cls._registry:
            available = ", ".join(sorted(cls._registry.keys()))
            raise ValueError(
                f"Unknown knowledge search: '{search_type}'. "
                f"Available: {available or 'none registered'}"
            )
        
        # Resolve params from preset
        resolved_params = cls._resolve_params(s_type, params, preset)
        
        return cls._registry[s_type](params=resolved_params)
    
    @classmethod
    def create_null(cls) -> KnowledgeSearch:
        """Create a null (disabled) search backend."""
        return NullKnowledgeSearch()
    
    @classmethod
    def create_from_config(cls, config: Dict[str, Any]) -> KnowledgeSearch:
        """Create search backend from a config dictionary."""
        if not config:
            return cls.create_null()
        
        # Check if explicitly disabled
        if config.get("enabled") is False:
            return cls.create_null()
        
        return cls.create(
            search_type=config.get("type", cls._default_type),
            params=config.get("params"),
            preset=config.get("preset"),
        )
    
    @classmethod
    def _resolve_params(
        cls,
        search_type: str,
        params: Optional[Dict[str, Any]],
        preset: Optional[str],
    ) -> Dict[str, Any]:
        """
        Resolve final params from defaults, preset, and overrides.
        
        Priority (highest to lowest):
        1. params (explicit overrides)
        2. preset params
        3. config defaults
        """
        # Start with defaults from config
        resolved = cls.get_defaults(search_type)
        
        # Apply preset params
        if preset:
            preset_params = cls.get_preset_params(search_type, preset)
            resolved.update(preset_params)
        
        # Apply explicit overrides
        if params:
            resolved.update(params)
        
        return resolved
    
    # =========================================================================
    # Configuration Access
    # =========================================================================
    
    @classmethod
    def get_defaults(cls, search_type: str) -> Dict[str, Any]:
        """Get default parameters from config."""
        cls._ensure_initialized()
        s_config = cls._configs.get(search_type.lower(), {})
        return s_config.get("defaults", {}).copy()
    
    @classmethod
    def get_preset_params(cls, search_type: str, preset: str) -> Dict[str, Any]:
        """Get parameters for a preset."""
        cls._ensure_initialized()
        
        s_config = cls._configs.get(search_type.lower(), {})
        presets = s_config.get("presets", {})
        
        if preset not in presets:
            return {}
        
        return presets[preset].get("params", {}).copy()
    
    @classmethod
    def get_default_preset(cls, search_type: str) -> str:
        """Get the default preset name."""
        cls._ensure_initialized()
        s_config = cls._configs.get(search_type.lower(), {})
        return s_config.get("default_preset", "DEFAULT")
    
    @classmethod
    def list_presets(cls, search_type: str) -> List[str]:
        """List available presets."""
        cls._ensure_initialized()
        s_config = cls._configs.get(search_type.lower(), {})
        return sorted(s_config.get("presets", {}).keys())
    
    @classmethod
    def get_default_type(cls) -> str:
        """Get the default search type."""
        cls._ensure_initialized()
        return cls._default_type
    
    # =========================================================================
    # Registry Access
    # =========================================================================
    
    @classmethod
    def list_available(cls) -> List[str]:
        """List all registered search backends."""
        cls._ensure_initialized()
        return sorted(cls._registry.keys())
    
    @classmethod
    def is_available(cls, name: str) -> bool:
        """Check if a search backend is registered."""
        cls._ensure_initialized()
        return name.lower() in cls._registry


def register_knowledge_search(name: str):
    """
    Decorator to register a knowledge search backend.
    
    Usage:
        @register_knowledge_search("kg_llm_navigation")
        class KGLLMNavigationSearch(KnowledgeSearch):
            ...
    """
    def decorator(cls: Type[KnowledgeSearch]) -> Type[KnowledgeSearch]:
        KnowledgeSearchFactory.register(name, cls)
        return cls
    return decorator


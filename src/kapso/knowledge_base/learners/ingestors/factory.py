# Ingestor Factory
#
# Factory pattern for creating ingestors with decorator-based registration.
#
# Usage:
#     from kapso.knowledge_base.learners.ingestors import IngestorFactory, register_ingestor
#     
#     @register_ingestor("my_source")
#     class MyIngestor(Ingestor):
#         ...
#     
#     ingestor = IngestorFactory.create("my_source", param1=value1)

from typing import Dict, Type, Any, Optional, List

from kapso.knowledge_base.learners.ingestors.base import Ingestor


# Global registry for ingestor classes
_INGESTOR_REGISTRY: Dict[str, Type[Ingestor]] = {}


def register_ingestor(name: str):
    """
    Decorator to register an ingestor class.
    
    Usage:
        @register_ingestor("repo")
        class RepoIngestor(Ingestor):
            ...
    
    Args:
        name: Unique identifier for this ingestor type (matches source_type)
    """
    def decorator(cls: Type[Ingestor]):
        if name in _INGESTOR_REGISTRY:
            raise ValueError(f"Ingestor '{name}' is already registered")
        _INGESTOR_REGISTRY[name] = cls
        return cls
    return decorator


class IngestorFactory:
    """
    Factory for creating ingestor instances.
    
    Supports both direct creation and source-based dispatch.
    """
    
    @staticmethod
    def create(source_type: str, **params) -> Ingestor:
        """
        Create an ingestor by source type name.
        
        Args:
            source_type: Registered name (e.g., 'repo', 'solution')
            **params: Parameters passed to the ingestor constructor
            
        Returns:
            Configured Ingestor instance
            
        Raises:
            ValueError: If source_type is not registered
        """
        if source_type not in _INGESTOR_REGISTRY:
            available = list(_INGESTOR_REGISTRY.keys())
            raise ValueError(
                f"Unknown ingestor type: '{source_type}'. "
                f"Available: {available}"
            )
        
        ingestor_cls = _INGESTOR_REGISTRY[source_type]
        return ingestor_cls(params=params)
    
    @staticmethod
    def for_source(source: Any, **params) -> Ingestor:
        """
        Create an ingestor for a given source object.
        
        Automatically determines the source type from the source class name.
        
        Args:
            source: A Source.* object (Source.Repo, Source.Solution)
            **params: Parameters passed to the ingestor constructor
            
        Returns:
            Appropriate Ingestor instance for the source type
            
        Raises:
            ValueError: If no ingestor is registered for the source type
        """
        # Get source type from class name (e.g., "Repo" -> "repo")
        source_type = source.__class__.__name__.lower()
        return IngestorFactory.create(source_type, **params)
    
    @staticmethod
    def list_ingestors() -> List[str]:
        """List all registered ingestor types."""
        return list(_INGESTOR_REGISTRY.keys())
    
    @staticmethod
    def is_registered(source_type: str) -> bool:
        """Check if an ingestor type is registered."""
        return source_type in _INGESTOR_REGISTRY
    
    @staticmethod
    def print_ingestors_info() -> None:
        """Print information about all registered ingestors."""
        print("\nAvailable Ingestors:")
        print("=" * 40)
        for name, cls in _INGESTOR_REGISTRY.items():
            doc = cls.__doc__ or "No description"
            # Get first line of docstring
            first_line = doc.strip().split('\n')[0]
            print(f"  {name}: {first_line}")
        print()


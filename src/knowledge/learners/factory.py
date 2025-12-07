# Learner Factory
#
# Factory pattern for creating learners with decorator-based registration.
# Follows the same pattern as SearchStrategyFactory and ContextManagerFactory.
#
# Usage:
#     from src.knowledge.learners import LearnerFactory, register_learner
#     
#     @register_learner("my_learner")
#     class MyLearner(Learner):
#         ...
#     
#     learner = LearnerFactory.create("my_learner", param1=value1)

from typing import Dict, Type, Any, Optional, List

from src.knowledge.learners.base import Learner


# Global registry for learner classes
_LEARNER_REGISTRY: Dict[str, Type[Learner]] = {}


def register_learner(name: str):
    """
    Decorator to register a learner class.
    
    Usage:
        @register_learner("repo")
        class RepoLearner(Learner):
            ...
    
    Args:
        name: Unique identifier for this learner type
    """
    def decorator(cls: Type[Learner]):
        if name in _LEARNER_REGISTRY:
            raise ValueError(f"Learner '{name}' is already registered")
        _LEARNER_REGISTRY[name] = cls
        return cls
    return decorator


class LearnerFactory:
    """
    Factory for creating learner instances.
    
    Supports both direct creation and configuration-based creation.
    """
    
    @staticmethod
    def create(learner_type: str, **params) -> Learner:
        """
        Create a learner by type name.
        
        Args:
            learner_type: Registered name (e.g., 'repo', 'paper', 'solution')
            **params: Parameters passed to the learner constructor
            
        Returns:
            Configured Learner instance
            
        Raises:
            ValueError: If learner_type is not registered
        """
        if learner_type not in _LEARNER_REGISTRY:
            available = list(_LEARNER_REGISTRY.keys())
            raise ValueError(
                f"Unknown learner type: '{learner_type}'. "
                f"Available: {available}"
            )
        
        learner_cls = _LEARNER_REGISTRY[learner_type]
        return learner_cls(params=params)
    
    @staticmethod
    def list_learners() -> List[str]:
        """List all registered learner types."""
        return list(_LEARNER_REGISTRY.keys())
    
    @staticmethod
    def is_registered(learner_type: str) -> bool:
        """Check if a learner type is registered."""
        return learner_type in _LEARNER_REGISTRY
    
    @staticmethod
    def print_learners_info() -> None:
        """Print information about all registered learners."""
        print("\nAvailable Learners:")
        print("=" * 40)
        for name, cls in _LEARNER_REGISTRY.items():
            doc = cls.__doc__ or "No description"
            # Get first line of docstring
            first_line = doc.strip().split('\n')[0]
            print(f"  {name}: {first_line}")
        print()


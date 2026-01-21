# Stop Condition Factory
#
# Factory and registry for creating stop conditions.
# Use @register_stop_condition decorator to add new conditions.
#
# Usage:
#     from src.environment.stop_conditions import StopConditionFactory
#     
#     # Create by name
#     condition = StopConditionFactory.create("threshold", threshold=0.95)
#     
#     # List available
#     StopConditionFactory.list_conditions()

from typing import Any, Callable, Dict, List, Type

from src.environment.stop_conditions.base import StopCondition


class StopConditionFactory:
    """
    Factory for creating stop conditions.
    
    Stop conditions are registered via the @register_stop_condition decorator.
    This follows the same pattern as SearchStrategyFactory and ContextManagerFactory.
    """
    
    _registry: Dict[str, Type[StopCondition]] = {}
    _default: str = "from_eval"  # Default: stop when evaluate.py signals STOP
    
    @classmethod
    def register(cls, name: str) -> Callable[[Type[StopCondition]], Type[StopCondition]]:
        """
        Decorator to register a stop condition.
        
        Usage:
            @StopConditionFactory.register("my_condition")
            class MyCondition(StopCondition):
                ...
        """
        def decorator(cond_cls: Type[StopCondition]) -> Type[StopCondition]:
            cond_cls.name = name
            cls._registry[name.lower()] = cond_cls
            return cond_cls
        return decorator
    
    @classmethod
    def create(cls, condition_type: str, **params) -> StopCondition:
        """
        Create a stop condition instance.
        
        Args:
            condition_type: Name of registered condition
            **params: Parameters to pass to condition constructor
            
        Returns:
            Configured StopCondition instance
            
        Raises:
            ValueError: If condition_type is not registered
        """
        cond_type = condition_type.lower()
        if cond_type not in cls._registry:
            available = ", ".join(sorted(cls._registry.keys()))
            raise ValueError(
                f"Unknown stop condition: '{condition_type}'. "
                f"Available: {available}"
            )
        return cls._registry[cond_type](**params)
    
    @classmethod
    def create_from_config(cls, config: Dict[str, Any]) -> StopCondition:
        """
        Create stop condition from config dictionary.
        
        Args:
            config: Dict with 'type' and optional 'params'
            
        Returns:
            Configured StopCondition instance
        """
        condition_type = config.get("type", cls._default)
        params = config.get("params", {})
        return cls.create(condition_type, **params)
    
    @classmethod
    def list_conditions(cls) -> List[Dict[str, str]]:
        """
        List all registered stop conditions with metadata.
        
        Returns:
            List of dicts with name and description
        """
        return [
            {
                "name": name,
                "description": cond_cls.description,
            }
            for name, cond_cls in sorted(cls._registry.items())
        ]
    
    @classmethod
    def print_conditions_info(cls) -> None:
        """Print formatted info about all registered stop conditions."""
        print("\n" + "=" * 60)
        print("Available Stop Conditions")
        print("=" * 60)
        
        for info in cls.list_conditions():
            print(f"\n  {info['name']}")
            print(f"    {info['description']}")
        
        print("\n" + "=" * 60)
    
    @classmethod
    def get_default(cls) -> str:
        """Get the default stop condition type name."""
        return cls._default


# Convenience decorator alias
register_stop_condition = StopConditionFactory.register


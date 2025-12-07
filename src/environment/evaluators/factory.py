# Evaluator Factory
#
# Factory and registry for creating evaluators.
# Use @register_evaluator decorator to add new evaluators.
#
# Usage:
#     from src.environment.evaluators import EvaluatorFactory
#     
#     # Create by name
#     evaluator = EvaluatorFactory.create("regex_pattern", pattern=r"Accuracy: ([\d.]+)")
#     
#     # List available
#     EvaluatorFactory.list_evaluators()

from typing import Any, Callable, Dict, List, Type

from src.environment.evaluators.base import Evaluator


class EvaluatorFactory:
    """
    Factory for creating evaluators.
    
    Evaluators are registered via the @register_evaluator decorator.
    This follows the same pattern as SearchStrategyFactory and ContextManagerFactory.
    """
    
    _registry: Dict[str, Type[Evaluator]] = {}
    _default: str = "no_score"
    
    @classmethod
    def register(cls, name: str) -> Callable[[Type[Evaluator]], Type[Evaluator]]:
        """
        Decorator to register an evaluator.
        
        Usage:
            @EvaluatorFactory.register("my_evaluator")
            class MyEvaluator(Evaluator):
                ...
        """
        def decorator(evaluator_cls: Type[Evaluator]) -> Type[Evaluator]:
            evaluator_cls.name = name
            cls._registry[name.lower()] = evaluator_cls
            return evaluator_cls
        return decorator
    
    @classmethod
    def create(cls, evaluator_type: str, **params) -> Evaluator:
        """
        Create an evaluator instance.
        
        Args:
            evaluator_type: Name of registered evaluator
            **params: Parameters to pass to evaluator constructor
            
        Returns:
            Configured Evaluator instance
            
        Raises:
            ValueError: If evaluator_type is not registered
        """
        eval_type = evaluator_type.lower()
        if eval_type not in cls._registry:
            available = ", ".join(sorted(cls._registry.keys()))
            raise ValueError(
                f"Unknown evaluator: '{evaluator_type}'. "
                f"Available: {available}"
            )
        return cls._registry[eval_type](**params)
    
    @classmethod
    def create_from_config(cls, config: Dict[str, Any]) -> Evaluator:
        """
        Create evaluator from config dictionary.
        
        Args:
            config: Dict with 'type' and optional 'params'
            
        Returns:
            Configured Evaluator instance
        """
        evaluator_type = config.get("type", cls._default)
        params = config.get("params", {})
        return cls.create(evaluator_type, **params)
    
    @classmethod
    def list_evaluators(cls) -> List[Dict[str, Any]]:
        """
        List all registered evaluators with metadata.
        
        Returns:
            List of dicts with name, description, requires_llm
        """
        return [
            {
                "name": name,
                "description": eval_cls.description,
                "requires_llm": eval_cls.requires_llm,
            }
            for name, eval_cls in sorted(cls._registry.items())
        ]
    
    @classmethod
    def print_evaluators_info(cls) -> None:
        """Print formatted info about all registered evaluators."""
        print("\n" + "=" * 60)
        print("Available Evaluators")
        print("=" * 60)
        
        for info in cls.list_evaluators():
            llm_tag = " [LLM]" if info["requires_llm"] else ""
            print(f"\n  {info['name']}{llm_tag}")
            print(f"    {info['description']}")
        
        print("\n" + "=" * 60)
    
    @classmethod
    def get_default(cls) -> str:
        """Get the default evaluator type name."""
        return cls._default


# Convenience decorator alias
register_evaluator = EvaluatorFactory.register


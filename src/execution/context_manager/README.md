# Context Manager Module

Modular context managers for gathering experiment context in the orchestration loop.

## Overview

The Context Manager is responsible for gathering all relevant context needed for solution generation:
- Problem description (from problem handler)
- Experiment history (from search strategy)
- External knowledge (from knowledge retriever)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ContextManagerFactory                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │ kg_enriched │  │   cached    │  │  your_cm    │  ...     │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                      ContextManager                          │
│  ┌─────────────────┐  ┌─────────────────┐                   │
│  │ problem_handler │  │ search_strategy │                   │
│  └─────────────────┘  └─────────────────┘                   │
│  ┌─────────────────┐  ┌─────────────────┐                   │
│  │knowledge_retriever│ │     params      │                   │
│  └─────────────────┘  └─────────────────┘                   │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
                     ContextData
```

## Quick Start

### Using an Existing Context Manager

```python
from src.agents.orchestrator_agent.context_manager import ContextManagerFactory

# Create context manager
cm = ContextManagerFactory.create(
    context_manager_type="kg_enriched",
    problem_handler=problem_handler,
    search_strategy=search_strategy,
    knowledge_retriever=knowledge_retriever,
    params={"max_experiment_history_count": 5},
)

# Get context for solution generation
context = cm.get_context(budget_progress=50)
```

### List Available Context Managers

```python
from src.agents.orchestrator_agent.context_manager import ContextManagerFactory

# List all registered context managers
print(ContextManagerFactory.list_available())
# Output: ['kg_enriched']

# List presets for a context manager
print(ContextManagerFactory.list_presets("kg_enriched"))
# Output: ['DEFAULT', 'DEEP_CONTEXT', 'MINIMAL']
```

---

## Adding a New Context Manager

Follow these 4 steps to add a new context manager:

### Step 1: Create the Implementation File

Copy `_template.py` to create your new context manager:

```bash
cp _template.py my_context_manager.py
```

### Step 2: Implement Your Context Manager

Edit the new file:

```python
from typing import Any, Dict, Optional

from src.agents.orchestrator_agent.types import ContextData, ExperimentHistoryProvider
from src.agents.orchestrator_agent.context_manager.base import ContextManager
from src.agents.orchestrator_agent.context_manager.factory import register_context_manager
from src.agents.orchestrator_agent.knowledge_retriever.base import KnowledgeRetriever
from src.problem_handler.base import ProblemHandler


@register_context_manager("my_custom")  # <- Your unique identifier
class MyCustomContextManager(ContextManager):
    """My custom context manager."""
    
    def __init__(
        self,
        problem_handler: ProblemHandler,
        search_strategy: ExperimentHistoryProvider,
        knowledge_retriever: Optional[KnowledgeRetriever] = None,
        params: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(problem_handler, search_strategy, knowledge_retriever, params)
        
        # Extract your custom params
        self.my_param = self.params.get("my_param", "default_value")
        self.another_param = self.params.get("another_param", 10)
    
    def get_context(self, budget_progress: float = 0) -> ContextData:
        """Gather context for solution generation."""
        
        # Get problem description
        problem = self.problem_handler.get_problem_context(budget_progress=budget_progress)
        
        # Get experiment history
        history = self.search_strategy.get_experiment_history(best_last=True)
        
        # Your custom logic here...
        additional_info = f"Custom info: {self.my_param}"
        
        # Optional: Use knowledge retriever
        kg_results = ""
        if self.knowledge_retriever.is_enabled():
            result = self.knowledge_retriever.retrieve(query=problem)
            kg_results = result.text_results
        
        return ContextData(
            problem=problem,
            additional_info=additional_info,
            kg_results=kg_results,
            kg_code_results="",
        )
```

### Step 3: Add Configuration Presets

Edit `context_manager.yaml`:

```yaml
context_managers:
  # ... existing entries ...
  
  my_custom:
    description: "My custom context manager"
    presets:
      DEFAULT:
        params:
          my_param: "default_value"
          another_param: 10
      ADVANCED:
        params:
          my_param: "advanced_value"
          another_param: 50
    default_preset: "DEFAULT"
```

### Step 4: Configure in Benchmark

In your benchmark's `config.yaml`:

```yaml
context_manager:
  type: "my_custom"
  params:
    my_param: "custom_value"
    another_param: 20
```

Or use a preset:

```yaml
context_manager:
  type: "my_custom"
  preset: "ADVANCED"
```

**That's it!** Your context manager is auto-discovered and ready to use.

---

## File Structure

```
context_manager/
├── README.md              # This file
├── __init__.py            # Module exports
├── base.py                # Abstract base class
├── factory.py             # Factory with auto-discovery
├── context_manager.yaml   # Configuration presets
├── _template.py           # Template for new implementations
└── kg_enriched_context_manager.py  # Default implementation
```

## Base Class API

```python
class ContextManager(ABC):
    """Abstract base class for context managers."""
    
    def __init__(
        self,
        problem_handler: ProblemHandler,
        search_strategy: ExperimentHistoryProvider,
        knowledge_retriever: Optional[KnowledgeRetriever] = None,
        params: Optional[Dict[str, Any]] = None,
    ):
        self.problem_handler = problem_handler
        self.search_strategy = search_strategy
        self.knowledge_retriever = knowledge_retriever or NullKnowledgeRetriever()
        self.params = params or {}
    
    @abstractmethod
    def get_context(self, budget_progress: float = 0) -> ContextData:
        """Gather context for solution generation."""
        pass
```

## ContextData Structure

```python
@dataclass
class ContextData:
    problem: str            # Problem description
    additional_info: str    # Experiment history and other info
    kg_results: str = ""    # Knowledge text results
    kg_code_results: str = ""  # Knowledge code snippets
```

## Factory API

| Method | Description |
|--------|-------------|
| `create(type, ...)` | Create context manager instance |
| `create_from_config(config, ...)` | Create from config dictionary |
| `list_available()` | List all registered context managers |
| `list_presets(type)` | List available presets for a type |
| `get_preset_params(type, preset)` | Get params for a preset |
| `is_available(name)` | Check if a type is registered |

## Tips

1. **Auto-discovery**: Just drop a `.py` file in this directory with `@register_context_manager()` - it's automatically discovered.

2. **Params flexibility**: Each context manager can have completely different params. Just document them in the YAML description.

3. **Knowledge retriever**: Always check `is_enabled()` before calling `retrieve()` to support disabled mode.

4. **Testing**: Test your context manager with different budget_progress values (0-100).


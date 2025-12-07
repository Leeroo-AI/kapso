# Search Strategies

Modular search strategies for experiment generation. Add your own strategy by following the simple steps below.

## Quick Start

```bash
# 1. Copy the template
cp _template.py my_awesome_strategy.py

# 2. Edit and implement your strategy
#    - Change @register_strategy("my_strategy") to your name
#    - Implement the abstract methods

# 3. Test it works
python -c "from src.agents.orchestrator_agent.search_strategies import SearchStrategyFactory; print(SearchStrategyFactory.list_available())"

# 4. Use in config.yaml
#    search_strategy:
#      type: "my_awesome_strategy"
```

## Architecture

```
search_strategies/
├── __init__.py          # Public exports
├── base.py              # SearchStrategy ABC + ExperimentResult
├── factory.py           # Registry + factory (auto-discovers strategies)
├── strategies.yaml      # Default presets for all strategies
├── _template.py         # 👈 COPY THIS to create your strategy
├── llm_tree_search.py   # Built-in: LLM-guided tree search
└── README.md            # This file
```

## Creating a New Strategy

### Step 1: Copy Template

```bash
cp _template.py my_strategy.py
```

### Step 2: Implement Required Methods

```python
from src.agents.orchestrator_agent.search_strategies import (
    SearchStrategy,
    SearchStrategyConfig,
    ExperimentResult,
    register_strategy,
)

@register_strategy("my_strategy")  # 👈 Unique name
class MyStrategy(SearchStrategy):
    
    def __init__(self, config: SearchStrategyConfig):
        super().__init__(config)
        # Access params: self.params.get("my_param", default)
        # Available: self.problem_handler, self.llm, self.workspace
    
    def run(self, context, budget_progress):
        """Main search loop - called each iteration."""
        # 1. Generate solutions using self.llm
        # 2. Implement using self._implement_n_debug()
        # 3. Store results in self.experiment_history
        pass
    
    def get_experiment_history(self, best_last=False):
        """Return all experiments (sorted if best_last=True)."""
        return self.experiment_history
    
    def get_best_experiment(self):
        """Return best successful experiment."""
        valid = [e for e in self.experiment_history if not e.had_error]
        return max(valid, key=lambda x: x.score) if valid else None
    
    def checkout_to_best_experiment_branch(self):
        """Checkout git to best experiment's branch."""
        best = self.get_best_experiment()
        if best:
            self.workspace.switch_branch(best.branch_name)
```

### Step 3: Add Presets (Optional)

Add to `strategies.yaml`:

```yaml
strategies:
  my_strategy:
    description: "My awesome search strategy"
    presets:
      FAST:
        params:
          iterations: 5
          code_debug_tries: 2
      THOROUGH:
        params:
          iterations: 20
          code_debug_tries: 5
    default_preset: "FAST"
```

### Step 4: Use in Benchmark Config

```yaml
# benchmarks/mle/config.yaml
modes:
  MY_MODE:
    search_strategy:
      type: "my_strategy"
      params:
        iterations: 10
        my_custom_param: "value"
    coding_agent:
      type: "aider"
```

## Available Base Class Features

When you extend `SearchStrategy`, you get:

| Attribute/Method | Description |
|-----------------|-------------|
| `self.problem_handler` | Run code, evaluate solutions |
| `self.llm` | LLM backend for generating solutions |
| `self.workspace` | Experiment workspace manager, creates branches |
| `self.params` | Your strategy's config params |
| `self._implement_n_debug()` | Implement solution + debug loop |
| `self.implement_solution()` | Generate code for a solution |
| `self.debug_solution()` | Debug a failed solution |

## Example: Simple Random Search

```python
@register_strategy("random_search")
class RandomSearch(SearchStrategy):
    """Simple baseline: generate N random solutions per iteration."""
    
    def __init__(self, config):
        super().__init__(config)
        self.samples_per_iter = self.params.get("samples_per_iteration", 3)
        self.experiment_history = []
    
    def run(self, context, budget_progress):
        for i in range(self.samples_per_iter):
            solution = self._generate_random_solution(context)
            branch = f"exp_{len(self.experiment_history)}"
            
            result = self._implement_n_debug(
                solution, context,
                code_debug_tries=2,
                branch_name=branch,
                parent_branch_name="main"
            )
            
            self.experiment_history.append(ExperimentResult(
                node_id=len(self.experiment_history),
                solution=solution,
                score=result.score,
                branch_name=branch,
                had_error=result.run_had_error,
            ))
    
    def _generate_random_solution(self, context):
        return self.llm.llm_completion(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": f"Propose a solution for: {context.problem}"}]
        )
    
    def get_experiment_history(self, best_last=False):
        if best_last:
            return sorted(self.experiment_history, key=lambda x: (not x.had_error, x.score))
        return self.experiment_history
    
    def get_best_experiment(self):
        valid = [e for e in self.experiment_history if not e.had_error]
        return max(valid, key=lambda x: x.score) if valid else None
    
    def checkout_to_best_experiment_branch(self):
        best = self.get_best_experiment()
        if best:
            self.workspace.switch_branch(best.branch_name)
```

## Factory API Reference

```python
from src.agents.orchestrator_agent.search_strategies import (
    SearchStrategyFactory
)

# List all registered strategies
SearchStrategyFactory.list_available()
# ['llm_tree_search', 'my_strategy']

# Check if strategy exists
SearchStrategyFactory.is_available("my_strategy")
# True

# Get preset params
SearchStrategyFactory.get_preset_params("llm_tree_search", "MINIMAL")
# {'code_debug_tries': 2, 'node_expansion_limit': 2, ...}

# List presets for a strategy
SearchStrategyFactory.list_presets("llm_tree_search")
# ['HEAVY_EXPERIMENTATION', 'HEAVY_THINKING', 'MINIMAL', 'PRODUCTION']

# Create strategy instance
strategy = SearchStrategyFactory.create(
    strategy_type="my_strategy",
    problem_handler=handler,
    llm=llm_backend,
    coding_agent_config=config,
    params={"my_param": 10},
)

# Print all strategy info
SearchStrategyFactory.print_strategies_info()
```

## Tips

1. **Start Simple**: Begin with a basic implementation, then add complexity
2. **Use Presets**: Define presets in `strategies.yaml` for easy configuration
3. **Log Progress**: Print iteration info for debugging
4. **Handle Errors**: Check `result.run_had_error` before storing scores
5. **Parallel Execution**: Use `ThreadPoolExecutor` for parallel experiments (see `llm_tree_search.py`)

## Need Help?

- Check `llm_tree_search.py` for a complete, production-ready example
- Check `_template.py` for annotated starter code
- Check `base.py` for the full `SearchStrategy` interface


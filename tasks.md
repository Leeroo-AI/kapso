# Search Strategy Refactoring - COMPLETED

## Summary

Successfully refactored search strategies to simplify the architecture:

### Changes Made

1. **Renamed `basic_linear_search` to `generic`**
   - `basic_linear_search.py` → `generic.py`
   - `BasicLinearSearch` → `GenericSearch`
   - Registration: `@register_strategy("generic")`
   - Branch naming: `generic_exp_` prefix

2. **Archived `linear_search` and `llm_tree_search`**
   - Both files moved to `/archive/`
   - All imports and references removed from active code

3. **Refactored `benchmark_tree_search`**
   - Now self-contained with all tree logic copied from `llm_tree_search`
   - Includes `TreeSearchNode` dataclass
   - Includes tree operations: `expand()`, `select()`, `prune_bad_solutions()`, `solution_generation()`
   - Inherits directly from `SearchStrategy` (not `LlmSteeredTreeSearch`)

4. **Updated configurations**
   - `src/config.yaml`: Updated to use `generic` strategy
   - `benchmarks/ale/config.yaml`: Removed `context_manager` sections
   - `benchmarks/mle/config.yaml`: Removed `context_manager` sections, removed LINEAR mode
   - `src/execution/search_strategies/strategies.yaml`: Updated presets

5. **Updated code references**
   - `src/execution/orchestrator.py`: Default strategy changed to `generic`
   - `src/execution/search_strategies/factory.py`: Default strategy changed to `generic`
   - Tests updated to use `generic` and `benchmark_tree_search`

6. **Updated documentation**
   - `docs/evolve/search-strategies.mdx`
   - `docs/evolve/orchestrator.mdx`
   - `docs/evolve/architecture.mdx`
   - `docs/evolve/execution-flow.mdx`
   - `src/execution/search_strategies/README.md`

## Final Architecture

### Active Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `generic` | Claude Code + MCP gates for ideation and implementation | General problem solving |
| `benchmark_tree_search` | Tree-based exploration with handler.run() | MLE-Bench, ALE-Bench |

### Archived Strategies

- `linear_search.py` → `/archive/`
- `llm_tree_search.py` → `/archive/`
- `basic_linear_search.py` → `/archive/`

### File Structure

```
src/execution/search_strategies/
├── __init__.py
├── _template.py
├── base.py
├── benchmark_tree_search.py  # Self-contained tree search
├── factory.py
├── generic.py                # Main strategy (renamed from basic_linear_search)
├── README.md
└── strategies.yaml
```

## Verification

- All Python files compile without errors
- No remaining references to archived strategies in active code
- Tests updated to use new strategy names
- Documentation updated to reflect new architecture

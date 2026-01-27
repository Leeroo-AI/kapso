# Task 09: Benchmark Backward Compatibility

## Overview

The benchmarks (`benchmarks/mle/` and `benchmarks/ale/`) use a different evaluation flow than the new `Kapso.evolve()` design. They have custom `ProblemHandler` subclasses with built-in evaluation logic via `handler.run()`.

**Goal:** Create a separate search strategy for benchmarks that uses `handler.run()` for evaluation, keeping the core `src/execution/` code intact for the new design.

## Problem Statement

### New Design (for `Kapso.evolve()`)
- Developer agent builds evaluation in `kapso_evaluation/`
- Developer agent runs evaluation and returns structured JSON
- `FeedbackGenerator` validates evaluation and decides stop/continue
- `SearchNode` is the unified data structure

### Benchmark Design (MLE-Bench, ALE-Bench)
- Custom `ProblemHandler` subclasses with built-in evaluation logic
- `handler.run()` executes code and grades with external tools (mlebench, ale_bench)
- `handler.stop_condition()` decides when to stop (e.g., `got_medal`)
- Returns `ProblemRunResult` with score, errors, feedbacks

### Key Differences

| Aspect | New Design | Benchmarks |
|--------|------------|------------|
| Evaluation | Agent builds dynamically | Handler has built-in `run()` |
| Stop decision | FeedbackGenerator | Handler's `stop_condition()` |
| Score source | Agent JSON output | Handler's `run()` result |
| Result type | `SearchNode` | `ProblemRunResult` |

## Solution: Hybrid Approach

1. Add shared utility methods to `SearchStrategy` base class
2. Create `BenchmarkTreeSearch` that inherits from `LlmSteeredTreeSearch`
3. Override only the evaluation step to use `handler.run()`
4. Update benchmark configs to use the new strategy

## Changes Required

### 1. Modify `src/execution/search_strategies/base.py`

Add two utility methods after `_generate_feedback()`:

```python
def _evaluate_with_handler(self, node: SearchNode, solution: str) -> SearchNode:
    """
    Evaluate using handler.run() for benchmark compatibility.
    
    Maps ProblemRunResult fields to SearchNode:
    - result.score -> node.score
    - result.output -> node.evaluation_output
    - result.run_had_error -> node.had_error
    - result.error_message -> node.error_message
    - result.feedbacks -> node.feedback
    """
    # Implementation details in code

def _check_handler_stop_condition(self) -> bool:
    """Check handler's stop_condition() for benchmark compatibility."""
    # Implementation details in code
```

### 2. Create `src/execution/search_strategies/benchmark_tree_search.py`

New file that inherits from `LlmSteeredTreeSearch`:

```python
@register_strategy("benchmark_tree_search")
class BenchmarkTreeSearch(LlmSteeredTreeSearch):
    """
    Tree search for benchmarks (MLE-Bench, ALE-Bench).
    
    Inherits all tree search logic but uses handler.run() for evaluation
    instead of agent-based evaluation with JSON extraction.
    """
    
    def _run_for_node(self, node, context, branch_name, budget_progress):
        # Same as parent but:
        # - Calls _evaluate_with_handler() instead of _extract_agent_result()
        # - Skips _generate_feedback() (handler provides feedback)
        # - Checks handler.stop_condition() for should_stop
```

### 3. Update `benchmarks/ale/config.yaml`

Change `search_strategy.type` in all modes:

```yaml
# All modes (ALE_CONFIGS, ALE_CONFIGS_LONG, HEAVY_THINKING, MINIMAL)
search_strategy:
  type: "benchmark_tree_search"  # was: "llm_tree_search"
```

### 4. Update `benchmarks/mle/config.yaml`

Change `search_strategy.type` in tree search modes:

```yaml
# Modes: MLE_CONFIGS, HEAVY_EXPERIMENTATION, MINIMAL
search_strategy:
  type: "benchmark_tree_search"  # was: "llm_tree_search"

# LINEAR mode stays as "linear_search" (testing only, not used by benchmarks)
```

## Flow Comparison

**Original `LlmSteeredTreeSearch._run_for_node()`:**
```
implement → _extract_agent_result() → _generate_feedback() → append to history
```

**New `BenchmarkTreeSearch._run_for_node()`:**
```
implement → _evaluate_with_handler() → _check_handler_stop_condition() → append to history
```

## Files Summary

| File | Action | Description |
|------|--------|-------------|
| `src/execution/search_strategies/base.py` | MODIFY | Add `_evaluate_with_handler()` and `_check_handler_stop_condition()` |
| `src/execution/search_strategies/benchmark_tree_search.py` | CREATE | New strategy using handler evaluation |
| `benchmarks/ale/config.yaml` | MODIFY | Change type to `benchmark_tree_search` (4 modes) |
| `benchmarks/mle/config.yaml` | MODIFY | Change type to `benchmark_tree_search` (3 modes) |

## Testing

### Test 1: MLE-Bench Still Works
```bash
python -m benchmarks.mle.runner --competition tabular-playground-series-dec-2021 --iterations 2 --mode MINIMAL
```

### Test 2: ALE-Bench Still Works
```bash
python -m benchmarks.ale.runner --problem ahc039 --iterations 2 --mode MINIMAL
```

### Test 3: New Kapso.evolve() Still Works
```bash
python tests/demo_task1_initialize_repo.py
```

## Benefits

1. **No changes to core logic** - `LlmSteeredTreeSearch` and `LinearSearch` remain unchanged
2. **Shared utility** - `_evaluate_with_handler()` in base class can be reused
3. **Clean separation** - Benchmark-specific behavior isolated in one file
4. **Minimal config changes** - Just change the strategy type name

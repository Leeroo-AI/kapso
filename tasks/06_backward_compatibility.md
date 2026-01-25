# Task 6: Backward Compatibility with Benchmarks

## Overview

The benchmarks (`benchmarks/mle/` and `benchmarks/ale/`) use a different entry point than `Kapso.evolve()`. They directly use `OrchestratorAgent` with custom `ProblemHandler` subclasses.

**Key insight:** Benchmarks have their own evaluation logic built into their handlers. They don't need the new "agent-builds-evaluation" design.

## Current Benchmark Architecture

```
benchmarks/mle/runner.py
    │
    ├─► MleBenchHandler (custom ProblemHandler)
    │     - run(): Executes code, validates submission, grades with mlebench
    │     - final_evaluate(): Grades on private test set
    │     - stop_condition(): Returns True when got_medal
    │
    └─► OrchestratorAgent(problem_handler, config_path, ...)
          └─► orchestrator.solve()
```

```
benchmarks/ale/runner.py
    │
    ├─► AleBench (custom ProblemHandler)
    │     - run(): Executes code, evaluates with ale_bench
    │     - final_evaluate(): Private evaluation
    │     - stop_condition(): Returns False (never stops early)
    │
    └─► OrchestratorAgent(problem_handler, config_path, ...)
          └─► orchestrator.solve()
```

## Why Benchmarks Are Different

| Aspect | Kapso.evolve() (new design) | Benchmarks |
|--------|----------------------------|------------|
| Evaluation | Agent builds dynamically | Handler has built-in evaluation |
| Stop condition | Feedback generator decides | Handler's stop_condition() |
| Problem context | Goal + constraints | Competition/problem description |
| Entry point | `Kapso.evolve()` | `OrchestratorAgent` directly |

## Backward Compatibility Strategy

### Option 1: Dual-Mode OrchestratorAgent (Recommended)

Keep `OrchestratorAgent` supporting both modes:

1. **Legacy mode** (for benchmarks): 
   - Uses `problem_handler.run()` for evaluation
   - Uses `problem_handler.stop_condition()` for stopping
   - No feedback generator

2. **New mode** (for Kapso.evolve()):
   - Agent builds evaluation
   - Feedback generator decides stop
   - No predefined evaluator

**Detection:** Check if `problem_handler` has a real `stop_condition()` implementation or if it's the simplified version.

```python
class OrchestratorAgent:
    def __init__(self, problem_handler, ..., use_feedback_generator: bool = None):
        # Auto-detect mode if not specified
        if use_feedback_generator is None:
            # Legacy handlers have meaningful stop_condition
            # New handlers use feedback generator
            self.use_feedback_generator = not self._has_custom_stop_condition(problem_handler)
        else:
            self.use_feedback_generator = use_feedback_generator
    
    def _has_custom_stop_condition(self, handler) -> bool:
        """Check if handler has custom stop_condition (legacy mode)."""
        # Check if it's a benchmark handler
        return hasattr(handler, 'maximize_scoring') and hasattr(handler, 'problem_id')
```

### Option 2: Separate Orchestrators

Create two orchestrator classes:

1. `LegacyOrchestratorAgent` - Current behavior for benchmarks
2. `EvolveOrchestratorAgent` - New behavior with feedback generator

**Pros:** Clean separation
**Cons:** Code duplication, maintenance burden

### Option 3: Handler Interface Extension

Add optional methods to `ProblemHandler`:

```python
class ProblemHandler(ABC):
    # Existing (required for benchmarks)
    @abstractmethod
    def run(self, file_path: str, ...) -> ProblemRunResult:
        pass
    
    @abstractmethod
    def stop_condition(self, **kwargs) -> bool:
        pass
    
    # New (optional, for new design)
    def uses_agent_evaluation(self) -> bool:
        """Return True if agent should build evaluation."""
        return False  # Default: use handler's run() for evaluation
```

## Recommended Implementation

### Keep Existing Files for Benchmarks

**DO NOT DELETE:**
- `src/environment/handlers/base.py` - Keep `ProblemHandler` and `ProblemRunResult`
- `src/environment/handlers/generic.py` - Keep but simplify for new design

**Benchmarks continue to work because:**
- They use `OrchestratorAgent` directly
- They provide custom `ProblemHandler` with `run()` and `stop_condition()`
- They don't use `Kapso.evolve()`

### Modify OrchestratorAgent for Dual Mode

```python
# src/execution/orchestrator.py

class OrchestratorAgent:
    def __init__(
        self,
        problem_handler: ProblemHandler,
        config_path: Optional[str] = None,
        mode: Optional[str] = None,
        coding_agent: Optional[str] = None,
        is_kg_active: bool = False,
        knowledge_search: Optional[KnowledgeSearch] = None,
        workspace_dir: Optional[str] = None,
        starting_repo_path: Optional[str] = None,
        # NEW: Explicit mode selection
        use_feedback_generator: bool = False,
        feedback_generator: Optional[FeedbackGenerator] = None,
    ):
        self.problem_handler = problem_handler
        self.use_feedback_generator = use_feedback_generator
        self.feedback_generator = feedback_generator
        # ... rest of init
    
    def solve(self, experiment_max_iter: int = 20, ...):
        for i in range(experiment_max_iter):
            # ... budget progress calculation
            
            if self.use_feedback_generator:
                # NEW MODE: Agent builds evaluation, feedback generator decides stop
                result = self.search_strategy.run(context, budget_progress)
                
                feedback = self.feedback_generator.generate(
                    goal=self.goal,
                    idea=result.solution,
                    implementation=result.code_diff,
                    evaluation_result=result.evaluation_output,
                )
                
                if feedback.stop:
                    break
                
                context.feedback = feedback.feedback
            else:
                # LEGACY MODE: Handler evaluates, handler decides stop
                if self.problem_handler.stop_condition() or budget_progress >= 100:
                    break
                
                context = self.context_manager.get_context(budget_progress)
                
                if self.context_manager.should_stop():
                    break
                
                self.search_strategy.run(context, budget_progress)
```

### Kapso.evolve() Uses New Mode

```python
# src/kapso.py

def evolve(self, goal, eval_dir=None, data_dir=None, initial_repo=None, ...):
    # ... setup
    
    # Create simplified handler (no evaluation logic)
    handler = SimpleProblemHandler(goal=goal)
    
    # Create feedback generator
    feedback_generator = FeedbackGenerator(coding_agent_config)
    
    # Create orchestrator in NEW mode
    orchestrator = OrchestratorAgent(
        handler,
        use_feedback_generator=True,  # NEW MODE
        feedback_generator=feedback_generator,
        # ... other params
    )
    
    orchestrator.solve(experiment_max_iter=max_iterations)
```

### Benchmarks Continue Using Legacy Mode

```python
# benchmarks/mle/runner.py (NO CHANGES NEEDED)

problem_handler = MleBenchHandler(competition_id)
orchestrator = OrchestratorAgent(
    problem_handler,
    config_path=CONFIG_PATH,
    # use_feedback_generator defaults to False
)
orchestrator.solve(experiment_max_iter=max_iterations)
```

## Files to Modify

1. **`src/execution/orchestrator.py`**
   - Add `use_feedback_generator` parameter
   - Add dual-mode logic in `solve()`

2. **`src/environment/handlers/base.py`**
   - Keep as-is (backward compatible)

3. **`src/environment/handlers/generic.py`**
   - Simplify for new design
   - Keep backward compatible interface

## Files to Keep (NOT Delete)

Despite `05_cleanup_dependencies.md`, we should **NOT delete**:
- `src/environment/handlers/base.py` - Needed by benchmarks
- `src/environment/handlers/generic.py` - Can be simplified but kept

We **CAN delete** (benchmarks don't use these):
- `src/environment/evaluators/` - Only used by GenericProblemHandler
- `src/environment/stop_conditions/` - Only used by GenericProblemHandler

## Testing

### Test 1: MLE-Bench Still Works

```bash
# Should work exactly as before
python -m benchmarks.mle.runner --competition tabular-playground-series-dec-2021 --iterations 5
```

### Test 2: ALE-Bench Still Works

```bash
# Should work exactly as before
python -m benchmarks.ale.runner --problem ahc039 --iterations 5
```

### Test 3: New Kapso.evolve() Works

```python
kapso = Kapso()
solution = kapso.evolve(
    goal="Build a text classifier",
    eval_dir="./my_eval/",
    data_dir="./my_data/",
)
```

## Summary

**Backward compatibility is achieved by:**

1. Keeping `ProblemHandler` interface unchanged
2. Adding `use_feedback_generator` flag to `OrchestratorAgent`
3. Benchmarks continue using legacy mode (default)
4. `Kapso.evolve()` uses new mode with feedback generator
5. Only deleting modules that benchmarks don't use (evaluators, stop_conditions)

## Cross-References
- Updates: `03_developer_agent_loop.md` - Don't delete handler base
- Updates: `05_cleanup_dependencies.md` - Keep handler files

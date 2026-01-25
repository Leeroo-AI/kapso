# Evolve System Redesign - Task Index

## Overview

This directory contains implementation tasks for the new Evolve system design.
See `/design.md` for the full architecture specification.

## Design Summary

```
evolve(goal, eval_dir?, data_dir?, initial_repo?)
    │
    ├─► 1. INITIALIZE REPO          → 01_initialize_repo.md
    ├─► 2. SETUP DIRECTORIES        → 02_setup_directories.md
    ├─► 3. DEVELOPER AGENT LOOP     → 03_developer_agent_loop.md
    └─► 4. RETURN RESULT            → 04_return_result.md

Additional:
    ├─► 5. CLEANUP & DEPENDENCIES   → 05_cleanup_dependencies.md
    └─► 6. BACKWARD COMPATIBILITY   → 06_backward_compatibility.md
```

## Task Files

| Task | File | Description | Complexity |
|------|------|-------------|------------|
| 1 | `01_initialize_repo.md` | Workflow search integration, repo initialization | Medium |
| 2 | `02_setup_directories.md` | Setup kapso_evaluation/ and kapso_datasets/ | Low |
| 3 | `03_developer_agent_loop.md` | New iteration flow, feedback generator | High |
| 4 | `04_return_result.md` | Result structure updates | Low |
| 5 | `05_cleanup_dependencies.md` | Remove evaluators/stop_conditions | Medium |
| 6 | `06_backward_compatibility.md` | Keep benchmarks working | Medium |

## Recommended Implementation Order

1. **`06_backward_compatibility.md`** - Understand what to preserve for benchmarks
2. **`05_cleanup_dependencies.md`** - Remove old code (but keep handler base)
3. **`01_initialize_repo.md`** - Setup repo initialization with workflow search
4. **`02_setup_directories.md`** - Add directory setup logic
5. **`03_developer_agent_loop.md`** - Implement new iteration flow (largest task)
6. **`04_return_result.md`** - Update result handling

## Key Changes Summary

### API Changes

**Before:**
```python
kapso.evolve(
    goal="...",
    starting_repo_path="/path/to/repo",
    data_dir="/path/to/data",
    evaluator="script",
    evaluator_params={},
    stop_condition="threshold",
    stop_condition_params={"threshold": 0.95},
    language="python",
    main_file="main.py",
    timeout=300,
)
```

**After:**
```python
kapso.evolve(
    goal="...",
    eval_dir="/path/to/evaluation",   # NEW
    data_dir="/path/to/data",          # Behavior change
    initial_repo="/path/to/repo",      # Renamed, accepts URL too
    max_iterations=10,
)
```

### Modules to Delete

- `src/environment/evaluators/` (entire directory)
- `src/environment/stop_conditions/` (entire directory)

### Modules to Keep (for benchmark compatibility)

- `src/environment/handlers/base.py` - ProblemHandler interface
- `src/environment/handlers/generic.py` - Simplified version

### Modules to Add

- `src/execution/feedback_generator.py`
- `src/execution/prompts/feedback_generator.md`

### Major Modifications

- `src/kapso.py` - evolve() method
- `src/execution/orchestrator.py` - dual-mode solve() loop
- `src/execution/search_strategies/base.py` - iteration logic
- `src/execution/prompts/coding_agent_implement.md` - agent instructions

## Backward Compatibility

**Benchmarks (`benchmarks/mle/`, `benchmarks/ale/`) continue to work because:**

1. They use `OrchestratorAgent` directly (not `Kapso.evolve()`)
2. They provide custom `ProblemHandler` with `run()` and `stop_condition()`
3. `OrchestratorAgent` supports dual-mode:
   - Legacy mode (default): Handler evaluates, handler decides stop
   - New mode: Agent builds evaluation, feedback generator decides stop

See `06_backward_compatibility.md` for details.

## Testing Strategy

1. Unit tests for each new component
2. Integration test for full evolve() flow
3. **Benchmark regression tests** - Ensure MLE-Bench and ALE-Bench still work
4. Test with all three use cases from design.md:
   - GPU Kernel Optimization
   - LLM Fine-tuning (DPO)
   - Multi-Agent System
5. End-to-end test with `data/wikis_llm_finetuning_test/` (PicoGPT workflow)

## Notes

- Backward compatible with existing benchmarks
- All changes can be made in a single PR
- Dual-mode orchestrator enables gradual migration

# Task 06: Remove `problem_handler.run()` from Implementation Flow

## Issue Description

In the current implementation, after the coding agent implements a solution, we call `problem_handler.run()` to execute `main.py`:

```python
# src/execution/search_strategies/base.py lines 319-325
session.generate_code(developer_prompt)  # Coding agent implements
return self.problem_handler.run(         # ← Then we run main.py separately
    session.session_folder,
    run_data_dir=session.run_dir,
    solution=solution,
)
```

This is **redundant** because in the new design (see `design.md`), the **developer agent itself** is responsible for:
1. Implementing the solution
2. Building evaluation in `kapso_evaluation/`
3. **Running the evaluation**
4. Reporting results

The `problem_handler.run()` call just executes `main.py` and checks the return code, which duplicates what the developer agent already does.

## Current Flow (Problematic)

```
implement_solution():
    │
    ├── session.generate_code(prompt)
    │   └── Developer agent:
    │       - Implements solution
    │       - Creates kapso_evaluation/
    │       - Runs evaluation
    │       - Reports results
    │
    └── problem_handler.run()  ← REDUNDANT
        └── Executes main.py again
        └── Checks return code
```

## Proposed Flow

```
implement_solution():
    │
    └── session.generate_code(prompt)
        └── Developer agent:
            - Implements solution
            - Creates kapso_evaluation/
            - Runs evaluation
            - Returns structured JSON with results
```

## Files to Modify

### 1. `src/execution/search_strategies/base.py`

**Changes:**
- Remove `problem_handler.run()` call from `implement_solution()`
- Remove `problem_handler.run()` call from `debug_solution()`
- Update `_implement_n_debug()` to not expect `ProblemRunResult` from these methods
- Instead, extract results from the coding agent's output (see Task 08)

**Current code to remove:**
```python
# In implement_solution() - line 321-325
return self.problem_handler.run(
    session.session_folder,
    run_data_dir=session.run_dir,
    solution=solution,
)

# In debug_solution() - line 375-380
return self.problem_handler.run(
    session.session_folder,
    run_data_dir=session.run_dir,
    solution=solution,
    debug=True,
)
```

### 2. `src/environment/handlers/generic.py`

**Changes:**
- The `run()` method can be simplified or deprecated
- Keep `get_problem_context()` as it's still needed for providing problem description
- Keep `final_evaluate()` for final evaluation on held-out test sets (benchmarks)

### 3. `src/environment/handlers/base.py`

**Changes:**
- Update `ProblemRunResult` dataclass or create a new result structure
- Consider deprecating `run()` as abstract method (make it optional)

## Dependencies

- **Task 08**: Developer agent returns structured JSON (needed to extract results without `problem_handler.run()`)

## Testing

After changes:
1. Run `tests/demo_task1_initialize_repo.py`
2. Verify developer agent still runs evaluation
3. Verify results are correctly extracted from agent output
4. Verify no duplicate execution of main.py

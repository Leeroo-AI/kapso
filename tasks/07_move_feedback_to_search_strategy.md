# Task 07: Move Feedback Generation into Search Strategy

## Issue Description

Currently, feedback generation happens in the **orchestrator** loop:

```python
# src/execution/orchestrator.py lines 414-436
for i in range(experiment_max_iter):
    experiment_result = self.search_strategy.run(context)  # Search strategy runs
    
    # Orchestrator calls feedback generator
    feedback_result = self.feedback_generator.generate(
        goal=self.goal,
        idea=experiment_result.solution,
        code_diff=experiment_result.code_diff,
        evaluation_script_path=eval_script_path,
        evaluation_result=experiment_result.evaluation_output,
        workspace_dir=workspace_dir,
    )
```

This breaks the clean separation of concerns:
1. The orchestrator shouldn't know about feedback generation details
2. Different search strategies may need different feedback handling
3. For `LlmTreeSearch`, feedback should be per-node, not per-iteration

## Current Flow (Problematic)

```
Orchestrator.solve():
    │
    for each iteration:
        │
        ├── search_strategy.run()
        │   └── Returns ExperimentResult
        │
        └── feedback_generator.generate()  ← In orchestrator
            └── Decides stop/continue
```

## Proposed Flow

The feedback generation should be part of the search strategy's **node lifecycle**:

```
SearchStrategy.run():
    │
    ├── 1. GENERATE SOLUTION
    │   └── expand/select (tree) or _generate_solution (linear)
    │   └── Node.solution populated
    │
    ├── 2. IMPLEMENT
    │   └── Coding agent implements and runs evaluation
    │   └── Node.code_diff, Node.evaluation_output populated
    │
    ├── 3. FEEDBACK GENERATION  ← Moved here
    │   └── feedback_generator.generate()
    │   └── Node.feedback, Node.should_stop populated
    │
    └── Return Node (or best node for tree search)

Orchestrator.solve():
    │
    for each iteration:
        │
        ├── node = search_strategy.run()
        │
        └── if node.should_stop: break  ← Simple check
```

## Benefits

1. **Unified Node structure** for both LinearSearch and LlmTreeSearch
2. **Search strategy owns the full lifecycle** - cleaner separation
3. **Orchestrator is simpler** - just calls run() and checks stop condition
4. **Tree search can do per-node feedback** - not blocked by current design

## Files to Modify

### 1. `src/execution/search_strategies/base.py`

**Changes:**
- Add `FeedbackGenerator` as a dependency of `SearchStrategy`
- Add `_generate_feedback()` method to base class
- Update `run()` signature to include feedback generation
- Define unified `SearchNode` dataclass (see below)

**New dataclass:**
```python
@dataclass
class SearchNode:
    node_id: int
    parent_node_id: Optional[int] = None
    
    # Step 1: Solution generation
    solution: str = ""
    
    # Step 2: Implementation
    branch_name: str = ""
    code_changes_summary: str = ""
    
    # Step 3: Evaluation
    evaluation_script_path: str = ""
    evaluation_output: str = ""
    
    # Step 4: Feedback
    feedback: str = ""
    score: Optional[float] = None
    should_stop: bool = False
    evaluation_valid: bool = True
    
    # Metadata
    had_error: bool = False
    error_message: str = ""
    workspace_dir: str = ""
```

### 2. `src/execution/search_strategies/linear_search.py`

**Changes:**
- Update `run()` to call feedback generator after implementation
- Return `SearchNode` instead of `ExperimentResult`
- Remove `ExperimentResult` usage, use `SearchNode`

### 3. `src/execution/search_strategies/llm_tree_search.py`

**Changes:**
- Update `_run_for_node()` to call feedback generator
- Use `SearchNode` structure for nodes
- Return best node from `run()` (currently returns `None`)

### 4. `src/execution/orchestrator.py`

**Changes:**
- Remove feedback generation code (lines 424-456)
- Simplify loop to just check `node.should_stop`
- Pass `FeedbackGenerator` to search strategy during initialization

**Simplified orchestrator loop:**
```python
for i in range(experiment_max_iter):
    node = self.search_strategy.run(context, budget_progress)
    
    if node is None:
        continue
    
    if node.should_stop:
        stopped_reason = "goal_achieved"
        break
    
    self.current_feedback = node.feedback
```

### 5. `src/execution/search_strategies/factory.py`

**Changes:**
- Update factory to pass `FeedbackGenerator` to search strategies

## Dependencies

- **Task 06**: Remove `problem_handler.run()` (simplifies implementation step)
- **Task 08**: Developer agent returns structured JSON (provides data for feedback)

## Testing

After changes:
1. Run `tests/demo_task1_initialize_repo.py` with LinearSearch
2. Verify feedback is generated within search strategy
3. Verify orchestrator correctly checks `should_stop`
4. Test with LlmTreeSearch to verify per-node feedback works

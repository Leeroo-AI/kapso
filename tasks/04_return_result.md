# Task 4: Return Result

## Design Reference
From `design.md` lines 55-56:
```
└─► 4. RETURN RESULT
        Final repo with solution
```

## Current Implementation

### What Exists

1. **`src/kapso.py` - `evolve()` method (lines 548-588)**
   ```python
   # Run experimentation
   orchestrator.solve(experiment_max_iter=max_iterations)
   
   # Collect results
   experiment_logs = self._extract_experiment_logs(orchestrator)
   workspace_path = orchestrator.search_strategy.workspace.workspace_dir
   
   # Checkout to best solution
   orchestrator.search_strategy.checkout_to_best_experiment_branch()
   
   # Final evaluation
   final_result = handler.final_evaluate(workspace_path)
   
   # Create solution result
   solution = SolutionResult(
       goal=goal,
       code_path=code_path,
       experiment_logs=experiment_logs,
       metadata={...}
   )
   ```

2. **`src/execution/solution.py` - `SolutionResult`**
   ```python
   @dataclass
   class SolutionResult:
       goal: str
       code_path: str
       experiment_logs: List[str]
       metadata: Dict[str, Any]
   ```

3. **`src/environment/handlers/generic.py` - `final_evaluate()`**
   - Returns summary of all runs
   - Uses predefined evaluator (being removed)

### Changes Required

#### DELETE
- None

#### MODIFY

1. **`src/kapso.py` - `evolve()` method (result section)**
   - Remove `handler.final_evaluate()` call (evaluator being removed)
   - Simplify result collection
   - New result section:
     ```python
     # Run experimentation
     final_feedback = orchestrator.solve(experiment_max_iter=max_iterations)
     
     # Collect results
     experiment_logs = self._extract_experiment_logs(orchestrator)
     workspace_path = orchestrator.search_strategy.workspace.workspace_dir
     
     # Checkout to best solution
     orchestrator.search_strategy.checkout_to_best_experiment_branch()
     
     # Create solution result
     solution = SolutionResult(
         goal=goal,
         code_path=workspace_path,
         experiment_logs=experiment_logs,
         final_feedback=final_feedback,  # NEW: from feedback generator
         metadata={
             "iterations": len(experiment_logs),
             "cost": f"${orchestrator.get_cumulative_cost():.3f}",
             "stopped_reason": final_feedback.stop_reason,
         }
     )
     ```

2. **`src/execution/solution.py` - `SolutionResult`**
   - Add `final_feedback` field
   - Update metadata structure
   ```python
   @dataclass
   class SolutionResult:
       goal: str
       code_path: str
       experiment_logs: List[str]
       final_feedback: Optional[FeedbackResult]  # NEW
       metadata: Dict[str, Any]
       
       @property
       def succeeded(self) -> bool:
           """True if goal was achieved."""
           return self.final_feedback and self.final_feedback.stop
       
       @property
       def final_score(self) -> Optional[float]:
           """Final evaluation score if available."""
           if self.final_feedback:
               return self.final_feedback.evaluation_result.get("score")
           return None
   ```

3. **`src/execution/orchestrator.py` - `solve()` method**
   - Return `FeedbackResult` from final iteration
   - Or return summary of why stopped (max iterations, etc.)

4. **`src/kapso.py` - `_extract_experiment_logs()` method**
   - Update to work with new experiment result structure
   - Include evaluation results in logs

#### ADD

1. **`src/execution/solution.py` - `FeedbackResult` import**
   - Import from feedback_generator module

### New Result Structure

```python
SolutionResult(
    goal="Fine-tune Llama for legal risk detection...",
    code_path="/path/to/workspace",
    experiment_logs=[
        "Iteration 1: Built DPO training, evaluation shows 15% improvement",
        "Iteration 2: Adjusted hyperparams, 28% improvement",
        "Iteration 3: Final tuning, 44% improvement - GOAL ACHIEVED",
    ],
    final_feedback=FeedbackResult(
        stop=True,
        evaluation_valid=True,
        feedback="Goal achieved: 44% improvement exceeds 40% target.",
        stop_reason="goal_achieved",
    ),
    metadata={
        "iterations": 3,
        "cost": "$12.50",
        "stopped_reason": "goal_achieved",
    }
)
```

### Files to Touch
- `src/kapso.py`
- `src/execution/solution.py`
- `src/execution/orchestrator.py`

### Cross-References
- Depends on: `03_developer_agent_loop.md` (FeedbackResult type)

### Testing Considerations
- Test result when goal achieved
- Test result when max iterations reached
- Test result when budget exhausted
- Verify code_path points to correct branch
- Verify experiment_logs capture all iterations

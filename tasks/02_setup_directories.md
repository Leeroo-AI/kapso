# Task 2: Setup Directories

## Design Reference
From `design.md` lines 32-35:
```
├─► 2. SETUP DIRECTORIES
│       Copy eval_dir → repo/kapso_evaluation/
│       Copy data_dir → repo/kapso_datasets/
│       Build repo_memory for workspace understanding
```

## Current Implementation

### What Exists

1. **`src/kapso.py` - `evolve()` method**
   - Has `data_dir` parameter but only passes it to `GenericProblemHandler`
   - No `eval_dir` parameter
   - Does NOT copy data to workspace

2. **`src/environment/handlers/generic.py` - `GenericProblemHandler`**
   - Stores `data_dir` path
   - References it in problem context string
   - Does NOT copy data into workspace

3. **`src/repo_memory/manager.py` - `RepoMemoryManager`**
   - `ensure_exists_in_worktree()` - creates repo_memory if missing
   - `bootstrap_baseline_model()` - builds initial repo model from seed
   - Called from `SearchStrategy.__init__()` in `base.py`

4. **`src/execution/search_strategies/base.py`**
   - Lines 117-133: Builds baseline RepoMemory on init
   - For seeded repos: calls `bootstrap_baseline_model()`
   - For empty repos: calls `ensure_exists_in_worktree()`

### Changes Required

#### DELETE
- None

#### MODIFY

1. **`src/kapso.py` - `evolve()` method**
   - Add `eval_dir` parameter
   - Remove `data_dir` from `GenericProblemHandler` (no longer needed there)
   - Remove these parameters that are no longer needed:
     - `evaluator`
     - `evaluator_params`
     - `stop_condition`
     - `stop_condition_params`
     - `language`
     - `main_file`
     - `timeout`

2. **`src/execution/orchestrator.py`**
   - Accept `eval_dir` and `data_dir` parameters
   - Pass to search strategy for directory setup

3. **`src/execution/search_strategies/base.py`**
   - Add `eval_dir` and `data_dir` to `SearchStrategyConfig`
   - In `__init__()`, after workspace init:
     ```python
     self._setup_kapso_directories(eval_dir, data_dir)
     ```

#### ADD

1. **`src/execution/search_strategies/base.py` - new method**
   ```python
   def _setup_kapso_directories(
       self, 
       eval_dir: Optional[str], 
       data_dir: Optional[str]
   ) -> None:
       """
       Setup kapso_evaluation/ and kapso_datasets/ directories.
       
       Copies user-provided directories into the workspace repo.
       """
       workspace = self.workspace.workspace_dir
       
       # Setup kapso_evaluation/
       kapso_eval = os.path.join(workspace, "kapso_evaluation")
       os.makedirs(kapso_eval, exist_ok=True)
       if eval_dir and os.path.exists(eval_dir):
           shutil.copytree(eval_dir, kapso_eval, dirs_exist_ok=True)
       
       # Setup kapso_datasets/
       kapso_data = os.path.join(workspace, "kapso_datasets")
       os.makedirs(kapso_data, exist_ok=True)
       if data_dir and os.path.exists(data_dir):
           shutil.copytree(data_dir, kapso_data, dirs_exist_ok=True)
       
       # Commit the directories
       self.workspace.repo.git.add([kapso_eval, kapso_data])
       if self.workspace.repo.is_dirty():
           self.workspace.repo.git.commit("-m", "chore(kapso): setup evaluation and data directories")
   ```

2. **Update `.gitignore` handling in `experiment_workspace.py`**
   - Ensure `kapso_evaluation/` and `kapso_datasets/` are NOT ignored
   - They should be committed and inherited by experiment branches

### API Changes

**Before:**
```python
kapso.evolve(
    goal="...",
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
    eval_dir="/path/to/evaluation",  # NEW
    data_dir="/path/to/data",         # Behavior changes: copied to workspace
    # REMOVED: evaluator, evaluator_params, stop_condition, stop_condition_params
    # REMOVED: language, main_file, timeout (agent decides these)
)
```

### Directory Structure After Setup

```
workspace/
  ├── kapso_evaluation/     # Created, contains eval_dir contents
  ├── kapso_datasets/       # Created, contains data_dir contents
  ├── .kapso/
  │     └── repo_memory.json  # Built by RepoMemoryManager
  └── (workflow code or empty)
```

### Files to Touch
- `src/kapso.py`
- `src/execution/orchestrator.py`
- `src/execution/search_strategies/base.py`
- `src/execution/experiment_workspace/experiment_workspace.py`

### Cross-References
- Depends on: `01_initialize_repo.md` (workspace must exist first)
- Related to: `03_developer_agent.md` (agent needs to know about these directories)

### Testing Considerations
- Test with both `eval_dir` and `data_dir` provided
- Test with only one provided
- Test with neither provided (empty directories created)
- Verify directories are committed and inherited by experiment branches
- Verify repo_memory is built after directory setup

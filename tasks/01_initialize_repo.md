# Task 1: Initialize Repo

## Design Reference
From `design.md` lines 24-30:
```
├─► 1. INITIALIZE REPO
│       if initial_repo provided:
│           use it as base
│       else:
│           workflow_search(goal) → find related workflow
│           if workflow found: init repo from workflow
│           else: create empty repo
```

## Current Implementation

### What Exists

1. **`src/kapso.py` - `evolve()` method (lines 423-588)**
   - Accepts `starting_repo_path` parameter for user-provided repo
   - Passes it to `OrchestratorAgent` which passes to `SearchStrategy`
   - Does NOT use workflow_search to find starter repos

2. **`src/execution/experiment_workspace/experiment_workspace.py`**
   - `_init_from_seed_repo()` - clones/copies user-provided repo
   - `is_seeded` flag tracks if workspace started from seed
   - Works correctly for user-provided repos
   - **Note:** `is_seeded` is still needed in new design to determine:
     - RepoMemory: full bootstrap (seeded) vs lightweight skeleton (empty)
     - Workspace init: skip placeholder creation (seeded) vs create empty main.py (empty)

3. **`src/knowledge/search/workflow_search.py`**
   - `WorkflowRepoSearch` class exists
   - `find_starter_repo(problem)` returns GitHub URL
   - Searches KG for Workflow pages with GitHub URLs
   - **NOT integrated into evolve() flow**

4. **`src/execution/orchestrator.py`**
   - Receives `starting_repo_path` and passes to search strategy
   - No workflow search integration

### Changes Required

#### DELETE
- None (keep existing seed repo functionality)

#### MODIFY

1. **`src/kapso.py` - `evolve()` method**
   - Rename `starting_repo_path` → `initial_repo` (API change per design)
   - `initial_repo` accepts TWO formats:
     - Local path: `/path/to/repo` or `./relative/path`
     - GitHub URL: `https://github.com/owner/repo`
   - Add logic to detect and handle both formats:
     ```python
     def _resolve_initial_repo(self, initial_repo: Optional[str]) -> Optional[str]:
         """
         Resolve initial_repo to a local path.
         
         Args:
             initial_repo: Local path or GitHub URL
             
         Returns:
             Local path to repo (cloned if URL), or None
         """
         if initial_repo is None:
             return None
         
         # Check if it's a GitHub URL
         if initial_repo.startswith("https://github.com/"):
             return self._clone_github_repo(initial_repo)
         
         # Assume local path
         return initial_repo
     ```
   - Add workflow search logic when `initial_repo` is None:
     ```python
     resolved_repo = self._resolve_initial_repo(initial_repo)
     
     if resolved_repo is None:
         # Search for workflow repo
         from src.knowledge.search.workflow_search import WorkflowRepoSearch
         workflow_search = WorkflowRepoSearch()
         starter_url = workflow_search.find_starter_repo(goal)
         if starter_url:
             resolved_repo = self._clone_github_repo(starter_url)
     ```

2. **`src/execution/orchestrator.py`**
   - Rename `starting_repo_path` → `initial_repo` for consistency

3. **`src/execution/search_strategies/base.py`**
   - Rename `seed_repo_path` → `initial_repo` in `SearchStrategyConfig`

4. **`src/execution/experiment_workspace/experiment_workspace.py`**
   - Rename `seed_repo_path` → `initial_repo` for consistency

#### ADD

1. **`src/kapso.py` - new helper method**
   ```python
   def _clone_github_repo(self, url: str) -> str:
       """
       Clone a GitHub repository to a temporary directory.
       
       Args:
           url: GitHub repository URL (https://github.com/owner/repo)
           
       Returns:
           Local path to cloned repository
       """
       import tempfile
       import git
       
       # Create temp directory
       temp_dir = tempfile.mkdtemp(prefix="kapso_repo_")
       
       # Clone the repo
       print(f"Cloning {url} to {temp_dir}...")
       git.Repo.clone_from(url, temp_dir)
       
       return temp_dir
   ```

2. **`src/kapso.py` - URL detection helper**
   ```python
   def _is_github_url(self, path: str) -> bool:
       """Check if path is a GitHub URL."""
       return path.startswith("https://github.com/") or path.startswith("git@github.com:")
   ```

### API Changes

**Before:**
```python
kapso.evolve(
    goal="...",
    starting_repo_path="/path/to/repo",  # Optional, local path only
)
```

**After:**
```python
kapso.evolve(
    goal="...",
    initial_repo="/path/to/repo",              # Local path
    # OR
    initial_repo="https://github.com/org/repo", # GitHub URL (will be cloned)
    # OR
    initial_repo=None,                          # Search for workflow repo
)
```

### Files to Touch
- `src/kapso.py`
- `src/execution/orchestrator.py`
- `src/execution/search_strategies/base.py`
- `src/execution/search_strategies/linear_search.py`
- `src/execution/search_strategies/llm_tree_search.py`
- `src/execution/experiment_workspace/experiment_workspace.py`

### Testing

#### Test Data
Use wiki data at: `data/wikis_llm_finetuning_test/`

This contains a workflow page:
- `workflows/Jaymody_PicoGPT_Text_Generation.md`
- GitHub URL in page: `https://github.com/leeroo-coder/workflow-jaymody-picogpt-text-generation`
- Domains: LLMs, Inference, Education

#### Test Cases

1. **Test with `initial_repo=None` and valid workflow in KG**
   ```python
   # Setup: Index the test wiki data first
   kapso = Kapso()
   kapso.index_kg(wiki_dir="data/wikis_llm_finetuning_test", save_to="test.index")
   
   # Test: Should find PicoGPT workflow and clone it
   kapso = Kapso(kg_index="test.index")
   solution = kapso.evolve(
       goal="Generate text using GPT-2 with NumPy",
       initial_repo=None,  # Should find Jaymody_PicoGPT workflow
   )
   
   # Verify: Workspace should contain cloned PicoGPT repo
   assert os.path.exists(os.path.join(solution.code_path, "gpt2.py"))
   ```

2. **Test with `initial_repo=None` and no matching workflow**
   ```python
   kapso = Kapso()  # No KG index
   solution = kapso.evolve(
       goal="Build a quantum computing simulator",
       initial_repo=None,  # No workflow will match
   )
   
   # Verify: Should create empty repo
   assert os.path.exists(solution.code_path)
   # Workspace should be empty (or have placeholder)
   ```

3. **Test with `initial_repo="/path/to/repo"` (local path)**
   ```python
   # Create a test repo
   test_repo = "/tmp/test_repo"
   os.makedirs(test_repo, exist_ok=True)
   with open(os.path.join(test_repo, "main.py"), "w") as f:
       f.write("print('hello')")
   
   solution = kapso.evolve(
       goal="Improve this code",
       initial_repo=test_repo,
   )
   
   # Verify: Workspace should contain the test repo content
   assert os.path.exists(os.path.join(solution.code_path, "main.py"))
   ```

4. **Test with `initial_repo="https://github.com/..."` (GitHub URL)**
   ```python
   solution = kapso.evolve(
       goal="Understand GPT-2 implementation",
       initial_repo="https://github.com/jaymody/picoGPT",
   )
   
   # Verify: Workspace should contain cloned repo
   assert os.path.exists(os.path.join(solution.code_path, "gpt2.py"))
   ```

5. **Test workflow search extracts correct GitHub URL**
   ```python
   from src.knowledge.search.workflow_search import WorkflowRepoSearch
   
   # Setup: Index test wiki
   # ...
   
   search = WorkflowRepoSearch()
   result = search.find_starter_repo("text generation with GPT")
   
   # Verify: Should return the workflow's GitHub URL
   assert result == "https://github.com/leeroo-coder/workflow-jaymody-picogpt-text-generation"
   ```

### Workflow Page Format Reference

From `data/wikis_llm_finetuning_test/workflows/Jaymody_PicoGPT_Text_Generation.md`:

```markdown
# Workflow: Text_Generation

{| class="wikitable" ...
|-
! Knowledge Sources
||
* [[source::Repo|PicoGPT|https://github.com/jaymody/picoGPT]]
...
|}

== GitHub URL ==
[[github_url::https://github.com/leeroo-coder/workflow-jaymody-picogpt-text-generation]]
```

The `workflow_search.py` extracts GitHub URL from:
1. `== Github URL ==` section
2. `[[source::Repo|name|URL]]` syntax
3. Raw GitHub URLs in content

### Cross-References
- Related to: `02_setup_directories.md` (runs after repo init)
- Related to: `03_developer_agent_loop.md` (uses initialized repo)

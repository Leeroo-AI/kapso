# Task 2: Setup Directories - Test Report

## Test Date
2026-01-25

## Test Environment
- Conda environment: `praxium_conda`
- Python: 3.11.14
- pytest: 8.4.2

## Test Results

### Unit Tests (7/7 passed)

```
tests/test_task2_setup_directories.py::TestSetupKapsoDirectories::test_both_directories_provided PASSED
tests/test_task2_setup_directories.py::TestSetupKapsoDirectories::test_only_eval_dir_provided PASSED
tests/test_task2_setup_directories.py::TestSetupKapsoDirectories::test_only_data_dir_provided PASSED
tests/test_task2_setup_directories.py::TestSetupKapsoDirectories::test_neither_directory_provided PASSED
tests/test_task2_setup_directories.py::TestSetupKapsoDirectories::test_directories_inherited_by_branches PASSED
tests/test_task2_setup_directories.py::TestSetupKapsoDirectories::test_nonexistent_eval_dir_ignored PASSED
tests/test_task2_setup_directories.py::TestKapsoEvolveWithDirectories::test_evolve_accepts_eval_dir_and_data_dir PASSED

======================== 7 passed, 2 warnings in 2.44s =========================
```

### Test Categories

#### 1. Directory Setup Tests (6 tests)
- Both eval_dir and data_dir provided - files copied correctly
- Only eval_dir provided - kapso_datasets/ has .gitkeep
- Only data_dir provided - kapso_evaluation/ has .gitkeep
- Neither provided - both directories have .gitkeep
- Directories inherited by git branches
- Non-existent eval_dir handled gracefully

#### 2. API Tests (1 test)
- Kapso.evolve() accepts eval_dir and data_dir parameters

### Demo Script Output (End-to-End Flow)

The demo script (`tests/demo_task1_initialize_repo.py`) shows the full flow:
1. Index wiki data → 2. Search for workflow → 3. Clone as initial_repo → 4. Add kapso directories

```
============================================================
Step 2: Search for workflow based on goal
============================================================
Goal: Generate text using GPT-2 model with pure NumPy implementation

Workflow search result:
  Title: Jaymody PicoGPT Text Generation
  GitHub URL: https://github.com/jaymody/picoGPT
  Score: 0.912

============================================================
Step 3: Clone workflow repo as initial_repo
============================================================
Cloned to: /tmp/kapso_repo_qga_4b6r

Initial repo contents:
  .git, .gitignore, LICENSE, README.md, encoder.py, gpt2.py, 
  gpt2_pico.py, requirements.txt, utils.py

============================================================
Step 6: Final workspace directory structure
============================================================

/tmp/demo_workspace_zy3maa6e/
├── .git/
├── .gitignore
├── .kapso/
│   └── repo_memory.json
├── LICENSE
├── README.md
├── encoder.py              # From workflow repo
├── gpt2.py                 # From workflow repo
├── gpt2_pico.py            # From workflow repo
├── kapso_datasets/         # Added by Kapso
│   └── prompts.txt
├── kapso_evaluation/       # Added by Kapso
│   └── evaluate.py
├── requirements.txt        # From workflow repo
├── utils.py                # From workflow repo

============================================================
Step 7: Verify git tracking
============================================================

Tracked kapso files:
  .kapso/repo_memory.json
  kapso_datasets/prompts.txt
  kapso_evaluation/evaluate.py

Recent commits:
  - chore(kapso): add baseline repo memory
  - chore(kapso): setup evaluation and data directories
  - chore(kapso): ignore experiment sessions
  - Update README.
  - Fix of 2 typos (#21)
```

## Files Modified

1. `src/kapso.py`
   - Added `eval_dir` parameter to `evolve()`
   - Passes `eval_dir` and `data_dir` to orchestrator

2. `src/execution/orchestrator.py`
   - Added `eval_dir` and `data_dir` parameters
   - Passes to search strategy factory

3. `src/execution/search_strategies/base.py`
   - Added `eval_dir` and `data_dir` to `SearchStrategyConfig`
   - Added `_setup_kapso_directories()` method
   - Calls setup before RepoMemory is built

4. `src/execution/search_strategies/factory.py`
   - Added `eval_dir` and `data_dir` parameters
   - Passes to `SearchStrategyConfig`

## New API

```python
kapso.evolve(
    goal="...",
    initial_repo="/path/to/repo",      # Or GitHub URL, or None for workflow search
    eval_dir="/path/to/evaluation",    # Copied to workspace/kapso_evaluation/
    data_dir="/path/to/data",          # Copied to workspace/kapso_datasets/
)
```

## Directory Structure After Setup

When `initial_repo` is provided (or found via workflow search), the workspace IS the cloned repo with kapso directories added as siblings:

```
workspace/                              # = cloned workflow repo
├── .git/
├── .gitignore
├── .kapso/
│   └── repo_memory.json
├── README.md                           # From workflow repo
├── src/                                # From workflow repo
│   └── ...
├── kapso_evaluation/                   # Added by Kapso
│   └── evaluate.py
├── kapso_datasets/                     # Added by Kapso
│   └── data.csv
└── requirements.txt                    # From workflow repo
```

## Test Files

- `tests/test_task2_setup_directories.py` - 7 unit tests
- `tests/demo_task1_initialize_repo.py` - Full end-to-end demo (Task 1 + Task 2)

## Status: COMPLETE ✓

All 7 tests pass. Task 2 implementation is complete and verified.

## Next Steps

Proceed to Task 3: Developer Agent Loop (`03_developer_agent_loop.md`)

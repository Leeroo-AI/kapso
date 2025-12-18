# tests_fetcher.py

**Path:** `/tmp/praxium_repo_d5p6fp4d/utils/tests_fetcher.py`

## Understanding

### Purpose
Determines impacted tests from code changes.

### Mechanism
This sophisticated test selection system operates in two stages: (1) identifies modified files by analyzing git diffs (either from the last commit or from the PR branching point), excluding changes that only affect docstrings or comments; (2) builds a reverse dependency map by analyzing import statements across all modules and test files to determine which tests are impacted by each modified file. It recursively traces dependencies through __init__.py files and handles the import structure to identify all affected tests. The system can optionally filter to only core models when too many models are impacted (more than half of all model tests), and integrates special handling for example tests and tiny model changes.

### Significance
Dramatically reduces CI execution time by running only relevant tests affected by code changes, rather than the entire test suite. This intelligent test selection is crucial for developer productivity in a large repository with thousands of tests, while still ensuring comprehensive coverage of impacted code. The filtering mechanism prevents CI overload when changes affect many models while maintaining thorough testing for targeted changes.

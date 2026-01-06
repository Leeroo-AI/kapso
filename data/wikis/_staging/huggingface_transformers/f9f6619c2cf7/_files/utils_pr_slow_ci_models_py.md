# File: `utils/pr_slow_ci_models.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 175 |
| Functions | `get_new_python_files_between_commits`, `get_new_python_files`, `get_new_model`, `parse_message`, `get_models`, `check_model_names` |
| Imports | argparse, git, json, os, pathlib, re, string |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Determines which model-specific slow tests should run in CI by detecting newly added models in pull requests and parsing special comments with "run-slow" prefixes that specify additional models to test.

**Mechanism:** The script uses GitPython to compare the current branch with main to identify new Python modeling files, extracts model names from file paths using regex patterns, parses PR comments or commit messages for "run-slow:", "run_slow", or "run slow" prefixes followed by comma-separated model names, validates model names against allowed characters, verifies that corresponding test directories exist, and outputs a JSON array of test paths suitable for GitHub Actions matrix strategies.

**Significance:** This tool optimizes CI resources by selectively running expensive slow tests only for relevant models rather than the entire test suite. It's crucial for maintaining fast PR feedback loops while ensuring comprehensive testing of new or specifically requested models, balancing thorough validation against CI time and cost constraints.

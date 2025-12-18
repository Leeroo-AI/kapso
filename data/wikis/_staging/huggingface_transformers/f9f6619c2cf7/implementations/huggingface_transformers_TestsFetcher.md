# TestsFetcher

## Metadata
| Key | Value |
|-----|-------|
| **Source** | `utils/tests_fetcher.py` |
| **Repository** | huggingface/transformers |
| **Domains** | CI/CD, Testing, Code Analysis |
| **Last Updated** | 2025-12-18 |

## Overview

The TestsFetcher module is a sophisticated test selection system that analyzes code changes in pull requests to determine which tests need to run. By building a dependency graph of the entire codebase, it identifies only the tests impacted by modified files, significantly reducing CI execution time while maintaining comprehensive test coverage.

## Description

This implementation provides intelligent test filtering for the Transformers library:

**Core Capabilities:**
- Analyzes git diffs to identify modified Python files (excluding docstring-only changes)
- Builds complete dependency maps by parsing import statements across the codebase
- Creates reverse dependency trees to identify all modules impacted by changes
- Filters tests to core models when too many models are affected
- Supports both PR-based and commit-based diff analysis
- Handles special cases (example tests, documentation tests, tiny model changes)
- Parses commit messages for CI control flags ([skip ci], [test all], etc.)

**Key Components:**
- Dependency extraction via regex-based import parsing
- Recursive dependency resolution with caching
- Reverse dependency map construction
- Test-to-module mapping with intelligent filtering
- Git integration for diff analysis
- Doctest file detection and filtering

**Integration Points:**
- Git repository access via GitPython
- GitHub Actions environment variables
- Transformers module structure and imports
- Important models configuration from `important_files.py`
- Test artifact generation for CI pipeline

## Usage

### Command Line Usage

```bash
# Basic usage: Fetch tests for a PR (diff with main branch)
python utils/tests_fetcher.py

# Fetch tests for main branch (diff with last commit)
python utils/tests_fetcher.py --diff_with_last_commit

# Show dependency tree for a specific module
python utils/tests_fetcher.py --print_dependencies_of src/transformers/models/bert/modeling_bert.py

# Filter specific test categories from results
python utils/tests_fetcher.py --filter_tests

# Force fetch all tests
python utils/tests_fetcher.py --fetch_all

# Custom output file
python utils/tests_fetcher.py --output_file my_test_list.txt
```

### Commit Message Flags

```bash
# Skip CI entirely
git commit -m "Fix typo [skip ci]"

# Run all tests regardless of changes
git commit -m "Major refactor [test all]"

# Disable model filtering (test all impacted models)
git commit -m "Update shared utility [no filter]"
```

## Code Reference

### Main Functions

```python
def get_modified_python_files(diff_with_last_commit: bool = False) -> list[str]:
    """
    Get list of Python files modified between current head and main branch.

    Args:
        diff_with_last_commit: If True, compare with parent commit instead

    Returns:
        List of relative file paths with real code changes
    """

def extract_imports(module_fname: str, cache: dict[str, list[str]] | None = None) -> list[str]:
    """
    Extract all imports from a Python module.

    Args:
        module_fname: Relative path to module file
        cache: Optional cache of previously computed imports

    Returns:
        List of (imported_module, [imported_objects]) tuples
    """

def get_module_dependencies(module_fname: str, cache: dict[str, list[str]] | None = None) -> list[str]:
    """
    Get refined list of module dependencies by resolving __init__ imports.

    Args:
        module_fname: Relative path to module file
        cache: Optional cache of previously computed dependencies

    Returns:
        List of module file paths that this module depends on
    """

def create_reverse_dependency_map() -> dict[str, list[str]]:
    """
    Create complete reverse dependency map for the repository.

    Returns:
        Dictionary mapping each file to all files that depend on it
    """

def create_module_to_test_map(
    reverse_map: dict[str, list[str]] | None = None,
    filter_models: bool = False
) -> dict[str, list[str]]:
    """
    Create mapping from modules to their test files.

    Args:
        reverse_map: Pre-computed reverse dependency map
        filter_models: Whether to filter to core models for broad changes

    Returns:
        Dictionary mapping each file to tests that should run if modified
    """

def infer_tests_to_run(
    output_file: str,
    diff_with_last_commit: bool = False,
    filter_models: bool = False,
    test_all: bool = False
):
    """
    Main function to determine tests to run based on changes.

    Args:
        output_file: Path to write test list
        diff_with_last_commit: Compare with parent commit instead of main
        filter_models: Filter to core models for broad changes
        test_all: Force running all tests
    """

def get_doctest_files(diff_with_last_commit: bool = False) -> list[str]:
    """
    Get list of files with modified documentation examples.

    Args:
        diff_with_last_commit: Compare with parent commit instead of main

    Returns:
        List of Python and Markdown files with doc example changes
    """
```

### Utility Functions

```python
def clean_code(content: str) -> str:
    """Remove docstrings, comments, and empty lines from code."""

def diff_is_docstring_only(repo: Repo, branching_point: str, filename: str) -> bool:
    """Check if diff only contains docstring/comment changes."""

def keep_doc_examples_only(content: str) -> str:
    """Extract only documentation examples from code."""

def diff_contains_doc_examples(repo: Repo, branching_point: str, filename: str) -> bool:
    """Check if diff contains changes to doc examples."""

def get_all_tests() -> list[str]:
    """Get complete list of test folders and files."""

def get_tree_starting_at(module: str, edges: list[tuple[str, str]]) -> list[str | list[str]]:
    """Build dependency tree starting from a given module."""

def print_tree_deps_of(module: str, all_edges=None):
    """Print dependency tree for a module."""

def parse_commit_message(commit_message: str) -> dict[str, bool]:
    """Parse commit message for CI control flags."""
```

### Constants

```python
# Repository paths
PATH_TO_REPO = Path(__file__).parent.parent.resolve()
PATH_TO_EXAMPLES = PATH_TO_REPO / "examples"
PATH_TO_TRANSFORMERS = PATH_TO_REPO / "src/transformers"
PATH_TO_TESTS = PATH_TO_REPO / "tests"

# Model filtering threshold
NUM_MODELS_TO_TRIGGER_FULL_CI = 30

# Job type regex mappings
JOB_TO_TEST_FILE = {
    "tests_torch": r"tests/models/.*/test_modeling_.*",
    "tests_generate": r"tests/models/.*/test_modeling_.*",
    "tests_tokenization": r"tests/(?:models/.*/test_tokenization.*|test_tokenization_mistral_common\.py)",
    "tests_processors": r"tests/models/.*/test_(?!(?:modeling_|tokenization_)).*",
    "examples_torch": r"examples/pytorch/.*test_.*",
    # ... additional mappings
}
```

### Import Patterns (Regex)

```python
# Single-line relative imports: from .module import obj
_re_single_line_relative_imports = re.compile(r"(?:^|\n)\s*from\s+(\.+\S+)\s+import\s+([^\n]+)(?=\n)")

# Multi-line relative imports: from .module import (obj1, obj2)
_re_multi_line_relative_imports = re.compile(r"(?:^|\n)\s*from\s+(\.+\S+)\s+import\s+\(([^\)]+)\)")

# Single-line direct imports: from transformers.module import obj
_re_single_line_direct_imports = re.compile(r"(?:^|\n)\s*from\s+transformers(\S*)\s+import\s+([^\n]+)(?=\n)")

# Multi-line direct imports: from transformers.module import (obj1, obj2)
_re_multi_line_direct_imports = re.compile(r"(?:^|\n)\s*from\s+transformers(\S*)\s+import\s+\(([^\)]+)\)")
```

### Key Imports

```python
import argparse
import collections
import glob
import json
import os
import re
from contextlib import contextmanager
from pathlib import Path

from git import Repo
from important_files import IMPORTANT_MODELS
```

## I/O Contract

### Inputs

| Input | Type | Source | Description |
|-------|------|--------|-------------|
| Git Repository | Repository | Current Working Directory | Transformers git repository for diff analysis |
| Modified Files | Git Diff | Git History | Files changed between commits |
| `important_files.py` | Python Module | Utils | List of core models that should always be tested |
| `utils/not_doctested.txt` | Text File | Utils | Files excluded from doctest |
| `utils/slow_documentation_tests.txt` | Text File | Utils | Slow doctests excluded from CI |
| `tests/utils/tiny_model_summary.json` | JSON File | Tests | Tiny model metadata for impact analysis |
| Commit Message | String | Git History | Commit message with optional CI flags |
| Command Line Args | Arguments | CLI | Configuration flags and output paths |

### Outputs

| Output | Type | Destination | Description |
|--------|------|-------------|-------------|
| `test_list.txt` | Text File | Specified Path | Space-separated list of tests to run |
| `test_preparation/*_test_list.txt` | Text Files | test_preparation/ | Test lists split by job type |
| `doctest_list.txt` | Text File | Output Directory | Space-separated list of doctest files |
| Console Output | stdout | Terminal/CI Logs | Modified files, impacted files, tests to run |
| Dependency Tree | stdout | Terminal (optional) | Visual representation of module dependencies |

### Side Effects

- Reads from git repository (may checkout different commits temporarily)
- Creates `test_preparation/` directory
- Writes multiple test list files
- Imports and analyzes all Python files in repository (via dependency analysis)
- May trigger full CI if too many models impacted

## Usage Examples

### Example 1: Basic Test Fetching for PR

```python
# Run test fetcher on a pull request
from tests_fetcher import infer_tests_to_run

infer_tests_to_run(
    output_file="test_list.txt",
    diff_with_last_commit=False,  # Compare with main branch
    filter_models=True,  # Filter to core models if needed
    test_all=False
)

# Output files created:
# - test_list.txt: Main test list
# - test_preparation/tests_torch_test_list.txt
# - test_preparation/tests_tokenization_test_list.txt
# - doctest_list.txt (if < 20 tests)
```

### Example 2: Analyzing Module Dependencies

```python
# Extract imports from a module
from tests_fetcher import extract_imports

imports = extract_imports("src/transformers/models/bert/modeling_bert.py")
# Returns: [
#   ("src/transformers/activations.py", ["gelu", "relu"]),
#   ("src/transformers/modeling_utils.py", ["PreTrainedModel"]),
#   ...
# ]

# Get full dependency tree
from tests_fetcher import get_module_dependencies

deps = get_module_dependencies("src/transformers/models/bert/modeling_bert.py")
# Returns: [
#   "src/transformers/activations.py",
#   "src/transformers/modeling_utils.py",
#   "src/transformers/utils/logging.py",
#   ...
# ]
```

### Example 3: Building Reverse Dependency Map

```python
# Create complete reverse dependency map
from tests_fetcher import create_reverse_dependency_map

reverse_map = create_reverse_dependency_map()
# reverse_map = {
#   "src/transformers/activations.py": [
#     "src/transformers/models/bert/modeling_bert.py",
#     "src/transformers/models/gpt2/modeling_gpt2.py",
#     "tests/models/bert/test_modeling_bert.py",
#     ...
#   ],
#   ...
# }

# Find all tests impacted by modifying activations.py
impacted_tests = [f for f in reverse_map["src/transformers/activations.py"]
                  if f.startswith("tests/")]
```

### Example 4: Printing Dependency Tree

```python
# Visualize dependencies of a specific module
from tests_fetcher import print_tree_deps_of

print_tree_deps_of("src/transformers/models/bert/modeling_bert.py")
# Output:
# src/transformers/models/bert/modeling_bert.py
#   tests/models/bert/test_modeling_bert.py
#   tests/test_modeling_common.py
#     tests/models/gpt2/test_modeling_gpt2.py
#     tests/models/roberta/test_modeling_roberta.py
#   ...
```

### Example 5: Filtering Tests by Job Type

```python
# Create test lists split by job type
from tests_fetcher import create_test_list_from_filter

full_test_list = [
    "tests/models/bert/test_modeling_bert.py",
    "tests/models/bert/test_tokenization_bert.py",
    "tests/models/gpt2/test_modeling_gpt2.py",
    "examples/pytorch/text-classification/test_text_classification.py"
]

create_test_list_from_filter(full_test_list, out_path="test_preparation/")

# Creates:
# test_preparation/tests_torch_test_list.txt:
#   tests/models/bert/test_modeling_bert.py
#   tests/models/gpt2/test_modeling_gpt2.py
#
# test_preparation/tests_tokenization_test_list.txt:
#   tests/models/bert/test_tokenization_bert.py
#
# test_preparation/examples_torch_test_list.txt:
#   examples/pytorch/text-classification/test_text_classification.py
```

### Example 6: Checking for Docstring-Only Changes

```python
# Check if a file only has docstring/comment changes
from tests_fetcher import diff_is_docstring_only
from git import Repo

repo = Repo(".")
is_doc_only = diff_is_docstring_only(
    repo,
    branching_point="origin/main",
    filename="src/transformers/models/bert/modeling_bert.py"
)

if is_doc_only:
    print("No functional changes, skip testing")
else:
    print("Real code changes detected")
```

### Example 7: Getting Doctest Files

```python
# Get files with modified doc examples
from tests_fetcher import get_doctest_files, get_all_doctest_files

# Get all doctest-eligible files
all_doctest = get_all_doctest_files()
# Returns: ["src/transformers/models/bert/modeling_bert.py", ...]

# Get only files with modified examples
modified_doctest = get_doctest_files(diff_with_last_commit=False)
# Returns subset with actual doc example changes
```

### Example 8: Parsing Commit Messages

```python
# Parse commit message for CI flags
from tests_fetcher import parse_commit_message

flags = parse_commit_message("Fix bug in tokenizer [test all]")
# flags = {"skip": False, "no_filter": False, "test_all": True}

flags = parse_commit_message("Update docs [skip ci]")
# flags = {"skip": True, "no_filter": False, "test_all": False}

flags = parse_commit_message("Major refactor [no filter]")
# flags = {"skip": False, "no_filter": True, "test_all": False}
```

### Example 9: Complete CI Integration

```python
#!/usr/bin/env python
# Complete example for CI integration

import sys
from pathlib import Path
from git import Repo
from tests_fetcher import (
    get_modified_python_files,
    create_reverse_dependency_map,
    create_module_to_test_map,
    infer_tests_to_run,
    parse_commit_message
)

# Check commit message
repo = Repo(".")
commit_flags = parse_commit_message(repo.head.commit.message)

if commit_flags["skip"]:
    print("CI skipped by commit message")
    sys.exit(0)

if commit_flags["test_all"]:
    print("Running all tests")
    test_all = True
else:
    # Analyze changes
    modified = get_modified_python_files()
    print(f"Modified files: {len(modified)}")

    # Check if we should test everything
    test_all = len(modified) > 50 or "setup.py" in modified

# Run test fetcher
infer_tests_to_run(
    output_file="test_list.txt",
    diff_with_last_commit=False,
    filter_models=not commit_flags["no_filter"],
    test_all=test_all
)

print("Test fetching complete")
```

## Related Pages

- [Notification Service Doc Tests Implementation](/wikis/huggingface_transformers_NotificationServiceDocTests.md)
- [Update Metadata Implementation](/wikis/huggingface_transformers_UpdateMetadata.md)

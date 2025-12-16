# File: `scripts/enforce_kwargs_spacing.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 179 |
| Functions | `enforce_spacing`, `remove_redundant_passes`, `process_file`, `main` |
| Imports | __future__, argparse, ast, collections, io, pathlib, sys, tokenize |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Code quality enforcement tool that ensures keyword arguments use proper spacing around the `=` sign and removes redundant `pass` statements from Python files.

**Mechanism:** The script performs two main operations: (1) `enforce_spacing()` uses Python's tokenize module to scan for `=` operators and inserts spaces before/after them when missing, tracking line offsets to handle multiple edits; (2) `remove_redundant_passes()` parses the AST to identify `pass` statements in blocks containing other statements, then removes them while preserving code structure. The `process_file()` function applies both transformations sequentially, only writing back to disk if changes were made. The script preserves file encoding and includes self-protection to prevent modifying itself during batch operations.

**Significance:** This is a code formatting utility specific to the Unsloth project's style requirements. While standard formatters like Ruff handle most formatting, this script enforces project-specific conventions about keyword argument spacing that may differ from default formatter behavior. It serves as a pre-commit or CI tool to maintain consistent code style across the codebase, complementing rather than replacing standard formatting tools.

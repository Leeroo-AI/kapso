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

**Purpose:** A code formatting script that enforces consistent spacing around `=` in keyword arguments and removes redundant `pass` statements from Python files.

**Mechanism:** The script has two main functions: `enforce_spacing()` uses Python's tokenize module to find all `=` operators and ensures they have spaces on both sides by inserting spaces where missing; `remove_redundant_passes()` uses Python's AST parser to identify `pass` statements in code blocks that contain other executable statements and removes them. The `process_file()` function reads a Python file with proper encoding detection, applies both transformations, and writes back only if changes were made. The script is designed to avoid self-modification and handles edge cases like multi-line pass statements.

**Significance:** This is a development utility for maintaining consistent code style in the Unsloth codebase. It complements standard formatters like Ruff by enforcing Unsloth's specific convention of having spaces around `=` in keyword arguments (e.g., `func(arg = value)` instead of `func(arg=value)`), which is a non-standard but explicit style choice for readability.

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

**Purpose:** Code style formatter that enforces spaces around '=' in keyword arguments and removes redundant pass statements from Python files.

**Mechanism:** Uses Python's tokenize module to identify '=' operators and add surrounding spaces where missing, then uses AST parsing to detect and remove pass statements that share a block with other executable code. Processes files in-place with proper encoding detection.

**Significance:** Development utility that maintains consistent code style for keyword arguments across the Unsloth codebase, complementing standard formatters like Ruff by handling specific spacing conventions not covered by typical formatters.

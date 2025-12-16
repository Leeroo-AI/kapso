# File: `scripts/enforce_kwargs_spacing.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 179 |
| Functions | `enforce_spacing`, `remove_redundant_passes`, `process_file`, `main` |
| Imports | __future__, argparse, ast, collections, io, pathlib, sys, tokenize |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Documented

**Purpose:** Enforces keyword argument spacing by adding spaces around `=` signs in Python function calls

**Mechanism:** Uses Python tokenizer to parse code and AST to identify function calls, then adds spaces around `=` in keyword arguments while preserving string literals and other contexts

**Significance:** Maintains consistent code style across the Unsloth codebase, ensuring keyword arguments follow the format `key = value` rather than `key=value`

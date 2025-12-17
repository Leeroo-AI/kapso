# File: `utils/check_copies.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 1044 |
| Functions | `find_block_end`, `split_code_into_blocks`, `find_code_in_transformers`, `replace_code`, `find_code_and_splits`, `get_indent`, `run_ruff`, `stylify`, `... +5 more` |
| Imports | argparse, collections, glob, os, re, subprocess, transformers |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Enforces code duplication consistency by verifying that code marked with "# Copied from" comments exactly matches the original source after applying specified transformations.

**Mechanism:** Parses Python files looking for "# Copied from transformers.X.Y" comments, extracts the referenced source code using AST-based introspection, applies replacement patterns (e.g., "with Bert->GPT2"), compares the result against the actual code in the file, and optionally overwrites mismatches using ruff for formatting. Also maintains consistency across localized READMEs by synchronizing model lists.

**Significance:** Critical quality control tool that prevents drift in duplicated code (e.g., when multiple model implementations share similar patterns), ensuring bug fixes and improvements propagate correctly across related implementations while allowing controlled variations via replacement patterns.

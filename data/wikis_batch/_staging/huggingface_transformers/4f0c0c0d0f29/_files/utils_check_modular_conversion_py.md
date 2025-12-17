# File: `utils/check_modular_conversion.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 247 |
| Functions | `process_file`, `compare_files`, `get_models_in_diff`, `guaranteed_no_diff` |
| Imports | argparse, create_dependency_mapping, difflib, functools, glob, io, logging, modular_model_converter, multiprocessing, os, ... +3 more |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Validates that modular modeling files (modular_*.py) correctly generate their corresponding traditional modeling files (modeling_*.py) without drift, supporting the modular model architecture pattern.

**Mechanism:** Uses modular_model_converter to generate modeling code from modular files, compares against actual modeling files using difflib unified_diff, processes files in dependency order (via topological sort) using multiprocessing, and can either check for diffs or auto-fix by overwriting. Optimizes by skipping models not in git diff (unless --check_all), and creates backup files before overwriting.

**Significance:** Critical consistency enforcer for the modular modeling architecture where models are defined in a modular style then mechanically converted to traditional format, ensuring the two representations stay in sync and preventing manual edits to generated files from being lost.

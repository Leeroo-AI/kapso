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

**Purpose:** Verifies that modular model files (modular_*.py) correctly generate their corresponding traditional modeling files through automated conversion.

**Mechanism:** Uses topological sorting to process modular files in dependency order, then converts each modular file to traditional modeling files and compares them with the actual files using unified diffs. Optimizes by only checking models that appear in the git diff or their dependencies. Can optionally overwrite files to fix discrepancies with the `--fix_and_overwrite` flag.

**Significance:** Essential for maintaining Transformers' transition to modular architecture where models inherit from each other. Ensures the generated traditional modeling files stay synchronized with their modular sources, preventing code divergence and maintaining backward compatibility.

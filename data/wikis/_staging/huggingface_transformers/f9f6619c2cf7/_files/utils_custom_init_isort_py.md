# File: `utils/custom_init_isort.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 331 |
| Functions | `get_indent`, `split_code_in_indented_blocks`, `ignore_underscore_and_lowercase`, `sort_objects`, `sort_objects_in_import`, `sort_imports`, `sort_imports_in_all_inits` |
| Imports | argparse, collections, os, re, typing |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Sorts import statements in custom __init__.py files that use lazy loading via _import_structure dictionaries. Maintains consistent alphabetical ordering following isort conventions (CONSTANTS, Classes, functions).

**Mechanism:** Parses __init__.py files to locate _import_structure dictionaries, splits code into indented blocks at various levels, extracts module keys and their imported objects, then sorts them alphabetically while ignoring underscores and case. Handles both direct dictionary assignments and append/extend operations. Can check-only or auto-fix files.

**Significance:** Essential code quality tool for Transformers' lazy import system. The library uses delayed imports to avoid loading all models at "import transformers", dramatically improving startup time. This script ensures imports remain sorted as part of make style command, preventing merge conflicts and maintaining code consistency.

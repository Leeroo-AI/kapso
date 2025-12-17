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

**Purpose:** Sorts imports in transformers' custom lazy-loading __init__ files following isort conventions.

**Mechanism:** Parses __init__.py files containing `_import_structure` dictionaries (used for delayed imports). Splits code into indented blocks, extracts import keys and values, then sorts alphabetically with special rules: uppercase constants first, CamelCase classes second, lowercase functions last. Handles both single-line and multi-line import definitions. Applied recursively to all __init__.py files in the transformers package.

**Significance:** Code quality tool used by `make style` to maintain consistent import ordering in Transformers' special delayed-import init files, which can't be processed by standard isort due to their non-traditional structure using dictionary-based imports instead of direct import statements.

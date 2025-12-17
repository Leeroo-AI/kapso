# File: `utils/check_inits.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 353 |
| Functions | `find_backend`, `parse_init`, `analyze_results`, `get_transformers_submodules`, `check_submodules` |
| Imports | collections, os, pathlib, re |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Validates that the custom __init__.py files use delayed imports correctly by ensuring objects in _import_structure match those in TYPE_CHECKING blocks, and that all submodules are registered.

**Mechanism:** Parses src/transformers/__init__.py to extract two dictionaries: (1) _import_structure mapping backends to object lists, and (2) TYPE_CHECKING imports for type checkers. Compares these to find missing/extra objects per backend. Also walks the transformers directory tree to verify every submodule appears as a key in _import_structure (even with empty values) to ensure importability.

**Significance:** Critical import system validator that prevents the delayed-import pattern from becoming inconsistent, which would break imports or type-checking. Ensures users can import objects even without optional dependencies installed (via dummy objects) while maintaining proper type hints for IDEs.

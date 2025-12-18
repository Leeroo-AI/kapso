# File: `utils/sort_auto_mappings.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 124 |
| Functions | `sort_auto_mapping`, `sort_all_auto_mappings` |
| Imports | argparse, os, re |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Automatically sorts the model name mappings in auto module files alphabetically to maintain consistent code formatting and organization.

**Mechanism:** Uses regex patterns to identify OrderedDict mapping definitions in auto module files (e.g., `MODEL_MAPPING_NAMES = OrderedDict`), extracts mapping blocks that may span single or multiple lines, sorts them alphabetically by their identifier strings (the model config names in quotes), and either overwrites the file with sorted content or reports inconsistencies. Operates on all Python files in `src/transformers/models/auto/`.

**Significance:** This is a code quality and maintenance utility integrated into the repository's style checks. It ensures auto-mappings remain alphabetically sorted across the codebase, making them easier to review and reducing merge conflicts. Used in both `make style` (auto-fix mode) and `make quality` (check-only mode) commands.

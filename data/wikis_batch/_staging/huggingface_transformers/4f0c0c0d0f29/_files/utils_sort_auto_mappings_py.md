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

**Purpose:** Enforces alphabetical ordering of model name mappings in auto module files for code consistency.

**Mechanism:** Uses regex patterns to identify OrderedDict mapping declarations in auto module files, extracts model identifier tuples (handling both single-line and multi-line definitions), sorts blocks alphabetically by model name, and either fixes the files in-place or checks for violations. Integrated into make style (auto-fix) and make quality (validation).

**Significance:** Code quality tool that maintains consistent ordering in critical mapping files like MODEL_FOR_CAUSAL_LM_MAPPING, making code reviews easier, reducing merge conflicts, and ensuring predictable ordering when models are dynamically discovered. Part of the automated style enforcement pipeline used in CI.

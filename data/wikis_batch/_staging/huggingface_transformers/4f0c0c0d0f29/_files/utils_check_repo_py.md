# File: `utils/check_repo.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 1309 |
| Functions | `check_missing_backends`, `check_model_list`, `get_model_modules`, `get_models`, `is_building_block`, `is_a_private_model`, `check_models_are_in_init`, `get_model_test_files`, `... +21 more` |
| Imports | ast, collections, difflib, os, pathlib, re, transformers, types, warnings |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Performs comprehensive repository consistency checks to ensure all models are properly defined, tested, documented, and configured in auto classes.

**Mechanism:** Uses AST parsing to inspect model definitions, test files, and documentation files. Performs multiple validation passes checking: model initialization, test coverage, documentation completeness, auto-class registration, deprecated model lists, and forward method signatures. Includes extensive whitelist/blacklist constants (PRIVATE_MODELS, IGNORE_NON_TESTED, UNCONVERTIBLE_MODEL_ARCHITECTURES) to handle exceptional cases. The main entry point runs all checks via `check_repo_quality()`.

**Significance:** Critical quality control script for the repository, invoked by `make repo-consistency` to ensure code and documentation standards are maintained. Prevents incomplete model integrations and missing test coverage before code reaches production.

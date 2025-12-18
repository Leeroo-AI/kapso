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

**Purpose:** Comprehensive repository consistency checker that validates models, tests, documentation, and auto-class mappings across the entire Transformers codebase.

**Mechanism:** Performs 10+ different consistency checks including: verifying all models are in the main init, all models have corresponding tests, all public objects are documented, all models are in auto classes, all auto mappings are valid and importable, decorator ordering is correct, and models accept **kwargs in forward methods. Uses AST parsing, regex matching, and introspection to analyze code structure.

**Significance:** Acts as the central quality assurance tool for repository consistency, catching a wide range of integration issues that would otherwise break the library. Essential for maintaining the complex interdependencies between models, configs, auto classes, tests, and documentation in this large multi-model library.

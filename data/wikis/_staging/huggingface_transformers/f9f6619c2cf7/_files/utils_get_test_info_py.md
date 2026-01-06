# File: `utils/get_test_info.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 197 |
| Functions | `get_module_path`, `get_test_module`, `get_tester_classes`, `get_test_classes`, `get_model_classes`, `get_model_tester_from_test_class`, `get_test_classes_for_model`, `get_tester_classes_for_model`, `... +4 more` |
| Imports | importlib, os, sys |

## Understanding

**Status:** ✅ Explored

**Purpose:** Dynamically introspects model test files to extract relationships between test classes, model classes, and tester classes.

**Mechanism:** Uses Python's importlib to dynamically import test modules from paths like `tests/models/*/test_modeling_*.py`, then inspects class attributes (particularly `all_model_classes`) and instantiates test classes to discover their associated model testers. Creates mappings between models, test classes, and tester classes through reflection and attribute inspection.

**Significance:** Provides programmatic access to the test architecture for automated tooling and documentation. Enables scripts to understand which test classes cover which models, which tester classes are used, and how the test infrastructure is organized without manual maintenance of these relationships.

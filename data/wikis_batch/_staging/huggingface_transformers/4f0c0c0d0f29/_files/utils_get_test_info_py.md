# File: `utils/get_test_info.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 197 |
| Functions | `get_module_path`, `get_test_module`, `get_tester_classes`, `get_test_classes`, `get_model_classes`, `get_model_tester_from_test_class`, `get_test_classes_for_model`, `get_tester_classes_for_model`, `... +4 more` |
| Imports | importlib, os, sys |

## Understanding

**Status:** ✅ Explored

**Purpose:** Introspects model test files to extract test classes, model classes, and tester classes with their relationships.

**Mechanism:** Dynamically imports test modules from `tests/models/*/test_modeling_*.py` paths and uses Python's introspection to find classes ending in `ModelTester`, classes with non-empty `all_model_classes` attributes, and builds mappings between models, tests, and testers. Includes utilities like `to_json()` for readable output formatting.

**Significance:** Critical for test infrastructure analysis and documentation generation, enabling automated discovery of test relationships, validation of test coverage, and generation of test mapping documentation for the transformers library's extensive model test suite.

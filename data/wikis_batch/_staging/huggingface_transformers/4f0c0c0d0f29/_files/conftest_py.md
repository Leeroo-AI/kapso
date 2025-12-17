# File: `conftest.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 152 |
| Classes | `CustomOutputChecker` |
| Functions | `pytest_configure`, `pytest_collection_modifyitems`, `pytest_addoption`, `pytest_terminal_summary`, `pytest_sessionfinish` |
| Imports | _pytest, doctest, os, pytest, sys, transformers, warnings |

## Understanding

**Status:** ✅ Explored

**Purpose:** Configures pytest behavior for the entire test suite, including custom markers, doctest handling, and test collection modifications.

**Mechanism:** Implements pytest hooks (pytest_configure, pytest_collection_modifyitems, pytest_addoption, pytest_terminal_summary, pytest_sessionfinish) to register custom markers (is_pipeline_test, is_staging_test, accelerate_tests, not_device_test, torch_compile_test, etc.), automatically tag CPU-only tests from NOT_DEVICE_TESTS set, customize doctest output checking with CustomOutputChecker and IGNORE_RESULT flag, inject HfDoctestModule and HfDocTestParser, enable TF32 control, patch torch.compile for fullgraph mode, and handle empty test collection (exit code 5 -> 0). Adds src directory to sys.path for development testing.

**Significance:** Critical test infrastructure file that centralizes pytest configuration for the entire Transformers repository. Ensures consistent test behavior across local development and CI environments, enables special test markers for different testing scenarios, and provides custom doctest functionality for documentation testing.

# File: `tests/conftest.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 86 |
| Classes | `ErrorOnDeprecation` |
| Functions | `pytest_addoption`, `pytest_configure`, `pytest_collection_modifyitems`, `pytest_runtest_makereport` |
| Imports | logging, platform, pytest, re |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests for pytest configuration and fixtures

**Mechanism:** Configures pytest with custom options (regression tests), sets up deprecation error handling for transformers library warnings, and implements MacOS-specific test skipping for torch.load vulnerability issues

**Significance:** Test coverage for test infrastructure configuration and platform-specific test behaviors

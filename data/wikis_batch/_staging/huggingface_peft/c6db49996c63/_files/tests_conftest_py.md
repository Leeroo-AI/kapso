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

**Purpose:** Tests for pytest configuration and test collection setup.

**Mechanism:** Provides pytest hooks for configuring test behavior: adds `--regression` CLI option for running regression tests, configures markers, creates custom logging handler (`ErrorOnDeprecation`) to raise errors on transformers deprecation warnings, and implements MacOS-specific workaround for torch.load vulnerability errors with older PyTorch versions.

**Significance:** Ensures consistent test execution across the test suite, catches deprecation warnings early, and handles platform-specific issues that would otherwise break CI on MacOS runners.

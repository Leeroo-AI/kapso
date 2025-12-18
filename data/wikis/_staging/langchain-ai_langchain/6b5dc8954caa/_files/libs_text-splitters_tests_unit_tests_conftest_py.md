# File: `libs/text-splitters/tests/unit_tests/conftest.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 86 |
| Functions | `pytest_addoption`, `pytest_collection_modifyitems` |
| Imports | collections, importlib, pytest |

## Understanding

**Status:** ✅ Explored

**Purpose:** Configures pytest test collection behavior for text-splitters unit tests with custom markers for optional dependencies.

**Mechanism:** Adds two custom CLI options (--only-extended, --only-core) and implements pytest_collection_modifyitems hook to handle @pytest.mark.requires("package") markers. Checks if required packages are installed using importlib.util.find_spec and skips tests or fails (in --only-extended mode) when dependencies are missing.

**Significance:** Critical testing infrastructure that enables selective test execution based on installed dependencies. Allows core tests to run without optional packages while ensuring extended tests validate all features when dependencies are present, supporting the monorepo's modular architecture.

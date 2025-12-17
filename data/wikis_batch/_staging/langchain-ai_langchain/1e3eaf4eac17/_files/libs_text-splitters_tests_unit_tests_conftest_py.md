# File: `libs/text-splitters/tests/unit_tests/conftest.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 86 |
| Functions | `pytest_addoption`, `pytest_collection_modifyitems` |
| Imports | collections, importlib, pytest |

## Understanding

**Status:** ✅ Explored

**Purpose:** Configures pytest behavior for unit tests, adding custom command-line options and implementing the `@pytest.mark.requires` marker for conditional test execution.

**Mechanism:** Adds `--only-extended` and `--only-core` CLI options to pytest. Implements collection hook that checks if required packages are installed using `importlib.util.find_spec`, automatically skipping tests marked with `@pytest.mark.requires("package_name")` if the package isn't available. Prevents conflicting options and caches package availability checks for performance.

**Significance:** Critical testing infrastructure that allows tests to gracefully skip when optional dependencies are missing. Enables flexible test execution strategies (core-only, extended-only, or all tests) and ensures CI/CD pipelines can run appropriate test subsets based on installed dependencies.

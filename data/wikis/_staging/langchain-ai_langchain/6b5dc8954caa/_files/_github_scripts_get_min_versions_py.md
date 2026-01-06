# File: `.github/scripts/get_min_versions.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 199 |
| Functions | `get_pypi_versions`, `get_minimum_version`, `get_min_version_from_toml`, `check_python_version` |
| Imports | collections, packaging, re, requests, sys, typing |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Determines minimum compatible versions of dependencies from pyproject.toml for testing backward compatibility.

**Mechanism:** Queries PyPI API to fetch all available versions, parses version specifiers (including caret notation like ^0.2.3), filters by Python version markers, and finds the lowest version satisfying constraints. Handles special cases for langchain-core and other internal packages that are skipped during pull requests.

**Significance:** Enables minimum version testing in CI to ensure packages work with the oldest advertised compatible versions, preventing regression in compatibility promises and catching breaking changes early.

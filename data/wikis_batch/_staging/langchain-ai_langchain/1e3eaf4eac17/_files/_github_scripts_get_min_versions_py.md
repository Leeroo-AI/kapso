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

**Purpose:** Extracts minimum compatible versions for critical dependencies from pyproject.toml files and resolves them against PyPI to find the actual lowest published version that satisfies the constraints. This enables testing against minimum supported versions to ensure backward compatibility.

**Mechanism:** The script parses pyproject.toml files to extract dependency specifications for libraries in MIN_VERSION_LIBS (langchain-core, langchain, langchain-text-splitters, numpy, SQLAlchemy). It converts caret syntax (^) version constraints to standard range specifiers (>=, <) and uses the packaging library to parse version specifiers. For each dependency, it queries the PyPI JSON API to get all available versions, filters them against the version constraints, and returns the minimum compatible version. The script also handles Python version markers to ensure dependencies are checked for the correct Python version. It supports both "release" and "pull_request" modes, with some libraries skipped during PRs to handle simultaneous multi-package changes.

**Significance:** Essential for testing backward compatibility by allowing CI to verify that packages work with their minimum declared dependencies, not just the latest versions. This catches cases where code uses newer APIs than the minimum version specifies, preventing broken installations for users who have older versions of dependencies. This script is used in the CI dependency testing workflow to create test matrices with minimum versions.

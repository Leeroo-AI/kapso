# File: `.github/scripts/check_diff.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 340 |
| Functions | `all_package_dirs`, `dependents_graph`, `add_dependents` |
| Imports | collections, get_min_versions, glob, json, os, packaging, pathlib, sys, tomllib, typing |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Analyzes git diffs to determine which LangChain packages need testing, linting, or building based on file changes and dependency relationships.

**Mechanism:** Parses pyproject.toml files to build a dependency graph between packages, maps changed files to affected directories, and generates GitHub Actions matrix configurations with appropriate Python versions. Handles special cases like core package changes triggering dependent tests and Pydantic version compatibility testing.

**Significance:** Critical CI/CD infrastructure script that optimizes test execution by running tests only on affected packages and their dependents, preventing unnecessary CI runs while ensuring comprehensive coverage. Used by the check_diffs workflow to dynamically configure test jobs.

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

**Purpose:** Analyzes git diffs to intelligently determine which LangChain packages need testing, linting, or building based on file changes. It maps changed files to affected packages and generates GitHub Actions CI job configurations.

**Mechanism:** The script takes changed file paths as arguments and categorizes them into job types (lint, test, extended-test, codspeed). It builds a dependency graph by parsing pyproject.toml files across the monorepo to identify which packages depend on others. When a file changes, it determines the affected package and adds all its dependents to the test matrix. It then generates JSON configurations for each CI job type, specifying working directories and Python versions to test. Special logic handles core package changes (which have many dependents), infrastructure changes (which trigger broad testing), and Pydantic version compatibility testing. The script outputs JSON configurations consumed by GitHub Actions workflows.

**Significance:** Critical CI optimization tool that prevents running unnecessary tests while ensuring dependent packages are tested when their dependencies change. This dramatically reduces CI time and resource usage in the monorepo by running only relevant tests for each PR, while maintaining safety through dependency graph awareness. Without this, every PR would need to test all packages or risk missing breakages in dependent packages.

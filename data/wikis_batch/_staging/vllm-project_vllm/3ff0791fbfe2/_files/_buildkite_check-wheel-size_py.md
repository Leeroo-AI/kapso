# File: `.buildkite/check-wheel-size.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 53 |
| Functions | `print_top_10_largest_files`, `check_wheel_size` |
| Imports | os, sys, zipfile |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Validates Python wheel file sizes against PyPI upload limits to ensure distribution packages don't exceed size constraints.

**Mechanism:** Scans directories for .whl files, checks their size against a configurable limit (default 500 MB), and reports the top 10 largest files within wheels that exceed the limit. Uses zipfile module to inspect compressed wheel contents without extraction.

**Significance:** CI/CD quality gate that prevents publishing oversized wheels to PyPI. The 500 MB default aligns with PyPI's 800 MB quota mentioned in comments, ensuring vLLM stays within platform limits.

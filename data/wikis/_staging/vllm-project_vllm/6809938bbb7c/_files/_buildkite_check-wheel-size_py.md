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

**Purpose:** CI script that validates wheel package size stays within PyPI limits.

**Mechanism:** Walks through a directory to find .whl files, checks their size against VLLM_MAX_SIZE_MB (default 500MB), and reports the top 10 largest files inside the wheel if it exceeds the limit. Uses zipfile to inspect compressed contents.

**Significance:** Critical for CI/CD - PyPI has an 800MB quota per release, and this prevents publishing oversized wheels that would fail upload or consume too much quota. Helps identify which compiled artifacts are bloating the package.

# File: `scripts/run_ruff_format.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 30 |
| Functions | `main` |
| Imports | __future__, pathlib, subprocess, sys |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Documented

**Purpose:** Executes ruff formatter on Python files

**Mechanism:** Runs `ruff format` command and pipes output through the kwargs spacing enforcer script to ensure consistent formatting

**Significance:** Integrated code formatting solution that combines ruff's formatting with custom spacing rules, providing a single command for complete code formatting

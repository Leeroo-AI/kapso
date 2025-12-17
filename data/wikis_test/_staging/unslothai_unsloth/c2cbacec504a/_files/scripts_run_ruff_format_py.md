# File: `scripts/run_ruff_format.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 30 |
| Functions | `main` |
| Imports | __future__, pathlib, subprocess, sys |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Orchestrates ruff formatter followed by custom kwarg spacing enforcement.

**Mechanism:** Runs ruff format command, then invokes enforce_kwargs_spacing.py script sequentially for consistent code formatting.

**Significance:** Development tool for consistent code formatting across the repository.

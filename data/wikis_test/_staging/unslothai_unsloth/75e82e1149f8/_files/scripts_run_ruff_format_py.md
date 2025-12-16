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

**Purpose:** Combined formatting script that runs ruff formatter followed by Unsloth's custom kwargs spacing enforcement.

**Mechanism:**
- Takes file paths as command-line arguments
- First runs `ruff format` on all provided files
- If ruff succeeds, runs `enforce_kwargs_spacing.py` on the same files
- Returns non-zero exit code if either step fails
- Uses subprocess to invoke both tools

**Significance:** Developer workflow tool that combines standard Python formatting (ruff) with Unsloth's custom style rule (spaces around `=` in kwargs). Used in pre-commit or CI to ensure consistent code style.

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

**Purpose:** A wrapper script that runs the Ruff formatter followed by Unsloth's custom keyword argument spacing enforcement on Python files.

**Mechanism:** The script filters command-line arguments to only include existing file paths, then executes two subprocess commands in sequence: first `python -m ruff format` to apply standard Python formatting rules, then the companion `enforce_kwargs_spacing.py` script to apply Unsloth's custom spacing convention. If Ruff formatting fails (non-zero return code), it exits early without running the spacing enforcement.

**Significance:** This is a developer convenience script that combines standard Ruff formatting with Unsloth's custom style rules into a single command. It ensures that code is first formatted according to standard conventions and then has the project-specific keyword argument spacing applied, maintaining the codebase's consistent style with minimal developer effort.

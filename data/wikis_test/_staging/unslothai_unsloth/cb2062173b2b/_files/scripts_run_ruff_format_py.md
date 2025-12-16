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

**Purpose:** Orchestrates a two-stage code formatting pipeline that runs Ruff formatter followed by custom keyword argument spacing enforcement.

**Mechanism:** The script filters command-line arguments to identify valid file paths, then executes two sequential subprocess calls: first `ruff format` for standard Python formatting, then `enforce_kwargs_spacing.py` for project-specific spacing rules. The script short-circuits on failure, only proceeding to the second stage if Ruff succeeds, and propagates return codes to allow build/CI systems to detect formatting failures.

**Significance:** This is a convenience wrapper that combines standard and custom formatting into a single command, ensuring consistent application of both Ruff's opinionated formatting and Unsloth's specific style requirements. It's likely used in pre-commit hooks or CI pipelines to automate code quality checks, providing a unified entry point for developers rather than requiring them to run multiple formatters manually.

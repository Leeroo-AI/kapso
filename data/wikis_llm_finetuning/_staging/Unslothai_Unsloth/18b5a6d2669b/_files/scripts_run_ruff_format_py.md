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

**Purpose:** Orchestration wrapper that runs Ruff code formatter followed by custom kwargs spacing enforcement in a single command.

**Mechanism:** Executes two sequential subprocess commands: first runs 'ruff format' on provided files, then runs 'enforce_kwargs_spacing.py' on the same files. Returns early with error code if Ruff fails, otherwise returns the spacing enforcement result.

**Significance:** Convenience utility that combines standard Ruff formatting with Unsloth's custom spacing rules, ensuring developers can apply both formatting steps with a single command during development or in pre-commit hooks.

# File: `utils/check_pipeline_typing.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 93 |
| Functions | `main` |
| Imports | re, transformers |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Auto-generates type-safe overload signatures for the `pipeline()` function based on all supported pipeline tasks.

**Mechanism:** Extracts the base pipeline function signature, then programmatically generates `@overload` decorated function signatures for each task in `SUPPORTED_TASKS`, mapping each task name to its corresponding pipeline class. Updates the pipelines init file by replacing content between marked comment boundaries with the generated overloads.

**Significance:** Provides precise type hints for IDE auto-completion and type checkers, allowing them to understand that `pipeline("text-classification")` returns a specific `TextClassificationPipeline` rather than a generic `Pipeline`. Keeps type annotations synchronized with the growing list of supported pipeline tasks.

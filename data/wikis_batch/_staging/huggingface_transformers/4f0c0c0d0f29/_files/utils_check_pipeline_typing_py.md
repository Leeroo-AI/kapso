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

**Purpose:** Generates @overload type signatures for the pipeline() function to provide precise return type hints for each task literal in IDEs and type checkers.

**Mechanism:** Reads src/transformers/pipelines/__init__.py, extracts the pipeline function signature, iterates through SUPPORTED_TASKS to get task names and their implementation classes, generates an @overload signature for each task (replacing "task: Optional[str]" with "task: Literal['task_name']" and setting return type), and writes the generated code between special markers in the file.

**Significance:** Developer experience tool that enables accurate type checking and autocomplete for pipeline() calls, allowing IDEs to know that pipeline(task="text-classification") returns a TextClassificationPipeline rather than a generic Pipeline type.

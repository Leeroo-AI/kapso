# File: `packages/@n8n/ai-workflow-builder.ee/evaluations/programmatic/python/src/__main__.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 8 |
| Imports | compare_workflows |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Module execution entry point

**Mechanism:** Imports and calls the main() function from compare_workflows when the module is run with `python -m compare_workflows`. Simple delegation pattern to enable module-level execution.

**Significance:** Allows the workflow comparison tool to be invoked as a Python module rather than directly executing the script file. Follows Python best practices for creating executable modules.

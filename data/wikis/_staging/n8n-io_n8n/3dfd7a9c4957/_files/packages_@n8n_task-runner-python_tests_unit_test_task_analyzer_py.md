# File: `packages/@n8n/task-runner-python/tests/unit/test_task_analyzer.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 197 |
| Classes | `TestTaskAnalyzer`, `TestImportValidation`, `TestAttributeAccessValidation`, `TestDynamicImportDetection`, `TestAllowAll` |
| Imports | pytest, src |

## Understanding

**Status:** ✅ Explored

**Purpose:** Task analyzer security unit tests

**Mechanism:** Tests AST-based code analysis for import detection, security validation, attribute access checking, dynamic import detection, and allow-all mode. Validates security enforcement logic.

**Significance:** Security validation coverage. Ensures the code analyzer correctly identifies blocked imports and dangerous patterns.

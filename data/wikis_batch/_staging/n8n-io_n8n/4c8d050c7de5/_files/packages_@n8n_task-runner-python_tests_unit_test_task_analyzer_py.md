# File: `packages/@n8n/task-runner-python/tests/unit/test_task_analyzer.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 197 |
| Classes | `TestTaskAnalyzer`, `TestImportValidation`, `TestAttributeAccessValidation`, `TestDynamicImportDetection`, `TestAllowAll` |
| Imports | pytest, src |

## Understanding

**Status:** ✅ Explored

**Purpose:** Static code analysis security validation tests

**Mechanism:** Tests TaskAnalyzer's static analysis (AST-based) that validates user Python code before execution. Validates import restrictions (allows configured stdlib modules, blocks os/sys/subprocess/socket), blocks relative imports, dangerous attribute access (BLOCKED_ATTRIBUTES like __subclasses__), blocked names (BLOCKED_NAMES like globals/locals/__import__), loader/spec exploitation attempts, name-mangled attributes, and dynamic imports. Tests wildcard mode (*) that bypasses validation.

**Significance:** First line of defense in the security sandbox - prevents execution of obviously dangerous code patterns before they can run. Critical for protecting the task runner and host system from malicious user code. Complements runtime restrictions with static analysis.

# File: `packages/@n8n/task-runner-python/src/import_validation.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 37 |
| Functions | `validate_module_import` |
| Imports | src, sys |

## Understanding

**Status:** ✅ Explored

**Purpose:** Module import security validation

**Mechanism:** validate_module_import() checks module names against allowed/blocked lists before permitting imports. Intercepts import attempts and raises SecurityViolationError for unauthorized modules.

**Significance:** Core sandbox security enforcement. Prevents user code from importing dangerous modules like os, subprocess, or network libraries.

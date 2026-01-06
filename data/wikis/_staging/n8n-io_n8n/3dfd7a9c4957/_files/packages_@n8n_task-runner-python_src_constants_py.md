# File: `packages/@n8n/task-runner-python/src/constants.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 193 |
| Imports | src |

## Understanding

**Status:** ✅ Explored

**Purpose:** Security and built-in module definitions

**Mechanism:** Defines lists of allowed/blocked Python modules for security enforcement, built-in function restrictions, and default execution environment settings. Contains ALLOWED_BUILTINS, BLOCKED_BUILTINS, DEFAULT_ALLOWED_MODULES, and related constants.

**Significance:** Core security policy definitions. Determines what Python capabilities are available to user code in the sandboxed execution environment.

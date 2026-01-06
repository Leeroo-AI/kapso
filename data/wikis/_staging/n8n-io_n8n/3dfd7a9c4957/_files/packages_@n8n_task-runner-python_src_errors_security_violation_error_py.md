# File: `packages/@n8n/task-runner-python/src/errors/security_violation_error.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 9 |
| Classes | `SecurityViolationError` |

## Understanding

**Status:** ✅ Explored

**Purpose:** Security policy violation error

**Mechanism:** Custom exception raised when code attempts forbidden operations like importing blocked modules or accessing restricted resources.

**Significance:** Core security enforcement mechanism. Blocks and reports unauthorized code execution attempts.

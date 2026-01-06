# File: `packages/@n8n/task-runner-python/src/config/security_config.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 9 |
| Classes | `SecurityConfig` |
| Imports | dataclasses |

## Understanding

**Status:** ✅ Explored

**Purpose:** Security settings configuration

**Mechanism:** Defines SecurityConfig dataclass with security-related settings for the task runner, controlling import restrictions and code execution boundaries.

**Significance:** Core security component controlling what Python code can execute. Prevents unauthorized module imports and dangerous operations.

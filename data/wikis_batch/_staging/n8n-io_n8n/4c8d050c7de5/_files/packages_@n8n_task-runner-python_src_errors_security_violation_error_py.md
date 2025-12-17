# File: `packages/@n8n/task-runner-python/src/errors/security_violation_error.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 9 |
| Classes | `SecurityViolationError` |

## Understanding

**Status:** ✅ Explored

**Purpose:** Security policy violation error.

**Mechanism:** Exception with `message` and `description` attributes. Used when user code attempts to use disallowed modules or builtins. The description field provides detailed context (line numbers, what was blocked, allowlist info).

**Significance:** Core security error type. Raised by TaskAnalyzer during static analysis or at runtime when import validation fails. Not sent to Sentry (expected user error). Provides clear feedback to users about sandbox restrictions.

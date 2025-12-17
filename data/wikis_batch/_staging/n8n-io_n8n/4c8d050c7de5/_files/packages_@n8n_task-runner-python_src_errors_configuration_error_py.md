# File: `packages/@n8n/task-runner-python/src/errors/configuration_error.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 6 |
| Classes | `ConfigurationError` |

## Understanding

**Status:** ✅ Explored

**Purpose:** Invalid runner configuration error.

**Mechanism:** Simple Exception subclass with custom message. Stores the message in both `self.message` attribute and passes to parent. Used when user-provided configuration values are invalid (e.g., invalid port numbers, negative timeouts).

**Significance:** Separates configuration validation errors from runtime errors. Enables clear error messages when environment variables or config files have invalid values.

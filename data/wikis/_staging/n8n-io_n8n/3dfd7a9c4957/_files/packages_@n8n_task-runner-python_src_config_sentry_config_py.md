# File: `packages/@n8n/task-runner-python/src/config/sentry_config.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 30 |
| Classes | `SentryConfig` |
| Imports | dataclasses, src |

## Understanding

**Status:** ✅ Explored

**Purpose:** Sentry error tracking configuration

**Mechanism:** Defines SentryConfig dataclass with DSN, environment, release version, and sampling rate settings. Uses environment variable loading for flexible deployment configuration.

**Significance:** Enables error monitoring and tracking via Sentry. Critical for production observability and debugging.

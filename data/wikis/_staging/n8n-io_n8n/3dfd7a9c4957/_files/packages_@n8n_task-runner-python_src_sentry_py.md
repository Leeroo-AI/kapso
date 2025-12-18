# File: `packages/@n8n/task-runner-python/src/sentry.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 88 |
| Classes | `TaskRunnerSentry` |
| Functions | `setup_sentry` |
| Imports | logging, src, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Sentry error tracking integration

**Mechanism:** TaskRunnerSentry class wraps the Sentry SDK for error reporting. setup_sentry() initializes with DSN, environment, and release info. Captures exceptions with context and handles sampling configuration.

**Significance:** Production observability component. Enables error tracking, performance monitoring, and debugging for deployed task runners.

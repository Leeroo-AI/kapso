# File: `packages/@n8n/task-runner-python/tests/unit/test_sentry.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 256 |
| Classes | `TestTaskRunnerSentry`, `TestSetupSentry`, `TestSentryConfig` |
| Functions | `sentry_config`, `disabled_sentry_config` |
| Imports | logging, pytest, src, unittest |

## Understanding

**Status:** ✅ Explored

**Purpose:** Sentry integration unit tests

**Mechanism:** Tests TaskRunnerSentry class, setup_sentry function, and SentryConfig handling. Validates error capture, configuration parsing, enable/disable behavior, and sampling configuration.

**Significance:** Validates error tracking setup. Ensures Sentry integration works correctly in various configurations.

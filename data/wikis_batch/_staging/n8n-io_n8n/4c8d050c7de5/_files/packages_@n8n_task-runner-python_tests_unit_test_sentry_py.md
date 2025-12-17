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

**Purpose:** Sentry error monitoring integration tests

**Mechanism:** Tests TaskRunnerSentry class initialization with proper SDK configuration (DSN, release version, environment, server name, integrations). Validates error filtering logic that excludes user code errors (syntax errors, errors from executor files), ignored error types, and errors identified by type name or stack frames. Tests SentryConfig.from_env() loading, setup_sentry() initialization with graceful degradation (logs warnings for missing SDK or setup failures).

**Significance:** Ensures error monitoring captures only legitimate system errors, not user code problems (which would pollute Sentry with noise). Critical for production monitoring and debugging of the task runner infrastructure itself while respecting user privacy and data separation.

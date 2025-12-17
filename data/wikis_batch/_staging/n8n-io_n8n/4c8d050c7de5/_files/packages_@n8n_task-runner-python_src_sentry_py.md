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

**Purpose:** Error tracking and monitoring integration

**Mechanism:** Integrates Sentry SDK for error reporting:
1. Initializes Sentry with DSN, release version (n8n@{version}), environment, deployment name
2. Sets custom tag "server_type: task_runner_python" for filtering
3. Implements before_send filter to exclude ignored error types (ConfigurationError, TaskRuntimeError, SecurityViolationError, etc.)
4. Filters out errors originating from user code (checks stacktrace for executor filenames)
5. Enables LoggingIntegration for ERROR level logs
6. Disables auto-enabling integrations, uses only default + logging
7. shutdown() flushes pending events with 2s timeout
8. setup_sentry() wraps initialization with ImportError handling (SDK may not be installed)

**Significance:** Provides production error monitoring for the task runner infrastructure (not user code errors). The filtering logic is critical to prevent noise: user code errors, expected errors (timeouts, cancellations), and security violations are intentionally excluded. The graceful degradation (continues if sentry-sdk not installed) supports optional monitoring. The server_type tag enables filtering task runner errors in a multi-component n8n deployment.

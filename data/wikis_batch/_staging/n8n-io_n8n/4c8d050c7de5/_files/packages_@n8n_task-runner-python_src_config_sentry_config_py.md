# File: `packages/@n8n/task-runner-python/src/config/sentry_config.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 30 |
| Classes | `SentryConfig` |
| Imports | dataclasses, src |

## Understanding

**Status:** ✅ Explored

**Purpose:** Configures Sentry error tracking integration settings

**Mechanism:** The `SentryConfig` dataclass stores four configuration fields: `dsn` (Data Source Name for Sentry endpoint), `n8n_version`, `environment`, and `deployment_name`. The `from_env()` class method reads these from environment variables (`N8N_SENTRY_DSN`, `N8N_VERSION`, `N8N_ENVIRONMENT`, `N8N_DEPLOYMENT_NAME`), defaulting to empty strings if not set. The `enabled` property returns `True` if a DSN is configured, providing a convenient way to check if Sentry integration is active.

**Significance:** This configuration enables error tracking and performance monitoring for the Python task runner. Sentry integration provides critical observability into production issues, capturing exceptions, stack traces, and contextual metadata (version, environment, deployment) that help developers diagnose and fix problems in distributed task execution environments.

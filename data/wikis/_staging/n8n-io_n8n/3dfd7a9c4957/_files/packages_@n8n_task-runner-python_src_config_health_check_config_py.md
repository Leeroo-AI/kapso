# File: `packages/@n8n/task-runner-python/src/config/health_check_config.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 34 |
| Classes | `HealthCheckConfig` |
| Imports | dataclasses, src |

## Understanding

**Status:** ✅ Explored

**Purpose:** Health check server configuration

**Mechanism:** Defines HealthCheckConfig dataclass with settings for health check endpoint including enabled flag, host binding, and port configuration. Uses environment variable loading from env module.

**Significance:** Enables Kubernetes/container orchestration health monitoring for the task runner. Essential for production deployments.

# File: `packages/@n8n/task-runner-python/src/config/health_check_config.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 34 |
| Classes | `HealthCheckConfig` |
| Imports | dataclasses, src |

## Understanding

**Status:** ✅ Explored

**Purpose:** Configures HTTP health check server endpoint settings

**Mechanism:** The `HealthCheckConfig` dataclass holds three configuration properties: `enabled` (boolean), `host` (string), and `port` (integer). The `from_env()` class method constructs the configuration by reading environment variables: `N8N_RUNNERS_HEALTH_CHECK_SERVER_ENABLED`, `N8N_RUNNERS_HEALTH_CHECK_SERVER_HOST`, and `N8N_RUNNERS_HEALTH_CHECK_SERVER_PORT`. Port validation ensures the value is between 0-65535, raising a `ConfigurationError` if invalid. Defaults are provided via constants: `DEFAULT_HEALTH_CHECK_SERVER_HOST` and `DEFAULT_HEALTH_CHECK_SERVER_PORT`. Health checks are disabled by default.

**Significance:** This configuration enables external monitoring systems to verify task runner health status via HTTP endpoints. The health check server provides observability for deployment orchestration, load balancers, and monitoring tools to detect unhealthy task runner instances and route traffic accordingly.

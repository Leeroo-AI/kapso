# File: `packages/@n8n/task-runner-python/src/config/task_runner_config.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 133 |
| Classes | `TaskRunnerConfig` |
| Functions | `parse_allowlist` |
| Imports | dataclasses, src |

## Understanding

**Status:** ✅ Explored

**Purpose:** Main task runner configuration

**Mechanism:** Defines TaskRunnerConfig dataclass with comprehensive settings including broker URL, runner ID, grant token, allowed/blocked modules, max payload size, task timeout, and idle timeout. Aggregates HealthCheckConfig, SentryConfig, and SecurityConfig. Includes parse_allowlist() for module restriction lists.

**Significance:** Central configuration hub for the task runner. Controls all runtime behavior including communication, security, and resource limits.

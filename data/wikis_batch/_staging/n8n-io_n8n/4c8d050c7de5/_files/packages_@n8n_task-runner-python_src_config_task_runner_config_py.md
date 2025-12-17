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

**Purpose:** Main task runner configuration with security and performance settings

**Mechanism:** The `TaskRunnerConfig` dataclass aggregates all core runtime settings including: authentication (`grant_token`), communication (`task_broker_uri`), performance (`max_concurrency`, `max_payload_size`, `task_timeout`), shutdown behavior (`auto_shutdown_timeout`, `graceful_shutdown_timeout`), security constraints (`stdlib_allow`, `external_allow`, `builtins_deny`, `env_deny`), and calculated timeouts (`pipe_reader_timeout`). The `from_env()` class method performs extensive validation: ensures grant token exists, validates positive timeouts, checks payload size against pipe limits (`PIPE_MSG_MAX_SIZE`), and calculates pipe reader timeout based on expected throughput (typical payload size / parse speed + safety buffer). The `parse_allowlist()` helper parses comma-separated module lists, validating that wildcard '*' is used alone if present.

**Significance:** This is the central configuration hub for the Python task runner, consolidating all operational parameters in one validated structure. It enforces critical constraints (required authentication, valid timeouts, reasonable payload limits) while providing sensible defaults. The calculated pipe reader timeout is particularly important for preventing deadlocks when reading large task results. The security allowlists/denylists are foundational for sandbox enforcement. The auto-shutdown feature enables efficient resource utilization in serverless/containerized deployments.

# File: `packages/@n8n/task-runner-python/src/constants.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 193 |
| Imports | src |

## Understanding

**Status:** ✅ Explored

**Purpose:** Centralized configuration constants and defaults

**Mechanism:** Defines application-wide constants organized by category:
1. **Message types**: WebSocket message type identifiers for broker/runner protocol
2. **Runner config**: Default values for concurrency (5), timeouts (60s), payload size (1GB)
3. **Executor**: Internal keys for user output, circular reference handling, subprocess exit codes
4. **Pipe communication**: Message prefix length (4 bytes), max size (~4GB)
5. **Broker connection**: Default URI, WebSocket path
6. **Health check**: Default host/port for health endpoint
7. **Environment variables**: Names for all configurable options (N8N_RUNNERS_*)
8. **Sentry**: Error tracking tags and ignored error types
9. **Logging**: Format strings, templates for structured logs
10. **Security**: Blocked names/attributes, default denied builtins, error messages
11. **Rejection reasons**: Standard messages for task rejection

**Significance:** This is the single source of truth for all configuration defaults and magic values. Centralizing constants improves maintainability and makes the system's defaults visible. The security constants (BLOCKED_NAMES, BLOCKED_ATTRIBUTES, BUILTINS_DENY_DEFAULT) are particularly critical as they define the security boundary. The comprehensive set of defaults allows the runner to work with minimal configuration.

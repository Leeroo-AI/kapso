# File: `packages/@n8n/task-runner-python/src/errors/websocket_connection_error.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 10 |
| Classes | `WebsocketConnectionError` |

## Understanding

**Status:** ✅ Explored

**Purpose:** WebSocket connection failure error

**Mechanism:** Custom exception raised when the task runner fails to establish or maintain a WebSocket connection with the task broker.

**Significance:** Handles broker connectivity issues. Enables appropriate retry logic and failure reporting for network problems.

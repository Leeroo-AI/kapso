# File: `packages/@n8n/task-runner-python/src/errors/websocket_connection_error.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 10 |
| Classes | `WebsocketConnectionError` |

## Understanding

**Status:** ✅ Explored

**Purpose:** Broker connection failure error.

**Mechanism:** ConnectionError subclass raised when WebSocket connection to broker fails. Includes the broker_uri in the error message with troubleshooting hint to check broker reachability.

**Significance:** Startup/connectivity error. Common causes: network issues, incorrect URI, broker service down. Inherits from ConnectionError for standard exception handling patterns. Critical for initial connection establishment.

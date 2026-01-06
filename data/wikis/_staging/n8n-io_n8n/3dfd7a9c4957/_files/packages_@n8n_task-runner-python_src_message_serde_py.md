# File: `packages/@n8n/task-runner-python/src/message_serde.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 138 |
| Classes | `MessageSerde` |
| Imports | dataclasses, json, src, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Message serialization/deserialization

**Mechanism:** MessageSerde class handles JSON encoding/decoding of broker and runner messages. Deserializes incoming broker messages into typed dataclasses, serializes runner responses for transmission.

**Significance:** Communication protocol implementation. Ensures type-safe message handling between task runner and broker.

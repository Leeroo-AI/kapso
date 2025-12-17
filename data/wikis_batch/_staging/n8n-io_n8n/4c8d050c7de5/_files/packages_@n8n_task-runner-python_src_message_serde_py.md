# File: `packages/@n8n/task-runner-python/src/message_serde.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 138 |
| Classes | `MessageSerde` |
| Imports | dataclasses, json, src, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** WebSocket message serialization/deserialization

**Mechanism:** Handles broker-runner protocol message conversion:
1. deserialize_broker_message(): Parses JSON to typed BrokerMessage objects
2. Uses MESSAGE_TYPE_MAP to dispatch to specialized parsers
3. Supports message types: InfoRequest, RunnerRegistered, TaskOfferAccept, TaskSettings, TaskCancel, RpcResponse
4. Converts nodeMode from string ("runOnceForAllItems", "runOnceForEachItem") to enum ("all_items", "per_item")
5. serialize_runner_message(): Converts dataclass messages to JSON
6. Applies snake_to_camel_case conversion for JavaScript interop (e.g., task_id → taskId)
7. Validates required fields and raises ValueError for malformed messages

**Significance:** This is the protocol translation layer between Python and TypeScript/JavaScript broker. The type-safe deserialization prevents runtime errors from malformed messages. The camelCase conversion ensures compatibility with JavaScript naming conventions. The specialized parsers extract and validate required fields, making the rest of the codebase work with typed objects rather than raw dicts. This separation of concerns keeps protocol details isolated from business logic.

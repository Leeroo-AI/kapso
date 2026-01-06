# File: `packages/@n8n/task-runner-python/src/message_types/broker.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 80 |
| Classes | `BrokerInfoRequest`, `BrokerRunnerRegistered`, `BrokerTaskOfferAccept`, `TaskSettings`, `BrokerTaskSettings`, `BrokerTaskCancel`, `BrokerRpcResponse` |
| Imports | dataclasses, src, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Broker message type definitions

**Mechanism:** Defines dataclass types for messages received from the task broker: info requests, registration confirmations, task offer acceptances, task settings with code/data, cancellation requests, and RPC responses.

**Significance:** Protocol contract for broker communication. Ensures type-safe handling of all incoming broker messages.

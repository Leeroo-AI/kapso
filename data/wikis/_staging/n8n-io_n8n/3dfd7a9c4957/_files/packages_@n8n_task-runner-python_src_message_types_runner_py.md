# File: `packages/@n8n/task-runner-python/src/message_types/runner.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 74 |
| Classes | `RunnerInfo`, `RunnerTaskOffer`, `RunnerTaskAccepted`, `RunnerTaskRejected`, `RunnerTaskDone`, `RunnerTaskError`, `RunnerRpcCall` |
| Imports | constants, dataclasses, src, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Runner message type definitions

**Mechanism:** Defines dataclass types for messages sent by the runner to broker: runner info/registration, task offers, task acceptance/rejection, task completion/error, and RPC calls for data fetching.

**Significance:** Protocol contract for runner communication. Ensures type-safe construction of all outgoing runner messages.

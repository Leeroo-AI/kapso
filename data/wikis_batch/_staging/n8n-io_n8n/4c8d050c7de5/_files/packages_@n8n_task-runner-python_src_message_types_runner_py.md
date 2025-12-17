# File: `packages/@n8n/task-runner-python/src/message_types/runner.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 74 |
| Classes | `RunnerInfo`, `RunnerTaskOffer`, `RunnerTaskAccepted`, `RunnerTaskRejected`, `RunnerTaskDone`, `RunnerTaskError`, `RunnerRpcCall` |
| Imports | constants, dataclasses, src, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Defines messages sent from runner to broker

**Mechanism:** This module defines the runner-to-broker message protocol using dataclasses with literal type discriminators. Each message class has a `type` field for type-safe discrimination. Key message types include: `RunnerInfo` (identifies runner with name and supported task types), `RunnerTaskOffer` (offers to accept a task with offer_id, task_type, and validity duration), `RunnerTaskAccepted` (confirms task acceptance), `RunnerTaskRejected` (declines task with reason), `RunnerTaskDone` (reports successful completion with result data dictionary), `RunnerTaskError` (reports execution failure with error dictionary), and `RunnerRpcCall` (initiates RPC call with call_id, task_id, name, and parameters). The `RunnerMessage` union type encompasses all possible runner messages.

**Significance:** This defines the complete runner-to-broker response vocabulary, complementing the broker messages. Together they form a bidirectional protocol for distributed task execution. The offer-based task acceptance pattern (RunnerTaskOffer) enables load balancing - runners advertise capacity and brokers decide which runner gets each task. The acceptance/rejection pattern provides graceful task routing when runners are busy or shutting down. Task completion messages (Done/Error) carry the actual execution results back to workflows. The RPC capability enables runners to call back into n8n during execution (e.g., for credential access or node helpers). The protocol supports the full task execution lifecycle with proper error handling and bidirectional communication.

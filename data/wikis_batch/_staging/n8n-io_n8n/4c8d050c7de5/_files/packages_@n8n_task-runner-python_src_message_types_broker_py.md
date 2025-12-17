# File: `packages/@n8n/task-runner-python/src/message_types/broker.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 80 |
| Classes | `BrokerInfoRequest`, `BrokerRunnerRegistered`, `BrokerTaskOfferAccept`, `TaskSettings`, `BrokerTaskSettings`, `BrokerTaskCancel`, `BrokerRpcResponse` |
| Imports | dataclasses, src, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Defines messages sent from broker to runner

**Mechanism:** This module defines the broker-to-runner message protocol using dataclasses with literal type discriminators. Each message class has a `type` field with a specific literal value (e.g., `"broker:inforequest"`) that enables type-safe message discrimination. Key message types include: `BrokerInfoRequest` (requests runner information), `BrokerRunnerRegistered` (confirms registration), `BrokerTaskOfferAccept` (accepts a task offer with task_id and offer_id), `BrokerTaskSettings` (delivers task execution settings including code, mode, items, workflow context), `BrokerTaskCancel` (cancels running task with reason), and `BrokerRpcResponse` (responds to RPC calls with status). The `TaskSettings` dataclass contains rich task metadata: Python code to execute, node execution mode (all_items/per_item), error handling flag, input items, workflow/node identifiers, and optional query parameter. The `BrokerMessage` union type encompasses all possible broker messages.

**Significance:** This defines the complete broker-to-runner command vocabulary for the task execution protocol. The type discriminators enable robust message parsing and routing without brittle type checks. The `TaskSettings` structure captures all context needed to execute Python code nodes in n8n workflows, including workflow provenance for debugging/logging. The protocol supports the full task lifecycle: offers, settings delivery, cancellation, and bidirectional RPC. The typed message structure ensures protocol correctness and enables easy serialization/deserialization.

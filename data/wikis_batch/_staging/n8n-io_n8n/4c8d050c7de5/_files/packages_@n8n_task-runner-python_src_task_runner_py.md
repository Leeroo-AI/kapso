# File: `packages/@n8n/task-runner-python/src/task_runner.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 501 |
| Classes | `TaskOffer`, `TaskRunner` |
| Imports | asyncio, dataclasses, logging, random, src, time, typing, urllib, websockets |

## Understanding

**Status:** ✅ Explored

**Purpose:** Core orchestrator for Python task execution

**Mechanism:** Implements the main task runner loop by:
1. Establishing WebSocket connection to n8n's task broker
2. Sending periodic task offers based on capacity (max_concurrency)
3. Handling broker messages (info requests, task offers, settings, cancellations)
4. Accepting tasks and managing their lifecycle (waiting → running → done/error)
5. Executing tasks via TaskExecutor in isolated subprocesses
6. Forwarding print() output from user code to browser via RPC calls
7. Managing idle timeout for auto-shutdown when no tasks are running
8. Gracefully shutting down by waiting for tasks to complete or terminating them

The runner maintains state for open offers, running tasks, and uses message serialization for broker communication. Tasks are validated with TaskAnalyzer before execution.

**Significance:** This is the heart of the task runner system. It implements the bidirectional protocol with n8n's broker, manages task capacity, coordinates execution, and ensures proper lifecycle management. The WebSocket-based communication pattern enables real-time task distribution and monitoring in n8n's distributed workflow execution architecture.

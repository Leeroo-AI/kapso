# File: `packages/@n8n/task-runner-python/src/task_runner.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 501 |
| Classes | `TaskOffer`, `TaskRunner` |
| Imports | asyncio, dataclasses, logging, random, src, time, typing, urllib, websockets |

## Understanding

**Status:** ✅ Explored

**Purpose:** Main task runner orchestrator

**Mechanism:** TaskRunner maintains WebSocket connection to broker, handles task offers, manages task lifecycle (accept, execute, complete/error), coordinates executor subprocesses, implements idle timeout, and handles reconnection logic.

**Significance:** Central orchestration component. Ties together all other modules into a functioning task execution service.

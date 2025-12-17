# File: `packages/@n8n/task-runner-python/tests/fixtures/local_task_broker.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 190 |
| Classes | `ActiveTask`, `LocalTaskBroker` |
| Imports | aiohttp, asyncio, collections, dataclasses, json, src, tests, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Mock WebSocket task broker for testing

**Mechanism:** Implements a local aiohttp-based WebSocket server that simulates the n8n task broker. It maintains connections, tracks active tasks, handles broker-runner message exchanges (info requests, task offers/accepts, task completion/errors), and stores RPC messages. Uses async queues for bidirectional message passing and provides utilities to wait for specific message types.

**Significance:** Critical testing infrastructure that enables integration tests to verify task runner behavior without requiring a full n8n server. Allows tests to control task lifecycle, verify message protocol compliance, and inspect RPC calls (like console.log output) from executed Python code.

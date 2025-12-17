# File: `examples/online_serving/kv_events_subscriber.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 117 |
| Classes | `EventBatch`, `KVCacheEvent`, `BlockStored`, `BlockRemoved`, `AllBlocksCleared`, `KVEventBatch` |
| Functions | `process_event`, `main` |
| Imports | msgspec, typing, vllm, zmq |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** KV cache event monitoring client

**Mechanism:** Subscribes to vLLM's KV cache events via ZeroMQ pub/sub sockets. Listens for events about cache block operations (stored, removed, cleared) and handles missed messages through a replay mechanism. Uses msgspec for efficient message serialization/deserialization.

**Significance:** Example demonstrating how to monitor and react to KV cache operations in vLLM. Useful for building external cache management systems, debugging cache behavior, or implementing cache-aware optimizations.

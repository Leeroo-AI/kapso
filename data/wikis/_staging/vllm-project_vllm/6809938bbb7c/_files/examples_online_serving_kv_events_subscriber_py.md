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

**Purpose:** ZeroMQ-based subscriber for KV cache events monitoring

**Mechanism:** Connects to vLLM's ZMQ event stream (ports 5557/5558) to receive real-time notifications about KV cache operations: block storage, removal, and clearing. Uses msgpack for efficient serialization. Implements replay mechanism to recover missed messages by requesting historical events from the replay socket when sequence gaps are detected.

**Significance:** Critical example for distributed KV cache management and monitoring. Enables external systems to track cache state for optimization, debugging, or coordination. The replay functionality ensures reliable event delivery even with network interruptions. Essential for advanced deployments requiring cache observability or cross-instance coordination.

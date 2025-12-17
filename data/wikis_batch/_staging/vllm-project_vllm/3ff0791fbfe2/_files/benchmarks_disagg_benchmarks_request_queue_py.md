# File: `benchmarks/disagg_benchmarks/request_queue.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 39 |
| Classes | `RequestQueue` |
| Imports | asyncio, collections |

## Understanding

**Status:** ✅ Explored

**Purpose:** Async request queue manager

**Mechanism:** Deque-based queue with semaphore-controlled concurrency and size limits. Provides enqueue() with capacity checks and process() loop that consumes tasks respecting max concurrent limit. Uses asyncio.Lock for thread-safe queue operations.

**Significance:** Concurrency control utility for disaggregation benchmarks. Prevents overwhelming backend services by limiting in-flight requests while queuing excess load, essential for stable performance measurements.
